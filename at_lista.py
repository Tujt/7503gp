"""
AT-LISTA (Adaptive Threshold LISTA) for MRI Reconstruction
============================================================
This adds the AT-LISTA model on top of mri_full.py.
Run: python3 at_lista.py

The key difference from standard LISTA:
  - LISTA:    scalar threshold θ_k shared across all coefficients
  - AT-LISTA: vector threshold θ_k ∈ R^n, one per coefficient per layer
This allows the network to learn different shrinkage strengths for different
spatial locations and frequency components.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py, os, time
from pathlib import Path
import torch
import torch.nn as nn

try:
    from skimage.metrics import structural_similarity as ssim_func
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

# ======================== CONFIG ========================
DATA_DIR = "/Users/lvke/Downloads/singlecoil_val"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== FFT ========================
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))

def center_crop(img, shape):
    h, w = img.shape[-2:]
    th, tw = shape
    return img[..., (h-th)//2:(h+th)//2, (w-tw)//2:(w+tw)//2]

def compute_psnr(img, ref):
    mse = np.mean((img - ref)**2)
    if mse < 1e-12: return 60.0
    return 10*np.log10(np.max(ref)**2 / mse)

def compute_ssim(img, ref):
    if not HAS_SSIM: return float('nan')
    return ssim_func(img, ref, data_range=ref.max()-ref.min())

def create_mask(W, acceleration=4, center_frac=0.08, seed=42):
    rng = np.random.RandomState(seed)
    mask = np.zeros(W)
    nc = max(int(W * center_frac), 1)
    mask[W//2 - nc//2 : W//2 - nc//2 + nc] = 1
    total = max(int(W / acceleration), nc)
    avail = np.where(mask == 0)[0]
    extra = min(total - nc, len(avail))
    if extra > 0:
        mask[rng.choice(avail, extra, replace=False)] = 1
    return mask.reshape(1, -1)

def load_slice(filepath, slice_idx=None):
    with h5py.File(filepath, 'r') as f:
        ks = f['kspace'][()]
        if slice_idx is None:
            slice_idx = ks.shape[0] // 2
        kspace = ks[slice_idx]
    img_full = np.abs(ifft2c(kspace))
    H, W = img_full.shape
    crop_size = min(H, W)
    crop_shape = (crop_size, crop_size)
    x_gt = center_crop(img_full, crop_shape)
    scale = x_gt.max()
    if scale > 0:
        x_gt /= scale
        kspace = kspace / scale
    return kspace, x_gt, crop_shape

# ======================== MODEL COMPONENTS ========================

class DataConsistency(nn.Module):
    """Replace measured k-space lines with original data."""
    def forward(self, x, kspace_u, mask):
        # x: (B,2,H,W), kspace_u: (B,2,H,W), mask: (B,1,W)
        xc = torch.complex(x[:,0], x[:,1])
        kx = torch.fft.fftshift(torch.fft.fft2(
            torch.fft.ifftshift(xc, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        ku = torch.complex(kspace_u[:,0], kspace_u[:,1])
        kx = kx * (1 - mask) + ku * mask
        xi = torch.fft.fftshift(torch.fft.ifft2(
            torch.fft.ifftshift(kx, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        return torch.stack([xi.real, xi.imag], dim=1)


class ConvBlock(nn.Module):
    """Standard CNN denoiser (used by both LISTA and AT-LISTA)."""
    def __init__(self, channels=2, features=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)


# ======================== STANDARD LISTA ========================

class LISTA_MRI(nn.Module):
    """Standard LISTA: scalar threshold per layer."""
    def __init__(self, T=10, features=32):
        super().__init__()
        self.T = T
        self.dc = DataConsistency()
        self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])

    def forward(self, kspace_u, mask):
        ku_complex = torch.complex(kspace_u[:,0], kspace_u[:,1])
        x0 = torch.fft.fftshift(torch.fft.ifft2(
            torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        x = torch.stack([x0.real, x0.imag], dim=1)
        for k in range(self.T):
            x = self.denoisers[k](x)
            x = self.dc(x, kspace_u, mask)
        return x


# ======================== AT-LISTA (NOVEL) ========================

class AdaptiveThresholdBlock(nn.Module):
    """
    Adaptive Threshold Block: learns a per-pixel, per-channel threshold map.

    Instead of a scalar θ_k, we learn θ_k(i,j) for each spatial location.
    This is implemented as a small CNN that predicts the threshold map
    from the current reconstruction, making it input-dependent.

    Soft-thresholding with learned vector threshold:
        T_{θ_k}(z)_i = sign(z_i) * max(|z_i| - θ_k,i, 0)
    """
    def __init__(self, channels=2, features=16):
        super().__init__()
        # Threshold predictor: input-dependent threshold map
        self.threshold_net = nn.Sequential(
            nn.Conv2d(channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels, 3, padding=1),
            nn.Softplus(),  # ensures threshold > 0
        )
        # Scale parameter to control initial threshold magnitude
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """
        x: (B, 2, H, W) — real/imag channels of complex image
        Returns: soft-thresholded x with adaptive per-pixel threshold
        """
        # Predict spatially-varying threshold map
        theta = self.threshold_net(x) * self.scale  # (B, 2, H, W), all positive

        # Soft-thresholding with vector threshold
        sign_x = torch.sign(x)
        abs_x = torch.abs(x)
        x_thresh = sign_x * torch.clamp(abs_x - theta, min=0)

        return x_thresh


class ATLISTA_MRI(nn.Module):
    """
    Adaptive Threshold LISTA for MRI Reconstruction.

    Each layer consists of:
        1. CNN denoiser (learned regularization)
        2. Adaptive soft-thresholding (per-pixel learned threshold)
        3. Data consistency (physics-based)

    Compared to standard LISTA which uses a scalar threshold per layer,
    AT-LISTA learns a threshold map θ_k ∈ R^{H×W} that adapts to the
    local signal characteristics at each spatial location.
    """
    def __init__(self, T=10, features=32):
        super().__init__()
        self.T = T
        self.dc = DataConsistency()
        self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])
        self.thresholds = nn.ModuleList([AdaptiveThresholdBlock(2, 16) for _ in range(T)])

    def forward(self, kspace_u, mask):
        """
        kspace_u: (B, 2, H, W) — undersampled k-space (real/imag)
        mask:     (B, 1, W)    — sampling mask
        """
        # Zero-filled initialization (same as LISTA)
        ku_complex = torch.complex(kspace_u[:,0], kspace_u[:,1])
        x0 = torch.fft.fftshift(torch.fft.ifft2(
            torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        x = torch.stack([x0.real, x0.imag], dim=1)

        for k in range(self.T):
            x = self.denoisers[k](x)          # Learned regularization (CNN)
            x = self.thresholds[k](x)         # Adaptive thresholding (novel)
            x = self.dc(x, kspace_u, mask)    # Data consistency (physics)

        return x


# ======================== DATA PREPARATION ========================

def prepare_torch_data(h5_files, acceleration=4, max_slices=500):
    all_kspace_u, all_gt, all_masks = [], [], []
    ref_shape = None

    for fp in h5_files:
        with h5py.File(fp, 'r') as f:
            ks_vol = f['kspace'][()]
        n = ks_vol.shape[0]
        for si in range(max(n//4,0), min(3*n//4,n), 2):
            kspace = ks_vol[si]
            if ref_shape is None:
                ref_shape = kspace.shape
            elif kspace.shape != ref_shape:
                continue
            img_full = np.abs(ifft2c(kspace))
            scale = img_full.max()
            if scale < 1e-6: continue
            kspace = kspace / scale
            W = kspace.shape[1]
            mask = create_mask(W, acceleration=acceleration, seed=si)
            kspace_u = kspace * mask
            gt = np.abs(ifft2c(kspace))
            all_kspace_u.append(np.stack([kspace_u.real, kspace_u.imag], axis=0))
            all_gt.append(gt)
            all_masks.append(mask.reshape(1, -1))
            if len(all_gt) >= max_slices: break
        if len(all_gt) >= max_slices: break

    print(f"  Prepared {len(all_gt)} slices (shape {ref_shape})")
    return all_kspace_u, all_gt, all_masks


# ======================== TRAINING ========================

def train_model(model, h5_files, model_name="Model", T=10, epochs=50,
                lr=1e-3, batch_size=2, acceleration=4):
    print(f"\n  Training {model_name} (T={T}, epochs={epochs})...")

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")

    all_ku, all_gt, all_masks = prepare_torch_data(h5_files, acceleration, max_slices=500)
    crop_h = min(g.shape[0] for g in all_gt)
    crop_w = min(g.shape[1] for g in all_gt)
    crop_shape = (crop_h, crop_w)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(all_gt)
    losses = []

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss, count = 0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) == 0: continue

            ku_batch = torch.tensor(
                np.stack([all_ku[j] for j in idx]), dtype=torch.float32).to(device)
            mask_batch = torch.tensor(
                np.stack([all_masks[j] for j in idx]), dtype=torch.float32).to(device)
            gt_batch = [all_gt[j] for j in idx]

            optimizer.zero_grad()
            out = model(ku_batch, mask_batch)
            out_mag = torch.sqrt(out[:,0]**2 + out[:,1]**2 + 1e-8)

            loss = 0
            for b, gt in enumerate(gt_batch):
                out_crop = center_crop(out_mag[b].unsqueeze(0), crop_shape).squeeze()
                gt_crop = torch.tensor(
                    center_crop(gt, crop_shape), dtype=torch.float32).to(device)
                loss = loss + torch.mean((out_crop - gt_crop)**2)
            loss = loss / len(gt_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(count, 1)
        losses.append(avg_loss)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss={avg_loss:.6f}")

    return model, crop_shape, losses


def run_inference(model, kspace, mask_1d, crop_shape, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        ku = kspace * mask_1d
        ku_t = torch.tensor(
            np.stack([ku.real, ku.imag])[None], dtype=torch.float32).to(device)
        m_t = torch.tensor(
            mask_1d.reshape(1, 1, -1), dtype=torch.float32).to(device)
        out = model(ku_t, m_t)
        out_np = out.cpu().numpy()[0]
        mag = np.sqrt(out_np[0]**2 + out_np[1]**2)
        return center_crop(mag, crop_shape)


# ======================== MAIN ========================

def main():
    h5_files = sorted(Path(DATA_DIR).glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files in {DATA_DIR}"); return
    print(f"Found {len(h5_files)} files")

    # Load test slice
    test_file = h5_files[0]
    kspace, x_gt, crop_shape = load_slice(test_file)
    H, W = kspace.shape
    mask = create_mask(W, acceleration=4)
    kspace_u = kspace * mask

    x_zf = center_crop(np.abs(ifft2c(kspace_u)), crop_shape)
    psnr_zf = compute_psnr(x_zf, x_gt)
    print(f"Zero-filled: PSNR={psnr_zf:.2f} dB")

    train_files = h5_files[1:20]
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # ---- Train LISTA ----
    lista_model = LISTA_MRI(T=10, features=32)
    lista_model, _, lista_losses = train_model(
        lista_model, train_files, model_name="LISTA", T=10, epochs=50, lr=1e-3)
    x_lista = run_inference(lista_model, kspace, mask, crop_shape, device)
    psnr_lista = compute_psnr(x_lista, x_gt)
    ssim_lista = compute_ssim(x_lista, x_gt)
    print(f"  LISTA:    PSNR={psnr_lista:.2f} dB, SSIM={ssim_lista:.4f}")

    # ---- Train AT-LISTA ----
    atlista_model = ATLISTA_MRI(T=10, features=32)
    atlista_model, _, atlista_losses = train_model(
        atlista_model, train_files, model_name="AT-LISTA", T=10, epochs=50, lr=1e-3)
    x_atlista = run_inference(atlista_model, kspace, mask, crop_shape, device)
    psnr_atlista = compute_psnr(x_atlista, x_gt)
    ssim_atlista = compute_ssim(x_atlista, x_gt)
    print(f"  AT-LISTA: PSNR={psnr_atlista:.2f} dB, SSIM={ssim_atlista:.4f}")

    # ======================== PLOTS ========================

    # Plot 1: Training loss comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(lista_losses)+1), lista_losses, 'b-', lw=2, label='LISTA')
    ax.plot(range(1, len(atlista_losses)+1), atlista_losses, 'r-', lw=2, label='AT-LISTA')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss (MSE)", fontsize=12)
    ax.set_title("Training Loss: LISTA vs AT-LISTA", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_loss_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved training_loss_comparison.png")

    # Plot 2: Visual comparison (LISTA vs AT-LISTA)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12); ax.axis('off')

    show(axes[0], x_zf, f"Zero-filled\nPSNR={psnr_zf:.1f} dB")
    show(axes[1], x_lista, f"LISTA (T=10)\nPSNR={psnr_lista:.1f} dB")
    show(axes[2], x_atlista, f"AT-LISTA (T=10)\nPSNR={psnr_atlista:.1f} dB")
    show(axes[3], x_gt, "Ground Truth")
    fig.suptitle("LISTA vs AT-LISTA (4x acceleration)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "lista_vs_atlista.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved lista_vs_atlista.png")

    # Plot 3: Error maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    err_zf = np.abs(x_zf - x_gt)
    err_lista = np.abs(x_lista - x_gt)
    err_atlista = np.abs(x_atlista - x_gt)
    vmax = max(err_zf.max(), err_lista.max(), err_atlista.max()) * 0.5

    axes[0].imshow(err_zf, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title(f"Zero-filled Error\nMSE={np.mean(err_zf**2):.6f}", fontsize=11)
    axes[0].axis('off')
    axes[1].imshow(err_lista, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(f"LISTA Error\nMSE={np.mean(err_lista**2):.6f}", fontsize=11)
    axes[1].axis('off')
    axes[2].imshow(err_atlista, cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title(f"AT-LISTA Error\nMSE={np.mean(err_atlista**2):.6f}", fontsize=11)
    axes[2].axis('off')
    fig.suptitle("Reconstruction Error Maps", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_maps.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved error_maps.png")

    # Results table
    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'Params':<12} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*60}")
    n_lista = sum(p.numel() for p in lista_model.parameters())
    n_atlista = sum(p.numel() for p in atlista_model.parameters())
    print(f"{'Zero-filled':<20} {'—':<12} {psnr_zf:<12.2f} {compute_ssim(x_zf, x_gt):<10.4f}")
    print(f"{'LISTA (T=10)':<20} {n_lista:<12,} {psnr_lista:<12.2f} {ssim_lista:<10.4f}")
    print(f"{'AT-LISTA (T=10)':<20} {n_atlista:<12,} {psnr_atlista:<12.2f} {ssim_atlista:<10.4f}")
    print(f"{'='*60}")

    # Save models
    torch.save(lista_model.state_dict(), os.path.join(OUTPUT_DIR, "lista_model.pth"))
    torch.save(atlista_model.state_dict(), os.path.join(OUTPUT_DIR, "atlista_model.pth"))
    print("Models saved.")


if __name__ == "__main__":
    main()
