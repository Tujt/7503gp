"""
MRI Reconstruction: ISTA / FISTA / LISTA
=========================================
pip install numpy matplotlib h5py scikit-image PyWavelets torch
Set DATA_DIR below, then: python mri_full.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py, os, time
from pathlib import Path

try:
    import pywt; HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[WARN] pip install PyWavelets -- needed for ISTA/FISTA")

try:
    from skimage.metrics import structural_similarity as ssim_func; HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import torch, torch.nn as nn; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] pip install torch -- needed for LISTA")

# ======================== CONFIG ========================
DATA_DIR = "/Users/lvke/Downloads/singlecoil_val"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== FFT ========================
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))

# ======================== UTILS ========================
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

# ======================== DATA ========================
def load_slice(filepath, slice_idx=None):
    """
    Load one slice. Returns kspace, ground_truth, crop_size.
    Ground truth = |IFFT(full_kspace)|, center-cropped.
    Everything computed from kspace directly (no reconstruction_esc mismatch).
    """
    with h5py.File(filepath, 'r') as f:
        ks = f['kspace'][()]
        if slice_idx is None:
            slice_idx = ks.shape[0] // 2
        kspace = ks[slice_idx]  # (H, W) complex

    # Ground truth from full kspace
    img_full = np.abs(ifft2c(kspace))  # (H, W)

    # Crop to square (same as fastMRI convention)
    H, W = img_full.shape
    crop_size = min(H, W)
    crop_shape = (crop_size, crop_size)
    x_gt = center_crop(img_full, crop_shape)

    # Normalize to [0, 1]
    scale = x_gt.max()
    if scale > 0:
        x_gt /= scale
        kspace = kspace / scale

    return kspace, x_gt, crop_shape

def create_mask(W, acceleration=4, center_frac=0.08, seed=42):
    rng = np.random.RandomState(seed)
    mask = np.zeros(W)
    # Center lines
    nc = max(int(W * center_frac), 1)
    mask[W//2 - nc//2 : W//2 - nc//2 + nc] = 1
    # Random lines
    total = max(int(W / acceleration), nc)
    avail = np.where(mask == 0)[0]
    extra = min(total - nc, len(avail))
    if extra > 0:
        mask[rng.choice(avail, extra, replace=False)] = 1
    return mask.reshape(1, -1)

# ======================== ISTA / FISTA ========================
def wavelet_thresh(x, lam, wavelet='db4', level=3):
    """Soft-threshold wavelet detail coefficients of complex image x."""
    def _do(r):
        c = pywt.wavedec2(r, wavelet, level=level)
        c_new = [c[0]]
        for detail in c[1:]:
            c_new.append(tuple(np.sign(d)*np.maximum(np.abs(d)-lam, 0) for d in detail))
        out = pywt.waverec2(c_new, wavelet)
        return out[:x.shape[0], :x.shape[1]]
    return _do(np.real(x)) + 1j * _do(np.imag(x))

def ista_mri(kspace_u, mask, lam, n_iter, x_gt=None, crop=None):
    x = ifft2c(kspace_u)  # initialize with zero-filled (much better than zeros!)
    psnrs = []
    for _ in range(n_iter):
        # Data consistency gradient step
        res = fft2c(x) * mask - kspace_u
        x = x - ifft2c(res * mask)
        # Wavelet regularization
        if HAS_PYWT:
            x = wavelet_thresh(x, lam)
        if x_gt is not None:
            xm = center_crop(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
    return x, psnrs

def fista_mri(kspace_u, mask, lam, n_iter, x_gt=None, crop=None):
    x = ifft2c(kspace_u)  # zero-filled init
    x_prev = x.copy()
    t = 1.0
    psnrs = []
    for _ in range(n_iter):
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x + (t-1)/t_new * (x - x_prev)
        res = fft2c(y) * mask - kspace_u
        x_prev = x.copy()
        x = y - ifft2c(res * mask)
        if HAS_PYWT:
            x = wavelet_thresh(x, lam)
        t = t_new
        if x_gt is not None:
            xm = center_crop(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
    return x, psnrs

# ======================== LISTA (PyTorch) ========================
if HAS_TORCH:
    class DataConsistency(nn.Module):
        """Replace measured k-space lines with original data."""
        def forward(self, x, kspace_u, mask):
            # x: (B,2,H,W), kspace_u: (B,2,H,W), mask: (B,1,W)
            xc = torch.complex(x[:,0], x[:,1])  # (B,H,W)
            kx = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            ku = torch.complex(kspace_u[:,0], kspace_u[:,1])  # (B,H,W)
            # mask (B,1,W) broadcasts with (B,H,W)
            kx = kx * (1 - mask) + ku * mask
            xi = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kx, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            return torch.stack([xi.real, xi.imag], dim=1)  # (B,2,H,W)

    class ConvBlock(nn.Module):
        """Simple CNN denoiser block (replaces wavelet thresholding)."""
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
            return x + self.net(x)  # residual connection

    class LISTA_MRI(nn.Module):
        """
        Unrolled ISTA for MRI: T layers of (data consistency + learned denoiser).
        """
        def __init__(self, T=10, features=32):
            super().__init__()
            self.T = T
            self.dc = DataConsistency()
            self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])

        def forward(self, kspace_u, mask):
            """
            kspace_u: (B, 2, H, W) - undersampled k-space (real/imag channels)
            mask:     (B, 1, 1, W) - sampling mask
            """
            # Zero-filled init
            ku_complex = torch.complex(kspace_u[:,0], kspace_u[:,1])
            x0 = torch.fft.fftshift(torch.fft.ifft2(
                torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            x = torch.stack([x0.real, x0.imag], dim=1)  # (B,2,H,W)

            for k in range(self.T):
                x = self.denoisers[k](x)              # learned regularization
                x = self.dc(x, kspace_u, mask)         # data consistency

            return x  # (B, 2, H, W)

    def prepare_torch_data(h5_files, acceleration=4, max_slices=200):
        """Load slices and prepare as torch tensors. Only keeps slices with matching shapes."""
        all_kspace_u = []
        all_gt = []
        all_masks = []
        ref_shape = None

        for fp in h5_files:
            with h5py.File(fp, 'r') as f:
                ks_vol = f['kspace'][()]

            # Take a few slices from each file (middle region has best anatomy)
            n = ks_vol.shape[0]
            start = max(n//4, 0)
            end = min(3*n//4, n)
            for si in range(start, end, 2):  # every other slice to save memory
                kspace = ks_vol[si]

                # Fix reference shape from first valid slice
                if ref_shape is None:
                    ref_shape = kspace.shape
                elif kspace.shape != ref_shape:
                    continue  # skip slices with different shape

                img_full = np.abs(ifft2c(kspace))
                scale = img_full.max()
                if scale < 1e-6:
                    continue
                kspace = kspace / scale

                W = kspace.shape[1]
                mask = create_mask(W, acceleration=acceleration, seed=si)

                kspace_u = kspace * mask
                gt = np.abs(ifft2c(kspace))  # normalized ground truth (full res)

                # Store as real/imag channels
                all_kspace_u.append(np.stack([kspace_u.real, kspace_u.imag], axis=0))
                all_gt.append(gt)
                all_masks.append(mask.reshape(1, -1))  # (1, W)

                if len(all_gt) >= max_slices:
                    break
            if len(all_gt) >= max_slices:
                break

        print(f"  Prepared {len(all_gt)} slices (shape {ref_shape}) for training")
        return all_kspace_u, all_gt, all_masks

    def train_lista(h5_files, T=10, epochs=20, lr=1e-3, batch_size=2, acceleration=4):
        """Train LISTA network."""
        print(f"\n  Training LISTA (T={T}, epochs={epochs})...")

        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"  Device: {device}")

        # Prepare data
        all_ku, all_gt, all_masks = prepare_torch_data(h5_files, acceleration, max_slices=500)
        crop_h = min(g.shape[0] for g in all_gt)
        crop_w = min(g.shape[1] for g in all_gt)
        crop_shape = (crop_h, crop_w)

        model = LISTA_MRI(T=T, features=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        n = len(all_gt)
        losses = []

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0
            count = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                if len(idx) == 0:
                    continue

                # Build batch
                ku_batch = torch.tensor(np.stack([all_ku[j] for j in idx]), dtype=torch.float32).to(device)
                mask_batch = torch.tensor(np.stack([all_masks[j] for j in idx]), dtype=torch.float32).to(device)
                gt_batch = [all_gt[j] for j in idx]

                optimizer.zero_grad()
                out = model(ku_batch, mask_batch)  # (B, 2, H, W)

                # Loss: MSE on magnitude in cropped region
                out_mag = torch.sqrt(out[:,0]**2 + out[:,1]**2 + 1e-8)
                loss = 0
                for b, gt in enumerate(gt_batch):
                    out_crop = center_crop(out_mag[b].unsqueeze(0), crop_shape).squeeze()
                    gt_crop = torch.tensor(center_crop(gt, crop_shape), dtype=torch.float32).to(device)
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

    def lista_inference(model, kspace, mask_1d, crop_shape, device=None):
        """Run trained LISTA on a single slice."""
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            ku = kspace * mask_1d
            ku_t = torch.tensor(
                np.stack([ku.real, ku.imag])[None], dtype=torch.float32
            ).to(device)
            m_t = torch.tensor(
                mask_1d.reshape(1, 1, -1), dtype=torch.float32
            ).to(device)  # (1, 1, W)
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

    # --- Load one test slice ---
    test_file = h5_files[0]
    kspace, x_gt, crop_shape = load_slice(test_file)
    H, W = kspace.shape
    print(f"Test slice: kspace={kspace.shape}, gt={x_gt.shape}")

    mask = create_mask(W, acceleration=4)
    kspace_u = kspace * mask
    print(f"Sampling rate: {mask.sum()/mask.size:.1%}")

    # Zero-filled
    x_zf = center_crop(np.abs(ifft2c(kspace_u)), crop_shape)
    psnr_zf = compute_psnr(x_zf, x_gt)
    print(f"Zero-filled: PSNR={psnr_zf:.2f} dB")

    # --- ISTA ---
    lam = 0.002
    n_iter = 100
    print(f"Running ISTA ({n_iter} iter, lam={lam})...", end=" ", flush=True)
    t0 = time.time()
    x_ista, psnrs_ista = ista_mri(kspace_u, mask, lam, n_iter, x_gt, crop_shape)
    t_ista = time.time()-t0
    x_ista_c = center_crop(np.abs(x_ista), crop_shape)
    psnr_ista = compute_psnr(x_ista_c, x_gt)
    print(f"PSNR={psnr_ista:.2f} dB ({t_ista:.1f}s)")

    # --- FISTA ---
    print(f"Running FISTA ({n_iter} iter, lam={lam})...", end=" ", flush=True)
    t0 = time.time()
    x_fista, psnrs_fista = fista_mri(kspace_u, mask, lam, n_iter, x_gt, crop_shape)
    t_fista = time.time()-t0
    x_fista_c = center_crop(np.abs(x_fista), crop_shape)
    psnr_fista = compute_psnr(x_fista_c, x_gt)
    print(f"PSNR={psnr_fista:.2f} dB ({t_fista:.1f}s)")

    # --- LISTA ---
    psnr_lista = None
    x_lista_c = None
    if HAS_TORCH:
        train_files = h5_files[1:20]  # use files 1-5 for training, file 0 for test
        model, lista_crop, losses = train_lista(train_files, T=10, epochs=50, lr=1e-3)

        # Inference on test slice
        device = next(model.parameters()).device
        x_lista_c = lista_inference(model, kspace, mask, crop_shape, device)
        psnr_lista = compute_psnr(x_lista_c, x_gt)
        ssim_lista = compute_ssim(x_lista_c, x_gt)
        print(f"  LISTA inference: PSNR={psnr_lista:.2f} dB, SSIM={ssim_lista:.4f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lista_model.pth"))

    # ======================== PLOTS ========================

    # Plot 1: PSNR vs iteration
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(psnrs_ista, label=f"ISTA ({psnr_ista:.1f} dB)", lw=2)
    ax.plot(psnrs_fista, label=f"FISTA ({psnr_fista:.1f} dB)", lw=2)
    if psnr_lista:
        ax.axhline(psnr_lista, color='green', ls='-', lw=2, label=f"LISTA T=10 ({psnr_lista:.1f} dB)")
    ax.axhline(psnr_zf, color='gray', ls='--', alpha=0.7, label=f"Zero-filled ({psnr_zf:.1f} dB)")
    ax.set_xlabel("Iteration $k$", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("MRI Reconstruction: PSNR vs Iteration (4x)", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence_mri.png"), dpi=150)
    plt.close(fig)
    print("Saved convergence_mri.png")

    # Plot 2: Visual comparison
    n_panels = 5 if x_lista_c is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))

    def show(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11); ax.axis('off')

    show(axes[0], x_zf, f"Zero-filled\nPSNR={psnr_zf:.1f} dB")
    show(axes[1], x_ista_c, f"ISTA ({n_iter} iter)\nPSNR={psnr_ista:.1f} dB")
    show(axes[2], x_fista_c, f"FISTA ({n_iter} iter)\nPSNR={psnr_fista:.1f} dB")
    if x_lista_c is not None:
        show(axes[3], x_lista_c, f"LISTA (T=10)\nPSNR={psnr_lista:.1f} dB")
        show(axes[4], x_gt, "Ground Truth")
    else:
        show(axes[3], x_gt, "Ground Truth")

    fig.suptitle("Knee MRI Reconstruction (4x acceleration)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "visual_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved visual_comparison.png")

    # Plot 3: Training loss (LISTA)
    if HAS_TORCH and losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(losses)+1), losses, 'b-', lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss (MSE)")
        ax.set_title("LISTA Training Loss"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "lista_training_loss.png"), dpi=150)
        plt.close(fig)
        print("Saved lista_training_loss.png")

    # Results table
    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*60}")
    print(f"{'Zero-filled':<20} {psnr_zf:<12.2f} {compute_ssim(x_zf, x_gt):<10.4f}")
    print(f"{'ISTA':<20} {psnr_ista:<12.2f} {compute_ssim(x_ista_c, x_gt):<10.4f}")
    print(f"{'FISTA':<20} {psnr_fista:<12.2f} {compute_ssim(x_fista_c, x_gt):<10.4f}")
    if psnr_lista:
        print(f"{'LISTA (T=10)':<20} {psnr_lista:<12.2f} {ssim_lista:<10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
