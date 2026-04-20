"""
MRI Reconstruction: ISTA, FISTA, LISTA (CNN), AT-LISTA (CNN + adaptive threshold)
====================================================================================
- ISTA: wavelet thresholding
- FISTA: Total Variation (convex, should outperform ISTA)
- LISTA: unrolled CNN denoiser (ConvBlock)
- AT-LISTA: unrolled CNN denoiser + adaptive threshold map (per-pixel, per-channel)
All models compared on a test slice. GPU accelerated.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py, os, time
from pathlib import Path

# ========================== USER CONFIGURATION =======================
RUN_LISTA = True          # Train standard LISTA (CNN denoiser)
RUN_AT_LISTA = True       # Train AT-LISTA (CNN + adaptive threshold)

# Common settings
ACCELERATION = 4
BATCH_SIZE = 2
DATA_DIR = r"C:\Users\MSI-NB\Downloads\knee_singlecoil_val\singlecoil_val"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ISTA / FISTA settings
LAMBDA = 0.0005
TV_WEIGHT = 0.01          # Regularisation weight for TV
FISTA_STEP_SIZE = 1.0     # Full step size (TV is convex)
N_ITER = 100

# LISTA / AT-LISTA settings (groupmate's configuration)
T = 10
FEATURES = 32
EPOCHS = 50
LR = 1e-3

# ========================== DEPENDENCIES ==============================
try:
    import pywt; HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("[WARN] pip install PyWavelets -- needed for ISTA")

try:
    from skimage.restoration import denoise_tv_chambolle; HAS_TV = True
except ImportError:
    HAS_TV = False
    print("[WARN] pip install scikit-image -- needed for TV denoising in FISTA")

try:
    from skimage.metrics import structural_similarity as ssim_func; HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] pip install torch -- needed for LISTA and AT-LISTA")

# ========================== FFT utilities =============================
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))

def center_crop_np(img, shape):
    h, w = img.shape[-2:]
    th, tw = shape
    return img[..., (h-th)//2:(h+th)//2, (w-tw)//2:(w+tw)//2]

def center_crop_tensor(img, shape):
    if img.dim() == 4:
        B, C, H, W = img.shape
    else:
        B, C, H, W = 1, img.shape[0], img.shape[1], img.shape[2]
        img = img.unsqueeze(0)
    th, tw = shape
    h_start = (H - th) // 2
    w_start = (W - tw) // 2
    cropped = img[:, :, h_start:h_start+th, w_start:w_start+tw]
    return cropped.squeeze() if B == 1 else cropped

def compute_psnr(img, ref):
    mse = np.mean((img - ref)**2)
    if mse < 1e-12: return 60.0
    return 10*np.log10(np.max(ref)**2 / mse)

def compute_ssim(img, ref):
    if not HAS_SSIM: return float('nan')
    return ssim_func(img, ref, data_range=ref.max()-ref.min())

# ========================== DATA LOADING =============================
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
    x_gt = center_crop_np(img_full, crop_shape)
    scale = x_gt.max()
    if scale > 0:
        x_gt /= scale
        kspace = kspace / scale
    return kspace, x_gt, crop_shape

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

# ======================== ISTA (wavelet) ===============================
def wavelet_thresh(x, lam, wavelet='db4', level=3):
    def _do(r):
        c = pywt.wavedec2(r, wavelet, level=level)
        c_new = [c[0]]
        for detail in c[1:]:
            c_new.append(tuple(np.sign(d)*np.maximum(np.abs(d)-lam, 0) for d in detail))
        out = pywt.waverec2(c_new, wavelet)
        return out[:x.shape[0], :x.shape[1]]
    return _do(np.real(x)) + 1j * _do(np.imag(x))

def ista_mri(kspace_u, mask, lam, n_iter, x_gt=None, crop=None):
    x = ifft2c(kspace_u)
    psnrs = []
    for _ in range(n_iter):
        res = fft2c(x) * mask - kspace_u
        x = x - ifft2c(res * mask)
        if HAS_PYWT:
            x = wavelet_thresh(x, lam)
        if x_gt is not None:
            xm = center_crop_np(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
    return x, psnrs

# ======================== FISTA (Total Variation) ======================
def tv_denoise_complex(x, weight):
    real = np.real(x)
    imag = np.imag(x)
    real_den = denoise_tv_chambolle(real, weight=weight, channel_axis=None)
    imag_den = denoise_tv_chambolle(imag, weight=weight, channel_axis=None)
    return real_den + 1j * imag_den

def fista_tv_mri(kspace_u, mask, tv_weight, n_iter, step_size=1.0, x_gt=None, crop=None):
    x = ifft2c(kspace_u)
    x_prev = x.copy()
    t = 1.0
    psnrs = []
    for _ in range(n_iter):
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x + (t-1)/t_new * (x - x_prev)
        res = fft2c(y) * mask - kspace_u
        x_prev = x.copy()
        x = y - step_size * ifft2c(res * mask)
        if HAS_TV:
            x = tv_denoise_complex(x, tv_weight)
        t = t_new
        if x_gt is not None:
            xm = center_crop_np(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
    return x, psnrs

# ======================== PyTorch components =========================
if HAS_TORCH:
    class DataConsistency(nn.Module):
        def forward(self, x, kspace_u, mask):
            xc = torch.complex(x[:,0], x[:,1])
            kx = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            ku = torch.complex(kspace_u[:,0], kspace_u[:,1])
            kx = kx * (1 - mask) + ku * mask
            xi = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kx, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            return torch.stack([xi.real, xi.imag], dim=1)

    # -------------------- LISTA (CNN denoiser) -------------------------
    class ConvBlock(nn.Module):
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

    class LISTA_MRI(nn.Module):
        def __init__(self, T=10, features=32):
            super().__init__()
            self.T = T
            self.dc = DataConsistency()
            self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])

        def forward(self, kspace_u, mask):
            ku_complex = torch.complex(kspace_u[:,0], kspace_u[:,1])
            x0 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            x = torch.stack([x0.real, x0.imag], dim=1)
            for k in range(self.T):
                x = self.denoisers[k](x)
                x = self.dc(x, kspace_u, mask)
            return x

    # -------------------- AT-LISTA (CNN + adaptive threshold) ----------
    class AdaptiveThresholdBlock(nn.Module):
        """
        Predicts a per-pixel, per-channel threshold map from the input.
        """
        def __init__(self, channels=2, features=16):
            super().__init__()
            self.threshold_net = nn.Sequential(
                nn.Conv2d(channels, features, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, channels, 3, padding=1),
                nn.Softplus(),   # ensures threshold > 0
            )
            self.scale = nn.Parameter(torch.tensor(0.01))

        def forward(self, x):
            theta = self.threshold_net(x) * self.scale
            return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0)

    class ATLISTA_MRI(nn.Module):
        def __init__(self, T=10, features=32):
            super().__init__()
            self.T = T
            self.dc = DataConsistency()
            self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])
            self.thresholds = nn.ModuleList([AdaptiveThresholdBlock(2, 16) for _ in range(T)])

        def forward(self, kspace_u, mask):
            ku_complex = torch.complex(kspace_u[:,0], kspace_u[:,1])
            x0 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
            x = torch.stack([x0.real, x0.imag], dim=1)
            for k in range(self.T):
                x = self.denoisers[k](x)
                x = self.thresholds[k](x)
                x = self.dc(x, kspace_u, mask)
            return x

    # -------------------- data preparation -----------------------------
    def prepare_torch_data(h5_files, acceleration=4, max_slices=500):
        all_kspace_u = []
        all_gt = []
        all_masks = []
        ref_shape = None
        for fp in h5_files:
            with h5py.File(fp, 'r') as f:
                ks_vol = f['kspace'][()]
            n = ks_vol.shape[0]
            start = max(n//4, 0)
            end = min(3*n//4, n)
            for si in range(start, end, 2):
                kspace = ks_vol[si]
                if ref_shape is None:
                    ref_shape = kspace.shape
                elif kspace.shape != ref_shape:
                    continue
                img_full = np.abs(ifft2c(kspace))
                scale = img_full.max()
                if scale < 1e-6:
                    continue
                kspace = kspace / scale
                W = kspace.shape[1]
                mask = create_mask(W, acceleration=acceleration, seed=si)
                kspace_u = kspace * mask
                gt = np.abs(ifft2c(kspace))
                all_kspace_u.append(np.stack([kspace_u.real, kspace_u.imag], axis=0))
                all_gt.append(gt)
                all_masks.append(mask.reshape(1, -1))
                if len(all_gt) >= max_slices:
                    break
            if len(all_gt) >= max_slices:
                break
        print(f"  Prepared {len(all_gt)} slices (shape {ref_shape}) for training")
        return all_kspace_u, all_gt, all_masks

    def train_model(model, h5_files, model_name, T=10, epochs=50, lr=1e-3, batch_size=2, acceleration=4):
        print(f"\n  Training {model_name} (T={T}, epochs={epochs})...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            count = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                if len(idx) == 0:
                    continue
                ku_batch = torch.tensor(np.stack([all_ku[j] for j in idx]), dtype=torch.float32).to(device)
                mask_batch = torch.tensor(np.stack([all_masks[j] for j in idx]), dtype=torch.float32).to(device)
                gt_batch = [all_gt[j] for j in idx]

                optimizer.zero_grad()
                out = model(ku_batch, mask_batch)
                out_mag = torch.sqrt(out[:,0]**2 + out[:,1]**2 + 1e-8)
                out_cropped = center_crop_tensor(out_mag, crop_shape)
                loss = 0.0
                for b, gt in enumerate(gt_batch):
                    gt_tensor = torch.tensor(center_crop_np(gt, crop_shape), dtype=torch.float32).to(device)
                    loss += torch.mean((out_cropped[b] - gt_tensor)**2)
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

    def inference(model, kspace, mask_1d, crop_shape, device=None):
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            ku = kspace * mask_1d
            ku_t = torch.tensor(np.stack([ku.real, ku.imag])[None], dtype=torch.float32).to(device)
            m_t = torch.tensor(mask_1d.reshape(1, 1, -1), dtype=torch.float32).to(device)
            out = model(ku_t, m_t)
            out_np = out.cpu().numpy()[0]
            mag = np.sqrt(out_np[0]**2 + out_np[1]**2)
            return center_crop_np(mag, crop_shape)

# ========================== MAIN ======================================
def main():
    h5_files = sorted(Path(DATA_DIR).glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files in {DATA_DIR}"); return
    print(f"Found {len(h5_files)} files")

    test_file = h5_files[0]
    kspace, x_gt, crop_shape = load_slice(test_file)
    H, W = kspace.shape
    print(f"Test slice: kspace={kspace.shape}, gt={x_gt.shape}")

    mask = create_mask(W, acceleration=ACCELERATION)
    kspace_u = kspace * mask
    print(f"Sampling rate: {mask.sum()/mask.size:.1%}")

    x_zf = center_crop_np(np.abs(ifft2c(kspace_u)), crop_shape)
    psnr_zf = compute_psnr(x_zf, x_gt)
    print(f"Zero-filled: PSNR={psnr_zf:.2f} dB")

    # ISTA (wavelet)
    print(f"Running ISTA ({N_ITER} iter, lam={LAMBDA})...", end=" ", flush=True)
    t0 = time.time()
    x_ista, psnrs_ista = ista_mri(kspace_u, mask, LAMBDA, N_ITER, x_gt, crop_shape)
    t_ista = time.time() - t0
    x_ista_c = center_crop_np(np.abs(x_ista), crop_shape)
    psnr_ista = compute_psnr(x_ista_c, x_gt)
    print(f"PSNR={psnr_ista:.2f} dB ({t_ista:.1f}s)")

    # FISTA with TV
    if HAS_TV:
        print(f"Running FISTA-TV ({N_ITER} iter, tv_weight={TV_WEIGHT})...", end=" ", flush=True)
        t0 = time.time()
        x_fista, psnrs_fista = fista_tv_mri(kspace_u, mask, TV_WEIGHT, N_ITER, FISTA_STEP_SIZE, x_gt, crop_shape)
        t_fista = time.time() - t0
        x_fista_c = center_crop_np(np.abs(x_fista), crop_shape)
        psnr_fista = compute_psnr(x_fista_c, x_gt)
        print(f"PSNR={psnr_fista:.2f} dB ({t_fista:.1f}s)")
    else:
        psnr_fista = np.nan
        x_fista_c = np.zeros_like(x_ista_c)
        psnrs_fista = [np.nan]

    # Standard LISTA (CNN)
    psnr_lista = None
    x_lista_c = None
    lista_losses = None
    if HAS_TORCH and RUN_LISTA:
        print("\n  Training standard LISTA (CNN denoiser)...")
        train_files = h5_files[1:20]
        model_lista = LISTA_MRI(T=T, features=FEATURES)
        model_lista, _, lista_losses = train_model(
            model_lista, train_files, model_name="LISTA",
            T=T, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, acceleration=ACCELERATION
        )
        x_lista_c = inference(model_lista, kspace, mask, crop_shape)
        psnr_lista = compute_psnr(x_lista_c, x_gt)
        ssim_lista = compute_ssim(x_lista_c, x_gt)
        print(f"  LISTA inference: PSNR={psnr_lista:.2f} dB, SSIM={ssim_lista:.4f}")

    # AT-LISTA (CNN + adaptive threshold)
    psnr_at = None
    x_at_c = None
    at_losses = None
    if HAS_TORCH and RUN_AT_LISTA:
        print("\n  Training AT-LISTA (CNN + adaptive threshold)...")
        train_files = h5_files[1:20]
        model_at = ATLISTA_MRI(T=T, features=FEATURES)
        model_at, _, at_losses = train_model(
            model_at, train_files, model_name="AT-LISTA",
            T=T, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, acceleration=ACCELERATION
        )
        x_at_c = inference(model_at, kspace, mask, crop_shape)
        psnr_at = compute_psnr(x_at_c, x_gt)
        ssim_at = compute_ssim(x_at_c, x_gt)
        print(f"  AT-LISTA inference: PSNR={psnr_at:.2f} dB, SSIM={ssim_at:.4f}")

    # ========================== PLOTS =================================
    # Convergence plot (ISTA/FISTA)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(psnrs_ista, label=f"ISTA (wavelet) {psnr_ista:.1f} dB", lw=2)
    if HAS_TV and not np.isnan(psnr_fista):
        ax.plot(psnrs_fista, label=f"FISTA-TV {psnr_fista:.1f} dB", lw=2)
    if psnr_lista:
        ax.axhline(psnr_lista, color='green', ls='-', lw=2, label=f"LISTA T={T} ({psnr_lista:.1f} dB)")
    if psnr_at:
        ax.axhline(psnr_at, color='orange', ls='-', lw=2, label=f"AT-LISTA T={T} ({psnr_at:.1f} dB)")
    ax.axhline(psnr_zf, color='gray', ls='--', alpha=0.7, label=f"Zero-filled ({psnr_zf:.1f} dB)")
    ax.set_xlabel("Iteration $k$", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("MRI Reconstruction: PSNR vs Iteration (4x)", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "convergence_mri.png"), dpi=150)
    plt.close(fig)

    # Visual comparison
    methods = [('Zero-filled', x_zf, psnr_zf),
               ('ISTA', x_ista_c, psnr_ista)]
    if HAS_TV and not np.isnan(psnr_fista):
        methods.append(('FISTA-TV', x_fista_c, psnr_fista))
    if psnr_lista:
        methods.append(('LISTA', x_lista_c, psnr_lista))
    if psnr_at:
        methods.append(('AT-LISTA', x_at_c, psnr_at))
    methods.append(('Ground Truth', x_gt, None))

    n_panels = len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    for idx, (name, img, psnr) in enumerate(methods):
        title = name
        if psnr is not None:
            title += f"\nPSNR={psnr:.1f} dB"
        axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].axis('off')
    fig.suptitle("Knee MRI Reconstruction (4x acceleration)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "visual_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Training loss curves
    if lista_losses is not None and at_losses is not None:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1, len(lista_losses)+1), lista_losses, 'b-', label='LISTA (CNN)')
        ax.plot(range(1, len(at_losses)+1), at_losses, 'orange', label='AT-LISTA (CNN+thresh)')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss (MSE)")
        ax.set_title("Training Loss Comparison"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "training_loss_comparison.png"), dpi=150)
        plt.close(fig)
    elif at_losses is not None:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1, len(at_losses)+1), at_losses, 'orange', label='AT-LISTA')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss (MSE)")
        ax.set_title("AT-LISTA Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "at_training_loss.png"), dpi=150)
        plt.close(fig)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*60}")
    print(f"{'Zero-filled':<20} {psnr_zf:<12.2f} {compute_ssim(x_zf, x_gt):<10.4f}")
    print(f"{'ISTA (wavelet)':<20} {psnr_ista:<12.2f} {compute_ssim(x_ista_c, x_gt):<10.4f}")
    if HAS_TV and not np.isnan(psnr_fista):
        print(f"{'FISTA-TV':<20} {psnr_fista:<12.2f} {compute_ssim(x_fista_c, x_gt):<10.4f}")
    if psnr_lista:
        print(f"{'LISTA (CNN)':<20} {psnr_lista:<12.2f} {ssim_lista:<10.4f}")
    if psnr_at:
        print(f"{'AT-LISTA (CNN+thresh)':<20} {psnr_at:<12.2f} {ssim_at:<10.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
