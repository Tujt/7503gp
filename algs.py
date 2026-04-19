# algs.py
import numpy as np
import torch
import torch.nn as nn
import pywt
from torch.utils.data import Dataset, DataLoader
import os
import h5py

# --- 通用工具函数 ---
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
    if mse < 1e-12:
        return 60.0
    return 10 * np.log10(np.max(ref)**2 / mse)

def compute_ssim(img, ref):
    from skimage.metrics import structural_similarity as ssim_func
    return ssim_func(img, ref, data_range=ref.max()-ref.min())

# --- 传统算法 (ISTA/FISTA) ---
def wavelet_thresh(x, lam, wavelet='db4', level=3):
    def _do(r):
        c = pywt.wavedec2(r, wavelet, level=level)
        c_new = [c[0]]
        for detail in c[1:]:
            c_new.append(tuple(np.sign(d) * np.maximum(np.abs(d) - lam, 0) for d in detail))
        out = pywt.waverec2(c_new, wavelet)
        return out[:x.shape[0], :x.shape[1]]
    return _do(np.real(x)) + 1j * _do(np.imag(x))

def ista_mri(kspace_u, mask, lam, n_iter, x_gt=None, crop=None, step_size=0.5):
    x = ifft2c(kspace_u)
    psnrs = []
    for _ in range(n_iter):
        # 数据一致性
        res = fft2c(x) * mask - kspace_u
        x = x - step_size*ifft2c(res * mask)
        # 正则化 (小波阈值)
        x = wavelet_thresh(x, lam)
        # 记录指标
        if x_gt is not None:
            xm = center_crop(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
    return x, psnrs

def fista_mri(kspace_u, mask, lam, n_iter, x_gt=None, crop=None, step_size=0.5):
    x = ifft2c(kspace_u)
    x_prev = x.copy() # 用于计算动量
    t = 1.0           # 动量系数
    psnrs = []
    
    for k in range(n_iter):
        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x + (t-1)/t_new * (x - x_prev)
        
        res = fft2c(y) * mask - kspace_u
        grad = ifft2c(res * mask)
        x_prev = x.copy()
        x = y - step_size * grad
        
        x = wavelet_thresh(x, lam * step_size) 

        t = t_new
        
        # 记录指标
        if x_gt is not None:
            xm = center_crop(np.abs(x), crop) if crop else np.abs(x)
            psnrs.append(compute_psnr(xm, x_gt))
            
    return x, psnrs


# --- 深度学习模型 (LISTA) ---
class DataConsistency(nn.Module):
    def forward(self, x, kspace_u, mask):
        xc = torch.complex(x[:, 0], x[:, 1])
        kx = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        ku = torch.complex(kspace_u[:, 0], kspace_u[:, 1])
        kx = kx * (1 - mask) + ku * mask
        xi = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kx, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        return torch.stack([xi.real, xi.imag], dim=1)

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
        return x + self.net(x) # 残差连接

class LISTA_MRI(nn.Module):
    def __init__(self, T=10, features=32):
        super().__init__()
        self.T = T
        self.dc = DataConsistency()
        self.denoisers = nn.ModuleList([ConvBlock(2, features) for _ in range(T)])

    def forward(self, kspace_u, mask):
        ku_complex = torch.complex(kspace_u[:, 0], kspace_u[:, 1])
        x0 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ku_complex, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))
        x = torch.stack([x0.real, x0.imag], dim=1)
        
        for k in range(self.T):
            x = self.denoisers[k](x)
            x = self.dc(x, kspace_u, mask)
        return x

# --- LISTA 工具函数 ---
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

def prepare_torch_data(h5_files, acceleration=4, max_slices=200):
    """
    修改点 1: 增加 acceleration 参数
    """
    all_kspace_u, all_gt, all_masks = [], [], []
    ref_shape = None
    for fp in h5_files:
        with h5py.File(fp, 'r') as f:
            ks_vol = f['kspace'][()]
            n = ks_vol.shape[0]
            start, end = max(n//4, 0), min(3*n//4, n)
            for si in range(start, end, 2):
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
                # 修改点 2: 使用传入的 acceleration 参数
                mask = create_mask(W, acceleration=acceleration, seed=si)
                kspace_u = kspace * mask
                gt = np.abs(ifft2c(kspace))
                all_kspace_u.append(np.stack([kspace_u.real, kspace_u.imag], axis=0))
                all_gt.append(gt)
                all_masks.append(mask.reshape(1, -1))
                if len(all_gt) >= max_slices: break
    return all_kspace_u, all_gt, all_masks

def train_lista_model(train_files, T=10, epochs=20, lr=1e-3, batch_size=2, acceleration=4, save_path=None):
    """
    修改点 3: 让训练函数也接收 acceleration 参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 修改点 4: 将 acceleration 传给 prepare_torch_data
    all_ku, all_gt, all_masks = prepare_torch_data(train_files, acceleration, max_slices=100) # 小样本演示
    crop_h = min(g.shape[0] for g in all_gt)
    crop_w = min(g.shape[1] for g in all_gt)
    crop_shape = (crop_h, crop_w)

    model = LISTA_MRI(T=T, features=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        perm = np.random.permutation(len(all_gt))
        
        for i in range(0, len(all_gt), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) == 0: continue
            
            optimizer.zero_grad()
            
            ku_batch = torch.tensor(np.stack([all_ku[j] for j in idx]), dtype=torch.float32).to(device)
            mask_batch = torch.tensor(np.stack([all_masks[j] for j in idx]), dtype=torch.float32).to(device)
            
            out = model(ku_batch, mask_batch)
            out_mag = torch.sqrt(out[:,0]**2 + out[:,1]**2 + 1e-8)
            
            loss = 0
            for b, j in enumerate(idx):
                out_crop = center_crop(out_mag[b].unsqueeze(0), crop_shape).squeeze()
                gt_crop = torch.tensor(center_crop(all_gt[j], crop_shape), dtype=torch.float32).to(device)
                loss += criterion(out_crop, gt_crop)
            
            loss = loss / len(idx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(all_gt):.6f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model, crop_shape

def lista_inference(model, kspace, mask_1d, crop_shape, device=None):
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
        return center_crop(mag, crop_shape)