# 7503gp

## Results in mri_full_2.py

Knee MRI reconstruction at 4x acceleration on fastMRI singlecoil data.

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Zero-filled | 26.94 | 0.5525 |
| ISTA (100 iter) | 26.72 | 0.5181 |
| FISTA (100 iter) | 26.30 | 0.4848 |
| **LISTA (T=10)** | **28.19** | **0.5608** |

### Note: change the line 31 with your own directory
