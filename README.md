# 7503gp

## Results

Knee MRI reconstruction at 4x acceleration on fastMRI singlecoil data.

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Zero-filled | 26.94 | 0.5525 |
| ISTA (100 iter) | 26.72 | 0.5181 |
| FISTA (100 iter) | 26.30 | 0.4848 |
| **LISTA (T=10)** | **28.19** | **0.5608** |

### Note: change the line 31 with your own directory
### Further improvement: consider changing to a smaller lambda.
`convergence_mri.png` lista_training_loss.png` `visual_comparison.png` are also results from `mri_full_2.py`

## Results for `at_lista.py`

Knee MRI reconstruction at 4x acceleration on fastMRI singlecoil data.

| Method | Params | PSNR (dB) | SSIM |
|---|---|---|---|
| Zero-filled | — | 26.94 | 0.5525 |
| LISTA (T=10) | 104,340 | 28.16 | 0.5614 |
| **AT-LISTA (T=10)** | **110,290** | **28.22** | **0.5624** |

### Note: change line 28 with your own directory
`error_maps.png`,`lista_vs_atlista.png`, and `training_loss_comparison_atlista.png` are results from `at_lista.py`
