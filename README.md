# 7503gp

mri_full_2.py contains code implementation of MRI image using ISTA FISTA LISTA
my results are
============================================================
Method               PSNR (dB)    SSIM      
------------------------------------------------------------
Zero-filled          26.94        0.5525    
ISTA                 26.72        0.5181    
FISTA                26.30        0.4848    
LISTA (T=10)         28.19        0.5608    
============================================================

The PSNR of LISTA > ISTA > Zero-filled

Note: change the line
