# Dual_Domain_Learning

## Abstract

***JPEG*** compression brings artifacts into the compressed image, which not only degrade visual quality, but also affect the performance of other image processing tasks. 

To address this issue, many learning-based compression artifacts removal methods have been developed in recent years, with remarkable success. However, existing learningbased methods generally only exploit spatial information and lack exploration of frequency domain information. 

Exploring frequency domain information is critical because ***JPEG*** compressionis actually performed in the frequency domain using the ***Discrete Cosine Transform (DCT)***. To effectively leverage information from both the spatial and frequency domains, we propose a novel ***Dual-Domain Learning Network for JPEG artifacts removal (D2LNet)***. 

Our approach first transforms the spatial domain image to the frequency domain by the ***fast Fourier transform (FFT)***. We then introduce two core modules, ***Amplitude Correction Module (ACM)*** and ***Phase Correction Module (PCM)***, which facilitate interactive learning of spatial and frequency domain information. 

Extensive experimental results performed on color and grayscale images have clearly demonstrated that our method achieves better results than the previous state-of-the-art methods. 

## Model

The architecture of our network, which consists of two main modules: 
  
the ***Amplitude Correction Module (ACM)*** and the ***Phase Correction Module (PCM)***.
  
Specifically, the ***ACM*** restores the amplitude spectrum of degraded images to remove JPEG artifacts, and the ***PCM*** restores the phase spectrum information to refine the highfrequency information.

![model](https://github.com/YeunkSuzy/Dual_Domain_Learning/assets/113883547/39d67d15-8bec-4e38-929e-37c350bfe621)

## Results

___Table 1___. ***PSNR/SSIM/PSNR-B*** results of our method comparaed to other nine methods in three datasets, with the best outcomes being highlighted in red.

![1](https://github.com/YeunkSuzy/Dual_Domain_Learning/assets/113883547/e439d374-b39d-488a-ac5d-01ef0e3ede5e)

___Table 2___. ***PSNR/SSIM/PSNR-B*** results of different methods on the three color datasets, with the best outcomes being highlighted in red.

![2](https://github.com/YeunkSuzy/Dual_Domain_Learning/assets/113883547/2a5fe5d6-1fb1-496f-b1db-be0fc47f55d5)

___Table 3___. The results of the ablation experiments conducted on the three datasets.

![3](https://github.com/YeunkSuzy/Dual_Domain_Learning/assets/113883547/59db89e0-260c-448b-be12-eae7c21d9e8e)

### Our code based on BasicSR.
