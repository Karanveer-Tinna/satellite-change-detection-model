## High-Precision Satellite Change Detection using Deep Learning
### Overview

This project presents a deep learning-based framework for detecting land-cover changes from multi-temporal satellite imagery. The system performs semantic segmentation on satellite images and compares predictions across different time periods to identify environmental and urban transformations. The framework supports applications such as urban growth monitoring, deforestation tracking, and waterbody dynamics analysis.

### Problem Statement
Manual monitoring of large-scale geographic changes from satellite imagery is time-consuming and expensive. This project aims to automate change detection using deep learning-based semantic segmentation techniques.

### Methodology
1) Image preprocessing and augmentation
2) CLAHE-based contrast enhancement
3) Semantic segmentation using DeepLabV3+, PSPNet, and SegFormer
4) Generation of land-cover masks
5) Dual-temporal mask comparison for change detection
6) Visualization of detected environmental changes