<div align="center">

# Deep Temporal Representation Learning for Sample-Efficient Human Activity Recognition from Video Sequences

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FPICC67314.2025.11291542-blue.svg?style=for-the-badge)](https://doi.org/10.1109/PICC67314.2025.11291542)
[![Publisher](https://img.shields.io/badge/Publisher-IEEE-blue.svg?style=for-the-badge)](https://ieeexplore.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Official Implementation and Associated Data**

*2025 International Conference on Power, Instrumentation, Control, and Computing (PICC)*<br>
*Thrissur, India | October 09-11, 2025*

</div>

---

## Abstract

Human Activity Recognition (HAR) from video is critical for surveillance, healthcare, and human-robot interaction. This study evaluates four deep learning models which are 3D CNN, CNN-LSTM, VideoMAE, and V-JEPA 2. Across convolutional, recurrent, and transformer paradigms for seven-class HAR. 

Implemented in a standardized pipeline, the 3D CNN-LSTM hybrid achieves the highest accuracy (**96.23%**), effectively modeling long-range temporal dependencies. Transformer models (VideoMAE, V-JEPA 2) leverage global attention but slightly underperform the hybrid approach. Quantitative analysis reveals persistent challenges in fine-grained activity discrimination and confirms the efficacy of recurrent-convolutional fusion for temporal modeling. We establish performance baselines and derive design principles for HAR systems.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Architectures Evaluated](#architectures-evaluated)
- [Getting Started](#getting-started)
- [Citation](#citation)

---

## Repository Structure

The repository is structured to facilitate reproducibility and ease of access to the experimental configurations detailed in the publication.

```text
.
├── 3d_cnn.py                # Core source code for 3D CNN model training and evaluation
├── 3d_cnn_output.txt        # Captured training logs, epoch metrics, and classification reports
└── dataset/                 # Dataset organization directory
    ├── train/               # Training split (70%)
    ├── val/                 # Validation split (15%)
    └── test/                # Testing split (15%)
```
> **Note:** The physical `.mp4` files are not hosted in this repository due to storage constraints. Please refer to `dataset/README.md` for download and organization instructions.

---

## Architectures Evaluated

This research standardizes and evaluates four primary architectural paradigms for seven-class human activity recognition:

1. **3D CNN** - Utilizes volumetric convolutions to extract spatial-temporal features.
2. **CNN-LSTM** - **(Highest Accuracy: 96.23%)** Fuses convolutional feature extraction with recurrent temporal modeling.
3. **VideoMAE** - Employs Masked Autoencoders for self-supervised video pre-training and global attention.
4. **V-JEPA 2** - Utilizes a Joint-Embedding Predictive Architecture for advanced temporal learning.

---

## Getting Started

### Prerequisites
To run the provided scripts, ensure your environment is configured with the following dependencies:
- Python 3.8+
- TensorFlow / Keras
- OpenCV (`opencv-python-headless`)
- Scikit-learn
- Matplotlib & Seaborn

### Dataset Preparation
The **Human Activity Recognition Video Dataset** used for this research can be downloaded from Kaggle. See `dataset/README.md` for the direct link. Once downloaded, partition the `.mp4` files into the respective `train`, `val`, and `test` directories matching the 7 classes evaluated in the study.

### Execution
To execute the 3D CNN pipeline:
```bash
python 3d_cnn.py
```

---

## Citation

If you find this repository or our research beneficial to your work, please consider citing the associated IEEE publication:

```bibtex
@inproceedings{palliparambil2025deep,
  title={Deep Temporal Representation Learning for Sample-Efficient Human Activity Recognition from Video Sequences},
  author={Palliparambil, Athul Joe Joseph and Shaji, Anandhu P and Rajan, Rajeev},
  booktitle={2025 International Conference on Power, Instrumentation, Control, and Computing (PICC)},
  year={2025},
  organization={IEEE},
  doi={10.1109/PICC67314.2025.11291542},
  url={https://doi.org/10.1109/PICC67314.2025.11291542}
}
```

<div align="center">
  <br>
  <em>Published by IEEE Xplore | 23 December 2025</em>
</div>
