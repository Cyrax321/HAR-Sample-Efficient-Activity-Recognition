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

## 📖 Abstract

Human Activity Recognition (HAR) from video is critical for surveillance, healthcare, and human-robot interaction. This study evaluates four deep learning models which are 3D CNN, CNN-LSTM, VideoMAE, and V-JEPA 2. Across convolutional, recurrent, and transformer paradigms for seven-class HAR. 

Implemented in a standardized pipeline, the 3D CNN-LSTM hybrid achieves the highest accuracy (**96.23%**), effectively modeling long-range temporal dependencies. Transformer models (VideoMAE, V-JEPA 2) leverage global attention but slightly underperform the hybrid approach. Quantitative analysis reveals persistent challenges in fine-grained activity discrimination and confirms the efficacy of recurrent-convolutional fusion for temporal modeling. We establish performance baselines and derive design principles for HAR systems.

---

## 📑 Table of Contents
- [Dataset Details](#dataset-details)
- [System Architectures Evaluated](#system-architectures-evaluated)
- [Experimental Setup & Hardware](#experimental-setup--hardware)
- [Performance Results](#performance-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Citation](#citation)

---

## 🗃️ Dataset Details

The dataset consists of **1,113 videos** systematically curated into seven distinct fine-grained activity classes:
1. `Clapping`
2. `Meet and Split`
3. `Sitting`
4. `Standing Still`
5. `Walking`
6. `Walking While Reading Book`
7. `Walking While Using Phone`

To tackle class imbalance and improve model generalization, rigorous data augmentation techniques (rotation, flipping, zooming, brightness adjustment) were applied. The dataset was organized into non-overlapping stratified splits:
- **70% Training**
- **15% Validation**
- **15% Testing**

---

## 🧠 System Architectures Evaluated

This research standardizes and extensively evaluates four primary architectural paradigms for video-based activity recognition:

### 1. 3D Convolutional Neural Network (3D CNN)
The 3D CNN processes 30-frame sequences at 64×64 resolution. It comprises three 3D convolutional layers (32, 64, 128 filters) with ReLU activations and batch normalization, mapping spatiotemporal features into a 256-neuron dense classifier.

### 2. CNN-LSTM Hybrid *(Best Performing)*
This architecture utilizes a 2D CNN (64 and 128 filters) to extract spatial frames at 224×224 resolution, routing the sequence features into LSTM layers. The LSTM maintains memory states across 16-frame segments to robustly model long-range temporal dependencies.

### 3. Video Masked Autoencoder (VideoMAE)
Utilizes a transformer backbone pre-trained with 80% spatiotemporal masking. Input videos are divided into 16-frame cubes tokenized into patches. The self-attention mechanisms evaluate relationships between all frames dynamically.

### 4. V-JEPA 2 (Video Joint-Embedding Predictive Architecture)
Employs self-supervised learning focused on latent space prediction. Processing 8-frame segments resized to 128×128, a large vision transformer encoder (307M parameters) predicts future latent states utilizing historical temporal contexts.

---

## 💻 Experimental Setup & Hardware

To ensure a fair evaluation pipeline, all models were trained using identical overarching hyperparameters:
* **Epochs:** 30
* **Batch distribution:** 50 videos per class 
* **Hardware utilized:**
  * Apple MacBook Pro M3 Pro (18GB RAM) for localized rapid prototyping and model definitions.
  * Google Colab TPU v4 for highly parallelized, accelerated transformer training.

---

## 📊 Performance Results

The CNN-LSTM hybrid decisively outperformed all standalone convolutional and modern transformer variants, demonstrating superior long-term feature aggregation.

| Model Architecture | Overall Accuracy | Macro F1-Score |
| :--- | :---: | :---: |
| **3D CNN** | 83.00% | 0.7000 |
| **CNN-LSTM** | **96.23%** | **0.9628** |
| **Transformer VideoMAE** | 94.00% | 0.9200 |
| **Transformer V-JEPA2** | 91.00% | 0.9000 |

*Analysis reveals that while shallow layers successfully isolate fine-grained visual details, recurrent architectures excel at aggregating deeper temporal semantics over time.*

---

## 📂 Repository Structure

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

## 🚀 Getting Started

### Prerequisites
To run the provided scripts, ensure your environment is configured with the following dependencies:
- Python 3.8+
- TensorFlow / Keras / PyTorch (Depending on the model evaluated)
- OpenCV (`opencv-python-headless`)
- Scikit-learn
- Matplotlib & Seaborn

### Dataset Preparation
The **Human Activity Recognition Video Dataset** used for this research can be downloaded from Kaggle. See `dataset/README.md` for the direct link. Once downloaded, partition the `.mp4` files into the respective `train`, `val`, and `test` directories.

### Execution
To execute the 3D CNN baseline pipeline:
```bash
python 3d_cnn.py
```

---

## 📝 Citation

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
