# Deep Temporal Representation Learning for Sample-Efficient Human Activity Recognition from Video Sequences

This repository contains the official implementation of the models and evaluation pipelines described in the IEEE PICC 2025 paper: *Deep Temporal Representation Learning for Sample-Efficient Human Activity Recognition from Video Sequences*.

This study evaluates four deep learning models across convolutional, recurrent, and transformer paradigms for seven-class Human Activity Recognition (HAR):
1. **3D CNN**
2. **CNN-LSTM** (Highest performing: 96.23% accuracy)
3. **VideoMAE**
4. **V-JEPA 2**

Implemented in a standardized pipeline, the 3D CNN-LSTM hybrid effectively models long-range temporal dependencies. Transformer models (VideoMAE, V-JEPA 2) leverage global attention but slightly underperform the hybrid approach. Quantitative analysis reveals persistent challenges in fine-grained activity discrimination and confirms the efficacy of recurrent-convolutional fusion for temporal modeling.

## Installation

We recommend using Python 3.8 or later. The codebase depends on TensorFlow/Keras for the convolutional and recurrent baselines, and PyTorch/Hugging Face for the transformer architectures. 

To set up the environment, install the following core dependencies :

```bash
pip install numpy pandas scikit-learn opencv-python-headless matplotlib seaborn tqdm
pip install tensorflow  # For 3D CNN and CNN-LSTM
```

## Dataset

The dataset consists of 1,113 videos categorized into seven distinct fine-grained activity classes (`Clapping`, `Meet and Split`, `Sitting`, `Standing Still`, `Walking`, `Walking While Reading Book`, `Walking While Using Phone`).

To maintain class balance, the dataset utilizes stratified splits:
- 70% Training
- 15% Validation
- 15% Testing

> **Note:** Due to storage constraints, the raw video files are not bundled in this repository. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset) and organize the `.mp4` files into the respective `dataset/train/`, `dataset/val/`, and `dataset/test/` directories prior to execution.

## Usage

### Training the baseline models

To train the baseline 3D CNN architecture on the prepared dataset, execute the primary script:

```bash
python 3d_cnn.py
```

This will run the full training pipeline, evaluate the model on the test split, and produce classification reports along with confusion matrices in the output directory.

## Experimental Results

The models were evaluated comprehensively over 30 epochs with 50 videos per class using a standard hardware environment (Apple MacBook Pro M3 Pro / Google Colab TPU v4). The table below summarizes the performance metrics from the study.

| Architecture | Overall Accuracy | Macro F1-Score |
| :--- | :---: | :---: |
| 3D CNN | 83.00% | 0.7000 |
| **CNN-LSTM** | **96.23%** | **0.9628** |
| Transformer VideoMAE | 94.00% | 0.9200 |
| Transformer V-JEPA2 | 91.00% | 0.9000 |

## Citing this work

If you use this code or data in your research, please cite our paper:

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

## License and Disclaimer

This project is licensed under the MIT License. 

*This is an independent academic research project and is not an officially supported Google product.*
