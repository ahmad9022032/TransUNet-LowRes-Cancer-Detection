# TransUNet for Low-Resolution Histopathology Cancer Detection

<p align="center">
  <b>Enhanced cancer detection using TransUNet for low-resolution histopathology images across multiple cancer types</b><br>
  <i>Presented at the AACR Annual Meeting 2025 · Cancer Research (2025) 85 (8_Supplement_1): 2431</i><br>
  <a href="https://doi.org/10.1158/1538-7445.AM2025-2431">DOI: 10.1158/1538-7445.AM2025-2431</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-TransUNet-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange" alt="Framework">
  <img src="https://img.shields.io/badge/Task-Segmentation%20%2B%20Classification-green" alt="Task">
  <img src="https://img.shields.io/badge/License-GPL--3.0-lightgrey" alt="License">
</p>

---

## 📌 Overview

Digital histopathology is an indispensable tool for cancer diagnosis and prognosis, but the high cost and limited availability of high-resolution whole-slide imaging (WSI) equipment poses a serious barrier in underdeveloped and resource-limited regions.

This project investigates **Low-Cost Low-Resolution (LCLR)** histopathology images for cancer detection using **TransUNet** — a hybrid architecture that combines the **encoder–decoder structure of U-Net** with **Transformer self-attention blocks**. The model performs **precise segmentation and classification** on noisy, low-resolution histopathology images across **four different cancer sites**:

- 🦷 **Oral cancer**
- 🍽️ **Gastrointestinal cancer**
- 🩺 **Colorectal cancer**
- 🎗️ **Breast cancer**

TransUNet's coarse-to-fine attention refinement enables accurate tumor-region segmentation even in noisy, low-resolution images, and it outperformed all other models we benchmarked.

## 🧠 What We Did

1. **Model benchmarking** — We compared four deep learning architectures on the Tabassum et al. (2020) OSCC dataset to select the best model:
   - AlexNet
   - EfficientNet
   - Vision Transformer (ViT)
   - **TransUNet** ✅ (winner)

2. **Segmentation** — TransUNet's U-Net-style decoder with skip connections and transformer-refined features enabled precise segmentation of cancerous regions in noisy, low-resolution histopathology images, outperforming the other models.

3. **Binary classification (per cancer site)** — The selected TransUNet model was trained on the **HistoVault v1** dataset (17,500 images at 10X and 40X magnification) to classify *cancerous vs. non-cancerous* tissue for each of the four cancer sites.

4. **Multi-class classification (8 classes)** — A single model differentiating **eight classes** (non-cancerous and cancerous tissue for each of the four cancer sites), demonstrating robust cross-site generalization.

5. **Grading & staging (ongoing / future work)** — With the objective of improving precision diagnosis, our ongoing research extends this work toward **cancer grading and staging** together with **explainable AI** techniques for model interpretability.

## 📊 Results

### Model selection — Tabassum et al. (2020) OSCC dataset (1,223 images, 10X & 40X)

| Model | Test Accuracy |
|---|---|
| **TransUNet** | **96.8%** (loss: 3%) — highest among all compared models |
| AlexNet / EfficientNet / ViT | outperformed by TransUNet |

### Per-site binary classification — HistoVault v1 dataset (17,500 images, 10X & 40X)

| Cancer Site | Test Accuracy |
|---|---|
| Oral | **98.7%** |
| Gastrointestinal | **95.9%** |
| Colorectal | **96.1%** |
| Breast | **97.2%** |

### Multi-class classification (8 classes)

| Task | Test Accuracy |
|---|---|
| Non-cancerous + cancerous across all 4 sites (8 classes) | **90.5%** |

## 🏗️ Architecture

The TransUNet implementation in this repo ([model.py](model.py)):

- **Encoder:** ResNet50 backbone (ImageNet pre-trained) extracting hierarchical features with skip connections at four resolution levels
- **Bottleneck:** Stacked Transformer blocks (multi-head self-attention + feed-forward network with layer normalization and dropout) applied to the encoded feature patches
- **Decoder:** U-Net-style upsampling path with convolutional blocks and skip-connection concatenation for coarse-to-fine refinement
- **Head:** Global average pooling + softmax classification layer

**Training configuration** (as reported in the paper): learning rate `1e-5`, batch size `32`, dropout `20%`, sparse categorical cross-entropy loss, evaluated with accuracy and confusion matrices.

## 📁 Repository Structure

```
├── main.py          # Entry point — builds, trains and evaluates the model
├── model.py         # TransUNet architecture (ResNet50 encoder + Transformer + U-Net decoder)
├── data_loader.py   # Generic data loading, balancing and train/val/test splitting
├── trainer.py       # Training loop, evaluation and metric plotting
├── libs.py          # Shared imports
├── LICENSE          # GPL-3.0 license
└── README.md
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### 2. Prepare your data

Organize your dataset as:

```
data_dir/
├── normal/    # non-cancerous .jpg images
└── cancer/    # cancerous .jpg images
```

> The provided [data_loader.py](data_loader.py) is a generic template — adapt it to your own directory structure if needed. It automatically balances classes and creates stratified train/validation/test splits.

### 3. Train

Set your dataset path in [main.py](main.py) and run:

```bash
python main.py
```

Multi-GPU training is supported out of the box via `tf.distribute.MirroredStrategy`. For multi-class training (e.g., the 8-class setup), change `num_classes` in [main.py](main.py) accordingly.

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{paracha2025transunet,
  title   = {Abstract 2431: Enhanced cancer detection using TransUnet for low-resolution histopathology images across multiple cancer types},
  author  = {Paracha, Muhammad Shaiq and Khan, Faisal F. and Riaz, Arsalan and Shirdel, Madina},
  journal = {Cancer Research},
  volume  = {85},
  number  = {8\_Supplement\_1},
  pages   = {2431},
  year    = {2025},
  doi     = {10.1158/1538-7445.AM2025-2431},
  note    = {Proceedings of the AACR Annual Meeting 2025, Chicago, IL}
}
```

**Affiliation:** Precision Medicine Lab, Peshawar, Pakistan

## ⚖️ License

This project is released under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

## 📬 Model Weights

The trained model weights are available **for research purposes** upon request.
Please email: **muhammadahmadkhan316@gmail.com**
