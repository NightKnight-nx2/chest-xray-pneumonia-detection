# Chest X-Ray Pneumonia Detection — Deep Learning Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/ResNet50-Transfer%20Learning-green" />
  <img src="https://img.shields.io/badge/Grad--CAM-Explainability-orange" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

An end-to-end deep learning project for detecting **pneumonia from chest X-ray images** using transfer learning (ResNet50), trained entirely on a local GPU with PyTorch. Includes a native desktop GUI application with real-time Grad-CAM heatmap visualization.

---

## ✨ Features

- 🔬 **Transfer Learning** — ResNet50 pre-trained on ImageNet, fine-tuned with gradual layer unfreezing
- 🎯 **High Recall Focus** — Optimized for medical screening (minimizing missed pneumonia cases)
- 🗺️ **Grad-CAM Explainability** — Visual heatmaps showing which lung regions the model focused on
- 🖥️ **Desktop GUI** — Tkinter-based dark-themed application with drag-and-drop image support
- 📋 **Analysis Logging** — Every prediction is silently logged to `analysis_log.txt`
- 🏋️ **Gradual Unfreezing** — Two-phase training strategy for robust feature learning

---

## 📊 Model Performance (Final Model)

Trained with a proper **80/20 stratified split** from the training set (1,043 validation samples), over **20 epochs** with gradual layer unfreezing.

| Metric               | NORMAL | PNEUMONIA |
|----------------------|:------:|:---------:|
| **Precision**        | 0.98   | 0.78      |
| **Recall**           | 0.53   | **0.99**  |
| **F1-Score**         | 0.69   | 0.87      |
| **Test Accuracy**    | —      | **82%**   |
| **Best Val Accuracy**| —      | **98.2%** |

> [!NOTE]
> In medical screening, **Recall (Sensitivity)** is the critical metric. A missed pneumonia case is far more dangerous than a false alarm. This model detects **99% of pneumonia cases** on the test set.

---

## 🗂️ Project Structure

```
.
├── desktop_app.py          # 🖥️  Main GUI application (run this!)
├── train_pytorch.py        # 🏋️  PyTorch training script (GPU-accelerated)
├── export_onnx.py          # 📦  Export trained model to ONNX format
├── eda.py                  # 📊  Exploratory Data Analysis & visualizations
├── evaluate.py             # 📈  TensorFlow model evaluation (legacy)
├── grad_cam.py             # 🗺️  Grad-CAM heatmap generator (standalone)
├── data_loader.py          # 📂  TensorFlow data pipeline (legacy)
├── model.py                # 🧠  TensorFlow model definition (legacy)
├── visualize_aug.py        # 🎨  Data augmentation visualizer
├── artifacts/              # 📁  Generated plots and charts (auto-created)
├── data/
│   └── chest_xray/         # 📁  Dataset (not included — see below)
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       └── test/
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/NightKnight-nx/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> **PyTorch with CUDA:** If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch for significantly faster training:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> ```
> CPU-only installations will work but training will be much slower.

### 4. Download the Dataset

Download the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:  
🔗 [kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Extract it so the folder structure looks like:
```
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

### 5. Train the Model

```bash
python train_pytorch.py
```

This will:
- Automatically detect and use your NVIDIA GPU (if available)
- Run 20 epochs with two-phase gradual unfreezing
- Save the best model as `best_pytorch_model.pth`
- Export training curves and confusion matrix to `artifacts/`

### 6. Run the Desktop Application

```bash
python desktop_app.py
```

---

## 🖥️ Desktop Application

The GUI application provides:

| Panel | Description |
|-------|-------------|
| **Original X-Ray** | The uploaded chest X-ray image |
| **Grad-CAM Heatmap** | Real-time heatmap showing which lung regions drove the model's decision |
| **Prediction** | 🔴 PNEUMONIA or 🟢 NORMAL with confidence score |
| **Analysis History** | View all past analyses logged in `analysis_log.txt` |

---

## 🧠 Architecture & Training Strategy

### Model
- **Base**: ResNet50 (ImageNet pre-trained)
- **Head**: `Dropout(0.5)` → `Linear(2048, 1)`
- **Loss**: `BCEWithLogitsLoss` with class-weighted `pos_weight`

### Two-Phase Training (Gradual Unfreezing)
```
Phase 1 (Epochs 1–6):   Only FC head trained         LR = 1e-3
Phase 2 (Epochs 7–20):  layer4 + FC head fine-tuned  LR = 1e-4 / 1e-3
                         Cosine Annealing LR Scheduler
```

### Data Augmentation (Training)
- Random Resized Crop (scale 0.75–1.0)
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast)
- ImageNet Normalization

### Validation Strategy
- Proper **80/20 stratified split** from training data (4,173 train / 1,043 val)
- The built-in `val/` folder (only 16 images) was intentionally excluded

---

## 📈 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to `layer4[-1]` of ResNet50. The heatmap always highlights which lung regions contribute to the **PNEUMONIA** score, making subtle findings visible even in NORMAL predictions.

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for the full list.

Key dependencies:
- `torch` ≥ 2.6 (+ CUDA recommended)
- `torchvision` ≥ 0.21
- `Pillow`, `opencv-python`, `numpy`
- `matplotlib`, `seaborn`, `scikit-learn`

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset: [Paul Mooney — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle
- Architecture: [He et al., 2015 — Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Explainability: [Selvaraju et al., 2017 — Grad-CAM](https://arxiv.org/abs/1610.02391)
