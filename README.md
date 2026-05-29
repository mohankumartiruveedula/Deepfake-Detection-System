<p align="center">
  <h1 align="center">🛡️ Deepfake Detection System</h1>
  <p align="center">
    <strong>High-accuracy deepfake detection powered by EfficientNet-B4 and Spatial Rich Model frequency analysis</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-key-features">Features</a> •
    <a href="#-cli-usage">CLI</a> •
    <a href="#-api-reference">API</a> •
    <a href="#-training">Training</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x"/>
  <img src="https://img.shields.io/badge/EfficientNet-B4-00C853?style=for-the-badge&logo=tensorflow&logoColor=white" alt="EfficientNet-B4"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-89.64%25-brightgreen?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.957-blue?style=flat-square" alt="AUC-ROC"/>
  <img src="https://img.shields.io/badge/Dataset-DF40%20(NeurIPS%202024)-purple?style=flat-square" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Methods-40%20Manipulation%20Types-red?style=flat-square" alt="Methods"/>
</p>

---

## 📖 Overview

**Deepfake Detection System** is an end-to-end pipeline for identifying AI-generated and manipulated facial imagery. It combines a fine-tuned **EfficientNet-B4** backbone with a **Spatial Rich Model (SRM)** frequency filter to jointly analyse RGB pixels *and* high-frequency noise residuals — the subtle forensic traces that generative models leave behind.

Trained on the **DF40 dataset** (NeurIPS 2024) spanning **40 distinct manipulation methods** and **72,424 images**, the system achieves **89.64% accuracy** and **0.957 AUC-ROC** on held-out test data. It ships with a modern React web UI, a FastAPI backend, and a powerful CLI — everything needed to go from a single image to a verified prediction in seconds.

### ✨ Highlights

| Metric | Value |
|:--|:--|
| **Accuracy** | 89.64% |
| **AUC-ROC** | 0.957 |
| **Dataset** | DF40 — 40 manipulation methods, 72,424 images |
| **Backbone** | EfficientNet-B4 (pretrained on ImageNet) |
| **Input Channels** | 6 (RGB + SRM high-frequency noise) |
| **Output Classes** | `REAL` · `FAKE` · `UNCERTAIN` |

---

## 🏗️ Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              INPUT IMAGE                    │
                         └────────────────────┬────────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────────┐
                         │     MediaPipe BlazeFace Detection            │
                         │     (5% tight-margin face crop)             │
                         └────────────────────┬────────────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              │                               │
                              ▼                               ▼
                    ┌──────────────────┐            ┌──────────────────┐
                    │   RGB Channels   │            │   SRM Filters    │
                    │   (3 channels)   │            │   (3 channels)   │
                    │                  │            │  High-frequency  │
                    │  Visual content  │            │  noise residuals │
                    └────────┬─────────┘            └────────┬─────────┘
                             │                               │
                             └───────────┬───────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │         6-Channel Concatenation              │
                         │           [RGB ⊕ SRM Noise]                 │
                         └────────────────────┬────────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────────┐
                         │          EfficientNet-B4 Backbone            │
                         │    (Modified first conv: 6 → 48 channels)   │
                         │    ImageNet pretrained weights (RGB half)    │
                         └────────────────────┬────────────────────────┘
                                              │
                                              ▼
                         ┌─────────────────────────────────────────────┐
                         │        Global Average Pooling + FC           │
                         │            Dropout (p=0.4)                  │
                         │          Softmax → 2 classes                │
                         └────────────────────┬────────────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                    ┌──────────────────┐            ┌──────────────────┐
                    │  Confidence Gate │            │   Grad-CAM       │
                    │                  │            │   Heatmap        │
                    │  ≥ threshold →   │            │   Explainability │
                    │  REAL / FAKE     │            │   Visualization  │
                    │  < threshold →   │            │                  │
                    │  UNCERTAIN       │            │                  │
                    └──────────────────┘            └──────────────────┘
```

> **Why SRM?** Generative models produce visually convincing images but struggle to replicate the natural noise patterns found in camera-captured photos. SRM high-pass filters extract these noise residuals, giving the network a second "forensic" view of the image that is robust across different manipulation techniques.

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🔬 Detection Engine
- **6-Channel Dual-Stream Input** — RGB pixels + SRM high-frequency noise residuals analyzed jointly
- **EfficientNet-B4 Backbone** — ImageNet-pretrained, fine-tuned for forensic classification
- **Confidence-Based Rejection** — 3-class output (`REAL` / `FAKE` / `UNCERTAIN`) reduces false positives
- **Test-Time Augmentation (TTA)** — Horizontal flip ensembling for more robust predictions

</td>
<td width="50%">

### 🧠 Explainability
- **Grad-CAM Heatmaps** — Visual explanations showing *where* the model detects manipulation
- **Per-prediction confidence scores** — Transparent probability outputs for both classes
- **Uncertainty flagging** — Low-confidence predictions explicitly marked rather than forced

</td>
</tr>
<tr>
<td>

### ⚡ Training Pipeline
- **MixUp Augmentation** — Convex combinations of training pairs for better generalization
- **Label Smoothing** — Soft targets to prevent overconfident predictions
- **Cosine Annealing LR** — Smooth learning rate decay with warm restarts
- **Mixed Precision (AMP)** — FP16 training for 2× speed on modern GPUs

</td>
<td>

### 🌐 Full-Stack Deployment
- **FastAPI Backend** — Async REST API with automatic OpenAPI documentation
- **React + Vite Frontend** — Glassmorphism dark-theme UI with drag-and-drop upload
- **Standalone HTML** — Zero-dependency single-file interface for instant use
- **CLI Interface** — Process images, folders, and videos from the command line

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Node.js 18+ (only if using Vite frontend)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

### 2. Download the Model Checkpoint

Download `best_model.pth` and place it in the checkpoints directory:

```
inference engine/outputs/checkpoints/best_model.pth
```

> [!IMPORTANT]
> The model checkpoint is required for inference. Ensure the file is placed at the exact path above before running the backend or CLI.

### 3. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the Backend Server

```bash
python server.py
```

The API server will launch at `http://localhost:8000`. You can verify it's running by visiting `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Launch the Frontend

You have two options:

<table>
<tr>
<td>

**Option A — Zero Setup** ⚡

Open the standalone HTML file directly in your browser:

```
frontend/standalone.html
```

No build tools, no dependencies — just double-click and go.

</td>
<td>

**Option B — Vite Dev Server** 🔧

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173` with hot reload.

</td>
</tr>
</table>

### 6. Detect!

Upload an image through the web interface and get an instant prediction with a confidence score and Grad-CAM heatmap. 🎉

---

## 💻 CLI Usage

The CLI supports single images, directories, and video files:

```bash
cd "inference engine"

# Single image
python inference.py --source face.jpg

# Single image with Grad-CAM visualization
python inference.py --source face.jpg --visualise

# Batch process a folder of images
python inference.py --source folder/

# Video analysis (sample every 4th frame)
python inference.py --source video.mp4 --sample-every 4
```

**Example Output:**

```
─────────────────────────────────────────
  Deepfake Detection Result
─────────────────────────────────────────
  File:        face.jpg
  Prediction:  FAKE
  Confidence:  94.2%
  Real prob:   0.058
  Fake prob:   0.942
─────────────────────────────────────────
```

---

## 📡 API Reference

### `POST /detect`

Analyze an uploaded image for deepfake manipulation.

**Request**

```
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|:--|:--|:--|
| `file` | `UploadFile` | Image file (JPEG, PNG, WebP) |

**cURL Example**

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg"
```

**Response** `200 OK`

```json
{
  "prediction": "FAKE",
  "confidence": 0.942,
  "real_probability": 0.058,
  "fake_probability": 0.942,
  "gradcam": "data:image/png;base64,iVBORw0KGgo..."
}
```

| Field | Type | Description |
|:--|:--|:--|
| `prediction` | `string` | `REAL`, `FAKE`, or `UNCERTAIN` |
| `confidence` | `float` | Confidence of the predicted class (0–1) |
| `real_probability` | `float` | Probability of the image being real |
| `fake_probability` | `float` | Probability of the image being fake |
| `gradcam` | `string` | Base64-encoded Grad-CAM heatmap overlay |

---

## 🏋️ Training

The model was trained on the **DF40 dataset** (NeurIPS 2024) containing 72,424 face images across 40 manipulation methods.

**Training Configuration:**

| Hyperparameter | Value |
|:--|:--|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Input Size | 380 × 380 × 6 channels |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts |
| Batch Size | 32 |
| Epochs | 30 |
| Augmentation | MixUp (α=0.2), HFlip, Color Jitter |
| Label Smoothing | 0.1 |
| Precision | Mixed (AMP FP16) |

```bash
cd "inference engine"
python train.py
```

> [!NOTE]
> Training requires the DF40 dataset to be downloaded and paths configured in `config.py`. A CUDA-capable GPU with ≥12 GB VRAM is recommended.

---

## 📁 Project Structure

```
Deepfake-Detection-System/
│
├── frontend/                        # React + Vite web interface
│   ├── src/
│   │   ├── App.jsx                  # Main React component
│   │   └── index.css                # Glassmorphism dark theme
│   ├── standalone.html              # Zero-dependency HTML version
│   └── package.json                 # Node.js dependencies
│
├── backend/                         # FastAPI inference server
│   ├── server.py                    # REST API endpoint (/detect)
│   └── requirements.txt            # Python dependencies
│
├── inference engine/                # Core ML pipeline
│   ├── model.py                     # EfficientNet-B4 + SRM architecture
│   ├── inference.py                 # Face detection, prediction, Grad-CAM
│   ├── config.py                    # Hyperparameters & paths
│   ├── train.py                     # Training script
│   ├── dataset.py                   # Data loading & augmentation
│   ├── evaluate.py                  # Metrics & evaluation plots
│   └── outputs/
│       └── checkpoints/
│           └── best_model.pth       # Trained model weights
│
├── DOCUMENTATION.md                 # Full technical documentation
└── README.md                        # ← You are here
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="20%"><strong>Category</strong></td>
<td align="center" width="30%"><strong>Technology</strong></td>
<td align="center" width="50%"><strong>Purpose</strong></td>
</tr>
<tr>
<td>🧠 Model</td>
<td>EfficientNet-B4</td>
<td>Feature extraction backbone</td>
</tr>
<tr>
<td>🔍 Forensics</td>
<td>SRM Filters</td>
<td>High-frequency noise residual extraction</td>
</tr>
<tr>
<td>🔥 Framework</td>
<td>PyTorch 2.x</td>
<td>Deep learning framework</td>
</tr>
<tr>
<td>👤 Face Detection</td>
<td>MediaPipe BlazeFace</td>
<td>Real-time face localization</td>
</tr>
<tr>
<td>📊 Explainability</td>
<td>Grad-CAM</td>
<td>Visual attribution heatmaps</td>
</tr>
<tr>
<td>⚡ Backend</td>
<td>FastAPI</td>
<td>Async REST API server</td>
</tr>
<tr>
<td>🎨 Frontend</td>
<td>React 18 + Vite</td>
<td>Modern web interface</td>
</tr>
<tr>
<td>📦 Dataset</td>
<td>DF40 (NeurIPS 2024)</td>
<td>40 manipulation methods, 72K+ images</td>
</tr>
</table>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Deepfake Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<p align="center">
  <sub>Built with ❤️ for a safer digital world</sub>
</p>
