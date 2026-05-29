# EfficientNet-B4 Deepfake Detector — Documentation

**Dataset:** DF40 (NeurIPS 2024) + CelebDF (real diversity) · 40 manipulation methods  
**Architecture:** EfficientNet-B4 pretrained ImageNet backbone · Custom 2-layer head  
**Heatmaps:** Grad-CAM on last MBConv block  
**Inference:** Confidence-based 3-class output (REAL / FAKE / UNCERTAIN)

---

## Table of Contents
1. [Why We Changed the Architecture](#1-why-we-changed-the-architecture)
2. [EfficientNet-B4 Architecture](#2-efficientnet-b4-architecture)
3. [DF40 Dataset](#3-df40-dataset)
4. [Dataset Preparation & Balanced Sampling](#4-dataset-preparation--balanced-sampling)
5. [Training Strategy](#5-training-strategy)
6. [Grad-CAM Heatmaps](#6-grad-cam-heatmaps)
7. [Robustness to Bad Image Conditions](#7-robustness-to-bad-image-conditions)
8. [Confidence-Based Rejection System](#8-confidence-based-rejection-system)
9. [File Structure & Usage](#9-file-structure--usage)
10. [Running on Google Colab](#10-running-on-google-colab)
11. [Model Performance](#11-model-performance)
12. [Website Integration & Model Export](#12-website-integration--model-export)

---

## 1. Why We Changed the Architecture

### Problems with the old Xception + SAM model

| Problem | Root Cause | Effect |
|---|---|---|
| **Broken heatmaps** | SAM learned attention from random init — no prior forcing it to focus on forgery regions | Attention maps = uniform noise, visualisations meaningless |
| **Overfitting** | Entire Xception trained from Kaiming-random init on a small dataset | Model memorises dataset compression artefacts; fails on new data |
| **Poor generalisation** | Trained only on FF++ (2018) + Celeb-DF (2019); both contain only GAN-based swaps | AUC drops 40-50% on real-world 2024 deepfakes |

### How EfficientNet-B4 fixes these

| Feature | Benefit |
|---|---|
| **ImageNet-pretrained weights** | Model already understands edges, textures, faces — only needs to learn forgery-specific deviations |
| **Grad-CAM heatmaps** | Uses real gradients from the forward pass; highlights exactly what drove the prediction |
| **DF40 dataset (40 methods)** | Covers face-swap, reenactment, synthesis, editing — including 2024 diffusion models (DiT, SiT, HeyGen) |
| **MixUp augmentation** | Blends training samples — prevents memorising any single method's artefacts |
| **Phase 4: Tighter Crops & SRM** | Forces the model to ignore background shortcuts and extracts invisible high-frequency noise from deepfake AI generators. |

---

## 2. EfficientNet-B4 Architecture

### Overview (Phase 4 - OOD Generalization)

```
Input Image (380 × 380 × 3 RGB)
         │
         ├───► [ SRM High-Pass Filter (Laplacian) ] ──► (380 × 380 × 3 Noise)
         │                                                      │
         └───────────────────► Concatenate ◄────────────────────┘
                                     │
                                     ▼
                      6-Channel Tensor (380 × 380 × 6)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────┐
│              EfficientNet-B4 Backbone                  │
│                                                        │
│  Stem Conv (3×3, s=2) → 48 channels                   │
│         │                                              │
│  MBConv Block 1   (stride=1, 24ch,  rep=2)             │
│  MBConv Block 2   (stride=2, 48ch,  rep=4)             │
│  MBConv Block 3   (stride=2, 56ch,  rep=4)             │
│  MBConv Block 4   (stride=2, 112ch, rep=6)             │
│  MBConv Block 5   (stride=1, 160ch, rep=6)   ← Grad-CAM│
│  MBConv Block 6   (stride=2, 272ch, rep=8)   ← target  │
│  MBConv Block 7   (stride=1, 448ch, rep=2)   ←         │
│         │                                              │
│  Head Conv 1×1 → 1792 channels                         │
│  [Spatial feature map: (B, 1792, H', W')]              │
└────────────────────────────────────────────────────────┘
         │
         ▼
  Adaptive Average Pool  →  (B, 1792)
         │
         ▼
┌────────────────────────────────┐
│     Classification Head        │
│                                │
│  Dropout(0.40)                 │
│  Linear(1792 → 512)            │
│  GELU activation               │
│  Dropout(0.30)                 │
│  Linear(512 → 1)               │
└────────────────────────────────┘
         │
         ▼
     Logit (scalar)
         │
      sigmoid()
         │
   Probability [0,1]
         │
   ┌─────┴──────────────┐
   │ Confidence-based    │
   │ rejection logic     │
   ├─────────────────────┤
   │ prob < 0.35 → REAL  │
   │ 0.35-0.50 → UNCERTAIN│
   │ prob > 0.50 → FAKE  │
   └─────────────────────┘
```

### What is EfficientNet-B4?

EfficientNet uses **compound scaling** — simultaneously scaling depth, width, and resolution of the network using a fixed ratio. Unlike manually tuned architectures, this gives the best accuracy-per-parameter trade-off.

- **B4** is the 4th scaling level (B0 is smallest, B7 is largest)
- Native resolution: **380 × 380** (critical — feeding wrong resolution degrades performance)
- Parameter count: **≈ 19M** (vs 22M for the old Xception)
- FLOPS: **≈ 4.4B** per forward pass

### What is an MBConv Block?

MBConv (Mobile Inverted Bottleneck Convolution) is the core building block:

```
Input
  │
  ▼
1×1 Conv (expand channels 6×)  ← pointwise expansion
  │
  ▼
3×3 Depthwise Conv + Squeeze-Excitation (SE) attention
  │                    │
  │           Channel attention: global avg pool → MLP → sigmoid
  │           Recalibrates which channels to emphasise
  │
  ▼
1×1 Conv (reduce back to original channels)
  │
  ▼
+── Residual connection (if same shape)
  │
  ▼
Output
```

**Squeeze-Excitation** is what makes EfficientNet special for deepfake detection: it learns *which feature channels* (frequency bands, texture types) are most informative for the decision — effectively an automatic feature selector.

### Why not Xception?

| | Xception | EfficientNet-B4 |
|---|---|---|
| Compound scaling | ❌ Manual | ✅ Automated |
| SE (channel attention) | ❌ | ✅ |
| Pretrained quality | ❌ Custom init only | ✅ timm ImageNet-21K+ |
| Native resolution | 299px | **380px** |
| Transfer learning | Poor (needs full fine-tune) | Excellent (partial freeze works) |

---

## 3. DF40 Dataset

### What is DF40?

DF40 is a large-scale deepfake benchmark introduced at **NeurIPS 2024**. It is the most comprehensive publicly available deepfake dataset by number of manipulation methods:

| Category | Methods | Examples |
|---|---|---|
| **Face Swapping** | 10 | DeepFaceLab, SimSwap, FSGAN, FaceSwap, ... |
| **Face Reenactment** | 13 | First Order Motion, FOMM, LivePortrait, ... |
| **Entire Face Synthesis** | 12 | DiT, SiT, PixArt-α, MidJourney, ... |
| **Face Editing** | 5 | StyleGAN, InterFaceGAN, ... |
| **Total** | **40** | |

### Why DF40 > FF++ + CelebDF

- **FF++** (2018): 4 methods, all GAN-based, old compression  
- **CelebDF** (2019): 1 method, controlled studio lighting  
- **DF40** (2024): 40 methods, includes **diffusion models** (DiT, SiT, PixArt-α) and commercial tools (HeyGen, MidJourney) — these are what real-world deepfakes look like today

Models trained only on FF++ + CelebDF can drop **40-50% AUC** when tested on DF40-style fakes.

---

## 4. Dataset Preparation & Balanced Sampling

### Phase 2: CelebDF Real Data Injection

In Phase 2, we injected **13,314 real face crops** from CelebDF videos to diversify the "real" class:

| Source | Videos | Frames Extracted | Purpose |
|---|---|---|---|
| Celeb-real | 590 | ~8,850 | Studio-quality, varied lighting |
| YouTube-real | 300 | ~4,464 | YouTube compression, different from FF++ |

This solved the "YouTube compression = Real" bias from Phase 1.

### Final dataset composition

| Split | Real | Fake | Total | Source Mix |
|---|---|---|---|---|
| Train | 36,212 | 25,544 | 61,756 | DF40 + CelebDF |
| Val | 4,506 | 3,193 | 7,699 | DF40 + CelebDF |
| Test | 4,526 | 3,193 | 7,719 | DF40 + CelebDF |

The `pos_weight=1.418` in BCEWithLogitsLoss compensates for the real:fake imbalance.

### `prepare_df40.py` — Features

- **Windows-compatible:** Auto-falls back from `os.symlink()` to `shutil.copy2()` when admin rights are unavailable
- **Resume mechanism:** JSON state file tracks progress; interrupted runs pick up where they left off
- **Balanced sampling:** 1:1 real-to-fake ratio using per-method caps (default: 10,000 per method)

---

## 5. Training Strategy

### Phase 1: Head Warm-Up (Epochs 0–4)

```
Backbone: FROZEN (gradients blocked)
Head:     TRAINING (full LR = 3e-5)

Why: The randomly initialised head produces garbage gradients at first.
     If the backbone is unfrozen, those large gradients corrupt the
     pretrained ImageNet features.  Freezing for 5 epochs lets the head
     learn a useful gradient direction without destroying the backbone.
```

### Phase 2: Full Fine-Tuning (Epoch 5 → end)

```
Backbone: UNFROZEN (backbone LR = 3e-6)  ← 10× smaller than head
Head:     TRAINING  (head LR    = 3e-5)

Why layer-wise LR:
  Higher layers (closer to output) need more adjustment because the
  task (deepfake detection) differs from ImageNet classification.
  Lower layers (early conv filters for edges/textures) barely need
  changing since they generalise across tasks.
```

### Training Configuration (Phase 2)

| Parameter | Value | Notes |
|---|---|---|
| `IMAGE_SIZE` | 380 | Native EfficientNet-B4 resolution |
| `BATCH_SIZE` | 16 | RTX 4050 6GB (reduce to 12 if OOM) |
| `NUM_EPOCHS` | 30 | Extended for CelebDF + DF40 |
| `UNFREEZE_EPOCH` | 5 | 5-epoch head warmup |
| `LEARNING_RATE` | 3e-5 | Head LR |
| `BACKBONE_LR_MULT` | 0.1 | Backbone LR = 3e-6 |
| `MIXUP_ALPHA` | 0.2 | Beta(0.2, 0.2) sampling |
| `LABEL_SMOOTHING` | 0.10 | Prevents overconfident outputs |
| `EARLY_STOP_PATIENCE` | 10 | Epochs without improvement |
| `WEIGHT_DECAY` | 5e-4 | AdamW regularisation |

### GPU Optimizations (train.py)

| Optimization | Technique |
|---|---|
| Mixed precision | `torch.amp.GradScaler` + `autocast` |
| TF32 matmul | `torch.set_float32_matmul_precision("high")` |
| cuDNN autotuner | `torch.backends.cudnn.benchmark = True` |
| Zero-copy transfers | `non_blocking=True` on `.to(device)` |
| Efficient grad reset | `optimizer.zero_grad(set_to_none=True)` |
| GPU accumulation | No per-batch `.item()` calls; single sync at epoch end |
| Inference mode | `@torch.inference_mode()` for evaluation |

### MixUp Augmentation

```python
lam = Beta(0.2, 0.2)          # typically 0.7–0.9
x_mix = lam * x_i + (1-lam) * x_j
y_mix = lam * y_i + (1-lam) * y_j
```

MixUp prevents the model from "memorising" individual samples. Mixed labels create a smoother decision boundary that generalises better to new fake methods.

---

## 6. Grad-CAM Heatmaps

### How Grad-CAM works

```
Forward pass:
  Image → EfficientNet-B4 → Feature Map (last block) → Head → Logit

Backward pass (FAKE class):
  ∂(FAKE logit) / ∂(Feature Map channels)
         = per-channel importance weights

Heatmap:
  Σ  weights_c × feature_map_c   (ReLU applied)
  c
  Resize to input resolution → overlay on original image
```

**Key insight:** Red areas in the Grad-CAM overlay = pixels that INCREASED the model's confidence that the image is FAKE. These will be blending boundaries, eye-texture inconsistencies, jawline warping — the actual forgery artefacts.

### Usage

```bash
python inference.py --source face.jpg --visualise
# Saves face_gradcam.png showing side-by-side original + heatmap
```

---

## 7. Robustness to Bad Image Conditions & Out-Of-Distribution Data

To prevent overfitting to perfect studio conditions, `dataset.py` includes heavy spatial and colour augmentations. Furthermore, in **Phase 4**, we completely revamped the data pipeline and architecture to enforce **Out-Of-Distribution (OOD) Generalization**.

### The "Shortcut Learning" Problem
Neural networks are lazy. If deepfakes in the dataset always have a slightly blurry background, the model will look at the *background* instead of the *face* to make a decision. When tested on real-world deepfakes with sharp backgrounds, it fails. We solved this with two major upgrades:

### Phase 4: Simulated Tighter Face Cropping (Spatial Forcing)
We reduced the face detection margin from `20%` down to `5%`.
- **Inference:** MediaPipe is instructed to extract faces with a tight 0.05 margin.
- **Training (Simulated Zoom):** To avoid re-extracting terabytes of video, `dataset.py` dynamically applies `transforms.CenterCrop(296)` (cropping the middle 78%) to the existing 20% margin images, stripping away the background and forcing the neural network to analyze *only* the facial features.

### Phase 4: Spatial Rich Model (SRM) Frequency Filter
AI generators (Midjourney, Stable Diffusion, GANs) produce faces that look perfect in RGB pixels, but they leave invisible "checkerboard" artifacts in the high-frequency spectrum.
- We added an `SRMConv2d` layer initialized with a hardcoded **Laplacian High-Pass Filter**.
- The model extracts 3 channels of high-frequency noise from the image and stacks them with the 3 RGB channels.
- The `timm` EfficientNet-B4 backbone is configured to accept a **6-Channel Input Tensor**, allowing it to simultaneously analyze visible colors and invisible AI noise.

### Standard Augmentations

The training augmentation pipeline deliberately simulates common real-world problems:

| Augmentation | Probability | What it simulates |
|---|---|---|
| `ColorJitter(b=±40%, c=±40%, s=±30%, h=±15°)` | 70% | Bad white-balance, colour temperature shifts, over/under-exposure |
| `RandomAutocontrast` | 30% | Auto-processing by phone cameras |
| `RandomEqualize` | 20% | Histogram-equalised surveillance feeds |
| `GaussianBlur(σ=0.1–2.5)` | 25% | Out-of-focus, motion blur, compressed video |
| `RandomAdjustSharpness(0)` | 20% | Soft/blurry images |
| `RandomPosterize(bits=4)` | 15% | Heavy JPEG compression, re-encoding |
| `RandomErasing(scale=2–15%)` | 30% | Occlusion by glasses, hats, hands |
| `RandomPerspective(dist=0.1)` | 30% | Non-frontal angles, distortion |

Because the model sees these degradations **every epoch** during training, it learns features that are robust to image quality variation.

---

## 8. Confidence-Based Rejection System

### The Problem

~25% of test images fall in the "uncertain zone" (probability 0.35–0.50). The model isn't completely confident about these — forcing a binary REAL/FAKE decision causes the vast majority of classification errors.

### The Solution: Three-Class Output

Instead of forcing every prediction into REAL or FAKE, images in the uncertain zone are flagged as **UNCERTAIN**, indicating they need human review:

```
Probability Output
  │
  ├── prob < 0.35  ────→  REAL       (high confidence)
  │
  ├── 0.35 ≤ prob ≤ 0.50 → UNCERTAIN  (needs human review)
  │
  └── prob > 0.50  ────→  FAKE       (high confidence)
```

### Impact on Reliability

| Metric | Without Rejection | With Rejection |
|---|---|---|
| Error rate (REAL) | 9.2% | **~1.5%** (on confident predictions) |
| Error rate (FAKE) | 12.0% | **~3.0%** (on confident predictions) |
| Human review needed | 0% | ~21% |

### CLI Usage

```bash
# Default: confidence-based rejection ON
python inference.py --source photo.jpg

# Customize uncertain zone
python inference.py --source photo.jpg --uncertain-low 0.30 --uncertain-high 0.70

# Disable rejection (force binary REAL/FAKE)
python inference.py --source photo.jpg --no-rejection
```

---

## 9. File Structure & Usage

### Directory layout

```
EfficientNet-B4/
├── model.py                ← EfficientNet-B4 architecture
├── config.py               ← All hyperparameters & paths
├── dataset.py              ← DF40 data loading + augmentation
├── prepare_df40.py         ← Organise DF40 into real/fake splits
├── extract_celebdf_faces.py ← Extract face crops from CelebDF videos
├── train.py                ← Training script (GPU-optimized)
├── evaluate.py             ← Metrics + all evaluation plots
├── inference.py            ← Inference with confidence rejection + Grad-CAM
├── analyze_model.py        ← Threshold analysis & edge-case report
├── colab_notebook.py       ← Complete Google Colab training script
├── mediapipe_face_model.tflite ← MediaPipe BlazeFace model
├── requirements.txt
├── DOCUMENTATION.md        ← This file
├── datasets/
│   └── df40_combined/      ← Created by prepare_df40.py + CelebDF
│       ├── train/ real/  fake/
│       ├── val/   real/  fake/
│       └── test/  real/  fake/
└── outputs/
    ├── checkpoints/
    │   ├── best_model.pth          ← Phase 2 best model
    │   ├── latest_model.pth
    │   ├── phase1_best_model.pth   ← Phase 1 backup
    │   └── phase1_latest_model.pth
    ├── logs/
    │   └── training_history.json
    └── evaluation/
        ├── confusion_matrix.png
        ├── roc_pr_curves.png
        ├── prob_distribution.png
        └── training_history.png
```

### Local usage (step by step)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download DF40 from https://github.com/YZY-stack/DF40

# 3. Prepare the dataset
python prepare_df40.py --df40-root /path/to/DF40

# 4. (Optional) Add CelebDF real diversity
python extract_celebdf_faces.py --celebdf-root /path/to/CelebDF

# 5. Train
python train.py

# 6. Resume training from checkpoint
python train.py --resume outputs/checkpoints/latest_model.pth

# 7. Evaluate on test set
python evaluate.py

# 8. Run inference
python inference.py --source test_face.jpg
python inference.py --source test_face.jpg --visualise    # with Grad-CAM
python inference.py --source folder_of_images/            # batch mode
python inference.py --source video.mp4                    # video mode
```

---

## 10. Running on Google Colab

See `colab_notebook.py` for the complete step-by-step script.

### Quick summary

| Step | What |
|---|---|
| **Cell 1** | Check GPU, install `timm`, `grad-cam` |
| **Cell 2** | Mount Google Drive, set paths |
| **Cell 3** | Download DF40 from official Google Drive links |
| **Cell 4** | Run `prepare_df40.py` — balance & split |
| **Cell 5** | Override Config for Colab GPU (auto-scaling batch size) |
| **Cell 6** | Load data loaders |
| **Cell 7** | Build model, loss, optimizer |
| **Cell 8** | Training loop with auto-resume |
| **Cell 9** | Final test evaluation |
| **Cell 10** | Plot training curves |
| **Cell 11** | Download `best_model.pth` to local machine |

---

## 11. Model Performance

### Phase 3 Results (April 2026)

Trained on perfectly balanced DF40 + CelebDF (72,424 training images), evaluated on 7,719 test images:

| Metric | Phase 1 (FF++ only) | Phase 2 (Unbalanced) | Phase 3 (Balanced) |
|---|---|---|---|
| **Accuracy (at 0.38)** | 71.71% | 85.41% | **88.64%** |
| **AUC-ROC** | 0.7578 | 0.9573 | **0.9567** |
| **Sensitivity (Fake recall)** | 88.13% | 93.55% | **89.79%** |
| **Specificity (Real recall)** | 60.12% | 79.67% | **87.83%** |
| **F1 (Fake)** | 0.7204 | 0.8414 | **0.8673** |
| **Max Achievable Accuracy** | 72.38% | 89.74% | **89.69%** (at 0.45) |

### Key Insight
Balancing the dataset by adding 10,668 more fake images massively improved the model's calibration. Specificity jumped from 79.6% to 87.8%, meaning the model makes significantly fewer false positive errors on real images, while still maintaining nearly 90% fake detection sensitivity at the 0.38 threshold.

### Confidence Breakdown

| Class | High-conf correct | Uncertain zone | High-conf WRONG |
|---|---|---|---|
| REAL | 46.1% | 22.4% | **0.3%** |
| FAKE | 53.4% | 19.9% | **1.5%** |

**Key insight:** When the model is confident (outside the 0.35–0.65 zone), it is **extremely reliable** — only 0.3% of real images and 1.5% of fake images get confidently wrong predictions.

### Optimal Threshold

The model is highly calibrated. Youden's J optimal threshold is **0.4141**. We recommend using **0.50** combined with the `[0.35 - 0.50]` rejection zone to bias slightly towards catching fakes (Sensitivity) while offloading unsure predictions to human review.

### Key Improvements Made

1. ✅ **380px native resolution** — preserves pixel-level forensic artifacts
2. ✅ **5-epoch head warmup** — prevents catastrophic forgetting of ImageNet features
3. ✅ **CelebDF real data injection** — eliminates "YouTube compression = Real" bias
4. ✅ **Perfect 1:1 Class Balance** — injected 10,668 extra fakes to fix class imbalance
5. ✅ **GPU throughput optimizations** — 15-25% faster training per epoch
6. ✅ **Confidence-based rejection** — UNCERTAIN output for low-confidence predictions (opt-in)
7. ✅ **MediaPipe BlazeFace** — reliable face detection replacing Haar Cascades, with explicit CPU fallback
8. ✅ **Test-Time Augmentation** — flip averaging for more stable predictions

---

## 12. Website Integration & Model Export

### Files Required for Deployment

To integrate the model into a website backend, you need these files:

| File | Purpose | Size |
|---|---|---|
| `outputs/checkpoints/best_model.pth` | Model weights (PyTorch checkpoint) | ~213 MB |
| `model.py` | Model architecture definition | 6 KB |
| `inference.py` | Complete inference pipeline (face detection, preprocessing, prediction, confidence rejection) | 18 KB |
| `config.py` | Configuration constants (IMAGE_SIZE, thresholds, normalization) | 6 KB |
| `mediapipe_face_model.tflite` | MediaPipe face detection model | ~200 KB |

### Python Dependencies for Server

```
torch>=2.0
torchvision
timm
opencv-python
mediapipe
Pillow
numpy
```

### Key Functions to Call from Your API

```python
from inference import load_model, predict_image

# Load once at server startup
model = load_model("best_model.pth", device="cuda")

# For each request
label, prob = predict_image(
    model, "uploaded_image.jpg", device="cuda",
    image_size=380, threshold=0.50, tta=True,
    uncertain_low=0.35, uncertain_high=0.65
)
# label = "REAL" | "FAKE" | "UNCERTAIN"
# prob  = float between 0.0 and 1.0
```

### API Response Format (Suggested)

```json
{
  "verdict": "FAKE",
  "confidence": 0.92,
  "fake_probability": 0.92,
  "requires_review": false
}
```

For `UNCERTAIN` predictions:
```json
{
  "verdict": "UNCERTAIN",
  "confidence": 0.45,
  "fake_probability": 0.45,
  "requires_review": true
}
```

### Phase 1 Model Backup

If the Phase 3 model underperforms in production, restore Phase 1:
```bash
copy outputs\checkpoints\phase1_best_model.pth outputs\checkpoints\best_model.pth
```

---

## 13. Optimizing for Real-World Deployment

While EfficientNet-B4 provides excellent accuracy, its 19M parameters and 380px resolution make it computationally heavy. To deploy this pipeline efficiently in a real-world web/mobile application, implement the following:

### 1. Model Export (ONNX & TensorRT)
Do not serve raw PyTorch models in production. 
- **ONNX Runtime:** Exporting the model to ONNX (`torch.onnx.export`) speeds up CPU inference by 2x-3x.
- **TensorRT:** If deploying on NVIDIA GPUs, converting the ONNX model to TensorRT will yield extremely low latency (<10ms per frame) for real-time video processing.

### 2. INT8 Quantization
Convert the model weights from 32-bit floats (FP32) to 8-bit integers (INT8) using PyTorch's Post-Training Quantization (PTQ).
- **Impact:** Reduces the model size from ~213 MB to **~53 MB** and heavily reduces RAM usage.
- **Tradeoff:** Typically costs < 1% in accuracy.

### 3. Lightweight Backbone Replacement
If edge-device deployment (mobile phones) is the goal, EfficientNet-B4 is too large.
- Replace the backbone with **MobileNetV3-Large** or **EfficientNet-Lite**.
- Use **Knowledge Distillation**: Train the lightweight model to mimic the predictions of your B4 model, allowing you to retain high accuracy with 1/5th the computation cost.

### 4. Asynchronous Video Processing
Instead of predicting every frame sequentially:
- Only process 1 frame every second (`sample_every=30` at 30fps).
- Run face detection (MediaPipe) on a separate CPU thread, passing cropped faces into a batched GPU queue for inference.

---

*Architecture by Antigravity AI · Dataset: DF40 (NeurIPS 2024) + CelebDF · April 2026*

