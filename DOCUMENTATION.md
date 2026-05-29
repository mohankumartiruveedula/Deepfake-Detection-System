# EfficientNet-B4 Deepfake Detector — Full System Documentation

**Model:** EfficientNet-B4 + SRM filter · Pretrained ImageNet backbone  
**Dataset:** DF40 (NeurIPS 2024, 40 methods) + CelebDF · ~72,000 images  
**Backend:** FastAPI (Python) · Grad-CAM heatmaps · MediaPipe face detection  
**Frontend:** React (Vite) + standalone HTML  
**Output:** REAL / FAKE / UNCERTAIN + confidence score + Grad-CAM overlay

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Model — EfficientNet-B4 Architecture](#3-model--efficientnet-b4-architecture)
4. [Dataset & Training](#4-dataset--training)
5. [Model Files — Exported EfficientNet-B4](#5-model-files--exported-efficientnet-b4)
6. [Backend — FastAPI Server](#6-backend--fastapi-server)
7. [Frontend — React UI](#7-frontend--react-ui)
8. [End-to-End Request Flow](#8-end-to-end-request-flow)
9. [Running the Project](#9-running-the-project)
10. [API Reference](#10-api-reference)
11. [Model Performance](#11-model-performance)
12. [Deployment & Optimisation Notes](#12-deployment--optimisation-notes)

---

## 1. Project Structure

```
exported model file structure/
├── DOCUMENTATION.md              ← This file (full system docs)
│
├── Exported EfficientNet-B4/     ← Model code + weights
│   ├── model.py                  ← EfficientNet-B4 architecture definition
│   ├── inference.py              ← Face detection, preprocessing, prediction, Grad-CAM
│   ├── config.py                 ← All hyperparameters and paths
│   ├── mediapipe_face_model.tflite ← BlazeFace detection model (~230 KB)
│   ├── requirements.txt          ← Model-only dependencies
│   ├── DOCUMENTATION.md          ← Model-specific docs (training, datasets)
│   └── outputs/
│       └── checkpoints/
│           └── best_model.pth    ← Trained weights (~213 MB) ← REQUIRED
│
├── backend/
│   ├── main.py                   ← FastAPI application (API server)
│   └── requirements.txt          ← Backend dependencies
│
└── frontend/
    ├── standalone.html            ← Zero-install HTML (open directly in browser)
    ├── index.html                 ← Vite entry point
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx               ← React root
        ├── App.jsx                ← Main React component
        └── index.css              ← All styles (glassmorphism dark theme)
```

> **The only file not included in the repo is `best_model.pth`** (~213 MB).  
> Place it at: `Exported EfficientNet-B4/outputs/checkpoints/best_model.pth`

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    USER'S BROWSER                    │
│                                                      │
│  1. Selects or drops an image / video file           │
│  2. React (Vite) or standalone.html sends            │
│     a multipart POST to http://localhost:8000/detect │
│  3. Displays: verdict, confidence bar, Grad-CAM      │
└────────────────────┬─────────────────────────────────┘
                     │  HTTP POST /detect (multipart/form-data)
                     ▼
┌──────────────────────────────────────────────────────┐
│               FASTAPI BACKEND  (port 8000)           │
│  backend/main.py                                     │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │  1. Receive uploaded file                   │     │
│  │  2. Save to temp file                       │     │
│  │  3. Call inference pipeline (image/video)   │     │
│  │  4. Generate Grad-CAM heatmap (images only) │     │
│  │  5. Return JSON result + base64 heatmaps    │     │
│  └─────────────────────────────────────────────┘     │
│                       │                              │
│            calls inference.py functions              │
└──────────────────┬────┴──────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│          MODEL PIPELINE  (inference.py)              │
│  Exported EfficientNet-B4/                           │
│                                                      │
│  1. Face Detection                                   │
│     MediaPipe BlazeFace → crop largest face          │
│     Fallback: OpenCV Haar Cascade                    │
│     Fallback: 70% centre crop                        │
│                                                      │
│  2. Preprocessing                                    │
│     Resize to 380×380 → ToTensor → ImageNet norm     │
│     TTA: average original + horizontal flip          │
│                                                      │
│  3. Forward Pass                                     │
│     SRM filter (high-freq noise) + RGB → 6-ch input  │
│     EfficientNet-B4 backbone → 1792-d feature        │
│     GAP → Dropout → Linear(512) → GELU → Linear(1)  │
│     sigmoid → probability [0, 1]                     │
│                                                      │
│  4. Decision Logic                                   │
│     prob < 0.35  → REAL                              │
│     0.35 ≤ prob ≤ 0.50 → UNCERTAIN                   │
│     prob > 0.50  → FAKE                              │
│                                                      │
│  5. Grad-CAM (images only)                           │
│     Gradient of FAKE score w.r.t. last MBConv block  │
│     → JET colourmap overlay on face crop             │
└──────────────────────────────────────────────────────┘
```

---

## 3. Model — EfficientNet-B4 Architecture

### Full Forward Pass

```
User Image (any resolution)
        │
        ▼  Face Detection (MediaPipe / Haar / centre crop)
Face Crop (variable size)
        │
        ▼  Resize to 380×380, ToTensor, ImageNet normalise
Tensor: (1, 3, 380, 380)
        │
        ├──────────────────────────────────────────┐
        │  SRM Conv2d (frozen Laplacian, depth-wise)│
        │  Extracts high-freq noise residuals       │
        └──► SRM Tensor: (1, 3, 380, 380)          │
        │                                           │
        ▼  torch.cat([rgb, srm], dim=1)             │
Concat Tensor: (1, 6, 380, 380) ◄──────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│           EfficientNet-B4 Backbone (timm)            │
│  in_chans=6, global_pool='', num_classes=0           │
│                                                      │
│  Stem Conv 3×3 s=2 → 48ch                            │
│  MBConv1  stride=1  24ch  ×2                         │
│  MBConv2  stride=2  48ch  ×4                         │
│  MBConv3  stride=2  56ch  ×4                         │
│  MBConv4  stride=2 112ch  ×6                         │
│  MBConv5  stride=1 160ch  ×6                         │
│  MBConv6  stride=2 272ch  ×8                         │
│  MBConv7  stride=1 448ch  ×2  ← Grad-CAM hooks here  │
│  Head Conv 1×1 → 1792ch                              │
└──────────────────────────────────────────────────────┘
        │  Spatial feature map: (1, 1792, H', W')
        ▼
  Adaptive Average Pool → (1, 1792)
        │
        ▼  Classification Head
  Dropout(0.40)
  Linear(1792 → 512)
  GELU
  Dropout(0.30)
  Linear(512 → 1)
        │
        ▼  logit (scalar)
  sigmoid() → probability p ∈ [0, 1]
        │
        ▼  Decision
  p < 0.35     → REAL
  0.35 ≤ p ≤ 0.50 → UNCERTAIN
  p > 0.50     → FAKE
```

### Key Design Choices

| Choice | Reason |
|---|---|
| **SRM filter** | Extracts checkerboard/compression noise invisible to human eye but highly predictive of deepfakes |
| **6-channel input** (RGB + SRM) | Forces backbone to jointly learn appearance + noise-residual features |
| **380px resolution** | EfficientNet-B4 native size; pixel-level deepfake artefacts are destroyed at lower resolutions |
| **global_pool=''** | Keeps spatial feature map alive for Grad-CAM; pooled manually after backbone |
| **GELU activation** | Smooth non-linearity; better gradient flow than ReLU for small classification heads |
| **Test-Time Augmentation** | Average probability over original + H-flip → more stable predictions |

---

## 4. Dataset & Training

### Dataset: DF40 + CelebDF

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 36,212 | 36,212 | 72,424 |
| Val | 4,506 | 3,193 | 7,699 |
| Test | 4,526 | 3,193 | 7,719 |

DF40 covers **40 manipulation methods**: face swapping (DeepFaceLab, SimSwap, FSGAN), face reenactment (FOMM, LivePortrait), entire face synthesis (DiT, SiT, PixArt-α, MidJourney), and face editing (StyleGAN, InterFaceGAN).

CelebDF real frames were injected to diversify the "real" class beyond FF++ YouTube compression bias.

### Training Strategy

| Phase | Epochs | Backbone | Head LR | Backbone LR |
|---|---|---|---|---|
| Warm-up | 0–4 | Frozen | 3e-5 | — |
| Full fine-tune | 5–30 | Unfrozen | 3e-5 | 3e-6 |

**Loss:** `BCEWithLogitsLoss` with `pos_weight=1.418` (compensates real:fake imbalance)  
**Augmentation:** ColorJitter, GaussianBlur, RandomErasing, MixUp (α=0.2), RandomPerspective  
**GPU optimisations:** Mixed precision (AMP), cuDNN benchmark, non-blocking transfers

---

## 5. Model Files — Exported EfficientNet-B4

### `config.py`
Central configuration. Key constants used at inference time:

```python
IMAGE_SIZE  = 380          # Input resolution — do not change
NUM_CLASSES = 1            # Binary: real=0, fake=1
IMG_MEAN = [0.485, 0.456, 0.406]   # ImageNet normalisation
IMG_STD  = [0.229, 0.224, 0.225]
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
```

### `model.py`
Defines `EfficientNetB4Detector` (alias `DeepfakeDetector`).

```python
model = DeepfakeDetector(
    num_classes=1,
    pretrained=False,   # weights loaded from checkpoint
    drop_rate=0.4,
    use_srm=True,       # SRM high-pass filter enabled
)
# Forward returns: (logits, None, embedding)
# logits: (B,) scalar per image
# embedding: (B, 512) feature vector (unused at inference)
```

### `inference.py`
The complete inference pipeline. Key public functions:

```python
# Load model from checkpoint
model = load_model(ckpt_path, device)

# Single image (with face detection + TTA)
label, prob = predict_image(
    model, path, device,
    image_size=380,
    threshold=0.50,
    tta=True,
    uncertain_low=0.35,
    uncertain_high=0.50,
)
# label: "REAL" | "FAKE" | "UNCERTAIN"
# prob:  float in [0, 1]  (fake probability)

# Video (sample every N-th frame)
label, avg_prob, per_frame_probs = predict_video(
    model, video_path, device,
    image_size=380,
    sample_every=8,
)
```

Face detection priority:
1. **MediaPipe BlazeFace** (CPU delegate, requires `mediapipe_face_model.tflite`)
2. **OpenCV Haar Cascade** (built-in fallback)
3. **70% centre crop** (no face found)

### `mediapipe_face_model.tflite`
Pre-downloaded BlazeFace model (~230 KB). Must be in the same directory as `inference.py`. Without it the system falls back to Haar Cascade automatically.

---

## 6. Backend — FastAPI Server

### `backend/main.py`

**Startup:** Loads model from `Exported EfficientNet-B4/outputs/checkpoints/best_model.pth` once at server start. Reads the `best_threshold` saved in the checkpoint; falls back to 0.38 if not present.

**CORS:** Fully open (`allow_origins=["*"]`) for local development.

### Endpoint: `POST /detect`

Accepts a `multipart/form-data` upload with a single field `file`.

**Supported formats:**
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

**Image processing flow:**
```
1. Save to temp file
2. predict_image()  →  (_, prob)
3. Apply threshold: label = "FAKE" if prob >= THRESHOLD else "REAL"
4. generate_heatmap_data()  →  face_crop + heatmap_overlay + heatmap_raw (base64 PNG)
5. Return JSON
6. Delete temp file
```

**Video processing flow:**
```
1. Save to temp file
2. predict_video()  →  (_, avg_prob, frame_probs)
3. Apply threshold: label = "FAKE" if avg_prob >= THRESHOLD else "REAL"
4. Return JSON (no heatmap for video)
5. Delete temp file
```

**JSON Response:**
```json
{
  "filename": "face.jpg",
  "label": "FAKE",
  "fake_prob": 0.8732,
  "confidence": 0.8732,
  "face_crop_base64": "<base64 PNG string>",
  "heatmap_overlay_base64": "<base64 PNG string>",
  "heatmap_raw_base64": "<base64 PNG string>"
}
```

For `REAL` images, `confidence = 1 - prob`.  
Heatmap fields are `null` for videos or if `grad-cam` is not installed.

---

## 7. Frontend — React UI

### Option A: Standalone HTML (`frontend/standalone.html`)
No build step required. Open directly in a browser:
```
File → Open → standalone.html
```
Uses React 18 + Babel transpiler from CDN. Works instantly without Node.js.

### Option B: React + Vite (`frontend/src/`)

**`src/App.jsx`** — Main component state machine:

```
States:
  idle     → upload zone displayed
  loading  → spinner shown
  error    → error box (fake-red style)
  result   → verdict box + previews + Grad-CAM
```

**Result display:**
- **Side-by-side row:** Original uploaded image | Grad-CAM heatmap overlay (if available)
- **Verdict box:** CSS class `real` / `fake` / `uncertain` drives colours and glow effects
  - REAL → green glow
  - FAKE → red glow + glitch animation on label
  - UNCERTAIN → amber/orange glow + human-review warning
- **Confidence bar:** Animated fill showing `confidence * 100%`
- **Stats row:** Confidence % + raw fake probability

**`src/index.css`** — Glassmorphism dark theme:
- Rotating conic-gradient border on glass panel
- Floating animated orb background
- Grid overlay
- Shimmer text animation on title
- Glitch animation on FAKE verdict
- Amber styles for UNCERTAIN verdict

---

## 8. End-to-End Request Flow

### Image upload: step by step

```
BROWSER
  │
  │  1. User drops "face.jpg" onto upload zone
  │     processFile() called
  │     URL.createObjectURL() → local preview shown
  │
  │  2. FormData built: { file: face.jpg }
  │     fetch("http://localhost:8000/detect", { method: "POST", body: formData })
  │
  ▼
BACKEND  main.py  /detect
  │
  │  3. File saved to temp path (NamedTemporaryFile)
  │
  │  4. ext = ".jpg" → IMAGE branch
  │     inference.predict_image(model, tmp_path, device, image_size=380)
  │       │
  │       │  a. cv2.imread(tmp_path) → BGR array
  │       │  b. crop_face():
  │       │       Try MediaPipe BlazeFace → detect bounding box → crop + margin
  │       │       Fallback: Haar Cascade → crop
  │       │       Fallback: 70% centre crop
  │       │  c. PIL.Image.fromarray(BGR→RGB)
  │       │  d. TTA:
  │       │       tensor1 = preprocess(img_pil, 380)     # original
  │       │       tensor2 = preprocess(img_pil.flip(), 380)  # H-flipped
  │       │       p1 = sigmoid(model(tensor1)[0])
  │       │       p2 = sigmoid(model(tensor2)[0])
  │       │       prob = (p1 + p2) / 2
  │       │  e. Returns (label_ignored, prob)
  │       │
  │  5. label = "FAKE" if prob >= THRESHOLD (0.38 default) else "REAL"
  │     confidence = prob if FAKE else (1 - prob)
  │
  │  6. generate_heatmap_data(tmp_path):
  │       crop_face() → face_bgr
  │       GradCAM(model, target_layers=[model.get_target_layer()])
  │         target_layer = backbone.blocks[-1]  (last MBConv block)
  │         gray_cam = cam(tensor, targets=[ClassifierOutputTarget(0)])
  │       cv2.resize(gray_cam, (face_w, face_h))
  │       cv2.applyColorMap(heatmap, COLORMAP_JET)
  │       overlay = addWeighted(face_bgr, 0.55, heatmap_coloured, 0.45, 0)
  │       Encode all 3 images as base64 PNG
  │
  │  7. Temp file deleted
  │
  │  8. JSON response returned:
  │     { filename, label, fake_prob, confidence,
  │       face_crop_base64, heatmap_overlay_base64, heatmap_raw_base64 }
  │
  ▼
BROWSER
  │
  │  9. setResult(data) → results-view rendered
  │     - preview-row: original | heatmap_overlay (if not null)
  │     - verdict-box with CSS class = label.toLowerCase()
  │     - confidence bar animated to confidence%
  │     - stats: confidence% + raw fake_prob
  │     - UNCERTAIN note shown if label === "UNCERTAIN"
```

### Video upload differences

- Backend calls `predict_video()` → samples 1 frame every 8 frames
- Each frame: face detect → preprocess → model forward → prob
- `avg_prob = mean(all frame probs)`
- Same threshold applied to `avg_prob`
- **No Grad-CAM for video** (heatmap fields are null)
- Frontend shows only the video player (no heatmap column)

---

## 9. Running the Project

### Prerequisites

```
Python 3.10+   (backend)
Node.js 18+    (frontend Vite, optional)
best_model.pth placed at:
  Exported EfficientNet-B4/outputs/checkpoints/best_model.pth
```

### Step 1 — Install backend dependencies

```powershell
cd "c:\Users\karth\Desktop\web\project\exported model file structure\backend"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

> For GPU (CUDA 12.x): replace `torch` install with:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 2 — Start the backend

```powershell
# From the backend/ directory with venv activated:
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
✅ Loaded: ...best_model.pth
   Val Acc : 0.8864
Model loaded successfully.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify at: `http://localhost:8000/` → `{"status": "Backend is running..."}`

### Step 3A — Frontend (standalone, no build needed)

Simply open in browser:
```
frontend/standalone.html
```
Right-click → Open with → Chrome / Edge

### Step 3B — Frontend (React + Vite)

```powershell
cd "c:\Users\karth\Desktop\web\project\exported model file structure\frontend"
npm install
npm run dev
```

Open: `http://localhost:5173`

---

## 10. API Reference

### `GET /`
Health check.

**Response:**
```json
{"status": "Backend is running. API is ready to accept files."}
```

### `POST /detect`

**Content-Type:** `multipart/form-data`  
**Field:** `file` — the image or video file

**Success Response (200):**
```json
{
  "filename": "photo.jpg",
  "label": "FAKE",
  "fake_prob": 0.8732,
  "confidence": 0.8732,
  "face_crop_base64": "iVBORw0KGgo...",
  "heatmap_overlay_base64": "iVBORw0KGgo...",
  "heatmap_raw_base64": "iVBORw0KGgo..."
}
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `filename` | string | Original upload filename |
| `label` | string | `"REAL"`, `"FAKE"`, or `"UNCERTAIN"` |
| `fake_prob` | float | Raw model output after sigmoid [0.0–1.0] |
| `confidence` | float | How confident the model is in its verdict |
| `face_crop_base64` | string\|null | Base64 PNG of the cropped face region |
| `heatmap_overlay_base64` | string\|null | Base64 PNG of face + Grad-CAM overlay |
| `heatmap_raw_base64` | string\|null | Base64 PNG of raw JET colourmap heatmap |

**Confidence calculation:**
```
if label == "FAKE":   confidence = fake_prob
if label == "REAL":   confidence = 1 - fake_prob
if label == "UNCERTAIN": confidence = fake_prob
```

**Error Responses:**

| Code | Cause |
|---|---|
| 400 | Unsupported file extension |
| 500 | Model not loaded / inference error |

---

## 11. Model Performance

### Phase 3 Results (April 2026)

Evaluated on 7,719 held-out test images (DF40 + CelebDF):

| Metric | Value |
|---|---|
| **Accuracy** (threshold 0.38) | **88.64%** |
| **AUC-ROC** | **0.9567** |
| **Sensitivity** (Fake recall) | **89.79%** |
| **Specificity** (Real recall) | **87.83%** |
| **F1 Score (Fake)** | **0.8673** |
| **Optimal threshold** (Youden's J) | 0.4141 |

### Confidence Zone Breakdown

| Zone | Prediction | Real error rate | Fake error rate |
|---|---|---|---|
| prob < 0.35 | REAL | **0.3%** | — |
| 0.35 ≤ prob ≤ 0.50 | UNCERTAIN | — (flagged) | — (flagged) |
| prob > 0.50 | FAKE | — | **1.5%** |

When the model is **confident** (outside the uncertain zone), it is extremely reliable.

---

## 12. Deployment & Optimisation Notes

### Threshold tuning

The backend uses `THRESHOLD = 0.38` by default (slightly below 0.5 to bias toward catching fakes). If the saved checkpoint contains `metrics.best_threshold`, that value is loaded automatically at startup.

To change the threshold at runtime, edit `THRESHOLD` in `backend/main.py` line 55.

### Production hardening

| Concern | Fix |
|---|---|
| CORS | Replace `allow_origins=["*"]` with your domain |
| File size limit | Add `File(..., max_size=50*1024*1024)` or Nginx `client_max_body_size` |
| HTTPS | Serve behind Nginx with TLS |
| Model reload | Use `@app.on_event("startup")` already implemented — model loaded once |

### Performance

| Optimisation | Impact |
|---|---|
| GPU inference | ~10x faster than CPU for images; essential for video |
| ONNX export | 2–3x CPU speedup; export with `torch.onnx.export` |
| INT8 quantisation | ~4x model size reduction (213 MB → 53 MB), <1% accuracy loss |
| `sample_every` | Reduce video processing time by sampling fewer frames |

### Troubleshooting

| Problem | Fix |
|---|---|
| `Model checkpoint not found` | Place `best_model.pth` at `Exported EfficientNet-B4/outputs/checkpoints/` |
| `timm not found` | `pip install timm>=0.9.12` |
| `mediapipe init failed` | Normal — falls back to Haar Cascade automatically |
| `grad-cam not installed` | `pip install grad-cam` — heatmaps disabled but detection still works |
| CORS error in browser | Backend must be running on port 8000; check firewall |
| Unicode emoji errors on Windows | Already handled in `main.py` (stdout reconfigured to UTF-8) |
| Video takes very long | Reduce `sample_every` parameter (default 8) or limit video length |

---

*EfficientNet-B4 Deepfake Detector · Antigravity AI · DF40 (NeurIPS 2024) + CelebDF · April 2026*
