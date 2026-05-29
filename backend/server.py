"""
FastAPI Backend — Deepfake Detection Server
============================================
Exposes a POST /detect endpoint that accepts an image or video upload,
runs the EfficientNet-B4 deepfake detector, and returns a JSON verdict
with an optional Grad-CAM heatmap overlay (base64-encoded PNG).

Usage
-----
    cd backend/
    pip install -r requirements.txt
    python server.py          # starts on http://localhost:8000

    # Or with uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Environment Variables
---------------------
    CHECKPOINT_PATH   Override the default model checkpoint location.
                      Default: ../inference engine/outputs/checkpoints/best_model.pth
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Add the inference engine directory to sys.path so we can import from it.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_INFERENCE_DIR = (_BACKEND_DIR / ".." / "inference engine").resolve()
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Imports from the inference engine
from inference import (
    Config,
    crop_face,
    load_model,
    predict,
    predict_image,
    predict_video,
    preprocess,
)

# ---------------------------------------------------------------------------
# Grad-CAM availability check
# ---------------------------------------------------------------------------
HAS_GRADCAM = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    HAS_GRADCAM = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_CHECKPOINT = str(
    (_INFERENCE_DIR / "outputs" / "checkpoints" / "best_model.pth").resolve()
)
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Detection API",
    description="EfficientNet-B4 deepfake detector with Grad-CAM heatmaps.",
    version="1.0.0",
)

# CORS — allow the Vite dev server and common local dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded global model (initialised on first request)
# ---------------------------------------------------------------------------
_model = None
_device: str | None = None


def _get_model():
    """Load the model lazily on first request and cache it."""
    global _model, _device

    if _model is not None:
        return _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(CHECKPOINT_PATH).exists():
        raise RuntimeError(
            f"Model checkpoint not found at: {CHECKPOINT_PATH}\n"
            "Set the CHECKPOINT_PATH environment variable to the correct path."
        )

    print(f"\n🔄 Loading model from: {CHECKPOINT_PATH}")
    print(f"   Device: {_device}")
    _model = load_model(CHECKPOINT_PATH, _device)
    print("✅ Model loaded and ready for inference.\n")
    return _model, _device


# ---------------------------------------------------------------------------
# Grad-CAM heatmap generation (returns base64 PNG or None)
# ---------------------------------------------------------------------------


def _generate_gradcam_base64(
    model, image_path: str, device: str
) -> str | None:
    """
    Produce a Grad-CAM overlay on the detected face and return it as a
    base64-encoded PNG string.  Returns None if grad-cam is unavailable or
    any error occurs.
    """
    if not HAS_GRADCAM:
        return None

    try:
        # Read image and crop face (same pipeline as inference.py)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        face_bgr, _ = crop_face(img_bgr)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(face_rgb)
        tensor = preprocess(img_pil, Config.IMAGE_SIZE).to(device)

        # Compute Grad-CAM
        cam_obj = GradCAM(
            model=model, target_layers=[model.get_target_layer()]
        )
        targets = [ClassifierOutputTarget(0)]
        gray_cam = cam_obj(input_tensor=tensor, targets=targets)[0]  # (H, W)

        # Resize to face crop dimensions and normalise
        resized = cv2.resize(gray_cam, (face_rgb.shape[1], face_rgb.shape[0]))
        resized = (resized - resized.min()) / (
            resized.max() - resized.min() + 1e-8
        )

        # Apply JET colourmap and blend with original face
        heatmap_colour = cv2.applyColorMap(
            (resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

        overlay = (
            face_rgb.astype(np.float32) / 255.0 * 0.55
            + heatmap_rgb.astype(np.float32) / 255.0 * 0.45
        )
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        # Encode to PNG buffer
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", overlay_bgr)
        if not success:
            return None

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# POST /detect — main detection endpoint
# ---------------------------------------------------------------------------


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accept an uploaded image or video file and return a deepfake detection
    result as JSON.

    Returns
    -------
    {
        "label":       "REAL" | "FAKE" | "UNCERTAIN",
        "confidence":  float,       # how confident in the label
        "fake_prob":   float,       # raw P(fake) from the model
        "filename":    str,
        "heatmap_overlay_base64": str | null   # base64 PNG (images only)
    }
    """
    # Validate filename / extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Supported images: {sorted(IMAGE_EXTENSIONS)}, "
                f"videos: {sorted(VIDEO_EXTENSIONS)}."
            ),
        )

    is_image = suffix in IMAGE_EXTENSIONS

    # Save uploaded bytes to a temp file
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read uploaded file: {e}"
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Lazy-load model
        try:
            model, device = _get_model()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        # Run prediction
        if is_image:
            label, prob = predict_image(model, tmp_path, device)
        else:
            label, prob, _ = predict_video(model, tmp_path, device)

        # Confidence: how sure we are about the predicted label
        confidence = prob if label == "FAKE" else 1 - prob

        # Grad-CAM heatmap (images only)
        heatmap_b64: str | None = None
        if is_image:
            heatmap_b64 = _generate_gradcam_base64(model, tmp_path, device)

        return {
            "label": label,
            "confidence": round(float(confidence), 6),
            "fake_prob": round(float(prob), 6),
            "filename": file.filename,
            "heatmap_overlay_base64": heatmap_b64,
        }

    except HTTPException:
        raise  # re-raise HTTP errors as-is
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {e}"
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": _device or "not initialised",
        "checkpoint": CHECKPOINT_PATH,
        "gradcam_available": HAS_GRADCAM,
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("=" * 64)
    print("  🛡️  Deepfake Detection API Server")
    print("=" * 64)
    print()
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Grad-CAM   : {'✅ available' if HAS_GRADCAM else '❌ not installed (pip install grad-cam)'}")
    print(f"  Device     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    print("  Endpoints:")
    print("    POST /detect   — upload an image/video for detection")
    print("    GET  /health   — server health check")
    print("    GET  /docs     — interactive Swagger UI")
    print()
    print("  Example (curl):")
    print('    curl -X POST http://localhost:8000/detect \\')
    print('         -F "file=@photo.jpg"')
    print()
    print("  Starting server on http://0.0.0.0:8000 ...")
    print("=" * 64)

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
