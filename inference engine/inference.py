"""
Inference — EfficientNet-B4 Deepfake Detector with Grad-CAM heatmaps.

Grad-CAM replaces the broken SAM attention from the old Xception model.
It computes the gradient of the FAKE score w.r.t. the last conv block,
producing spatial heatmaps that highlight actual forgery regions
(blending edges, eye-texture inconsistencies, jawline warping, etc.)

Usage
-----
    python inference.py --source face.jpg                     # single image
    python inference.py --source dir/                         # directory
    python inference.py --source video.mp4 --sample-every 4  # video
    python inference.py --source face.jpg --visualise         # Grad-CAM
    python inference.py --source dir/ --output results.csv    # CSV export
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from model import DeepfakeDetector

# ── MediaPipe face detector (new Tasks API, mediapipe>=0.10.30) ───────────────
# Requires: mediapipe_face_model.tflite next to this script.
# Download: https://storage.googleapis.com/mediapipe-models/face_detector/
#           blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
_MP_DETECTOR  = None
HAS_MEDIAPIPE = False

def _load_mp_detector():
    global _MP_DETECTOR, HAS_MEDIAPIPE
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
    model_path = Path(__file__).parent / "mediapipe_face_model.tflite"
    if not model_path.exists():
        print("[INFO] mediapipe_face_model.tflite not found — using Haar Cascade.")
        return None
    try:
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        
        # Explicitly use CPU delegate to fix issues on PCs with no dedicated GPU
        opts = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU
            ),
            min_detection_confidence=0.4,
        )
        _MP_DETECTOR  = mp_vision.FaceDetector.create_from_options(opts)
        HAS_MEDIAPIPE = True
        return _MP_DETECTOR
    except Exception as e:
        print(f"[INFO] MediaPipe init failed ({e}) — using Haar Cascade.")
        return None



# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> DeepfakeDetector:
    model = DeepfakeDetector(num_classes=Config.NUM_CLASSES,
                             pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    metrics = ckpt.get("metrics", {})
    print(f"\n✅ Loaded: {ckpt_path}")
    if metrics:
        va = metrics.get("val_acc")
        au = metrics.get("val_auc")
        va = max(v for v in va if v is not None) if isinstance(va, list) and va else va
        au = max(v for v in au if v is not None) if isinstance(au, list) and au else au
        print(f"   Val Acc : {va:.4f}" if va else "   Val Acc : N/A")
        print(f"   Val AUC : {au:.4f}" if au else "   Val AUC : N/A")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _transform(image_size: int = 380) -> transforms.Compose:
    cfg  = Config
    mean = getattr(cfg, "IMG_MEAN", [0.485, 0.456, 0.406])
    std  = getattr(cfg, "IMG_STD",  [0.229, 0.224, 0.225])
    crop_size = int(image_size * 0.78)  # 296 for 380px — matches training
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess(img: Image.Image, image_size: int = 380) -> torch.Tensor:
    return _transform(image_size)(img.convert("RGB")).unsqueeze(0)


# ── Face detection ────────────────────────────────────────────────────────────

_CASCADE = None


def _load_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def crop_face(img_bgr: np.ndarray,
              margin: float = 0.05) -> tuple[np.ndarray, bool]:
    """Crop the largest face from an image, with margin padding.
    Uses MediaPipe BlazeFace (Tasks API) if the model file is present,
    otherwise falls back to OpenCV Haar Cascade.
    """
    h, w = img_bgr.shape[:2]

    # ── MediaPipe path (Tasks API, mediapipe>=0.10.30) ───────────────────────
    detector = _load_mp_detector()
    if detector is not None:
        try:
            import mediapipe as mp
            rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)
            if result.detections:
                best = max(result.detections, key=lambda d: d.categories[0].score)
                bb   = best.bounding_box
                fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height
                px = int(fw * margin); py = int(fh * margin)
                x1 = max(fx - px, 0);  y1 = max(fy - py, 0)
                x2 = min(fx + fw + px, w); y2 = min(fy + fh + py, h)
                if x2 > x1 and y2 > y1:
                    return img_bgr[y1:y2, x1:x2], True
        except Exception:
            pass  # fall through to Haar Cascade


    # ── Haar Cascade fallback ─────────────────────────────────────────────────
    gray  = _load_cascade().detectMultiScale(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(gray) > 0:
        areas = [fw * fh for (_, _, fw, fh) in gray]
        fx, fy, fw, fh = gray[int(np.argmax(areas))]
        px = int(fw * margin); py = int(fh * margin)
        x1 = max(fx - px, 0);  y1 = max(fy - py, 0)
        x2 = min(fx + fw + px, w); y2 = min(fy + fh + py, h)
        return img_bgr[y1:y2, x1:x2], True

    # ── No face found — centre crop ───────────────────────────────────────────
    s  = int(min(h, w) * 0.70)
    cx, cy = w // 2, h // 2
    x1 = max(cx - s // 2, 0); y1 = max(cy - s // 2, 0)
    x2 = min(x1 + s, w);      y2 = min(y1 + s, h)
    return img_bgr[y1:y2, x1:x2], False


# ── Core prediction ───────────────────────────────────────────────────────────

DEFAULT_UNCERTAIN_LOW  = 0.35
DEFAULT_UNCERTAIN_HIGH = 0.50


def _classify(prob: float, threshold: float,
              uncertain_low: float = DEFAULT_UNCERTAIN_LOW,
              uncertain_high: float = DEFAULT_UNCERTAIN_HIGH) -> str:
    """Classify with confidence-based rejection.
    Returns REAL, FAKE, or UNCERTAIN."""
    if uncertain_low <= prob <= uncertain_high:
        return "UNCERTAIN"
    return "FAKE" if prob > threshold else "REAL"


@torch.no_grad()
def predict(model, tensor: torch.Tensor,
            device: str, threshold: float = 0.50,
            uncertain_low: float = DEFAULT_UNCERTAIN_LOW,
            uncertain_high: float = DEFAULT_UNCERTAIN_HIGH) -> tuple[str, float]:
    tensor = tensor.to(device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits, _, _ = model(tensor)
    prob  = torch.sigmoid(logits).item()
    label = _classify(prob, threshold, uncertain_low, uncertain_high)
    return label, prob


def predict_image(model, path: str, device: str,
                  image_size: int = Config.IMAGE_SIZE,
                  threshold: float = 0.50,
                  tta: bool = True,
                  uncertain_low: float = DEFAULT_UNCERTAIN_LOW,
                  uncertain_high: float = DEFAULT_UNCERTAIN_HIGH) -> tuple[str, float]:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        img_pil = Image.open(path).convert("RGB")
    else:
        face_bgr, found = crop_face(img_bgr)
        if not found:
            print(f"   [WARN] No face detected in {Path(path).name} -- using centre crop")
        img_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

    if tta:
        # Test-Time Augmentation: average over original + horizontal flip
        t1 = preprocess(img_pil, image_size)
        t2 = preprocess(img_pil.transpose(Image.FLIP_LEFT_RIGHT), image_size)
        _, p1 = predict(model, t1, device, threshold, uncertain_low, uncertain_high)
        _, p2 = predict(model, t2, device, threshold, uncertain_low, uncertain_high)
        prob  = (p1 + p2) / 2.0
        label = _classify(prob, threshold, uncertain_low, uncertain_high)
        return label, prob

    return predict(model, preprocess(img_pil, image_size), device,
                   threshold, uncertain_low, uncertain_high)


def predict_video(model, video_path: str, device: str,
                  image_size: int = 380,
                  sample_every: int = 8) -> tuple[str, float, list[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return "UNKNOWN", 0.5, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 1
    print(f"\n   🎬 {Path(video_path).name}  |  {total/fps:.1f}s  |  {total} frames")
    probs: list[float] = []
    faces_found = 0
    fidx = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        if fidx % sample_every == 0:
            face, found = crop_face(frm)
            if found:
                faces_found += 1
            pil    = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            _, p   = predict(model, preprocess(pil, image_size), device)
            probs.append(p)
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            print(f"   Frame {fidx:5d}  [{bar}]  {p:.3f}  → {'FAKE' if p>0.5 else 'REAL'}")
        fidx += 1
    cap.release()
    if not probs:
        return "UNKNOWN", 0.5, []
    avg  = float(np.mean(probs))
    n_f  = sum(1 for p in probs if p > 0.5)
    print(f"\n   FAKE frames: {n_f}/{len(probs)}  |  avg prob: {avg:.4f}")
    return ("FAKE" if avg > 0.5 else "REAL"), avg, probs


# ── Grad-CAM heatmap ──────────────────────────────────────────────────────────

def visualise_gradcam(model, image_path: str, device: str,
                      save_path: str | None = None):
    """
    Grad-CAM spatial heatmap overlaid on the input face.

    Highlights the pixels that INCREASED the FAKE prediction score.
    Grad-CAM is always meaningful because it uses real gradients from the
    forward pass — unlike the old SAM attention which was trained from scratch
    and degenerated to near-uniform maps.

    Requires:  pip install grad-cam
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as mpl_cm
    except ImportError:
        print("❌  matplotlib required — pip install matplotlib")
        return

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        HAS_GRADCAM = True
    except ImportError:
        HAS_GRADCAM = False
        print("⚠️  pytorch-grad-cam not found — pip install grad-cam")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌  Cannot read: {image_path}")
        return

    face_bgr, _ = crop_face(img_bgr)
    face_rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_pil     = Image.fromarray(face_rgb)
    tensor      = preprocess(img_pil, Config.IMAGE_SIZE).to(device)

    label, prob = predict(model, tensor, device)
    colour = "#e53935" if label == "FAKE" else "#43a047"

    if HAS_GRADCAM:
        cam_obj   = GradCAM(model=model,
                             target_layers=[model.get_target_layer()])
        targets   = [ClassifierOutputTarget(0)]
        gray_cam  = cam_obj(input_tensor=tensor, targets=targets)[0]  # (H, W)
        resized   = cv2.resize(gray_cam,
                               (face_rgb.shape[1], face_rgb.shape[0]))
        resized   = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        coloured  = mpl_cm.jet(resized)[..., :3]
        overlay   = (face_rgb / 255.0 * 0.55 + coloured * 0.45).clip(0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e293b"); ax.axis("off")
        ax1.imshow(face_rgb)
        ax1.set_title("Original Face", color="#f1f5f9",
                      fontsize=12, fontweight="bold")
        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM  →  {label}  ({prob:.3f})",
                      color=colour, fontsize=12, fontweight="bold")
        fig.suptitle(f"Grad-CAM  —  {Path(image_path).name}",
                     color="#f1f5f9", fontsize=13, y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.imshow(face_rgb); ax.axis("off")
        ax.set_title(f"{label}  ({prob:.3f})", color=colour,
                     fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150,
                    facecolor="#0f172a")
        print(f"\n   💾 Grad-CAM → {save_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B4 Deepfake Detector — Inference"
    )
    parser.add_argument("--source",       required=True)
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--output",       default=None,
                        help="Save results CSV")
    parser.add_argument("--visualise",    action="store_true",
                        help="Grad-CAM heatmap (single image only)")
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--threshold",    type=float, default=0.50,
                        help="FAKE probability threshold (default: 0.50)")
    parser.add_argument("--uncertain-low",  type=float, default=0.35,
                        help="Lower bound of uncertain zone (default: 0.35)")
    parser.add_argument("--uncertain-high", type=float, default=0.50,
                        help="Upper bound of uncertain zone (default: 0.50)")
    parser.add_argument("--no-tta",       action="store_true",
                        help="Disable test-time augmentation")
    parser.add_argument("--no-rejection", action="store_true",
                        help="Disable confidence-based rejection (outputs FAKE/REAL only)")
    args = parser.parse_args()

    # By default we use the rejection zone unless explicitly disabled
    unc_lo = 999.0 if args.no_rejection else args.uncertain_low
    unc_hi = -1.0 if args.no_rejection else args.uncertain_high

    cfg    = Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    _load_mp_detector()
    print(f"Face detector: {'MediaPipe BlazeFace' if HAS_MEDIAPIPE else 'Haar Cascade (fallback)'}")
    print(f"Threshold    : {args.threshold}")
    print(f"TTA          : {'off' if args.no_tta else 'on (flip)'}")
    print(f"Rejection    : [{unc_lo:.2f} — {unc_hi:.2f}] → UNCERTAIN")

    ckpt = args.checkpoint or str(cfg.CHECKPOINT_DIR / "best_model.pth")
    if not Path(ckpt).exists():
        print(f"❌  Checkpoint not found: {ckpt}"); sys.exit(1)

    model   = load_model(ckpt, device)
    source  = Path(args.source)
    results: list[tuple] = []

    VIDEO  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source.is_file():
        if source.suffix.lower() in VIDEO:
            label, avg, _ = predict_video(model, str(source), device,
                                          cfg.IMAGE_SIZE, args.sample_every)
            conf = avg if label == "FAKE" else 1 - avg
            print(f"\n{'='*50}\n  VERDICT: {label}  (conf={conf:.4f})\n{'='*50}\n")
            results.append((str(source), label, avg))
        elif source.suffix.lower() in IMAGES:
            label, prob = predict_image(model, str(source), device,
                                        cfg.IMAGE_SIZE, args.threshold,
                                        tta=not args.no_tta,
                                        uncertain_low=unc_lo,
                                        uncertain_high=unc_hi)
            conf = prob if label == "FAKE" else 1 - prob
            verdict_icon = "⚠️" if label == "UNCERTAIN" else ""
            print(f"\n{'='*50}\n  {source.name}\n  VERDICT: {label} {verdict_icon} "
                  f"conf={conf:.4f}  fake_prob={prob:.4f}\n{'='*50}\n")
            results.append((str(source), label, prob))
            if args.visualise:
                save_vis = str(source.parent / (source.stem + "_gradcam.png"))
                visualise_gradcam(model, str(source), device, save_vis)
        else:
            print(f"❌  Unsupported extension: {source.suffix}"); sys.exit(1)

    elif source.is_dir():
        files = sorted(p for p in source.iterdir()
                       if p.suffix.lower() in IMAGES | VIDEO)
        if not files:
            print(f"❌  No media in: {source}"); sys.exit(1)
        print(f"\nInference on {len(files)} files…\n")
        print(f"  {'File':<45} {'Verdict':<11} {'Conf':>6}  {'FakeProb':>8}")
        print(f"  {'-'*45} {'-'*11} {'-'*6}  {'-'*8}")
        for p in files:
            if p.suffix.lower() in VIDEO:
                label, prob, _ = predict_video(model, str(p), device,
                                               cfg.IMAGE_SIZE, args.sample_every)
            else:
                label, prob = predict_image(model, str(p), device,
                                            cfg.IMAGE_SIZE, args.threshold,
                                            tta=not args.no_tta,
                                            uncertain_low=unc_lo,
                                            uncertain_high=unc_hi)
            conf = prob if label == "FAKE" else 1 - prob
            print(f"  {p.name:<45} {label:<11} {conf:>6.4f}  {prob:>8.4f}")
            results.append((str(p), label, prob))
    else:
        print(f"❌  Not found: {source}"); sys.exit(1)

    if args.output and results:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "prediction", "prob_fake"])
            writer.writerows(results)
        print(f"Results → {args.output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Explicitly close the MediaPipe detector to avoid a harmless
        # __del__ TypeError on Windows during interpreter shutdown.
        if _MP_DETECTOR is not None:
            try:
                _MP_DETECTOR.close()
            except Exception:
                pass
