"""
extract_celebdf_faces.py — Extract face crops from CelebDF real videos.

Extracts frames at a configurable interval, detects faces using MediaPipe
(with Haar Cascade fallback), crops them, and saves as JPEG images into
the existing DF40 training set to diversify the 'real' class.

Usage
-----
    python extract_celebdf_faces.py \
        --celebdf-root "C:\path\to\CelebDF" \
        --out-dir "datasets\df40_combined" \
        --frames-per-video 15 \
        --target-size 380
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Face detection setup ──────────────────────────────────────────────────────

_MP_DETECTOR = None
HAS_MEDIAPIPE = False

def _init_mediapipe():
    global _MP_DETECTOR, HAS_MEDIAPIPE
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
    model_path = Path(__file__).parent / "mediapipe_face_model.tflite"
    if not model_path.exists():
        return None
    try:
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions
        opts = mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5,
        )
        _MP_DETECTOR = mp_vision.FaceDetector.create_from_options(opts)
        HAS_MEDIAPIPE = True
        return _MP_DETECTOR
    except Exception:
        return None


_CASCADE = None

def _load_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def detect_and_crop_face(img_bgr: np.ndarray,
                         target_size: int = 380,
                         margin: float = 0.30):
    """Detect the largest face and return a square crop resized to target_size.
    Returns None if no face is found."""
    h, w = img_bgr.shape[:2]

    # ── MediaPipe path ────────────────────────────────────────────────────
    detector = _init_mediapipe()
    if detector is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.detections:
                best = max(result.detections, key=lambda d: d.categories[0].score)
                bb = best.bounding_box
                fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height
                px = int(fw * margin); py = int(fh * margin)
                x1 = max(fx - px, 0);  y1 = max(fy - py, 0)
                x2 = min(fx + fw + px, w); y2 = min(fy + fh + py, h)
                if x2 > x1 + 20 and y2 > y1 + 20:
                    crop = img_bgr[y1:y2, x1:x2]
                    return cv2.resize(crop, (target_size, target_size))
        except Exception:
            pass

    # ── Haar Cascade fallback ─────────────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _load_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        areas = [fw * fh for (_, _, fw, fh) in faces]
        fx, fy, fw, fh = faces[int(np.argmax(areas))]
        px = int(fw * margin); py = int(fh * margin)
        x1 = max(fx - px, 0);  y1 = max(fy - py, 0)
        x2 = min(fx + fw + px, w); y2 = min(fy + fh + py, h)
        if x2 > x1 + 20 and y2 > y1 + 20:
            crop = img_bgr[y1:y2, x1:x2]
            return cv2.resize(crop, (target_size, target_size))

    return None


def extract_faces_from_video(video_path: Path,
                             out_dir: Path,
                             prefix: str,
                             frames_per_video: int = 15,
                             target_size: int = 380) -> int:
    """Extract evenly-spaced face crops from a video.
    Returns the number of faces saved."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frames_per_video:
        step = 1
    else:
        step = total_frames // frames_per_video

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0 and saved < frames_per_video:
            crop = detect_and_crop_face(frame, target_size)
            if crop is not None:
                fname = f"{prefix}_f{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_dir / fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from CelebDF real videos into DF40 training set."
    )
    parser.add_argument("--celebdf-root", required=True, type=Path,
                        help="Root of CelebDF dataset.")
    parser.add_argument("--out-dir", default=None, type=Path,
                        help="DF40 combined dataset directory. "
                             "Default: datasets/df40_combined")
    parser.add_argument("--frames-per-video", type=int, default=15,
                        help="Number of frames to extract per video (default: 15)")
    parser.add_argument("--target-size", type=int, default=380,
                        help="Output face crop size in pixels (default: 380)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to add faces to (default: train val test)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    out_dir = args.out_dir or (script_dir / "datasets" / "df40_combined")

    # Collect all real video folders
    real_dirs = []
    for name in ["Celeb-real", "YouTube-real"]:
        d = args.celebdf_root / name
        if d.is_dir():
            real_dirs.append((name, d))
            print(f"Found: {d}  ({len(list(d.glob('*.mp4')))} videos)")

    if not real_dirs:
        print("[ERROR] No Celeb-real or YouTube-real folders found.")
        sys.exit(1)

    # Count existing real images
    train_real_dir = out_dir / "train" / "real"
    existing_count = sum(1 for _ in train_real_dir.glob("*")
                         if _.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if train_real_dir.exists() else 0
    print(f"\nExisting real images in train/real: {existing_count:,}")

    # Gather all real videos
    all_videos = []
    for folder_name, folder_path in real_dirs:
        for v in sorted(folder_path.glob("*.mp4")):
            all_videos.append((folder_name, v))
    print(f"Total CelebDF real videos: {len(all_videos)}")

    # Split videos 80/10/10 to match train/val/test
    import random
    random.seed(42)
    random.shuffle(all_videos)
    n = len(all_videos)
    n_test = int(n * 0.10)
    n_val  = int(n * 0.10)
    splits = {
        "test":  all_videos[:n_test],
        "val":   all_videos[n_test:n_test + n_val],
        "train": all_videos[n_test + n_val:],
    }

    total_saved = 0
    for split_name in args.splits:
        videos = splits.get(split_name, [])
        if not videos:
            continue

        dest = out_dir / split_name / "real"
        dest.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Extracting faces for [{split_name}] — {len(videos)} videos")
        print(f"  Output: {dest}")
        print(f"{'='*60}")

        split_saved = 0
        for i, (folder_name, vpath) in enumerate(videos):
            prefix = f"celebdf_{folder_name}_{vpath.stem}"
            count = extract_faces_from_video(
                vpath, dest, prefix,
                frames_per_video=args.frames_per_video,
                target_size=args.target_size,
            )
            split_saved += count
            if (i + 1) % 20 == 0 or (i + 1) == len(videos):
                print(f"  [{i+1}/{len(videos)}] {vpath.name}: "
                      f"{count} faces | split total: {split_saved:,}")

        total_saved += split_saved
        print(f"  → {split_name}: {split_saved:,} face crops saved")

    print(f"\n{'='*60}")
    print(f"  DONE — {total_saved:,} total CelebDF face crops extracted")
    print(f"{'='*60}")

    # Show new balance
    for split_name in ("train", "val", "test"):
        real_dir = out_dir / split_name / "real"
        fake_dir = out_dir / split_name / "fake"
        n_real = sum(1 for _ in real_dir.glob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if real_dir.exists() else 0
        n_fake = sum(1 for _ in fake_dir.glob("*")
                     if _.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if fake_dir.exists() else 0
        ratio = n_real / n_fake if n_fake > 0 else 0
        print(f"  {split_name}: {n_real:,} real / {n_fake:,} fake  "
              f"(ratio: {ratio:.2f})")

    print(f"\n[NOTE] Phase 1 model backed up at: "
          f"outputs/checkpoints/phase1_best_model.pth")
    print(f"[NEXT] Run:  python train.py  to retrain with diverse real data")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _MP_DETECTOR is not None:
            try:
                _MP_DETECTOR.close()
            except Exception:
                pass
