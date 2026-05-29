"""
Dataset class for the EfficientNet-B4 Deepfake Detector.

Expected folder structure (produced by prepare_df40.py):
    DATA_DIR/
        train/ real/  fake/
        val/   real/  fake/
        test/  real/  fake/

Labels:
    real images → 0    fake images → 1

Augmentation strategy (training only)
--------------------------------------
Designed to be robust to bad image conditions commonly seen when deepfakes
are distributed in the wild (bad lighting, colour temperature shifts, blur,
compression artefacts, partial occlusion):

  · ColorJitter (±40% brightness/contrast, ±30% saturation, ±15° hue)
        → exposure shifts, warm/cool colour temperature, over/under-exposed
  · RandomAutocontrast + RandomEqualize
        → normalisation artefacts, histogram-stretched images
  · GaussianBlur + RandomAdjustSharpness
        → motion blur, out-of-focus, phone-quality images
  · RandomPosterize
        → JPEG compression, low bit-depth, re-encoded video frames
  · RandomErasing
        → partial occlusion by hands, glasses, hats, watermarks
  · RandomPerspective
        → geometric distortion, non-frontal camera angles
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """Loads real / fake face-crop images from the prepared DF40 splits."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        data_dir,
        split: str = "train",
        image_size: int = 380,
        label_smoothing: float = 0.0,
        img_mean=None,
        img_std=None,
    ):
        self.split           = split
        self.label_smoothing = label_smoothing
        split_dir = Path(data_dir) / split

        # ImageNet stats — required for pretrained EfficientNet-B4 backbone
        if img_mean is None:
            img_mean = [0.485, 0.456, 0.406]
        if img_std is None:
            img_std  = [0.229, 0.224, 0.225]

        # ── Collect samples ───────────────────────────────────────────────────
        self.samples: list[tuple[str, int]] = []
        for label_name, label_int in [("real", 0), ("fake", 1)]:
            folder = split_dir / label_name
            if not folder.exists():
                raise FileNotFoundError(
                    f"Missing folder: {folder}\n"
                    f"Run  python prepare_df40.py  first."
                )
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(p), label_int))

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[{split:5s}] {len(self.samples):7,} images  "
              f"(real: {n_real:,}, fake: {n_fake:,})")

        # ── Transforms ───────────────────────────────────────────────────────
        norm = transforms.Normalize(mean=img_mean, std=img_std)

        # Simulated tighter crop (5% margin instead of 20% original)
        crop_size = int(image_size * 0.78)

        if split == "train":
            self.transform = transforms.Compose([
                # ── Spatial ──────────────────────────────────────────────────
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.Resize((image_size + 40, image_size + 40)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.10, p=0.3),

                # ── Colour / exposure robustness ─────────────────────────────
                # Simulates bad white-balance, colour temperature shifts,
                # over/under-exposed images.
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.40,   # ±40% — dark rooms / harsh sun
                        contrast=0.40,     # ±40% — hazy / overexposed
                        saturation=0.30,   # ±30% — warm/cool tones
                        hue=0.15,          # ±15° — colour temperature shift
                    )
                ], p=0.70),
                transforms.RandomAutocontrast(p=0.30),
                transforms.RandomEqualize(p=0.20),

                # ── Sharpness / blur robustness ──────────────────────────────
                transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.20),
                transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.20),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))
                ], p=0.25),

                # ── Compression / quantisation simulation ────────────────────
                transforms.RandomApply([
                    transforms.RandomPosterize(bits=4)
                ], p=0.15),
                transforms.RandomGrayscale(p=0.05),

                # ── Tensor ───────────────────────────────────────────────────
                transforms.ToTensor(),
                norm,

                # ── Occlusion simulation ─────────────────────────────────────
                transforms.RandomErasing(
                    p=0.30,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value=0,
                ),
            ])
        else:
            # Val / test: deterministic — simulate tight crop, then standard resize
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                norm,
            ])

    # ── Label helpers ────────────────────────────────────────────────────────

    def _smooth_label(self, label_int: int) -> float:
        eps = self.label_smoothing
        if eps <= 0.0:
            return float(label_int)
        return eps / 2.0 if label_int == 0 else 1.0 - eps / 2.0

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_int = self.samples[idx]
        img   = Image.open(img_path).convert("RGB")
        img   = self.transform(img)
        label = torch.tensor(self._smooth_label(label_int), dtype=torch.float32)
        return img, label, img_path

    def class_counts(self) -> tuple[int, int]:
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        return n_real, n_fake


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_data_loaders(config):
    """
    Build train / val / test DataLoaders from Config.

    Returns
    -------
    train_loader, val_loader, test_loader, pos_weight_tensor_or_None
    """
    img_mean = getattr(config, "IMG_MEAN", [0.485, 0.456, 0.406])
    img_std  = getattr(config, "IMG_STD",  [0.229, 0.224, 0.225])

    train_ds = DeepfakeDataset(
        config.DATA_DIR, split="train",
        image_size=config.IMAGE_SIZE,
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0),
        img_mean=img_mean, img_std=img_std,
    )
    val_ds = DeepfakeDataset(
        config.DATA_DIR, split="val",
        image_size=config.IMAGE_SIZE,
        img_mean=img_mean, img_std=img_std,
    )
    test_ds = DeepfakeDataset(
        config.DATA_DIR, split="test",
        image_size=config.IMAGE_SIZE,
        img_mean=img_mean, img_std=img_std,
    )

    # ── Weighted sampler ─────────────────────────────────────────────────────
    pos_weight = None
    sampler    = None

    if getattr(config, "USE_WEIGHTED_LOSS", True):
        n_real, n_fake = train_ds.class_counts()
        if n_real > 0 and n_fake > 0:
            total          = n_real + n_fake
            sample_weights = [
                total / (2.0 * (n_real if l == 0 else n_fake))
                for _, l in train_ds.samples
            ]
            sampler    = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )
            pos_weight = torch.tensor([n_real / n_fake], dtype=torch.float32)
            print(f"Class balance  real={n_real:,}  fake={n_fake:,}  "
                  f"pos_weight={pos_weight.item():.3f}")

    nw = config.NUM_WORKERS
    pf = 2 if nw > 0 else None   # prefetch 2 batches ahead per worker

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=pf,
    )

    return train_loader, val_loader, test_loader, pos_weight
