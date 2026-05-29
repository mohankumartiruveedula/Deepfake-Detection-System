"""
Configuration for EfficientNet-B4 Deepfake Detector — DF40 Dataset.

Dataset
-------
  DF40 (NeurIPS 2024) — 40 manipulation methods, million-scale.
  Pre-processed face crops downloaded from: https://github.com/YZY-stack/DF40
  After running prepare_df40.py the layout is:
      datasets/df40_combined/
          train/ real/  fake/
          val/   real/  fake/
          test/  real/  fake/

Colab Usage
-----------
  The colab_notebook.py overrides DATA_DIR and CHECKPOINT_DIR at runtime,
  so no manual path edits are needed when training on Google Colab.
"""

from pathlib import Path


class Config:
    # ─────────────────────────────────────────────────────────────────────────
    # Paths  (Colab overrides these at runtime — see colab_notebook.py)
    # ─────────────────────────────────────────────────────────────────────────
    PROJECT_DIR = Path(__file__).parent

    # DF40 root — where you downloaded / extracted the DF40 dataset
    DF40_RAW_DIR = Path(r"/content/drive/MyDrive/DF40")   # Colab default

    # Prepared combined split (output of prepare_df40.py)
    DATA_DIR = PROJECT_DIR / "datasets" / "df40_combined"

    # Outputs
    OUTPUT_DIR     = PROJECT_DIR / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR        = OUTPUT_DIR / "logs"

    # ─────────────────────────────────────────────────────────────────────────
    # DF40 Balanced Sampling (used by prepare_df40.py)
    # ─────────────────────────────────────────────────────────────────────────
    # Maximum FAKE images per method to keep balanced training.
    # Set to None to use all available (may create imbalance if methods vary
    # greatly in size).
    MAX_FAKE_PER_METHOD = 10000   # Will auto-balance to real count / n_methods

    # Fraction of data to set aside for val and test.
    VAL_SPLIT  = 0.10   # 10% validation
    TEST_SPLIT = 0.10   # 10% test

    # ─────────────────────────────────────────────────────────────────────────
    # Model / Input
    # ─────────────────────────────────────────────────────────────────────────
    # EfficientNet-B4 native resolution is 380px.  Deepfake artefacts are
    # pixel-level — downsampling to 320px destroys forensic evidence.
    # Use 380px for best accuracy.  If you get GPU OOM, reduce BATCH_SIZE to 12.
    IMAGE_SIZE  = 380
    NUM_CLASSES = 1     # Binary: real=0, fake=1

    # ImageNet normalisation — REQUIRED for pretrained EfficientNet weights.
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    # ─────────────────────────────────────────────────────────────────────────
    # Training Hyperparameters
    # ─────────────────────────────────────────────────────────────────────────
    # Batch size:
    #   RTX 4050 6GB  (380px, frozen backbone)   → 16
    #   RTX 4050 6GB  (380px, unfrozen backbone) → 12 if OOM, else 16
    #   Colab T4 16GB → 32  (auto-set by colab_notebook.py)
    #   Colab A100 40GB→ 48  (auto-set by colab_notebook.py)
    BATCH_SIZE   = 16
    NUM_WORKERS  = 6
    NUM_EPOCHS   = 50   # Extended for Phase 4 (SRM + Tight Crops)

    # Fine-tuning learning rate — keep small to preserve pretrained features.
    LEARNING_RATE    = 3e-5

    # Backbone receives 10x smaller LR than the head during full fine-tuning.
    BACKBONE_LR_MULT = 0.1   # effective backbone LR = 3e-6

    WEIGHT_DECAY  = 5e-4
    LABEL_SMOOTHING = 0.10

    # Freeze backbone for first N epochs (head-only warm-up).
    UNFREEZE_EPOCH = 5

    # MixUp: blends pairs of training samples to prevent overfitting.
    MIXUP_ALPHA = 0.2   # 0.0 = disabled

    # Weighted sampling + loss to handle real/fake imbalance.
    USE_WEIGHTED_LOSS = True

    # Early stopping
    EARLY_STOP_PATIENCE = 10


    # ─────────────────────────────────────────────────────────────────────────
    # Reproducibility
    # ─────────────────────────────────────────────────────────────────────────
    RANDOM_SEED = 42

    # ─────────────────────────────────────────────────────────────────────────
    # Device
    # ─────────────────────────────────────────────────────────────────────────
    DEVICE = "cuda"   # auto-detected in train.py / colab_notebook.py
