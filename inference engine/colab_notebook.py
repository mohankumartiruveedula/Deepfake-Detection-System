"""
EfficientNet-B4 Deepfake Detector — Google Colab Training Notebook
====================================================================
HOW TO USE THIS FILE:
  This file is structured as code blocks separated by # ===CELL N=== headers.
  Copy each block into a SEPARATE Colab cell and run them ONE BY ONE in order.

  Do NOT paste the whole file into one cell — it won't work!

BEFORE RUNNING ANY CELL:
  1. Runtime → Change runtime type → GPU (T4 = free, A100 = Pro)
  2. Upload the entire EfficientNet-B4/ folder to Google Drive at:
         MyDrive/Colab Notebooks/EfficientNet-B4/
     (all .py files must be there)
  3. For DF40: either add the Drive shortcut (Option A in Cell 3)
     or get File IDs from https://github.com/YZY-stack/DF40 (Option B)
"""


# ===CELL 1=== GPU check & install dependencies
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 1 of your Colab notebook."""
import subprocess, sys

def run(cmd):
    """Run a shell command and print output. Raises on failure."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")

# Show which GPU Colab gave us
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

# Install required packages
run(f"{sys.executable} -m pip install timm grad-cam scikit-learn matplotlib seaborn tqdm gdown --quiet")

print("\n✅ Cell 1 complete — dependencies installed")


# ===CELL 2=== Mount Google Drive & configure paths
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 2 of your Colab notebook."""
import os, sys
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

# ── CONFIGURE THESE PATHS ────────────────────────────────────────────────────
DRIVE_ROOT     = "/content/drive/MyDrive"
PROJECT_DIR    = f"{DRIVE_ROOT}/Colab Notebooks/EfficientNet-B4"  # where you uploaded .py files

# DF40 location — see Cell 3 to decide which path to use:
#   Option A (Drive Shortcut, BEST)   → keep as-is below
#   Option B (gdown to /content/)     → change to "/content/DF40"
DF40_RAW_DIR   = f"{DRIVE_ROOT}/Colab Notebooks/DF40_train"       # ← change if needed

# ── DATA STRATEGY: what goes on Drive vs local SSD ──────────────────────────
# EXTRACT_DIR  → Drive (persists across sessions — zips extracted once)
# PREPARED_DIR → Colab local SSD (/content/) — fast training I/O,
#                 recreated in seconds from symlinks after any restart
# CHECKPOINT_DIR/LOG_DIR → Drive — permanent model saves
# ─────────────────────────────────────────────────────────────────────────────
EXTRACT_DIR    = f"{DRIVE_ROOT}/Colab Notebooks/_df40_ext"  # Drive: persists 💾
PREPARED_DIR   = "/content/df40_combined"                   # local SSD: fast ⚡
CHECKPOINT_DIR = f"{PROJECT_DIR}/outputs/checkpoints"       # Drive: permanent
LOG_DIR        = f"{PROJECT_DIR}/outputs/logs"              # Drive: permanent
STATE_FILE     = f"{PROJECT_DIR}/prepare_state.json"        # Drive: savepoint
# ─────────────────────────────────────────────────────────────────────────────

# Add project to Python path so imports work
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

print(f"Working dir      : {os.getcwd()}")
print(f"PROJECT_DIR      : {PROJECT_DIR}  → exists: {Path(PROJECT_DIR).exists()}")
print(f"DF40_RAW_DIR     : {DF40_RAW_DIR} → exists: {Path(DF40_RAW_DIR).exists()}")
print(f"PREPARED_DIR     : {PREPARED_DIR} (Colab local SSD)")
print(f"EXTRACT_DIR      : {EXTRACT_DIR}  (Colab local SSD)")
print(f"CHECKPOINT_DIR   : {CHECKPOINT_DIR} (Drive)")

# Verify the .py files are present
required = ["model.py", "config.py", "dataset.py", "prepare_df40.py",
            "train.py", "evaluate.py", "inference.py"]
missing = [f for f in required if not Path(PROJECT_DIR, f).exists()]
if missing:
    print(f"\n❌ Missing files: {missing}")
    print("   Upload the full EfficientNet-B4/ folder to MyDrive first!")
else:
    print(f"\n✅ Cell 2 complete — all {len(required)} project files found")


# ===CELL 3=== Access DF40 (choose ONE option below)
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 3 of your Colab notebook.
   Read the options and uncomment ONLY the one you need.
"""
from pathlib import Path   # may already be imported — no harm running again

# ══════════════════════════════════════════════════════════════════════════════
# OPTION A ★ BEST — Drive Shortcut (ZERO download, ZERO quota used)
# ══════════════════════════════════════════════════════════════════════════════
# Use when the DF40 Github link opens a BROWSEABLE FOLDER in Drive.
#
# Steps (do ONCE in your browser, NOT in Colab):
#   1. Open the DF40 Drive link in your browser
#   2. At the top, click the folder name → "Add shortcut to Drive"
#   3. Choose "My Drive" → click "Add shortcut"
#   4. That's it — no download at all!
#
# The folder will appear as MyDrive/Colab Notebooks/DF40_train/.
# Make sure DF40_RAW_DIR in Cell 2 matches that folder name.
# ──────────────────────────────────────────────────────────────────────────────
if Path(DF40_RAW_DIR).exists():
    contents = list(Path(DF40_RAW_DIR).iterdir())
    print(f"✅ OPTION A — DF40 found at: {DF40_RAW_DIR}")
    print(f"   {len(contents)} items: {[c.name for c in contents[:8]]}")
    print("   → Skip to Cell 4")
else:
    print(f"⚠️  {DF40_RAW_DIR} not found.")
    print("   Did you add the Drive shortcut? (see instructions above)")
    print("   If not, use Option B below.")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# OPTION B — gdown to Colab local disk (if Option A fails / link is a zip)
# ══════════════════════════════════════════════════════════════════════════════
# Use when the Drive link is a downloadable ZIP FILE (not a folder).
# Downloads to /content/DF40/ — Colab's local disk (~100 GB free).
# Does NOT use your Drive quota! But re-downloads on session restart.
#
# HOW TO GET THE FILE ID:
#   From: https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
#   Copy:                              ^^^^^^^^^^^^ this part
#   Or folder link: https://drive.google.com/drive/folders/FOLDER_ID_HERE
#
# STEP 1: Paste your IDs below, then STEP 2: uncomment the download block.

DF40_TRAIN_FILE_ID = "PASTE_TRAIN_FILE_ID_HERE"   # ← from DF40 GitHub README
DF40_TEST_FILE_ID  = "PASTE_TEST_FILE_ID_HERE"    # ← from DF40 GitHub README

# import gdown                                     # ← STEP 2: uncomment all below
# LOCAL_DF40 = "/content/DF40"
# Path(LOCAL_DF40).mkdir(parents=True, exist_ok=True)
#
# if DF40_TRAIN_FILE_ID != "PASTE_TRAIN_FILE_ID_HERE":
#     print("⬇️  Downloading training data (~50 GB) to /content/ ...")
#     gdown.download(id=DF40_TRAIN_FILE_ID,
#                    output=f"{LOCAL_DF40}/df40_train.zip", fuzzy=True)
#     print("📦 Extracting...")
#     run(f"unzip -q {LOCAL_DF40}/df40_train.zip -d {LOCAL_DF40}")
#     run(f"rm {LOCAL_DF40}/df40_train.zip")
#
# if DF40_TEST_FILE_ID != "PASTE_TEST_FILE_ID_HERE":
#     print("⬇️  Downloading test data (~93 GB) to /content/ ...")
#     gdown.download(id=DF40_TEST_FILE_ID,
#                    output=f"{LOCAL_DF40}/df40_test.zip", fuzzy=True)
#     print("📦 Extracting...")
#     run(f"unzip -q {LOCAL_DF40}/df40_test.zip -d {LOCAL_DF40}")
#     run(f"rm {LOCAL_DF40}/df40_test.zip")
#
# # IMPORTANT: Update DF40_RAW_DIR after Option B
# DF40_RAW_DIR = LOCAL_DF40    # ← this overrides what was set in Cell 2
# print(f"\n✅ Option B done — DF40 at: {LOCAL_DF40}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL CHECK — runs regardless of which option you used
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"DF40_RAW_DIR : {DF40_RAW_DIR}")
print(f"Exists       : {Path(DF40_RAW_DIR).exists()}")
if Path(DF40_RAW_DIR).exists():
    items = list(Path(DF40_RAW_DIR).iterdir())
    print(f"Contents     : {[i.name for i in items[:10]]}")
    print("\n✅ Cell 3 complete — proceed to Cell 4")
else:
    print("\n❌ DF40 not accessible — fix the path or complete an option above")


# ===CELL 4=== Prepare DF40 — balanced real/fake splits
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 4 of your Colab notebook."""
import os, sys, subprocess
from pathlib import Path

# run() redefined here so Cell 4 works even after a kernel restart
# (it was originally defined in Cell 1, but kernels wipe all variables)
def run(cmd):
    """Run a shell command, print output, raise on failure."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")

# These variables must exist from Cell 2.
# If you restarted the kernel, re-run Cells 1-3 first!
try:
    _ = PROJECT_DIR, DF40_RAW_DIR, PREPARED_DIR
except NameError:
    raise RuntimeError(
        "❌ Variables not defined!\n"
        "   You must run Cells 1 → 2 → 3 before running Cell 4.\n"
        "   If you restarted the kernel, re-run those cells first."
    )

# Verify working directory is correct
print(f"Working dir : {os.getcwd()}")
print(f"prepare_df40 exists: {Path('prepare_df40.py').exists()}")

if not Path("prepare_df40.py").exists():
    # Try to fix the working directory
    os.chdir(PROJECT_DIR)
    print(f"Fixed working dir to: {os.getcwd()}")

if not Path(DF40_RAW_DIR).exists():
    raise FileNotFoundError(
        f"❌ DF40 directory not found: {DF40_RAW_DIR}\n"
        f"   Complete Cell 3 (Option A or B) first to make DF40 accessible."
    )

# ── Detect layout: zip-bundle (Layout C) vs extracted folders (Layout A/B) ──
df40_items   = list(Path(DF40_RAW_DIR).iterdir())
df40_zips    = [p for p in df40_items if p.suffix.lower() == ".zip"]
df40_subdirs = [p for p in df40_items if p.is_dir()]
print(f"\nDF40 folder contents: {len(df40_zips)} zip(s), {len(df40_subdirs)} subfolder(s)")
print(f"Sample items: {[p.name for p in df40_items[:8]]}")

# ── REAL DATA — only needed for Layout C (zip bundle with no real/ folder) ──
# ─────────────────────────────────────────────────────────────────────────────
# The DF40 training Drive folder (DF40_train) contains ONLY fake images —
# one zip per deepfake method (e.g. blendface.zip, fsgan.zip …).
# Real (genuine) images come from FaceForensics++ and MUST be downloaded
# separately before Cell 4 can proceed.
#
# ▶ HOW TO GET THE REAL DATA (do this ONCE in your browser):
#   1. Open this Google Drive link:
#      https://drive.google.com/file/d/1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb
#   2. Click "Add shortcut to Drive" → choose "My Drive" → Add shortcut.
#      (Or download it — the file is ff_real.zip, ~8–15 GB.)
#   3. In a Colab cell, extract it ONCE:
#        !unzip "/content/drive/MyDrive/ff_real.zip" \
#               -d "/content/drive/MyDrive/Colab Notebooks/ff_real"
#   4. Update FF_REAL_DIR below to match the extraction path.
# ─────────────────────────────────────────────────────────────────────────────

# ← SET THIS to wherever the FF++ real images were extracted
FF_REAL_DIR = f"/content/drive/MyDrive/Colab Notebooks/ff_real"

# Uncomment to extract real images (run ONCE — skip on subsequent runs):
# run(f'unzip -q "/content/drive/MyDrive/ff_real.zip" -d "{FF_REAL_DIR}"')

# ── Auto-detect Layout C and build the --real-dir flag ───────────────────────
_REAL_NAMES = {"real", "original", "00_original", "pristine", "source"}
REAL_DIR_FLAG = ""

if df40_zips and not any(d.name.lower() in _REAL_NAMES for d in df40_subdirs):
    # ── Layout C: per-method zip files, no real/ folder ──────────────────────
    print("\n📁 Layout C detected: per-method zip files with no real/ folder.")
    print(f"   Fake zips  : {[z.name for z in df40_zips[:5]]} ...")
    print(f"   Real images expected at: {FF_REAL_DIR}")

    if not Path(FF_REAL_DIR).exists():
        raise FileNotFoundError(
            f"\n❌ Real-image folder not found: {FF_REAL_DIR}\n"
            "\n"
            "   The DF40 training folder only contains FAKE images (one zip per method).\n"
            "   You need to download the FF++ real data separately:\n"
            "\n"
            "   Step 1 — Download real images:\n"
            "     https://drive.google.com/file/d/1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb\n"
            "\n"
            "   Step 2 — Extract in a Colab cell:\n"
            f'     !unzip "/content/drive/MyDrive/ff_real.zip" -d "{FF_REAL_DIR}"\n'
            "\n"
            f"   Step 3 — Update FF_REAL_DIR in Cell 4 to: {FF_REAL_DIR}\n"
            "   Then re-run Cell 4."
        )

    n_real = sum(1 for p in Path(FF_REAL_DIR).rglob("*") if p.is_file())
    print(f"   ✅ Real-image folder found — {n_real:,} image files")
    REAL_DIR_FLAG = f'--real-dir "{FF_REAL_DIR}"'

else:
    # ── Layout A/B: extracted subfolders already present ─────────────────────
    print("\n📁 Layout A/B detected: extracted method folders found — no --real-dir needed.")

# ── Run prepare_df40.py ───────────────────────────────────────────────────────
prepared_real = Path(PREPARED_DIR) / "train" / "real"
# ══════════════════════════════════════════════════════════════════════════════
# LOCAL EXECUTION ALTERNATIVE (run on your Windows PC if Colab keeps dying)
# ══════════════════════════════════════════════════════════════════════════════
# If Colab keeps timing out, you can run preparation locally.
# Requirements: Google Drive for Desktop installed (so Drive appears as a drive
# letter, e.g. G:\My Drive\).
#
# In PowerShell on your PC:
#   cd C:\Users\<you>\..\deepfake_detection\EfficientNet-B4
#   pip install tqdm
#   python prepare_df40.py ^
#     --df40-root "G:\My Drive\Colab Notebooks\DF40_train" ^
#     --real-dir  "G:\My Drive\Colab Notebooks\ff_real" ^
#     --out-dir   "C:\Users\<you>\df40_combined" ^
#     --extract-dir "C:\Users\<you>\df40_extract" ^
#     --state-file "G:\My Drive\Colab Notebooks\prepare_state.json" ^
#     --no-symlinks ^
#     --max-fake-per-method 5000
#
# Output goes to C:\...\df40_combined (local, ~15 GB).
# Then upload df40_combined/ to Drive and update PREPARED_DIR in Cell 2 to
# point to its Drive location for Colab training.
# ══════════════════════════════════════════════════════════════════════════════

if prepared_real.exists() and any(prepared_real.iterdir()):
    print(f"\n✅ Prepared data already exists — skipping prepare_df40.py")
    for split in ("train", "val", "test"):
        for cls in ("real", "fake"):
            d = Path(PREPARED_DIR) / split / cls
            if d.exists():
                count = sum(1 for _ in d.iterdir())
                print(f"   {split}/{cls}: {count:,} images")
else:
    # ── Pre-flight: verify Drive copy is the updated version ───────────────
    chk = subprocess.run(
        [sys.executable, "prepare_df40.py", "--help"],
        capture_output=True, text=True
    )
    if "--state-file" not in chk.stdout:
        raise RuntimeError(
            "❌ The prepare_df40.py on your Drive is OUTDATED.\n"
            "   Re-upload the latest version from:\n"
            "   C:\\...\\deepfake_detection\\EfficientNet-B4\\prepare_df40.py\n"
            f"   to Drive: {PROJECT_DIR}/prepare_df40.py\n"
            "   Then re-run Cells 1-4."
        )

    print(f"\nRunning prepare_df40.py ...")
    print(f"  DF40 source  : {DF40_RAW_DIR}")
    print(f"  Extract dir  : {EXTRACT_DIR}  (Drive — persistent)")
    if REAL_DIR_FLAG:
        print(f"  Real dir     : {FF_REAL_DIR}")
    print(f"  Output dir   : {PREPARED_DIR}  (local SSD)")
    print(f"  State file   : {STATE_FILE}  (Drive savepoint)")
    print(f"  Max per method: 5000")
    print()
    print("⚠️  If this runtime gets killed, just re-run Cells 1→2→3→4.")
    print("   The savepoint on Drive will resume from the last completed zip.\n")

    # ── Build arg list ─────────────────────────────────────────────────────
    cmd_args = [
        sys.executable, "prepare_df40.py",
        "--df40-root",    DF40_RAW_DIR,
        "--out-dir",      PREPARED_DIR,
        "--extract-dir",  EXTRACT_DIR,          # Drive — survives restarts
        "--state-file",   STATE_FILE,           # Drive savepoint file
        "--val-split",   "0.10",
        "--test-split",  "0.10",
        "--max-fake-per-method", "5000",
        "--seed",        "42",
    ]
    if REAL_DIR_FLAG:
        cmd_args += ["--real-dir", FF_REAL_DIR]
    print(f"$ {' '.join(cmd_args)}\n")

    # ── Stream output in real time + keepalive to prevent idle timeout ───────
    # subprocess.run(capture_output=True) buffers ALL output and shows nothing
    # until the process finishes.  Colab sees no output and kills the session.
    # Popen with real-time readline() keeps Colab alive and shows progress.
    import time as _time
    proc = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True, bufsize=1
    )
    last_output = _time.time()
    while True:
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
            last_output = _time.time()
        elif proc.poll() is not None:
            break
        # Keepalive: if no output for 30 s, print a heartbeat
        # (prevents Colab's idle-detection from killing the session)
        if _time.time() - last_output > 30:
            print(f"  ⏳ still running — {_time.strftime('%H:%M:%S')} …", flush=True)
            last_output = _time.time()
    # Drain any remaining stderr
    remaining = proc.stdout.read()
    if remaining:
        print(remaining, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "prepare_df40.py exited with an error (see output above).\n"
            "Check the STDERR lines above for the cause.\n"
            "If you hit a Drive quota error, wait a few minutes and re-run Cell 4."
        )

print("\n✅ Cell 4 complete — data is ready for training")


# ===CELL 5=== Config override + model sanity check
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 5 of your Colab notebook."""
import torch
from config import Config

gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {gpu_mem_gb:.1f} GB")

if gpu_mem_gb >= 35:        # A100 / H100
    Config.BATCH_SIZE = 48
elif gpu_mem_gb >= 14:      # T4 / V100
    Config.BATCH_SIZE = 32
else:
    Config.BATCH_SIZE = 24

Config.DATA_DIR       = Path(PREPARED_DIR)
Config.OUTPUT_DIR     = Path(f"{PROJECT_DIR}/outputs")
Config.CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
Config.LOG_DIR        = Path(LOG_DIR)
Config.NUM_WORKERS    = 4

Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nBATCH_SIZE : {Config.BATCH_SIZE}")
print(f"DATA_DIR   : {Config.DATA_DIR}")
print(f"IMAGE_SIZE : {Config.IMAGE_SIZE}")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.manual_seed(Config.RANDOM_SEED)
torch.cuda.manual_seed_all(Config.RANDOM_SEED)

from model import DeepfakeDetector
model = DeepfakeDetector(pretrained=True).to(device)
dummy = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE, device=device)
with torch.no_grad():
    lgts, _, emb = model(dummy)
print(f"\n✅ Forward pass OK")
print(f"   Logits    : {lgts.shape}")
print(f"   Embedding : {emb.shape}")
print(f"   Params    : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
del model, dummy
torch.cuda.empty_cache()
print("\n✅ Cell 5 complete")


# ===CELL 6=== Load data loaders
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 6."""
from dataset import get_data_loaders

print(f"Loading data from: {Config.DATA_DIR}")
train_loader, val_loader, test_loader, pos_weight = get_data_loaders(Config)
print(f"\n✅ Cell 6 complete — loaders ready (batch={Config.BATCH_SIZE})")


# ===CELL 7=== Build model, loss, optimizer
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 7."""
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from train import train_epoch, evaluate, save_checkpoint

model = DeepfakeDetector(num_classes=Config.NUM_CLASSES,
                          pretrained=True).to(device)
print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

criterion = (nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
             if pos_weight is not None else nn.BCEWithLogitsLoss())

backbone_lr = Config.LEARNING_RATE * getattr(Config, "BACKBONE_LR_MULT", 0.1)
optimizer   = AdamW([
    {"params": model.get_head_params(),     "lr": Config.LEARNING_RATE},
    {"params": model.get_backbone_params(), "lr": backbone_lr},
], weight_decay=Config.WEIGHT_DECAY)
scheduler   = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
scaler      = torch.amp.GradScaler("cuda")

model.freeze_backbone()
print(f"Backbone FROZEN for first {Config.UNFREEZE_EPOCH} warm-up epochs")
print(f"\n✅ Cell 7 complete")


# ===CELL 8=== Training loop
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 8. This is the main training loop."""
import json, time
import numpy as np

start_epoch  = 0
best_val_auc = 0.0
best_val_acc = 0.0
patience_ctr = 0
mixup_alpha  = getattr(Config, "MIXUP_ALPHA", 0.2)
unfreeze_ep  = getattr(Config, "UNFREEZE_EPOCH", 3)

history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [],
    "val_auc":    [], "val_f1":    [], "lr": [],
}

# Auto-resume from last checkpoint if it exists
resume_path = Config.CHECKPOINT_DIR / "latest_model.pth"
if resume_path.exists():
    print(f"Resuming from: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch  = ckpt["epoch"] + 1
    saved_metrics = ckpt.get("metrics", {})
    if isinstance(saved_metrics.get("val_acc"), list):
        history.update(saved_metrics)
        best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
        valid_auc    = [v for v in history.get("val_auc", []) if v is not None]
        best_val_auc = max(valid_auc) if valid_auc else 0.0
    print(f"  Resumed from epoch {start_epoch}, best_auc={best_val_auc:.4f}")

print(f"\n🚀 Training for up to {Config.NUM_EPOCHS} epochs "
      f"(patience={Config.EARLY_STOP_PATIENCE})\n")

for epoch in range(start_epoch, Config.NUM_EPOCHS):
    t0 = time.time()

    if epoch == unfreeze_ep:
        model.unfreeze_backbone()
        print(f"\n🔓 Backbone UNFROZEN at epoch {epoch+1} "
              f"(backbone LR={backbone_lr:.2e})\n")

    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device,
        epoch=epoch+1, total_epochs=Config.NUM_EPOCHS,
        scaler=scaler, mixup_alpha=mixup_alpha,
    )
    val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(
        model, val_loader, criterion, device,
    )

    torch.cuda.empty_cache()
    scheduler.step()
    lr      = scheduler.get_last_lr()[0]
    elapsed = time.time() - t0

    auc_s = f"{val_auc:.4f}" if val_auc is not None else "  N/A "
    f1_s  = f"{val_f1:.4f}"  if val_f1  is not None else "  N/A "
    print(
        f"Ep {epoch+1:3d}/{Config.NUM_EPOCHS} | "
        f"Loss {train_loss:.4f}/{val_loss:.4f} | "
        f"Acc {train_acc:.4f}/{val_acc:.4f} | "
        f"AUC {auc_s} | F1 {f1_s} | "
        f"LR {lr:.2e} | {elapsed:.0f}s"
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)
    history["val_f1"].append(val_f1)
    history["lr"].append(lr)

    primary = val_auc if val_auc is not None else val_acc
    best    = best_val_auc if val_auc is not None else best_val_acc

    if primary > best:
        best_val_auc = val_auc if val_auc is not None else best_val_auc
        best_val_acc = val_acc
        patience_ctr = 0
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"val_acc": val_acc, "val_auc": val_auc,
             "val_loss": val_loss, "val_f1": val_f1},
            Config.CHECKPOINT_DIR / "best_model.pth",
        )
    else:
        patience_ctr += 1

    save_checkpoint(model, optimizer, scheduler, epoch, history,
                    Config.CHECKPOINT_DIR / "latest_model.pth")

    if patience_ctr >= Config.EARLY_STOP_PATIENCE:
        print(f"\n⛔ Early stopping at epoch {epoch+1}")
        break

print("\n✅ Cell 8 complete — training finished")


# ===CELL 9=== Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 9."""
best_path = Config.CHECKPOINT_DIR / "best_model.pth"
if best_path.exists():
    print("Loading best model for final test evaluation…")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

test_loss, test_acc, test_auc, test_f1, _, _ = evaluate(
    model, test_loader, criterion, device,
)
print(f"\n{'='*55}")
print(f"  DF40 TEST RESULTS")
print(f"{'='*55}")
print(f"  Loss     : {test_loss:.4f}")
print(f"  Accuracy : {test_acc:.4f}  ({'✅ ≥90%' if test_acc >= 0.90 else '⚠️ <90%'})")
print(f"  AUC-ROC  : {test_auc:.4f}" if test_auc else "  AUC-ROC  : N/A")
print(f"  F1 Score : {test_f1:.4f}"  if test_f1  else "  F1 Score : N/A")
print(f"{'='*55}")

history.update({"test_acc": test_acc, "test_loss": test_loss,
                "test_auc": test_auc, "test_f1": test_f1})
with open(Config.LOG_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("History saved.")
print("\n✅ Cell 9 complete")


# ===CELL 10=== Plot training curves
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 10."""
from plot_history import plot
plot(history,
     save_path=Config.LOG_DIR / "training_curves.png",
     show=False)
print(f"✅ Curves saved → {Config.LOG_DIR / 'training_curves.png'}")


# ===CELL 11=== Download best_model.pth
# ─────────────────────────────────────────────────────────────────────────────
"""Paste this block into Cell 11."""
from google.colab import files

best_path = Config.CHECKPOINT_DIR / "best_model.pth"
if best_path.exists():
    print("⬇️  Downloading best_model.pth to your local machine…")
    files.download(str(best_path))
    print("\n✅ Download started — check your browser downloads folder")
    print("\nTo use locally:")
    print("  1. Copy best_model.pth → EfficientNet-B4/outputs/checkpoints/")
    print("  2. python evaluate.py")
    print("  3. python inference.py --source face.jpg --visualise")
else:
    print(f"❌ {best_path} not found — training may not have completed yet")
