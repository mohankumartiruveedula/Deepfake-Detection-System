"""
Training script - EfficientNet-B4 Deepfake Detector on DF40.

Training strategy
-----------------
  Phase 1  (epochs 0 -> UNFREEZE_EPOCH-1)
    Backbone is FROZEN.  Only the classification head trains.
    This prevents the pretrained ImageNet features from being corrupted
    before the head has learned a useful gradient direction.

  Phase 2  (epoch UNFREEZE_EPOCH -> end)
    Backbone is UNFROZEN with layer-wise learning rates:
      - Head params     -> LEARNING_RATE     (3e-5)
      - Backbone params -> LEARNING_RATE x BACKBONE_LR_MULT  (3e-6)
    Lower backbone LR preserves pretrained representations while allowing
    fine-tuning toward forgery-specific features in DF40.

  MixUp augmentation (on-GPU)
    Blends pairs of training images with a Beta(alpha, alpha) coefficient.
    Prevents the model from memorising specific method artefacts and
    creates smoother decision boundaries.

Usage
-----
    python train.py
    python train.py --resume outputs/checkpoints/latest_model.pth
    python train.py --data-dir datasets/df40_combined
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output so emoji/unicode in print() never crash on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from config import Config
from dataset import get_data_loaders
from model import DeepfakeDetector

try:
    from sklearn.metrics import roc_auc_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# MixUp
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(images: torch.Tensor, labels: torch.Tensor,
                alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp to a training batch.
    x_mix = lam*xi + (1-lam)*xj,   y_mix = lam*yi + (1-lam)*yj
    """
    if alpha <= 0.0:
        return images, labels
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return (lam * images + (1 - lam) * images[idx],
            lam * labels + (1 - lam) * labels[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device,
                grad_clip=5.0, epoch=1, total_epochs=1,
                scaler=None, mixup_alpha=0.2):
    model.train()
    # Accumulate on GPU tensors — avoids per-batch CPU-GPU sync from .item()
    running_loss    = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device, dtype=torch.long)
    total_n = 0

    pbar = tqdm(loader,
                desc=f"Epoch {epoch}/{total_epochs} [Train]",
                leave=True,
                bar_format="{l_bar}{bar:20}{r_bar} | {percentage:3.0f}%")
    for images, labels, _ in pbar:
        # non_blocking=True overlaps data transfer with previous GPU computation
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, labels = mixup_batch(images, labels, alpha=mixup_alpha)

        # set_to_none=True avoids costly memset-to-zero; just deallocates grads
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, _, _ = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = images.size(0)
        # Accumulate on GPU — no .item() per batch
        running_loss += loss.detach() * bs
        preds        = (logits.detach() > 0.0)
        hard_labels  = (labels > 0.5)
        running_correct += (preds == hard_labels).sum()
        total_n += bs

    # Single CPU-GPU sync at epoch end
    epoch_loss = running_loss.item() / total_n
    epoch_acc  = running_correct.item() / total_n
    return epoch_loss, epoch_acc


@torch.inference_mode()   # faster than @torch.no_grad — disables view tracking too
def evaluate(model, loader, criterion, device):
    model.eval()
    # Accumulate on GPU tensors — single sync at end
    running_loss    = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device, dtype=torch.long)
    total_n = 0
    all_probs_gpu  = []   # collect on GPU, convert once at end
    all_labels_gpu = []

    pbar = tqdm(loader, desc="Evaluating", leave=True,
                bar_format="{l_bar}{bar:20}{r_bar} | {percentage:3.0f}%")
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits, _, _ = model(images)
            loss = criterion(logits, labels)

        bs = images.size(0)
        running_loss += loss * bs
        probs        = torch.sigmoid(logits)
        preds        = (logits > 0.0)
        hard_labels  = (labels > 0.5)
        running_correct += (preds == hard_labels).sum()
        total_n += bs
        all_probs_gpu.append(probs)
        all_labels_gpu.append(hard_labels.float())

    # Single CPU-GPU sync at epoch end
    avg_loss = running_loss.item() / total_n
    accuracy = running_correct.item() / total_n

    # Concat on GPU first, then transfer to CPU once
    all_probs  = torch.cat(all_probs_gpu).cpu().tolist()
    all_labels = torch.cat(all_labels_gpu).cpu().tolist()

    auc = f1 = None
    if HAS_SKLEARN and len(set(all_labels)) == 2:
        try:
            auc      = roc_auc_score(all_labels, all_probs)
            bin_pred = [1 if p > 0.5 else 0 for p in all_probs]
            f1       = f1_score(all_labels, bin_pred, zero_division=0)
        except Exception:
            pass
    return avg_loss, accuracy, auc, f1, all_probs, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics":              metrics,
    }, path)
    print(f"  [SAVED] Checkpoint -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B4 on DF40"
    )
    parser.add_argument("--resume",   default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--data-dir", default=None,
                        help="Override Config.DATA_DIR")
    args = parser.parse_args()

    cfg = Config
    if args.data_dir:
        cfg.DATA_DIR = Path(args.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        # TF32 gives up to 2x speedup on RTX 30/40 series Tensor Cores
        torch.set_float32_matmul_precision("high")
    else:
        print("\n[WARNING] CUDA not available -- training on CPU will be very slow.")

    torch.manual_seed(cfg.RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nLoading DF40 data from: {cfg.DATA_DIR}")
    train_loader, val_loader, test_loader, pos_weight = get_data_loaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading pretrained EfficientNet-B4 (ImageNet weights)...")
    model = DeepfakeDetector(num_classes=cfg.NUM_CLASSES, pretrained=True, use_srm=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params/1e6:.1f}M")

    model.freeze_backbone()
    print(f"Backbone FROZEN for first {cfg.UNFREEZE_EPOCH} epochs (head warm-up)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )

    # ── Layer-wise optimizer ───────────────────────────────────────────────────
    backbone_lr = cfg.LEARNING_RATE * getattr(cfg, "BACKBONE_LR_MULT", 0.1)
    optimizer   = AdamW([
        {"params": model.get_head_params(),     "lr": cfg.LEARNING_RATE},
        {"params": model.get_backbone_params(), "lr": backbone_lr},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                            eta_min=1e-7)

    # ── State ─────────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    patience_ctr = 0
    history      = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_auc":    [], "val_f1":    [],
        "lr":         [],
    }

    # ── Resume (MUST happen BEFORE torch.compile) ──────────────────────────────
    # torch.compile wraps the model and prefixes all state_dict keys with
    # "_orig_mod.". Loading a plain checkpoint into a compiled model fails with
    # a key mismatch. Always load weights first, then compile.
    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        metrics      = ckpt["metrics"]
        if isinstance(metrics.get("val_acc"), list):
            history.update(metrics)
            best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
            valid_aucs   = [v for v in history.get("val_auc", []) if v is not None]
            best_val_auc = max(valid_aucs) if valid_aucs else 0.0
        else:
            best_val_acc = metrics.get("val_acc", 0.0)
            best_val_auc = metrics.get("val_auc", 0.0) or 0.0
        print(f"Resumed from epoch {start_epoch} "
              f"(best_acc={best_val_acc:.4f}, best_auc={best_val_auc:.4f})")

    # NOTE: torch.compile requires Triton which is Linux-only.
    # Speedup on Windows comes from TF32, AMP (GradScaler), cudnn.benchmark,
    # and prefetch_factor in the DataLoader — all already active above.

    # ── Training loop ─────────────────────────────────────────────────────────
    mixup_alpha = getattr(cfg, "MIXUP_ALPHA", 0.2)
    unfreeze_ep = getattr(cfg, "UNFREEZE_EPOCH", 3)
    scaler      = torch.amp.GradScaler("cuda") if device == "cuda" else None

    print(f"\nTraining up to {cfg.NUM_EPOCHS} epochs "
          f"(patience={cfg.EARLY_STOP_PATIENCE})\n")

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        t0 = time.time()

        if epoch == unfreeze_ep:
            model.unfreeze_backbone()
            print(f"\n[UNFROZEN] Backbone UNFROZEN -- epoch {epoch+1} "
                  f"(backbone LR={backbone_lr:.2e})\n")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch+1, total_epochs=cfg.NUM_EPOCHS,
            scaler=scaler, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc, val_auc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device,
        )

        # Removed torch.cuda.empty_cache() — it forces a full CUDA sync
        # and wastes 200-400ms per epoch for no benefit. PyTorch's caching
        # allocator handles memory reuse efficiently on its own.
        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        auc_s = f"{val_auc:.4f}" if val_auc is not None else "  N/A "
        f1_s  = f"{val_f1:.4f}"  if val_f1  is not None else "  N/A "
        print(
            f"Ep {epoch+1:3d}/{cfg.NUM_EPOCHS} | "
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
                cfg.CHECKPOINT_DIR / "best_model.pth",
            )
        else:
            patience_ctr += 1

        save_checkpoint(model, optimizer, scheduler, epoch, history,
                        cfg.CHECKPOINT_DIR / "latest_model.pth")

        if patience_ctr >= cfg.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"({patience_ctr} epochs without improvement)")
            break

    # ── Final test ────────────────────────────────────────────────────────────
    best_path = cfg.CHECKPOINT_DIR / "best_model.pth"
    if best_path.exists():
        print("\nLoading best model for final test evaluation...")
        best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
        # Strip _orig_mod. prefix in case checkpoint was saved from compiled model
        raw_sd   = best_ckpt["model_state_dict"]
        fixed_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
        eval_model = DeepfakeDetector(num_classes=cfg.NUM_CLASSES,
                                      pretrained=False).to(device)
        try:
            eval_model.load_state_dict(fixed_sd)
        except RuntimeError:
            eval_model.load_state_dict(raw_sd)
        model = eval_model

    test_loss, test_acc, test_auc, test_f1, _, _ = evaluate(
        model, test_loader, criterion, device,
    )
    print(f"\n{'='*55}")
    print(f"  TEST RESULTS (DF40)")
    print(f"{'='*55}")
    print(f"  Loss     : {test_loss:.4f}")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  AUC-ROC  : {test_auc}")
    print(f"  F1 Score : {test_f1}")
    print(f"{'='*55}")

    history.update({"test_acc": test_acc, "test_loss": test_loss,
                    "test_auc": test_auc, "test_f1": test_f1})
    log_path = cfg.LOG_DIR / "training_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {log_path}")


if __name__ == "__main__":
    main()
