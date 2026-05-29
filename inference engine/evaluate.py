"""
evaluate.py — Confusion matrix, ROC, Precision-Recall, and score
distribution plots for the EfficientNet-B4 + DF40 model.

Usage
-----
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python evaluate.py --split val
    python evaluate.py --threshold 0.45
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch
import torch.amp
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from tqdm import tqdm

from config import Config
from dataset import get_data_loaders
from model import DeepfakeDetector


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":     "#0f172a", "card":  "#1e293b",
    "accent": "#6366f1", "real":  "#22c55e",
    "fake":   "#ef4444", "text":  "#f1f5f9",
    "muted":  "#94a3b8",
}


def _dark(ax, title=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=12,
                     fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["bg"])


def load_model(ckpt_path: str, device: str) -> DeepfakeDetector:
    model = DeepfakeDetector(num_classes=Config.NUM_CLASSES,
                             pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    metrics = ckpt.get("metrics", {})
    print(f"✅ Loaded: {ckpt_path}")
    if metrics:
        va = metrics.get("val_acc"); va = max(va) if isinstance(va, list) and va else va
        au = metrics.get("val_auc"); au = (max(v for v in au if v is not None) if isinstance(au, list) and au else au)
        print(f"   Val Acc  : {va:.4f}" if va is not None else "   Val Acc  : N/A")
        print(f"   Val AUC  : {au:.4f}" if au is not None else "   Val AUC  : N/A")
    return model


@torch.no_grad()
def collect_predictions(model, loader, device, threshold=0.5):
    all_probs, all_labels = [], []
    use_amp = device == "cuda"
    for images, labels, _ in tqdm(loader, desc="Running inference"):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _, _ = model(images)
        probs       = torch.sigmoid(logits).cpu().numpy()
        hard_labels = (labels.numpy() > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_labels.extend(hard_labels.tolist())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= threshold).astype(int)
    return all_probs, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cell_c  = [["#166534", "#7f1d1d"], ["#7f1d1d", "#14532d"]]
    text_c  = [["#bbf7d0", "#fecaca"], ["#fecaca", "#bbf7d0"]]
    for i in range(2):
        for j in range(2):
            y_base = 1 - i
            ax.add_patch(plt.Rectangle([j, y_base], 1, 1,
                         facecolor=cell_c[i][j],
                         edgecolor=PALETTE["bg"], linewidth=3))
            ax.text(j + 0.5, y_base + 0.5,
                    f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]:,})",
                    ha="center", va="center",
                    color=text_c[i][j], fontsize=15, fontweight="bold")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted REAL", "Predicted FAKE"],
                       color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.set_yticklabels(["Actual FAKE", "Actual REAL"],
                       color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Confusion Matrix", color=PALETTE["text"],
                 fontsize=14, fontweight="bold", pad=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#166534", edgecolor="none", label="Correct"),
        Patch(facecolor="#7f1d1d", edgecolor="none", label="Wrong"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2,
       facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
       framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"   💾 Confusion matrix → {save_path}")


def plot_roc_pr(probs, labels, auc_roc, ap, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fpr, tpr, _ = roc_curve(labels, probs)
    ax1.plot(fpr, tpr, color=PALETTE["accent"], lw=2,
             label=f"AUC = {auc_roc:.4f}")
    ax1.plot([0,1],[0,1], color=PALETTE["muted"], lw=1, linestyle="--")
    ax1.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent"])
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    _dark(ax1, "ROC Curve")
    prec, rec, _ = precision_recall_curve(labels, probs)
    ax2.plot(rec, prec, color=PALETTE["fake"], lw=2, label=f"AP = {ap:.4f}")
    ax2.fill_between(rec, prec, alpha=0.15, color=PALETTE["fake"])
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    _dark(ax2, "Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"   💾 ROC & PR → {save_path}")


def plot_prob_distribution(probs, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    bins = np.linspace(0, 1, 40)
    ax.hist(probs[labels==0], bins=bins, color=PALETTE["real"], alpha=0.75,
            label=f"REAL (n={int((labels==0).sum()):,})", density=True)
    ax.hist(probs[labels==1], bins=bins, color=PALETTE["fake"], alpha=0.75,
            label=f"FAKE (n={int((labels==1).sum()):,})", density=True)
    ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", lw=1.5,
               label="Threshold (0.5)")
    ax.set_xlabel("Predicted FAKE probability"); ax.set_ylabel("Density")
    ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    _dark(ax, "Score Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"   💾 Score dist → {save_path}")


def plot_training_history(history_path, save_path):
    with open(history_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig    = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs     = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    plots  = [
        ("Loss",     "train_loss", "val_loss", PALETTE["accent"], PALETTE["fake"]),
        ("Accuracy", "train_acc",  "val_acc",  PALETTE["accent"], PALETTE["fake"]),
        ("AUC-ROC",  None,         "val_auc",  None, PALETTE["real"]),
        ("F1 Score", None,         "val_f1",   None, "#f59e0b"),
    ]
    for idx, (title, tk, vk, tc, vc) in enumerate(plots):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor(PALETTE["card"])
        if tk and tk in h:
            ax.plot(epochs, h[tk], color=tc, lw=2, label="Train")
        if vk and vk in h:
            ep_v = [e for e, v in zip(epochs, h[vk]) if v is not None]
            va_v = [v for v in h[vk] if v is not None]
            ax.plot(ep_v, va_v, color=vc, lw=2, linestyle="--", label="Val")
        ax.set_xlabel("Epoch")
        ax.legend(facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
                  fontsize=9)
        _dark(ax, title)
    plt.suptitle("Training History — EfficientNet-B4 on DF40",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()
    print(f"   💾 Training history → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet-B4 Deepfake Detector")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir",   default=None)
    args = parser.parse_args()

    cfg    = Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    ckpt_path = args.checkpoint or str(cfg.CHECKPOINT_DIR / "best_model.pth")
    out_dir   = Path(args.out_dir) if args.out_dir else cfg.OUTPUT_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device)
    print(f"\nLoading {args.split} split from: {cfg.DATA_DIR}")
    _, val_loader, test_loader, _ = get_data_loaders(cfg)
    loader = test_loader if args.split == "test" else val_loader

    print(f"\nThreshold : {args.threshold}")
    probs, preds, labels = collect_predictions(model, loader, device,
                                               args.threshold)

    cm          = confusion_matrix(labels, preds)
    report      = classification_report(labels, preds,
                                        target_names=["REAL", "FAKE"], digits=4)
    auc         = roc_auc_score(labels, probs)
    ap          = average_precision_score(labels, probs)
    f1          = f1_score(labels, preds, zero_division=0)
    acc         = (preds == labels).mean()
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  EVALUATION — {args.split.upper()} split")
    print(f"{'='*55}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  F1 (FAKE)    : {f1:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (FAKE recall)")
    print(f"  Specificity  : {specificity:.4f}  (REAL recall)")
    print(f"\n{report}")

    print("Saving plots…")
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    plot_roc_pr(probs, labels, auc, ap, out_dir / "roc_pr_curves.png")
    plot_prob_distribution(probs, labels, out_dir / "prob_distribution.png")
    hist_path = cfg.LOG_DIR / "training_history.json"
    if hist_path.exists():
        plot_training_history(str(hist_path), str(out_dir / "training_history.png"))

    summary = {
        "split": args.split, "threshold": args.threshold,
        "accuracy": float(acc), "auc_roc": float(auc),
        "avg_precision": float(ap), "f1_fake": float(f1),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ All outputs → {out_dir}")


if __name__ == "__main__":
    main()
