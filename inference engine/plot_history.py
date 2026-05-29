"""
Plot training history saved by train.py.

Usage
-----
    python plot_history.py
    python plot_history.py --log outputs/logs/training_history.json
    python plot_history.py --save outputs/logs/training_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor"  : "#1a1d27",
    "axes.edgecolor"  : "#3a3d4d", "axes.labelcolor" : "#c8ccd4",
    "xtick.color"     : "#c8ccd4", "ytick.color"     : "#c8ccd4",
    "text.color"      : "#c8ccd4", "grid.color"      : "#2a2d3d",
    "grid.linestyle"  : "--",      "grid.linewidth"  : 0.6,
    "font.family"     : "sans-serif", "font.size"    : 10,
    "legend.framealpha": 0.2,      "legend.edgecolor": "#3a3d4d",
})

PALETTE = {"train": "#4fc3f7", "val": "#ff8a65"}


def plot(history: dict, save_path: Path | None = None, show: bool = False):
    epochs   = list(range(1, len(history["train_loss"]) + 1))
    has_auc  = any(v is not None for v in history.get("val_auc", []))
    has_f1   = any(v is not None for v in history.get("val_f1",  []))
    has_lr   = "lr" in history and len(history["lr"]) > 0
    n_panels = 2 + int(has_auc) + int(has_f1) + int(has_lr)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    fig.suptitle("Training History — EfficientNet-B4 on DF40",
                 fontsize=12, fontweight="bold", color="#e8eaf6", y=1.02)
    ax_iter = iter(axes if n_panels > 1 else [axes])

    def _next():
        ax = next(ax_iter); ax.grid(True, alpha=0.4); return ax

    ax = _next()
    ax.plot(epochs, history["train_loss"], color=PALETTE["train"],
            lw=1.8, marker="o", ms=3, label="Train")
    ax.plot(epochs, history["val_loss"],   color=PALETTE["val"],
            lw=1.8, marker="s", ms=3, label="Val")
    ax.set_title("Loss (BCE)"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss"); ax.legend()

    ax = _next()
    ax.plot(epochs, [v*100 for v in history["train_acc"]],
            color=PALETTE["train"], lw=1.8, marker="o", ms=3, label="Train")
    ax.plot(epochs, [v*100 for v in history["val_acc"]],
            color=PALETTE["val"],   lw=1.8, marker="s", ms=3, label="Val")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend()

    if has_auc:
        ax = _next()
        vals = [v if v is not None else float("nan") for v in history["val_auc"]]
        ax.plot(epochs, vals, color="#ce93d8", lw=1.8, marker="D", ms=3,
                label="Val AUC")
        ax.set_title("AUC-ROC"); ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.9, color="#66bb6a", lw=0.8, ls="--", alpha=0.6,
                   label="0.90 target")
        ax.legend()

    if has_f1:
        ax = _next()
        vals = [v if v is not None else float("nan") for v in history["val_f1"]]
        ax.plot(epochs, vals, color="#80cbc4", lw=1.8, marker="^", ms=3,
                label="Val F1")
        ax.set_title("F1 Score"); ax.set_xlabel("Epoch")
        ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.legend()

    if has_lr:
        ax = _next()
        ax.plot(epochs, history["lr"], color="#ffcc80", lw=1.8)
        ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
        ax.set_ylabel("LR"); ax.set_yscale("log")

    lines = []
    if "test_acc" in history:
        lines.append("Test Acc : %.2f%%" % (history["test_acc"] * 100))
    if history.get("test_auc") is not None:
        lines.append("Test AUC : %.4f" % history["test_auc"])
    if history.get("test_f1") is not None:
        lines.append("Test F1  : %.4f" % history["test_f1"])
    if lines:
        fig.text(0.5, -0.04, "  |  ".join(lines),
                 ha="center", fontsize=11, color="#a5d6a7", fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print("Saved: " + str(save_path))
    if show:
        matplotlib.use("TkAgg"); plt.show()
    plt.close(fig)


def main():
    from config import Config
    cfg = Config
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--log",  default=str(cfg.LOG_DIR / "training_history.json"))
    parser.add_argument("--save", default=str(cfg.LOG_DIR / "training_curves.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print("Not found:", log_path); return
    with open(log_path) as f:
        history = json.load(f)
    print(f"Loaded {len(history.get('train_loss', []))} epochs from {log_path}")
    plot(history, save_path=Path(args.save), show=args.show)


if __name__ == "__main__":
    main()
