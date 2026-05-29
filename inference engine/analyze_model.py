"""Comprehensive threshold analysis and edge-case report for the trained model."""
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from config import Config
from dataset import get_data_loaders
from model import DeepfakeDetector
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config

    # Load model
    model = DeepfakeDetector(num_classes=1, pretrained=False).to(device)
    print(f"Evaluating checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    raw_sd = ckpt["model_state_dict"]
    fixed_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    try:
        model.load_state_dict(fixed_sd)
    except RuntimeError:
        model.load_state_dict(raw_sd)
    model.eval()

    # Load test data
    _, _, test_loader, _ = get_data_loaders(cfg)

    # Collect all predictions
    all_probs = []
    all_labels = []
    all_paths = []
    with torch.inference_mode():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits, _, _ = model(images)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend((labels > 0.5).float().cpu().tolist())
            all_paths.extend(paths)

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print("=" * 75)
    print("  COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 75)
    print(f"  AUC-ROC: {roc_auc_score(labels_np, probs_np):.4f}")
    print(f"  Optimal threshold (Youden J): {optimal_threshold:.4f}")
    print()

    # Test multiple thresholds
    header = f"{'Thresh':>7} | {'Accuracy':>8} | {'Sensitivity':>11} | {'Specificity':>11} | {'F1-Fake':>7} | {'F1-Real':>7}"
    print(header)
    print("-" * 75)
    for t in [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, optimal_threshold]:
        preds = (probs_np >= t).astype(float)
        acc = accuracy_score(labels_np, preds)
        tp = ((preds == 1) & (labels_np == 1)).sum()
        fn = ((preds == 0) & (labels_np == 1)).sum()
        fp = ((preds == 1) & (labels_np == 0)).sum()
        tn = ((preds == 0) & (labels_np == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_f = f1_score(labels_np, preds, zero_division=0)
        f1_r = f1_score(1 - labels_np, 1 - preds, zero_division=0)
        marker = " <-- OPTIMAL" if abs(t - optimal_threshold) < 0.001 else ""
        print(f"  {t:.4f} | {acc:8.4f} | {sens:11.4f} | {spec:11.4f} | {f1_f:7.4f} | {f1_r:7.4f}{marker}")

    # Distribution analysis
    print()
    print("=" * 75)
    print("  PROBABILITY DISTRIBUTION")
    print("=" * 75)
    real_probs = probs_np[labels_np == 0]
    fake_probs = probs_np[labels_np == 1]
    print(f"  REAL images  - mean: {real_probs.mean():.4f}, median: {np.median(real_probs):.4f}, std: {real_probs.std():.4f}")
    print(f"  FAKE images  - mean: {fake_probs.mean():.4f}, median: {np.median(fake_probs):.4f}, std: {fake_probs.std():.4f}")
    print(f"  REAL in [0.3, 0.7] (uncertain): {((real_probs >= 0.3) & (real_probs <= 0.7)).sum()} / {len(real_probs)}")
    print(f"  FAKE in [0.3, 0.7] (uncertain): {((fake_probs >= 0.3) & (fake_probs <= 0.7)).sum()} / {len(fake_probs)}")

    # Confidence breakdown
    print()
    print("=" * 75)
    print("  CONFIDENCE BREAKDOWN")
    print("=" * 75)
    for label_name, lbl_probs in [("REAL", real_probs), ("FAKE", fake_probs)]:
        if label_name == "REAL":
            high_conf = (lbl_probs < 0.2).sum()
            wrong_conf = (lbl_probs > 0.8).sum()
        else:
            high_conf = (lbl_probs > 0.8).sum()
            wrong_conf = (lbl_probs < 0.2).sum()
        uncertain = ((lbl_probs >= 0.35) & (lbl_probs <= 0.65)).sum()
        total = len(lbl_probs)
        print(f"  {label_name}: High-conf CORRECT: {high_conf}/{total} ({100*high_conf/total:.1f}%)")
        print(f"  {label_name}: Uncertain [0.35-0.65]: {uncertain}/{total} ({100*uncertain/total:.1f}%)")
        print(f"  {label_name}: High-conf WRONG:   {wrong_conf}/{total} ({100*wrong_conf/total:.1f}%)")
        print()

    # Worst misclassifications at threshold=0.5
    print("=" * 75)
    print("  WORST MISCLASSIFICATIONS (threshold=0.5)")
    print("=" * 75)

    # False Negatives (FAKE labeled as REAL) — most dangerous
    fn_mask = (labels_np == 1) & (probs_np < 0.5)
    fn_probs = probs_np[fn_mask]
    fn_paths = [all_paths[i] for i in range(len(all_paths)) if fn_mask[i]]
    if len(fn_probs) > 0:
        sorted_idx = np.argsort(fn_probs)
        print(f"\n  False Negatives (FAKE missed as REAL): {fn_mask.sum()} / {(labels_np==1).sum()}")
        for i in sorted_idx[:10]:
            print(f"    prob={fn_probs[i]:.4f}  {os.path.basename(fn_paths[i])}")

    # False Positives (REAL labeled as FAKE)
    fp_mask = (labels_np == 0) & (probs_np >= 0.5)
    fp_probs = probs_np[fp_mask]
    fp_paths = [all_paths[i] for i in range(len(all_paths)) if fp_mask[i]]
    if len(fp_probs) > 0:
        sorted_idx = np.argsort(-fp_probs)
        print(f"\n  False Positives (REAL flagged as FAKE): {fp_mask.sum()} / {(labels_np==0).sum()}")
        for i in sorted_idx[:10]:
            print(f"    prob={fp_probs[i]:.4f}  {os.path.basename(fp_paths[i])}")

    print()
    print("=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    best_acc_t = None
    best_acc = 0
    for t in np.arange(0.25, 0.60, 0.01):
        preds = (probs_np >= t).astype(float)
        acc = accuracy_score(labels_np, preds)
        if acc > best_acc:
            best_acc = acc
            best_acc_t = t
    print(f"  Best accuracy achievable: {best_acc:.4f} at threshold {best_acc_t:.2f}")
    print(f"  Youden optimal threshold: {optimal_threshold:.4f}")
    print(f"  Total test samples:       {len(labels_np)}")
    print(f"  Total REAL:               {(labels_np == 0).sum()}")
    print(f"  Total FAKE:               {(labels_np == 1).sum()}")


if __name__ == "__main__":
    main()
