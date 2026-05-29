"""
prepare_df40.py — Prepare the DF40 dataset for EfficientNet-B4 training.

What this script does
---------------------
1. Scans the raw DF40 directory for real and fake image folders.
2. DEDUPLICATES real images (the same source videos appear across many
   fake-method folders in DF40).
3. Samples up to --max-fake-per-method images from each fake method
   so every method contributes equally (balanced across 40 methods).
4. Undersamples the real set to match the total fake count (1:1 ratio).
5. Shuffles and splits 80 / 10 / 10 into train / val / test.
6. Links (symlink on Linux/Mac, file-copy on Windows) or copies each
   image into:
       <out_dir>/train/real/   <out_dir>/train/fake/
       <out_dir>/val/real/     <out_dir>/val/fake/
       <out_dir>/test/real/    <out_dir>/test/fake/

Resume support
--------------
Progress is saved to --state-file (default: prepare_state.json).
If the script is interrupted and re-run, it skips already-completed steps.

Windows compatibility
---------------------
On Windows, os.symlink() requires either Developer Mode or admin rights.
This script automatically falls back to shutil.copy2() when symlinks fail.
Use --copy-files to force copying regardless of OS.

Usage
-----
    python prepare_df40.py --df40-root /path/to/DF40 --out-dir datasets/df40_combined
    python prepare_df40.py --df40-root /path/to/DF40 --out-dir datasets/df40_combined \\
        --max-fake-per-method 5000
    python prepare_df40.py --df40-root /path/to/DF40 --out-dir datasets/df40_combined \\
        --copy-files       # Force copy instead of symlink (slower, uses disk space)
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ── Configuration defaults ────────────────────────────────────────────────────
try:
    from config import Config
    _DEFAULT_MAX = Config.MAX_FAKE_PER_METHOD  # respect config.py setting
    _DEFAULT_VAL  = Config.VAL_SPLIT
    _DEFAULT_TEST = Config.TEST_SPLIT
    _DEFAULT_SEED = Config.RANDOM_SEED
except ImportError:
    _DEFAULT_MAX  = 5000
    _DEFAULT_VAL  = 0.10
    _DEFAULT_TEST = 0.10
    _DEFAULT_SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── State helpers ─────────────────────────────────────────────────────────────

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            print(f"[RESUME] Loaded state from {state_file}")
            return state
        except (json.JSONDecodeError, IOError):
            print(f"[WARN] Could not parse {state_file} — starting fresh.")
    return {}


def save_state(state: dict, state_file: Path) -> None:
    tmp = state_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_file)


# ── File linking ──────────────────────────────────────────────────────────────

_USE_COPY: bool = False   # set by --copy-files or after first symlink failure


def _link(src: Path, dst: Path) -> None:
    """Create a symlink src→dst, falling back to copy on Windows / permission errors."""
    global _USE_COPY
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return  # already done
    if _USE_COPY:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        # Windows without Developer Mode or admin: symlink not permitted
        _USE_COPY = True
        shutil.copy2(src, dst)


# ── Image discovery ───────────────────────────────────────────────────────────

def find_images(folder: Path) -> list[Path]:
    """Recursively find all images under folder."""
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


# ── Main preparation logic ────────────────────────────────────────────────────

def prepare(
    df40_root: Path,
    out_dir: Path,
    max_fake_per_method: int | None,
    val_split: float,
    test_split: float,
    seed: int,
    state: dict,
    state_file: Path,
) -> None:
    random.seed(seed)

    # ── Step 1: Discover DF40 structure ──────────────────────────────────────
    # DF40 layout varies by version but usually looks like:
    #   DF40/
    #     real/          ← shared real images (may also live per-method)
    #     <method_name>/
    #       fake/
    #       real/        ← (optional, same images as top-level real)

    print(f"\n[1/5] Scanning DF40 root: {df40_root}")

    # Collect all fake methods (subdirectory names that contain a 'fake' sub-folder)
    method_dirs: dict[str, Path] = {}
    for d in sorted(df40_root.iterdir()):
        if d.is_dir():
            fake_sub = d / "fake"
            if fake_sub.is_dir():
                method_dirs[d.name] = fake_sub

    if not method_dirs:
        # Alternative flat layout: df40_root/fake/<method>/ or df40_root/<method>/
        for d in sorted(df40_root.iterdir()):
            if d.is_dir() and d.name not in ("real", "__pycache__"):
                imgs = find_images(d)
                if imgs:
                    method_dirs[d.name] = d

    if not method_dirs:
        print(f"[ERROR] No fake method directories found under {df40_root}")
        print("        Expected DF40 structure:  df40_root/<method_name>/fake/")
        sys.exit(1)

    print(f"         Found {len(method_dirs)} fake methods: {', '.join(list(method_dirs)[:5])}...")

    # ── Step 2: Collect & deduplicate real images ─────────────────────────────
    if "real_paths" not in state:
        print("\n[2/5] Collecting real images (deduplicating)...")

        real_candidates: set[str] = set()

        # Top-level real directory
        top_real = df40_root / "real"
        if top_real.is_dir():
            for p in find_images(top_real):
                real_candidates.add(str(p))

        # Per-method real directories
        for method_name, fake_dir in method_dirs.items():
            method_real = fake_dir.parent / "real"
            if method_real.is_dir():
                for p in find_images(method_real):
                    real_candidates.add(str(p))   # set deduplicates

        if not real_candidates:
            print("[ERROR] No real images found. Check your DF40 directory structure.")
            sys.exit(1)

        real_paths = sorted(real_candidates)
        random.shuffle(real_paths)
        state["real_paths"] = real_paths
        save_state(state, state_file)
        print(f"         Found {len(real_paths):,} unique real images.")
    else:
        real_paths = state["real_paths"]
        print(f"\n[2/5] (skipped) {len(real_paths):,} real images loaded from state.")

    # ── Step 3: Sample fake images per method ────────────────────────────────
    if "fake_paths" not in state:
        print("\n[3/5] Sampling fake images (balanced per method)...")

        n_methods = len(method_dirs)
        cap = max_fake_per_method  # may be None → use all

        all_fake: list[str] = []
        for method_name, fake_dir in method_dirs.items():
            imgs = [str(p) for p in find_images(fake_dir)]
            random.shuffle(imgs)
            if cap is not None:
                imgs = imgs[:cap]
            all_fake.extend(imgs)
            print(f"         {method_name:<30} {len(imgs):>6,} fakes")

        random.shuffle(all_fake)
        state["fake_paths"] = all_fake
        save_state(state, state_file)
        print(f"         Total fakes: {len(all_fake):,}")
    else:
        all_fake = state["fake_paths"]
        print(f"\n[3/5] (skipped) {len(all_fake):,} fake paths loaded from state.")

    # ── Step 4: Balance real vs fake (1:1) ───────────────────────────────────
    if "manifest" not in state:
        print("\n[4/5] Balancing real:fake 1:1 and splitting train/val/test...")

        n_fake = len(all_fake)
        n_real_available = len(real_paths)

        if n_real_available < n_fake:
            print(f"[WARN] Fewer real images ({n_real_available:,}) than fake ({n_fake:,}). "
                  "Using all real images.")
            real_selected = real_paths
            # Under-sample fake to match
            all_fake = all_fake[:n_real_available]
            n_fake = len(all_fake)
        else:
            real_selected = real_paths[:n_fake]

        print(f"         Balanced: {n_fake:,} real + {n_fake:,} fake = {2*n_fake:,} total")

        # Split each class independently to maintain balance in every split
        def split_list(lst, val_frac, test_frac):
            n = len(lst)
            n_test = int(n * test_frac)
            n_val  = int(n * val_frac)
            random.shuffle(lst)
            return lst[n_val + n_test:], lst[n_test:n_val + n_test], lst[:n_test]

        real_train, real_val, real_test = split_list(
            list(real_selected), val_split, test_split
        )
        fake_train, fake_val, fake_test = split_list(
            list(all_fake), val_split, test_split
        )

        manifest = {
            "train": {"real": real_train, "fake": fake_train},
            "val":   {"real": real_val,   "fake": fake_val},
            "test":  {"real": real_test,  "fake": fake_test},
        }
        state["manifest"] = manifest
        save_state(state, state_file)

        total_train = len(real_train) + len(fake_train)
        total_val   = len(real_val)   + len(fake_val)
        total_test  = len(real_test)  + len(fake_test)
        print(f"         train={total_train:,}  val={total_val:,}  test={total_test:,}")
    else:
        manifest = state["manifest"]
        print(f"\n[4/5] (skipped) Manifest loaded from state.")
        total_train = len(manifest["train"]["real"]) + len(manifest["train"]["fake"])
        total_val   = len(manifest["val"]["real"])   + len(manifest["val"]["fake"])
        total_test  = len(manifest["test"]["real"])  + len(manifest["test"]["fake"])
        print(f"         train={total_train:,}  val={total_val:,}  test={total_test:,}")

    # ── Step 5: Link / copy files into output structure ───────────────────────
    print(f"\n[5/5] Linking images into {out_dir} ...")
    print("       (Windows: auto-falls-back to copy if symlinks not permitted)")

    linked_count = 0
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            paths = manifest[split][label]
            dest_dir = out_dir / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            key = f"done_{split}_{label}"
            if state.get(key):
                n_existing = sum(1 for _ in dest_dir.iterdir()
                                 if _.suffix.lower() in IMG_EXTS)
                print(f"   [SKIP] {split}/{label}: {n_existing:,} files already in place")
                continue

            print(f"   Linking {split}/{label}: {len(paths):,} files...", end="", flush=True)
            for i, src_str in enumerate(paths):
                src = Path(src_str)
                if not src.exists():
                    continue
                dst = dest_dir / f"{i:07d}{src.suffix.lower()}"
                _link(src, dst)
                linked_count += 1

            state[key] = True
            save_state(state, state_file)
            mode = "copied" if _USE_COPY else "linked"
            print(f" done ({len(paths):,} {mode})")

    link_mode = "Copied" if _USE_COPY else "Symlinked"
    print(f"\n[DONE] {link_mode} {linked_count:,} new files.")
    print(f"       Output → {out_dir}")
    print(f"       State  → {state_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DF40 dataset for EfficientNet-B4 deepfake training."
    )
    parser.add_argument(
        "--df40-root", required=True, type=Path,
        help="Path to the extracted DF40 root directory."
    )
    parser.add_argument(
        "--out-dir", default=None, type=Path,
        help="Output directory for prepared splits. "
             "Default: EfficientNet-B4/datasets/df40_combined"
    )
    parser.add_argument(
        "--max-fake-per-method", default=str(_DEFAULT_MAX), type=str,
        help=f"Max fake images per method (default: {_DEFAULT_MAX}). "
             "Use 'none' for unlimited."
    )
    parser.add_argument(
        "--val-split", default=_DEFAULT_VAL, type=float,
        help=f"Fraction for validation (default: {_DEFAULT_VAL})"
    )
    parser.add_argument(
        "--test-split", default=_DEFAULT_TEST, type=float,
        help=f"Fraction for test (default: {_DEFAULT_TEST})"
    )
    parser.add_argument(
        "--seed", default=_DEFAULT_SEED, type=int,
        help=f"Random seed (default: {_DEFAULT_SEED})"
    )
    parser.add_argument(
        "--state-file", default=None, type=Path,
        help="Path to the JSON resume state file. "
             "Default: <out_dir>/../prepare_state.json"
    )
    parser.add_argument(
        "--copy-files", action="store_true",
        help="Force file copy instead of symlinks (slower, uses more disk)."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Ignore existing state file and start fresh."
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    out_dir    = args.out_dir or (script_dir / "datasets" / "df40_combined")
    state_file = args.state_file or (script_dir / "prepare_state.json")

    # Parse max_fake_per_method (allows "none" / "null" / "" for unlimited)
    raw_max = str(args.max_fake_per_method).strip().lower()
    if raw_max in ("none", "null", ""):
        max_fake = None
    else:
        try:
            max_fake = int(raw_max)
        except ValueError:
            print(f"[ERROR] Invalid --max-fake-per-method value: {args.max_fake_per_method}")
            sys.exit(1)

    # Force-copy mode
    global _USE_COPY
    if args.copy_files:
        _USE_COPY = True

    # Load or reset state
    state = {} if args.reset else load_state(state_file)

    prepare(
        df40_root=Path(args.df40_root),
        out_dir=out_dir,
        max_fake_per_method=max_fake,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        state=state,
        state_file=state_file,
    )


if __name__ == "__main__":
    main()