"""
Prepare the Kaggle "Fruits Fresh and Rotten" dataset for multi-stage
shelf-life regression.

WHAT THIS DOES
--------------
The Kaggle archive (sriramr/fruits-fresh-and-rotten-for-classification)
provides only TWO labels per fruit: fresh and rotten. Our pipeline expects
multi-stage labels. This script derives 4 ordered stages per fruit by:

  1. Treating the original "fresh" and "rotten" labels as ANCHOR classes
     (ground truth at the two extremes).
  2. Computing a per-image VISUAL DECAY SCORE -- the mean V (brightness)
     channel of the foreground region. Decay manifests as darker spots,
     mould, dehydration, and bruising, all of which lower the mean V.
     The foreground is approximated by masking out very bright (V>230)
     pixels which are usually background.
  3. Within each anchor, sorting images by decay score and splitting at
     the median into two sub-stages.

The resulting four stages are ORDINAL (stage 1 = lowest decay,
stage 4 = highest decay), with anchors at both ends grounded in the
original Kaggle labels.

CAVEAT (please disclose in the write-up)
----------------------------------------
The within-anchor sub-staging is a HEURISTIC ordinal extension, not
validated time-to-spoilage data. Day-counts in `config.FRUITS` are
interpolated between max_days and 0 using a non-linear schedule.
Future work would validate against time-stamped ripeness photographs.

USAGE
-----
    python scripts/prepare_dataset.py /path/to/kaggle/extracted

By default this creates symlinks (fast, no extra disk). Pass --copy to
copy files instead.

    python scripts/prepare_dataset.py /path/to/kaggle --copy

After this script finishes, you can run `scripts/build_splits.py`,
then the training scripts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/prepare_dataset.py ...` to find project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_ROOT, FRUITS

# Map Kaggle class folder name -> (fruit, anchor)
# anchor = "fresh" or "rotten"
KAGGLE_TO_ANCHOR: dict[str, tuple[str, str]] = {
    "freshapples":   ("apple",  "fresh"),
    "rottenapples":  ("apple",  "rotten"),
    "freshbanana":   ("banana", "fresh"),
    "rottenbanana":  ("banana", "rotten"),
    "freshoranges":  ("orange", "fresh"),
    "rottenoranges": ("orange", "rotten"),
}
NORMALISED = {k.lower(): v for k, v in KAGGLE_TO_ANCHOR.items()}

# Within each anchor, sub-stage 0 (lower-decay half) gets these stage IDs;
# sub-stage 1 (higher-decay half) gets the next ID.
#   fresh  -> stages 1, 2
#   rotten -> stages 3, 4
ANCHOR_STAGE_OFFSET = {"fresh": 1, "rotten": 3}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Decay score

def decay_score(img_path: Path) -> float:
    """Compute a visual decay score for one image.

    Higher score = more decay (darker fruit, more spots/mould/bruising).

    Method: take the central 70% crop, convert to HSV, mask out
    likely-background pixels (V > 230), and return 1 - mean(V_foreground).
    Returned values lie in [0, 1] approximately (V is normalised to [0,1]).
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return float("nan")

    h, w = img.shape[:2]
    # Center crop 70% to drop most of the background border
    ch, cw = int(h * 0.15), int(w * 0.15)
    img = img[ch : h - ch, cw : w - cw]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    # Foreground = not extremely bright (white background) AND has some saturation,
    # i.e., we keep colourful pixels and dark pixels, and drop pure-white background.
    fg_mask = (v < 0.90) | (s > 0.15)
    fg_v = v[fg_mask]
    if fg_v.size < 50:
        # Failed mask -- fall back to whole-image mean.
        fg_v = v.flatten()

    # Decay = 1 - mean foreground brightness.
    return float(1.0 - fg_v.mean())


# Discovery

def find_class_dirs(root: Path) -> list[tuple[Path, str, str, str]]:
    """Return (source_dir, fruit, anchor, origin) tuples.

    `origin` is the parent folder name (typically "train" or "test"), used
    to disambiguate filenames when we copy/symlink.
    """
    found: list[tuple[Path, str, str, str]] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        key = d.name.lower()
        if key not in NORMALISED:
            continue
        fruit, anchor = NORMALISED[key]
        if fruit not in FRUITS:
            print(f"  warn: skipping {d} -- {fruit} not in config.FRUITS")
            continue
        origin = d.parent.name  # train / test
        found.append((d, fruit, anchor, origin))
    return found


# Main pipeline

def prepare(src_root: Path, dst_root: Path, copy: bool) -> None:
    if not src_root.exists():
        sys.exit(f"Source path does not exist: {src_root}")

    class_dirs = find_class_dirs(src_root)
    if not class_dirs:
        sys.exit(
            f"No Kaggle class folders found under {src_root}.\n"
            f"Expected layout: <src_root>/train/freshapples/*.png  etc."
        )

    print(f"Found {len(class_dirs)} class folders.")
    dst_root.mkdir(parents=True, exist_ok=True)

    # Step 1: collect all images per (fruit, anchor) 
    pool: dict[tuple[str, str], list[tuple[Path, str]]] = defaultdict(list)
    for src, fruit, anchor, origin in class_dirs:
        for img in src.iterdir():
            if img.suffix.lower() in VALID_EXTS:
                pool[(fruit, anchor)].append((img, origin))

    print("\nImage counts per (fruit, anchor):")
    for (fruit, anchor), imgs in sorted(pool.items()):
        print(f"  {fruit:7s}  {anchor:6s}  {len(imgs):>5,}")

    # Step 2: score every image 
    print("\nScoring images by visual decay...")
    scores: dict[tuple[str, str], list[tuple[Path, str, float]]] = {}
    for key, imgs in pool.items():
        scored = []
        for path, origin in imgs:
            s = decay_score(path)
            if not np.isnan(s):
                scored.append((path, origin, s))
        scored.sort(key=lambda t: t[2])  # ascending: low decay first
        scores[key] = scored
        score_arr = np.array([t[2] for t in scored])
        print(f"  {key[0]:7s} {key[1]:6s}  n={len(scored):>5,}  "
              f"decay mean={score_arr.mean():.3f}  std={score_arr.std():.3f}  "
              f"min={score_arr.min():.3f}  max={score_arr.max():.3f}")

    # ---- Step 3: split each anchor at the median, write to stage folders ----
    print("\nWriting stage folders...")
    counts: dict[tuple[str, int], int] = {}
    stage_score_stats: dict[tuple[str, int], list[float]] = defaultdict(list)
    total = 0

    for (fruit, anchor), scored in scores.items():
        n = len(scored)
        if n < 4:
            print(f"  warn: only {n} images for {fruit}/{anchor}, skipping")
            continue
        median = n // 2
        offset = ANCHOR_STAGE_OFFSET[anchor]   # 1 for fresh, 3 for rotten

        for sub_idx, slice_ in enumerate([scored[:median], scored[median:]]):
            stage = offset + sub_idx           # fresh -> 1,2  rotten -> 3,4
            dst = dst_root / fruit / f"stage_{stage}"
            dst.mkdir(parents=True, exist_ok=True)
            for path, origin, score in slice_:
                target = dst / f"{origin}_{path.name}"
                if not target.exists():
                    if copy:
                        shutil.copy2(path, target)
                    else:
                        target.symlink_to(path.resolve())
                counts[(fruit, stage)] = counts.get((fruit, stage), 0) + 1
                stage_score_stats[(fruit, stage)].append(score)
                total += 1

    # Step 4: validation report 
    print(f"\nWrote {total:,} images into {dst_root}/")
    print("\nPer-stage validation (decay score should rise with stage number):")
    print(f"  {'fruit':7s}  {'stage':>5s}  {'n':>5s}  {'mean':>6s}  {'std':>6s}  {'origin'}")
    for fruit in sorted({k[0] for k in counts}):
        for stage in range(1, 5):
            key = (fruit, stage)
            if key not in counts:
                continue
            scores_arr = np.array(stage_score_stats[key])
            anchor = "fresh " if stage <= 2 else "rotten"
            print(f"  {fruit:7s}  {stage:>5d}  {counts[key]:>5,}  "
                  f"{scores_arr.mean():.3f}  {scores_arr.std():.3f}  "
                  f"{anchor}")

    # Sanity check: stage means should be monotonically increasing per fruit
    print("\nMonotonicity check (stage decay-score means per fruit):")
    ok = True
    for fruit in sorted({k[0] for k in counts}):
        means = [np.mean(stage_score_stats[(fruit, s)])
                 for s in range(1, 5) if (fruit, s) in counts]
        is_monotonic = all(means[i] < means[i + 1] for i in range(len(means) - 1))
        status = "OK" if is_monotonic else "WARN"
        print(f"  {fruit:7s}  means={[f'{m:.3f}' for m in means]}  [{status}]")
        ok = ok and is_monotonic

    if not ok:
        print("\nNon-monotonic ordering detected -- inspect score distributions.")
        print("This usually means the decay heuristic is fighting against background "
              "or lighting variation in one of the fruits. Consider adjusting "
              "`decay_score()` or using --copy to inspect borderline images.")
    else:
        print("\nAll fruits show monotonic decay-score progression across stages.")
    print("\nNext step:  python scripts/build_splits.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("src", type=Path,
                    help="Path to the extracted Kaggle archive (the directory "
                         "containing train/ and test/ subfolders).")
    ap.add_argument("--dst", type=Path, default=DATA_ROOT,
                    help=f"Destination dataset/ folder (default: {DATA_ROOT})")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking.")
    args = ap.parse_args()
    prepare(args.src.resolve(), args.dst.resolve(), copy=args.copy)


if __name__ == "__main__":
    main()
