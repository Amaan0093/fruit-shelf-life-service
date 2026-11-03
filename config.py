"""
Central configuration for the fruit shelf-life regression project.

All paths, hyperparameters, and the stage-to-days mapping live here so
experiments are reproducible and easy to tweak.
"""
from pathlib import Path

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset"          # where the raw images live
ARTIFACTS = PROJECT_ROOT / "outputs"           # trained models, predictions, plots
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Reproducibility

SEED = 42


# Fruits and their shelf-life metadata
#
# CURRENT CONFIG: 3 fruits x 4 derived stages each.
# Source images come from the Kaggle "Fruits Fresh and Rotten" dataset
# (sriramr/fruits-fresh-and-rotten-for-classification), which provides
# only fresh/rotten labels. Sub-stages within each anchor are derived
# from a visual decay score by `scripts/prepare_dataset.py`.
#
# Stages (1 = freshest, 4 = most rotten):
#   stage_1 = "very fresh"     (low-decay half of fresh anchor)
#   stage_2 = "starting to age" (high-decay half of fresh anchor)
#   stage_3 = "going off"      (low-decay half of rotten anchor)
#   stage_4 = "fully rotten"   (high-decay half of rotten anchor)
#
# Days schedule: non-linear ("concave") -- fruit stays good for a
# while, then drops off fast. Multipliers of max_days:
#   stage_1 -> 1.00 * max_days
#   stage_2 -> 0.85 * max_days   (small drop -- still good)
#   stage_3 -> 0.40 * max_days   (big drop -- visibly going off)
#   stage_4 -> 0.00              (spoiled)
#
# Shelf-life ranges (max_days) at room temperature:
#   apple : ~14 days   (proposal Table I)
#   banana: ~7 days    (Dadzie & Orchard 1997 [3])
#   orange: ~10 days   (typical room-temperature post-purchase range)

STAGE_DAYS_MULTIPLIERS = (1.0, 0.85, 0.40, 0.0)


def _build_mapping(max_days: float) -> dict[int, float]:
    """Convert max_days + STAGE_DAYS_MULTIPLIERS into an explicit stage->days map."""
    return {i + 1: round(max_days * m, 2) for i, m in enumerate(STAGE_DAYS_MULTIPLIERS)}


FRUITS = {
    "apple":  {"stages": 4, "max_days": 14, "mapping": _build_mapping(14)},
    "banana": {"stages": 4, "max_days": 7,  "mapping": _build_mapping(7)},
    "orange": {"stages": 4, "max_days": 10, "mapping": _build_mapping(10)},
}
FRUIT_LIST = list(FRUITS.keys())          # canonical order for one-hot encoding
N_FRUITS = len(FRUIT_LIST)

# Image / feature settings

TRADITIONAL_IMG_SIZE = (100, 100)         # for SVR + Random Forest
DEEP_IMG_SIZE = (224, 224)                # MobileNetV2 expects 224x224

# HSV histogram: 32 bins per channel, 96-d total (per the proposal)
HSV_BINS = 32

# HOG: standard scikit-image defaults work well at 100x100
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# Train / val / test split

SPLIT_RATIOS = (0.70, 0.15, 0.15)         # train / val / test
# We stratify by (fruit, stage) so every cell is represented in each split.


# Hyperparameters

# SVR grid search
SVR_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1],
}

# Random Forest
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=SEED,
)

# MobileNetV2 training schedule
MOBILENET = dict(
    batch_size=32,
    frozen_epochs=10,        # head-only training
    finetune_epochs=20,      # full-network fine-tune
    head_lr=1e-3,
    finetune_lr=1e-5,
    weight_decay=1e-4,
    fruit_embed_dim=16,      # size of fruit-type embedding fused before regression head
)
