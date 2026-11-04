# How Long Has This Got? — Predicting Fruit Shelf Life from Images

**Author:** Amaan Hussain Shaik - 24211745
**Module:** H9MLAI — Machine Learning
**Programme:** MSc Artificial Intelligence (MSCAI1A), National College of Ireland
**Lecturer:** Jaswinder Singh
**Submission:** 2 May 2026

This repository contains the full implementation, dataset, trained models, and
evaluation outputs for the H9MLAI final project. The project frames fruit
shelf-life estimation as a regression problem (predicting remaining shelf life
in days) rather than the ordinal classification used in prior work, and
compares three pipelines: SVR on HSV histograms, Random Forest on HSV+HOG, and
a fine-tuned MobileNetV2.

---

## 1. Headline result

| Model | Test MAE | Test RMSE | Test R² | CPU latency | Size on disk |
| ----- | -------- | --------- | ------- | ----------- | ------------ |
| **MobileNetV2** (served by the API) | **0.967 d** | **1.335 d** | **0.910** | 31 ms | 9.4 MB |
| RandomForest | 1.217 d | 2.007 d | 0.797 | 46 ms | 17.2 MB |
| SVR | 1.577 d | 2.322 d | 0.728 | **3.8 ms** | **6.5 MB** |

**These results replace those in the submitted report.** The dataset holds 9 copies
of each of its 1,511 photos, and the original random split put copies of every test
photo into training. That made the SVR look best (0.295 d MAE). Splitting by source
photo removes the overlap and reverses the ranking.

**Limitations**

- Labels come from image brightness, not observed spoilage, so these figures measure
  agreement with those labels rather than real shelf life in days.
- The labels fail the stage ordering check for apples, and oranges are almost tied:
  mean decay is lower for stage 3 than stage 2 (apple 0.183 vs 0.324, orange 0.154 vs 0.160).
- The API does not check that an image actually shows the selected fruit.

---

## 2. Repository layout

```
fruit_shelf_life/
├── README.md                   ← this file
├── requirements.txt            ← Python dependencies
├── config.py                   ← all paths, hyperparameters, fruit metadata
│
├── data/                       ← dataset indexing and label mapping
│   ├── dataset_loader.py
│   └── label_mapping.py
│
├── features/                   ← hand-crafted feature extractors
│   ├── color_histogram.py      ← 32-bin per-channel HSV histogram
│   └── hog_features.py         ← Histogram of Oriented Gradients
│
├── models/                     ← three model pipelines
│   ├── svr_model.py            ← Pipeline A
│   ├── random_forest.py        ← Pipeline B (+ feature-group importance)
│   └── mobilenet_model.py      ← Pipeline C (frozen-then-fine-tuned)
│
├── evaluation/                 ← metrics and benchmarking
│   ├── metrics.py              ← MAE / RMSE / R² overall and per fruit/stage
│   └── benchmark.py            ← per-image CPU latency timing
│
├── scripts/                    ← runnable entry points
│   ├── prepare_dataset.py      ← Kaggle archive → 4-stage layout
│   ├── build_splits.py         ← deterministic train/val/test split
│   ├── train_svr.py
│   ├── train_rf.py
│   ├── train_mobilenet.py
│   ├── evaluate_all.py         ← all metrics + latency + size
│   ├── ablation_study.py       ← unified vs per-fruit specialist
│   └── make_figures.py         ← all 9 paper figures from CSVs
│
├── dataset/                    ← SAMPLE ONLY (~5 images per stage for reference)
│   ├── apple/   stage_1/  stage_2/  stage_3/  stage_4/
│   ├── banana/  stage_1/  stage_2/  stage_3/  stage_4/
│   └── orange/  stage_1/  stage_2/  stage_3/  stage_4/
│   ↑ The full ~13,599-image dataset is not bundled (size). To rebuild
│     it locally, see §4.2 below.
│
└── outputs/                    ← all artefacts (trained models + CSVs + figures)
    ├── train.csv               ← train split index
    ├── val.csv                 ← validation split index
    ├── test.csv                ← test split index
    ├── svr.joblib              ← trained SVR pipeline (scaler + model)
    ├── rf.joblib               ← trained Random Forest
    ├── mobilenet.pt            ← trained MobileNetV2 weights
    ├── predictions_svr.csv     ← per-image SVR predictions on test set
    ├── predictions_rf.csv      ← per-image Random Forest predictions
    ├── predictions_mobilenet.csv  ← per-image MobileNetV2 predictions
    ├── metrics_summary.csv     ← overall MAE / RMSE / R² per model
    ├── per_fruit_metrics.csv   ← per-fruit metrics per model
    ├── latency.csv             ← CPU inference latency per model
    ├── ablation.csv            ← unified vs specialist comparison
    └── figures/                ← all 9 figures referenced in the report (PNG)
```

---

## 3. Quick verification

The `outputs/` folder already contains every artefact needed to verify the
reported numbers **without running any model**:

* **Pre-computed metrics** in `metrics_summary.csv`, `per_fruit_metrics.csv`,
  `latency.csv`, and `ablation.csv` — these are exactly the numbers quoted
  in the report and on the slides.
* **Per-image test predictions** in `predictions_svr.csv`,
  `predictions_rf.csv`, and `predictions_mobilenet.csv` (one row per
  test image, columns `path, fruit, stage, days, y_pred`). These let you
  re-derive every metric independently — for example with one line of
  pandas — without trusting our scoring code.
* **All 9 figures** as PNG in `outputs/figures/`.

To re-run inference end-to-end with the trained models in `outputs/`,
the full dataset is required (it is not bundled — see §4.2):

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# (Step 4.2 first to rebuild the dataset/ tree)
python scripts/evaluate_all.py
```

Expected: byte-identical CSVs to the ones already in `outputs/` and the
same headline figures as in §1 of this README and Table I in the report.

---

## 4. Full reproduction from scratch (~ a few hours on CPU)

### 4.1 Install

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA GPU, install the matching `torch` / `torchvision` wheels
from <https://pytorch.org/get-started/locally/> *before* running the
`pip install` above, otherwise PyTorch will be installed CPU-only.

### 4.2 Rebuild the four-stage dataset (REQUIRED)

The full ~13,599-image dataset is not bundled with this submission to keep
the archive a reasonable size. The `dataset/` folder included here contains
only ~5 sample images per stage so a marker can see the directory structure
the loader expects. To rebuild the full dataset:

1. Download the Kaggle archive from
   <https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification>
   (free, no account required for download).
2. Extract it locally to any folder.
3. Delete or rename the bundled sample `dataset/` folder so it does not
   collide with the rebuilt one.
4. Run:

   ```
   python scripts/prepare_dataset.py /path/to/extracted/kaggle/archive --copy
   ```

The script computes a brightness-based decay score per image and
median-splits each (fruit, fresh|rotten) anchor into two sub-stages,
producing the four-stage `dataset/<fruit>/stage_<n>/` layout the rest of
the pipeline expects. It prints a validation report — confirm that the
per-stage decay-score means rise monotonically across stages 1 → 4 for
every fruit before continuing.

This step is documented in detail in §IV-B of the report and on slide 5
of the presentation.

### 4.3 Build the deterministic splits

```
python scripts/build_splits.py
```

Writes `outputs/train.csv`, `outputs/val.csv`, `outputs/test.csv`
(70 / 15 / 15, stratified by fruit and stage, seed = 42 in `config.py`).

### 4.4 Train the three models (run in any order)

```
python scripts/train_svr.py          # ~1 minute
python scripts/train_rf.py           # ~3 minutes
python scripts/train_mobilenet.py    # ~30–90 minutes on CPU; ~5 minutes on GPU
```

Each script loads the train and validation splits from §4.3, trains its
model, and writes the serialised result to `outputs/`.

### 4.5 Evaluate everything on the held-out test set

```
python scripts/evaluate_all.py
```

Writes `outputs/predictions_*.csv`, `metrics_summary.csv`,
`per_fruit_metrics.csv`, and `latency.csv`.

### 4.6 Ablation: unified vs per-fruit specialists

```
python scripts/ablation_study.py
```

Trains a separate per-fruit specialist for SVR and Random Forest, compares
test-set MAE against the unified models from §4.4, and writes
`outputs/ablation.csv`.

### 4.7 Regenerate the figures

```
python scripts/make_figures.py
```

Produces all 9 figures referenced in the report from the CSVs in
`outputs/`. Output goes to `outputs/figures/`.

---

## 5. Reproducibility notes

`SEED = 42` in `config.py` controls the train/val/test split, the NumPy
random generators inside the dataset loader, the Random Forest's
bootstrap sampling, and PyTorch's CPU/GPU random state. Re-running any
training script on the same input produces identical artefacts.

The only non-deterministic step is the Random Forest's parallel tree
construction with `n_jobs=-1`; the reported metrics agree to four decimal
places across runs.

The PyTorch fine-tuning step uses `torch.backends.cudnn.benchmark = False`
and a deterministic Adam optimiser. Run-to-run variance on the
MobileNetV2 test MAE is ≤ 0.01 days.

---

## 6. Modifying the experiment

Most common changes are one-line edits to `config.py`:

| To change …                  | Edit …                                    |
| ---------------------------- | ----------------------------------------- |
| Number of stages             | `STAGE_DAYS_MULTIPLIERS` (length = stages) |
| Per-fruit shelf-life range   | `max_days` in `FRUITS`                    |
| Random seed                  | `SEED`                                    |
| Train/val/test ratios        | `SPLIT_RATIOS`                            |
| Image size for traditional   | `TRADITIONAL_IMG_SIZE`                    |
| HSV histogram bin count      | `HSV_BINS`                                |
| HOG parameters               | `HOG_PARAMS`                              |
| SVR grid-search range        | `SVR_GRID`                                |
| Random Forest hyperparams    | `RF_PARAMS`                               |
| MobileNetV2 schedule         | `MOBILENET`                               |

The data loader, splitter, feature extractors, and evaluators all read
from `config.py`, so changes propagate automatically.

To swap the decay metric used for pseudo-labelling, edit `decay_score()`
in `scripts/prepare_dataset.py` — the rest of the pipeline is agnostic.

---

## 7. Required environment

* Python 3.10 or 3.11 (3.12 also works; not tested on 3.13)
* About 1 GB free disk for the dataset + outputs
* About 2 GB RAM for SVR/Random Forest training; 4 GB+ recommended for
  MobileNetV2 fine-tuning on CPU
* Optional: any CUDA 11+ GPU dramatically speeds up §4.4 step three

Tested on Windows 11 (Python 3.11) and Ubuntu 22.04 (Python 3.10).

---

## 8. Citations and acknowledgements

The Kaggle "Fruits Fresh and Rotten for Classification" dataset by
S. R. Kalluri (2018) is the sole image source. Stage-to-days mappings
are anchored to post-harvest ranges from Dadzie & Orchard (1997) for
banana and standard food-science literature for apple and orange.

MobileNetV2 weights are the IMAGENET1K_V2 weights from torchvision,
trained on ImageNet (Deng et al., 2009). All other code is original
to this project.

Full bibliography is in §References of the final report.

---

## 9. Files for assessment

In addition to this codebase, the following are submitted separately:

* `H9MLAI_Final_Report.pdf` — IEEE-format final report (12 pages)
* `H9MLAI_Presentation.pptx` — slide deck for the 7-minute video
* `presentation_video.mp4` — the recorded presentation

For any reproduction or environment issues, please check `requirements.txt`
first — version mismatches (especially scikit-learn ≥ 1.3 and PyTorch ≥ 2.0)
are the most common source of inconsistency.
