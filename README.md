# Fruit Shelf-Life Prediction Service

![CI](https://github.com/Amaan0093/fruit-shelf-life-service/actions/workflows/ci.yml/badge.svg)

Predicts how many days an apple, banana or orange has left from a photo. The
model is a fine-tuned MobileNetV2, served as a FastAPI service in Docker,
tested and security-scanned in GitHub Actions, deployed to Azure Container
Apps, and monitored for data drift.

The project started as MSc coursework (H9MLAI Machine Learning, National
College of Ireland, May 2026). After submission, a data leakage problem was
found and fixed, which reversed the original results. Everything below
reflects the corrected version.

---

## Results

Test set: 2,034 images from 226 photos the models never saw in training.

| Model | MAE (days) | RMSE (days) | R² | CPU time per image | Size |
| ----- | ---------- | ----------- | -- | ------------------ | ---- |
| **MobileNetV2** (served) | **0.967** | **1.335** | **0.910** | 31 ms | 9.4 MB |
| Random Forest (HSV + HOG) | 1.217 | 2.007 | 0.797 | 46 ms | 17.2 MB |
| SVR (HSV histogram) | 1.577 | 2.322 | 0.728 | **3.8 ms** | **6.5 MB** |

### What changed from the coursework

The dataset holds 9 copies of each of its 1,511 photos (the original plus
rotations, flips and noise). The original random split put a copy of every test
photo into training, so the models were tested on images they had effectively
already seen. That made the SVR look best, at 0.295 days MAE.

The split now keeps every copy of a photo in the same set
(`data/dataset_loader.py`, tested in `tests/test_splits.py`). With that fixed,
the SVR's error rose about 5x and MobileNetV2 became the most accurate model.

### Limitations

- **Labels are not real shelf life.** The dataset only says fresh or rotten.
  Each half is split into two stages by image brightness, and stages map to
  days on a fixed schedule. The figures measure agreement with those labels.
- **The labels are weakest for apples and oranges.** Mean decay score should
  rise from stage 1 to stage 4, but drops from stage 2 to stage 3 for apples
  (0.324 to 0.183) and is almost tied for oranges (0.160 to 0.154).
- **One split, one seed.** Small differences, such as Random Forest edging
  MobileNetV2 on banana MAE (0.60 vs 0.66 days), may not hold up.
- **The API does not check that an image shows the selected fruit.**

---

## How it works

```
Photo + fruit type
      │
      ▼
FastAPI  (app/main.py) ── validates input, max 5 MB
      │
      ▼
MobileNetV2 + fruit embedding  (models/mobilenet_model.py)
      │
      ├──► JSON response: predicted days, model version, inference time
      └──► Structured log line: summary statistics only, never the image
                    │
                    ▼
      Drift check  (scripts/check_drift.py) ── PSI against held-out test data
```

---

## Using the API

`GET /health` returns the service status and model version.

`POST /predict` takes a form with `fruit` (`apple`, `banana` or `orange`) and
`file` (an image up to 5 MB):

```bash
curl -X POST http://127.0.0.1:8000/predict -F "fruit=apple" -F "file=@apple.png"
```

```json
{"fruit": "apple", "predicted_days": 13.29, "model_version": "mobilenetv2-v1", "inference_ms": 27.5}
```

Interactive documentation is available at `/docs` while the service runs.

---

## Running it

### With Docker

```bash
docker build -t fruit-api .
docker run --rm -p 8000:8000 fruit-api
```

The image installs CPU-only PyTorch, runs as a non-root user and contains only
the files the API needs. It is about 1.7 GB, mostly PyTorch.

### Locally

Requires Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests
python -m uvicorn app.main:app --port 8000
```

---

## Testing and CI

Every push and pull request runs `.github/workflows/ci.yml`:

1. **Tests:** API behaviour and input validation, prediction parity with the
   saved test results, split leakage checks, and drift calculations.
2. **Bandit:** scans the code for security issues.
3. **pip-audit:** checks pinned dependencies for known vulnerabilities.
4. **Docker build**, then a **Trivy** scan that fails on fixable high or
   critical vulnerabilities.
5. **Publish:** on `main` only, after every check passes, the image is pushed
   to the GitHub Container Registry, tagged with the commit ID.

Third-party actions are pinned to exact commits.

---

## Deployment

The image runs on Azure Container Apps with scale-to-zero, so it uses no
compute when idle. The first request after a quiet period can take a minute
or more while the container starts. The service has no authentication, so it
is intended for demonstrations, not real traffic.

Deploying a new version is a manual step:

```bash
az containerapp update --name fruit-api --resource-group <resource-group> --image ghcr.io/amaan0093/fruit-shelf-life-service:<commit-id>
```

---

## Monitoring and retraining

The API logs summary numbers for each prediction, never the image. A drift check
(`scripts/check_drift.py`) compares them with held-out test data using PSI, and a
value above 0.25 recommends a review. It cannot measure accuracy, because live
requests have no true labels.

It was validated with simulated traffic of 250 validation images per run
(`scripts/send_test_traffic.py`):

| Simulated traffic | Brightness PSI | Predicted days PSI | Result |
| ----------------- | -------------- | ------------------ | ------ |
| Normal | 0.051 | 0.121 | No action |
| 15% darker | 1.928 | 0.157 | Review recommended |
| 40% darker | 8.277 | 0.111 | Review recommended |

Predictions barely moved even with much darker images, so watching predictions
alone would have missed this change. The 40% darker result was confirmed on the
Azure deployment.

When a review is recommended:

1. Find the cause, such as a real change in photos or just a small sample.
2. Collect and label new data that represents the change.
3. Retrain, keeping all copies of a photo in the same split.
4. Deploy only if the new model beats the current one on held-out photos.
5. Rebuild the reference with `scripts/build_reference.py`.

---

## Reproducing the training

### 1. Install

```bash
pip install -r requirements.txt
```

This installs CPU-only PyTorch on most systems. For GPU training, install a
CUDA build of PyTorch from <https://pytorch.org/get-started/locally/> first.
CI and Docker use the versions pinned in `requirements-api.txt`.

### 2. Build the dataset

Download the Kaggle dataset
[Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
and extract it outside this project. Then:

```bash
python scripts/prepare_dataset.py /path/to/extracted/dataset --copy
```

This builds `dataset/<fruit>/stage_<n>/` (13,599 images, not committed) and
prints a stage ordering check. Expect the warnings described under
Limitations.

### 3. Split, train and evaluate

```bash
python scripts/build_splits.py
python scripts/train_svr.py
python scripts/train_rf.py
python scripts/train_mobilenet.py
python scripts/evaluate_all.py
```

The split is 70 / 15 / 15 by source photo, with `SEED = 42` in `config.py`.
MobileNetV2 training is much faster on a GPU.

### 4. Refresh the derived files

```bash
python scripts/ablation_study.py
python scripts/make_figures.py
python scripts/copy_test_images.py
python scripts/build_reference.py
```

These refresh the per-fruit comparison, the figures in `outputs/figures/`, the
parity test images and the drift reference. Run them whenever the models change.

---

## Repository layout

```
.github/workflows/ci.yml   CI: tests, security scans, image build and publish
app/                       FastAPI service and prediction logging
config.py                  Paths, hyperparameters and fruit settings
data/                      Dataset indexing, labels and the grouped split
evaluation/                Metrics, latency benchmark and drift checks
features/                  HSV histogram and HOG features (SVR, Random Forest)
models/                    SVR, Random Forest and MobileNetV2
scripts/                   Dataset, training, evaluation and monitoring scripts
tests/                     pytest suite and small test images
outputs/                   Models, splits, predictions, metrics and figures
Dockerfile                 API image
requirements-api.txt       API dependencies (pinned)
requirements-dev.txt       Adds test and feature-extraction dependencies
requirements.txt           Adds training and plotting dependencies
```

`outputs/rf.joblib` is not committed because of its size. Run
`scripts/train_rf.py` to recreate it.

---

## Acknowledgements

Images come from the Kaggle dataset Fruits Fresh and Rotten for
Classification by S. R. Kalluri (2018). MobileNetV2 starts from torchvision's
ImageNet weights. Stage-to-days mappings follow published room-temperature
shelf-life ranges, such as Dadzie & Orchard (1997) for bananas.
