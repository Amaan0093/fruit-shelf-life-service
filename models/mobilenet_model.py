"""
Pipeline C from the proposal:
    MobileNetV2 (ImageNet pretrained) + fruit-type embedding -> 1-d regression head.

Training schedule (from the proposal):
    - Stage 1: freeze backbone, train head only for 10 epochs at lr=1e-3.
    - Stage 2: unfreeze everything, fine-tune for ~20 epochs at lr=1e-5.

Loss = mean absolute error on remaining shelf life in days. We use MAE rather
than MSE because we report MAE as the headline metric and want training and
evaluation aligned.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

from config import (
    ARTIFACTS,
    DEEP_IMG_SIZE,
    FRUIT_LIST,
    MOBILENET,
    N_FRUITS,
    SEED,
)
from data.dataset_loader import fruit_to_index


# Dataset

def eval_transform() -> transforms.Compose:
    """Preprocessing for inference, shared by evaluation and the API."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(DEEP_IMG_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

class FruitImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, training: bool):
        self.paths: List[str] = df["path"].tolist()
        self.fruits = np.array([fruit_to_index(f) for f in df["fruit"]], dtype=np.int64)
        self.days = df["days"].to_numpy(dtype=np.float32)
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        normalize = transforms.Normalize(
            mean=weights.transforms().mean, std=weights.transforms().std
        )
        if training:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(DEEP_IMG_SIZE[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.tf = eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.fruits[i]), float(self.days[i])


# Model

class FruitShelfLifeNet(nn.Module):
    """MobileNetV2 backbone + fruit-type embedding -> scalar shelf life."""

    def __init__(self, fruit_embed_dim: int = MOBILENET["fruit_embed_dim"], pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.last_channel  # 1280

        self.fruit_embed = nn.Embedding(N_FRUITS, fruit_embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim + fruit_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def freeze_backbone(self, freeze: bool = True):
        for p in self.features.parameters():
            p.requires_grad = not freeze

    def forward(self, img: torch.Tensor, fruit_idx: torch.Tensor) -> torch.Tensor:
        x = self.features(img)
        x = self.pool(x).flatten(1)
        e = self.fruit_embed(fruit_idx)
        x = torch.cat([x, e], dim=1)
        return self.head(x).squeeze(-1)


# Train / eval

def _set_seeds():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def _epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    loss_fn = nn.L1Loss()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, fruit, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            img, fruit, y = img.to(device), fruit.to(device), y.to(device)
            preds = model(img, fruit)
            loss = loss_fn(preds, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * img.size(0)
            n += img.size(0)
    return total / max(n, 1)


def train_mobilenet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str | None = None,
) -> FruitShelfLifeNet:
    _set_seeds()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mobilenet] device = {device}")

    train_loader = DataLoader(
        FruitImageDataset(train_df, training=True),
        batch_size=MOBILENET["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        FruitImageDataset(val_df, training=False),
        batch_size=MOBILENET["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    model = FruitShelfLifeNet().to(device)
    best_state, best_val = None, float("inf")

    # ----- Stage 1: head only -----
    model.freeze_backbone(True)
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=MOBILENET["head_lr"],
                           weight_decay=MOBILENET["weight_decay"])
    print(f"[mobilenet] Stage 1: training head for {MOBILENET['frozen_epochs']} epochs")
    for ep in range(MOBILENET["frozen_epochs"]):
        tr = _epoch(model, train_loader, opt, device, train=True)
        va = _epoch(model, val_loader, opt, device, train=False)
        print(f"  epoch {ep+1:02d}  train MAE={tr:.3f}  val MAE={va:.3f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ----- Stage 2: full fine-tune -----
    model.freeze_backbone(False)
    opt = torch.optim.Adam(model.parameters(), lr=MOBILENET["finetune_lr"],
                           weight_decay=MOBILENET["weight_decay"])
    print(f"[mobilenet] Stage 2: full fine-tune for {MOBILENET['finetune_epochs']} epochs")
    for ep in range(MOBILENET["finetune_epochs"]):
        tr = _epoch(model, train_loader, opt, device, train=True)
        va = _epoch(model, val_loader, opt, device, train=False)
        print(f"  epoch {ep+1:02d}  train MAE={tr:.3f}  val MAE={va:.3f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"[mobilenet] Best val MAE = {best_val:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save(model: FruitShelfLifeNet, out_dir: Path = ARTIFACTS):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "mobilenet.pt")
    print(f"[mobilenet] Saved -> {out_dir / 'mobilenet.pt'}")


def load(out_dir: Path = ARTIFACTS, device: str | None = None) -> FruitShelfLifeNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitShelfLifeNet(pretrained=False).to(device)
    state = torch.load(out_dir / "mobilenet.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: FruitShelfLifeNet, df: pd.DataFrame, device: str | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    loader = DataLoader(
        FruitImageDataset(df, training=False),
        batch_size=MOBILENET["batch_size"],
        shuffle=False,
        num_workers=2,
    )
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for img, fruit, _ in tqdm(loader, desc="predict"):
            img, fruit = img.to(device), fruit.to(device)
            preds.append(model(img, fruit).cpu().numpy())
    return np.concatenate(preds)
