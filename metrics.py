from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}
CANONICAL_CLASSES = {
    "covid-19": ["covid", "covid19", "covid-19", "sarscov2", "sars-cov-2", "positive"],
    "non-covid": ["non-covid", "noncovid", "non covid", "negative"],
    "normal": ["normal", "healthy", "control"],
}


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def infer_class_from_path(path: Path, dataset_root: Path) -> str:
    parts = [_normalize_name(p) for p in path.relative_to(dataset_root).parts[:-1]]
    for part in reversed(parts):
        if part in SPLIT_NAMES:
            continue
        compact = part.replace("-", "").replace(" ", "")
        for canonical, aliases in CANONICAL_CLASSES.items():
            for alias in aliases:
                alias_compact = alias.replace("-", "").replace(" ", "")
                if alias_compact in compact:
                    return canonical
    for part in reversed(parts):
        if part not in SPLIT_NAMES:
            return part
    raise ValueError(f"Could not infer class for image: {path}")


def discover_image_records(dataset_root: Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    rows = []
    for image_path in dataset_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({"path": str(image_path), "label": infer_class_from_path(image_path, dataset_root)})

    if not rows:
        raise RuntimeError(
            f"No images found in {dataset_root}. Place class folders under project/dataset, "
            "for example dataset/COVID-19, dataset/Non-COVID, dataset/Normal."
        )
    df = pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)
    return df


def make_stratified_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    labels = df["label"].tolist()
    counts = Counter(labels)
    if min(counts.values()) < 3 or len(df) < 10:
        train_df, temp_df = train_test_split(df, train_size=train_size, random_state=seed, shuffle=True)
        val_fraction = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(temp_df, train_size=val_fraction, random_state=seed, shuffle=True)
    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, temp_idx = next(sss1.split(df["path"], df["label"]))
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]
        val_fraction = val_size / (val_size + test_size)
        temp_counts = Counter(temp_df["label"])
        if min(temp_counts.values()) < 2:
            val_df, test_df = train_test_split(temp_df, train_size=val_fraction, random_state=seed, shuffle=True)
        else:
            sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_fraction, random_state=seed)
            val_idx, test_idx = next(sss2.split(temp_df["path"], temp_df["label"]))
            val_df = temp_df.iloc[val_idx]
            test_df = temp_df.iloc[test_idx]

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def build_label_maps(labels: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted(set(labels), key=lambda name: ["covid-19", "non-covid", "normal"].index(name) if name in CANONICAL_CLASSES else 99)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def get_transforms(image_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class CTImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label, row["path"]


def make_weighted_sampler(train_df: pd.DataFrame, class_to_idx: dict[str, int]) -> WeightedRandomSampler:
    counts = Counter(train_df["label"])
    weights_by_class = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = torch.DoubleTensor([weights_by_class[label] for label in train_df["label"]])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def dataset_statistics(df: pd.DataFrame) -> dict:
    per_class = dict(Counter(df["label"]))
    return {"total_images": int(len(df)), "class_counts": per_class}


def save_split_csvs(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_df in splits.items():
        split_df.to_csv(output_dir / f"{split}_split.csv", index=False)


def save_dataset_visualizations(df: pd.DataFrame, output_dir: Path, max_per_class: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        if len(grouped[row["label"]]) < max_per_class:
            grouped[row["label"]].append(row["path"])

    classes = sorted(grouped)
    if not classes:
        return
    cols = max_per_class
    rows = len(classes)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3), squeeze=False)
    for r, label in enumerate(classes):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(grouped[label]):
                img = Image.open(grouped[label][c]).convert("RGB")
                ax.imshow(img, cmap="gray")
                ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    counts = Counter(df["label"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values(), color=["#2a9d8f", "#e76f51", "#457b9d"][: len(counts)])
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset class distribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
