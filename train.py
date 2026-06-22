from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CTImageDataset, get_transforms
from metrics import collect_predictions, evaluate_predictions, plot_confusion_matrix, plot_roc_curve, save_metrics
from model import create_convnext_tiny, load_checkpoint
from utils import MODEL_DIR, OUTPUT_DIR, get_device, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ConvNeXt-Tiny checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pth")
    parser.add_argument("--split-csv", type=Path, default=OUTPUT_DIR / "logs" / "test_split.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    device = get_device()
    import pandas as pd

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    model = create_convnext_tiny(len(class_names), pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    df = pd.read_csv(args.split_csv)
    ds = CTImageDataset(df, class_to_idx, get_transforms(args.image_size, train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    y_true, y_pred, y_prob, _ = collect_predictions(model, loader, device)
    metrics = evaluate_predictions(y_true, y_pred, y_prob, class_names)
    save_metrics(metrics, OUTPUT_DIR / "logs")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, OUTPUT_DIR / "confusion_matrix")
    plot_roc_curve(y_true, y_prob, class_names, OUTPUT_DIR / "roc_curve")
    logger.info("Evaluation complete: %s", metrics)


if __name__ == "__main__":
    main()
