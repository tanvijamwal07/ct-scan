from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import (
    CTImageDataset,
    build_label_maps,
    dataset_statistics,
    discover_image_records,
    get_transforms,
    make_stratified_splits,
    make_weighted_sampler,
    save_dataset_visualizations,
    save_split_csvs,
)
from metrics import collect_predictions, evaluate_predictions, plot_confusion_matrix, plot_roc_curve, plot_training_curves, save_metrics
from model import create_convnext_tiny, load_checkpoint
from utils import EarlyStopping, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, auto_batch_size, ensure_project_dirs, get_device, gpu_info, model_summary_text, save_json, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny for COVID-19 CT classification.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto-adjust by GPU memory.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)
    logger = setup_logging()
    device = get_device(require_cuda=args.require_cuda)
    logger.info("Using device: %s", device)
    logger.info("GPU info: %s", gpu_info())

    df = discover_image_records(args.dataset)
    stats = dataset_statistics(df)
    logger.info("Dataset statistics: %s", stats)
    save_json(stats, OUTPUT_DIR / "logs" / "dataset_statistics.json")
    save_dataset_visualizations(df, OUTPUT_DIR / "graphs")

    splits = make_stratified_splits(df, seed=args.seed)
    save_split_csvs(splits, OUTPUT_DIR / "logs")
    class_to_idx, idx_to_class = build_label_maps(df["label"])
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    save_json({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, OUTPUT_DIR / "logs" / "label_maps.json")

    small_dataset = len(df) < 7000
    pretrained = not args.no_pretrained or small_dataset
    batch_size = args.batch_size or auto_batch_size()

    train_ds = CTImageDataset(splits["train"], class_to_idx, get_transforms(args.image_size, train=True))
    val_ds = CTImageDataset(splits["val"], class_to_idx, get_transforms(args.image_size, train=False))
    test_ds = CTImageDataset(splits["test"], class_to_idx, get_transforms(args.image_size, train=False))
    sampler = make_weighted_sampler(splits["train"], class_to_idx)
    loader_args = dict(batch_size=batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = create_convnext_tiny(num_classes=len(class_names), pretrained=pretrained, dropout=0.5).to(device)
    # Discriminative learning rates: backbone 0.1x, head 1x
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * 0.1}, {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    # Warmup + cosine annealing
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopping = EarlyStopping(patience=args.patience)

    start_epoch = 1
    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
        history = checkpoint.get("history", history)
        logger.info("Resumed from %s at epoch %s", args.resume, start_epoch)

    (OUTPUT_DIR / "logs" / "model_summary.txt").write_text(model_summary_text(model), encoding="utf-8")
    logger.info("Training with batch_size=%s, pretrained=%s, small_dataset=%s", batch_size, pretrained, small_dataset)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        logger.info(
            "Epoch %03d/%03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
            "history": history,
            "class_to_idx": class_to_idx,
            "args": vars(args) | {"dataset": str(args.dataset), "resume": str(args.resume) if args.resume else None},
        }
        torch.save(checkpoint, MODEL_DIR / "last_checkpoint.pth")
        if early_stopping.step(val_acc):
            best_val_acc = val_acc
            torch.save(checkpoint, MODEL_DIR / "best_model.pth")
            logger.info("Saved new best model with validation accuracy %.4f", best_val_acc)
        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %s", epoch)
            break

    torch.save(checkpoint, MODEL_DIR / "final_model.pth")
    plot_training_curves(history, OUTPUT_DIR / "graphs")
    y_true, y_pred, y_prob, _ = collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(y_true, y_pred, y_prob, class_names)
    save_metrics(metrics, OUTPUT_DIR / "logs")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, OUTPUT_DIR / "confusion_matrix")
    plot_roc_curve(y_true, y_prob, class_names, OUTPUT_DIR / "roc_curve")
    logger.info("Test metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
