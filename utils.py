from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def evaluate_predictions(y_true, y_pred, y_prob, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }
    try:
        if len(class_names) == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except ValueError:
        metrics["auc_roc"] = None
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def save_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = metrics["classification_report"]
    lines = ["class,precision,recall,f1-score,support"]
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(
                f"{key},{value.get('precision', 0):.6f},{value.get('recall', 0):.6f},"
                f"{value.get('f1-score', 0):.6f},{value.get('support', 0)}"
            )
    (output_dir / "classification_report.csv").write_text("\n".join(lines), encoding="utf-8")


def plot_training_curves(history: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    style = {"linewidth": 2.4, "marker": "o", "markersize": 3}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history.get("train_loss", []), label="Training loss", color="#264653", **style)
    ax.plot(epochs, history.get("val_loss", []), label="Validation loss", color="#e76f51", **style)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history.get("train_acc", []), label="Training accuracy", color="#2a9d8f", **style)
    ax.plot(epochs, history.get("val_acc", []), label="Validation accuracy", color="#457b9d", **style)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Training and validation accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}", linewidth=2.4)
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for idx, name in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
                ax.plot(fpr, tpr, label=f"{name} AUC = {auc(fpr, tpr):.3f}", linewidth=2.0)
            except ValueError:
                continue
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []
    for images, labels, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        paths.extend(batch_paths)
    return y_true, y_pred, y_prob, paths
