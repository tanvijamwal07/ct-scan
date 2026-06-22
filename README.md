from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import get_transforms
from model import create_convnext_tiny, load_checkpoint
from utils import MODEL_DIR, OUTPUT_DIR, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image inference.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pth")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    model = create_convnext_tiny(len(class_names), pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transform = get_transforms(args.image_size, train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    pred_idx = int(probs.argmax().item())
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx].item())
    print(f"Prediction: {pred_label} | confidence={confidence:.4f}")

    OUTPUT_DIR.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{pred_label} ({confidence:.2%})")
    output_path = OUTPUT_DIR / "predictions" / f"{args.image.stem}_prediction.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
