# ConvNeXt-Tiny COVID-19 CT Scan Classification

This repository contains a complete PyTorch research pipeline for three-class COVID-19 CT scan classification using ConvNeXt-Tiny with a regularised classifier head (Dense 512 → ReLU → Dropout 0.5). It includes automatic dataset discovery, stratified 70/15/15 splitting, discriminative learning rates, warmup + cosine annealing, CUDA training with mixed precision, class-imbalance handling, publication-quality metrics, and an Elsevier `cas-dc` paper scaffold.

## Project Structure

```text
project/
├── dataset/
├── outputs/
│   ├── graphs/
│   ├── confusion_matrix/
│   ├── roc_curve/
│   ├── predictions/
│   └── logs/
├── models/
├── paper/
├── src/
├── requirements.txt
├── README.md
├── run.sh
└── run.ps1
```

## Dataset

Place the uploaded CT image dataset under `dataset/`. The loader automatically detects common folder layouts and class aliases for:

- `COVID-19`
- `Non-COVID`
- `Normal`

Recommended layout:

```text
dataset/
├── COVID-19/
├── Non-COVID/
└── Normal/
```

Nested layouts such as `dataset/train/COVID-19` are also supported. The pipeline writes split CSV files and class statistics to `outputs/logs/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CUDA Setup

Install a PyTorch build matching your NVIDIA driver and CUDA runtime from the official PyTorch selector. Verify CUDA with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

To require GPU execution:

```bash
python src/train.py --require-cuda
```

## Training

```bash
python src/train.py --dataset dataset --epochs 30
```

The training script uses:

- ConvNeXt-Tiny
- AdamW optimizer
- CrossEntropyLoss
- cosine learning-rate scheduling
- mixed precision on CUDA
- weighted sampling for class imbalance
- early stopping
- checkpoint resume support

Resume training:

```bash
python src/train.py --resume models/last_checkpoint.pth --epochs 50
```

## Testing

```bash
python src/test.py --checkpoint models/best_model.pth
```

Outputs include accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix, ROC curves, and a classification report.

## Inference

```bash
python src/inference.py --image path/to/ct_image.png --checkpoint models/best_model.pth
```

Prediction images are saved in `outputs/predictions/`.

## Research Paper

Generate or refresh the IEEE paper after training:

```bash
python src/generate_paper.py
```

The paper source is in `paper/main.tex`, references are in `paper/references.bib`, and the compiled PDF is written as `paper/main.pdf` when `pdflatex` is installed.

## Results and Screenshots

After training, inspect:

- `outputs/graphs/loss_curve.png`
- `outputs/graphs/accuracy_curve.png`
- `outputs/graphs/dataset_samples.png`
- `outputs/confusion_matrix/confusion_matrix.png`
- `outputs/roc_curve/roc_curve.png`
- `outputs/logs/metrics.json`
- `outputs/predictions/*_prediction.png`

## Future Improvements

- Add external validation with hospital-level dataset separation.
- Compare ConvNeXt-Tiny against DenseNet, EfficientNetV2, Swin Transformer, and ViT.
- Add segmentation-aware lung cropping before classification.
- Calibrate predicted probabilities for clinical decision support.
- Add explainability methods (attention maps, SHAP) reviewed by radiologists.
