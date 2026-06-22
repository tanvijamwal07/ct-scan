import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


def ensure_project_dirs() -> None:
    for path in [
        PROJECT_ROOT / "dataset",
        OUTPUT_DIR / "graphs",
        OUTPUT_DIR / "confusion_matrix",
        OUTPUT_DIR / "roc_curve",
        OUTPUT_DIR / "predictions",
        OUTPUT_DIR / "logs",
        MODEL_DIR,
        PROJECT_ROOT / "paper" / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA GPU is required but was not detected by PyTorch.")
    return torch.device("cpu")


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    ensure_project_dirs()
    log_dir = log_dir or OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("covid_ct_convnext")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"experiment_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def gpu_info() -> dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(idx),
                "device_index": idx,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def auto_batch_size(default: int = 32) -> int:
    if not torch.cuda.is_available():
        return min(default, 16)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb >= 20:
        return 64
    if total_gb >= 10:
        return 32
    if total_gb >= 6:
        return 16
    return 8


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def model_summary_text(model: torch.nn.Module, input_size=(1, 3, 224, 224)) -> str:
    lines = [str(model)]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"\nTotal parameters: {total:,}")
    lines.append(f"Trainable parameters: {trainable:,}")
    lines.append(f"Input size: {input_size}")
    return "\n".join(lines)


def run_latex(tex_path: Path, logger: logging.Logger | None = None) -> bool:
    try:
        user_miktex = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
        pdflatex = shutil.which("pdflatex") or str(user_miktex / "pdflatex.exe")
        bibtex = shutil.which("bibtex") or str(user_miktex / "bibtex.exe")
        commands = [
            [pdflatex, "-interaction=nonstopmode", tex_path.name],
            [bibtex, tex_path.stem],
            [pdflatex, "-interaction=nonstopmode", tex_path.name],
            [pdflatex, "-interaction=nonstopmode", tex_path.name],
        ]
        ok = True
        for command in commands:
            result = subprocess.run(
                command,
                cwd=str(tex_path.parent),
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if logger:
                logger.info("%s exit code: %s", command[0], result.returncode)
            ok = ok and result.returncode == 0
        generated = tex_path.with_suffix(".pdf")
        if ok and generated.exists():
            shutil.copyfile(generated, tex_path.parent / "paper.pdf")
        return ok
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if logger:
            logger.warning("Could not compile LaTeX automatically: %s", exc)
        return False
