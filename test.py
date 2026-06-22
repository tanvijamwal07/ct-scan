from __future__ import annotations

import json
from pathlib import Path

from utils import PROJECT_ROOT, run_latex, setup_logging


def read_metrics() -> dict:
    path = PROJECT_ROOT / "outputs" / "logs" / "metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "accuracy": 0.0,
        "precision_weighted": 0.0,
        "recall_weighted": 0.0,
        "f1_weighted": 0.0,
        "auc_roc": None,
    }


def write_results_table(metrics: dict) -> None:
    auc = "--" if metrics.get("auc_roc") is None else f"{metrics['auc_roc']:.4f}"
    table = rf"""\begin{{table}}[t]
\caption{{Overall Test Performance of ConvNeXt-Tiny}}
\label{{tab:overall_results}}
\centering
\begin{{tabular}}{{lc}}
\hline
\textbf{{Metric}} & \textbf{{Value}} \\
\hline
Accuracy & {metrics.get('accuracy', 0.0):.4f} \\
Precision (weighted) & {metrics.get('precision_weighted', 0.0):.4f} \\
Recall (weighted) & {metrics.get('recall_weighted', 0.0):.4f} \\
F1-score (weighted) & {metrics.get('f1_weighted', 0.0):.4f} \\
AUC-ROC & {auc} \\
\hline
\end{{tabular}}
\end{{table}}
"""
    (PROJECT_ROOT / "paper" / "results_table.tex").write_text(table, encoding="utf-8")

    comparison = rf"""\begin{{table}}[t]
\caption{{Performance Comparison Table}}
\label{{tab:comparison}}
\centering
\begin{{tabular}}{{lcc}}
\hline
\textbf{{Method}} & \textbf{{Backbone}} & \textbf{{Accuracy}} \\
\hline
Proposed method & ConvNeXt-Tiny & {metrics.get('accuracy', 0.0):.4f} \\
Baseline 1 & DenseNet-121 & To be evaluated \\
Baseline 2 & EfficientNetV2-S & To be evaluated \\
Baseline 3 & Swin-Tiny & To be evaluated \\
\hline
\end{{tabular}}
\end{{table}}
"""
    (PROJECT_ROOT / "paper" / "comparison_table.tex").write_text(comparison, encoding="utf-8")


def main():
    logger = setup_logging()
    metrics = read_metrics()
    write_results_table(metrics)
    ok = run_latex(PROJECT_ROOT / "paper" / "main.tex", logger=logger)
    if ok:
        logger.info("Paper compiled to %s", PROJECT_ROOT / "paper" / "paper.pdf")
    else:
        logger.warning("Paper source generated, but automatic PDF compilation was unavailable or failed.")


if __name__ == "__main__":
    main()
