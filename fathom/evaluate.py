"""evaluate.py -- real evaluation metrics computed from a model's
real predictions on a held-out split, not just accuracy (accuracy
alone is misleading on this dataset's real 3:1 class imbalance --
a model that always predicts "no_ship" scores 75% accuracy while
catching zero real ships).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalReport:
    accuracy: float
    precision_per_class: List[float]  # index 0 = ship, 1 = no_ship
    recall_per_class: List[float]
    f1_per_class: List[float]
    confusion: List[List[int]]  # rows = true label, cols = predicted label
    n_examples: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "confusion_matrix": self.confusion,
            "n_examples": self.n_examples,
        }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> EvalReport:
    model.eval()
    model.to(device)
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return metrics_from_predictions(all_labels, all_preds)


def metrics_from_predictions(labels: List[int], preds: List[int]) -> EvalReport:
    """Separated from evaluate_model so the metric math itself can be
    unit-tested directly against a known, hand-computed confusion
    matrix, with no model or DataLoader involved."""
    if len(labels) != len(preds):
        raise ValueError("labels and preds must be the same length")
    n = len(labels)
    accuracy = sum(1 for lbl, pred in zip(labels, preds) if lbl == pred) / n if n else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return EvalReport(
        accuracy=accuracy,
        precision_per_class=precision.tolist(),
        recall_per_class=recall.tolist(),
        f1_per_class=f1.tolist(),
        confusion=cm.tolist(),
        n_examples=n,
    )
