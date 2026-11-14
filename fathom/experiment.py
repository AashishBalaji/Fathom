"""experiment.py -- CLI entrypoint: runs the real tuning/experimentation
comparison this project exists to demonstrate -- three model variants,
trained identically (same data, same split, same training loop, same
class-weighted loss, same seed) except for the one architectural
choice being compared, then evaluated on the same real held-out test
set with the same metrics.

Variants:
  1. baseline_cnn        SimpleCNN, trained from scratch.
  2. resnet18_frozen      Pretrained ResNet18, backbone frozen
                          (linear probe -- only the new head trains).
  3. resnet18_finetuned   Pretrained ResNet18, every layer fine-tuned.

Usage:
    python3 -m fathom.download_data   # once
    python3 -m fathom.experiment --epochs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from fathom.data import (
    EVAL_TRANSFORM,
    TRAIN_TRANSFORM,
    ShipPatchDataset,
    class_counts,
    class_weights,
    load_cached_dataset,
    stratified_split,
)
from fathom.evaluate import evaluate_model
from fathom.models import SimpleCNN, build_resnet18
from fathom.train import train_model

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "artifacts" / "results.json"

VARIANTS = ["baseline_cnn", "resnet18_frozen", "resnet18_finetuned"]


def _build_model(variant: str):
    if variant == "baseline_cnn":
        return SimpleCNN()
    if variant == "resnet18_frozen":
        return build_resnet18(freeze_backbone=True)
    if variant == "resnet18_finetuned":
        return build_resnet18(freeze_backbone=False)
    raise ValueError(f"unknown variant: {variant}")


def run_experiment(epochs: int, batch_size: int, lr: float, seed: int = 42) -> dict:
    hf_ds = load_cached_dataset()
    labels = hf_ds["label"]
    split = stratified_split(labels, seed=seed)

    train_ds = ShipPatchDataset(hf_ds, split.train, TRAIN_TRANSFORM)
    val_ds = ShipPatchDataset(hf_ds, split.val, EVAL_TRANSFORM)
    test_ds = ShipPatchDataset(hf_ds, split.test, EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    weights = class_weights(hf_ds, split.train)

    results: dict = {
        "dataset": "jonathan-roberts1/Ships-In-Satellite-Imagery",
        "run_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "class_counts": {
            "train": class_counts(hf_ds, split.train),
            "val": class_counts(hf_ds, split.val),
            "test": class_counts(hf_ds, split.test),
        },
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed},
        "variants": {},
    }

    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        model = _build_model(variant)
        t0 = time.time()
        history = train_model(model, train_loader, val_loader, weights, epochs, lr, seed=seed)
        train_seconds = time.time() - t0

        test_report = evaluate_model(model, test_loader)
        print(
            f"test accuracy={test_report.accuracy:.3f} "
            f"ship-precision={test_report.precision_per_class[0]:.3f} "
            f"ship-recall={test_report.recall_per_class[0]:.3f} "
            f"train_seconds={train_seconds:.1f}"
        )

        results["variants"][variant] = {
            "history": {
                "train_loss": history.train_loss,
                "train_acc": history.train_acc,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
            },
            "test_metrics": test_report.to_dict(),
            "train_seconds": train_seconds,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote real results to {RESULTS_PATH}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
