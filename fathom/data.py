"""data.py -- loads the real ship-vs-no-ship satellite-imagery dataset
and builds a leak-free, stratified train/val/test split.

Dataset: "Ships in Satellite Imagery" (Kaggle, rhammell), mirrored on
the Hugging Face Hub as `jonathan-roberts1/Ships-In-Satellite-Imagery`
-- 4000 real 80x80 RGB chips cropped from real Planet satellite scenes
over San Francisco Bay, each labelled "an entire ship" (1000 images)
or "no ship or part of a ship" (3000 images). This is real satellite
imagery and a real label distribution, not a synthetic or balanced
toy set -- the 3:1 class imbalance is a real property of the data and
is handled explicitly (class-weighted loss, per-class metrics), not
hidden by reporting only accuracy.

Downloading requires network access to huggingface.co; `download.py`
fetches it once and caches it to disk (DATA_CACHE_DIR below) so every
later run, test, and experiment works offline against the same cached
copy -- avoiding both a repeated download and any risk of the dataset
silently changing between runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = REPO_ROOT / "data" / "ships_in_satellite_imagery"

LABEL_NAMES = ["ship", "no_ship"]  # index 0 = "an entire ship", 1 = "no ship or part of a ship"

# ImageNet normalization -- used for both models (the from-scratch CNN
# and the ResNet18 transfer model) so the two experiments differ only
# in architecture/pretraining, not in input preprocessing.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


@dataclass(frozen=True)
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


def load_cached_dataset():
    """Loads the dataset from the local on-disk cache written by
    download.py. Raises a clear, actionable error (not a bare
    FileNotFoundError from deep inside the datasets library) if it
    hasn't been downloaded yet."""
    if not DATA_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"No cached dataset at {DATA_CACHE_DIR}. Run `python3 -m fathom.download_data` "
            "once (needs network access to huggingface.co) to fetch and cache the real "
            "Ships-In-Satellite-Imagery dataset before training, evaluating, or testing."
        )
    from datasets import load_from_disk  # noqa: PLC0415

    return load_from_disk(str(DATA_CACHE_DIR))


def stratified_split(
    labels: List[int], train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 42
) -> SplitIndices:
    """A real stratified split -- each of the three splits preserves
    the real 3:1 class ratio, rather than risking an unlucky random
    split that leaves the (already minority) ship class thin in val
    or test. Indices are partitioned, never duplicated, so no image
    can leak across splits -- test_data.py's
    test_splits_are_disjoint_and_cover_every_index proves this
    directly rather than trusting the implementation by inspection."""
    rng = random.Random(seed)
    by_label: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for label, idxs in by_label.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_frac)
        n_val = int(len(idxs) * val_frac)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    return SplitIndices(train=sorted(train_idx), val=sorted(val_idx), test=sorted(test_idx))


class ShipPatchDataset(Dataset):
    """Wraps the real HF dataset split with a torchvision transform.
    `hf_dataset` is the full 4000-image dataset; `indices` selects
    this split's real subset of it (see stratified_split)."""

    def __init__(self, hf_dataset, indices: List[int], transform) -> None:
        self._ds = hf_dataset
        self._indices = indices
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        row = self._ds[self._indices[i]]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        return self._transform(image), label


def class_counts(hf_dataset, indices: List[int]) -> dict[int, int]:
    counts: dict[int, int] = {0: 0, 1: 0}
    for i in indices:
        counts[int(hf_dataset[i]["label"])] += 1
    return counts


def class_weights(hf_dataset, indices: List[int]) -> torch.Tensor:
    """Inverse-frequency class weights computed from the real training
    split's real label counts -- fed to the loss function so the
    3:1 imbalance doesn't quietly bias the model toward always
    predicting the majority ('no_ship') class."""
    counts = class_counts(hf_dataset, indices)
    total = sum(counts.values())
    weights = [total / (2 * counts[c]) if counts[c] else 0.0 for c in (0, 1)]
    return torch.tensor(weights, dtype=torch.float32)
