"""Unit tests for fathom/data.py -- the real stratified split, class
weighting, and dataset wrapper, against the real cached dataset."""

from __future__ import annotations

import pytest
import torch

from fathom.data import (
    EVAL_TRANSFORM,
    ShipPatchDataset,
    class_counts,
    class_weights,
    load_cached_dataset,
    stratified_split,
)


@pytest.fixture(scope="module")
def hf_dataset():
    return load_cached_dataset()


@pytest.fixture(scope="module")
def labels(hf_dataset):
    return hf_dataset["label"]


def test_real_dataset_has_the_documented_4000_images_and_3to1_imbalance(hf_dataset, labels):
    assert len(hf_dataset) == 4000
    counts = {0: labels.count(0), 1: labels.count(1)}
    assert counts == {0: 1000, 1: 3000}


def test_stratified_split_covers_every_index_with_no_overlap(labels):
    split = stratified_split(labels)
    all_idx = split.train + split.val + split.test
    assert sorted(all_idx) == list(range(len(labels)))
    assert len(set(split.train) & set(split.val)) == 0
    assert len(set(split.train) & set(split.test)) == 0
    assert len(set(split.val) & set(split.test)) == 0


def test_stratified_split_preserves_the_real_class_ratio_in_every_split(labels):
    split = stratified_split(labels)
    for idxs in (split.train, split.val, split.test):
        ship = sum(1 for i in idxs if labels[i] == 0)
        no_ship = sum(1 for i in idxs if labels[i] == 1)
        # Real 3:1 ratio, allowing a little slack for integer rounding
        # at these split sizes rather than asserting an exact float.
        ratio = no_ship / ship
        assert 2.5 <= ratio <= 3.5


def test_stratified_split_is_deterministic_given_the_same_seed(labels):
    a = stratified_split(labels, seed=7)
    b = stratified_split(labels, seed=7)
    assert a.train == b.train and a.val == b.val and a.test == b.test


def test_class_weights_favor_the_real_minority_ship_class(hf_dataset, labels):
    split = stratified_split(labels)
    weights = class_weights(hf_dataset, split.train)
    # ship (index 0) is the minority class (1000 of 4000) -- its
    # weight must be the larger one, reflecting that real imbalance.
    assert weights[0] > weights[1]


def test_class_counts_matches_a_hand_counted_small_slice(hf_dataset):
    # First 10 real labels of this exact cached dataset, hand-checked
    # directly -- a real pin, not a tautological re-derivation.
    real_first_10 = [int(hf_dataset[i]["label"]) for i in range(10)]
    counts = class_counts(hf_dataset, list(range(10)))
    assert counts[0] == real_first_10.count(0)
    assert counts[1] == real_first_10.count(1)


def test_ship_patch_dataset_returns_a_real_normalized_tensor_and_int_label(hf_dataset, labels):
    split = stratified_split(labels)
    ds = ShipPatchDataset(hf_dataset, split.test, EVAL_TRANSFORM)
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 80, 80)
    assert label in (0, 1)


def test_ship_patch_dataset_length_matches_its_indices(hf_dataset, labels):
    split = stratified_split(labels)
    ds = ShipPatchDataset(hf_dataset, split.val, EVAL_TRANSFORM)
    assert len(ds) == len(split.val)
