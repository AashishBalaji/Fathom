"""Unit tests for fathom/evaluate.py's metrics math, pinned against a
hand-computed confusion matrix -- not just checking the code runs."""

from __future__ import annotations

import pytest

from fathom.evaluate import metrics_from_predictions


def test_perfect_predictions_score_1_0_everywhere():
    labels = [0, 0, 1, 1, 1]
    preds = [0, 0, 1, 1, 1]
    report = metrics_from_predictions(labels, preds)
    assert report.accuracy == 1.0
    assert report.precision_per_class == [1.0, 1.0]
    assert report.recall_per_class == [1.0, 1.0]


def test_metrics_match_a_hand_computed_confusion_matrix():
    # 2 real ships correctly caught, 1 real ship missed (predicted
    # no_ship), 1 real no_ship wrongly flagged as a ship.
    labels = [0, 0, 0, 1, 1]
    preds = [0, 0, 1, 1, 0]
    report = metrics_from_predictions(labels, preds)

    # confusion[true][pred]: row 0 (true=ship) -> [2 correct, 1 missed]
    assert report.confusion == [[2, 1], [1, 1]]
    # ship precision = TP / (TP + FP) = 2 / (2 + 1) = 0.667
    assert report.precision_per_class[0] == pytest.approx(2 / 3)
    # ship recall = TP / (TP + FN) = 2 / (2 + 1) = 0.667
    assert report.recall_per_class[0] == pytest.approx(2 / 3)
    assert report.accuracy == pytest.approx(3 / 5)


def test_always_predicting_the_majority_class_scores_high_accuracy_but_zero_ship_recall():
    # The exact failure mode class-weighted loss exists to avoid --
    # this test is the honest proof that accuracy alone would hide
    # it on this dataset's real 3:1 imbalance.
    labels = [0, 1, 1, 1]  # 1 ship, 3 no_ship -- the real ratio
    preds = [1, 1, 1, 1]  # always predicts no_ship
    report = metrics_from_predictions(labels, preds)
    assert report.accuracy == 0.75
    assert report.recall_per_class[0] == 0.0  # caught zero real ships


def test_mismatched_lengths_raise_rather_than_silently_truncating():
    with pytest.raises(ValueError):
        metrics_from_predictions([0, 1], [0])
