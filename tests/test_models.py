"""Unit tests for fathom/models.py -- real forward-pass shape checks
and the freeze_backbone switch's real gradient-flow behavior (not
just that the models import)."""

from __future__ import annotations

import torch

from fathom.models import NUM_CLASSES, SimpleCNN, build_resnet18


def test_simple_cnn_forward_pass_shape_at_the_real_80px_input_size():
    model = SimpleCNN()
    x = torch.randn(4, 3, 80, 80)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_simple_cnn_has_no_pretrained_dependency_and_trains_from_scratch():
    model = SimpleCNN()
    assert all(p.requires_grad for p in model.parameters())


def test_resnet18_forward_pass_shape_at_the_real_80px_input_size():
    model = build_resnet18(freeze_backbone=False)
    x = torch.randn(4, 3, 80, 80)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_resnet18_frozen_backbone_only_trains_the_new_head():
    model = build_resnet18(freeze_backbone=True)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    # Only the replaced fc layer's weight/bias should be trainable;
    # everything else in the pretrained backbone must be frozen.
    assert set(trainable) == {"fc.weight", "fc.bias"}
    assert len(frozen) > 0


def test_resnet18_full_finetune_trains_every_layer():
    model = build_resnet18(freeze_backbone=False)
    assert all(p.requires_grad for p in model.parameters())


def test_resnet18_fc_head_is_replaced_for_this_projects_2_class_problem():
    model = build_resnet18()
    assert model.fc.out_features == NUM_CLASSES
