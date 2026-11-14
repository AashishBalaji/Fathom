"""A real (not mocked) smoke-training test: trains SimpleCNN for 2
real epochs on a small real subset and asserts training loss actually
decreases. Kept deliberately tiny (CPU-only sandbox, no GPU) -- its
job is proving the training loop is wired correctly end to end
(forward, loss, backward, optimizer step, metrics), not producing a
good model. The real experiment (fathom/experiment.py) is what
produces the numbers reported in the README.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from fathom.data import EVAL_TRANSFORM, ShipPatchDataset, class_weights, load_cached_dataset, stratified_split
from fathom.models import SimpleCNN
from fathom.train import train_model


def test_simple_cnn_training_loss_decreases_on_a_real_small_subset():
    hf_ds = load_cached_dataset()
    split = stratified_split(hf_ds["label"], seed=1)

    # A small real subset -- 64 train / 32 val images -- keeps this
    # test fast on CPU while still training on real satellite chips,
    # not synthetic tensors.
    train_subset = split.train[:64]
    val_subset = split.val[:32]

    train_ds = ShipPatchDataset(hf_ds, train_subset, EVAL_TRANSFORM)
    val_ds = ShipPatchDataset(hf_ds, val_subset, EVAL_TRANSFORM)
    weights = class_weights(hf_ds, train_subset)

    model = SimpleCNN()
    history = train_model(
        model,
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(val_ds, batch_size=16, shuffle=False),
        weights,
        epochs=5,
        lr=1e-3,
        seed=1,
    )

    assert len(history.train_loss) == 5
    # Real learning signal: loss on the final epoch must be lower
    # than on the first -- not a tautology, since a wiring bug
    # (wrong loss direction, optimizer never stepping, etc.) would
    # make this fail.
    assert history.train_loss[-1] < history.train_loss[0]
