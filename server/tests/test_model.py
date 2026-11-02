"""Tests for HeteroSAGE forward pass, train step, and checkpoint reload."""

from __future__ import annotations

from pathlib import Path

import torch

from aml.convert import NODE_PARTY, REL_TRANSFER
from aml.dataset import make_tiny_dataset
from aml.model import (
    HeteroSAGE,
    ModelConfig,
    build_model_from_checkpoint,
    score_snapshot,
)
from aml.train import (
    batch_loss,
    compute_pos_weight,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def test_forward_shapes() -> None:
    ds = make_tiny_dataset(events=150, snapshot_every=75)
    assert ds.train, "expected at least one training graph"
    data = ds.train[0]
    model = HeteroSAGE(ModelConfig.from_schema(ds.schema))
    model.eval()
    node_logits, edge_logits = model(data)

    n_party = data[NODE_PARTY].x.size(0)
    n_edge = data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].edge_index.size(1)
    assert node_logits.shape == (n_party,)
    assert edge_logits.shape == (n_edge,)


def test_one_train_step_and_checkpoint(tmp_path: Path) -> None:
    ds = make_tiny_dataset(
        train_seeds=[42, 43],
        val_seeds=[50],
        events=120,
        snapshot_every=60,
    )
    assert len(ds.train) >= 2
    assert ds.val

    party_w, edge_w = compute_pos_weight(ds.train)
    model = HeteroSAGE(ModelConfig.from_schema(ds.schema))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    party_crit = torch.nn.BCEWithLogitsLoss(pos_weight=party_w)
    edge_crit = torch.nn.BCEWithLogitsLoss(pos_weight=edge_w)

    model.train()
    opt.zero_grad()
    loss = batch_loss(
        model,
        ds.train[:2],
        torch.device("cpu"),
        party_crit,
        edge_crit,
    )
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    # Short train + checkpoint round-trip via score_snapshot.
    ckpt = train_model(
        ds,
        epochs=2,
        batch_size=2,
        patience=2,
        train_baseline=False,
        device=torch.device("cpu"),
    )
    path = tmp_path / "toy.pt"
    save_checkpoint(ckpt, path)
    loaded = load_checkpoint(path)
    restored = build_model_from_checkpoint(loaded)
    scores = score_snapshot(restored, ds.test[0] if ds.test else ds.train[0])
    assert scores.party_scores
    assert all(0.0 <= v <= 1.0 for v in scores.party_scores.values())
    if scores.transfer_scores:
        assert all(0.0 <= v <= 1.0 for v in scores.transfer_scores.values())

    metrics = evaluate(restored, ds.val, torch.device("cpu"))
    assert metrics.early_stop_score >= 0.0 or metrics.early_stop_score != metrics.early_stop_score
