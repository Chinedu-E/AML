"""Tests for sim/replay transaction sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aml.dataset import make_tiny_dataset
from aml.domain import Network, NodeKind, Typology
from aml.engine import EngineConfig, RealtimeEngine
from aml.population import PopulationConfig
from aml.sources import (
    ReplaySource,
    build_replay_source,
    parse_transaction_row,
)
from aml.train import save_checkpoint, train_model


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _sample_rows() -> list[dict]:
    return [
        {
            "tx_id": "tx-00000001",
            "timestamp": 1.0,
            "source_id": "bank-0001",
            "dest_id": "bank-0002",
            "amount": 1000.0,
            "currency": "USD",
            "channel": "wire",
            "network": "banking",
            "typology": "benign",
            "pattern_id": None,
        },
        {
            "tx_id": "tx-00000002",
            "timestamp": 2.5,
            "source_id": "bank-0002",
            "dest_id": "shell-0001",
            "amount": 2500.0,
            "currency": "USD",
            "channel": "invoice",
            "network": "cross_border",
            "typology": "layering",
            "pattern_id": "pat-1",
        },
        {
            "tx_id": "tx-00000003",
            "timestamp": 4.0,
            "source_id": "wallet-0001",
            "dest_id": "mixer-0001",
            "amount": 0.4,
            "currency": "BTC",
            "channel": "mixer",
            "network": "crypto",
            "typology": "benign",
            "pattern_id": None,
        },
    ]


def test_parse_transaction_row() -> None:
    tx = parse_transaction_row(_sample_rows()[1])
    assert tx.tx_id == "tx-00000002"
    assert tx.network == Network.CROSS_BORDER
    assert tx.typology == Typology.LAYERING
    assert tx.pattern_id == "pat-1"


def test_replay_creates_parties_and_applies(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, _sample_rows())
    src = build_replay_source(path)
    tx = src.next_transaction()
    assert tx.tx_id == "tx-00000001"
    assert src.store.g.nodes["bank-0001"]["kind"] == NodeKind.PARTY.value
    assert src.store.g.nodes["bank-0002"]["kind"] == NodeKind.PARTY.value
    assert "lat" in src.store.g.nodes["bank-0001"]
    assert src.store.has_tx("tx-00000001")
    assert src.now == 1.0


def test_replay_loop_suffixes_tx_ids(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, _sample_rows()[:1])
    src = ReplaySource(path)
    first = src.next_transaction()
    second = src.next_transaction()
    assert first.tx_id == "tx-00000001"
    assert second.tx_id == "tx-00000001:1"
    assert src.store.has_tx("tx-00000001")
    assert src.store.has_tx("tx-00000001:1")


def test_replay_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_replay_source(tmp_path / "missing.jsonl")


def test_replay_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        ReplaySource(path)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ds = make_tiny_dataset(
        train_seeds=[42, 43],
        val_seeds=[50],
        events=120,
        snapshot_every=60,
    )
    ckpt = train_model(
        ds,
        epochs=2,
        batch_size=2,
        patience=2,
        train_baseline=False,
        device=torch.device("cpu"),
    )
    path = tmp_path_factory.mktemp("ckpt") / "replay_toy.pt"
    save_checkpoint(ckpt, path)
    return path


def test_engine_scores_replay_source(tmp_path: Path, tiny_checkpoint: Path) -> None:
    path = tmp_path / "events.jsonl"
    # Longer stream so warmup + scoring have enough edges.
    rows = []
    for i in range(40):
        rows.append(
            {
                "tx_id": f"tx-{i:08d}",
                "timestamp": float(i + 1),
                "source_id": f"bank-{i % 8:04d}",
                "dest_id": f"bank-{(i + 1) % 8:04d}",
                "amount": 500.0 + i,
                "currency": "USD",
                "channel": "ach",
                "network": "banking",
                "typology": "benign" if i % 5 else "structuring",
                "pattern_id": None if i % 5 else f"p-{i}",
            }
        )
    _write_jsonl(path, rows)
    source = build_replay_source(path)
    cfg = EngineConfig(
        seed=1,
        warmup=10,
        batch_size=5,
        threshold=0.5,
        population=PopulationConfig(banking=20, crypto=10, cross_border=8),
        checkpoint=tiny_checkpoint,
        device="cpu",
        reveal_labels=True,
    )
    engine = RealtimeEngine(cfg, source=source)
    warmed = engine.warmup(10)
    assert warmed == 10
    hello = engine.hello_dict()
    assert hello["source"] == "replay"
    assert hello["sources"] == ["replay"]
    assert hello["replay"] == str(path)

    batch = [engine.next_transaction() for _ in range(5)]
    scored = engine.score_batch(batch)
    assert scored.batch_id == 1
    assert len(scored.transfer_scores) >= 1
    assert engine.sim_time >= batch[-1].timestamp
