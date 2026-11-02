"""Tests for JSONL snapshots and sim+IBM portfolio datasets."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from aml.dataset import (
    SOURCE_IBM,
    SOURCE_SIM,
    DatasetConfig,
    JsonlDatasetConfig,
    collect_jsonl_snapshots,
    build_jsonl_dataset,
    build_portfolio_dataset,
    split_records_chronological,
)
from aml.population import PopulationConfig
from aml.report import build_training_report, format_report_markdown, format_report_text
from aml.sources import ReplaySource
from aml.train import evaluate_by_source, train_model


def _row(
    i: int,
    *,
    typology: str = "benign",
    amount: float = 100.0,
) -> dict:
    src = f"bank-{i % 7:04d}"
    dst = f"bank-{(i + 3) % 7:04d}"
    if src == dst:
        dst = f"bank-{(i + 1) % 7:04d}"
    return {
        "tx_id": f"tx-{i:08d}",
        "timestamp": float(i + 1),
        "source_id": src,
        "dest_id": dst,
        "amount": amount,
        "currency": "USD",
        "channel": "wire",
        "network": "banking",
        "typology": typology,
        "pattern_id": f"ibm-{i}" if typology != "benign" else None,
    }


def _write_jsonl(path: Path, n: int, *, illicit_every: int = 10) -> None:
    rows = []
    for i in range(n):
        typ = "structuring" if illicit_every and i % illicit_every == 0 else "benign"
        rows.append(_row(i, typology=typ, amount=500.0 if typ != "benign" else 100.0))
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_replay_loop_false_stops(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(path, 3)
    src = ReplaySource(path, loop=False)
    assert src.n_rows == 3
    for _ in range(3):
        src.next_transaction()
    assert src.exhausted
    try:
        src.next_transaction()
        raise AssertionError("expected StopIteration")
    except StopIteration:
        pass


def test_collect_jsonl_snapshots_and_time_split(tmp_path: Path) -> None:
    path = tmp_path / "ibm.jsonl"
    _write_jsonl(path, 60, illicit_every=8)
    cfg = JsonlDatasetConfig(snapshot_every=20, final_snapshot=True)
    recs = collect_jsonl_snapshots(path, cfg)
    # 20, 40, 60 (+ final only if not on boundary — 60 is on boundary → 3)
    assert len(recs) == 3
    assert all(r.source == SOURCE_IBM for r in recs)
    assert [r.event_index for r in recs] == [20, 40, 60]

    train, val, test = split_records_chronological(recs, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == 3
    assert len(train) >= 1 and len(val) >= 1 and len(test) >= 1
    assert train[0].event_index < test[-1].event_index


def test_build_jsonl_dataset(tmp_path: Path) -> None:
    path = tmp_path / "ibm.jsonl"
    _write_jsonl(path, 100, illicit_every=7)
    ds = build_jsonl_dataset(
        path,
        JsonlDatasetConfig(snapshot_every=20, train_frac=0.6, val_frac=0.2),
    )
    assert ds.meta["mode"] == "ibm"
    assert len(ds.train) >= 1
    assert all(getattr(g, "source") == SOURCE_IBM for g in ds.train)
    assert ds.schema.party_mean is not None


def test_build_portfolio_and_train(tmp_path: Path) -> None:
    path = tmp_path / "ibm.jsonl"
    _write_jsonl(path, 80, illicit_every=6)
    sim_cfg = DatasetConfig(
        train_seeds=[42],
        val_seeds=[50],
        test_seeds=[60],
        events_per_run=120,
        snapshot_every=60,
        pattern_event_fraction=0.2,
        population=PopulationConfig(
            banking=30,
            crypto=20,
            cross_border=12,
            mixers=2,
            smart_contracts=2,
            on_ramps=3,
            ip_pool_size=10,
            address_pool_size=8,
            preferred_counterparties=3,
            initial_balance_min=50_000.0,
            initial_balance_max=150_000.0,
        ),
    )
    ds = build_portfolio_dataset(
        sim_cfg,
        jsonl_path=path,
        jsonl_config=JsonlDatasetConfig(snapshot_every=20, train_frac=0.6, val_frac=0.2),
    )
    assert ds.meta["mode"] == "portfolio"
    counts = ds.meta["counts"]["train"]
    assert counts.get(SOURCE_SIM, 0) >= 1
    assert counts.get(SOURCE_IBM, 0) >= 1
    sources = {getattr(g, "source") for g in ds.train + ds.val + ds.test}
    assert SOURCE_SIM in sources
    assert SOURCE_IBM in sources

    ckpt = train_model(
        ds,
        epochs=2,
        batch_size=2,
        patience=2,
        train_baseline=True,
        device=torch.device("cpu"),
        hidden_channels=32,
    )
    assert ckpt["dataset_meta"]["mode"] == "portfolio"
    by_src = ckpt["metrics"]["test_by_source"]
    assert SOURCE_SIM in by_src or SOURCE_IBM in by_src
    assert "n_graphs" in next(iter(by_src.values()))

    report = build_training_report(ckpt, checkpoint_path="portfolio.pt")
    assert report["portfolio"] is not None
    assert report["portfolio"]["mode"] == "portfolio"
    text = format_report_text(report)
    assert "Portfolio by source" in text
    md = format_report_markdown(report)
    assert "## Portfolio by source" in md

    # Per-source eval helper stays usable standalone.
    from aml.model import build_model_from_checkpoint

    model = build_model_from_checkpoint(ckpt)
    grouped = evaluate_by_source(model, ds.test, torch.device("cpu"))
    assert grouped


def test_split_records_requires_valid_fracs() -> None:
    try:
        split_records_chronological([], train_frac=0.9, val_frac=0.2)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
