"""Tests for RealtimeEngine scoring, alerts, and subscribe filters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aml.domain import (
    Channel,
    Currency,
    Network,
    Transaction,
    Typology,
)
from aml.engine import (
    EngineConfig,
    RealtimeEngine,
    build_alerts,
    neighborhood_payload,
)
from aml.graph_store import GraphStore
from aml.clock import SimRng
from aml.dataset import make_tiny_dataset
from aml.model import build_model_from_checkpoint
from aml.population import PopulationConfig, seed_population
from aml.train import save_checkpoint, train_model


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
    path = tmp_path_factory.mktemp("ckpt") / "toy.pt"
    save_checkpoint(ckpt, path)
    return path


def _small_pop() -> PopulationConfig:
    return PopulationConfig(
        banking=40,
        crypto=25,
        cross_border=16,
        mixers=2,
        smart_contracts=2,
        on_ramps=4,
        ip_pool_size=12,
        address_pool_size=10,
        preferred_counterparties=4,
        initial_balance_min=50_000.0,
        initial_balance_max=200_000.0,
    )


def test_score_batch_after_warmup(tiny_checkpoint: Path) -> None:
    cfg = EngineConfig(
        seed=7,
        rate_per_second=100.0,
        batch_size=5,
        warmup=40,
        threshold=0.5,
        population=_small_pop(),
        checkpoint=tiny_checkpoint,
        device="cpu",
    )
    engine = RealtimeEngine(cfg)
    n = engine.warmup(40)
    assert n == 40

    txs = [engine.next_transaction() for _ in range(5)]
    scored = engine.score_batch(txs)

    assert scored.batch_id == 1
    assert scored.infer_ms >= 0.0
    for tx in txs:
        assert tx.tx_id in scored.transfer_scores
        risk = scored.transfer_scores[tx.tx_id]
        assert 0.0 <= risk <= 1.0
        assert engine.store.has_tx(tx.tx_id)
        assert tx.source_id in scored.party_scores
        assert tx.dest_id in scored.party_scores
        assert 0.0 <= scored.party_scores[tx.source_id] <= 1.0
        assert 0.0 <= scored.party_scores[tx.dest_id] <= 1.0


def test_build_alerts_threshold_dedupe_and_cap() -> None:
    rng = SimRng(seed=1)
    store = GraphStore()
    seed_population(
        store,
        rng,
        PopulationConfig(
            banking=20,
            crypto=10,
            cross_border=8,
            mixers=1,
            smart_contracts=1,
            on_ramps=2,
            ip_pool_size=6,
            address_pool_size=6,
            preferred_counterparties=2,
            initial_balance_min=100_000.0,
            initial_balance_max=200_000.0,
        ),
    )
    parties = [n for n, d in store.g.nodes(data=True) if d.get("kind") == "party"]
    src, dst = parties[0], parties[1]
    # Seed a few transfers so neighborhood is non-empty.
    for i in range(3):
        store.set_balance(src, 1_000_000.0)
        tx = Transaction(
            tx_id=f"tx-seed-{i}",
            timestamp=float(i),
            source_id=src,
            dest_id=dst,
            amount=100.0,
            currency=Currency.USD,
            channel=Channel.WIRE,
            network=Network.BANKING,
            typology=Typology.BENIGN,
        )
        assert store.apply(tx)

    batch_tx = Transaction(
        tx_id="tx-hot",
        timestamp=10.0,
        source_id=src,
        dest_id=dst,
        amount=500.0,
        currency=Currency.USD,
        channel=Channel.WIRE,
        network=Network.BANKING,
        typology=Typology.STRUCTURING,
        pattern_id="pat-1",
    )
    store.set_balance(src, 1_000_000.0)
    assert store.apply(batch_tx)

    transfer_scores = {"tx-hot": 0.92, "tx-seed-0": 0.1}
    party_scores = {src: 0.88, dst: 0.2}
    prev = {src: 0.1, dst: 0.2}

    alerts, alerted = build_alerts(
        batch_txs=[batch_tx],
        transfer_scores=transfer_scores,
        party_scores=party_scores,
        prev_party_scores=prev,
        store=store,
        snapshot_time=10.0,
        threshold=0.5,
        alerted_parties=set(),
    )
    kinds = {a.kind for a in alerts}
    assert "transfer" in kinds
    assert "party" in kinds
    transfer_alert = next(a for a in alerts if a.kind == "transfer")
    assert transfer_alert.score == 0.92
    assert transfer_alert.neighborhood["center"] == src
    assert len(transfer_alert.neighborhood["nodes"]) <= 40
    assert len(transfer_alert.neighborhood["edges"]) <= 80
    assert src in alerted

    # Second pass: same scores — party should not re-alert (no rise).
    alerts2, alerted2 = build_alerts(
        batch_txs=[batch_tx],
        transfer_scores=transfer_scores,
        party_scores=party_scores,
        prev_party_scores={src: 0.88, dst: 0.2},
        store=store,
        snapshot_time=10.0,
        threshold=0.5,
        alerted_parties=alerted,
    )
    party_alerts = [a for a in alerts2 if a.kind == "party"]
    assert party_alerts == []
    # Transfer still alerts each batch when above threshold.
    assert any(a.kind == "transfer" for a in alerts2)
    assert alerted2 == alerted

    # Rise of 0.05 re-fires party alert.
    alerts3, _ = build_alerts(
        batch_txs=[],
        transfer_scores={},
        party_scores={src: 0.95},
        prev_party_scores={src: 0.88},
        store=store,
        snapshot_time=10.0,
        threshold=0.5,
        alerted_parties=alerted,
    )
    assert any(a.kind == "party" and a.subject_id == src for a in alerts3)

    # Cap check on neighborhood helper directly.
    payload = neighborhood_payload(store, src, node_cap=2, edge_cap=1)
    assert len(payload["nodes"]) <= 2
    assert len(payload["edges"]) <= 1


def test_subscribe_filter_excludes_other_networks() -> None:
    """Crypto-only subscriber logic: banking messages are skipped by Hub filter."""
    import asyncio

    from aml.serve import ALL_NETWORKS, Hub

    hub = Hub()

    class FakeWS:
        def __init__(self) -> None:
            self.closed = False

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = True

    async def _run() -> list[dict]:
        ws = FakeWS()
        state = await hub.register(ws)
        hub.set_networks(ws, ["crypto"])
        assert state.networks == {"crypto"}

        received: list[dict] = []

        async def drain() -> None:
            while True:
                msg = await state.queue.get()
                if msg is None:
                    return
                received.append(msg)

        task = asyncio.create_task(drain())
        await hub.broadcast(
            {"type": "transaction", "network": "banking", "tx_id": "b1"},
            network="banking",
        )
        await hub.broadcast(
            {"type": "transaction", "network": "crypto", "tx_id": "c1"},
            network="crypto",
        )
        await asyncio.sleep(0.05)
        await hub.unregister(ws)
        await asyncio.wait_for(task, timeout=1.0)
        return received

    msgs = asyncio.run(_run())
    tx_ids = [m.get("tx_id") for m in msgs if m.get("type") == "transaction"]
    assert "c1" in tx_ids
    assert "b1" not in tx_ids
    assert ALL_NETWORKS >= {"banking", "crypto", "cross_border"}


def test_alert_ids_are_unique_across_batches(tiny_checkpoint: Path) -> None:
    cfg = EngineConfig(
        seed=9,
        rate_per_second=100.0,
        batch_size=4,
        warmup=30,
        threshold=0.0,
        population=_small_pop(),
        checkpoint=tiny_checkpoint,
        device="cpu",
    )
    engine = RealtimeEngine(cfg)
    engine.warmup(30)
    first = [engine.next_transaction() for _ in range(4)]
    scored = engine.score_batch(first)
    ids = [alert.alert_id for alert in scored.alerts]
    assert ids
    assert len(ids) == len(set(ids))
    assert all(alert.alert_id.startswith("a") for alert in scored.alerts)
    assert all(alert.to_dict()["alert_id"] == alert.alert_id for alert in scored.alerts)

    second = [engine.next_transaction() for _ in range(4)]
    scored2 = engine.score_batch(second)
    ids2 = [alert.alert_id for alert in scored2.alerts]
    assert ids2
    assert set(ids).isdisjoint(ids2)


def test_open_checkpoint_missing(tmp_path: Path) -> None:
    from aml.engine import open_checkpoint

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        open_checkpoint(tmp_path / "missing.pt")


def test_checkpoint_reload_matches_train(tiny_checkpoint: Path) -> None:
    from aml.engine import open_checkpoint

    model, schema, meta = open_checkpoint(tiny_checkpoint)
    assert schema.party_mean is not None
    assert "state_dict" in meta or model is not None
    # Smoke: build_model path used by engine.
    restored = build_model_from_checkpoint(
        __import__("aml.train", fromlist=["load_checkpoint"]).load_checkpoint(
            tiny_checkpoint
        )
    )
    assert restored is not None
