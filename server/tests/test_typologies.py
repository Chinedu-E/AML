"""Tests for money-laundering typology injectors."""

from __future__ import annotations

from aml.domain import Typology
from aml.graph_store import GraphStore
from aml.typologies import (
    InjectionContext,
    inject_cycle,
    inject_layering,
    inject_scatter_gather,
    inject_structuring,
)


def _apply_all(store: GraphStore, txs) -> None:
    for tx in txs:
        assert store.apply(tx) is True


def test_structuring_fan_out(injection_ctx: InjectionContext) -> None:
    txs = inject_structuring(injection_ctx)
    assert txs, "structuring should succeed on the small test population"
    assert 8 <= len(txs) <= 12
    sources = {tx.source_id for tx in txs}
    dests = {tx.dest_id for tx in txs}
    assert len(sources) == 1
    assert len(dests) == len(txs)
    assert all(tx.typology == Typology.STRUCTURING for tx in txs)
    assert len({tx.pattern_id for tx in txs}) == 1
    assert all(8_000.0 <= tx.amount <= 9_900.0 for tx in txs)
    _apply_all(injection_ctx.store, txs)


def test_layering_path(injection_ctx: InjectionContext) -> None:
    txs = inject_layering(injection_ctx)
    assert txs
    assert 4 <= len(txs) <= 6
    assert all(tx.typology == Typology.LAYERING for tx in txs)
    assert len({tx.pattern_id for tx in txs}) == 1
    timestamps = [tx.timestamp for tx in txs]
    assert timestamps == sorted(timestamps)
    assert all(t2 > t1 for t1, t2 in zip(timestamps, timestamps[1:], strict=False))
    # Path continuity and amount conservation within fee/retain band.
    for i in range(len(txs) - 1):
        assert txs[i].dest_id == txs[i + 1].source_id
        assert txs[i + 1].amount <= txs[i].amount
        assert txs[i + 1].amount >= txs[i].amount * 0.90
    _apply_all(injection_ctx.store, txs)


def test_scatter_gather_fan_in(injection_ctx: InjectionContext) -> None:
    txs = inject_scatter_gather(injection_ctx)
    assert txs
    assert 6 <= len(txs) <= 15
    hubs = {tx.dest_id for tx in txs}
    sources = {tx.source_id for tx in txs}
    assert len(hubs) == 1
    assert len(sources) == len(txs)
    assert all(tx.typology == Typology.SCATTER_GATHER for tx in txs)
    assert len({tx.pattern_id for tx in txs}) == 1
    _apply_all(injection_ctx.store, txs)


def test_cycle_returns_to_start(injection_ctx: InjectionContext) -> None:
    txs = inject_cycle(injection_ctx)
    assert txs
    assert 3 <= len(txs) <= 5
    assert all(tx.typology == Typology.CYCLE for tx in txs)
    # Follow the path; last dest should equal first source.
    assert txs[0].source_id == txs[-1].dest_id
    for i in range(len(txs) - 1):
        assert txs[i].dest_id == txs[i + 1].source_id
    _apply_all(injection_ctx.store, txs)


def test_injection_fails_closed_when_parties_unavailable(
    injection_ctx: InjectionContext,
) -> None:
    # Reserve everyone so injectors cannot find participants.
    injection_ctx.reserved.update(injection_ctx.population.parties.keys())
    edges_before = injection_ctx.store.transfer_edge_count()
    balances = {
        pid: injection_ctx.store.balance(pid)
        for pid in injection_ctx.population.parties
    }
    assert inject_structuring(injection_ctx) == []
    assert inject_layering(injection_ctx) == []
    assert inject_scatter_gather(injection_ctx) == []
    assert inject_cycle(injection_ctx) == []
    assert injection_ctx.store.transfer_edge_count() == edges_before
    for pid, bal in balances.items():
        assert injection_ctx.store.balance(pid) == bal
