"""Tests for GraphStore mutations and queries."""

from __future__ import annotations

from aml.domain import (
    Channel,
    Currency,
    EdgeKind,
    Network,
    Transaction,
    Typology,
)
from aml.graph_store import GraphStore
from aml.population import Population


def _tx(
    tx_id: str,
    src: str,
    dst: str,
    *,
    amount: float = 100.0,
    ts: float = 1.0,
    typology: Typology = Typology.BENIGN,
    pattern_id: str | None = None,
) -> Transaction:
    return Transaction(
        tx_id=tx_id,
        timestamp=ts,
        source_id=src,
        dest_id=dst,
        amount=amount,
        currency=Currency.USD,
        channel=Channel.ACH,
        network=Network.BANKING,
        typology=typology,
        pattern_id=pattern_id,
    )


def test_apply_adds_directed_transfer(
    store: GraphStore, population: Population
) -> None:
    src, dst = population.party_ids_in(Network.BANKING)[:2]
    src_bal = store.balance(src)
    dst_bal = store.balance(dst)
    ok = store.apply(_tx("t1", src, dst, amount=250.0, ts=10.0))
    assert ok is True
    assert store.has_tx("t1")
    assert store.transfer_edge_count() == 1
    assert store.balance(src) == src_bal - 250.0
    assert store.balance(dst) == dst_bal + 250.0
    assert store.tx_count(src) == 1
    assert store.tx_count(dst) == 1
    assert store.g.has_edge(src, dst, key="t1")


def test_duplicate_tx_id_rejected(store: GraphStore, population: Population) -> None:
    src, dst = population.party_ids_in(Network.BANKING)[:2]
    assert store.apply(_tx("dup", src, dst, amount=50.0)) is True
    src_bal = store.balance(src)
    dst_bal = store.balance(dst)
    edges_before = store.transfer_edge_count()
    assert store.apply(_tx("dup", src, dst, amount=50.0, ts=2.0)) is False
    assert store.balance(src) == src_bal
    assert store.balance(dst) == dst_bal
    assert store.transfer_edge_count() == edges_before


def test_neighborhood_k_hop(store: GraphStore, population: Population) -> None:
    banks = population.party_ids_in(Network.BANKING)
    a, b, c = banks[0], banks[1], banks[2]
    store.apply(_tx("e1", a, b, amount=10.0, ts=1.0))
    store.apply(_tx("e2", b, c, amount=10.0, ts=2.0))
    one_hop = store.neighborhood(a, k=1)
    assert set(one_hop.nodes) == {a, b}
    two_hop = store.neighborhood(a, k=2)
    assert set(two_hop.nodes) == {a, b, c}


def test_edges_in_window(store: GraphStore, population: Population) -> None:
    src, dst = population.party_ids_in(Network.BANKING)[:2]
    store.apply(_tx("w1", src, dst, amount=10.0, ts=5.0))
    store.apply(_tx("w2", src, dst, amount=10.0, ts=15.0))
    store.apply(_tx("w3", src, dst, amount=10.0, ts=25.0))
    window = store.edges_in_window(src, 10.0, 20.0)
    assert [e["tx_id"] for e in window] == ["w2"]


def test_retention_keeps_identity_edges(
    store: GraphStore, population: Population
) -> None:
    store.max_transfer_edges = 2
    banks = population.party_ids_in(Network.BANKING)
    src, dst = banks[0], banks[1]
    identity_before = store.identity_edge_count()
    assert identity_before > 0
    store.apply(_tx("r1", src, dst, amount=1.0, ts=1.0))
    store.apply(_tx("r2", src, dst, amount=1.0, ts=2.0))
    store.apply(_tx("r3", src, dst, amount=1.0, ts=3.0))
    assert store.transfer_edge_count() == 2
    assert not store.has_tx("r1")
    assert store.has_tx("r2")
    assert store.has_tx("r3")
    assert store.identity_edge_count() == identity_before
    # Identity edges still present on the party.
    assert store.parties_with_ip(src)
    assert store.parties_with_address(src)


def test_snapshot_stats_counts(
    store: GraphStore, population: Population
) -> None:
    src, dst = population.party_ids_in(Network.BANKING)[:2]
    store.apply(
        _tx(
            "s1",
            src,
            dst,
            amount=10.0,
            typology=Typology.STRUCTURING,
            pattern_id="pat-1",
        )
    )
    stats = store.snapshot_stats()
    assert stats["transfer_edges"] == 1
    assert stats["edges_by_typology"][Typology.STRUCTURING.value] == 1
    assert stats["nodes_by_network"][Network.BANKING.value] >= 1
    assert stats["identity_edges"] > 0
    assert any(
        d.get("edge_kind") == EdgeKind.HAS_IP.value
        for _, _, d in store.g.edges(data=True)
    )
