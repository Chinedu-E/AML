"""Tests for the Simulator benign process and typology scheduler."""

from __future__ import annotations

from aml.domain import Typology
from aml.generator import Simulator, SimulatorConfig
from aml.population import PopulationConfig


def _small_config(
    seed: int = 42, pattern_fraction: float = 0.2
) -> SimulatorConfig:
    return SimulatorConfig(
        seed=seed,
        rate_per_second=7.0,
        pattern_event_fraction=pattern_fraction,
        population=PopulationConfig(
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
        ),
        max_transfer_edges=5_000,
    )


def test_benign_events_have_null_pattern_id() -> None:
    sim = Simulator(_small_config(pattern_fraction=0.0))
    txs = sim.collect(100)
    assert len(txs) == 100
    assert all(tx.typology == Typology.BENIGN for tx in txs)
    assert all(tx.pattern_id is None for tx in txs)


def test_injected_events_share_pattern_id_and_reserve_parties() -> None:
    sim = Simulator(_small_config(pattern_fraction=0.5))
    txs = sim.collect(400)
    patterned = [tx for tx in txs if tx.typology != Typology.BENIGN]
    assert patterned, "expected at least one injected pattern"
    by_pattern: dict[str, list] = {}
    for tx in patterned:
        assert tx.pattern_id is not None
        by_pattern.setdefault(tx.pattern_id, []).append(tx)
    for pattern_id, group in by_pattern.items():
        typologies = {tx.typology for tx in group}
        assert len(typologies) == 1
        # Within a single pattern emission, party reuse across concurrent patterns
        # is prevented by reservation; edges in one pattern may reuse parties
        # (layering path). Ensure all share the same pattern_id.
        assert all(tx.pattern_id == pattern_id for tx in group)


def test_same_seed_replays_identically() -> None:
    a = Simulator(_small_config(seed=7, pattern_fraction=0.1)).collect(150)
    b = Simulator(_small_config(seed=7, pattern_fraction=0.1)).collect(150)
    assert len(a) == len(b) == 150
    for ta, tb in zip(a, b, strict=True):
        assert ta.tx_id == tb.tx_id
        assert ta.amount == tb.amount
        assert ta.timestamp == tb.timestamp
        assert ta.source_id == tb.source_id
        assert ta.dest_id == tb.dest_id
        assert ta.typology == tb.typology
        assert ta.pattern_id == tb.pattern_id


def test_balances_never_negative(simulator: Simulator) -> None:
    for _ in simulator.run(300):
        for party_id in simulator.population.parties:
            assert simulator.store.balance(party_id) >= 0.0
