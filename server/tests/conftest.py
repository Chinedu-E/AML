"""Shared fixtures: small, seed-locked populations for fast tests."""

from __future__ import annotations

import pytest

from aml.clock import SimClock, SimRng
from aml.generator import Simulator, SimulatorConfig
from aml.graph_store import GraphStore
from aml.population import Population, PopulationConfig, seed_population
from aml.typologies import InjectionContext


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def rng(seed: int) -> SimRng:
    return SimRng(seed=seed)


@pytest.fixture
def clock() -> SimClock:
    return SimClock(now=0.0)


@pytest.fixture
def small_pop_config() -> PopulationConfig:
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


@pytest.fixture
def store() -> GraphStore:
    return GraphStore(max_transfer_edges=1_000)


@pytest.fixture
def population(
    store: GraphStore, rng: SimRng, small_pop_config: PopulationConfig
) -> Population:
    return seed_population(store, rng, small_pop_config, opened_at=0.0)


@pytest.fixture
def injection_ctx(
    store: GraphStore,
    population: Population,
    rng: SimRng,
    clock: SimClock,
) -> InjectionContext:
    seq = {"n": 0}

    def next_tx_id() -> str:
        seq["n"] += 1
        return f"test-tx-{seq['n']:04d}"

    return InjectionContext(
        store=store,
        population=population,
        rng=rng,
        clock=clock,
        reserved=set(),
        next_tx_id=next_tx_id,
    )


@pytest.fixture
def simulator(seed: int, small_pop_config: PopulationConfig) -> Simulator:
    return Simulator(
        SimulatorConfig(
            seed=seed,
            rate_per_second=7.0,
            pattern_event_fraction=0.15,
            population=small_pop_config,
            max_transfer_edges=5_000,
        )
    )
