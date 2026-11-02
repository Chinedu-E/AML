"""Tests for leakage-safe GraphStore -> HeteroData conversion."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aml.convert import (
    NODE_PARTY,
    REL_TRANSFER,
    flip_transfer_typology,
    graph_store_to_hetero,
    snapshot_to_hetero,
)
from aml.domain import Typology
from aml.features import build_snapshot_arrays
from aml.generator import Simulator, SimulatorConfig
from aml.population import PopulationConfig


@pytest.fixture
def small_sim(seed: int, small_pop_config: PopulationConfig) -> Simulator:
    return Simulator(
        SimulatorConfig(
            seed=seed,
            rate_per_second=7.0,
            pattern_event_fraction=0.2,
            population=small_pop_config,
            max_transfer_edges=5_000,
        )
    )


def test_conversion_counts_and_ids(small_sim: Simulator) -> None:
    n_events = 300
    list(small_sim.run(n_events))
    store = small_sim.store
    data = graph_store_to_hetero(store, small_sim.clock.now)

    n_parties = sum(
        1 for _, d in store.g.nodes(data=True) if d.get("kind") == "party"
    )
    n_transfers = store.transfer_edge_count()

    assert data[NODE_PARTY].x.size(0) == n_parties
    assert data[NODE_PARTY].x.size(0) == len(data[NODE_PARTY].node_ids)
    store_edge = data[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
    assert store_edge.edge_index.size(1) == n_transfers
    assert len(store_edge.tx_ids) == n_transfers
    assert torch.isfinite(data[NODE_PARTY].x).all()
    assert torch.isfinite(store_edge.edge_attr).all()

    # Round-trip: party ids are unique and match store.
    assert len(set(data[NODE_PARTY].node_ids)) == n_parties
    for pid in data[NODE_PARTY].node_ids:
        assert pid in store.g


def test_typology_leakage_changes_labels_only(small_sim: Simulator) -> None:
    list(small_sim.run(250))
    store = small_sim.store
    snap_a = build_snapshot_arrays(store, small_sim.clock.now)
    assert snap_a.transfer_tx_ids, "need at least one transfer"

    # Flip first transfer typology on the graph.
    tx_id = snap_a.transfer_tx_ids[0]
    original_typ = snap_a.transfer_typologies[0]
    new_typ = (
        Typology.STRUCTURING.value
        if original_typ == Typology.BENIGN.value
        else Typology.BENIGN.value
    )
    flip_transfer_typology(store, tx_id, typology=new_typ)

    snap_b = build_snapshot_arrays(store, small_sim.clock.now)

    np.testing.assert_allclose(snap_a.party_x, snap_b.party_x)
    np.testing.assert_allclose(snap_a.ip_x, snap_b.ip_x)
    np.testing.assert_allclose(snap_a.address_x, snap_b.address_x)
    np.testing.assert_allclose(snap_a.transfer_x, snap_b.transfer_x)

    # Labels must differ for that edge (and possibly incident parties).
    idx = snap_b.transfer_tx_ids.index(tx_id)
    assert snap_a.transfer_y[idx] != snap_b.transfer_y[idx]

    data_a = snapshot_to_hetero(snap_a)
    data_b = snapshot_to_hetero(snap_b)
    assert not torch.equal(
        data_a[NODE_PARTY, REL_TRANSFER, NODE_PARTY].y,
        data_b[NODE_PARTY, REL_TRANSFER, NODE_PARTY].y,
    )
    torch.testing.assert_close(data_a[NODE_PARTY].x, data_b[NODE_PARTY].x)
