"""Transaction wire and neighborhood payloads include coordinates."""

from __future__ import annotations

from aml.clock import SimRng
from aml.domain import (
    Channel,
    Currency,
    Network,
    Transaction,
    Typology,
)
from aml.engine import EngineConfig, RealtimeEngine, neighborhood_payload
from aml.features import default_schema
from aml.geo import haversine_km
from aml.graph_store import GraphStore
from aml.model import empty_model_for_schema
from aml.population import PopulationConfig, seed_population


def _small_pop() -> PopulationConfig:
    return PopulationConfig(
        banking=20,
        crypto=10,
        cross_border=8,
        mixers=1,
        smart_contracts=1,
        on_ramps=2,
        ip_pool_size=6,
        address_pool_size=5,
        preferred_counterparties=3,
        initial_balance_min=50_000.0,
        initial_balance_max=200_000.0,
    )


def test_neighborhood_payload_includes_lat_lng() -> None:
    store = GraphStore()
    pop = seed_population(store, SimRng(seed=3), _small_pop())
    pid = next(iter(pop.parties))
    payload = neighborhood_payload(store, pid, node_cap=10, edge_cap=10)
    assert payload["nodes"]
    for node in payload["nodes"]:
        assert "lat" in node
        assert "lng" in node
        assert "city" in node
        assert "country" in node
        assert isinstance(node["lat"], float)
        assert isinstance(node["lng"], float)


def test_tx_wire_dict_includes_endpoint_geo() -> None:
    schema = default_schema()
    model = empty_model_for_schema(schema)
    cfg = EngineConfig(
        seed=5,
        warmup=0,
        population=_small_pop(),
        rate_per_second=5.0,
        batch_size=5,
    )
    engine = RealtimeEngine(cfg, model=model, schema=schema)
    parties = [
        n
        for n, data in engine.store.g.nodes(data=True)
        if data.get("kind") == "party"
    ]
    src, dst = parties[0], parties[1]
    tx = Transaction(
        tx_id="wire-geo-1",
        timestamp=1.0,
        source_id=src,
        dest_id=dst,
        amount=250.0,
        currency=Currency.USD,
        channel=Channel.WIRE,
        network=Network.BANKING,
        typology=Typology.BENIGN,
    )
    wire = engine.tx_wire_dict(tx, transfer_risk=0.1)
    assert "source_lat" in wire
    assert "source_lng" in wire
    assert "dest_lat" in wire
    assert "dest_lng" in wire
    assert "source_city" in wire
    assert "dest_country" in wire
    expected = haversine_km(
        wire["source_lat"], wire["source_lng"], wire["dest_lat"], wire["dest_lng"]
    )
    assert abs(wire["distance_km"] - expected) < 1e-6
