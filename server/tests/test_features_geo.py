"""Spatial feature rows: lat/lng encodings and haversine edge attrs."""

from __future__ import annotations

import math

from aml.domain import (
    Channel,
    Currency,
    Network,
    Party,
    PartyType,
    Transaction,
    Typology,
)
from aml.features import build_snapshot_arrays, transfer_feature_names
from aml.geo import FINANCIAL_CENTERS, haversine_km
from aml.graph_store import GraphStore


def _party(
    party_id: str,
    *,
    country: str,
    lat: float,
    lng: float,
    city: str,
    network: Network = Network.BANKING,
) -> Party:
    return Party(
        party_id=party_id,
        party_type=PartyType.INDIVIDUAL,
        network=network,
        country=country,
        opened_at=0.0,
        city=city,
        lat=lat,
        lng=lng,
    )


def test_transfer_row_ends_with_distance_and_cross_border() -> None:
    store = GraphStore()
    nyc = FINANCIAL_CENTERS["US"]
    zurich = FINANCIAL_CENTERS["CH"]
    store.add_party(
        _party("a", country="US", lat=nyc.lat, lng=nyc.lng, city=nyc.city),
        balance=10_000,
    )
    store.add_party(
        _party("b", country="CH", lat=zurich.lat, lng=zurich.lng, city=zurich.city),
        balance=10_000,
    )
    store.apply(
        Transaction(
            tx_id="tx-1",
            timestamp=10.0,
            source_id="a",
            dest_id="b",
            amount=5_000,
            currency=Currency.USD,
            channel=Channel.WIRE,
            network=Network.BANKING,
            typology=Typology.BENIGN,
        )
    )
    snap = build_snapshot_arrays(store, snapshot_time=10.0)
    names = transfer_feature_names()
    assert names[-2:] == ["log1p_distance_km", "cross_border"]
    row = snap.transfer_x[0]
    assert len(row) == len(names)
    expected_km = haversine_km(nyc.lat, nyc.lng, zurich.lat, zurich.lng)
    assert abs(float(row[-2]) - math.log1p(expected_km)) < 1e-5
    assert float(row[-1]) == 1.0


def test_same_country_transfer_is_not_cross_border() -> None:
    store = GraphStore()
    nyc = FINANCIAL_CENTERS["US"]
    store.add_party(
        _party("a", country="US", lat=nyc.lat, lng=nyc.lng, city=nyc.city),
        balance=10_000,
    )
    store.add_party(
        _party(
            "b",
            country="US",
            lat=nyc.lat + 0.1,
            lng=nyc.lng + 0.1,
            city=nyc.city,
        ),
        balance=10_000,
    )
    store.apply(
        Transaction(
            tx_id="tx-2",
            timestamp=5.0,
            source_id="a",
            dest_id="b",
            amount=100,
            currency=Currency.USD,
            channel=Channel.ACH,
            network=Network.BANKING,
        )
    )
    snap = build_snapshot_arrays(store, snapshot_time=5.0)
    assert float(snap.transfer_x[0][-1]) == 0.0
    assert float(snap.transfer_x[0][-2]) > 0.0


def test_party_row_includes_lat_lng_encoding() -> None:
    store = GraphStore()
    london = FINANCIAL_CENTERS["GB"]
    store.add_party(
        _party(
            "p1",
            country="GB",
            lat=london.lat,
            lng=london.lng,
            city=london.city,
        ),
        balance=1_000,
    )
    snap = build_snapshot_arrays(store, snapshot_time=0.0)
    row = snap.party_x[0]
    assert abs(float(row[-3]) - london.lat / 90.0) < 1e-6
    lng_rad = math.radians(london.lng)
    assert abs(float(row[-2]) - math.sin(lng_rad)) < 1e-6
    assert abs(float(row[-1]) - math.cos(lng_rad)) < 1e-6


def test_open_checkpoint_rejects_stale_schema(tmp_path) -> None:
    import torch

    from aml.engine import open_checkpoint
    from aml.features import FeatureSchema

    path = tmp_path / "stale.pt"
    stale = FeatureSchema(
        party_names=["a"],
        ip_names=["b"],
        address_names=["c"],
        transfer_names=["d"],
    )
    torch.save(
        {
            "feature_schema": stale.to_dict(),
            "model_config": {
                "hidden_channels": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "party_in": 1,
                "ip_in": 1,
                "address_in": 1,
                "transfer_edge_dim": 1,
            },
            "state_dict": {},
        },
        path,
    )
    try:
        open_checkpoint(path)
        raised = False
    except ValueError as exc:
        raised = True
        assert "stale" in str(exc).lower() or "Retrain" in str(exc)
    assert raised
