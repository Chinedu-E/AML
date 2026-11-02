"""Tests for financial-center coordinates and haversine distance."""

from __future__ import annotations

import math

from aml.clock import SimRng
from aml.geo import FINANCIAL_CENTERS, assign_location, haversine_km
from aml.graph_store import GraphStore
from aml.population import PopulationConfig, seed_population


def test_haversine_same_point_is_zero() -> None:
    assert haversine_km(40.7128, -74.0060, 40.7128, -74.0060) == 0.0


def test_haversine_nyc_london_about_5570_km() -> None:
    nyc = FINANCIAL_CENTERS["US"]
    london = FINANCIAL_CENTERS["GB"]
    km = haversine_km(nyc.lat, nyc.lng, london.lat, london.lng)
    assert 5400 < km < 5700


def test_assign_location_jitter_is_seeded() -> None:
    a = assign_location(SimRng(seed=7), "CH")
    b = assign_location(SimRng(seed=7), "CH")
    c = assign_location(SimRng(seed=8), "CH")
    assert a.city == "Zurich"
    assert a.country == "CH"
    assert a.lat == b.lat and a.lng == b.lng
    assert (a.lat, a.lng) != (c.lat, c.lng)
    assert abs(a.lat - FINANCIAL_CENTERS["CH"].lat) <= 0.35 + 1e-9
    assert abs(a.lng - FINANCIAL_CENTERS["CH"].lng) <= 0.35 + 1e-9


def test_seeded_population_stores_finite_lat_lng(
    store: GraphStore, small_pop_config: PopulationConfig
) -> None:
    pop = seed_population(store, SimRng(seed=11), small_pop_config)
    for pid, party in pop.parties.items():
        assert math.isfinite(party.lat)
        assert math.isfinite(party.lng)
        assert party.city
        assert party.country
        node = store.g.nodes[pid]
        assert node["lat"] == party.lat
        assert node["lng"] == party.lng
        assert node["city"] == party.city
