"""Financial-center coordinates and great-circle helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aml.clock import SimRng

EARTH_RADIUS_KM = 6371.0
# Half-width of the uniform jitter around a city center (degrees).
LOCATION_JITTER_DEG = 0.35


@dataclass(frozen=True)
class GeoCenter:
    city: str
    country: str
    lat: float
    lng: float


# One primary financial center per country code used by Population.
FINANCIAL_CENTERS: dict[str, GeoCenter] = {
    "US": GeoCenter("New York", "US", 40.7128, -74.0060),
    "GB": GeoCenter("London", "GB", 51.5074, -0.1278),
    "DE": GeoCenter("Frankfurt", "DE", 50.1109, 8.6821),
    "NG": GeoCenter("Lagos", "NG", 6.5244, 3.3792),
    "SG": GeoCenter("Singapore", "SG", 1.3521, 103.8198),
    "AE": GeoCenter("Dubai", "AE", 25.2048, 55.2708),
    "CH": GeoCenter("Zurich", "CH", 47.3769, 8.5417),
    "KY": GeoCenter("George Town", "KY", 19.2869, -81.3674),
    "PA": GeoCenter("Panama City", "PA", 8.9824, -79.5199),
    "HK": GeoCenter("Hong Kong", "HK", 22.3193, 114.1694),
}


@dataclass(frozen=True)
class Location:
    city: str
    country: str
    lat: float
    lng: float


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 points."""
    if lat1 == lat2 and lng1 == lng2:
        return 0.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def assign_location(rng: SimRng, country: str) -> Location:
    """Pick the country's financial center and jitter so parties do not stack."""
    center = FINANCIAL_CENTERS.get(country)
    if center is None:
        # Fall back to New York for unexpected country codes.
        center = FINANCIAL_CENTERS["US"]
        country = center.country
    j = LOCATION_JITTER_DEG
    lat = center.lat + rng.uniform(-j, j)
    lng = center.lng + rng.uniform(-j, j)
    lat = max(-90.0, min(90.0, lat))
    lng = ((lng + 180.0) % 360.0) - 180.0
    return Location(city=center.city, country=country, lat=lat, lng=lng)
