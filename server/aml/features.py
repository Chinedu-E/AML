"""Leakage-safe snapshot features for GraphStore -> PyG conversion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from aml.domain import (
    Channel,
    Currency,
    EdgeKind,
    Network,
    NodeKind,
    PartyType,
    Typology,
)
from aml.geo import haversine_km
from aml.graph_store import GraphStore
from aml.population import COUNTRIES

PARTY_TYPES: tuple[str, ...] = tuple(p.value for p in PartyType)
NETWORKS: tuple[str, ...] = tuple(n.value for n in Network)
CURRENCIES: tuple[str, ...] = tuple(c.value for c in Currency)
CHANNELS: tuple[str, ...] = tuple(c.value for c in Channel)


def _one_hot(value: str, categories: tuple[str, ...]) -> list[float]:
    return [1.0 if value == c else 0.0 for c in categories]


def _log1p(x: float) -> float:
    return math.log1p(max(float(x), 0.0))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


@dataclass
class FeatureConfig:
    """Window lengths in simulated seconds."""

    short_window_seconds: float = 120.0


@dataclass
class FeatureSchema:
    """Feature names plus optional train-set normalization stats."""

    party_names: list[str]
    ip_names: list[str]
    address_names: list[str]
    transfer_names: list[str]
    party_mean: list[float] | None = None
    party_std: list[float] | None = None
    ip_mean: list[float] | None = None
    ip_std: list[float] | None = None
    address_mean: list[float] | None = None
    address_std: list[float] | None = None
    transfer_mean: list[float] | None = None
    transfer_std: list[float] | None = None
    short_window_seconds: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSchema:
        return cls(**data)


def party_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"party_type_{t}" for t in PARTY_TYPES)
    names.extend(f"network_{n}" for n in NETWORKS)
    names.append("is_on_ramp")
    names.extend(
        [
            "log1p_balance",
            "log1p_tx_count",
            "account_age",
            "in_degree",
            "out_degree",
            "unique_counterparties",
            "short_count",
            "short_amount_sum",
            "short_amount_std",
            "life_count",
            "life_amount_sum",
            "life_amount_std",
            "life_amount_max",
            "life_in_sum",
            "life_out_sum",
            "life_in_out_ratio",
            "short_to_long_count_ratio",
            "lat_norm",
            "lng_sin",
            "lng_cos",
        ]
    )
    return names


def ip_feature_names() -> list[str]:
    return ["log1p_linked_parties"]


def address_feature_names() -> list[str]:
    return ["log1p_linked_parties", *[f"country_{c}" for c in COUNTRIES]]


def transfer_feature_names() -> list[str]:
    names = ["log1p_amount"]
    names.extend(f"currency_{c}" for c in CURRENCIES)
    names.extend(f"channel_{c}" for c in CHANNELS)
    names.extend(f"network_{n}" for n in NETWORKS)
    names.append("log1p_secs_since_prev")
    names.append("log1p_distance_km")
    names.append("cross_border")
    return names


def default_schema(config: FeatureConfig | None = None) -> FeatureSchema:
    cfg = config or FeatureConfig()
    return FeatureSchema(
        party_names=party_feature_names(),
        ip_names=ip_feature_names(),
        address_names=address_feature_names(),
        transfer_names=transfer_feature_names(),
        short_window_seconds=cfg.short_window_seconds,
    )


@dataclass
class SnapshotArrays:
    """Numpy feature matrices and label side-channels for one graph snapshot."""

    party_ids: list[str]
    ip_ids: list[str]
    address_ids: list[str]
    party_x: np.ndarray
    ip_x: np.ndarray
    address_x: np.ndarray
    party_y: np.ndarray
    party_mask: np.ndarray
    # Transfer edges in chronological / store order.
    transfer_src: list[str]
    transfer_dst: list[str]
    transfer_tx_ids: list[str]
    transfer_typologies: list[str]
    transfer_x: np.ndarray
    transfer_y: np.ndarray
    # Identity / on-ramp relations as (src_id, dst_id) pairs.
    has_ip: list[tuple[str, str]] = field(default_factory=list)
    has_address: list[tuple[str, str]] = field(default_factory=list)
    on_ramp: list[tuple[str, str]] = field(default_factory=list)
    snapshot_time: float = 0.0
    schema: FeatureSchema = field(default_factory=default_schema)


def _party_ids(store: GraphStore) -> list[str]:
    return [
        n
        for n, d in store.g.nodes(data=True)
        if d.get("kind") == NodeKind.PARTY.value
    ]


def _ids_of_kind(store: GraphStore, kind: NodeKind) -> list[str]:
    return [n for n, d in store.g.nodes(data=True) if d.get("kind") == kind.value]


def build_snapshot_arrays(
    store: GraphStore,
    snapshot_time: float,
    config: FeatureConfig | None = None,
    schema: FeatureSchema | None = None,
) -> SnapshotArrays:
    """Extract leakage-safe features and labels from ``store`` at ``snapshot_time``."""
    cfg = config or FeatureConfig()
    sch = schema or default_schema(cfg)
    short_start = snapshot_time - cfg.short_window_seconds

    party_ids = _party_ids(store)
    ip_ids = _ids_of_kind(store, NodeKind.IP_ADDRESS)
    address_ids = _ids_of_kind(store, NodeKind.PHYSICAL_ADDRESS)

    # Collect transfer edges (no typology in features).
    transfers: list[dict[str, Any]] = []
    for u, v, _key, data in store.g.edges(keys=True, data=True):
        if data.get("edge_kind") != EdgeKind.TRANSFER.value:
            continue
        transfers.append(
            {
                "source": u,
                "dest": v,
                "tx_id": data.get("tx_id", ""),
                "timestamp": float(data["timestamp"]),
                "amount": float(data["amount"]),
                "currency": data.get("currency", Currency.USD.value),
                "channel": data.get("channel", Channel.WIRE.value),
                "network": data.get("network", Network.BANKING.value),
                "typology": data.get("typology", Typology.BENIGN.value),
            }
        )
    transfers.sort(key=lambda e: (float(e["timestamp"]), str(e["tx_id"])))

    # Per-party incident transfers for stats and labels.
    out_by: dict[str, list[dict[str, Any]]] = {pid: [] for pid in party_ids}
    in_by: dict[str, list[dict[str, Any]]] = {pid: [] for pid in party_ids}
    for e in transfers:
        out_by.setdefault(e["source"], []).append(e)
        in_by.setdefault(e["dest"], []).append(e)

    # Seconds since previous outbound transfer from the same source (by time).
    last_out_ts: dict[str, float] = {}
    secs_since_prev: list[float] = []
    for e in transfers:
        src = e["source"]
        ts = float(e["timestamp"])
        if src in last_out_ts:
            secs_since_prev.append(max(0.0, ts - last_out_ts[src]))
        else:
            secs_since_prev.append(0.0)
        last_out_ts[src] = ts

    party_rows: list[list[float]] = []
    party_y: list[float] = []
    party_mask: list[float] = []

    for pid in party_ids:
        node = store.g.nodes[pid]
        outs = out_by.get(pid, [])
        ins = in_by.get(pid, [])
        all_inc = outs + ins

        short = [
            e
            for e in all_inc
            if short_start <= float(e["timestamp"]) <= snapshot_time
        ]
        short_amts = [float(e["amount"]) for e in short]
        life_amts = [float(e["amount"]) for e in all_inc]
        in_amts = [float(e["amount"]) for e in ins]
        out_amts = [float(e["amount"]) for e in outs]

        counterparties = {e["dest"] for e in outs} | {e["source"] for e in ins}
        life_count = float(len(all_inc))
        short_count = float(len(short))
        in_sum = float(sum(in_amts))
        out_sum = float(sum(out_amts))
        # Avoid div-by-zero; encode as log1p(in / (out + eps)) style ratio in [0,inf).
        in_out_ratio = in_sum / (out_sum + 1.0)
        short_long_ratio = short_count / (life_count + 1.0)

        row: list[float] = []
        row.extend(_one_hot(str(node.get("party_type", "")), PARTY_TYPES))
        row.extend(_one_hot(str(node.get("network", "")), NETWORKS))
        row.append(1.0 if node.get("is_on_ramp") else 0.0)
        row.append(_log1p(float(node.get("balance", 0.0))))
        row.append(_log1p(float(node.get("tx_count", 0))))
        row.append(max(0.0, snapshot_time - float(node.get("opened_at", 0.0))))
        row.append(float(len(ins)))
        row.append(float(len(outs)))
        row.append(float(len(counterparties)))
        row.append(short_count)
        row.append(_log1p(sum(short_amts)))
        row.append(_safe_std(short_amts))
        row.append(life_count)
        row.append(_log1p(sum(life_amts)))
        row.append(_safe_std(life_amts))
        row.append(_log1p(max(life_amts) if life_amts else 0.0))
        row.append(_log1p(in_sum))
        row.append(_log1p(out_sum))
        row.append(in_out_ratio)
        row.append(short_long_ratio)
        lat = float(node.get("lat", 0.0))
        lng = float(node.get("lng", 0.0))
        lng_rad = math.radians(lng)
        row.append(lat / 90.0)
        row.append(math.sin(lng_rad))
        row.append(math.cos(lng_rad))
        party_rows.append(row)

        positive = any(
            str(e.get("typology", Typology.BENIGN.value)) != Typology.BENIGN.value
            for e in all_inc
        )
        party_y.append(1.0 if positive else 0.0)
        party_mask.append(1.0 if int(node.get("tx_count", 0)) > 0 else 0.0)

    # Identity degree for IP / address.
    ip_degree: dict[str, int] = {i: 0 for i in ip_ids}
    addr_degree: dict[str, int] = {a: 0 for a in address_ids}
    has_ip: list[tuple[str, str]] = []
    has_address: list[tuple[str, str]] = []
    on_ramp: list[tuple[str, str]] = []

    for u, v, data in store.g.edges(data=True):
        kind = data.get("edge_kind")
        if kind == EdgeKind.HAS_IP.value:
            has_ip.append((u, v))
            if v in ip_degree:
                ip_degree[v] += 1
        elif kind == EdgeKind.HAS_ADDRESS.value:
            has_address.append((u, v))
            if v in addr_degree:
                addr_degree[v] += 1
        elif kind == "on_ramp":
            on_ramp.append((u, v))

    ip_rows = [[_log1p(ip_degree.get(i, 0))] for i in ip_ids]
    address_rows: list[list[float]] = []
    for aid in address_ids:
        node = store.g.nodes[aid]
        row = [_log1p(addr_degree.get(aid, 0))]
        row.extend(_one_hot(str(node.get("country", "")), COUNTRIES))
        address_rows.append(row)

    transfer_rows: list[list[float]] = []
    transfer_y: list[float] = []
    transfer_src: list[str] = []
    transfer_dst: list[str] = []
    transfer_tx_ids: list[str] = []
    transfer_typologies: list[str] = []
    for e, dt in zip(transfers, secs_since_prev, strict=True):
        row = [_log1p(float(e["amount"]))]
        row.extend(_one_hot(str(e["currency"]), CURRENCIES))
        row.extend(_one_hot(str(e["channel"]), CHANNELS))
        row.extend(_one_hot(str(e["network"]), NETWORKS))
        row.append(_log1p(dt))
        src_node = store.g.nodes.get(e["source"], {})
        dst_node = store.g.nodes.get(e["dest"], {})
        src_lat = src_node.get("lat")
        src_lng = src_node.get("lng")
        dst_lat = dst_node.get("lat")
        dst_lng = dst_node.get("lng")
        if (
            src_lat is not None
            and src_lng is not None
            and dst_lat is not None
            and dst_lng is not None
        ):
            distance_km = haversine_km(
                float(src_lat), float(src_lng), float(dst_lat), float(dst_lng)
            )
        else:
            distance_km = 0.0
        src_country = str(src_node.get("country", ""))
        dst_country = str(dst_node.get("country", ""))
        cross_border = (
            1.0
            if src_country and dst_country and src_country != dst_country
            else 0.0
        )
        row.append(_log1p(distance_km))
        row.append(cross_border)
        transfer_rows.append(row)
        typ = str(e["typology"])
        transfer_y.append(0.0 if typ == Typology.BENIGN.value else 1.0)
        transfer_src.append(e["source"])
        transfer_dst.append(e["dest"])
        transfer_tx_ids.append(str(e["tx_id"]))
        transfer_typologies.append(typ)

    n_party = len(party_ids)
    n_ip = len(ip_ids)
    n_addr = len(address_ids)
    n_tx = len(transfers)
    party_dim = len(sch.party_names)
    ip_dim = len(sch.ip_names)
    addr_dim = len(sch.address_names)
    tx_dim = len(sch.transfer_names)

    return SnapshotArrays(
        party_ids=party_ids,
        ip_ids=ip_ids,
        address_ids=address_ids,
        party_x=np.asarray(party_rows, dtype=np.float32).reshape(n_party, party_dim)
        if n_party
        else np.zeros((0, party_dim), dtype=np.float32),
        ip_x=np.asarray(ip_rows, dtype=np.float32).reshape(n_ip, ip_dim)
        if n_ip
        else np.zeros((0, ip_dim), dtype=np.float32),
        address_x=np.asarray(address_rows, dtype=np.float32).reshape(n_addr, addr_dim)
        if n_addr
        else np.zeros((0, addr_dim), dtype=np.float32),
        party_y=np.asarray(party_y, dtype=np.float32),
        party_mask=np.asarray(party_mask, dtype=np.float32),
        transfer_src=transfer_src,
        transfer_dst=transfer_dst,
        transfer_tx_ids=transfer_tx_ids,
        transfer_typologies=transfer_typologies,
        transfer_x=np.asarray(transfer_rows, dtype=np.float32).reshape(n_tx, tx_dim)
        if n_tx
        else np.zeros((0, tx_dim), dtype=np.float32),
        transfer_y=np.asarray(transfer_y, dtype=np.float32),
        has_ip=has_ip,
        has_address=has_address,
        on_ramp=on_ramp,
        snapshot_time=snapshot_time,
        schema=sch,
    )


def fit_normalization(snapshots: list[SnapshotArrays]) -> FeatureSchema:
    """Compute mean/std over training snapshots and return an updated schema."""
    if not snapshots:
        return default_schema()
    base = snapshots[0].schema

    def _stats(mats: list[np.ndarray], names: list[str]) -> tuple[list[float], list[float]]:
        if not mats or mats[0].size == 0:
            zeros = [0.0] * len(names)
            ones = [1.0] * len(names)
            return zeros, ones
        stacked = np.concatenate(mats, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float64).tolist(), std.astype(np.float64).tolist()

    party_mean, party_std = _stats([s.party_x for s in snapshots], base.party_names)
    ip_mean, ip_std = _stats([s.ip_x for s in snapshots], base.ip_names)
    addr_mean, addr_std = _stats([s.address_x for s in snapshots], base.address_names)
    tx_mean, tx_std = _stats([s.transfer_x for s in snapshots], base.transfer_names)

    return FeatureSchema(
        party_names=list(base.party_names),
        ip_names=list(base.ip_names),
        address_names=list(base.address_names),
        transfer_names=list(base.transfer_names),
        party_mean=party_mean,
        party_std=party_std,
        ip_mean=ip_mean,
        ip_std=ip_std,
        address_mean=addr_mean,
        address_std=addr_std,
        transfer_mean=tx_mean,
        transfer_std=tx_std,
        short_window_seconds=base.short_window_seconds,
    )


def apply_normalization(snap: SnapshotArrays, schema: FeatureSchema) -> SnapshotArrays:
    """Return a copy of ``snap`` with features z-scored using ``schema`` stats."""

    def _norm(x: np.ndarray, mean: list[float] | None, std: list[float] | None) -> np.ndarray:
        if mean is None or std is None or x.size == 0:
            return x.copy()
        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        return ((x - m) / s).astype(np.float32)

    return SnapshotArrays(
        party_ids=list(snap.party_ids),
        ip_ids=list(snap.ip_ids),
        address_ids=list(snap.address_ids),
        party_x=_norm(snap.party_x, schema.party_mean, schema.party_std),
        ip_x=_norm(snap.ip_x, schema.ip_mean, schema.ip_std),
        address_x=_norm(snap.address_x, schema.address_mean, schema.address_std),
        party_y=snap.party_y.copy(),
        party_mask=snap.party_mask.copy(),
        transfer_src=list(snap.transfer_src),
        transfer_dst=list(snap.transfer_dst),
        transfer_tx_ids=list(snap.transfer_tx_ids),
        transfer_typologies=list(snap.transfer_typologies),
        transfer_x=_norm(snap.transfer_x, schema.transfer_mean, schema.transfer_std),
        transfer_y=snap.transfer_y.copy(),
        has_ip=list(snap.has_ip),
        has_address=list(snap.has_address),
        on_ramp=list(snap.on_ramp),
        snapshot_time=snap.snapshot_time,
        schema=schema,
    )
