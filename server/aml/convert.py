"""Convert GraphStore snapshots to PyTorch Geometric HeteroData."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData

from aml.domain import EdgeKind, Typology
from aml.features import (
    FeatureConfig,
    FeatureSchema,
    SnapshotArrays,
    apply_normalization,
    build_snapshot_arrays,
    default_schema,
)
from aml.graph_store import GraphStore

NODE_PARTY = "party"
NODE_IP = "ip_address"
NODE_ADDRESS = "physical_address"

REL_TRANSFER = "transfer"
REL_REV_TRANSFER = "rev_transfer"
REL_HAS_IP = "has_ip"
REL_REV_HAS_IP = "rev_has_ip"
REL_HAS_ADDRESS = "has_address"
REL_REV_HAS_ADDRESS = "rev_has_address"
REL_ON_RAMP = "on_ramp"
REL_REV_ON_RAMP = "rev_on_ramp"


def _index_map(ids: list[str]) -> dict[str, int]:
    return {nid: i for i, nid in enumerate(ids)}


def _edge_index(
    pairs: list[tuple[str, str]], src_map: dict[str, int], dst_map: dict[str, int]
) -> torch.Tensor:
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [src_map[u] for u, v in pairs if u in src_map and v in dst_map]
    dst = [dst_map[v] for u, v in pairs if u in src_map and v in dst_map]
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _reverse(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    return edge_index.flip(0)


def snapshot_to_hetero(
    snap: SnapshotArrays,
    *,
    schema: FeatureSchema | None = None,
    normalize: bool = False,
) -> HeteroData:
    """Build a HeteroData object from precomputed snapshot arrays."""
    if normalize:
        if schema is None or schema.party_mean is None:
            raise ValueError("normalize=True requires a schema with fitted stats")
        snap = apply_normalization(snap, schema)
    elif schema is not None:
        snap.schema = schema

    data = HeteroData()

    data[NODE_PARTY].x = torch.from_numpy(snap.party_x)
    data[NODE_PARTY].y = torch.from_numpy(snap.party_y)
    data[NODE_PARTY].train_mask = torch.from_numpy(snap.party_mask).bool()
    data[NODE_PARTY].node_ids = snap.party_ids

    data[NODE_IP].x = torch.from_numpy(snap.ip_x)
    data[NODE_IP].node_ids = snap.ip_ids

    data[NODE_ADDRESS].x = torch.from_numpy(snap.address_x)
    data[NODE_ADDRESS].node_ids = snap.address_ids

    party_map = _index_map(snap.party_ids)
    ip_map = _index_map(snap.ip_ids)
    addr_map = _index_map(snap.address_ids)

    # Transfer edges (multi-edges as repeated pairs).
    transfer_pairs = list(zip(snap.transfer_src, snap.transfer_dst, strict=True))
    transfer_ei = _edge_index(transfer_pairs, party_map, party_map)
    transfer_attr = torch.from_numpy(snap.transfer_x)
    transfer_y = torch.from_numpy(snap.transfer_y)

    data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].edge_index = transfer_ei
    data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].edge_attr = transfer_attr
    data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].y = transfer_y
    data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].tx_ids = list(snap.transfer_tx_ids)
    data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].typologies = list(
        snap.transfer_typologies
    )

    # Reverse transfer: same attrs/labels so neighborhood messages flow back.
    data[NODE_PARTY, REL_REV_TRANSFER, NODE_PARTY].edge_index = _reverse(transfer_ei)
    data[NODE_PARTY, REL_REV_TRANSFER, NODE_PARTY].edge_attr = transfer_attr.clone()

    # Identity edges with constant scalar attr.
    has_ip_ei = _edge_index(snap.has_ip, party_map, ip_map)
    ones_ip = torch.ones((has_ip_ei.size(1), 1), dtype=torch.float32)
    data[NODE_PARTY, REL_HAS_IP, NODE_IP].edge_index = has_ip_ei
    data[NODE_PARTY, REL_HAS_IP, NODE_IP].edge_attr = ones_ip
    data[NODE_IP, REL_REV_HAS_IP, NODE_PARTY].edge_index = _reverse(has_ip_ei)
    data[NODE_IP, REL_REV_HAS_IP, NODE_PARTY].edge_attr = ones_ip.clone()

    has_addr_ei = _edge_index(snap.has_address, party_map, addr_map)
    ones_addr = torch.ones((has_addr_ei.size(1), 1), dtype=torch.float32)
    data[NODE_PARTY, REL_HAS_ADDRESS, NODE_ADDRESS].edge_index = has_addr_ei
    data[NODE_PARTY, REL_HAS_ADDRESS, NODE_ADDRESS].edge_attr = ones_addr
    data[NODE_ADDRESS, REL_REV_HAS_ADDRESS, NODE_PARTY].edge_index = _reverse(
        has_addr_ei
    )
    data[NODE_ADDRESS, REL_REV_HAS_ADDRESS, NODE_PARTY].edge_attr = ones_addr.clone()

    on_ramp_ei = _edge_index(snap.on_ramp, party_map, party_map)
    ones_or = torch.ones((on_ramp_ei.size(1), 1), dtype=torch.float32)
    data[NODE_PARTY, REL_ON_RAMP, NODE_PARTY].edge_index = on_ramp_ei
    data[NODE_PARTY, REL_ON_RAMP, NODE_PARTY].edge_attr = ones_or
    data[NODE_PARTY, REL_REV_ON_RAMP, NODE_PARTY].edge_index = _reverse(on_ramp_ei)
    data[NODE_PARTY, REL_REV_ON_RAMP, NODE_PARTY].edge_attr = ones_or.clone()

    data.snapshot_time = snap.snapshot_time
    data.feature_schema = snap.schema.to_dict()
    return data


def graph_store_to_hetero(
    store: GraphStore,
    snapshot_time: float,
    *,
    config: FeatureConfig | None = None,
    schema: FeatureSchema | None = None,
    normalize: bool = False,
) -> HeteroData:
    """Build HeteroData directly from a live GraphStore."""
    snap = build_snapshot_arrays(
        store,
        snapshot_time,
        config=config,
        schema=schema or default_schema(config),
    )
    return snapshot_to_hetero(snap, schema=schema, normalize=normalize)


def flip_transfer_typology(
    store: GraphStore, tx_id: str, typology: str = Typology.STRUCTURING.value
) -> None:
    """Mutate a transfer edge typology in-place (for leakage tests)."""
    for _u, _v, _key, data in store.g.edges(keys=True, data=True):
        if data.get("edge_kind") != EdgeKind.TRANSFER.value:
            continue
        if data.get("tx_id") == tx_id:
            data["typology"] = typology
            return
    raise KeyError(f"transfer not found: {tx_id}")


def hetero_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    """Canonical node types and edge types for HeteroSAGE construction."""
    node_types = [NODE_PARTY, NODE_IP, NODE_ADDRESS]
    edge_types = [
        (NODE_PARTY, REL_TRANSFER, NODE_PARTY),
        (NODE_PARTY, REL_REV_TRANSFER, NODE_PARTY),
        (NODE_PARTY, REL_HAS_IP, NODE_IP),
        (NODE_IP, REL_REV_HAS_IP, NODE_PARTY),
        (NODE_PARTY, REL_HAS_ADDRESS, NODE_ADDRESS),
        (NODE_ADDRESS, REL_REV_HAS_ADDRESS, NODE_PARTY),
        (NODE_PARTY, REL_ON_RAMP, NODE_PARTY),
        (NODE_PARTY, REL_REV_ON_RAMP, NODE_PARTY),
    ]
    return node_types, edge_types


def feature_dims_from_schema(schema: FeatureSchema) -> dict[str, int]:
    return {
        NODE_PARTY: len(schema.party_names),
        NODE_IP: len(schema.ip_names),
        NODE_ADDRESS: len(schema.address_names),
        "transfer_edge": len(schema.transfer_names),
    }


def count_positives(data: HeteroData) -> dict[str, Any]:
    store_edge = data[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
    party_y = data[NODE_PARTY].y
    mask = data[NODE_PARTY].train_mask
    return {
        "n_party": int(party_y.numel()),
        "n_party_active": int(mask.sum().item()),
        "n_party_pos": int((party_y[mask] > 0.5).sum().item()) if mask.any() else 0,
        "n_transfer": int(store_edge.y.numel()),
        "n_transfer_pos": int((store_edge.y > 0.5).sum().item()),
    }
