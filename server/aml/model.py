"""Heterogeneous GraphSAGE encoder with party and transfer risk heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from aml.convert import (
    NODE_ADDRESS,
    NODE_IP,
    NODE_PARTY,
    REL_TRANSFER,
    feature_dims_from_schema,
    hetero_metadata,
)
from aml.features import FeatureSchema, default_schema


@dataclass
class ModelConfig:
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    party_in: int = 0
    ip_in: int = 0
    address_in: int = 0
    transfer_edge_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_schema(
        cls,
        schema: FeatureSchema,
        *,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> ModelConfig:
        dims = feature_dims_from_schema(schema)
        return cls(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            party_in=dims[NODE_PARTY],
            ip_in=dims[NODE_IP],
            address_in=dims[NODE_ADDRESS],
            transfer_edge_dim=dims["transfer_edge"],
        )


class HeteroSAGE(nn.Module):
    """2-layer HeteroConv SAGE with party node head and transfer edge head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_channels
        self.dropout = config.dropout

        self.lin_dict = nn.ModuleDict(
            {
                NODE_PARTY: nn.Linear(config.party_in, h),
                NODE_IP: nn.Linear(config.ip_in, h),
                NODE_ADDRESS: nn.Linear(config.address_in, h),
            }
        )

        _node_types, edge_types = hetero_metadata()
        self.convs = nn.ModuleList()
        for _ in range(config.num_layers):
            conv = HeteroConv(
                {
                    etype: SAGEConv(h, h)
                    for etype in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.node_head = nn.Linear(h, 1)
        edge_in = 2 * h + config.transfer_edge_dim
        self.edge_head = nn.Sequential(
            nn.Linear(edge_in, h),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, 1),
        )

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {
            NODE_PARTY: self.lin_dict[NODE_PARTY](data[NODE_PARTY].x),
            NODE_IP: self.lin_dict[NODE_IP](data[NODE_IP].x),
            NODE_ADDRESS: self.lin_dict[NODE_ADDRESS](data[NODE_ADDRESS].x),
        }
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in data.edge_types
            if data[etype].edge_index.numel() > 0
            or data[etype].edge_index.size(-1) == 0
        }
        # Always include all declared edge types (empty is fine for HeteroConv).
        _node_types, edge_types = hetero_metadata()
        for etype in edge_types:
            if etype not in edge_index_dict:
                edge_index_dict[etype] = torch.zeros(
                    (2, 0), dtype=torch.long, device=data[NODE_PARTY].x.device
                )

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                k: F.relu(F.dropout(v, p=self.dropout, training=self.training))
                for k, v in x_dict.items()
            }
        return x_dict

    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (party_logits [N], transfer_logits [E])."""
        x_dict = self.encode(data)
        party_h = x_dict[NODE_PARTY]
        node_logits = self.node_head(party_h).view(-1)

        store = data[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
        ei = store.edge_index
        if ei.numel() == 0:
            edge_logits = torch.zeros(0, device=party_h.device)
            return node_logits, edge_logits

        src, dst = ei[0], ei[1]
        edge_attr = store.edge_attr
        edge_in = torch.cat([party_h[src], party_h[dst], edge_attr], dim=-1)
        edge_logits = self.edge_head(edge_in).view(-1)
        return node_logits, edge_logits


class FeatureOnlyBaseline(nn.Module):
    """Linear party-feature classifier (no message passing)."""

    def __init__(self, party_in: int) -> None:
        super().__init__()
        self.node_head = nn.Linear(party_in, 1)
        # Dummy edge head so training loop can share code paths.
        self.edge_head = nn.Sequential(
            nn.Linear(party_in * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.party_in = party_in

    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        party_x = data[NODE_PARTY].x
        node_logits = self.node_head(party_x).view(-1)
        store = data[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
        ei = store.edge_index
        if ei.numel() == 0:
            return node_logits, torch.zeros(0, device=party_x.device)
        src, dst = ei[0], ei[1]
        # Use mean of edge_attr dims as a scalar so edge head stays simple.
        edge_scalar = store.edge_attr.mean(dim=-1, keepdim=True)
        edge_in = torch.cat([party_x[src], party_x[dst], edge_scalar], dim=-1)
        edge_logits = self.edge_head(edge_in).view(-1)
        return node_logits, edge_logits


@dataclass
class ScoreResult:
    party_scores: dict[str, float]
    transfer_scores: dict[str, float]


@torch.no_grad()
def score_snapshot(
    model: nn.Module,
    data: HeteroData,
    *,
    device: torch.device | None = None,
) -> ScoreResult:
    """Apply sigmoid and return risk scores keyed by party id / tx id."""
    model.eval()
    if device is not None:
        data = data.to(device)
        model = model.to(device)
    node_logits, edge_logits = model(data)
    party_probs = torch.sigmoid(node_logits).cpu().tolist()
    edge_probs = torch.sigmoid(edge_logits).cpu().tolist()

    party_ids: list[str] = list(data[NODE_PARTY].node_ids)
    tx_ids: list[str] = list(
        data[NODE_PARTY, REL_TRANSFER, NODE_PARTY].tx_ids
    )
    return ScoreResult(
        party_scores={pid: float(p) for pid, p in zip(party_ids, party_probs, strict=True)},
        transfer_scores={
            tid: float(p) for tid, p in zip(tx_ids, edge_probs, strict=True)
        },
    )


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> HeteroSAGE:
    """Reload HeteroSAGE from an ``aml-train`` checkpoint dict."""
    schema = FeatureSchema.from_dict(ckpt["feature_schema"])
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    # Prefer dims from schema if config was saved without them.
    if cfg.party_in <= 0:
        cfg = ModelConfig.from_schema(schema)
    model = HeteroSAGE(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model


def empty_model_for_schema(schema: FeatureSchema | None = None) -> HeteroSAGE:
    sch = schema or default_schema()
    return HeteroSAGE(ModelConfig.from_schema(sch))
