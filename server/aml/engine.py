"""Real-time micro-batch inference over a live simulator GraphStore."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aml.convert import graph_store_to_hetero
from aml.domain import Network, Transaction, Typology
from aml.features import FeatureConfig, FeatureSchema, default_schema
from aml.generator import SimulatorConfig
from aml.geo import haversine_km
from aml.graph_store import GraphStore
from aml.model import ScoreResult, build_model_from_checkpoint, score_snapshot
from aml.population import PopulationConfig
from aml.sources import ReplaySource, SimSource, TransactionSource, build_sim_source
from aml.train import load_checkpoint

# Max nodes/edges included in an alert neighborhood payload.
NEIGHBORHOOD_NODE_CAP = 40
NEIGHBORHOOD_EDGE_CAP = 80
PARTY_ALERT_RISE = 0.05
RECENT_TX_RING = 50


@dataclass
class EngineConfig:
    seed: int = 42
    rate_per_second: float = 8.0
    batch_size: int = 10
    max_batch_wait: float = 1.0
    warmup: int = 2_000
    threshold: float = 0.5
    pattern_event_fraction: float = 0.02
    population: PopulationConfig = field(default_factory=PopulationConfig)
    max_transfer_edges: int = 50_000
    device: str = "cpu"
    reveal_labels: bool = False
    checkpoint: Path = field(default_factory=lambda: Path("artifacts/hetero_sage.pt"))


@dataclass
class Alert:
    kind: str  # "transfer" | "party"
    subject_id: str
    score: float
    network: str
    source_id: str | None = None
    dest_id: str | None = None
    reason: str = ""
    neighborhood: dict[str, Any] = field(default_factory=dict)
    alert_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "score": self.score,
            "network": self.network,
            "source_id": self.source_id,
            "dest_id": self.dest_id,
            "reason": self.reason,
            "neighborhood": self.neighborhood,
        }


@dataclass
class BatchScores:
    batch_id: int
    infer_ms: float
    transfer_scores: dict[str, float]
    party_scores: dict[str, float]
    alerts: list[Alert]
    label_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "infer_ms": self.infer_ms,
            "transfer_scores": self.transfer_scores,
            "party_scores": self.party_scores,
            "alerts": [a.to_dict() for a in self.alerts],
            "label_metrics": self.label_metrics,
        }


@dataclass
class RecentTx:
    tx: dict[str, Any]
    transfer_risk: float | None


def _schema_names_match(saved: FeatureSchema, expected: FeatureSchema) -> bool:
    return (
        list(saved.party_names) == list(expected.party_names)
        and list(saved.ip_names) == list(expected.ip_names)
        and list(saved.address_names) == list(expected.address_names)
        and list(saved.transfer_names) == list(expected.transfer_names)
    )


def open_checkpoint(
    path: Path, device: str = "cpu"
) -> tuple[nn.Module, FeatureSchema, dict[str, Any]]:
    """Load HeteroSAGE and its feature schema from an ``aml-train`` checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "Train a model first with: aml-train --out artifacts/hetero_sage.pt"
        )
    ckpt = load_checkpoint(path)
    schema = FeatureSchema.from_dict(ckpt["feature_schema"])
    expected = default_schema()
    if not _schema_names_match(schema, expected):
        raise ValueError(
            f"checkpoint feature schema is stale: {path}\n"
            "Retrain with the current feature set:\n"
            "  aml-train --out artifacts/hetero_sage.pt"
        )
    model = build_model_from_checkpoint(ckpt)
    model.eval()
    model.to(torch.device(device))
    return model, schema, ckpt


def neighborhood_payload(
    store: GraphStore,
    node_id: str,
    *,
    k: int = 2,
    node_cap: int = NEIGHBORHOOD_NODE_CAP,
    edge_cap: int = NEIGHBORHOOD_EDGE_CAP,
) -> dict[str, Any]:
    """Capped k-hop transfer neighborhood for alert context (not model input)."""
    try:
        sub = store.neighborhood(node_id, k=k)
    except KeyError:
        return {"nodes": [], "edges": [], "center": node_id, "k": k}

    nodes: list[dict[str, Any]] = []
    for n, data in sub.nodes(data=True):
        node: dict[str, Any] = {
            "id": n,
            "network": data.get("network"),
            "party_type": data.get("party_type"),
            "tx_count": int(data.get("tx_count", 0)),
        }
        if data.get("country") is not None:
            node["country"] = data.get("country")
        if data.get("city") is not None:
            node["city"] = data.get("city")
        if data.get("lat") is not None:
            node["lat"] = float(data["lat"])
        if data.get("lng") is not None:
            node["lng"] = float(data["lng"])
        nodes.append(node)
        if len(nodes) >= node_cap:
            break

    keep = {n["id"] for n in nodes}
    edges: list[dict[str, Any]] = []
    for u, v, data in sub.edges(data=True):
        if u not in keep or v not in keep:
            continue
        edges.append(
            {
                "source": u,
                "dest": v,
                "tx_id": data.get("tx_id"),
                "amount": data.get("amount"),
                "timestamp": data.get("timestamp"),
                "channel": data.get("channel"),
                "network": data.get("network"),
            }
        )
        if len(edges) >= edge_cap:
            break
    return {"nodes": nodes, "edges": edges, "center": node_id, "k": k}


def build_reason(
    store: GraphStore,
    *,
    subject_id: str,
    score: float,
    snapshot_time: float,
    short_window: float = 120.0,
    kind: str = "party",
) -> str:
    """Template natural-language reason from graph facts in the short window."""
    start = snapshot_time - short_window
    try:
        window = store.edges_in_window(subject_id, start, snapshot_time)
    except KeyError:
        window = []
    counterparties: set[str] = set()
    amount_sum = 0.0
    for e in window:
        amount_sum += float(e.get("amount", 0.0))
        if e.get("source") == subject_id:
            counterparties.add(str(e.get("dest")))
        else:
            counterparties.add(str(e.get("source")))
    n = len(window)
    label = "Account" if kind == "party" else "Transfer involving account"
    return (
        f"Flagged High Risk ({score:.2f}): {label} {subject_id} engaged in "
        f"{n} transfer(s) totaling ${amount_sum:,.0f} across "
        f"{len(counterparties)} counterparties within {int(short_window)} seconds."
    )


def build_alerts(
    *,
    batch_txs: list[Transaction],
    transfer_scores: dict[str, float],
    party_scores: dict[str, float],
    prev_party_scores: dict[str, float],
    store: GraphStore,
    snapshot_time: float,
    threshold: float,
    short_window: float = 120.0,
    alerted_parties: set[str] | None = None,
) -> tuple[list[Alert], set[str]]:
    """Emit transfer alerts and party alerts with threshold / rise dedupe.

    Party alerts re-fire only when a party newly crosses ``threshold`` or its
    score rises by at least ``PARTY_ALERT_RISE`` after already being alerted.
    """
    alerted = set(alerted_parties or ())
    alerts: list[Alert] = []

    for tx in batch_txs:
        score = transfer_scores.get(tx.tx_id)
        if score is None or score < threshold:
            continue
        center = tx.source_id
        alerts.append(
            Alert(
                kind="transfer",
                subject_id=tx.tx_id,
                score=score,
                network=tx.network.value,
                source_id=tx.source_id,
                dest_id=tx.dest_id,
                reason=build_reason(
                    store,
                    subject_id=center,
                    score=score,
                    snapshot_time=snapshot_time,
                    short_window=short_window,
                    kind="transfer",
                ),
                neighborhood=neighborhood_payload(store, center),
            )
        )

    for pid, score in party_scores.items():
        if score < threshold:
            continue
        prev = prev_party_scores.get(pid, 0.0)
        newly_crossed = prev < threshold <= score
        rose = pid in alerted and (score - prev) >= PARTY_ALERT_RISE
        first_time = pid not in alerted and score >= threshold
        if not (newly_crossed or rose or first_time):
            continue
        network = ""
        if pid in store.g:
            network = str(store.g.nodes[pid].get("network", ""))
        alerts.append(
            Alert(
                kind="party",
                subject_id=pid,
                score=score,
                network=network,
                reason=build_reason(
                    store,
                    subject_id=pid,
                    score=score,
                    snapshot_time=snapshot_time,
                    short_window=short_window,
                    kind="party",
                ),
                neighborhood=neighborhood_payload(store, pid),
            )
        )
        alerted.add(pid)

    return alerts, alerted


def batch_label_metrics(
    txs: list[Transaction], transfer_scores: dict[str, float], threshold: float
) -> dict[str, float]:
    """Hidden-label precision/recall for transfers in this batch."""
    tp = fp = fn = tn = 0
    for tx in txs:
        score = transfer_scores.get(tx.tx_id, 0.0)
        pred = score >= threshold
        pos = tx.typology != Typology.BENIGN
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif not pred and pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
    }


class RealtimeEngine:
    """Owns a TransactionSource + HeteroSAGE; paces wire events and scored batches."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        source: TransactionSource | None = None,
        sources: dict[str, TransactionSource] | None = None,
        model: nn.Module | None = None,
        schema: FeatureSchema | None = None,
        checkpoint_meta: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if model is None or schema is None:
            model, schema, checkpoint_meta = open_checkpoint(
                config.checkpoint, device=config.device
            )
        self.model = model
        self.schema = schema
        self.checkpoint_meta = checkpoint_meta or {}
        self.feature_config = FeatureConfig(
            short_window_seconds=schema.short_window_seconds
        )

        default = source or build_sim_source(
            SimulatorConfig(
                seed=config.seed,
                rate_per_second=config.rate_per_second,
                pattern_event_fraction=config.pattern_event_fraction,
                population=config.population,
                max_transfer_edges=config.max_transfer_edges,
            )
        )
        self._sources: dict[str, TransactionSource] = dict(sources or {})
        if default.name not in self._sources:
            self._sources[default.name] = default
        self.source: TransactionSource = self._sources[default.name]
        # Backward-compatible alias for tests / callers that still touch ``.sim``.
        self.sim = self.source.sim if isinstance(self.source, SimSource) else None
        self._batch_id = 0
        self._alert_seq = 0
        self.party_scores: dict[str, float] = {}
        self.transfer_scores: dict[str, float] = {}
        self._alerted_parties: set[str] = set()
        self.recent_txs: deque[RecentTx] = deque(maxlen=RECENT_TX_RING)
        self.warmup_done = 0
        self.live_events = 0
        self._warmup_by_source: dict[str, int] = {}

    @property
    def store(self) -> GraphStore:
        return self.source.store

    @property
    def sim_time(self) -> float:
        return self.source.now

    @property
    def available_sources(self) -> list[str]:
        """Source names the UI may select, in a stable order."""
        order = ("sim", "replay")
        names = [name for name in order if name in self._sources]
        for name in sorted(self._sources):
            if name not in names:
                names.append(name)
        return names

    def warmup(self, n: int | None = None) -> int:
        """Drain ``n`` events with no scoring so the live graph resembles training."""
        n = self.config.warmup if n is None else n
        if n <= 0:
            self._warmup_by_source[self.source.name] = 0
            self.warmup_done = 0
            return 0
        produced = self.source.warmup(n)
        self._warmup_by_source[self.source.name] = produced
        self.warmup_done = produced
        return produced

    def warmup_all(self, n: int | None = None) -> dict[str, int]:
        """Warm every registered source; keep the originally active source selected."""
        n = self.config.warmup if n is None else n
        active = self.source.name
        counts: dict[str, int] = {}
        for name, src in self._sources.items():
            self.source = src
            self.sim = src.sim if isinstance(src, SimSource) else None
            counts[name] = self.warmup(n)
        self.set_source(active)
        return counts

    def set_source(self, name: str) -> bool:
        """Activate a registered source and clear live scoring state.

        Returns False when ``name`` is unknown or already active.
        """
        if name not in self._sources:
            return False
        if self.source.name == name:
            return False
        self.source = self._sources[name]
        self.sim = self.source.sim if isinstance(self.source, SimSource) else None
        self.warmup_done = self._warmup_by_source.get(name, 0)
        self.party_scores = {}
        self.transfer_scores = {}
        self._alerted_parties = set()
        self.recent_txs.clear()
        self.live_events = 0
        self._batch_id = 0
        self._alert_seq = 0
        return True

    def next_transaction(self) -> Transaction:
        """Pull the next applied transaction from the configured source."""
        tx = self.source.next_transaction()
        self.live_events += 1
        payload = self.tx_wire_dict(tx, transfer_risk=None)
        self.recent_txs.append(RecentTx(tx=payload, transfer_risk=None))
        return tx

    def _party_geo(self, party_id: str) -> dict[str, Any]:
        """Lat/lng/city/country for a party node, or empty if missing."""
        if party_id not in self.store.g:
            return {}
        node = self.store.g.nodes[party_id]
        out: dict[str, Any] = {}
        if node.get("country") is not None:
            out["country"] = node.get("country")
        if node.get("city") is not None:
            out["city"] = node.get("city")
        if node.get("lat") is not None:
            out["lat"] = float(node["lat"])
        if node.get("lng") is not None:
            out["lng"] = float(node["lng"])
        return out

    def tx_wire_dict(
        self, tx: Transaction, *, transfer_risk: float | None
    ) -> dict[str, Any]:
        src_geo = self._party_geo(tx.source_id)
        dst_geo = self._party_geo(tx.dest_id)
        data: dict[str, Any] = {
            "tx_id": tx.tx_id,
            "timestamp": tx.timestamp,
            "source_id": tx.source_id,
            "dest_id": tx.dest_id,
            "amount": tx.amount,
            "currency": tx.currency.value,
            "channel": tx.channel.value,
            "network": tx.network.value,
            "transfer_risk": transfer_risk,
        }
        if "lat" in src_geo:
            data["source_lat"] = src_geo["lat"]
        if "lng" in src_geo:
            data["source_lng"] = src_geo["lng"]
        if "city" in src_geo:
            data["source_city"] = src_geo["city"]
        if "country" in src_geo:
            data["source_country"] = src_geo["country"]
        if "lat" in dst_geo:
            data["dest_lat"] = dst_geo["lat"]
        if "lng" in dst_geo:
            data["dest_lng"] = dst_geo["lng"]
        if "city" in dst_geo:
            data["dest_city"] = dst_geo["city"]
        if "country" in dst_geo:
            data["dest_country"] = dst_geo["country"]
        if "lat" in src_geo and "lng" in src_geo and "lat" in dst_geo and "lng" in dst_geo:
            data["distance_km"] = haversine_km(
                src_geo["lat"], src_geo["lng"], dst_geo["lat"], dst_geo["lng"]
            )
        if self.config.reveal_labels:
            data["typology"] = tx.typology.value
            data["pattern_id"] = tx.pattern_id
        return data

    def score_batch(self, txs: list[Transaction]) -> BatchScores:
        """Score the full retained graph; return scores for ``txs`` endpoints only."""
        if not txs:
            return BatchScores(
                batch_id=self._batch_id,
                infer_ms=0.0,
                transfer_scores={},
                party_scores={},
                alerts=[],
                label_metrics={
                    "tp": 0.0,
                    "fp": 0.0,
                    "fn": 0.0,
                    "tn": 0.0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                },
            )

        t0 = time.perf_counter()
        data = graph_store_to_hetero(
            self.store,
            snapshot_time=self.sim_time,
            config=self.feature_config,
            schema=self.schema,
            normalize=True,
        )
        scores: ScoreResult = score_snapshot(self.model, data, device=self.device)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        self._batch_id += 1
        batch_id = self._batch_id

        tx_ids = {tx.tx_id for tx in txs}
        transfer_out = {
            tid: float(scores.transfer_scores[tid])
            for tid in tx_ids
            if tid in scores.transfer_scores
        }
        touched: set[str] = set()
        for tx in txs:
            touched.add(tx.source_id)
            touched.add(tx.dest_id)
        party_out = {
            pid: float(scores.party_scores[pid])
            for pid in touched
            if pid in scores.party_scores
        }

        prev_party = dict(self.party_scores)
        # Refresh full party cache from this forward (compact for snapshot).
        self.party_scores = {
            pid: float(s) for pid, s in scores.party_scores.items()
        }
        self.transfer_scores.update(transfer_out)

        # Patch recent ring with transfer risks.
        risk_by_id = transfer_out
        for recent in self.recent_txs:
            tid = recent.tx.get("tx_id")
            if tid in risk_by_id:
                recent.transfer_risk = risk_by_id[tid]
                recent.tx["transfer_risk"] = risk_by_id[tid]

        alerts, self._alerted_parties = build_alerts(
            batch_txs=txs,
            transfer_scores=transfer_out,
            party_scores=party_out,
            prev_party_scores=prev_party,
            store=self.store,
            snapshot_time=self.sim_time,
            threshold=self.config.threshold,
            short_window=self.schema.short_window_seconds,
            alerted_parties=self._alerted_parties,
        )
        self._stamp_alerts(alerts)
        metrics = batch_label_metrics(txs, transfer_out, self.config.threshold)

        return BatchScores(
            batch_id=batch_id,
            infer_ms=infer_ms,
            transfer_scores=transfer_out,
            party_scores=party_out,
            alerts=alerts,
            label_metrics=metrics,
        )

    def _stamp_alerts(self, alerts: list[Alert]) -> None:
        """Assign session-unique ids so analysts can disposition each alert."""
        for alert in alerts:
            self._alert_seq += 1
            alert.alert_id = f"a{self._alert_seq}"

    def hello_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "hello",
            "seed": self.config.seed,
            "rate": self.config.rate_per_second,
            "checkpoint": str(self.config.checkpoint),
            "networks": [n.value for n in Network],
            "sim_time": self.sim_time,
            "warmup_done": self.warmup_done,
            "threshold": self.config.threshold,
            "reveal_labels": self.config.reveal_labels,
            "source": self.source.name,
            "sources": self.available_sources,
        }
        replay = self._sources.get("replay")
        if isinstance(replay, ReplaySource):
            data["replay"] = replay.replay_path
        elif isinstance(self.source, ReplaySource):
            data["replay"] = self.source.replay_path
        return data

    def snapshot_dict(self) -> dict[str, Any]:
        recent = []
        for r in self.recent_txs:
            row = dict(r.tx)
            row["transfer_risk"] = r.transfer_risk
            recent.append(row)
        return {
            "type": "snapshot",
            "sim_time": self.sim_time,
            "party_scores": dict(self.party_scores),
            "recent_transactions": recent,
            "stats": self.store.snapshot_stats(),
        }
