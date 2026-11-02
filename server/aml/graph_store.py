"""In-memory dynamic MultiDiGraph for AML parties and transfers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from aml.domain import (
    EdgeKind,
    Network,
    NodeKind,
    Party,
    Transaction,
    Typology,
)


class DuplicateTransactionError(ValueError):
    """Raised when a transaction id is applied more than once."""


class GraphStore:
    """NetworkX-backed dynamic graph with retention and neighborhood queries."""

    def __init__(self, max_transfer_edges: int = 50_000) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()
        self.max_transfer_edges = max_transfer_edges
        self._tx_ids: set[str] = set()
        # Ordered list of (timestamp, edge_key) for transfer retention.
        self._transfer_order: list[tuple[float, tuple[str, str, str]]] = []
        self._on_ramp_links: dict[str, str] = {}

    # --- seeding helpers -------------------------------------------------

    def add_party(self, party: Party, balance: float) -> None:
        self.g.add_node(
            party.party_id,
            kind=NodeKind.PARTY.value,
            party_type=party.party_type.value,
            network=party.network.value,
            country=party.country,
            city=party.city,
            lat=float(party.lat),
            lng=float(party.lng),
            opened_at=party.opened_at,
            is_on_ramp=party.is_on_ramp,
            balance=float(balance),
            tx_count=0,
            last_seen=party.opened_at,
            pattern_ids=[],
        )

    def add_identity_node(
        self, node_id: str, kind: NodeKind, attrs: dict[str, Any] | None = None
    ) -> None:
        data = {"kind": kind.value}
        if attrs:
            data.update(attrs)
        self.g.add_node(node_id, **data)

    def add_identity_edge(
        self, party_id: str, identity_id: str, kind: EdgeKind
    ) -> None:
        self.g.add_edge(
            party_id,
            identity_id,
            key=f"{kind.value}:{party_id}:{identity_id}",
            edge_kind=kind.value,
        )

    def link_on_ramp(self, bank_id: str, wallet_id: str) -> None:
        self._on_ramp_links[bank_id] = wallet_id
        self._on_ramp_links[wallet_id] = bank_id
        self.g.add_edge(
            bank_id,
            wallet_id,
            key=f"on_ramp:{bank_id}:{wallet_id}",
            edge_kind="on_ramp",
        )

    def on_ramp_partner(self, party_id: str) -> str | None:
        return self._on_ramp_links.get(party_id)

    # --- mutations -------------------------------------------------------

    def apply(self, tx: Transaction) -> bool:
        """Apply a transfer. Returns False if ``tx_id`` was already seen."""
        if tx.tx_id in self._tx_ids:
            return False

        for pid in (tx.source_id, tx.dest_id):
            if pid not in self.g or self.g.nodes[pid].get("kind") != NodeKind.PARTY.value:
                raise KeyError(f"unknown party: {pid}")

        src = self.g.nodes[tx.source_id]
        dst = self.g.nodes[tx.dest_id]
        if src["balance"] < tx.amount:
            raise ValueError(
                f"insufficient balance on {tx.source_id}: "
                f"{src['balance']} < {tx.amount}"
            )

        src["balance"] = float(src["balance"]) - float(tx.amount)
        dst["balance"] = float(dst["balance"]) + float(tx.amount)
        for node in (src, dst):
            node["tx_count"] = int(node["tx_count"]) + 1
            node["last_seen"] = max(float(node["last_seen"]), float(tx.timestamp))

        if tx.typology != Typology.BENIGN and tx.pattern_id:
            for node in (src, dst):
                pids: list[str] = list(node.get("pattern_ids") or [])
                if tx.pattern_id not in pids:
                    pids.append(tx.pattern_id)
                node["pattern_ids"] = pids

        edge_key = tx.tx_id
        self.g.add_edge(
            tx.source_id,
            tx.dest_id,
            key=edge_key,
            edge_kind=EdgeKind.TRANSFER.value,
            tx_id=tx.tx_id,
            timestamp=float(tx.timestamp),
            amount=float(tx.amount),
            currency=tx.currency.value,
            channel=tx.channel.value,
            network=tx.network.value,
            typology=tx.typology.value,
            pattern_id=tx.pattern_id,
        )
        self._tx_ids.add(tx.tx_id)
        self._transfer_order.append(
            (float(tx.timestamp), (tx.source_id, tx.dest_id, edge_key))
        )
        self._enforce_retention()
        return True

    def _enforce_retention(self) -> None:
        while len(self._transfer_order) > self.max_transfer_edges:
            _, (u, v, key) = self._transfer_order.pop(0)
            if self.g.has_edge(u, v, key=key):
                data = self.g.edges[u, v, key]
                tx_id = data.get("tx_id")
                self.g.remove_edge(u, v, key=key)
                if tx_id in self._tx_ids:
                    self._tx_ids.discard(tx_id)

    # --- queries ---------------------------------------------------------

    def balance(self, party_id: str) -> float:
        return float(self.g.nodes[party_id]["balance"])

    def set_balance(self, party_id: str, balance: float) -> None:
        self.g.nodes[party_id]["balance"] = float(balance)

    def tx_count(self, party_id: str) -> int:
        return int(self.g.nodes[party_id]["tx_count"])

    def has_tx(self, tx_id: str) -> bool:
        return tx_id in self._tx_ids

    def transfer_edge_count(self) -> int:
        return len(self._transfer_order)

    def identity_edge_count(self) -> int:
        count = 0
        for _, _, data in self.g.edges(data=True):
            kind = data.get("edge_kind")
            if kind in (EdgeKind.HAS_IP.value, EdgeKind.HAS_ADDRESS.value, "on_ramp"):
                count += 1
        return count

    def neighborhood(self, node_id: str, k: int = 1) -> nx.MultiDiGraph:
        """Induced k-hop subgraph over transfer edges only."""
        if node_id not in self.g:
            raise KeyError(f"unknown node: {node_id}")
        if k < 0:
            raise ValueError("k must be non-negative")

        transfer = self._transfer_view()
        if node_id not in transfer:
            # Isolated party with no transfers yet.
            sub = nx.MultiDiGraph()
            sub.add_node(node_id, **dict(self.g.nodes[node_id]))
            return sub

        nodes: set[str] = {node_id}
        frontier = {node_id}
        for _ in range(k):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(transfer.successors(n))
                nxt.update(transfer.predecessors(n))
            nxt -= nodes
            nodes |= nxt
            frontier = nxt
            if not frontier:
                break
        return transfer.subgraph(nodes).copy()

    def edges_in_window(
        self, node_id: str, start: float, end: float
    ) -> list[dict[str, Any]]:
        """Transfer edges incident to ``node_id`` with timestamp in [start, end]."""
        if start > end:
            raise ValueError("start must be <= end")
        if node_id not in self.g:
            raise KeyError(f"unknown node: {node_id}")

        results: list[dict[str, Any]] = []
        for u, v, key, data in self.g.out_edges(node_id, keys=True, data=True):
            if data.get("edge_kind") != EdgeKind.TRANSFER.value:
                continue
            ts = float(data["timestamp"])
            if start <= ts <= end:
                results.append({"source": u, "dest": v, "key": key, **data})
        for u, v, key, data in self.g.in_edges(node_id, keys=True, data=True):
            if data.get("edge_kind") != EdgeKind.TRANSFER.value:
                continue
            ts = float(data["timestamp"])
            if start <= ts <= end:
                results.append({"source": u, "dest": v, "key": key, **data})
        results.sort(key=lambda e: float(e["timestamp"]))
        return results

    def snapshot_stats(self) -> dict[str, Any]:
        nodes_by_network: dict[str, int] = defaultdict(int)
        nodes_by_kind: dict[str, int] = defaultdict(int)
        edges_by_typology: dict[str, int] = defaultdict(int)
        transfer_edges = 0
        identity_edges = 0

        for _, data in self.g.nodes(data=True):
            kind = data.get("kind", "unknown")
            nodes_by_kind[kind] += 1
            if kind == NodeKind.PARTY.value:
                nodes_by_network[data.get("network", "unknown")] += 1

        for _, _, data in self.g.edges(data=True):
            kind = data.get("edge_kind")
            if kind == EdgeKind.TRANSFER.value:
                transfer_edges += 1
                edges_by_typology[data.get("typology", "unknown")] += 1
            else:
                identity_edges += 1

        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "transfer_edges": transfer_edges,
            "identity_edges": identity_edges,
            "nodes_by_network": dict(nodes_by_network),
            "nodes_by_kind": dict(nodes_by_kind),
            "edges_by_typology": dict(edges_by_typology),
        }

    def parties_with_ip(self, party_id: str) -> list[str]:
        """Return IP node ids linked from this party."""
        ips: list[str] = []
        for _, v, data in self.g.out_edges(party_id, data=True):
            if data.get("edge_kind") == EdgeKind.HAS_IP.value:
                ips.append(v)
        return ips

    def parties_with_address(self, party_id: str) -> list[str]:
        addrs: list[str] = []
        for _, v, data in self.g.out_edges(party_id, data=True):
            if data.get("edge_kind") == EdgeKind.HAS_ADDRESS.value:
                addrs.append(v)
        return addrs

    def _transfer_view(self) -> nx.MultiDiGraph:
        edges = [
            (u, v, k, d)
            for u, v, k, d in self.g.edges(keys=True, data=True)
            if d.get("edge_kind") == EdgeKind.TRANSFER.value
        ]
        view = nx.MultiDiGraph()
        for n, data in self.g.nodes(data=True):
            if data.get("kind") == NodeKind.PARTY.value:
                view.add_node(n, **data)
        for u, v, k, d in edges:
            view.add_edge(u, v, key=k, **d)
        return view
