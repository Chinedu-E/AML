"""Transaction sources for the live inference engine (sim or JSONL replay)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aml.clock import SimRng
from aml.domain import (
    Channel,
    Currency,
    Network,
    NodeKind,
    Party,
    PartyType,
    Transaction,
    Typology,
)
from aml.generator import Simulator, SimulatorConfig
from aml.geo import Location, assign_location
from aml.graph_store import GraphStore
from aml.population import COUNTRIES

# Opening balance large enough that replay rarely stalls on first sight.
REPLAY_OPENING_BALANCE = 1_000_000.0
REPLAY_TOP_UP_MULTIPLIER = 5.0


class TransactionSource(Protocol):
    """Produces applied transactions on a shared GraphStore."""

    @property
    def store(self) -> GraphStore: ...

    @property
    def now(self) -> float: ...

    @property
    def name(self) -> str: ...

    def warmup(self, n: int) -> int: ...

    def next_transaction(self) -> Transaction: ...


@dataclass
class SimSource:
    """Live synthetic generator wrapping ``Simulator``."""

    sim: Simulator
    _stream: Iterator[Transaction] | None = None

    @property
    def store(self) -> GraphStore:
        return self.sim.store

    @property
    def now(self) -> float:
        return self.sim.clock.now

    @property
    def name(self) -> str:
        return "sim"

    def warmup(self, n: int) -> int:
        if n <= 0:
            return 0
        produced = 0
        for _tx in self.sim.run(n):
            produced += 1
        self._stream = None
        return produced

    def next_transaction(self) -> Transaction:
        if self._stream is None:
            self._stream = self.sim.run(10**12)
        return next(self._stream)


def build_sim_source(config: SimulatorConfig) -> SimSource:
    return SimSource(sim=Simulator(config))


def parse_transaction_row(row: dict[str, Any], *, tx_id: str | None = None) -> Transaction:
    """Build a ``Transaction`` from an ``aml-sim --jsonl`` row."""
    typology_raw = row.get("typology", Typology.BENIGN.value)
    pattern_id = row.get("pattern_id")
    if pattern_id is not None and not isinstance(pattern_id, str):
        pattern_id = str(pattern_id) if pattern_id else None
    return Transaction(
        tx_id=tx_id if tx_id is not None else str(row["tx_id"]),
        timestamp=float(row["timestamp"]),
        source_id=str(row["source_id"]),
        dest_id=str(row["dest_id"]),
        amount=float(row["amount"]),
        currency=Currency(str(row["currency"])),
        channel=Channel(str(row["channel"])),
        network=Network(str(row["network"])),
        typology=Typology(str(typology_raw)),
        pattern_id=pattern_id,
    )


def _stable_location(party_id: str) -> Location:
    """Deterministic country/geo for a party id (no RNG drift across runs)."""
    digest = 0
    for ch in party_id:
        digest = (digest * 31 + ord(ch)) & 0xFFFFFFFF
    country = COUNTRIES[digest % len(COUNTRIES)]
    # Seeded jitter from party id so co-located accounts do not stack.
    rng = SimRng(seed=digest % (2**31 - 1) or 1)
    return assign_location(rng, country)


def _infer_party_type(party_id: str, network: Network) -> PartyType:
    lower = party_id.lower()
    if "mixer" in lower:
        return PartyType.MIXER
    if "contract" in lower:
        return PartyType.SMART_CONTRACT
    if "shell" in lower:
        return PartyType.SHELL
    if "wallet" in lower:
        return PartyType.WALLET
    if network == Network.CRYPTO:
        return PartyType.WALLET
    if network == Network.CROSS_BORDER and "trade" in lower:
        return PartyType.BUSINESS
    if "bank" in lower or "onramp" in lower:
        return PartyType.INDIVIDUAL
    if network == Network.BANKING:
        return PartyType.INDIVIDUAL
    return PartyType.BUSINESS


class ReplaySource:
    """Replay ``aml-sim`` JSONL, auto-creating parties and looping when exhausted."""

    def __init__(
        self,
        path: Path,
        *,
        max_transfer_edges: int = 50_000,
        opening_balance: float = REPLAY_OPENING_BALANCE,
        loop: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"replay file not found: {self.path}")
        self._rows = self._load_rows(self.path)
        if not self._rows:
            raise ValueError(f"replay file is empty: {self.path}")
        self._store = GraphStore(max_transfer_edges=max_transfer_edges)
        self._opening_balance = float(opening_balance)
        self._allow_loop = bool(loop)
        self._index = 0
        self._loop_count = 0
        self._now = 0.0

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on line {line_no} of {path}: {exc}"
                    ) from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"JSONL row on line {line_no} must be an object")
                rows.append(raw)
        return rows

    @property
    def store(self) -> GraphStore:
        return self._store

    @property
    def now(self) -> float:
        return self._now

    @property
    def name(self) -> str:
        return "replay"

    @property
    def replay_path(self) -> str:
        return str(self.path)

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    @property
    def exhausted(self) -> bool:
        return (not self._allow_loop) and self._index >= len(self._rows)

    def warmup(self, n: int) -> int:
        if n <= 0:
            return 0
        produced = 0
        for _ in range(n):
            self.next_transaction()
            produced += 1
        return produced

    def next_transaction(self) -> Transaction:
        if self._index >= len(self._rows):
            if not self._allow_loop:
                raise StopIteration("replay JSONL exhausted (loop=False)")
            self._index = 0
            self._loop_count += 1
        row = self._rows[self._index]
        self._index += 1
        base_id = str(row["tx_id"])
        tx_id = (
            base_id if self._loop_count == 0 else f"{base_id}:{self._loop_count}"
        )
        tx = parse_transaction_row(row, tx_id=tx_id)
        self._ensure_party(tx.source_id, tx.network, tx.timestamp)
        self._ensure_party(tx.dest_id, tx.network, tx.timestamp)
        self._ensure_funds(tx.source_id, tx.amount)
        if not self._store.apply(tx):
            # Should not happen with loop suffixes; skip by advancing again.
            return self.next_transaction()
        self._now = float(tx.timestamp)
        return tx

    def _ensure_party(self, party_id: str, network: Network, opened_at: float) -> None:
        if party_id in self._store.g:
            node = self._store.g.nodes[party_id]
            if node.get("kind") == NodeKind.PARTY.value:
                return
        loc = _stable_location(party_id)
        party = Party(
            party_id=party_id,
            party_type=_infer_party_type(party_id, network),
            network=network,
            country=loc.country,
            opened_at=float(opened_at),
            city=loc.city,
            lat=loc.lat,
            lng=loc.lng,
        )
        self._store.add_party(party, balance=self._opening_balance)

    def _ensure_funds(self, party_id: str, amount: float) -> None:
        bal = self._store.balance(party_id)
        if bal >= amount:
            return
        needed = max(amount * REPLAY_TOP_UP_MULTIPLIER, self._opening_balance)
        self._store.set_balance(party_id, needed)


def build_replay_source(
    path: Path,
    *,
    max_transfer_edges: int = 50_000,
    loop: bool = True,
) -> ReplaySource:
    return ReplaySource(path, max_transfer_edges=max_transfer_edges, loop=loop)
