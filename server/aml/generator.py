"""Benign Poisson transaction process and typology scheduler."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from aml.clock import SimClock, SimRng
from aml.domain import (
    NETWORK_CHANNELS,
    NETWORK_CURRENCIES,
    Channel,
    Currency,
    Network,
    Transaction,
    Typology,
)
from aml.graph_store import GraphStore
from aml.population import Population, PopulationConfig, seed_population
from aml.typologies import (
    INJECTORS,
    InjectionContext,
    pick_typology,
)

# Channel-specific lognormal params (mu, sigma) for amount draws.
CHANNEL_AMOUNT: dict[Channel, tuple[float, float]] = {
    Channel.WIRE: (8.5, 0.6),
    Channel.ACH: (7.8, 0.5),
    Channel.PAYROLL: (8.2, 0.3),
    Channel.CARD: (4.5, 0.8),
    Channel.TRANSFER: (6.5, 0.9),
    Channel.CONTRACT: (7.0, 0.7),
    Channel.MIXER: (7.2, 0.6),
    Channel.INVOICE: (9.0, 0.5),
}

WIRE_FEE_RATE = 0.001  # 0.1% fee on wires (deducted from sender, not recepient)


@dataclass
class SimulatorConfig:
    seed: int = 42
    rate_per_second: float = 7.0
    pattern_event_fraction: float = 0.02
    population: PopulationConfig = field(default_factory=PopulationConfig)
    max_transfer_edges: int = 50_000


class Simulator:
    """Yields labeled transactions and updates an in-memory graph."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.rng = SimRng(seed=self.config.seed)
        self.clock = SimClock(now=0.0)
        self.store = GraphStore(max_transfer_edges=self.config.max_transfer_edges)
        self.population = seed_population(
            self.store, self.rng, self.config.population, opened_at=0.0
        )
        self._tx_seq = 0
        self._reserved: set[str] = set()
        self._pending_pattern: list[Transaction] = []

    def next_tx_id(self) -> str:
        self._tx_seq += 1
        return f"tx-{self._tx_seq:08d}"

    def run(self, n_events: int) -> Iterator[Transaction]:
        """Yield up to ``n_events`` applied transactions."""
        produced = 0
        while produced < n_events:
            # Drain an in-flight pattern first.
            if self._pending_pattern:
                tx = self._pending_pattern.pop(0)
                if not self._pending_pattern:
                    # Release participants when the pattern completes.
                    self._reserved.clear()
                if self._try_apply(tx):
                    produced += 1
                    yield tx
                continue

            # Decide whether to inject a pattern or emit a benign transfer.
            inject = self.rng.random() < self.config.pattern_event_fraction
            if inject:
                txs = self._try_inject()
                if txs:
                    self._pending_pattern = list(txs)
                    continue

            tx = self._benign_transaction()
            if tx is None:
                # Advance time a bit and retry.
                self.clock.advance(
                    self.rng.poisson_interarrival(self.config.rate_per_second)
                )
                continue
            if self._try_apply(tx):
                produced += 1
                yield tx

    def collect(self, n_events: int) -> list[Transaction]:
        return list(self.run(n_events))

    def _try_apply(self, tx: Transaction) -> bool:
        try:
            return self.store.apply(tx)
        except ValueError:
            return False

    def _try_inject(self) -> list[Transaction]:
        typology = pick_typology(self.rng)
        injector = INJECTORS[typology]
        ctx = InjectionContext(
            store=self.store,
            population=self.population,
            rng=self.rng,
            clock=self.clock,
            reserved=self._reserved,
            next_tx_id=self.next_tx_id,
        )
        before_reserved = set(self._reserved)
        txs = injector(ctx)
        if not txs:
            # Ensure failed injection does not leave reservations behind.
            self._reserved.clear()
            self._reserved.update(before_reserved)
            return []
        return txs

    def _benign_transaction(self) -> Transaction | None:
        dt = self.rng.poisson_interarrival(self.config.rate_per_second)
        ts = self.clock.advance(dt)

        networks = [Network.BANKING, Network.CRYPTO, Network.CROSS_BORDER]
        # Weight toward banking traffic.
        weights = [0.55, 0.30, 0.15]
        r = self.rng.random()
        cumulative = 0.0
        network = Network.BANKING
        for n, w in zip(networks, weights, strict=True):
            cumulative += w
            if r <= cumulative:
                network = n
                break

        ids = [
            pid
            for pid in self.population.party_ids_in(network)
            if pid not in self._reserved
        ]
        if len(ids) < 2:
            return None

        source = self.rng.choice(ids)
        preferred = [
            p
            for p in self.population.preferred.get(source, [])
            if p in ids and p != source
        ]
        if preferred and self.rng.random() < 0.7:
            dest = preferred[self.rng.zipf_index(len(preferred))]
        else:
            others = [pid for pid in ids if pid != source]
            dest = self.rng.choice(others)

        channel = self.rng.choice(NETWORK_CHANNELS[network])
        currency = self.rng.choice(NETWORK_CURRENCIES[network])
        mu, sigma = CHANNEL_AMOUNT[channel]
        amount = self.rng.round_amount(self.rng.lognormal(mu, sigma))
        # Round payroll/invoices more aggressively.
        if channel in (Channel.PAYROLL, Channel.INVOICE):
            amount = float(round(amount / 100.0) * 100.0)
            amount = max(amount, 100.0)

        fee = 0.0
        if channel == Channel.WIRE:
            fee = self.rng.round_amount(amount * WIRE_FEE_RATE)

        needed = amount + fee
        bal = self.store.balance(source)
        if bal < needed:
            # Top up occasionally so the stream does not stall.
            if self.rng.random() < 0.3:
                self.store.set_balance(source, needed * self.rng.uniform(2.0, 5.0))
            else:
                return None

        # Deduct fee from source without creating a transfer edge.
        if fee > 0:
            self.store.set_balance(source, self.store.balance(source) - fee)

        return Transaction(
            tx_id=self.next_tx_id(),
            timestamp=ts,
            source_id=source,
            dest_id=dest,
            amount=amount,
            currency=currency,
            channel=channel,
            network=network,
            typology=Typology.BENIGN,
            pattern_id=None,
        )
