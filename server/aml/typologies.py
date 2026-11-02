"""Money-laundering typology injectors with ground-truth labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from aml.clock import SimClock, SimRng
from aml.domain import (
    Channel,
    Currency,
    Network,
    Transaction,
    Typology,
)
from aml.graph_store import GraphStore
from aml.population import Population


@dataclass
class InjectionContext:
    store: GraphStore
    population: Population
    rng: SimRng
    clock: SimClock
    reserved: set[str]
    next_tx_id: Callable[[], str]


def _new_pattern_id(rng: SimRng) -> str:
    # Deterministic-ish id from rng state + uuid suffix avoided; use rng counters.
    return f"pat-{rng.randint(0, 1_000_000_000):010d}"


def _eligible_ids(
    pop: Population,
    network: Network,
    reserved: set[str],
    *,
    exclude_types: set[str] | None = None,
) -> list[str]:
    ids: list[str] = []
    for pid in pop.party_ids_in(network):
        if pid in reserved:
            continue
        party = pop.parties[pid]
        if exclude_types and party.party_type.value in exclude_types:
            continue
        ids.append(pid)
    return ids


def inject_structuring(ctx: InjectionContext) -> list[Transaction]:
    """Fan-out: one banking source sends 8–12 sub-threshold wires."""
    network = Network.BANKING
    candidates = _eligible_ids(ctx.population, network, ctx.reserved)
    if len(candidates) < 9:
        return []

    n_legs = ctx.rng.randint(8, 12)
    # Prefer a funded source.
    funded = [pid for pid in candidates if ctx.store.balance(pid) >= 100_000.0]
    if not funded:
        # Top up the richest candidate so injection can proceed in tests/sims.
        source = max(candidates, key=ctx.store.balance)
        if ctx.store.balance(source) < 100_000.0:
            ctx.store.set_balance(source, 120_000.0)
    else:
        source = ctx.rng.choice(funded)

    dest_pool = [pid for pid in candidates if pid != source]
    # Prefer low-activity destinations.
    dest_pool.sort(key=lambda pid: ctx.store.tx_count(pid))
    if len(dest_pool) < n_legs:
        return []
    destinations = dest_pool[:n_legs]

    pattern_id = _new_pattern_id(ctx.rng)
    participants = {source, *destinations}
    ctx.reserved.update(participants)

    txs: list[Transaction] = []
    base_ts = ctx.clock.now
    # Spread across a few simulated minutes.
    span = ctx.rng.uniform(60.0, 180.0)
    for i, dest in enumerate(destinations):
        amount = ctx.rng.round_amount(ctx.rng.uniform(8_000.0, 9_900.0))
        ts = base_ts + (span * (i + 1) / n_legs)
        if ctx.store.balance(source) < amount:
            # Abort cleanly: release and return nothing applied yet.
            ctx.reserved.difference_update(participants)
            return []
        txs.append(
            Transaction(
                tx_id=ctx.next_tx_id(),
                timestamp=ts,
                source_id=source,
                dest_id=dest,
                amount=amount,
                currency=Currency.USD,
                channel=Channel.WIRE,
                network=network,
                typology=Typology.STRUCTURING,
                pattern_id=pattern_id,
            )
        )
    ctx.clock.set(base_ts + span)
    return txs


def inject_layering(ctx: InjectionContext) -> list[Transaction]:
    """Pass-through path of 4–6 hops with minimal balance retention."""
    hops = ctx.rng.randint(4, 6)
    use_cross = bool(ctx.population.on_ramp_ids) and ctx.rng.random() < 0.35

    path: list[str] = []
    if use_cross:
        # bank -> ... -> onramp-bank -> onramp-wallet -> crypto/shell
        bank_ids = _eligible_ids(ctx.population, Network.BANKING, ctx.reserved)
        if len(bank_ids) < 2:
            use_cross = False
        else:
            onramp_banks = [
                pid
                for pid in ctx.population.on_ramp_ids
                if pid.startswith("onramp-bank-") and pid not in ctx.reserved
            ]
            if not onramp_banks:
                use_cross = False
            else:
                start = ctx.rng.choice(bank_ids)
                bridge_bank = ctx.rng.choice(onramp_banks)
                bridge_wallet = ctx.store.on_ramp_partner(bridge_bank)
                if bridge_wallet is None or bridge_wallet in ctx.reserved:
                    use_cross = False
                else:
                    mid = [p for p in bank_ids if p not in {start, bridge_bank}]
                    path = [start]
                    while len(path) < hops - 1 and mid:
                        path.append(mid.pop(ctx.rng.randint(0, len(mid) - 1)))
                    path.append(bridge_bank)
                    # After bridge, continue into crypto.
                    crypto_ids = _eligible_ids(
                        ctx.population, Network.CRYPTO, ctx.reserved
                    )
                    crypto_ids = [c for c in crypto_ids if c != bridge_wallet]
                    path.append(bridge_wallet)
                    while len(path) < hops + 1 and crypto_ids:
                        path.append(
                            crypto_ids.pop(ctx.rng.randint(0, len(crypto_ids) - 1))
                        )

    if not use_cross or len(path) < hops + 1:
        # Single-network path; prefer banking, fall back to crypto.
        for network in (Network.BANKING, Network.CRYPTO, Network.CROSS_BORDER):
            ids = _eligible_ids(ctx.population, network, ctx.reserved)
            if len(ids) >= hops + 1:
                ctx.rng.shuffle(ids)
                path = ids[: hops + 1]
                break
        else:
            return []

    if len(path) < hops + 1:
        return []

    # Ensure source can fund the chain.
    start_amount = ctx.rng.round_amount(ctx.rng.uniform(20_000.0, 80_000.0))
    if ctx.store.balance(path[0]) < start_amount:
        ctx.store.set_balance(path[0], start_amount * 1.2)

    pattern_id = _new_pattern_id(ctx.rng)
    participants = set(path)
    ctx.reserved.update(participants)

    txs: list[Transaction] = []
    amount = start_amount
    ts = ctx.clock.now
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        retain = ctx.rng.uniform(0.005, 0.03)
        send_amount = ctx.rng.round_amount(amount * (1.0 - retain))
        if send_amount <= 0 or ctx.store.balance(src) < send_amount:
            ctx.reserved.difference_update(participants)
            return []
        ts += ctx.rng.uniform(0.05, 1.9)
        src_party = ctx.population.parties[src]
        network = src_party.network
        if network == Network.CRYPTO:
            channel = Channel.TRANSFER
            currency = Currency.USDT
        elif network == Network.CROSS_BORDER:
            channel = Channel.WIRE
            currency = Currency.USD
        else:
            channel = Channel.WIRE
            currency = Currency.USD
        txs.append(
            Transaction(
                tx_id=ctx.next_tx_id(),
                timestamp=ts,
                source_id=src,
                dest_id=dst,
                amount=send_amount,
                currency=currency,
                channel=channel,
                network=network,
                typology=Typology.LAYERING,
                pattern_id=pattern_id,
            )
        )
        amount = send_amount
    ctx.clock.set(ts)
    return txs


def inject_scatter_gather(ctx: InjectionContext) -> list[Transaction]:
    """Fan-in: many mules send similar amounts to one hub."""
    n_mules = ctx.rng.randint(6, 15)
    # Hub prefers shell or wallet.
    hub_candidates = [
        pid
        for pid in (ctx.population.shell_ids + ctx.population.wallet_ids)
        if pid not in ctx.reserved
    ]
    if not hub_candidates:
        return []
    hub = ctx.rng.choice(hub_candidates)
    hub_party = ctx.population.parties[hub]
    network = hub_party.network

    mule_pool = _eligible_ids(ctx.population, network, ctx.reserved)
    mule_pool = [pid for pid in mule_pool if pid != hub]
    if len(mule_pool) < n_mules:
        # Fall back to banking mules into a shell/wallet via same-network only.
        if network != Network.BANKING:
            mule_pool = _eligible_ids(
                ctx.population, Network.BANKING, ctx.reserved
            )
            mule_pool = [pid for pid in mule_pool if pid != hub]
            # Keep transfers on banking if hub is not banking — require same network.
            if hub_party.network != Network.BANKING:
                # Use mules from hub network only; if short, fail closed.
                mule_pool = _eligible_ids(ctx.population, network, ctx.reserved)
                mule_pool = [pid for pid in mule_pool if pid != hub]
        if len(mule_pool) < n_mules:
            return []

    mules = ctx.rng.sample(mule_pool, n_mules)
    base_amount = ctx.rng.round_amount(ctx.rng.uniform(1_500.0, 4_500.0))
    for mule in mules:
        needed = base_amount * 1.1
        if ctx.store.balance(mule) < needed:
            ctx.store.set_balance(mule, needed)

    pattern_id = _new_pattern_id(ctx.rng)
    participants = {hub, *mules}
    ctx.reserved.update(participants)

    txs: list[Transaction] = []
    base_ts = ctx.clock.now
    span = ctx.rng.uniform(30.0, 120.0)
    channel = Channel.TRANSFER if network == Network.CRYPTO else Channel.WIRE
    currency = Currency.USDT if network == Network.CRYPTO else Currency.USD
    for i, mule in enumerate(mules):
        jitter = ctx.rng.uniform(0.9, 1.1)
        amount = ctx.rng.round_amount(base_amount * jitter)
        if ctx.store.balance(mule) < amount:
            ctx.reserved.difference_update(participants)
            return []
        ts = base_ts + (span * (i + 1) / n_mules)
        txs.append(
            Transaction(
                tx_id=ctx.next_tx_id(),
                timestamp=ts,
                source_id=mule,
                dest_id=hub,
                amount=amount,
                currency=currency,
                channel=channel,
                network=network,
                typology=Typology.SCATTER_GATHER,
                pattern_id=pattern_id,
            )
        )
    ctx.clock.set(base_ts + span)
    return txs


def inject_cycle(ctx: InjectionContext) -> list[Transaction]:
    """Directed cycle of 3–5 parties with nearly equal amounts."""
    n = ctx.rng.randint(3, 5)
    for network in (Network.BANKING, Network.CRYPTO, Network.CROSS_BORDER):
        ids = _eligible_ids(ctx.population, network, ctx.reserved)
        if len(ids) >= n:
            break
    else:
        return []

    members = ctx.rng.sample(ids, n)
    amount = ctx.rng.round_amount(ctx.rng.uniform(5_000.0, 25_000.0))
    for pid in members:
        if ctx.store.balance(pid) < amount * 1.05:
            ctx.store.set_balance(pid, amount * 1.2)

    pattern_id = _new_pattern_id(ctx.rng)
    participants = set(members)
    ctx.reserved.update(participants)

    txs: list[Transaction] = []
    ts = ctx.clock.now
    network = ctx.population.parties[members[0]].network
    if network == Network.CRYPTO:
        channel, currency = Channel.TRANSFER, Currency.USDT
    elif network == Network.CROSS_BORDER:
        channel, currency = Channel.WIRE, Currency.USD
    else:
        channel, currency = Channel.ACH, Currency.USD

    for i in range(n):
        src = members[i]
        dst = members[(i + 1) % n]
        leg_amount = ctx.rng.round_amount(amount * ctx.rng.uniform(0.98, 1.02))
        if ctx.store.balance(src) < leg_amount:
            ctx.reserved.difference_update(participants)
            return []
        ts += ctx.rng.uniform(0.5, 5.0)
        txs.append(
            Transaction(
                tx_id=ctx.next_tx_id(),
                timestamp=ts,
                source_id=src,
                dest_id=dst,
                amount=leg_amount,
                currency=currency,
                channel=channel,
                network=network,
                typology=Typology.CYCLE,
                pattern_id=pattern_id,
            )
        )
    ctx.clock.set(ts)
    return txs


INJECTORS: dict[Typology, Callable[[InjectionContext], list[Transaction]]] = {
    Typology.STRUCTURING: inject_structuring,
    Typology.LAYERING: inject_layering,
    Typology.SCATTER_GATHER: inject_scatter_gather,
    Typology.CYCLE: inject_cycle,
}


def pick_typology(rng: SimRng, network: Network | None = None) -> Typology:
    """Choose a typology that can run on the given network (or any)."""
    options = [
        Typology.STRUCTURING,
        Typology.LAYERING,
        Typology.SCATTER_GATHER,
        Typology.CYCLE,
    ]
    if network == Network.CRYPTO:
        options = [Typology.LAYERING, Typology.SCATTER_GATHER, Typology.CYCLE]
    elif network == Network.CROSS_BORDER:
        options = [Typology.LAYERING, Typology.SCATTER_GATHER, Typology.CYCLE]
    return rng.choice(options)
