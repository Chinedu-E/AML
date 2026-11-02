"""Seed multi-network party populations and identity nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from aml.clock import SimRng
from aml.domain import (
    EdgeKind,
    Network,
    NodeKind,
    Party,
    PartyType,
)
from aml.geo import assign_location
from aml.graph_store import GraphStore

COUNTRIES = ("US", "GB", "DE", "NG", "SG", "AE", "CH", "KY", "PA", "HK")


@dataclass
class PopulationConfig:
    banking: int = 400
    crypto: int = 150
    cross_border: int = 60
    mixers: int = 3
    smart_contracts: int = 5
    on_ramps: int = 8
    # Shared identity pools so address/IP reuse is visible.
    ip_pool_size: int = 80
    address_pool_size: int = 60
    preferred_counterparties: int = 8
    initial_balance_min: float = 5_000.0
    initial_balance_max: float = 250_000.0


@dataclass
class Population:
    parties: dict[str, Party] = field(default_factory=dict)
    preferred: dict[str, list[str]] = field(default_factory=dict)
    by_network: dict[Network, list[str]] = field(default_factory=dict)
    on_ramp_ids: list[str] = field(default_factory=list)
    mixer_ids: list[str] = field(default_factory=list)
    contract_ids: list[str] = field(default_factory=list)
    shell_ids: list[str] = field(default_factory=list)
    wallet_ids: list[str] = field(default_factory=list)

    def parties_in(self, network: Network) -> list[Party]:
        return [self.parties[pid] for pid in self.by_network.get(network, [])]

    def party_ids_in(self, network: Network) -> list[str]:
        return list(self.by_network.get(network, []))


def seed_population(
    store: GraphStore,
    rng: SimRng,
    config: PopulationConfig | None = None,
    opened_at: float = 0.0,
) -> Population:
    """Create parties, identity nodes, and preferred-counterparty sets on ``store``."""
    cfg = config or PopulationConfig()
    pop = Population()
    for network in Network:
        pop.by_network[network] = []

    ip_ids = [f"ip-{i:04d}" for i in range(cfg.ip_pool_size)]
    addr_ids = [f"addr-{i:04d}" for i in range(cfg.address_pool_size)]
    for ip_id in ip_ids:
        store.add_identity_node(ip_id, NodeKind.IP_ADDRESS, {"value": ip_id})
    for addr_id in addr_ids:
        store.add_identity_node(
            addr_id,
            NodeKind.PHYSICAL_ADDRESS,
            {"value": addr_id, "country": rng.choice(COUNTRIES)},
        )

    def _add_party(
        party_id: str,
        party_type: PartyType,
        network: Network,
        *,
        is_on_ramp: bool = False,
    ) -> Party:
        country = rng.choice(COUNTRIES)
        loc = assign_location(rng, country)
        party = Party(
            party_id=party_id,
            party_type=party_type,
            network=network,
            country=loc.country,
            opened_at=opened_at,
            is_on_ramp=is_on_ramp,
            city=loc.city,
            lat=loc.lat,
            lng=loc.lng,
        )
        balance = rng.uniform(cfg.initial_balance_min, cfg.initial_balance_max)
        store.add_party(party, balance=balance)
        ip_id = rng.choice(ip_ids)
        addr_id = rng.choice(addr_ids)
        store.add_identity_edge(party_id, ip_id, EdgeKind.HAS_IP)
        store.add_identity_edge(party_id, addr_id, EdgeKind.HAS_ADDRESS)
        pop.parties[party_id] = party
        pop.by_network[network].append(party_id)
        return party

    # Banking: mostly individuals, some businesses.
    n_business = max(1, cfg.banking // 5)
    for i in range(cfg.banking):
        ptype = PartyType.BUSINESS if i < n_business else PartyType.INDIVIDUAL
        _add_party(f"bank-{i:04d}", ptype, Network.BANKING)

    # Crypto: wallets + mixers + contracts.
    n_wallets = max(0, cfg.crypto - cfg.mixers - cfg.smart_contracts)
    for i in range(n_wallets):
        party = _add_party(f"wallet-{i:04d}", PartyType.WALLET, Network.CRYPTO)
        pop.wallet_ids.append(party.party_id)
    for i in range(cfg.mixers):
        party = _add_party(f"mixer-{i:04d}", PartyType.MIXER, Network.CRYPTO)
        pop.mixer_ids.append(party.party_id)
    for i in range(cfg.smart_contracts):
        party = _add_party(
            f"contract-{i:04d}", PartyType.SMART_CONTRACT, Network.CRYPTO
        )
        pop.contract_ids.append(party.party_id)

    # Cross-border: shells and trading firms (businesses).
    n_shells = max(1, cfg.cross_border // 2)
    for i in range(cfg.cross_border):
        if i < n_shells:
            party = _add_party(f"shell-{i:04d}", PartyType.SHELL, Network.CROSS_BORDER)
            pop.shell_ids.append(party.party_id)
        else:
            _add_party(f"trade-{i:04d}", PartyType.BUSINESS, Network.CROSS_BORDER)

    # On-ramp bridge parties: same identity lives in banking and crypto graphs.
    for i in range(cfg.on_ramps):
        bank_id = f"onramp-bank-{i:04d}"
        wallet_id = f"onramp-wallet-{i:04d}"
        _add_party(bank_id, PartyType.INDIVIDUAL, Network.BANKING, is_on_ramp=True)
        _add_party(wallet_id, PartyType.WALLET, Network.CRYPTO, is_on_ramp=True)
        pop.on_ramp_ids.extend([bank_id, wallet_id])
        pop.wallet_ids.append(wallet_id)
        # Link the pair so later phases can treat them as one corridor.
        store.link_on_ramp(bank_id, wallet_id)

    # Preferred counterparties (Zipf-biased within network).
    for network, ids in pop.by_network.items():
        if len(ids) < 2:
            continue
        k = min(cfg.preferred_counterparties, len(ids) - 1)
        for pid in ids:
            others = [oid for oid in ids if oid != pid]
            # Rank others, then sample Zipf among ranks for a preferred set.
            ranked = list(others)
            rng.shuffle(ranked)
            preferred: list[str] = []
            # Bias toward the first ranks of the shuffled list via Zipf picks.
            attempts = 0
            while len(preferred) < k and attempts < k * 5:
                attempts += 1
                idx = rng.zipf_index(len(ranked))
                cand = ranked[idx]
                if cand not in preferred:
                    preferred.append(cand)
            pop.preferred[pid] = preferred

    return pop
