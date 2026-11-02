"""Tests for multi-network population seeding."""

from __future__ import annotations

from aml.clock import SimRng
from aml.domain import Network, NodeKind, PartyType
from aml.graph_store import GraphStore
from aml.population import PopulationConfig, seed_population


def test_three_networks_with_expected_party_types(
    store: GraphStore, population, small_pop_config: PopulationConfig
) -> None:
    assert set(population.by_network) == set(Network)
    assert len(population.party_ids_in(Network.BANKING)) == (
        small_pop_config.banking + small_pop_config.on_ramps
    )
    assert len(population.party_ids_in(Network.CRYPTO)) == (
        small_pop_config.crypto + small_pop_config.on_ramps
    )
    assert len(population.party_ids_in(Network.CROSS_BORDER)) == (
        small_pop_config.cross_border
    )

    types = {p.party_type for p in population.parties.values()}
    assert PartyType.INDIVIDUAL in types
    assert PartyType.BUSINESS in types
    assert PartyType.WALLET in types
    assert PartyType.MIXER in types
    assert PartyType.SMART_CONTRACT in types
    assert PartyType.SHELL in types
    assert len(population.mixer_ids) == small_pop_config.mixers
    assert len(population.contract_ids) == small_pop_config.smart_contracts
    assert len(population.shell_ids) >= 1


def test_every_party_has_ip_and_address(store: GraphStore, population) -> None:
    for party_id in population.parties:
        ips = store.parties_with_ip(party_id)
        addrs = store.parties_with_address(party_id)
        assert len(ips) == 1
        assert len(addrs) == 1
        assert store.g.nodes[ips[0]]["kind"] == NodeKind.IP_ADDRESS.value
        assert store.g.nodes[addrs[0]]["kind"] == NodeKind.PHYSICAL_ADDRESS.value


def test_on_ramp_parties_in_banking_and_crypto(store: GraphStore, population) -> None:
    assert population.on_ramp_ids
    bank_onramps = [p for p in population.on_ramp_ids if p.startswith("onramp-bank-")]
    wallet_onramps = [
        p for p in population.on_ramp_ids if p.startswith("onramp-wallet-")
    ]
    assert len(bank_onramps) == len(wallet_onramps)
    for bank_id in bank_onramps:
        assert population.parties[bank_id].network == Network.BANKING
        assert population.parties[bank_id].is_on_ramp
        partner = store.on_ramp_partner(bank_id)
        assert partner is not None
        assert partner in wallet_onramps
        assert population.parties[partner].network == Network.CRYPTO


def test_same_seed_rebuilds_same_ids(small_pop_config: PopulationConfig) -> None:
    store_a = GraphStore()
    store_b = GraphStore()
    pop_a = seed_population(store_a, SimRng(seed=99), small_pop_config)
    pop_b = seed_population(store_b, SimRng(seed=99), small_pop_config)
    assert sorted(pop_a.parties) == sorted(pop_b.parties)
    for pid in pop_a.parties:
        assert pop_a.parties[pid].party_type == pop_b.parties[pid].party_type
        assert pop_a.parties[pid].network == pop_b.parties[pid].network
        assert pop_a.preferred[pid] == pop_b.preferred[pid]
