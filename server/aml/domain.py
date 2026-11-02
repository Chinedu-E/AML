"""Domain types for the AML synthetic transaction simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Network(str, Enum):
    BANKING = "banking"
    CRYPTO = "crypto"
    CROSS_BORDER = "cross_border"


class PartyType(str, Enum):
    INDIVIDUAL = "individual"
    BUSINESS = "business"
    WALLET = "wallet"
    MIXER = "mixer"
    SMART_CONTRACT = "smart_contract"
    SHELL = "shell"


class NodeKind(str, Enum):
    PARTY = "party"
    IP_ADDRESS = "ip_address"
    PHYSICAL_ADDRESS = "physical_address"


class Channel(str, Enum):
    WIRE = "wire"
    ACH = "ach"
    PAYROLL = "payroll"
    CARD = "card"
    TRANSFER = "transfer"
    CONTRACT = "contract"
    MIXER = "mixer"
    INVOICE = "invoice"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    BTC = "BTC"
    USDT = "USDT"


class Typology(str, Enum):
    BENIGN = "benign"
    STRUCTURING = "structuring"
    LAYERING = "layering"
    SCATTER_GATHER = "scatter_gather"
    CYCLE = "cycle"


class EdgeKind(str, Enum):
    TRANSFER = "transfer"
    HAS_IP = "has_ip"
    HAS_ADDRESS = "has_address"


@dataclass(frozen=True)
class Party:
    party_id: str
    party_type: PartyType
    network: Network
    country: str
    opened_at: float
    is_on_ramp: bool = False
    city: str = ""
    lat: float = 0.0
    lng: float = 0.0


@dataclass(frozen=True)
class Transaction:
    tx_id: str
    timestamp: float
    source_id: str
    dest_id: str
    amount: float
    currency: Currency
    channel: Channel
    network: Network
    typology: Typology = Typology.BENIGN
    pattern_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["currency"] = self.currency.value
        data["channel"] = self.channel.value
        data["network"] = self.network.value
        data["typology"] = self.typology.value
        return data


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    typology: Typology
    network: Network
    party_ids: tuple[str, ...] = field(default_factory=tuple)
    start_ts: float = 0.0
    end_ts: float = 0.0


BANKING_CHANNELS: tuple[Channel, ...] = (
    Channel.WIRE,
    Channel.ACH,
    Channel.PAYROLL,
    Channel.CARD,
)
CRYPTO_CHANNELS: tuple[Channel, ...] = (
    Channel.TRANSFER,
    Channel.CONTRACT,
    Channel.MIXER,
)
CROSS_BORDER_CHANNELS: tuple[Channel, ...] = (
    Channel.INVOICE,
    Channel.WIRE,
)

NETWORK_CHANNELS: dict[Network, tuple[Channel, ...]] = {
    Network.BANKING: BANKING_CHANNELS,
    Network.CRYPTO: CRYPTO_CHANNELS,
    Network.CROSS_BORDER: CROSS_BORDER_CHANNELS,
}

NETWORK_CURRENCIES: dict[Network, tuple[Currency, ...]] = {
    Network.BANKING: (Currency.USD, Currency.EUR),
    Network.CRYPTO: (Currency.BTC, Currency.USDT),
    Network.CROSS_BORDER: (Currency.USD, Currency.EUR),
}
