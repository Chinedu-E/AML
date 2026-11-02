"""Convert IBM AML HI-Small (and sibling) CSVs into aml-serve replay JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from aml.domain import Channel, Currency, Network, Transaction, Typology

# IBM Payment Format → our Channel.
FORMAT_TO_CHANNEL: dict[str, Channel] = {
    "ach": Channel.ACH,
    "wire": Channel.WIRE,
    "credit card": Channel.CARD,
    "card": Channel.CARD,
    "cheque": Channel.ACH,
    "check": Channel.ACH,
    "cash": Channel.CARD,
    "reinvestment": Channel.TRANSFER,
    "bitcoin": Channel.TRANSFER,
    "exchange": Channel.TRANSFER,
}

# IBM currency names → our Currency enum (unsupported → USD fallback).
CURRENCY_TO_ENUM: dict[str, Currency] = {
    "us dollar": Currency.USD,
    "usd": Currency.USD,
    "euro": Currency.EUR,
    "eur": Currency.EUR,
    "bitcoin": Currency.BTC,
    "btc": Currency.BTC,
    "usdt": Currency.USDT,
    "tether": Currency.USDT,
}

TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


@dataclass
class ConvertStats:
    rows_read: int = 0
    rows_written: int = 0
    dropped_self: int = 0
    dropped_bad: int = 0
    by_network: Counter[str] | None = None
    by_typology: Counter[str] | None = None
    by_channel: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.by_network = self.by_network or Counter()
        self.by_typology = self.by_typology or Counter()
        self.by_channel = self.by_channel or Counter()


def _norm_header(name: str) -> str:
    return " ".join(name.strip().lower().split())


def uniquify_fieldnames(names: list[str]) -> list[str]:
    """Rename duplicate CSV headers (IBM uses Account twice → Account.1)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}.{count}")
    return out


def _pick(row: dict[str, str], *aliases: str) -> str | None:
    lookup = {_norm_header(k): v for k, v in row.items()}
    for alias in aliases:
        key = _norm_header(alias)
        if key in lookup and lookup[key] not in ("", None):
            return str(lookup[key]).strip()
    return None


def parse_ibm_timestamp(raw: str) -> datetime:
    text = raw.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized IBM timestamp: {raw!r}")


def map_currency(raw: str | None) -> Currency:
    if not raw:
        return Currency.USD
    return CURRENCY_TO_ENUM.get(raw.strip().lower(), Currency.USD)


def map_channel(raw: str | None) -> Channel:
    if not raw:
        return Channel.TRANSFER
    return FORMAT_TO_CHANNEL.get(raw.strip().lower(), Channel.TRANSFER)


def map_network(
    *,
    from_bank: str,
    to_bank: str,
    payment_currency: str | None,
    receiving_currency: str | None,
) -> Network:
    currencies = " ".join(
        c.lower() for c in (payment_currency or "", receiving_currency or "")
    )
    if "bitcoin" in currencies or "btc" in currencies or "usdt" in currencies:
        return Network.CRYPTO
    if from_bank != to_bank and map_currency(payment_currency) != Currency.USD:
        return Network.CROSS_BORDER
    if from_bank != to_bank and map_currency(receiving_currency) not in (
        Currency.USD,
        map_currency(payment_currency),
    ):
        return Network.CROSS_BORDER
    # Distinct banks with fiat still sit in the banking corridor for this adapter.
    return Network.BANKING


def map_typology(is_laundering: str | None) -> Typology:
    if is_laundering is None:
        return Typology.BENIGN
    flag = is_laundering.strip().lower()
    if flag in {"1", "true", "yes", "y"}:
        # IBM binary label — treat as structuring-class illicit for our enum.
        return Typology.STRUCTURING
    return Typology.BENIGN


def party_id(bank: str, account: str) -> str:
    return f"ibm-{bank}:{account}"


def row_to_transaction(
    row: dict[str, str],
    *,
    index: int,
) -> tuple[Transaction, datetime] | None:
    """Convert one IBM CSV row (timestamp left absolute until epoch is known)."""
    ts_raw = _pick(row, "Timestamp")
    src_acct = _pick(row, "Account")
    # pandas renames the second Account column to Account.1
    dst_acct = _pick(row, "Account.1", "Account 1", "To Account")
    from_bank = _pick(row, "From Bank") or "0"
    to_bank = _pick(row, "To Bank") or "0"
    if not ts_raw or not src_acct or not dst_acct:
        return None

    amount_raw = _pick(row, "Amount Paid", "Amount Received")
    if amount_raw is None:
        return None
    amount = float(amount_raw.replace(",", ""))
    if amount <= 0:
        return None

    pay_ccy = _pick(row, "Payment Currency")
    recv_ccy = _pick(row, "Receiving Currency")
    fmt = _pick(row, "Payment Format")
    is_laundering = _pick(row, "Is Laundering")
    typology = map_typology(is_laundering)

    ts = parse_ibm_timestamp(ts_raw)
    source = party_id(from_bank, src_acct)
    dest = party_id(to_bank, dst_acct)
    tx = Transaction(
        tx_id=f"ibm-{index:08d}",
        timestamp=0.0,  # filled by iterator once the kept-row epoch is known
        source_id=source,
        dest_id=dest,
        amount=amount,
        currency=map_currency(pay_ccy or recv_ccy),
        channel=map_channel(fmt),
        network=map_network(
            from_bank=from_bank,
            to_bank=to_bank,
            payment_currency=pay_ccy,
            receiving_currency=recv_ccy,
        ),
        typology=typology,
        pattern_id=f"ibm-launder-{index}" if typology != Typology.BENIGN else None,
    )
    return tx, ts


def iter_ibm_transactions(
    csv_path: Path,
    *,
    limit: int | None = None,
    drop_self: bool = True,
    stats: ConvertStats | None = None,
) -> Iterator[Transaction]:
    """Yield converted transactions; mutate ``stats`` with counters."""
    stats = stats if stats is not None else ConvertStats()
    t0: datetime | None = None
    written = 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        # IBM HI-Small headers repeat "Account" for source and dest.
        reader.fieldnames = uniquify_fieldnames(list(reader.fieldnames))
        for index, row in enumerate(reader):
            stats.rows_read += 1
            try:
                parsed = row_to_transaction(row, index=index)
            except (ValueError, KeyError):
                stats.dropped_bad += 1
                continue
            if parsed is None:
                stats.dropped_bad += 1
                continue
            tx, ts = parsed
            if drop_self and tx.source_id == tx.dest_id:
                stats.dropped_self += 1
                continue
            if t0 is None:
                t0 = ts
            rel = max(0.0, (ts - t0).total_seconds())
            tx = Transaction(
                tx_id=tx.tx_id,
                timestamp=rel,
                source_id=tx.source_id,
                dest_id=tx.dest_id,
                amount=tx.amount,
                currency=tx.currency,
                channel=tx.channel,
                network=tx.network,
                typology=tx.typology,
                pattern_id=tx.pattern_id,
            )
            stats.by_network[tx.network.value] += 1
            stats.by_typology[tx.typology.value] += 1
            stats.by_channel[tx.channel.value] += 1
            stats.rows_written += 1
            written += 1
            yield tx
            if limit is not None and written >= limit:
                return


def convert_ibm_csv(
    csv_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    drop_self: bool = True,
) -> ConvertStats:
    """Write replay-compatible JSONL and return conversion stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ConvertStats()
    with out_path.open("w", encoding="utf-8") as out:
        for tx in iter_ibm_transactions(
            csv_path, limit=limit, drop_self=drop_self, stats=stats
        ):
            out.write(json.dumps(tx.to_dict(), sort_keys=True) + "\n")
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml-ibm-convert",
        description=(
            "Convert IBM AML transaction CSVs (e.g. HI-Small_Trans.csv) "
            "into JSONL for aml-serve --source replay"
        ),
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to HI-Small_Trans.csv (or HI/LI Medium/Large)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ibm_hi_small.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max transactions to write (useful for demos)",
    )
    p.add_argument(
        "--keep-self",
        action="store_true",
        help="Keep same-account self transfers (dropped by default)",
    )
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    try:
        stats = convert_ibm_csv(
            args.csv,
            args.out,
            limit=args.limit,
            drop_self=not args.keep_self,
        )
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("IBM AML → replay JSONL complete")
    print(f"  csv:           {args.csv}")
    print(f"  out:           {args.out}")
    print(f"  rows read:     {stats.rows_read}")
    print(f"  rows written:  {stats.rows_written}")
    print(f"  dropped self:  {stats.dropped_self}")
    print(f"  dropped bad:   {stats.dropped_bad}")
    print(f"  by network:    {dict(stats.by_network)}")
    print(f"  by typology:   {dict(stats.by_typology)}")
    print(f"  by channel:    {dict(stats.by_channel)}")
    print()
    print("Next:")
    print(
        f"  uv run aml-serve --source replay --replay {args.out} "
        f"--rate 2 --warmup 200 --reveal-labels"
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
