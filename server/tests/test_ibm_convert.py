"""Tests for IBM AML CSV → replay JSONL conversion."""

from __future__ import annotations

import json
from pathlib import Path

from aml.ibm_convert import (
    convert_ibm_csv,
    map_channel,
    map_currency,
    map_network,
    run,
    uniquify_fieldnames,
)
from aml.sources import ReplaySource, parse_transaction_row


def _write_csv(path: Path, rows: list[dict[str, str]], *, ibm_dup_account: bool = False) -> None:
    # Real IBM files repeat the header name "Account" twice (not Account.1).
    if ibm_dup_account:
        headers = [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering",
        ]
        key_order = [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering",
        ]
    else:
        headers = [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering",
        ]
        key_order = headers
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row[h] for h in key_order))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_uniquify_duplicate_account_headers() -> None:
    assert uniquify_fieldnames(
        ["Timestamp", "Account", "To Bank", "Account", "Amount Paid"]
    ) == ["Timestamp", "Account", "To Bank", "Account.1", "Amount Paid"]


def test_currency_and_channel_maps() -> None:
    assert map_currency("US Dollar").value == "USD"
    assert map_currency("Euro").value == "EUR"
    assert map_currency("Bitcoin").value == "BTC"
    assert map_channel("ACH").value == "ach"
    assert map_channel("Wire").value == "wire"
    assert map_channel("Credit Card").value == "card"


def test_network_inference() -> None:
    assert map_network(
        from_bank="1", to_bank="1", payment_currency="US Dollar", receiving_currency="US Dollar"
    ).value == "banking"
    assert map_network(
        from_bank="1", to_bank="2", payment_currency="Bitcoin", receiving_currency="Bitcoin"
    ).value == "crypto"
    assert map_network(
        from_bank="1", to_bank="9", payment_currency="Euro", receiving_currency="Euro"
    ).value == "cross_border"


def test_convert_drops_self_and_labels_laundering(tmp_path: Path) -> None:
    csv_path = tmp_path / "HI-Small_Trans.csv"
    out = tmp_path / "out.jsonl"
    _write_csv(
        csv_path,
        [
            {
                "Timestamp": "2022/09/01 00:00",
                "From Bank": "10",
                "Account": "AAA",
                "To Bank": "10",
                "Account.1": "AAA",
                "Amount Received": "100.0",
                "Receiving Currency": "US Dollar",
                "Amount Paid": "100.0",
                "Payment Currency": "US Dollar",
                "Payment Format": "Reinvestment",
                "Is Laundering": "0",
            },
            {
                "Timestamp": "2022/09/01 00:05",
                "From Bank": "1",
                "Account": "SRC1",
                "To Bank": "2",
                "Account.1": "DST1",
                "Amount Received": "2500.5",
                "Receiving Currency": "US Dollar",
                "Amount Paid": "2500.5",
                "Payment Currency": "US Dollar",
                "Payment Format": "ACH",
                "Is Laundering": "1",
            },
            {
                "Timestamp": "2022/09/01 00:10",
                "From Bank": "3",
                "Account": "SRC2",
                "To Bank": "4",
                "Account.1": "DST2",
                "Amount Received": "10.0",
                "Receiving Currency": "Euro",
                "Amount Paid": "10.0",
                "Payment Currency": "Euro",
                "Payment Format": "Wire",
                "Is Laundering": "0",
            },
        ],
        ibm_dup_account=True,
    )
    stats = convert_ibm_csv(csv_path, out, drop_self=True)
    assert stats.dropped_self == 1
    assert stats.rows_written == 2
    assert stats.rows_read == 3

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["typology"] == "structuring"
    assert rows[0]["channel"] == "ach"
    assert rows[0]["source_id"] == "ibm-1:SRC1"
    assert rows[0]["dest_id"] == "ibm-2:DST1"
    assert rows[0]["timestamp"] == 0.0
    assert rows[1]["timestamp"] == 300.0  # 5 minutes later
    assert rows[1]["network"] == "cross_border"
    assert rows[1]["currency"] == "EUR"

    # ReplaySource can consume the file.
    src = ReplaySource(out)
    tx = src.next_transaction()
    assert tx.tx_id == rows[0]["tx_id"]
    assert parse_transaction_row(rows[0]).amount == 2500.5


def test_cli_limit_and_missing_csv(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "tiny.csv"
    out = tmp_path / "limited.jsonl"
    _write_csv(
        csv_path,
        [
            {
                "Timestamp": "2022/09/01 01:00",
                "From Bank": "1",
                "Account": "A",
                "To Bank": "2",
                "Account.1": "B",
                "Amount Received": "1",
                "Receiving Currency": "US Dollar",
                "Amount Paid": "1",
                "Payment Currency": "US Dollar",
                "Payment Format": "ACH",
                "Is Laundering": "0",
            },
            {
                "Timestamp": "2022/09/01 01:01",
                "From Bank": "1",
                "Account": "C",
                "To Bank": "2",
                "Account.1": "D",
                "Amount Received": "2",
                "Receiving Currency": "US Dollar",
                "Amount Paid": "2",
                "Payment Currency": "US Dollar",
                "Payment Format": "Wire",
                "Is Laundering": "0",
            },
        ],
    )
    assert run(["--csv", str(csv_path), "--out", str(out), "--limit", "1"]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "IBM AML → replay JSONL complete" in capsys.readouterr().out

    assert run(["--csv", str(tmp_path / "missing.csv"), "--out", str(out)]) == 1
