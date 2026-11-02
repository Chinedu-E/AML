"""Tests for the aml-sim CLI."""

from __future__ import annotations

import json
from pathlib import Path

from aml.cli import run


def test_cli_writes_summary_and_jsonl(tmp_path: Path, capsys) -> None:
    out = tmp_path / "events.jsonl"
    code = run(
        [
            "--events",
            "50",
            "--seed",
            "42",
            "--banking",
            "30",
            "--crypto",
            "20",
            "--cross-border",
            "12",
            "--jsonl",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "AML synthetic run complete" in captured.out
    assert "events:            50" in captured.out
    assert out.exists()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 50
    for row in rows:
        assert "tx_id" in row
        assert "amount" in row
        assert "typology" in row
        assert "network" in row
        if row["typology"] == "benign":
            assert row["pattern_id"] is None
        else:
            assert row["pattern_id"] is not None
