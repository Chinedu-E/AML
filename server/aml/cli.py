"""Console entry point for the AML synthetic simulator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from aml.generator import Simulator, SimulatorConfig
from aml.population import PopulationConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml-sim",
        description="Synthetic multi-network AML transaction generator",
    )
    p.add_argument("--events", type=int, default=2000, help="Number of transactions")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--rate",
        type=float,
        default=7.0,
        help="Target benign events per simulated second",
    )
    p.add_argument(
        "--pattern-fraction",
        type=float,
        default=0.02,
        help="Fraction of draws that attempt typology injection",
    )
    p.add_argument("--jsonl", type=Path, default=None, help="Write events as JSONL")
    p.add_argument("--banking", type=int, default=400)
    p.add_argument("--crypto", type=int, default=150)
    p.add_argument("--cross-border", type=int, default=60)
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulatorConfig(
        seed=args.seed,
        rate_per_second=args.rate,
        pattern_event_fraction=args.pattern_fraction,
        population=PopulationConfig(
            banking=args.banking,
            crypto=args.crypto,
            cross_border=args.cross_border,
        ),
    )
    sim = Simulator(config)
    events_by_network: Counter[str] = Counter()
    events_by_typology: Counter[str] = Counter()
    pattern_parties: set[str] = set()
    rows: list[dict] = []

    for tx in sim.run(args.events):
        events_by_network[tx.network.value] += 1
        events_by_typology[tx.typology.value] += 1
        if tx.pattern_id:
            pattern_parties.add(tx.source_id)
            pattern_parties.add(tx.dest_id)
        rows.append(tx.to_dict())

    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    stats = sim.store.snapshot_stats()
    print("AML synthetic run complete")
    print(f"  seed:              {args.seed}")
    print(f"  events:            {len(rows)}")
    print(f"  by network:        {dict(events_by_network)}")
    print(f"  by typology:       {dict(events_by_typology)}")
    print(f"  pattern parties:   {len(pattern_parties)}")
    print(f"  graph nodes:       {stats['nodes']}")
    print(f"  transfer edges:    {stats['transfer_edges']}")
    print(f"  identity edges:    {stats['identity_edges']}")
    if args.jsonl is not None:
        print(f"  jsonl:             {args.jsonl}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
