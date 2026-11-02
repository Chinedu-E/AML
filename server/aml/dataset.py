"""Seeded simulator + JSONL replay runs that yield HeteroData training snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch_geometric.data import HeteroData

from aml.convert import graph_store_to_hetero, snapshot_to_hetero
from aml.features import (
    FeatureConfig,
    FeatureSchema,
    SnapshotArrays,
    build_snapshot_arrays,
    default_schema,
    fit_normalization,
)
from aml.generator import Simulator, SimulatorConfig
from aml.population import PopulationConfig
from aml.sources import ReplaySource

SOURCE_SIM = "sim"
SOURCE_IBM = "ibm"


@dataclass
class DatasetConfig:
    train_seeds: list[int] = field(
        default_factory=lambda: list(range(100, 124))
    )
    val_seeds: list[int] = field(
        default_factory=lambda: list(range(200, 204))
    )
    test_seeds: list[int] = field(
        default_factory=lambda: list(range(300, 304))
    )
    events_per_run: int = 4_000
    snapshot_every: int = 2_000
    pattern_event_fraction: float = 0.02
    rate_per_second: float = 7.0
    population: PopulationConfig = field(default_factory=PopulationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    max_transfer_edges: int = 50_000


@dataclass
class JsonlDatasetConfig:
    """One-pass JSONL (IBM / replay) → chronological snapshot splits."""

    snapshot_every: int = 2_000
    max_events: int | None = None
    train_frac: float = 0.70
    val_frac: float = 0.15
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    max_transfer_edges: int = 50_000
    source_label: str = SOURCE_IBM
    final_snapshot: bool = True


@dataclass
class SnapshotRecord:
    seed: int
    event_index: int
    arrays: SnapshotArrays
    source: str = SOURCE_SIM


@dataclass
class PreparedDataset:
    train: list[HeteroData]
    val: list[HeteroData]
    test: list[HeteroData]
    schema: FeatureSchema
    train_records: list[SnapshotRecord]
    val_records: list[SnapshotRecord]
    test_records: list[SnapshotRecord]
    meta: dict[str, Any] = field(default_factory=dict)


def collect_run_snapshots(
    seed: int,
    config: DatasetConfig,
    *,
    source: str = SOURCE_SIM,
) -> list[SnapshotRecord]:
    """Run one simulator seed and snapshot the graph every ``snapshot_every`` events."""
    sim = Simulator(
        SimulatorConfig(
            seed=seed,
            rate_per_second=config.rate_per_second,
            pattern_event_fraction=config.pattern_event_fraction,
            population=config.population,
            max_transfer_edges=config.max_transfer_edges,
        )
    )
    schema = default_schema(config.feature)
    records: list[SnapshotRecord] = []
    produced = 0
    for _tx in sim.run(config.events_per_run):
        produced += 1
        if produced % config.snapshot_every == 0:
            arrays = build_snapshot_arrays(
                sim.store,
                snapshot_time=sim.clock.now,
                config=config.feature,
                schema=schema,
            )
            records.append(
                SnapshotRecord(
                    seed=seed,
                    event_index=produced,
                    arrays=arrays,
                    source=source,
                )
            )
    return records


def collect_split_arrays(
    seeds: list[int], config: DatasetConfig, *, source: str = SOURCE_SIM
) -> list[SnapshotRecord]:
    out: list[SnapshotRecord] = []
    for seed in seeds:
        out.extend(collect_run_snapshots(seed, config, source=source))
    return out


def collect_jsonl_snapshots(
    path: Path,
    config: JsonlDatasetConfig | None = None,
) -> list[SnapshotRecord]:
    """Replay a JSONL file once (no loop) and snapshot every N applied events."""
    cfg = config or JsonlDatasetConfig()
    src = ReplaySource(
        Path(path),
        max_transfer_edges=cfg.max_transfer_edges,
        loop=False,
    )
    schema = default_schema(cfg.feature)
    records: list[SnapshotRecord] = []
    produced = 0
    limit = cfg.max_events if cfg.max_events is not None else src.n_rows
    limit = min(limit, src.n_rows)

    while produced < limit:
        try:
            src.next_transaction()
        except StopIteration:
            break
        produced += 1
        if produced % cfg.snapshot_every == 0:
            arrays = build_snapshot_arrays(
                src.store,
                snapshot_time=src.now,
                config=cfg.feature,
                schema=schema,
            )
            records.append(
                SnapshotRecord(
                    seed=0,
                    event_index=produced,
                    arrays=arrays,
                    source=cfg.source_label,
                )
            )

    if (
        cfg.final_snapshot
        and produced > 0
        and (not records or records[-1].event_index != produced)
    ):
        arrays = build_snapshot_arrays(
            src.store,
            snapshot_time=src.now,
            config=cfg.feature,
            schema=schema,
        )
        records.append(
            SnapshotRecord(
                seed=0,
                event_index=produced,
                arrays=arrays,
                source=cfg.source_label,
            )
        )
    return records


def split_records_chronological(
    records: list[SnapshotRecord],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[list[SnapshotRecord], list[SnapshotRecord], list[SnapshotRecord]]:
    """Time-ordered split (earlier snapshots → train) to avoid future leakage."""
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            f"need train_frac>0, val_frac>=0, train+val<1; got "
            f"train={train_frac}, val={val_frac}"
        )
    n = len(records)
    if n == 0:
        return [], [], []
    if n == 1:
        return list(records), [], []
    if n == 2:
        return [records[0]], [records[1]], []

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val
    # Prefer a non-empty test split when we have enough snapshots.
    while n_test < 1 and n_train > 1:
        n_train -= 1
        n_test = n - n_train - n_val
    while n_test < 1 and n_val > 1:
        n_val -= 1
        n_test = n - n_train - n_val
    if n_test < 1:
        # n == 3 with aggressive fracs: force 1/1/1.
        n_train, n_val, n_test = 1, 1, n - 2
    return (
        records[:n_train],
        records[n_train : n_train + n_val],
        records[n_train + n_val :],
    )


def _records_to_hetero(
    recs: list[SnapshotRecord], schema: FeatureSchema
) -> list[HeteroData]:
    graphs: list[HeteroData] = []
    for rec in recs:
        g = snapshot_to_hetero(rec.arrays, schema=schema, normalize=True)
        g.source = rec.source  # type: ignore[attr-defined]
        graphs.append(g)
    return graphs


def _count_by_source(recs: list[SnapshotRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.source] = counts.get(r.source, 0) + 1
    return counts


def _prepare_from_records(
    train_recs: list[SnapshotRecord],
    val_recs: list[SnapshotRecord],
    test_recs: list[SnapshotRecord],
    *,
    meta: dict[str, Any] | None = None,
) -> PreparedDataset:
    if not train_recs:
        raise ValueError("cannot build dataset: train split is empty")
    schema = fit_normalization([r.arrays for r in train_recs])
    return PreparedDataset(
        train=_records_to_hetero(train_recs, schema),
        val=_records_to_hetero(val_recs, schema),
        test=_records_to_hetero(test_recs, schema),
        schema=schema,
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        meta=meta or {},
    )


def build_dataset(config: DatasetConfig | None = None) -> PreparedDataset:
    """Generate train/val/test HeteroData lists with seed split and fitted normalization."""
    cfg = config or DatasetConfig()
    train_recs = collect_split_arrays(cfg.train_seeds, cfg, source=SOURCE_SIM)
    val_recs = collect_split_arrays(cfg.val_seeds, cfg, source=SOURCE_SIM)
    test_recs = collect_split_arrays(cfg.test_seeds, cfg, source=SOURCE_SIM)
    return _prepare_from_records(
        train_recs,
        val_recs,
        test_recs,
        meta={
            "mode": "sim",
            "counts": {
                "train": _count_by_source(train_recs),
                "val": _count_by_source(val_recs),
                "test": _count_by_source(test_recs),
            },
        },
    )


def build_jsonl_dataset(
    path: Path,
    config: JsonlDatasetConfig | None = None,
) -> PreparedDataset:
    """Build a dataset solely from one JSONL replay (chronological split)."""
    cfg = config or JsonlDatasetConfig()
    all_recs = collect_jsonl_snapshots(path, cfg)
    if not all_recs:
        raise ValueError(
            f"no snapshots from {path}; need events >= snapshot_every "
            f"({cfg.snapshot_every}) or enable final_snapshot"
        )
    train_recs, val_recs, test_recs = split_records_chronological(
        all_recs, train_frac=cfg.train_frac, val_frac=cfg.val_frac
    )
    return _prepare_from_records(
        train_recs,
        val_recs,
        test_recs,
        meta={
            "mode": "ibm",
            "jsonl": str(path),
            "events_snapshotted": all_recs[-1].event_index if all_recs else 0,
            "counts": {
                "train": _count_by_source(train_recs),
                "val": _count_by_source(val_recs),
                "test": _count_by_source(test_recs),
            },
        },
    )


def build_portfolio_dataset(
    sim_config: DatasetConfig | None = None,
    *,
    jsonl_path: Path,
    jsonl_config: JsonlDatasetConfig | None = None,
) -> PreparedDataset:
    """Mix simulator (crypto/cross-border typologies) with IBM banking JSONL.

    Normalization is fit on the **combined** train split. Val/test keep both
    sources so metrics can be reported per domain.
    """
    sim_cfg = sim_config or DatasetConfig()
    jcfg = jsonl_config or JsonlDatasetConfig()

    sim_train = collect_split_arrays(sim_cfg.train_seeds, sim_cfg, source=SOURCE_SIM)
    sim_val = collect_split_arrays(sim_cfg.val_seeds, sim_cfg, source=SOURCE_SIM)
    sim_test = collect_split_arrays(sim_cfg.test_seeds, sim_cfg, source=SOURCE_SIM)

    ibm_all = collect_jsonl_snapshots(jsonl_path, jcfg)
    if not ibm_all:
        raise ValueError(
            f"portfolio requires IBM snapshots from {jsonl_path}; got none"
        )
    ibm_train, ibm_val, ibm_test = split_records_chronological(
        ibm_all, train_frac=jcfg.train_frac, val_frac=jcfg.val_frac
    )

    train_recs = sim_train + ibm_train
    val_recs = sim_val + ibm_val
    test_recs = sim_test + ibm_test

    return _prepare_from_records(
        train_recs,
        val_recs,
        test_recs,
        meta={
            "mode": "portfolio",
            "jsonl": str(jsonl_path),
            "ibm_events_snapshotted": ibm_all[-1].event_index,
            "counts": {
                "train": _count_by_source(train_recs),
                "val": _count_by_source(val_recs),
                "test": _count_by_source(test_recs),
            },
        },
    )


def make_tiny_dataset(
    *,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    events: int = 200,
    snapshot_every: int = 100,
    population: PopulationConfig | None = None,
) -> PreparedDataset:
    """Small deterministic dataset for unit tests."""
    pop = population or PopulationConfig(
        banking=40,
        crypto=25,
        cross_border=16,
        mixers=2,
        smart_contracts=2,
        on_ramps=4,
        ip_pool_size=12,
        address_pool_size=10,
        preferred_counterparties=4,
        initial_balance_min=50_000.0,
        initial_balance_max=200_000.0,
    )
    cfg = DatasetConfig(
        train_seeds=train_seeds or [42, 43],
        val_seeds=val_seeds or [50],
        test_seeds=[60],
        events_per_run=events,
        snapshot_every=snapshot_every,
        pattern_event_fraction=0.15,
        population=pop,
    )
    return build_dataset(cfg)


def live_store_to_hetero(
    sim: Simulator,
    schema: FeatureSchema,
    config: FeatureConfig | None = None,
) -> HeteroData:
    """Score helper: convert current simulator store with a fitted schema."""
    return graph_store_to_hetero(
        sim.store,
        sim.clock.now,
        config=config,
        schema=schema,
        normalize=True,
    )
