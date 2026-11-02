"""Train HeteroSAGE on simulator snapshots and write a reloadable checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from aml.convert import NODE_PARTY, REL_TRANSFER, count_positives
from aml.dataset import (
    SOURCE_IBM,
    SOURCE_SIM,
    DatasetConfig,
    JsonlDatasetConfig,
    PreparedDataset,
    build_dataset,
    build_jsonl_dataset,
    build_portfolio_dataset,
    make_tiny_dataset,
)
from aml.domain import Typology
from aml.model import (
    FeatureOnlyBaseline,
    HeteroSAGE,
    ModelConfig,
    ScoreResult,
    build_model_from_checkpoint,
    score_snapshot,
)
from aml.population import PopulationConfig
from aml.report import (
    format_report_text,
    load_report_or_checkpoint,
    write_training_report,
)


def _safe_auc(y_true: list[float], y_score: list[float]) -> float:
    """ROC-AUC without sklearn; returns nan if undefined."""
    if not y_true or len(set(int(y) for y in y_true)) < 2:
        return float("nan")
    pairs = sorted(zip(y_score, y_true), key=lambda t: t[0])
    # Mann–Whitney / Wilcoxon form of AUC
    n_pos = sum(1 for _, y in pairs if y >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y >= 0.5:
            rank_sum += i
    # Average ranks for ties would be more precise; scores are continuous enough.
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(y_true: list[float], y_score: list[float]) -> float:
    """PR-AUC (average precision) without sklearn."""
    if not y_true or sum(1 for y in y_true if y >= 0.5) == 0:
        return float("nan")
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    tp = 0
    fp = 0
    n_pos = sum(1 for y in y_true if y >= 0.5)
    prev_recall = 0.0
    ap = 0.0
    for idx in order:
        if y_true[idx] >= 0.5:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def _f1_at_threshold(
    y_true: list[float], y_score: list[float], threshold: float = 0.5
) -> float:
    tp = fp = fn = 0
    for yt, ys in zip(y_true, y_score, strict=True):
        pred = ys >= threshold
        pos = yt >= 0.5
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif not pred and pos:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class SplitMetrics:
    node_pr_auc: float
    node_roc_auc: float
    node_f1: float
    edge_pr_auc: float
    edge_roc_auc: float
    edge_f1: float
    typology_recall: dict[str, float]

    @property
    def early_stop_score(self) -> float:
        vals = [self.node_pr_auc, self.edge_pr_auc]
        finite = [v for v in vals if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_pr_auc": self.node_pr_auc,
            "node_roc_auc": self.node_roc_auc,
            "node_f1": self.node_f1,
            "edge_pr_auc": self.edge_pr_auc,
            "edge_roc_auc": self.edge_roc_auc,
            "edge_f1": self.edge_f1,
            "typology_recall": self.typology_recall,
            "early_stop_score": self.early_stop_score,
        }


def compute_pos_weight(graphs: list[HeteroData]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (party_pos_weight, edge_pos_weight) scalars for BCEWithLogitsLoss."""
    party_pos = party_neg = 0
    edge_pos = edge_neg = 0
    for g in graphs:
        mask = g[NODE_PARTY].train_mask
        y = g[NODE_PARTY].y
        if mask.any():
            yy = y[mask]
            party_pos += int((yy > 0.5).sum().item())
            party_neg += int((yy <= 0.5).sum().item())
        ey = g[NODE_PARTY, REL_TRANSFER, NODE_PARTY].y
        edge_pos += int((ey > 0.5).sum().item())
        edge_neg += int((ey <= 0.5).sum().item())
    party_w = party_neg / max(party_pos, 1)
    edge_w = edge_neg / max(edge_pos, 1)
    return torch.tensor([party_w], dtype=torch.float32), torch.tensor(
        [edge_w], dtype=torch.float32
    )


def evaluate(model: nn.Module, graphs: list[HeteroData], device: torch.device) -> SplitMetrics:
    model.eval()
    node_true: list[float] = []
    node_score: list[float] = []
    edge_true: list[float] = []
    edge_score: list[float] = []
    typ_true: dict[str, list[float]] = defaultdict(list)
    typ_pred: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            n_logits, e_logits = model(g)
            n_prob = torch.sigmoid(n_logits).cpu()
            e_prob = torch.sigmoid(e_logits).cpu()
            mask = g[NODE_PARTY].train_mask.cpu()
            y = g[NODE_PARTY].y.cpu()
            if mask.any():
                node_true.extend(y[mask].tolist())
                node_score.extend(n_prob[mask].tolist())
            store = g[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
            ey = store.y.cpu()
            edge_true.extend(ey.tolist())
            edge_score.extend(e_prob.tolist())
            typs = list(store.typologies)
            for i, typ in enumerate(typs):
                if typ == Typology.BENIGN.value:
                    continue
                typ_true[typ].append(1.0)
                typ_pred[typ].append(1.0 if e_prob[i].item() >= 0.5 else 0.0)

    typology_recall: dict[str, float] = {}
    for typ, truths in typ_true.items():
        preds = typ_pred[typ]
        if not truths:
            typology_recall[typ] = float("nan")
        else:
            typology_recall[typ] = sum(preds) / len(truths)

    return SplitMetrics(
        node_pr_auc=_average_precision(node_true, node_score),
        node_roc_auc=_safe_auc(node_true, node_score),
        node_f1=_f1_at_threshold(node_true, node_score),
        edge_pr_auc=_average_precision(edge_true, edge_score),
        edge_roc_auc=_safe_auc(edge_true, edge_score),
        edge_f1=_f1_at_threshold(edge_true, edge_score),
        typology_recall=typology_recall,
    )


def graphs_by_source(graphs: list[HeteroData]) -> dict[str, list[HeteroData]]:
    grouped: dict[str, list[HeteroData]] = defaultdict(list)
    for g in graphs:
        src = getattr(g, "source", None) or "unknown"
        grouped[str(src)].append(g)
    return dict(grouped)


def evaluate_by_source(
    model: nn.Module, graphs: list[HeteroData], device: torch.device
) -> dict[str, dict[str, Any]]:
    """Per-domain metrics (sim typologies vs IBM banking) for portfolio reports."""
    out: dict[str, dict[str, Any]] = {}
    for source, subset in sorted(graphs_by_source(graphs).items()):
        metrics = evaluate(model, subset, device)
        row = metrics.to_dict()
        row["n_graphs"] = len(subset)
        out[source] = row
    return out


def batch_loss(
    model: nn.Module,
    batch: list[HeteroData],
    device: torch.device,
    party_criterion: nn.Module,
    edge_criterion: nn.Module,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for g in batch:
        g = g.to(device)
        n_logits, e_logits = model(g)
        mask = g[NODE_PARTY].train_mask
        if mask.any():
            total = total + party_criterion(n_logits[mask], g[NODE_PARTY].y[mask])
        store = g[NODE_PARTY, REL_TRANSFER, NODE_PARTY]
        if store.y.numel() > 0:
            total = total + edge_criterion(e_logits, store.y)
    return total / max(len(batch), 1)


def train_model(
    dataset: PreparedDataset,
    *,
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: torch.device | None = None,
    train_baseline: bool = True,
    hidden_channels: int = 64,
) -> dict[str, Any]:
    """Train HeteroSAGE (and optional baseline); return checkpoint dict + metrics."""
    device = device or torch.device("cpu")
    schema = dataset.schema
    cfg = ModelConfig.from_schema(schema, hidden_channels=hidden_channels)
    model = HeteroSAGE(cfg).to(device)

    party_w, edge_w = compute_pos_weight(dataset.train)
    party_w = party_w.to(device)
    edge_w = edge_w.to(device)
    party_crit = nn.BCEWithLogitsLoss(pos_weight=party_w)
    edge_crit = nn.BCEWithLogitsLoss(pos_weight=edge_w)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -1.0
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    train_graphs = dataset.train
    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle indices each epoch.
        order = torch.randperm(len(train_graphs)).tolist()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            batch = [train_graphs[i] for i in idxs]
            opt.zero_grad()
            loss = batch_loss(model, batch, device, party_crit, edge_crit)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_metrics = evaluate(model, dataset.val, device)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(n_batches, 1),
            "val": val_metrics.to_dict(),
        }
        history.append(row)
        score = val_metrics.early_stop_score
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch {epoch:03d}  loss={row['train_loss']:.4f}  "
            f"val_pr=({val_metrics.node_pr_auc:.3f},{val_metrics.edge_pr_auc:.3f})  "
            f"stop={score:.3f}"
        )
        if stale >= patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, dataset.test, device)
    print("test:", json.dumps(test_metrics.to_dict(), indent=2))
    test_by_source = evaluate_by_source(model, dataset.test, device)
    if len(test_by_source) > 1:
        print("test by source:", json.dumps(test_by_source, indent=2))

    baseline_metrics: dict[str, Any] | None = None
    baseline_by_source: dict[str, dict[str, Any]] | None = None
    if train_baseline:
        baseline_metrics, baseline_by_source = _train_baseline(
            dataset,
            device=device,
            epochs=min(epochs, 20),
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            party_w=party_w,
            edge_w=edge_w,
        )
        print("baseline test:", json.dumps(baseline_metrics, indent=2))
        if baseline_by_source and len(baseline_by_source) > 1:
            print(
                "baseline test by source:",
                json.dumps(baseline_by_source, indent=2),
            )

    ckpt: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "feature_schema": schema.to_dict(),
        "hetero_metadata": {
            "node_types": list(HeteroSAGE(cfg).lin_dict.keys()),
        },
        "dataset_meta": dict(dataset.meta or {}),
        "metrics": {
            "test": test_metrics.to_dict(),
            "test_by_source": test_by_source,
            "best_val_score": best_score,
            "baseline_test": baseline_metrics,
            "baseline_test_by_source": baseline_by_source,
        },
        "history": history,
        "pos_weight": {
            "party": float(party_w.item()),
            "edge": float(edge_w.item()),
        },
    }
    return ckpt


def _train_baseline(
    dataset: PreparedDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    party_w: torch.Tensor,
    edge_w: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    party_in = len(dataset.schema.party_names)
    model = FeatureOnlyBaseline(party_in).to(device)
    party_crit = nn.BCEWithLogitsLoss(pos_weight=party_w)
    edge_crit = nn.BCEWithLogitsLoss(pos_weight=edge_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_graphs = dataset.train
    best_score = -1.0
    best_state = None
    for _epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_graphs)).tolist()
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            batch = [train_graphs[i] for i in idxs]
            opt.zero_grad()
            loss = batch_loss(model, batch, device, party_crit, edge_crit)
            loss.backward()
            opt.step()
        val = evaluate(model, dataset.val, device)
        if val.early_stop_score > best_score:
            best_score = val.early_stop_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    overall = evaluate(model, dataset.test, device).to_dict()
    by_source = evaluate_by_source(model, dataset.test, device)
    return overall, by_source


def save_checkpoint(ckpt: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml-train",
        description=(
            "Train HeteroSAGE on simulator snapshots, IBM JSONL, "
            "or a portfolio mix (sim + IBM)"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/hetero_sage.pt"),
        help="Checkpoint path",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Training report markdown path (default: <out-dir>/training_report.md)",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the training summary report",
    )
    p.add_argument(
        "--print-report",
        action="store_true",
        help="Print the training summary to the terminal after training",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu | mps | cuda",
    )
    p.add_argument("--events", type=int, default=4_000)
    p.add_argument("--snapshot-every", type=int, default=2_000)
    p.add_argument("--train-runs", type=int, default=24)
    p.add_argument("--val-runs", type=int, default=4)
    p.add_argument("--test-runs", type=int, default=4)
    p.add_argument("--pattern-fraction", type=float, default=0.02)
    p.add_argument("--banking", type=int, default=400)
    p.add_argument("--crypto", type=int, default=150)
    p.add_argument("--cross-border", type=int, default=60)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help=(
            "IBM/replay JSONL path. With default settings this builds a "
            "portfolio (sim + IBM); use --ibm-only for JSONL alone"
        ),
    )
    p.add_argument(
        "--ibm-only",
        action="store_true",
        help="Train only on --jsonl (no simulator graphs)",
    )
    p.add_argument(
        "--jsonl-snapshot-every",
        type=int,
        default=None,
        help="Snapshot cadence for JSONL (default: --snapshot-every)",
    )
    p.add_argument(
        "--jsonl-max-events",
        type=int,
        default=None,
        help="Cap JSONL events applied when building snapshots",
    )
    p.add_argument(
        "--jsonl-train-frac",
        type=float,
        default=0.70,
        help="Chronological train fraction for JSONL snapshots",
    )
    p.add_argument(
        "--jsonl-val-frac",
        type=float,
        default=0.15,
        help="Chronological val fraction for JSONL snapshots (rest = test)",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help="Use a tiny dataset (for smoke tests)",
    )
    p.add_argument("--no-baseline", action="store_true")
    return p


def _build_dataset_from_args(args: argparse.Namespace) -> PreparedDataset:
    if args.tiny and args.jsonl is not None:
        raise SystemExit("--tiny cannot be combined with --jsonl")
    if args.ibm_only and args.jsonl is None:
        raise SystemExit("--ibm-only requires --jsonl")
    if args.tiny:
        return make_tiny_dataset()

    sim_cfg = DatasetConfig(
        train_seeds=list(range(100, 100 + args.train_runs)),
        val_seeds=list(range(200, 200 + args.val_runs)),
        test_seeds=list(range(300, 300 + args.test_runs)),
        events_per_run=args.events,
        snapshot_every=args.snapshot_every,
        pattern_event_fraction=args.pattern_fraction,
        population=PopulationConfig(
            banking=args.banking,
            crypto=args.crypto,
            cross_border=args.cross_border,
        ),
    )
    if args.jsonl is None:
        print(
            f"Building sim dataset: train={len(sim_cfg.train_seeds)} "
            f"val={len(sim_cfg.val_seeds)} test={len(sim_cfg.test_seeds)} "
            f"events={sim_cfg.events_per_run}"
        )
        return build_dataset(sim_cfg)

    jcfg = JsonlDatasetConfig(
        snapshot_every=args.jsonl_snapshot_every or args.snapshot_every,
        max_events=args.jsonl_max_events,
        train_frac=args.jsonl_train_frac,
        val_frac=args.jsonl_val_frac,
        source_label=SOURCE_IBM,
        max_transfer_edges=sim_cfg.max_transfer_edges,
    )
    if args.ibm_only:
        print(
            f"Building IBM JSONL dataset from {args.jsonl} "
            f"(snapshot_every={jcfg.snapshot_every})"
        )
        return build_jsonl_dataset(args.jsonl, jcfg)

    print(
        f"Building portfolio dataset: sim runs "
        f"train/val/test="
        f"{len(sim_cfg.train_seeds)}/{len(sim_cfg.val_seeds)}/{len(sim_cfg.test_seeds)} "
        f"+ IBM {args.jsonl}"
    )
    return build_portfolio_dataset(
        sim_cfg, jsonl_path=args.jsonl, jsonl_config=jcfg
    )


def build_parser() -> argparse.ArgumentParser:
    """Backward-compatible alias used by tests / callers."""
    return build_train_parser()


def default_report_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "training_report.md"


def run_train(argv: list[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    device = torch.device(args.device)
    dataset = _build_dataset_from_args(args)

    print(
        f"graphs: train={len(dataset.train)} val={len(dataset.val)} "
        f"test={len(dataset.test)} mode={dataset.meta.get('mode', 'sim')}"
    )
    if dataset.meta.get("counts"):
        print("source counts:", json.dumps(dataset.meta["counts"]))
    if dataset.train:
        print("sample counts:", count_positives(dataset.train[0]))
        src0 = getattr(dataset.train[0], "source", SOURCE_SIM)
        print(f"sample source: {src0}")

    ckpt = train_model(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        train_baseline=not args.no_baseline,
        hidden_channels=args.hidden,
    )
    save_checkpoint(ckpt, args.out)
    print(f"wrote {args.out}")

    report_path = args.report or default_report_path(args.out)
    if not args.no_report:
        report = write_training_report(
            ckpt,
            report_path,
            checkpoint_path=args.out,
            also_json=True,
        )
        print(f"wrote {report_path}")
        print(f"wrote {report_path.with_suffix('.json')}")
        if args.print_report:
            print()
            print(format_report_text(report), end="")

    # Smoke: reload and score first test graph.
    loaded = load_checkpoint(args.out)
    model = build_model_from_checkpoint(loaded)
    if dataset.test:
        scores: ScoreResult = score_snapshot(model, dataset.test[0])
        print(
            f"score_snapshot smoke: {len(scores.party_scores)} parties, "
            f"{len(scores.transfer_scores)} transfers"
        )
    return 0


def run_report_command(argv: list[str] | None = None) -> int:
    """``aml-train report <path>`` — display a saved report or checkpoint summary."""
    p = argparse.ArgumentParser(
        prog="aml-train report",
        description="Display a training summary from a report or checkpoint",
    )
    p.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("artifacts/training_report.json"),
        help="training_report.json / .md / .pt checkpoint",
    )
    p.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
    )
    args = p.parse_args(argv)
    report = load_report_or_checkpoint(args.path)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.format == "markdown":
        from aml.report import format_report_markdown

        print(format_report_markdown(report), end="")
    else:
        print(format_report_text(report), end="")
    return 0


def run(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else None
    # ``aml-train report ...`` delegates to the report viewer.
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "report":
        return run_report_command(argv[1:])
    return run_train(argv)


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
