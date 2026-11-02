"""Training summary report: build, write, and display scores / improvement."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def _delta(a: float | None, b: float | None) -> str:
    """Format b - a with a sign; positive means improvement for AUC metrics."""
    if a is None or b is None:
        return "n/a"
    if isinstance(a, float) and math.isnan(a):
        return "n/a"
    if isinstance(b, float) and math.isnan(b):
        return "n/a"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _metric(block: dict[str, Any] | None, key: str) -> float | None:
    if not block:
        return None
    v = block.get(key)
    if v is None:
        return None
    return float(v)


def build_training_report(
    ckpt: dict[str, Any],
    *,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Derive a structured summary from an ``aml-train`` checkpoint."""
    history: list[dict[str, Any]] = list(ckpt.get("history") or [])
    metrics = ckpt.get("metrics") or {}
    test = metrics.get("test") or {}
    baseline = metrics.get("baseline_test")
    best_val = metrics.get("best_val_score")

    epochs: list[dict[str, Any]] = []
    best_epoch = None
    best_score = -1.0
    for row in history:
        val = row.get("val") or {}
        score = float(val.get("early_stop_score") or 0.0)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_epoch = int(row["epoch"])
        epochs.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row.get("train_loss", 0.0)),
                "val_node_pr_auc": _metric(val, "node_pr_auc"),
                "val_edge_pr_auc": _metric(val, "edge_pr_auc"),
                "val_node_f1": _metric(val, "node_f1"),
                "val_edge_f1": _metric(val, "edge_f1"),
                "early_stop_score": score,
                "is_best": improved and best_epoch == int(row["epoch"]),
            }
        )
    # Mark only the final best epoch (last improvement).
    for e in epochs:
        e["is_best"] = best_epoch is not None and e["epoch"] == best_epoch

    first = epochs[0] if epochs else None
    last = epochs[-1] if epochs else None
    best_row = next((e for e in epochs if e["is_best"]), last)

    improvement = {
        "epochs_ran": len(epochs),
        "best_epoch": best_epoch,
        "best_val_score": float(best_val) if best_val is not None else (
            best_row["early_stop_score"] if best_row else None
        ),
        "train_loss_first": first["train_loss"] if first else None,
        "train_loss_last": last["train_loss"] if last else None,
        "train_loss_delta": (
            (last["train_loss"] - first["train_loss"]) if first and last else None
        ),
        "val_score_first": first["early_stop_score"] if first else None,
        "val_score_best": best_row["early_stop_score"] if best_row else None,
        "val_score_delta_first_to_best": (
            (best_row["early_stop_score"] - first["early_stop_score"])
            if first and best_row
            else None
        ),
        "val_node_pr_first": first["val_node_pr_auc"] if first else None,
        "val_node_pr_best": best_row["val_node_pr_auc"] if best_row else None,
        "val_edge_pr_first": first["val_edge_pr_auc"] if first else None,
        "val_edge_pr_best": best_row["val_edge_pr_auc"] if best_row else None,
    }

    comparison: dict[str, Any] | None = None
    if baseline:
        comparison = {
            "node_pr_auc": {
                "gnn": _metric(test, "node_pr_auc"),
                "baseline": _metric(baseline, "node_pr_auc"),
                "delta": None,
            },
            "edge_pr_auc": {
                "gnn": _metric(test, "edge_pr_auc"),
                "baseline": _metric(baseline, "edge_pr_auc"),
                "delta": None,
            },
            "node_f1": {
                "gnn": _metric(test, "node_f1"),
                "baseline": _metric(baseline, "node_f1"),
                "delta": None,
            },
            "edge_f1": {
                "gnn": _metric(test, "edge_f1"),
                "baseline": _metric(baseline, "edge_f1"),
                "delta": None,
            },
        }
        for key, block in comparison.items():
            g = block["gnn"]
            b = block["baseline"]
            if g is not None and b is not None and not (
                math.isnan(g) or math.isnan(b)
            ):
                block["delta"] = g - b

    metrics_block = metrics
    test_by_source = metrics_block.get("test_by_source") or {}
    baseline_by_source = metrics_block.get("baseline_test_by_source") or {}
    dataset_meta = ckpt.get("dataset_meta") or {}

    portfolio: dict[str, Any] | None = None
    if test_by_source or dataset_meta.get("mode") in ("portfolio", "ibm"):
        sources: dict[str, Any] = {}
        for name, block in sorted(test_by_source.items()):
            base = (baseline_by_source or {}).get(name) or {}
            sources[name] = {
                "n_graphs": block.get("n_graphs"),
                "node_pr_auc": _metric(block, "node_pr_auc"),
                "edge_pr_auc": _metric(block, "edge_pr_auc"),
                "node_f1": _metric(block, "node_f1"),
                "edge_f1": _metric(block, "edge_f1"),
                "typology_recall": block.get("typology_recall") or {},
                "baseline_edge_pr_auc": _metric(base, "edge_pr_auc"),
                "baseline_node_pr_auc": _metric(base, "node_pr_auc"),
            }
        portfolio = {
            "mode": dataset_meta.get("mode"),
            "jsonl": dataset_meta.get("jsonl"),
            "counts": dataset_meta.get("counts") or {},
            "sources": sources,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": checkpoint_path,
        "model_config": ckpt.get("model_config") or {},
        "pos_weight": ckpt.get("pos_weight") or {},
        "dataset_meta": dataset_meta,
        "epochs": epochs,
        "improvement": improvement,
        "test": test,
        "baseline_test": baseline,
        "comparison": comparison,
        "typology_recall": (test.get("typology_recall") or {}),
        "portfolio": portfolio,
    }


def format_report_text(report: dict[str, Any]) -> str:
    """Plain-text report suitable for CLI stdout."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("AML HeteroSAGE — Training Summary")
    lines.append("=" * 64)
    if report.get("checkpoint_path"):
        lines.append(f"Checkpoint:  {report['checkpoint_path']}")
    lines.append(f"Generated:   {report.get('generated_at', 'n/a')}")
    cfg = report.get("model_config") or {}
    if cfg:
        lines.append(
            f"Model:       hidden={cfg.get('hidden_channels')} "
            f"layers={cfg.get('num_layers')} dropout={cfg.get('dropout')}"
        )
    pw = report.get("pos_weight") or {}
    if pw:
        lines.append(
            f"Pos weight:  party={_fmt(pw.get('party'))}  edge={_fmt(pw.get('edge'))}"
        )

    imp = report.get("improvement") or {}
    lines.append("")
    lines.append("Improvement")
    lines.append("-" * 64)
    lines.append(
        f"  Epochs ran:           {imp.get('epochs_ran', 0)}  "
        f"(best @ epoch {imp.get('best_epoch', 'n/a')})"
    )
    lines.append(
        f"  Train loss:           {_fmt(imp.get('train_loss_first'))} → "
        f"{_fmt(imp.get('train_loss_last'))}  "
        f"(Δ {_fmt(imp.get('train_loss_delta'))})"
    )
    lines.append(
        f"  Val early-stop score: {_fmt(imp.get('val_score_first'))} → "
        f"{_fmt(imp.get('val_score_best'))}  "
        f"(Δ {_delta(imp.get('val_score_first'), imp.get('val_score_best'))})"
    )
    lines.append(
        f"  Val node PR-AUC:      {_fmt(imp.get('val_node_pr_first'))} → "
        f"{_fmt(imp.get('val_node_pr_best'))}  "
        f"(Δ {_delta(imp.get('val_node_pr_first'), imp.get('val_node_pr_best'))})"
    )
    lines.append(
        f"  Val edge PR-AUC:      {_fmt(imp.get('val_edge_pr_first'))} → "
        f"{_fmt(imp.get('val_edge_pr_best'))}  "
        f"(Δ {_delta(imp.get('val_edge_pr_first'), imp.get('val_edge_pr_best'))})"
    )

    lines.append("")
    lines.append("Epoch history (validation)")
    lines.append("-" * 64)
    lines.append(
        f"  {'ep':>3}  {'loss':>8}  {'nodePR':>7}  {'edgePR':>7}  "
        f"{'nodeF1':>7}  {'edgeF1':>7}  {'score':>7}"
    )
    for e in report.get("epochs") or []:
        mark = " *" if e.get("is_best") else "  "
        lines.append(
            f"{mark}{e['epoch']:3d}  {_fmt(e['train_loss']):>8}  "
            f"{_fmt(e['val_node_pr_auc'], 3):>7}  "
            f"{_fmt(e['val_edge_pr_auc'], 3):>7}  "
            f"{_fmt(e['val_node_f1'], 3):>7}  "
            f"{_fmt(e['val_edge_f1'], 3):>7}  "
            f"{_fmt(e['early_stop_score'], 3):>7}"
        )
    if report.get("epochs"):
        lines.append("  (* = best validation early-stop score)")

    test = report.get("test") or {}
    lines.append("")
    lines.append("Test metrics (best checkpoint)")
    lines.append("-" * 64)
    lines.append(
        f"  Node  PR-AUC={_fmt(test.get('node_pr_auc'))}  "
        f"ROC-AUC={_fmt(test.get('node_roc_auc'))}  "
        f"F1={_fmt(test.get('node_f1'))}"
    )
    lines.append(
        f"  Edge  PR-AUC={_fmt(test.get('edge_pr_auc'))}  "
        f"ROC-AUC={_fmt(test.get('edge_roc_auc'))}  "
        f"F1={_fmt(test.get('edge_f1'))}"
    )

    portfolio = report.get("portfolio")
    if portfolio and portfolio.get("sources"):
        lines.append("")
        lines.append(
            f"Portfolio by source (mode={portfolio.get('mode', 'n/a')})"
        )
        lines.append("-" * 64)
        if portfolio.get("jsonl"):
            lines.append(f"  JSONL: {portfolio['jsonl']}")
        lines.append(
            f"  {'source':8s}  {'graphs':>6}  {'nodePR':>7}  {'edgePR':>7}  "
            f"{'nodeF1':>7}  {'edgeF1':>7}"
        )
        for name, block in portfolio["sources"].items():
            lines.append(
                f"  {name:8s}  {_fmt(block.get('n_graphs'), 0):>6}  "
                f"{_fmt(block.get('node_pr_auc'), 3):>7}  "
                f"{_fmt(block.get('edge_pr_auc'), 3):>7}  "
                f"{_fmt(block.get('node_f1'), 3):>7}  "
                f"{_fmt(block.get('edge_f1'), 3):>7}"
            )
            typ_src = block.get("typology_recall") or {}
            for typ_name in sorted(typ_src):
                lines.append(
                    f"           recall {typ_name}: {_fmt(typ_src[typ_name])}"
                )

    typ = report.get("typology_recall") or {}
    if typ:
        lines.append("")
        lines.append("Typology recall (test edges, threshold 0.5)")
        lines.append("-" * 64)
        for name in sorted(typ):
            lines.append(f"  {name:16s}  {_fmt(typ[name])}")

    comp = report.get("comparison")
    if comp:
        lines.append("")
        lines.append("GNN vs feature-only baseline (test)")
        lines.append("-" * 64)
        lines.append(
            f"  {'metric':16s}  {'GNN':>8}  {'baseline':>8}  {'Δ':>8}"
        )
        for key, label in (
            ("node_pr_auc", "node PR-AUC"),
            ("edge_pr_auc", "edge PR-AUC"),
            ("node_f1", "node F1"),
            ("edge_f1", "edge F1"),
        ):
            block = comp.get(key) or {}
            lines.append(
                f"  {label:16s}  {_fmt(block.get('gnn')):>8}  "
                f"{_fmt(block.get('baseline')):>8}  "
                f"{_delta(block.get('baseline'), block.get('gnn')):>8}"
            )

    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines) + "\n"


def format_report_markdown(report: dict[str, Any]) -> str:
    """Markdown report for artifacts/training_report.md."""
    lines: list[str] = []
    lines.append("# AML HeteroSAGE — Training Summary")
    lines.append("")
    if report.get("checkpoint_path"):
        lines.append(f"- **Checkpoint:** `{report['checkpoint_path']}`")
    lines.append(f"- **Generated:** {report.get('generated_at', 'n/a')}")
    cfg = report.get("model_config") or {}
    if cfg:
        lines.append(
            f"- **Model:** hidden={cfg.get('hidden_channels')}, "
            f"layers={cfg.get('num_layers')}, dropout={cfg.get('dropout')}"
        )
    pw = report.get("pos_weight") or {}
    if pw:
        lines.append(
            f"- **Pos weight:** party={_fmt(pw.get('party'))}, "
            f"edge={_fmt(pw.get('edge'))}"
        )

    imp = report.get("improvement") or {}
    lines.append("")
    lines.append("## Improvement")
    lines.append("")
    lines.append("| Metric | First | Best / Last | Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| Train loss | {_fmt(imp.get('train_loss_first'))} | "
        f"{_fmt(imp.get('train_loss_last'))} | "
        f"{_fmt(imp.get('train_loss_delta'))} |"
    )
    lines.append(
        f"| Val early-stop score | {_fmt(imp.get('val_score_first'))} | "
        f"{_fmt(imp.get('val_score_best'))} | "
        f"{_delta(imp.get('val_score_first'), imp.get('val_score_best'))} |"
    )
    lines.append(
        f"| Val node PR-AUC | {_fmt(imp.get('val_node_pr_first'))} | "
        f"{_fmt(imp.get('val_node_pr_best'))} | "
        f"{_delta(imp.get('val_node_pr_first'), imp.get('val_node_pr_best'))} |"
    )
    lines.append(
        f"| Val edge PR-AUC | {_fmt(imp.get('val_edge_pr_first'))} | "
        f"{_fmt(imp.get('val_edge_pr_best'))} | "
        f"{_delta(imp.get('val_edge_pr_first'), imp.get('val_edge_pr_best'))} |"
    )
    lines.append("")
    lines.append(
        f"Best checkpoint selected at **epoch {imp.get('best_epoch', 'n/a')}** "
        f"after **{imp.get('epochs_ran', 0)}** epoch(s)."
    )

    lines.append("")
    lines.append("## Epoch history")
    lines.append("")
    lines.append(
        "| Epoch | Train loss | Node PR-AUC | Edge PR-AUC | "
        "Node F1 | Edge F1 | Score | Best |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for e in report.get("epochs") or []:
        best = "✓" if e.get("is_best") else ""
        lines.append(
            f"| {e['epoch']} | {_fmt(e['train_loss'])} | "
            f"{_fmt(e['val_node_pr_auc'])} | {_fmt(e['val_edge_pr_auc'])} | "
            f"{_fmt(e['val_node_f1'])} | {_fmt(e['val_edge_f1'])} | "
            f"{_fmt(e['early_stop_score'])} | {best} |"
        )

    test = report.get("test") or {}
    lines.append("")
    lines.append("## Test metrics")
    lines.append("")
    lines.append("| Head | PR-AUC | ROC-AUC | F1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| Node | {_fmt(test.get('node_pr_auc'))} | "
        f"{_fmt(test.get('node_roc_auc'))} | {_fmt(test.get('node_f1'))} |"
    )
    lines.append(
        f"| Edge | {_fmt(test.get('edge_pr_auc'))} | "
        f"{_fmt(test.get('edge_roc_auc'))} | {_fmt(test.get('edge_f1'))} |"
    )

    portfolio = report.get("portfolio")
    if portfolio and portfolio.get("sources"):
        lines.append("")
        lines.append("## Portfolio by source")
        lines.append("")
        lines.append(f"- **Mode:** `{portfolio.get('mode', 'n/a')}`")
        if portfolio.get("jsonl"):
            lines.append(f"- **JSONL:** `{portfolio['jsonl']}`")
        lines.append("")
        lines.append(
            "| Source | Graphs | Node PR-AUC | Edge PR-AUC | Node F1 | Edge F1 |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, block in portfolio["sources"].items():
            lines.append(
                f"| {name} | {_fmt(block.get('n_graphs'), 0)} | "
                f"{_fmt(block.get('node_pr_auc'))} | "
                f"{_fmt(block.get('edge_pr_auc'))} | "
                f"{_fmt(block.get('node_f1'))} | "
                f"{_fmt(block.get('edge_f1'))} |"
            )
        # Per-source typology recall when present.
        for name, block in portfolio["sources"].items():
            typ_src = block.get("typology_recall") or {}
            if not typ_src:
                continue
            lines.append("")
            lines.append(f"### {name} typology recall")
            lines.append("")
            lines.append("| Typology | Recall |")
            lines.append("| --- | ---: |")
            for typ_name in sorted(typ_src):
                lines.append(f"| {typ_name} | {_fmt(typ_src[typ_name])} |")

    typ = report.get("typology_recall") or {}
    if typ:
        lines.append("")
        lines.append("## Typology recall")
        lines.append("")
        lines.append("| Typology | Recall |")
        lines.append("| --- | ---: |")
        for name in sorted(typ):
            lines.append(f"| {name} | {_fmt(typ[name])} |")

    comp = report.get("comparison")
    if comp:
        lines.append("")
        lines.append("## GNN vs feature-only baseline")
        lines.append("")
        lines.append("| Metric | GNN | Baseline | Δ (GNN − baseline) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key, label in (
            ("node_pr_auc", "Node PR-AUC"),
            ("edge_pr_auc", "Edge PR-AUC"),
            ("node_f1", "Node F1"),
            ("edge_f1", "Edge F1"),
        ):
            block = comp.get(key) or {}
            lines.append(
                f"| {label} | {_fmt(block.get('gnn'))} | "
                f"{_fmt(block.get('baseline'))} | "
                f"{_delta(block.get('baseline'), block.get('gnn'))} |"
            )

    lines.append("")
    return "\n".join(lines)


def write_training_report(
    ckpt: dict[str, Any],
    report_path: Path,
    *,
    checkpoint_path: str | Path | None = None,
    also_json: bool = True,
) -> dict[str, Any]:
    """Write markdown (and optional JSON) report; return the structured report."""
    report = build_training_report(
        ckpt,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report_markdown(report), encoding="utf-8")
    if also_json:
        json_path = report_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def load_report_or_checkpoint(path: Path) -> dict[str, Any]:
    """Load a JSON report dict, or build one from a ``.pt`` checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"report/checkpoint not found: {path}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".md":
        # Prefer sibling JSON if present; otherwise require checkpoint.
        sibling = path.with_suffix(".json")
        if sibling.exists():
            return json.loads(sibling.read_text(encoding="utf-8"))
        raise ValueError(
            f"cannot display markdown alone; use {sibling.name} or a .pt checkpoint"
        )
    # Assume torch checkpoint.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return build_training_report(ckpt, checkpoint_path=str(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml-report",
        description="Display an AML training summary from a report or checkpoint",
    )
    p.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("artifacts/training_report.json"),
        help="Path to training_report.json, .md (uses sibling .json), or .pt checkpoint",
    )
    p.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format (default: text)",
    )
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = load_report_or_checkpoint(args.path)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(format_report_markdown(report), end="")
    else:
        print(format_report_text(report), end="")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
