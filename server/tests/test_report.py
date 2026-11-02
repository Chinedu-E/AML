"""Tests for training summary report generation and CLI display."""

from __future__ import annotations

from pathlib import Path

import pytest

from aml.dataset import make_tiny_dataset
from aml.report import (
    build_training_report,
    format_report_markdown,
    format_report_text,
    load_report_or_checkpoint,
    write_training_report,
)
from aml.train import run_report_command, save_checkpoint, train_model


def _tiny_ckpt() -> dict:
    ds = make_tiny_dataset(
        train_seeds=[42, 43],
        val_seeds=[50],
        events=100,
        snapshot_every=50,
    )
    return train_model(
        ds,
        epochs=2,
        batch_size=2,
        patience=2,
        train_baseline=True,
        device=__import__("torch").device("cpu"),
    )


def test_write_and_display_report(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt()
    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(ckpt, ckpt_path)

    report_path = tmp_path / "training_report.md"
    report = write_training_report(
        ckpt, report_path, checkpoint_path=ckpt_path, also_json=True
    )

    assert report_path.exists()
    json_path = report_path.with_suffix(".json")
    assert json_path.exists()

    md = report_path.read_text(encoding="utf-8")
    assert "Training Summary" in md
    assert "Improvement" in md
    assert "Epoch history" in md
    assert "Test metrics" in md

    text = format_report_text(report)
    assert "AML HeteroSAGE" in text
    assert "Val early-stop score" in text
    assert "epoch" in text.lower() or "Epoch" in text

    # JSON round-trip via loader.
    loaded = load_report_or_checkpoint(json_path)
    assert loaded["improvement"]["epochs_ran"] == report["improvement"]["epochs_ran"]

    # Checkpoint path also works.
    from_ckpt = load_report_or_checkpoint(ckpt_path)
    assert from_ckpt["test"]["node_pr_auc"] == report["test"]["node_pr_auc"]

    # Markdown formatter stays in sync with structured fields.
    md2 = format_report_markdown(report)
    assert "GNN vs feature-only baseline" in md2 or report.get("baseline_test") is None


def test_report_cli_prints(tmp_path: Path, capsys) -> None:
    ckpt = _tiny_ckpt()
    report_path = tmp_path / "training_report.md"
    write_training_report(ckpt, report_path, also_json=True)
    rc = run_report_command([str(report_path.with_suffix(".json"))])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Training Summary" in out
    assert "Test metrics" in out


def test_build_report_marks_best_epoch() -> None:
    fake = {
        "history": [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val": {
                    "node_pr_auc": 0.5,
                    "edge_pr_auc": 0.4,
                    "node_f1": 0.3,
                    "edge_f1": 0.2,
                    "early_stop_score": 0.45,
                },
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "val": {
                    "node_pr_auc": 0.7,
                    "edge_pr_auc": 0.6,
                    "node_f1": 0.5,
                    "edge_f1": 0.4,
                    "early_stop_score": 0.65,
                },
            },
            {
                "epoch": 3,
                "train_loss": 0.75,
                "val": {
                    "node_pr_auc": 0.6,
                    "edge_pr_auc": 0.55,
                    "node_f1": 0.45,
                    "edge_f1": 0.35,
                    "early_stop_score": 0.575,
                },
            },
        ],
        "metrics": {
            "test": {
                "node_pr_auc": 0.7,
                "edge_pr_auc": 0.6,
                "node_roc_auc": 0.7,
                "edge_roc_auc": 0.6,
                "node_f1": 0.5,
                "edge_f1": 0.4,
                "typology_recall": {"structuring": 0.8},
            },
            "best_val_score": 0.65,
            "baseline_test": {
                "node_pr_auc": 0.55,
                "edge_pr_auc": 0.5,
                "node_f1": 0.4,
                "edge_f1": 0.3,
            },
        },
        "model_config": {"hidden_channels": 64, "num_layers": 2, "dropout": 0.2},
        "pos_weight": {"party": 2.0, "edge": 3.0},
    }
    report = build_training_report(fake, checkpoint_path="x.pt")
    assert report["improvement"]["best_epoch"] == 2
    assert [e["is_best"] for e in report["epochs"]] == [False, True, False]
    assert report["comparison"]["node_pr_auc"]["delta"] == pytest.approx(0.15)
    text = format_report_text(report)
    assert "*  2" in text or "  2" in text
    assert "structuring" in text
