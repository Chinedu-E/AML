"""Live WebSocket round-trip for aml-serve."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

from aml.dataset import make_tiny_dataset
from aml.engine import EngineConfig, RealtimeEngine
from aml.population import PopulationConfig
from aml.serve import Hub, handler, producer_loop
from aml.train import save_checkpoint, train_model


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ds = make_tiny_dataset(
        train_seeds=[42, 43],
        val_seeds=[50],
        events=120,
        snapshot_every=60,
    )
    ckpt = train_model(
        ds,
        epochs=2,
        batch_size=2,
        patience=2,
        train_baseline=False,
        device=torch.device("cpu"),
    )
    path = tmp_path_factory.mktemp("ckpt") / "serve_toy.pt"
    save_checkpoint(ckpt, path)
    return path


def _small_pop() -> PopulationConfig:
    return PopulationConfig(
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


def test_websocket_hello_transaction_scores(tiny_checkpoint: Path) -> None:
    try:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets import connect  # type: ignore[no-redef]
        from websockets.server import serve  # type: ignore[no-redef]

    async def _run() -> None:
        cfg = EngineConfig(
            seed=11,
            rate_per_second=200.0,
            batch_size=3,
            max_batch_wait=0.5,
            warmup=20,
            threshold=0.5,
            population=_small_pop(),
            checkpoint=tiny_checkpoint,
            device="cpu",
        )
        engine = RealtimeEngine(cfg)
        engine.warmup(20)
        hub = Hub()
        tick_lock = asyncio.Lock()
        batch: list = []

        async def _handler(websocket) -> None:  # noqa: ANN001
            await handler(
                websocket, engine, hub, tick_lock=tick_lock, batch=batch
            )

        producer = asyncio.create_task(
            producer_loop(engine, hub, tick_lock=tick_lock, batch=batch)
        )
        try:
            async with serve(_handler, "127.0.0.1", 0) as server:
                sockets = server.sockets
                assert sockets
                port = sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    hello_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    hello = json.loads(hello_raw)
                    assert hello["type"] == "hello"
                    assert hello["seed"] == 11
                    assert hello["source"] == "sim"
                    assert hello["sources"] == ["sim"]

                    snap_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    snap = json.loads(snap_raw)
                    assert snap["type"] == "snapshot"

                    saw_tx = False
                    saw_scores = False
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + 15.0
                    while loop.time() < deadline:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "transaction":
                            saw_tx = True
                            assert "tx_id" in msg
                            assert msg.get("transfer_risk") is None
                        elif msg.get("type") == "scores":
                            saw_scores = True
                            assert "transfer_scores" in msg
                            assert "party_scores" in msg
                            assert msg["batch_id"] >= 1
                        if saw_tx and saw_scores:
                            break
                    assert saw_tx, "expected at least one transaction message"
                    assert saw_scores, "expected at least one scores message"
        finally:
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


class _FakeEngine:
    def hello_dict(self) -> dict:
        return {
            "type": "hello",
            "seed": 1,
            "rate": 1,
            "checkpoint": "toy",
            "networks": ["banking", "crypto", "cross_border"],
            "sim_time": 0,
            "warmup_done": 0,
            "threshold": 0.5,
            "reveal_labels": False,
            "source": "sim",
            "sources": ["sim"],
        }

    def snapshot_dict(self) -> dict:
        return {
            "type": "snapshot",
            "sim_time": 0,
            "party_scores": {},
            "recent_transactions": [],
            "stats": {},
        }

    def set_source(self, name: str) -> bool:
        return False


def test_handler_replays_alerts_and_dispositions() -> None:
    class WS:
        def __init__(self) -> None:
            self.inbox: asyncio.Queue[str | None] = asyncio.Queue()
            self.outbox: asyncio.Queue[str] = asyncio.Queue()

        def __aiter__(self) -> "WS":
            return self

        async def __anext__(self) -> str:
            msg = await self.inbox.get()
            if msg is None:
                raise StopAsyncIteration
            return msg

        async def send(self, data: str) -> None:
            await self.outbox.put(data)

    async def _run() -> None:
        hub = Hub()
        alert = {
            "type": "alert",
            "alert_id": "a1",
            "kind": "party",
            "subject_id": "bank-0001",
            "score": 0.91,
            "network": "banking",
            "source_id": None,
            "dest_id": None,
            "reason": "Flagged High Risk",
            "neighborhood": {
                "nodes": [],
                "edges": [],
                "center": "bank-0001",
                "k": 2,
            },
        }
        await hub.note_alert(alert)
        await hub.note_alert({"type": "alert", "reason": "missing id"})

        ws = WS()
        task = asyncio.create_task(handler(ws, _FakeEngine(), hub))
        seen: list[dict] = []
        while True:
            raw = await asyncio.wait_for(ws.outbox.get(), timeout=2)
            seen.append(json.loads(raw))
            if any(item.get("type") == "cases" for item in seen):
                break

        assert [item["type"] for item in seen[:3]] == ["hello", "snapshot", "alert"]
        assert seen[2]["alert_id"] == "a1"
        assert seen[3]["type"] == "cases"
        assert seen[3]["dispositions"] == []

        await ws.inbox.put(
            json.dumps({"op": "disposition", "alert_id": "a1", "action": "nope"})
        )
        await ws.inbox.put(
            json.dumps(
                {"op": "disposition", "alert_id": "missing", "action": "approve"}
            )
        )
        await ws.inbox.put(
            json.dumps({"op": "disposition", "alert_id": "a1", "action": "escalate"})
        )

        disposition = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2
        while loop.time() < deadline:
            raw = await asyncio.wait_for(ws.outbox.get(), timeout=2)
            msg = json.loads(raw)
            if msg.get("type") == "disposition":
                disposition = msg
                break
        assert disposition == {
            "type": "disposition",
            "alert_id": "a1",
            "action": "escalate",
        }

        replayed = await hub.replay()
        assert replayed[-1]["dispositions"] == [
            {"alert_id": "a1", "action": "escalate"}
        ]

        await ws.inbox.put(None)
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(_run())


def test_serve_replay_requires_path(capsys) -> None:
    from aml.serve import run

    # --no-replay disables the shipped demo auto-load.
    code = run(["--source", "replay", "--no-replay"])
    assert code == 1
    err = capsys.readouterr().err
    assert "requires --replay" in err


def test_resolve_replay_path_auto_demo(tmp_path: Path, monkeypatch) -> None:
    from aml import serve as serve_mod
    from aml.serve import build_parser, resolve_replay_path

    demo = tmp_path / "ibm_hi_small_demo.jsonl"
    demo.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(serve_mod, "DEMO_REPLAY_PATH", demo)

    auto = build_parser().parse_args([])
    assert resolve_replay_path(auto) == demo

    explicit = build_parser().parse_args(["--replay", str(tmp_path / "other.jsonl")])
    assert resolve_replay_path(explicit) == tmp_path / "other.jsonl"

    disabled = build_parser().parse_args(["--no-replay"])
    assert resolve_replay_path(disabled) is None


def _write_mini_jsonl(path: Path, n: int = 40) -> None:
    rows = []
    for i in range(n):
        rows.append(
            {
                "tx_id": f"tx-{i:04d}",
                "timestamp": float(i),
                "source_id": f"bank-{(i % 5) + 1:04d}",
                "dest_id": f"bank-{((i + 1) % 5) + 1:04d}",
                "amount": 1000.0 + i,
                "currency": "USD",
                "channel": "ach",
                "network": "banking",
                "typology": "benign",
                "pattern_id": None,
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_engine_registers_both_sources(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    from aml.serve import build_engine_from_args, build_parser

    replay = tmp_path / "mini.jsonl"
    _write_mini_jsonl(replay)
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(tiny_checkpoint),
            "--source",
            "sim",
            "--replay",
            str(replay),
            "--warmup",
            "5",
            "--banking",
            "20",
            "--crypto",
            "10",
            "--cross-border",
            "8",
        ]
    )
    engine = build_engine_from_args(args)
    assert engine.source.name == "sim"
    assert engine.available_sources == ["sim", "replay"]
    hello = engine.hello_dict()
    assert hello["source"] == "sim"
    assert hello["sources"] == ["sim", "replay"]
    assert hello["replay"] == str(replay)


def test_source_switch_clears_hub_and_updates_hello(
    tiny_checkpoint: Path, tmp_path: Path
) -> None:
    try:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
    except ImportError:
        from websockets import connect  # type: ignore[no-redef]
        from websockets.server import serve  # type: ignore[no-redef]

    from aml.serve import build_engine_from_args, build_parser

    replay = tmp_path / "mini.jsonl"
    _write_mini_jsonl(replay, n=80)

    async def _run() -> None:
        args = build_parser().parse_args(
            [
                "--checkpoint",
                str(tiny_checkpoint),
                "--source",
                "sim",
                "--replay",
                str(replay),
                "--warmup",
                "5",
                "--rate",
                "200",
                "--batch-size",
                "3",
                "--banking",
                "20",
                "--crypto",
                "10",
                "--cross-border",
                "8",
            ]
        )
        engine = build_engine_from_args(args)
        hub = Hub()
        tick_lock = asyncio.Lock()
        batch: list = []

        async def _handler(websocket) -> None:  # noqa: ANN001
            await handler(
                websocket, engine, hub, tick_lock=tick_lock, batch=batch
            )

        producer = asyncio.create_task(
            producer_loop(engine, hub, tick_lock=tick_lock, batch=batch)
        )
        try:
            async with serve(_handler, "127.0.0.1", 0) as server:
                sockets = server.sockets
                assert sockets
                port = sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    assert hello["type"] == "hello"
                    assert hello["source"] == "sim"
                    snap = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    assert snap["type"] == "snapshot"

                    # Seed an alert so a switch must clear the hub ring.
                    await hub.note_alert(
                        {
                            "type": "alert",
                            "alert_id": "a-old",
                            "kind": "party",
                            "subject_id": "bank-0001",
                            "score": 0.9,
                            "network": "banking",
                            "source_id": None,
                            "dest_id": None,
                            "reason": "stale",
                            "neighborhood": {
                                "nodes": [],
                                "edges": [],
                                "center": "bank-0001",
                                "k": 2,
                            },
                        }
                    )

                    await ws.send(json.dumps({"op": "source", "name": "replay"}))

                    saw_reset = False
                    saw_hello = False
                    saw_snapshot = False
                    saw_cases = False
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + 10.0
                    while loop.time() < deadline:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "reset":
                            saw_reset = True
                        elif msg.get("type") == "hello" and saw_reset:
                            assert msg["source"] == "replay"
                            assert msg["sources"] == ["sim", "replay"]
                            assert msg["replay"] == str(replay)
                            saw_hello = True
                        elif msg.get("type") == "snapshot" and saw_hello:
                            saw_snapshot = True
                        elif msg.get("type") == "cases" and saw_snapshot:
                            assert msg["dispositions"] == []
                            saw_cases = True
                            break
                    assert saw_reset and saw_hello and saw_snapshot and saw_cases
                    replayed = await hub.replay()
                    assert replayed == [
                        {"type": "cases", "dispositions": []}
                    ]
                    assert engine.source.name == "replay"
        finally:
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_producer_pauses_without_clients(tiny_checkpoint: Path) -> None:
    """No scoring while the hub is empty; production resumes after register."""

    class CountingEngine:
        def __init__(self, real: RealtimeEngine) -> None:
            self._real = real
            self.pulls = 0
            self.config = real.config
            self.source_epoch = real.source_epoch

        def next_transaction(self):  # noqa: ANN201
            self.pulls += 1
            return self._real.next_transaction()

        def score_batch(self, txs):  # noqa: ANN001, ANN201
            return self._real.score_batch(txs)

        def tx_wire_dict(self, tx, *, transfer_risk=None):  # noqa: ANN001, ANN201
            return self._real.tx_wire_dict(tx, transfer_risk=transfer_risk)

    async def _run() -> None:
        cfg = EngineConfig(
            seed=3,
            rate_per_second=50.0,
            batch_size=2,
            max_batch_wait=0.2,
            warmup=10,
            threshold=0.5,
            population=_small_pop(),
            checkpoint=tiny_checkpoint,
            device="cpu",
        )
        real = RealtimeEngine(cfg)
        real.warmup(10)
        engine = CountingEngine(real)
        hub = Hub()
        tick_lock = asyncio.Lock()
        batch: list = []
        producer = asyncio.create_task(
            producer_loop(engine, hub, tick_lock=tick_lock, batch=batch)
        )
        try:
            await asyncio.sleep(0.15)
            assert engine.pulls == 0
            assert hub.client_count == 0

            class DummyWS:
                pass

            client = DummyWS()
            await hub.register(client)
            deadline = asyncio.get_running_loop().time() + 3.0
            while engine.pulls == 0 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
            assert engine.pulls >= 1

            await hub.unregister(client)
            assert hub.client_count == 0
            pulls_at_idle = engine.pulls
            await asyncio.sleep(0.2)
            assert engine.pulls == pulls_at_idle
        finally:
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_health_request_returns_ok() -> None:
    from aml.serve import health_request

    class Conn:
        def respond(self, status: int, text: str):
            return {"status": status, "text": text}

    class Req:
        def __init__(self, path: str) -> None:
            self.path = path

    async def _run() -> None:
        ok = await health_request(Conn(), Req("/health"))
        assert ok == {"status": 200, "text": "ok\n"}
        # Root must stay open for the WebSocket upgrade.
        assert await health_request(Conn(), Req("/")) is None
        assert await health_request(Conn(), Req("/ws")) is None

    asyncio.run(_run())
