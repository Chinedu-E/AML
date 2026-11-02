"""Asyncio WebSocket server for the real-time AML inference engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.exceptions import ConnectionClosed

from aml.domain import Network, Transaction
from aml.engine import EngineConfig, RealtimeEngine, open_checkpoint
from aml.generator import SimulatorConfig
from aml.population import PopulationConfig
from aml.sources import TransactionSource, build_replay_source, build_sim_source

# Drop a client if its outbound queue grows beyond this.
OUTBOUND_QUEUE_CAP = 256
ALERT_RING = 100
DISPOSITIONS = {"approve", "escalate"}
ALL_NETWORKS = {n.value for n in Network}


@dataclass
class ClientState:
    websocket: Any
    networks: set[str] = field(default_factory=lambda: set(ALL_NETWORKS))
    queue: asyncio.Queue[dict[str, Any] | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTBOUND_QUEUE_CAP)
    )
    writer_task: asyncio.Task[None] | None = None


class Hub:
    """Fan-out hub with per-connection network filters and bounded queues.

    Alert history and analyst dispositions live here, in memory, for the
    process lifetime (or until a source switch clears them). They are not
    part of model state.
    """

    def __init__(self) -> None:
        self._clients: dict[Any, ClientState] = {}
        self._lock = asyncio.Lock()
        self._alerts: deque[dict[str, Any]] = deque(maxlen=ALERT_RING)
        self._seen_alerts: set[str] = set()
        self._dispositions: dict[str, str] = {}

    async def register(self, websocket: Any) -> ClientState:
        state = ClientState(websocket=websocket)
        async with self._lock:
            self._clients[websocket] = state
        return state

    async def unregister(self, websocket: Any) -> None:
        async with self._lock:
            state = self._clients.pop(websocket, None)
        if state is None:
            return
        if state.writer_task is not None and not state.writer_task.done():
            state.writer_task.cancel()
            try:
                await state.writer_task
            except asyncio.CancelledError:
                pass
        # Unblock writer if waiting.
        try:
            state.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def set_networks(self, websocket: Any, networks: list[str]) -> None:
        state = self._clients.get(websocket)
        if state is None:
            return
        cleaned = {n for n in networks if n in ALL_NETWORKS}
        state.networks = cleaned or set(ALL_NETWORKS)

    async def note_alert(self, message: dict[str, Any]) -> None:
        """Remember an alert so late subscribers and dispositions can see it."""
        alert_id = message.get("alert_id")
        if not isinstance(alert_id, str) or not alert_id:
            return
        async with self._lock:
            self._seen_alerts.add(alert_id)
            self._alerts.append(dict(message))

    async def apply_disposition(
        self, alert_id: Any, action: Any
    ) -> dict[str, Any] | None:
        """Record approve/escalate for a known alert. Unknown ids are ignored."""
        if not isinstance(alert_id, str) or not alert_id:
            return None
        if not isinstance(action, str) or action not in DISPOSITIONS:
            return None
        async with self._lock:
            if alert_id not in self._seen_alerts:
                return None
            self._dispositions[alert_id] = action
            return {
                "type": "disposition",
                "alert_id": alert_id,
                "action": action,
            }

    async def clear_session(self) -> None:
        """Drop alert history, dispositions, and queued outbound messages."""
        async with self._lock:
            self._alerts.clear()
            self._seen_alerts.clear()
            self._dispositions.clear()
            clients = list(self._clients.values())
        for state in clients:
            while True:
                try:
                    state.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def replay(self) -> list[dict[str, Any]]:
        """Alerts still in the ring, then the current disposition map."""
        async with self._lock:
            alerts = [dict(msg) for msg in self._alerts]
            cases = {
                "type": "cases",
                "dispositions": [
                    {"alert_id": alert_id, "action": action}
                    for alert_id, action in self._dispositions.items()
                ],
            }
        return [*alerts, cases]

    async def broadcast(
        self,
        message: dict[str, Any],
        *,
        network: str | None = None,
    ) -> None:
        """Enqueue ``message`` for clients subscribed to ``network`` (or all)."""
        dead: list[Any] = []
        async with self._lock:
            clients = list(self._clients.values())
        for state in clients:
            if network is not None and network not in state.networks:
                continue
            try:
                state.queue.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(state.websocket)
        for ws in dead:
            try:
                await ws.close(code=1013, reason="outbound queue overflow")
            except Exception:
                pass
            await self.unregister(ws)


async def _writer_loop(state: ClientState) -> None:
    try:
        while True:
            msg = await state.queue.get()
            if msg is None:
                return
            await state.websocket.send(json.dumps(msg, sort_keys=True))
    except ConnectionClosed:
        return


def _format_metrics(metrics: dict[str, float]) -> str:
    def _f(v: float) -> str:
        if math.isnan(v):
            return "nan"
        return f"{v:.2f}"

    return (
        f"p={_f(metrics.get('precision', float('nan')))} "
        f"r={_f(metrics.get('recall', float('nan')))} "
        f"tp={int(metrics.get('tp', 0))} "
        f"fp={int(metrics.get('fp', 0))} "
        f"fn={int(metrics.get('fn', 0))}"
    )


async def producer_loop(
    engine: RealtimeEngine,
    hub: Hub,
    *,
    tick_lock: asyncio.Lock,
    batch: list[Transaction],
) -> None:
    """Pace the simulator, emit transactions, and score micro-batches.

    ``tick_lock`` is held while pulling/scoring/broadcasting and released
    during the pacing sleep so a source switch can run between ticks.
    """
    cfg = engine.config
    interval = 1.0 / max(cfg.rate_per_second, 1e-6)
    batch_started = asyncio.get_event_loop().time()

    while True:
        async with tick_lock:
            t_start = asyncio.get_event_loop().time()
            tx = await asyncio.to_thread(engine.next_transaction)
            wire = engine.tx_wire_dict(tx, transfer_risk=None)
            await hub.broadcast(
                {"type": "transaction", **wire},
                network=tx.network.value,
            )
            batch.append(tx)
            now = asyncio.get_event_loop().time()
            flush = len(batch) >= cfg.batch_size or (
                batch and (now - batch_started) >= cfg.max_batch_wait
            )
            if flush:
                scored = await asyncio.to_thread(engine.score_batch, list(batch))
                scores_msg = {
                    "type": "scores",
                    "batch_id": scored.batch_id,
                    "infer_ms": round(scored.infer_ms, 2),
                    "transfer_scores": scored.transfer_scores,
                    "party_scores": scored.party_scores,
                }
                await hub.broadcast(scores_msg)
                for alert in scored.alerts:
                    payload = {"type": "alert", **alert.to_dict()}
                    await hub.note_alert(payload)
                    await hub.broadcast(
                        payload,
                        network=alert.network if alert.network in ALL_NETWORKS else None,
                    )
                print(
                    f"batch {scored.batch_id:04d}  events={len(batch)}  "
                    f"infer_ms={scored.infer_ms:.1f}  alerts={len(scored.alerts)}  "
                    f"{_format_metrics(scored.label_metrics)}",
                    flush=True,
                )
                batch.clear()
                batch_started = asyncio.get_event_loop().time()

            elapsed = asyncio.get_event_loop().time() - t_start
            sleep_for = interval - elapsed
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


async def switch_source(
    engine: RealtimeEngine,
    hub: Hub,
    name: str,
    *,
    tick_lock: asyncio.Lock,
    batch: list[Transaction],
) -> bool:
    """Swap the live source under ``tick_lock`` and push a cleared view."""
    async with tick_lock:
        if not engine.set_source(name):
            return False
        batch.clear()
        await hub.clear_session()
        await hub.broadcast({"type": "reset"})
        await hub.broadcast(engine.hello_dict())
        await hub.broadcast(engine.snapshot_dict())
        await hub.broadcast({"type": "cases", "dispositions": []})
        print(f"source switched to {engine.source.name}", flush=True)
        return True


async def handler(
    websocket: Any,
    engine: RealtimeEngine,
    hub: Hub,
    *,
    tick_lock: asyncio.Lock | None = None,
    batch: list[Transaction] | None = None,
) -> None:
    state = await hub.register(websocket)
    state.writer_task = asyncio.create_task(_writer_loop(state))
    try:
        await state.queue.put(engine.hello_dict())
        await state.queue.put(engine.snapshot_dict())
        for prior in await hub.replay():
            await state.queue.put(prior)
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("op") == "subscribe":
                nets = msg.get("networks") or list(ALL_NETWORKS)
                if isinstance(nets, list):
                    hub.set_networks(websocket, [str(n) for n in nets])
            elif msg.get("op") == "disposition":
                record = await hub.apply_disposition(
                    msg.get("alert_id"), msg.get("action")
                )
                if record is not None:
                    await hub.broadcast(record)
            elif msg.get("op") == "source":
                name = msg.get("name")
                if (
                    isinstance(name, str)
                    and tick_lock is not None
                    and batch is not None
                ):
                    await switch_source(
                        engine, hub, name, tick_lock=tick_lock, batch=batch
                    )
    except ConnectionClosed:
        return
    finally:
        await hub.unregister(websocket)


async def run_server(engine: RealtimeEngine, host: str, port: int) -> None:
    try:
        from websockets.asyncio.server import serve
    except ImportError:  # pragma: no cover - older websockets
        from websockets.server import serve  # type: ignore[no-redef]

    hub = Hub()
    tick_lock = asyncio.Lock()
    batch: list[Transaction] = []
    producer = asyncio.create_task(
        producer_loop(engine, hub, tick_lock=tick_lock, batch=batch)
    )

    async def _handler(websocket: Any) -> None:
        await handler(
            websocket, engine, hub, tick_lock=tick_lock, batch=batch
        )

    print(
        f"aml-serve listening on ws://{host}:{port}  "
        f"source={engine.source.name}  "
        f"sources={','.join(engine.available_sources)}  "
        f"rate={engine.config.rate_per_second}/s  "
        f"batch={engine.config.batch_size}  "
        f"checkpoint={engine.config.checkpoint}",
        flush=True,
    )
    try:
        async with serve(_handler, host, port):
            await asyncio.Future()  # run forever
    finally:
        producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aml-serve",
        description="Real-time AML WebSocket inference engine",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/hetero_sage.pt"),
        help="Path to HeteroSAGE checkpoint from aml-train",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--rate",
        type=float,
        default=3.0,
        help="Wall-clock transactions per second. 3 leaves room for the dashboard to animate each arrival",
    )
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument(
        "--max-batch-wait",
        type=float,
        default=1.0,
        help="Flush a partial batch after this many wall seconds",
    )
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    p.add_argument(
        "--reveal-labels",
        action="store_true",
        help="Include typology/pattern_id on the wire (lab debugging)",
    )
    p.add_argument("--pattern-fraction", type=float, default=0.02)
    p.add_argument("--banking", type=int, default=400)
    p.add_argument("--crypto", type=int, default=150)
    p.add_argument("--cross-border", type=int, default=60)
    p.add_argument(
        "--source",
        choices=("sim", "replay"),
        default="sim",
        help="Transaction source: live simulator or JSONL replay",
    )
    p.add_argument(
        "--replay",
        type=Path,
        default=None,
        help=(
            "JSONL path from aml-sim --jsonl or aml-ibm-convert. "
            "Required with --source replay; also registers Replay for the UI switcher"
        ),
    )
    return p


def build_engine_from_args(args: argparse.Namespace) -> RealtimeEngine:
    if args.source == "replay" and args.replay is None:
        raise ValueError("--source replay requires --replay PATH")

    config = EngineConfig(
        seed=args.seed,
        rate_per_second=args.rate,
        batch_size=args.batch_size,
        max_batch_wait=args.max_batch_wait,
        warmup=args.warmup,
        threshold=args.threshold,
        pattern_event_fraction=args.pattern_fraction,
        population=PopulationConfig(
            banking=args.banking,
            crypto=args.crypto,
            cross_border=args.cross_border,
        ),
        device=args.device,
        reveal_labels=args.reveal_labels,
        checkpoint=args.checkpoint,
    )
    model, schema, meta = open_checkpoint(config.checkpoint, device=config.device)

    sim_source = build_sim_source(
        SimulatorConfig(
            seed=config.seed,
            rate_per_second=config.rate_per_second,
            pattern_event_fraction=config.pattern_event_fraction,
            population=config.population,
            max_transfer_edges=config.max_transfer_edges,
        )
    )
    sources: dict[str, TransactionSource] = {"sim": sim_source}
    if args.replay is not None:
        sources["replay"] = build_replay_source(
            args.replay, max_transfer_edges=config.max_transfer_edges
        )

    if args.source not in sources:
        raise ValueError(f"unknown --source {args.source!r}")

    engine = RealtimeEngine(
        config,
        source=sources[args.source],
        sources=sources,
        model=model,
        schema=schema,
        checkpoint_meta=meta,
    )
    counts = engine.warmup_all(config.warmup)
    warmed = ", ".join(f"{name}={n}" for name, n in counts.items())
    print(
        f"warmup complete: {warmed}  sim_time={engine.sim_time:.1f}s  "
        f"source={engine.source.name}  sources={','.join(engine.available_sources)}",
        flush=True,
    )
    return engine


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = build_engine_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    asyncio.run(run_server(engine, args.host, args.port))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
