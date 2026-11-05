# AML Command Center

Real-time financial crime intelligence: a synthetic multi-network transaction generator, HeteroSAGE training, a live WebSocket inference engine, and a React dashboard.

```
aml/
  server/      Python package (simulator, train, convert, serve)
  dashboard/   React UI (Vite)
```

## Setup

Requires Python 3.13+ ([uv](https://docs.astral.sh/uv/)) and Node.js 20+.

```bash
cd server
uv sync

cd ../dashboard
npm install
```

Train a checkpoint before serving (or use an existing `artifacts/hetero_sage.pt`):

```bash
cd server
uv run aml-train --out artifacts/hetero_sage.pt
```

Run the engine and dashboard in two terminals:

```bash
# terminal 1 — live simulator (default)
cd server
uv run aml-serve --checkpoint artifacts/hetero_sage.pt

# terminal 2
cd dashboard
npm run dev
```

Open http://127.0.0.1:5173. The dashboard connects to `ws://127.0.0.1:8765` (override with `VITE_AML_WS`).

To start on the simulator and let the UI switch to the shipped demo replay:

```bash
cd server
uv run aml-serve \
  --checkpoint artifacts/hetero_sage.pt \
  --source sim \
  --replay artifacts/ibm_hi_small_demo.jsonl
```

`--source` is the source that is live when the process starts. If `artifacts/ibm_hi_small_demo.jsonl` is present, `aml-serve` registers it for the UI Replay switcher automatically (override with `--replay PATH`, or disable with `--no-replay`).

The producer **pauses when no dashboard clients are connected** (no scoring / no new txs) and resumes on the next WebSocket connect.

---

## Commands

All Python commands are run from `server/` with `uv run`. Relative paths such as `artifacts/` are relative to that directory.

### `aml-sim`

Synthetic multi-network transaction generator. Optionally writes JSONL for later replay or training.

```bash
uv run aml-sim [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--events N` | `2000` | Number of transactions |
| `--seed N` | `42` | RNG seed |
| `--rate F` | `7.0` | Target benign events per simulated second |
| `--pattern-fraction F` | `0.02` | Fraction of draws that attempt typology injection |
| `--jsonl PATH` | _(none)_ | Write events as JSONL |
| `--banking N` | `400` | Banking population size |
| `--crypto N` | `150` | Crypto population size |
| `--cross-border N` | `60` | Cross-border population size |

```bash
uv run aml-sim --events 2000 --jsonl events.jsonl
```

### `aml-train`

Train HeteroSAGE and write a checkpoint plus a training report.

```bash
uv run aml-train [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--out PATH` | `artifacts/hetero_sage.pt` | Checkpoint path |
| `--report PATH` | `<out-dir>/training_report.md` | Training report markdown path |
| `--no-report` | off | Skip writing the training summary report |
| `--print-report` | off | Print the training summary to the terminal |
| `--epochs N` | `40` | Training epochs |
| `--batch-size N` | `4` | Batch size |
| `--lr F` | `1e-3` | Learning rate |
| `--weight-decay F` | `1e-4` | Weight decay |
| `--patience N` | `8` | Early-stopping patience |
| `--hidden N` | `64` | Hidden size |
| `--device {cpu,mps,cuda}` | `cpu` | Torch device |
| `--events N` | `4000` | Events per simulator run |
| `--snapshot-every N` | `2000` | Snapshot cadence for simulator runs |
| `--train-runs N` | `24` | Simulator train runs |
| `--val-runs N` | `4` | Simulator val runs |
| `--test-runs N` | `4` | Simulator test runs |
| `--pattern-fraction F` | `0.02` | Typology injection fraction |
| `--banking N` | `400` | Banking population size |
| `--crypto N` | `150` | Crypto population size |
| `--cross-border N` | `60` | Cross-border population size |
| `--jsonl PATH` | _(none)_ | IBM/replay JSONL. Default mix is sim + JSONL; use `--ibm-only` for JSONL alone |
| `--ibm-only` | off | Train only on `--jsonl` (requires `--jsonl`) |
| `--jsonl-snapshot-every N` | `--snapshot-every` | Snapshot cadence for JSONL |
| `--jsonl-max-events N` | _(none)_ | Cap JSONL events when building snapshots |
| `--jsonl-train-frac F` | `0.70` | Chronological train fraction for JSONL |
| `--jsonl-val-frac F` | `0.15` | Chronological val fraction (rest = test) |
| `--tiny` | off | Tiny smoke dataset (cannot combine with `--jsonl`) |
| `--no-baseline` | off | Skip the baseline model |

Examples:

```bash
# Simulator only
uv run aml-train --out artifacts/hetero_sage.pt

# Portfolio (simulator + JSONL)
uv run aml-train --jsonl artifacts/ibm_hi_small.jsonl --out artifacts/hetero_sage.pt

# JSONL only
uv run aml-train --ibm-only --jsonl artifacts/ibm_hi_small.jsonl

# Smoke test
uv run aml-train --tiny --out artifacts/hetero_sage_tiny.pt
```

#### `aml-train report`

Display a saved training summary (same job as `aml-report`).

```bash
uv run aml-train report [PATH] [--format {text,markdown,json}]
```

| Argument / flag | Default | Description |
|-----------------|---------|-------------|
| `PATH` | `artifacts/training_report.json` | Report JSON/markdown or `.pt` checkpoint |
| `--format` | `text` | `text`, `markdown`, or `json` |

### `aml-report`

Display a training summary from a report file or checkpoint.

```bash
uv run aml-report [PATH] [--format {text,markdown,json}]
```

| Argument / flag | Default | Description |
|-----------------|---------|-------------|
| `PATH` | `artifacts/training_report.json` | Report JSON/markdown or `.pt` checkpoint |
| `--format` | `text` | `text`, `markdown`, or `json` |

```bash
uv run aml-report
uv run aml-report artifacts/hetero_sage.pt --format markdown
```

### `aml-ibm-convert`

Convert an IBM AML transaction CSV (HI-Small and HI/LI Medium/Large siblings) into replay JSONL for `aml-serve` and training.

```bash
uv run aml-ibm-convert --csv PATH [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--csv PATH` | _(required)_ | Path to `HI-Small_Trans.csv` (or sibling) |
| `--out PATH` | `artifacts/ibm_hi_small.jsonl` | Output JSONL path |
| `--limit N` | _(none)_ | Max transactions to write |
| `--keep-self` | off | Keep same-account self transfers (dropped by default) |

```bash
uv run aml-ibm-convert --csv HI-Small_Trans.csv --out artifacts/ibm_hi_small.jsonl
uv run aml-ibm-convert --csv HI-Small_Trans.csv --limit 5000
```

### `aml-serve`

Load a HeteroSAGE checkpoint and serve a WebSocket inference engine for the dashboard.

```bash
uv run aml-serve [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | `artifacts/hetero_sage.pt` | Checkpoint from `aml-train` |
| `--host HOST` | `127.0.0.1` | Bind address |
| `--port N` | `8765` | WebSocket port |
| `--rate F` | `3.0` | Wall-clock transactions per second |
| `--batch-size N` | `10` | Score micro-batch size |
| `--max-batch-wait F` | `1.0` | Flush a partial batch after this many wall seconds |
| `--warmup N` | `2000` | Events drained before live scoring |
| `--threshold F` | `0.5` | Alert threshold |
| `--seed N` | `42` | Simulator RNG seed |
| `--device {cpu,mps,cuda}` | `cpu` | Torch device |
| `--reveal-labels` | off | Include typology/pattern_id on the wire (lab debugging) |
| `--pattern-fraction F` | `0.02` | Simulator typology injection fraction |
| `--banking N` | `400` | Simulator banking population |
| `--crypto N` | `150` | Simulator crypto population |
| `--cross-border N` | `60` | Simulator cross-border population |
| `--source {sim,replay}` | `sim` | Source that is live when the process starts |
| `--replay PATH` | auto demo if present | JSONL from `aml-sim --jsonl` or `aml-ibm-convert`. Defaults to `artifacts/ibm_hi_small_demo.jsonl` when that file exists. Required with `--source replay` if no demo is shipped |
| `--no-replay` | off | Do not register Replay (simulator only) |

Examples:

```bash
# Live simulator only
uv run aml-serve --checkpoint artifacts/hetero_sage.pt

# Start on simulator; UI can switch to the shipped demo replay
uv run aml-serve \
  --checkpoint artifacts/hetero_sage.pt \
  --source sim \
  --replay artifacts/ibm_hi_small_demo.jsonl

# Start on replay
uv run aml-serve \
  --checkpoint artifacts/hetero_sage.pt \
  --source replay \
  --replay artifacts/ibm_hi_small_demo.jsonl \
  --rate 2 \
  --warmup 200 \
  --reveal-labels
```

Default URL: `ws://127.0.0.1:8765`. With zero connected clients the producer stays idle.

### Tests (server)

```bash
cd server
uv run pytest
```

---

## Dashboard

From `dashboard/`:

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server on http://127.0.0.1:5173 |
| `npm run build` | Typecheck, then production build |
| `npm run typecheck` | Typecheck only |
| `npm run test` | Vitest (one-shot) |

Optional env: `VITE_AML_WS` overrides the WebSocket URL (default `ws://127.0.0.1:8765`).

In the UI, the **Source** switcher selects Simulator or Replay. Replay is available only when `aml-serve` was started with `--replay`. Switching clears the live feed, alerts, and graph, then continues from the chosen source’s warmed graph.

---

## Shipped artifacts

These files are in git and copied into the server Docker image:

| Path | Purpose |
|------|---------|
| `server/artifacts/hetero_sage.pt` | HeteroSAGE checkpoint (~600KB) |
| `server/artifacts/ibm_hi_small_demo.jsonl` | First 5k IBM-style rows for Replay (~1MB) |

The full `ibm_hi_small.jsonl` (~1GB) stays local/gitignored. Refresh the demo slice after regenerating IBM JSONL:

```bash
cd server
head -n 5000 artifacts/ibm_hi_small.jsonl > artifacts/ibm_hi_small_demo.jsonl
# or: uv run aml-ibm-convert --csv HI-Small_Trans.csv --out artifacts/ibm_hi_small_demo.jsonl --limit 5000
```

---

## Deploy (Railway)

Two services from the same repo. The server Dockerfile listens on `0.0.0.0` and uses Railway’s `PORT`. `GET /health` returns `200` for probes (`/` stays free for the WebSocket upgrade). Inference only runs while at least one WebSocket client is connected.

### 1. Server service

- Root directory: `server`
- Builder: Dockerfile ([`server/Dockerfile`](server/Dockerfile))
- No secrets required for artifacts (checkpoint + demo JSONL are in the image)
- Optional vars: `HOST` (default `0.0.0.0` in the image), `PORT` (set by Railway)

After deploy, note the public HTTPS URL (e.g. `https://aml-server-production.up.railway.app`). The dashboard must use the WebSocket form: `wss://aml-server-production.up.railway.app`.

### 2. Dashboard service

- Root directory: `dashboard`
- Builder: Dockerfile ([`dashboard/Dockerfile`](dashboard/Dockerfile))
- Build arg / variable: `VITE_AML_WS=wss://<your-server-host>` (no path; same host the server exposes)

Example Railway build arg:

```
VITE_AML_WS=wss://aml-server-production.up.railway.app
```

Redeploy the dashboard whenever the server public hostname changes (Vite inlines `VITE_AML_WS` at build time).

### Local Docker smoke check

```bash
# server
cd server
docker build -t aml-server .
docker run --rm -p 8080:8080 -e PORT=8080 aml-server

# dashboard (point at local server)
cd ../dashboard
docker build --build-arg VITE_AML_WS=ws://127.0.0.1:8080 -t aml-dashboard .
docker run --rm -p 3000:3000 -e PORT=3000 aml-dashboard
```
