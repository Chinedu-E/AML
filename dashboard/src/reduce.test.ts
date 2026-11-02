import { describe, expect, it } from "vitest";

import { parseServerMessage } from "./protocol";
import { ALERT_CAP, FEED_CAP, initialState, reduce, visibleAlerts } from "./reduce";
import type { ServerMessage, TxRow } from "./protocol";

function tx(id: string, network: TxRow["network"] = "banking", risk: number | null = null): TxRow {
  return {
    tx_id: id,
    timestamp: Number(id.replace(/\D/g, "")) || 1,
    source_id: "bank-0001",
    dest_id: "bank-0002",
    amount: 9500,
    currency: "USD",
    channel: "wire",
    network,
    transfer_risk: risk,
  };
}

function send(state: typeof initialState, message: ServerMessage) {
  return reduce(state, { type: "server", message });
}

describe("parseServerMessage", () => {
  it("rejects junk and accepts a transaction", () => {
    expect(parseServerMessage(null)).toBeNull();
    expect(parseServerMessage({ type: "nope" })).toBeNull();
    const msg = parseServerMessage({ type: "transaction", ...tx("tx-0001") });
    expect(msg?.type).toBe("transaction");
  });

  it("keeps source fields on hello and accepts reset", () => {
    const hello = parseServerMessage({
      type: "hello",
      seed: 1,
      rate: 3,
      checkpoint: "artifacts/hetero_sage.pt",
      networks: ["banking"],
      sim_time: 12,
      warmup_done: 100,
      threshold: 0.5,
      reveal_labels: false,
      source: "sim",
      sources: ["sim", "replay"],
      replay: "artifacts/ibm_hi_small.jsonl",
    });
    expect(hello).toEqual({
      type: "hello",
      seed: 1,
      rate: 3,
      checkpoint: "artifacts/hetero_sage.pt",
      networks: ["banking"],
      sim_time: 12,
      warmup_done: 100,
      threshold: 0.5,
      reveal_labels: false,
      source: "sim",
      sources: ["sim", "replay"],
      replay: "artifacts/ibm_hi_small.jsonl",
    });
    expect(parseServerMessage({ type: "reset" })).toEqual({ type: "reset" });
    expect(
      parseServerMessage({
        type: "hello",
        seed: 1,
        rate: 3,
        checkpoint: "x",
        networks: [],
        sim_time: 0,
        warmup_done: 0,
        threshold: 0.5,
        reveal_labels: false,
      }),
    ).toBeNull();
  });
});

describe("reduce", () => {
  it("joins scores onto transactions by id", () => {
    let state = send(initialState, { type: "transaction", ...tx("tx-7") });
    state = send(state, {
      type: "scores",
      batch_id: 3,
      infer_ms: 12.5,
      transfer_scores: { "tx-7": 0.91 },
      party_scores: { "bank-0001": 0.4 },
    });
    expect(state.feed[0].transfer_risk).toBe(0.91);
    expect(state.partyRisk["bank-0001"]).toBe(0.4);
    expect(state.inferMs).toBe(12.5);
    expect(state.batchId).toBe(3);
  });

  it("caps the feed and keeps the newest rows", () => {
    let state = initialState;
    for (let i = 0; i < FEED_CAP + 5; i += 1) {
      state = send(state, { type: "transaction", ...tx(`tx-${i}`) });
    }
    expect(state.feed).toHaveLength(FEED_CAP);
    expect(state.feed[0].tx_id).toBe(`tx-${FEED_CAP + 4}`);
    expect(state.feed.some((row) => row.tx_id === "tx-0")).toBe(false);
  });

  it("replaces the feed from a snapshot", () => {
    let state = send(initialState, { type: "transaction", ...tx("tx-live") });
    state = send(state, {
      type: "snapshot",
      sim_time: 40,
      party_scores: { "bank-0008": 0.2 },
      recent_transactions: [tx("tx-old"), tx("tx-new")],
      stats: null,
    });
    expect(state.feed.map((row) => row.tx_id)).toEqual(["tx-new", "tx-old"]);
    expect(state.partyRisk).toEqual({ "bank-0008": 0.2 });
    expect(state.simTime).toBe(40);
  });

  it("keeps a disposition that arrives before the alert", () => {
    let state = send(initialState, {
      type: "disposition",
      alert_id: "a1",
      action: "approve",
    });
    state = send(state, {
      type: "alert",
      alert_id: "a1",
      kind: "party",
      subject_id: "bank-0001",
      score: 0.88,
      network: "banking",
      source_id: null,
      dest_id: null,
      reason: "Flagged",
      neighborhood: { nodes: [], edges: [], center: "bank-0001", k: 2 },
    });
    expect(state.alerts[0].disposition).toBe("approve");
  });

  it("drops a selected alert when the ring evicts it", () => {
    let state = send(initialState, {
      type: "alert",
      alert_id: "a0",
      kind: "party",
      subject_id: "bank-0001",
      score: 0.9,
      network: "banking",
      source_id: null,
      dest_id: null,
      reason: "first",
      neighborhood: { nodes: [], edges: [], center: "bank-0001", k: 2 },
    });
    state = reduce(state, { type: "select-alert", alertId: "a0" });
    expect(state.graphMode).toBe("case");
    for (let i = 1; i <= ALERT_CAP; i += 1) {
      state = send(state, {
        type: "alert",
        alert_id: `a${i}`,
        kind: "transfer",
        subject_id: `tx-${i}`,
        score: 0.8,
        network: "banking",
        source_id: "bank-0001",
        dest_id: "bank-0002",
        reason: "later",
        neighborhood: { nodes: [], edges: [], center: "bank-0001", k: 2 },
      });
    }
    expect(state.alerts).toHaveLength(ALERT_CAP);
    expect(state.alerts.some((alert) => alert.alert_id === "a0")).toBe(false);
    expect(state.selectedAlertId).toBeNull();
    expect(state.graphMode).toBe("live");
  });

  it("clears a case that falls outside the corridor filter", () => {
    let state = send(initialState, {
      type: "alert",
      alert_id: "c1",
      kind: "party",
      subject_id: "wallet-0001",
      score: 0.7,
      network: "crypto",
      source_id: null,
      dest_id: null,
      reason: "mixer",
      neighborhood: { nodes: [], edges: [], center: "wallet-0001", k: 2 },
    });
    state = reduce(state, { type: "select-alert", alertId: "c1" });
    state = reduce(state, { type: "filter", networks: ["banking"] });
    expect(state.selectedAlertId).toBeNull();
    expect(state.graphMode).toBe("live");
    expect(visibleAlerts(state)).toHaveLength(0);
  });

  it("opens the newest case when a flagged node is picked", () => {
    let state = send(initialState, {
      type: "alert",
      alert_id: "old",
      kind: "party",
      subject_id: "bank-0004",
      score: 0.6,
      network: "banking",
      source_id: null,
      dest_id: null,
      reason: "old",
      neighborhood: { nodes: [], edges: [], center: "bank-0004", k: 2 },
    });
    state = send(state, {
      type: "alert",
      alert_id: "new",
      kind: "transfer",
      subject_id: "tx-9",
      score: 0.95,
      network: "banking",
      source_id: "bank-0004",
      dest_id: "bank-0005",
      reason: "new",
      neighborhood: { nodes: [], edges: [], center: "bank-0004", k: 2 },
    });
    state = reduce(state, { type: "select-node", nodeId: "bank-0004" });
    expect(state.selectedAlertId).toBe("new");
    expect(state.graphMode).toBe("case");
  });

  it("clears the live view on reset while keeping the corridor filter", () => {
    let state = send(initialState, { type: "transaction", ...tx("tx-live") });
    state = send(state, {
      type: "alert",
      alert_id: "a1",
      kind: "party",
      subject_id: "bank-0001",
      score: 0.9,
      network: "banking",
      source_id: null,
      dest_id: null,
      reason: "risk",
      neighborhood: { nodes: [], edges: [], center: "bank-0001", k: 2 },
    });
    state = reduce(state, { type: "filter", networks: ["banking"] });
    state = reduce(state, { type: "select-alert", alertId: "a1" });
    state = reduce(state, { type: "status", status: "live" });
    state = send(state, { type: "reset" });
    expect(state.feed).toEqual([]);
    expect(state.alerts).toEqual([]);
    expect(state.dispositions).toEqual({});
    expect(state.partyRisk).toEqual({});
    expect(state.selectedAlertId).toBeNull();
    expect(state.graphMode).toBe("live");
    expect(state.networkFilter).toEqual(["banking"]);
    expect(state.status).toBe("live");
  });
});
