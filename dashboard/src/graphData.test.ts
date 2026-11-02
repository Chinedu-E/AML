import { describe, expect, it } from "vitest";

import { buildGraph, LIVE_EDGE_CAP, summarizeNeighborhood } from "./graphData";
import { initialState, reduce } from "./reduce";
import type { ServerMessage, TxRow } from "./protocol";

function tx(id: number, source = "bank-0001", dest = "bank-0002"): TxRow {
  return {
    tx_id: `tx-${id}`,
    timestamp: id,
    source_id: source,
    dest_id: dest,
    amount: 1000 + id,
    currency: "USD",
    channel: "wire",
    network: "banking",
    transfer_risk: id % 2 === 0 ? 0.2 : 0.9,
  };
}

function withFeed(rows: TxRow[]) {
  return rows.reduce(
    (state, row) => reduce(state, { type: "server", message: { type: "transaction", ...row } }),
    initialState,
  );
}

describe("buildGraph", () => {
  it("keeps the newest edges and unique ids", () => {
    const rows = Array.from({ length: LIVE_EDGE_CAP + 10 }, (_, i) =>
      tx(i, `bank-${i % 7}`, `bank-${(i % 7) + 1}`),
    );
    const elements = buildGraph(withFeed(rows), "");
    const edges = elements.filter((el) => el.data.source);
    const ids = elements.map((el) => el.data.id);
    expect(edges).toHaveLength(LIVE_EDGE_CAP);
    expect(new Set(ids).size).toBe(ids.length);
    expect(edges[edges.length - 1].data.txId).toBe(`tx-${LIVE_EDGE_CAP + 9}`);
  });

  it("dims accounts that do not match the search", () => {
    const elements = buildGraph(withFeed([tx(1), tx(2, "wallet-0003", "bank-0001")]), "wallet");
    const wallet = elements.find((el) => el.data.id === "wallet-0003");
    const bank = elements.find((el) => el.data.id === "bank-0002");
    expect(String(wallet?.classes)).toContain("hit");
    expect(String(bank?.classes)).toContain("dim");
  });

  it("draws the alert neighborhood and marks the center", () => {
    const message: ServerMessage = {
      type: "alert",
      alert_id: "a9",
      kind: "party",
      subject_id: "bank-0004",
      score: 0.92,
      network: "banking",
      source_id: null,
      dest_id: null,
      reason: "layering",
      neighborhood: {
        center: "bank-0004",
        k: 2,
        nodes: [
          { id: "bank-0004", party_type: "individual" },
          { id: "mixer-0001", party_type: "mixer" },
        ],
        edges: [
          { source: "bank-0004", dest: "mixer-0001", amount: 9500, tx_id: "tx-1", channel: "wire" },
        ],
      },
    };
    let state = reduce(initialState, { type: "server", message });
    state = reduce(state, { type: "select-alert", alertId: "a9" });
    const elements = buildGraph(state, "");
    const center = elements.find((el) => el.data.id === "bank-0004");
    const mixer = elements.find((el) => el.data.id === "mixer-0001");
    expect(String(center?.classes)).toContain("center");
    expect(String(center?.classes)).toContain("risk-high");
    expect(String(mixer?.classes)).toContain("kind-mixer");
    expect(elements.some((el) => el.data.source === "bank-0004")).toBe(true);
  });
});

describe("summarizeNeighborhood", () => {
  it("totals amount, channels, and time span", () => {
    const summary = summarizeNeighborhood({
      center: "a",
      k: 2,
      nodes: [{ id: "a" }, { id: "b" }],
      edges: [
        { source: "a", dest: "b", amount: 100, timestamp: 10, channel: "wire" },
        { source: "b", dest: "a", amount: 40, timestamp: 70, channel: "wire" },
      ],
    });
    expect(summary).toMatchObject({
      nodes: 2,
      edges: 2,
      amount: 140,
      channels: ["wire"],
      spanSeconds: 60,
    });
  });
});
