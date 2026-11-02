import { describe, expect, it } from "vitest";

import { buildGlobe, formatDistanceKm, GLOBE_EDGE_CAP } from "./globeData";
import type { ServerMessage, TxRow } from "./protocol";
import { initialState, reduce } from "./reduce";

function geoTx(
  id: number,
  source = "bank-0001",
  dest = "bank-0002",
  extras: Partial<TxRow> = {},
): TxRow {
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
    source_lat: 40.71,
    source_lng: -74.0,
    source_city: "New York",
    source_country: "US",
    dest_lat: 51.5,
    dest_lng: -0.12,
    dest_city: "London",
    dest_country: "GB",
    distance_km: 5570,
    ...extras,
  };
}

function withFeed(rows: TxRow[]) {
  return rows.reduce(
    (state, row) => reduce(state, { type: "server", message: { type: "transaction", ...row } }),
    initialState,
  );
}

describe("buildGlobe", () => {
  it("builds live arcs from the newest geo transactions", () => {
    const rows = Array.from({ length: GLOBE_EDGE_CAP + 5 }, (_, i) =>
      geoTx(i, `bank-${i % 5}`, `bank-${(i % 5) + 1}`),
    );
    const model = buildGlobe(withFeed(rows), "");
    expect(model.arcs).toHaveLength(GLOBE_EDGE_CAP);
    expect(model.points.length).toBeGreaterThan(0);
    expect(model.arcs[model.arcs.length - 1].txId).toBe(`tx-${GLOBE_EDGE_CAP + 4}`);
  });

  it("skips transfers without coordinates", () => {
    const rows = [
      geoTx(1),
      {
        tx_id: "tx-bare",
        timestamp: 2,
        source_id: "bank-0003",
        dest_id: "bank-0004",
        amount: 50,
        currency: "USD",
        channel: "ach",
        network: "banking" as const,
        transfer_risk: 0.1,
      },
    ];
    const model = buildGlobe(withFeed(rows), "");
    expect(model.arcs).toHaveLength(1);
    expect(model.arcs[0].txId).toBe("tx-1");
  });

  it("colors high-risk arcs and dims non-matching search hits", () => {
    const model = buildGlobe(
      withFeed([
        geoTx(1, "bank-0001", "bank-0002", { transfer_risk: 0.9 }),
        geoTx(2, "wallet-0003", "bank-0001", {
          transfer_risk: 0.2,
          source_lat: 1.35,
          source_lng: 103.8,
          source_city: "Singapore",
          source_country: "SG",
        }),
      ]),
      "wallet",
    );
    const high = model.arcs.find((arc) => arc.txId === "tx-1");
    const wallet = model.arcs.find((arc) => arc.txId === "tx-2");
    expect(high?.band).toBe("high");
    expect(high?.dim).toBe(true);
    expect(wallet?.dim).toBe(false);
    const walletPoint = model.points.find((p) => p.id === "wallet-0003");
    expect(walletPoint?.dim).toBe(false);
  });

  it("draws the alert neighborhood in case mode", () => {
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
          {
            id: "bank-0004",
            party_type: "individual",
            lat: 40.7,
            lng: -74.0,
            city: "New York",
            country: "US",
          },
          {
            id: "shell-0001",
            party_type: "shell",
            lat: 19.3,
            lng: -81.4,
            city: "George Town",
            country: "KY",
          },
        ],
        edges: [
          {
            source: "bank-0004",
            dest: "shell-0001",
            amount: 9500,
            tx_id: "tx-1",
            channel: "wire",
          },
        ],
      },
    };
    let state = reduce(initialState, { type: "server", message });
    state = reduce(state, { type: "select-alert", alertId: "a9" });
    const model = buildGlobe(state, "");
    expect(model.points).toHaveLength(2);
    expect(model.arcs).toHaveLength(1);
    const center = model.points.find((p) => p.id === "bank-0004");
    expect(center?.center).toBe(true);
    expect(center?.band).toBe("high");
  });
});

describe("formatDistanceKm", () => {
  it("formats short and long distances", () => {
    expect(formatDistanceKm(3.2)).toBe("3.2 km");
    expect(formatDistanceKm(5570)).toMatch(/5,?570 km/);
    expect(formatDistanceKm(undefined)).toBe("—");
  });
});
