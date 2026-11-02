import { riskBand, type RiskBand } from "./format";
import type { Neighborhood } from "./protocol";
import type { AlertRow, EngineState } from "./reduce";
import { thresholdOf, visibleFeed } from "./reduce";

/** Live globe keeps a tighter window than the trail so arcs do not thrash. */
export const GLOBE_EDGE_CAP = 14;

export const RISK_COLORS: Record<RiskBand, string> = {
  low: "#8fbf73",
  mid: "#e0a34a",
  high: "#e15d4a",
  unknown: "#6e6a60",
};

/** Dim solid corridor under the traveling pulse. */
export const RISK_COLORS_MUTED: Record<RiskBand, string> = {
  low: "rgba(143, 191, 115, 0.35)",
  mid: "rgba(224, 163, 74, 0.4)",
  high: "rgba(225, 93, 74, 0.45)",
  unknown: "rgba(110, 106, 96, 0.3)",
};

/** Bright pulse head traveling source → destination. */
export const RISK_COLORS_PULSE: Record<RiskBand, string> = {
  low: "#c8f0a8",
  mid: "#ffd28a",
  high: "#ff8a7a",
  unknown: "#d0cbb8",
};

export type GlobePoint = {
  id: string;
  lat: number;
  lng: number;
  city?: string;
  country?: string;
  band: RiskBand;
  score: number | null;
  dim: boolean;
  selected: boolean;
  center: boolean;
};

export type GlobeArc = {
  txId: string;
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  sourceId: string;
  destId: string;
  sourceCity?: string;
  destCity?: string;
  distanceKm?: number;
  band: RiskBand;
  score: number | null;
  amount?: number;
  dim: boolean;
  selected: boolean;
};

export type GlobeModel = {
  points: GlobePoint[];
  arcs: GlobeArc[];
};

type LatLng = { lat: number; lng: number };

function readCoords(lat: unknown, lng: unknown): LatLng | null {
  if (typeof lat !== "number" || !Number.isFinite(lat)) return null;
  if (typeof lng !== "number" || !Number.isFinite(lng)) return null;
  return { lat, lng };
}

function matchesQuery(id: string, city: string | undefined, query: string): boolean {
  if (!query) return true;
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  if (id.toLowerCase().includes(needle)) return true;
  if (city && city.toLowerCase().includes(needle)) return true;
  return false;
}

function upsertPoint(
  points: Map<string, GlobePoint>,
  id: string,
  lat: number,
  lng: number,
  extras: Partial<GlobePoint>,
) {
  const existing = points.get(id);
  if (existing) {
    points.set(id, {
      ...existing,
      ...extras,
      lat,
      lng,
      selected: existing.selected || Boolean(extras.selected),
      center: existing.center || Boolean(extras.center),
      dim: existing.dim && (extras.dim ?? existing.dim),
    });
    return;
  }
  points.set(id, {
    id,
    lat,
    lng,
    band: extras.band ?? "unknown",
    score: extras.score ?? null,
    dim: extras.dim ?? false,
    selected: extras.selected ?? false,
    center: extras.center ?? false,
    city: extras.city,
    country: extras.country,
  });
}

function liveGlobe(state: EngineState, query: string): GlobeModel {
  const threshold = thresholdOf(state);
  // Newest first in the feed → take a short window, then chronological so
  // new arcs append instead of reshuffling the whole set every tick.
  const rows = visibleFeed(state).slice(0, GLOBE_EDGE_CAP).reverse();
  const points = new Map<string, GlobePoint>();
  const arcs: GlobeArc[] = [];
  const needle = query.trim().toLowerCase();

  for (const tx of rows) {
    const src = readCoords(tx.source_lat, tx.source_lng);
    const dst = readCoords(tx.dest_lat, tx.dest_lng);
    if (!src || !dst) continue;
    const band = riskBand(tx.transfer_risk, threshold);
    const hit =
      !needle ||
      matchesQuery(tx.source_id, tx.source_city, needle) ||
      matchesQuery(tx.dest_id, tx.dest_city, needle) ||
      tx.tx_id.toLowerCase().includes(needle);
    upsertPoint(points, tx.source_id, src.lat, src.lng, {
      city: tx.source_city,
      country: tx.source_country,
      band: riskBand(state.partyRisk[tx.source_id], threshold),
      score: state.partyRisk[tx.source_id] ?? null,
      dim: Boolean(needle) && !matchesQuery(tx.source_id, tx.source_city, needle),
      selected: state.selectedNodeId === tx.source_id,
    });
    upsertPoint(points, tx.dest_id, dst.lat, dst.lng, {
      city: tx.dest_city,
      country: tx.dest_country,
      band: riskBand(state.partyRisk[tx.dest_id], threshold),
      score: state.partyRisk[tx.dest_id] ?? null,
      dim: Boolean(needle) && !matchesQuery(tx.dest_id, tx.dest_city, needle),
      selected: state.selectedNodeId === tx.dest_id,
    });
    arcs.push({
      txId: tx.tx_id,
      startLat: src.lat,
      startLng: src.lng,
      endLat: dst.lat,
      endLng: dst.lng,
      sourceId: tx.source_id,
      destId: tx.dest_id,
      sourceCity: tx.source_city,
      destCity: tx.dest_city,
      distanceKm: tx.distance_km,
      band,
      score: tx.transfer_risk,
      amount: tx.amount,
      dim: Boolean(needle) && !hit,
      selected: tx.tx_id === state.selectedTxId,
    });
  }

  return { points: [...points.values()], arcs };
}

function caseGlobe(alert: AlertRow, partyRisk: Record<string, number>, threshold: number, query: string, selectedTxId: string | null): GlobeModel {
  const neighborhood: Neighborhood = alert.neighborhood;
  const points = new Map<string, GlobePoint>();
  const arcs: GlobeArc[] = [];
  const needle = query.trim().toLowerCase();
  const byId = new Map(neighborhood.nodes.map((node) => [node.id, node]));

  for (const node of neighborhood.nodes) {
    const coords = readCoords(node.lat, node.lng);
    if (!coords) continue;
    const score = alert.kind === "party" && node.id === alert.subject_id ? alert.score : partyRisk[node.id] ?? null;
    upsertPoint(points, node.id, coords.lat, coords.lng, {
      city: node.city,
      country: node.country,
      band: riskBand(score, threshold),
      score,
      dim: Boolean(needle) && !matchesQuery(node.id, node.city, needle),
      selected: false,
      center: node.id === neighborhood.center || node.id === alert.subject_id,
    });
  }

  neighborhood.edges.forEach((edge, index) => {
    const srcNode = byId.get(edge.source);
    const dstNode = byId.get(edge.dest);
    if (!srcNode || !dstNode) return;
    const src = readCoords(srcNode.lat, srcNode.lng);
    const dst = readCoords(dstNode.lat, dstNode.lng);
    if (!src || !dst) return;
    const txId = edge.tx_id ?? `case-${index}:${edge.source}:${edge.dest}`;
    const selected = alert.kind === "transfer" && edge.tx_id === alert.subject_id;
    const hit =
      !needle ||
      matchesQuery(edge.source, srcNode.city, needle) ||
      matchesQuery(edge.dest, dstNode.city, needle) ||
      txId.toLowerCase().includes(needle);
    arcs.push({
      txId,
      startLat: src.lat,
      startLng: src.lng,
      endLat: dst.lat,
      endLng: dst.lng,
      sourceId: edge.source,
      destId: edge.dest,
      sourceCity: srcNode.city,
      destCity: dstNode.city,
      band: selected ? riskBand(alert.score, threshold) : "unknown",
      score: selected ? alert.score : null,
      amount: edge.amount,
      dim: Boolean(needle) && !hit,
      selected: selected || txId === selectedTxId,
    });
  });

  return { points: [...points.values()], arcs };
}

/** Build points and arcs for the globe from live feed or selected alert. */
export function buildGlobe(state: EngineState, query: string): GlobeModel {
  const threshold = thresholdOf(state);
  if (state.graphMode === "case" && state.selectedAlertId) {
    const alert = state.alerts.find((item) => item.alert_id === state.selectedAlertId);
    if (alert) {
      return caseGlobe(alert, state.partyRisk, threshold, query, state.selectedTxId);
    }
  }
  return liveGlobe(state, query);
}

/** Color helpers shared with GlobeView. */
export function bandColor(band: RiskBand): string {
  return RISK_COLORS[band];
}

export function bandMutedColor(band: RiskBand): string {
  return RISK_COLORS_MUTED[band];
}

export function bandPulseColor(band: RiskBand): string {
  return RISK_COLORS_PULSE[band];
}

export function arcAltitude(band: RiskBand): number {
  // Keep arcs low so altitude churn does not read as camera shake.
  if (band === "high") return 0.28;
  if (band === "mid") return 0.18;
  return 0.1;
}

export function formatDistanceKm(km: number | undefined): string {
  if (km == null || !Number.isFinite(km)) return "—";
  if (km < 10) return `${km.toFixed(1)} km`;
  return `${Math.round(km).toLocaleString("en-US")} km`;
}
