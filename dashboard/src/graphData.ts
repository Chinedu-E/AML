import type { ElementDefinition } from "cytoscape";

import { formatPlainAmount, riskBand } from "./format";
import type { AlertRow, EngineState } from "./reduce";
import { thresholdOf, visibleFeed } from "./reduce";
import type { Neighborhood } from "./protocol";

export const LIVE_EDGE_CAP = 28;

const PARTY_KINDS = new Set([
  "mixer",
  "shell",
  "smart_contract",
  "wallet",
  "business",
  "individual",
]);

export type NeighborhoodSummary = {
  nodes: number;
  edges: number;
  amount: number;
  channels: string[];
  spanSeconds: number;
};

export function summarizeNeighborhood(neighborhood: Neighborhood): NeighborhoodSummary {
  let amount = 0;
  const channels = new Set<string>();
  const times: number[] = [];
  for (const edge of neighborhood.edges) {
    if (typeof edge.amount === "number") amount += edge.amount;
    if (edge.channel) channels.add(edge.channel);
    if (typeof edge.timestamp === "number") times.push(edge.timestamp);
  }
  const spanSeconds =
    times.length >= 2 ? Math.max(...times) - Math.min(...times) : 0;
  return {
    nodes: neighborhood.nodes.length,
    edges: neighborhood.edges.length,
    amount,
    channels: [...channels],
    spanSeconds,
  };
}

function classesFor(
  band: string,
  extra: string[] = [],
): string {
  return [`risk-${band}`, ...extra].filter(Boolean).join(" ");
}

function liveElements(
  state: EngineState,
): ElementDefinition[] {
  const threshold = thresholdOf(state);
  const rows = visibleFeed(state).slice(0, LIVE_EDGE_CAP).reverse();
  const nodes = new Map<string, ElementDefinition>();
  const edges: ElementDefinition[] = [];

  for (const tx of rows) {
    for (const id of [tx.source_id, tx.dest_id]) {
      if (nodes.has(id)) continue;
      const score = state.partyRisk[id];
      const band = riskBand(score, threshold);
      const extra = state.selectedNodeId === id ? ["picked"] : [];
      nodes.set(id, {
        data: { id, label: id },
        classes: classesFor(band, extra),
      });
    }
    const band = riskBand(tx.transfer_risk, threshold);
    const extra = tx.tx_id === state.selectedTxId ? ["selected"] : [];
    edges.push({
      data: {
        id: `e:${tx.tx_id}`,
        source: tx.source_id,
        target: tx.dest_id,
        label: "",
        txId: tx.tx_id,
      },
      classes: classesFor(band, extra),
    });
  }
  return [...nodes.values(), ...edges];
}

function caseElements(alert: AlertRow, partyRisk: Record<string, number>, threshold: number): ElementDefinition[] {
  const nodes = new Map<string, ElementDefinition>();
  const ensure = (id: string, partyType?: string) => {
    if (nodes.has(id)) return;
    const score = alert.kind === "party" && id === alert.subject_id ? alert.score : partyRisk[id];
    const band = riskBand(score, threshold);
    const extra: string[] = [];
    if (id === alert.neighborhood.center || id === alert.subject_id) extra.push("center");
    if (partyType && PARTY_KINDS.has(partyType)) extra.push(`kind-${partyType}`);
    nodes.set(id, {
      data: { id, label: id },
      classes: classesFor(band, extra),
    });
  };

  for (const node of alert.neighborhood.nodes) ensure(node.id, node.party_type);
  if (alert.neighborhood.center) ensure(alert.neighborhood.center);
  if (alert.subject_id) ensure(alert.subject_id);
  if (alert.source_id) ensure(alert.source_id);
  if (alert.dest_id) ensure(alert.dest_id);

  const edges: ElementDefinition[] = [];
  const used = new Set<string>();
  alert.neighborhood.edges.forEach((edge, index) => {
    if (!nodes.has(edge.source) || !nodes.has(edge.dest)) return;
    let id = edge.tx_id ? `e:${edge.tx_id}` : `e:${index}:${edge.source}:${edge.dest}`;
    if (used.has(id)) id = `${id}:${index}`;
    used.add(id);
    const selected = alert.kind === "transfer" && edge.tx_id === alert.subject_id;
    edges.push({
      data: {
        id,
        source: edge.source,
        target: edge.dest,
        label: typeof edge.amount === "number" ? formatPlainAmount(edge.amount) : "",
        txId: edge.tx_id ?? "",
      },
      classes: classesFor("unknown", selected ? ["selected"] : []),
    });
  });
  return [...nodes.values(), ...edges];
}

function applyQuery(elements: ElementDefinition[], query: string): ElementDefinition[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return elements;
  const hitNodes = new Set(
    elements
      .filter((el) => !el.data.source && String(el.data.label ?? el.data.id).toLowerCase().includes(needle))
      .map((el) => String(el.data.id)),
  );
  return elements.map((el) => {
    const isEdge = Boolean(el.data.source);
    const hit = isEdge
      ? hitNodes.has(String(el.data.source)) || hitNodes.has(String(el.data.target))
      : hitNodes.has(String(el.data.id));
    const kept = String(el.classes ?? "")
      .split(/\s+/)
      .filter((name) => name && name !== "dim" && name !== "hit");
    kept.push(hit ? "hit" : "dim");
    return { ...el, classes: kept.join(" ") };
  });
}

export function buildGraph(state: EngineState, query: string): ElementDefinition[] {
  const threshold = thresholdOf(state);
  if (state.graphMode === "case" && state.selectedAlertId) {
    const alert = state.alerts.find((item) => item.alert_id === state.selectedAlertId);
    if (alert) return applyQuery(caseElements(alert, state.partyRisk, threshold), query);
  }
  return applyQuery(liveElements(state), query);
}
