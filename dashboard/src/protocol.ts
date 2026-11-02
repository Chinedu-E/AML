export const ALL_NETWORKS = ["banking", "crypto", "cross_border"] as const;

export type NetworkId = (typeof ALL_NETWORKS)[number];

export type TxRow = {
  tx_id: string;
  timestamp: number;
  source_id: string;
  dest_id: string;
  amount: number;
  currency: string;
  channel: string;
  network: NetworkId;
  transfer_risk: number | null;
  typology?: string;
  pattern_id?: string | null;
  source_lat?: number;
  source_lng?: number;
  source_city?: string;
  source_country?: string;
  dest_lat?: number;
  dest_lng?: number;
  dest_city?: string;
  dest_country?: string;
  distance_km?: number;
};

export type NeighborhoodNode = {
  id: string;
  network?: string;
  party_type?: string;
  tx_count?: number;
  lat?: number;
  lng?: number;
  city?: string;
  country?: string;
};

export type NeighborhoodEdge = {
  source: string;
  dest: string;
  tx_id?: string;
  amount?: number;
  timestamp?: number;
  channel?: string;
  network?: string;
};

export type Neighborhood = {
  nodes: NeighborhoodNode[];
  edges: NeighborhoodEdge[];
  center: string;
  k: number;
};

export type SourceId = "sim" | "replay";

export type HelloMessage = {
  type: "hello";
  seed: number;
  rate: number;
  checkpoint: string;
  networks: string[];
  sim_time: number;
  warmup_done: number;
  threshold: number;
  reveal_labels: boolean;
  source: SourceId;
  sources: SourceId[];
  replay?: string;
};

export type ResetMessage = {
  type: "reset";
};

export type TransactionMessage = TxRow & { type: "transaction" };

export type ScoresMessage = {
  type: "scores";
  batch_id: number;
  infer_ms: number;
  transfer_scores: Record<string, number>;
  party_scores: Record<string, number>;
};

export type AlertMessage = {
  type: "alert";
  alert_id: string;
  kind: "transfer" | "party";
  subject_id: string;
  score: number;
  network: string;
  source_id: string | null;
  dest_id: string | null;
  reason: string;
  neighborhood: Neighborhood;
};

export type GraphStats = {
  nodes: number;
  edges: number;
  transfer_edges: number;
  nodes_by_network: Record<string, number>;
  edges_by_typology: Record<string, number>;
};

export type SnapshotMessage = {
  type: "snapshot";
  sim_time: number;
  party_scores: Record<string, number>;
  recent_transactions: TxRow[];
  stats: GraphStats | null;
};

export type DispositionAction = "approve" | "escalate";

export type DispositionMessage = {
  type: "disposition";
  alert_id: string;
  action: DispositionAction;
};

export type CasesMessage = {
  type: "cases";
  dispositions: { alert_id: string; action: DispositionAction }[];
};

export type ServerMessage =
  | HelloMessage
  | ResetMessage
  | TransactionMessage
  | ScoresMessage
  | AlertMessage
  | SnapshotMessage
  | DispositionMessage
  | CasesMessage;

function isSourceId(value: unknown): value is SourceId {
  return value === "sim" || value === "replay";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isNetwork(value: unknown): value is NetworkId {
  return value === "banking" || value === "crypto" || value === "cross_border";
}

function numMap(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  const out: Record<string, number> = {};
  for (const [key, raw] of Object.entries(value)) {
    const n = finite(raw);
    if (n !== null) out[key] = n;
  }
  return out;
}

export function parseTx(raw: unknown): TxRow | null {
  if (!isRecord(raw)) return null;
  const timestamp = finite(raw.timestamp);
  const amount = finite(raw.amount);
  const risk = raw.transfer_risk === null ? null : finite(raw.transfer_risk);
  if (
    typeof raw.tx_id !== "string" ||
    typeof raw.source_id !== "string" ||
    typeof raw.dest_id !== "string" ||
    typeof raw.currency !== "string" ||
    typeof raw.channel !== "string" ||
    !isNetwork(raw.network) ||
    timestamp === null ||
    amount === null
  ) {
    return null;
  }
  const row: TxRow = {
    tx_id: raw.tx_id,
    timestamp,
    source_id: raw.source_id,
    dest_id: raw.dest_id,
    amount,
    currency: raw.currency,
    channel: raw.channel,
    network: raw.network,
    transfer_risk: risk,
  };
  if (typeof raw.typology === "string") row.typology = raw.typology;
  if (typeof raw.pattern_id === "string" || raw.pattern_id === null) {
    row.pattern_id = raw.pattern_id;
  }
  const sourceLat = finite(raw.source_lat);
  const sourceLng = finite(raw.source_lng);
  const destLat = finite(raw.dest_lat);
  const destLng = finite(raw.dest_lng);
  const distanceKm = finite(raw.distance_km);
  if (sourceLat !== null) row.source_lat = sourceLat;
  if (sourceLng !== null) row.source_lng = sourceLng;
  if (destLat !== null) row.dest_lat = destLat;
  if (destLng !== null) row.dest_lng = destLng;
  if (distanceKm !== null) row.distance_km = distanceKm;
  if (typeof raw.source_city === "string") row.source_city = raw.source_city;
  if (typeof raw.source_country === "string") row.source_country = raw.source_country;
  if (typeof raw.dest_city === "string") row.dest_city = raw.dest_city;
  if (typeof raw.dest_country === "string") row.dest_country = raw.dest_country;
  return row;
}

function parseNeighborhood(raw: unknown, fallbackCenter: string): Neighborhood {
  const empty: Neighborhood = { nodes: [], edges: [], center: fallbackCenter, k: 2 };
  if (!isRecord(raw)) return empty;
  const nodes: NeighborhoodNode[] = [];
  if (Array.isArray(raw.nodes)) {
    for (const item of raw.nodes) {
      if (!isRecord(item) || typeof item.id !== "string") continue;
      const node: NeighborhoodNode = { id: item.id };
      if (typeof item.network === "string") node.network = item.network;
      if (typeof item.party_type === "string") node.party_type = item.party_type;
      const txCount = finite(item.tx_count);
      if (txCount !== null) node.tx_count = txCount;
      const lat = finite(item.lat);
      const lng = finite(item.lng);
      if (lat !== null) node.lat = lat;
      if (lng !== null) node.lng = lng;
      if (typeof item.city === "string") node.city = item.city;
      if (typeof item.country === "string") node.country = item.country;
      nodes.push(node);
    }
  }
  const edges: NeighborhoodEdge[] = [];
  if (Array.isArray(raw.edges)) {
    for (const item of raw.edges) {
      if (!isRecord(item) || typeof item.source !== "string" || typeof item.dest !== "string") {
        continue;
      }
      const edge: NeighborhoodEdge = { source: item.source, dest: item.dest };
      if (typeof item.tx_id === "string") edge.tx_id = item.tx_id;
      const amount = finite(item.amount);
      if (amount !== null) edge.amount = amount;
      const timestamp = finite(item.timestamp);
      if (timestamp !== null) edge.timestamp = timestamp;
      if (typeof item.channel === "string") edge.channel = item.channel;
      if (typeof item.network === "string") edge.network = item.network;
      edges.push(edge);
    }
  }
  return {
    nodes,
    edges,
    center: typeof raw.center === "string" ? raw.center : fallbackCenter,
    k: finite(raw.k) ?? 2,
  };
}

function parseStats(raw: unknown): GraphStats | null {
  if (!isRecord(raw)) return null;
  const nodes = finite(raw.nodes);
  const edges = finite(raw.edges);
  const transfer = finite(raw.transfer_edges);
  if (nodes === null || edges === null || transfer === null) return null;
  return {
    nodes,
    edges,
    transfer_edges: transfer,
    nodes_by_network: numMap(raw.nodes_by_network),
    edges_by_typology: numMap(raw.edges_by_typology),
  };
}

function parseDispositionAction(value: unknown): DispositionAction | null {
  return value === "approve" || value === "escalate" ? value : null;
}

export function parseServerMessage(value: unknown): ServerMessage | null {
  if (!isRecord(value) || typeof value.type !== "string") return null;

  switch (value.type) {
    case "hello": {
      const threshold = finite(value.threshold);
      const simTime = finite(value.sim_time);
      const rate = finite(value.rate);
      const seed = finite(value.seed);
      const warmup = finite(value.warmup_done);
      if (
        threshold === null ||
        simTime === null ||
        rate === null ||
        seed === null ||
        warmup === null ||
        typeof value.checkpoint !== "string" ||
        typeof value.reveal_labels !== "boolean" ||
        !Array.isArray(value.networks) ||
        !isSourceId(value.source) ||
        !Array.isArray(value.sources)
      ) {
        return null;
      }
      const sources = value.sources.filter(isSourceId);
      if (sources.length === 0 || !sources.includes(value.source)) {
        return null;
      }
      const hello: HelloMessage = {
        type: "hello",
        seed,
        rate,
        checkpoint: value.checkpoint,
        networks: value.networks.filter((item): item is string => typeof item === "string"),
        sim_time: simTime,
        warmup_done: warmup,
        threshold,
        reveal_labels: value.reveal_labels,
        source: value.source,
        sources,
      };
      if (typeof value.replay === "string") hello.replay = value.replay;
      return hello;
    }
    case "reset":
      return { type: "reset" };
    case "transaction": {
      const tx = parseTx(value);
      return tx ? { type: "transaction", ...tx } : null;
    }
    case "scores": {
      const batchId = finite(value.batch_id);
      const infer = finite(value.infer_ms);
      if (batchId === null || infer === null) return null;
      return {
        type: "scores",
        batch_id: batchId,
        infer_ms: infer,
        transfer_scores: numMap(value.transfer_scores),
        party_scores: numMap(value.party_scores),
      };
    }
    case "alert": {
      const score = finite(value.score);
      if (
        typeof value.subject_id !== "string" ||
        score === null ||
        (value.kind !== "transfer" && value.kind !== "party")
      ) {
        return null;
      }
      const alertId =
        typeof value.alert_id === "string" && value.alert_id
          ? value.alert_id
          : `local-${value.kind}-${value.subject_id}`;
      return {
        type: "alert",
        alert_id: alertId,
        kind: value.kind,
        subject_id: value.subject_id,
        score,
        network: typeof value.network === "string" ? value.network : "",
        source_id: typeof value.source_id === "string" ? value.source_id : null,
        dest_id: typeof value.dest_id === "string" ? value.dest_id : null,
        reason: typeof value.reason === "string" ? value.reason : "",
        neighborhood: parseNeighborhood(value.neighborhood, value.subject_id),
      };
    }
    case "snapshot": {
      const simTime = finite(value.sim_time);
      if (simTime === null) return null;
      const recent: TxRow[] = [];
      if (Array.isArray(value.recent_transactions)) {
        for (const item of value.recent_transactions) {
          const tx = parseTx(item);
          if (tx) recent.push(tx);
        }
      }
      return {
        type: "snapshot",
        sim_time: simTime,
        party_scores: numMap(value.party_scores),
        recent_transactions: recent,
        stats: parseStats(value.stats),
      };
    }
    case "disposition": {
      const action = parseDispositionAction(value.action);
      if (typeof value.alert_id !== "string" || !value.alert_id || !action) return null;
      return { type: "disposition", alert_id: value.alert_id, action };
    }
    case "cases": {
      const dispositions: { alert_id: string; action: DispositionAction }[] = [];
      if (Array.isArray(value.dispositions)) {
        for (const item of value.dispositions) {
          if (!isRecord(item) || typeof item.alert_id !== "string") continue;
          const action = parseDispositionAction(item.action);
          if (!action) continue;
          dispositions.push({ alert_id: item.alert_id, action });
        }
      }
      return { type: "cases", dispositions };
    }
    default:
      return null;
  }
}
