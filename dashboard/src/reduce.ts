import type {
  AlertMessage,
  DispositionAction,
  GraphStats,
  NetworkId,
  ServerMessage,
  TxRow,
} from "./protocol";

export const FEED_CAP = 200;
export const ALERT_CAP = 100;

export type ConnectionStatus = "connecting" | "live" | "reconnecting";
export type GraphMode = "live" | "case";
export type Disposition = "open" | DispositionAction;

export type AlertRow = Omit<AlertMessage, "type"> & { disposition: Disposition };

export type EngineState = {
  status: ConnectionStatus;
  hello: Extract<ServerMessage, { type: "hello" }> | null;
  simTime: number;
  inferMs: number | null;
  batchId: number | null;
  partyRisk: Record<string, number>;
  feed: TxRow[];
  alerts: AlertRow[];
  dispositions: Record<string, DispositionAction>;
  stats: GraphStats | null;
  buffered: number;
  networkFilter: NetworkId[] | null;
  selectedAlertId: string | null;
  selectedTxId: string | null;
  selectedNodeId: string | null;
  graphMode: GraphMode;
};

export const initialState: EngineState = {
  status: "connecting",
  hello: null,
  simTime: 0,
  inferMs: null,
  batchId: null,
  partyRisk: {},
  feed: [],
  alerts: [],
  dispositions: {},
  stats: null,
  buffered: 0,
  networkFilter: null,
  selectedAlertId: null,
  selectedTxId: null,
  selectedNodeId: null,
  graphMode: "live",
};

export type Action =
  | { type: "status"; status: ConnectionStatus }
  | { type: "buffered"; count: number }
  | { type: "flush"; messages: ServerMessage[] }
  | { type: "filter"; networks: NetworkId[] | null }
  | { type: "select-alert"; alertId: string | null }
  | { type: "select-tx"; txId: string | null }
  | { type: "select-node"; nodeId: string }
  | { type: "server"; message: ServerMessage };

export function matchesNetwork(network: string, filter: NetworkId[] | null): boolean {
  if (filter === null) return true;
  return filter.some((item) => item === network);
}

export function thresholdOf(state: EngineState): number {
  return state.hello?.threshold ?? 0.5;
}

export function visibleFeed(state: EngineState): TxRow[] {
  return state.feed.filter((tx) => matchesNetwork(tx.network, state.networkFilter));
}

export function visibleAlerts(state: EngineState): AlertRow[] {
  return state.alerts.filter((alert) => matchesNetwork(alert.network, state.networkFilter));
}

export function openAlertCount(alerts: AlertRow[]): number {
  return alerts.filter((alert) => alert.disposition === "open").length;
}

function mergeTx(prev: TxRow, row: TxRow): TxRow {
  return {
    ...prev,
    ...row,
    transfer_risk: row.transfer_risk ?? prev.transfer_risk,
    typology: row.typology ?? prev.typology,
    pattern_id: row.pattern_id ?? prev.pattern_id,
  };
}

function noteTx(feed: TxRow[], row: TxRow): TxRow[] {
  const index = feed.findIndex((tx) => tx.tx_id === row.tx_id);
  if (index === -1) return [row, ...feed].slice(0, FEED_CAP);
  const next = feed.slice();
  next[index] = mergeTx(next[index], row);
  return next;
}

function dropSelection(state: EngineState, alerts: AlertRow[]): Pick<
  EngineState,
  "selectedAlertId" | "graphMode"
> {
  if (
    state.selectedAlertId &&
    !alerts.some((alert) => alert.alert_id === state.selectedAlertId)
  ) {
    return {
      selectedAlertId: null,
      graphMode: state.graphMode === "case" ? "live" : state.graphMode,
    };
  }
  return { selectedAlertId: state.selectedAlertId, graphMode: state.graphMode };
}

function rememberAlert(state: EngineState, message: AlertMessage): EngineState {
  const disposition = state.dispositions[message.alert_id] ?? "open";
  const row: AlertRow = {
    alert_id: message.alert_id,
    kind: message.kind,
    subject_id: message.subject_id,
    score: message.score,
    network: message.network,
    source_id: message.source_id,
    dest_id: message.dest_id,
    reason: message.reason,
    neighborhood: message.neighborhood,
    disposition,
  };
  const index = state.alerts.findIndex((alert) => alert.alert_id === row.alert_id);
  const alerts =
    index >= 0
      ? state.alerts.map((alert, i) => (i === index ? { ...alert, ...row } : alert))
      : [row, ...state.alerts].slice(0, ALERT_CAP);
  return { ...state, alerts, ...dropSelection(state, alerts) };
}

function applyDispositions(
  state: EngineState,
  updates: Record<string, DispositionAction>,
): EngineState {
  const dispositions = { ...state.dispositions, ...updates };
  const alerts = state.alerts.map((alert) => {
    const action = dispositions[alert.alert_id];
    return action ? { ...alert, disposition: action } : alert;
  });
  return { ...state, dispositions, alerts };
}

function clearLiveView(state: EngineState): EngineState {
  return {
    ...state,
    simTime: 0,
    inferMs: null,
    batchId: null,
    partyRisk: {},
    feed: [],
    alerts: [],
    dispositions: {},
    stats: null,
    buffered: 0,
    selectedAlertId: null,
    selectedTxId: null,
    selectedNodeId: null,
    graphMode: "live",
  };
}

function applyMessage(state: EngineState, message: ServerMessage): EngineState {
  switch (message.type) {
    case "reset":
      return clearLiveView(state);
    case "hello":
      return { ...state, hello: message, simTime: message.sim_time };
    case "transaction": {
      const row: TxRow = {
        tx_id: message.tx_id,
        timestamp: message.timestamp,
        source_id: message.source_id,
        dest_id: message.dest_id,
        amount: message.amount,
        currency: message.currency,
        channel: message.channel,
        network: message.network,
        transfer_risk: message.transfer_risk,
      };
      if (message.typology !== undefined) row.typology = message.typology;
      if (message.pattern_id !== undefined) row.pattern_id = message.pattern_id;
      if (message.source_lat !== undefined) row.source_lat = message.source_lat;
      if (message.source_lng !== undefined) row.source_lng = message.source_lng;
      if (message.source_city !== undefined) row.source_city = message.source_city;
      if (message.source_country !== undefined) row.source_country = message.source_country;
      if (message.dest_lat !== undefined) row.dest_lat = message.dest_lat;
      if (message.dest_lng !== undefined) row.dest_lng = message.dest_lng;
      if (message.dest_city !== undefined) row.dest_city = message.dest_city;
      if (message.dest_country !== undefined) row.dest_country = message.dest_country;
      if (message.distance_km !== undefined) row.distance_km = message.distance_km;
      return {
        ...state,
        simTime: Math.max(state.simTime, message.timestamp),
        feed: noteTx(state.feed, row),
      };
    }
    case "scores": {
      const feed = state.feed.map((tx) => {
        const risk = message.transfer_scores[tx.tx_id];
        return risk === undefined ? tx : { ...tx, transfer_risk: risk };
      });
      return {
        ...state,
        feed,
        partyRisk: { ...state.partyRisk, ...message.party_scores },
        inferMs: message.infer_ms,
        batchId: message.batch_id,
      };
    }
    case "alert":
      return rememberAlert(state, message);
    case "snapshot": {
      const chronological = message.recent_transactions;
      const feed = [...chronological].reverse().slice(0, FEED_CAP);
      return {
        ...state,
        simTime: message.sim_time,
        partyRisk: { ...message.party_scores },
        feed,
        stats: message.stats,
      };
    }
    case "disposition":
      return applyDispositions(state, { [message.alert_id]: message.action });
    case "cases": {
      const updates: Record<string, DispositionAction> = {};
      for (const item of message.dispositions) updates[item.alert_id] = item.action;
      return applyDispositions(state, updates);
    }
    default: {
      const _exhaustive: never = message;
      return _exhaustive;
    }
  }
}

function applyFilter(state: EngineState, networks: NetworkId[] | null): EngineState {
  const next: EngineState = { ...state, networkFilter: networks };
  if (!state.selectedAlertId) return next;
  const alert = state.alerts.find((item) => item.alert_id === state.selectedAlertId);
  if (!alert || !matchesNetwork(alert.network, networks)) {
    next.selectedAlertId = null;
    next.graphMode = "live";
  }
  return next;
}

export function reduce(state: EngineState, action: Action): EngineState {
  switch (action.type) {
    case "status":
      return { ...state, status: action.status };
    case "buffered":
      return { ...state, buffered: action.count };
    case "flush": {
      const next = action.messages.reduce(applyMessage, state);
      return { ...next, buffered: 0 };
    }
    case "filter":
      return applyFilter(state, action.networks);
    case "select-alert":
      return {
        ...state,
        selectedAlertId: action.alertId,
        graphMode: action.alertId ? "case" : "live",
      };
    case "select-tx":
      return { ...state, selectedTxId: action.txId, graphMode: "live" };
    case "select-node": {
      const match = state.alerts.find(
        (alert) =>
          alert.subject_id === action.nodeId ||
          alert.source_id === action.nodeId ||
          alert.dest_id === action.nodeId,
      );
      if (!match) {
        return { ...state, selectedNodeId: action.nodeId, graphMode: "live" };
      }
      return {
        ...state,
        selectedNodeId: action.nodeId,
        selectedAlertId: match.alert_id,
        graphMode: "case",
      };
    }
    case "server":
      return applyMessage(state, action.message);
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}
