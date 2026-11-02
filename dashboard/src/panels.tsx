import {
  dispositionLabel,
  formatAmount,
  formatPlainAmount,
  formatSpan,
  networkLabel,
  riskBand,
} from "./format";
import { summarizeNeighborhood } from "./graphData";
import type { TxRow } from "./protocol";
import type { AlertRow } from "./reduce";

export function Feed({
  rows,
  selectedTxId,
  threshold,
  showTypology,
  onSelect,
}: {
  rows: TxRow[];
  selectedTxId: string | null;
  threshold: number;
  showTypology: boolean;
  onSelect: (txId: string) => void;
}) {
  if (rows.length === 0) {
    return <p className="empty">Waiting for transactions in this corridor.</p>;
  }
  return (
    <ul className="rows">
      {rows.map((tx) => {
        const band = riskBand(tx.transfer_risk, threshold);
        return (
          <li key={tx.tx_id}>
            <button
              type="button"
              className={tx.tx_id === selectedTxId ? "row selected" : "row"}
              onClick={() => onSelect(tx.tx_id)}
            >
              <span className="row-main">
                <span className="mono">
                  {tx.source_id} → {tx.dest_id}
                </span>
                <span className={`score ${band}`}>
                  {tx.transfer_risk == null ? "scoring" : tx.transfer_risk.toFixed(2)}
                </span>
              </span>
              <span className="row-meta">
                {formatAmount(tx.amount, tx.currency)} · {tx.channel} · {networkLabel(tx.network)}
                {showTypology && tx.typology ? ` · ${tx.typology}` : ""}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export function AlertQueue({
  alerts,
  selectedAlertId,
  threshold,
  onSelect,
}: {
  alerts: AlertRow[];
  selectedAlertId: string | null;
  threshold: number;
  onSelect: (alertId: string) => void;
}) {
  if (alerts.length === 0) {
    return <p className="empty">No alerts in this corridor.</p>;
  }
  return (
    <ul className="rows compact">
      {alerts.map((alert) => {
        const band = riskBand(alert.score, threshold);
        const title =
          alert.kind === "transfer"
            ? `${alert.source_id ?? "?"} → ${alert.dest_id ?? "?"}`
            : alert.subject_id;
        return (
          <li key={alert.alert_id}>
            <button
              type="button"
              className={alert.alert_id === selectedAlertId ? "row selected" : "row"}
              onClick={() => onSelect(alert.alert_id)}
            >
              <span className="row-main">
                <span className="mono">{title}</span>
                <span className={`score ${band}`}>{alert.score.toFixed(2)}</span>
              </span>
              <span className="row-meta">
                <span className={`badge ${alert.disposition}`}>{dispositionLabel(alert.disposition)}</span>
                {alert.kind === "transfer" ? "Transfer" : "Account"} · {networkLabel(alert.network)}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export function Investigator({
  alert,
  tx,
  nodeId,
  nodeScore,
  threshold,
  onDisposition,
}: {
  alert: AlertRow | null;
  tx: TxRow | null;
  nodeId: string | null;
  nodeScore: number | null;
  threshold: number;
  onDisposition: (alertId: string, action: "approve" | "escalate") => void;
}) {
  if (alert) {
    const summary = summarizeNeighborhood(alert.neighborhood);
    const trail = [...alert.neighborhood.edges]
      .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
      .slice(0, 8);
    return (
      <div className="case">
        <div className="row-main">
          <span className="kicker">{alert.kind === "transfer" ? "Transfer case" : "Account case"}</span>
          <span className={`score ${riskBand(alert.score, threshold)}`}>{alert.score.toFixed(2)}</span>
        </div>
        <p className="reason">{alert.reason}</p>
        <dl className="facts">
          <div>
            <dt>Accounts</dt>
            <dd>{summary.nodes}</dd>
          </div>
          <div>
            <dt>Transfers</dt>
            <dd>{summary.edges}</dd>
          </div>
          <div>
            <dt>Amount</dt>
            <dd>{formatPlainAmount(summary.amount)}</dd>
          </div>
          <div>
            <dt>Span</dt>
            <dd>{formatSpan(summary.spanSeconds)}</dd>
          </div>
        </dl>
        {summary.channels.length > 0 && (
          <p className="row-meta">Channels: {summary.channels.join(", ")}</p>
        )}
        {trail.length > 0 && (
          <ol className="trail">
            {trail.map((edge, index) => (
              <li key={`${edge.tx_id ?? index}-${edge.source}`}>
                <span className="mono">
                  {edge.source} → {edge.dest}
                </span>
                <span>
                  {typeof edge.amount === "number" ? formatPlainAmount(edge.amount) : "—"}
                  {edge.channel ? ` · ${edge.channel}` : ""}
                </span>
              </li>
            ))}
          </ol>
        )}
        {summary.edges > trail.length && (
          <p className="row-meta">{summary.edges - trail.length} more transfers in the graph.</p>
        )}
        <div className="actions">
          <button
            type="button"
            className="ghost"
            aria-pressed={alert.disposition === "approve"}
            onClick={() => onDisposition(alert.alert_id, "approve")}
          >
            Approve
          </button>
          <button
            type="button"
            className="solid"
            aria-pressed={alert.disposition === "escalate"}
            onClick={() => onDisposition(alert.alert_id, "escalate")}
          >
            Escalate
          </button>
        </div>
      </div>
    );
  }

  if (tx) {
    return (
      <div className="case">
        <p className="kicker">Selected transfer</p>
        <p className="mono">
          {tx.source_id} → {tx.dest_id}
        </p>
        <p className="reason">
          {formatAmount(tx.amount, tx.currency)} on {tx.channel}. Risk{" "}
          {tx.transfer_risk == null ? "is still scoring" : tx.transfer_risk.toFixed(2)}.
        </p>
      </div>
    );
  }

  if (nodeId) {
    return (
      <div className="case">
        <p className="kicker">Account</p>
        <p className="mono">{nodeId}</p>
        <p className="reason">
          {nodeScore == null
            ? "No score yet, and no open case for this account."
            : `Risk ${nodeScore.toFixed(2)}. No open case for this account.`}
        </p>
      </div>
    );
  }

  return <p className="empty">Select an alert to read the reason and the money-flow trail.</p>;
}
