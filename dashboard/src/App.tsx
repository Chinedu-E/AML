import { useEffect, useMemo, useRef, useState } from "react";

import { formatSimTime, networkLabel } from "./format";
import { GraphCanvas, type GraphControls } from "./GraphCanvas";
import { buildGraph } from "./graphData";
import { buildGlobe } from "./globeData";
import { GlobeView } from "./GlobeView";
import { AlertQueue, Feed, Investigator } from "./panels";
import { ALL_NETWORKS, type NetworkId } from "./protocol";
import { openAlertCount, thresholdOf, visibleAlerts, visibleFeed } from "./reduce";
import { useEngine } from "./useEngine";

type GraphViewMode = "trail" | "globe";

const WS_URL = import.meta.env.VITE_AML_WS ?? "ws://127.0.0.1:8765";

const CORRIDORS: { id: string; label: string; value: NetworkId[] | null }[] = [
  { id: "all", label: "All", value: null },
  ...ALL_NETWORKS.map((network) => ({
    id: network,
    label: networkLabel(network),
    value: [network] as NetworkId[],
  })),
];

type PanelId = "feed" | "graph" | "alerts" | "investigator";

function sameFilter(a: NetworkId[] | null, b: NetworkId[] | null): boolean {
  if (a === null || b === null) return a === b;
  return a.length === b.length && a.every((network, index) => network === b[index]);
}

function useNarrowScreen() {
  const [narrow, setNarrow] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia("(max-width: 980px)").matches : false,
  );
  useEffect(() => {
    const query = window.matchMedia("(max-width: 980px)");
    const onChange = () => setNarrow(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

function ExpandIcon({ collapse }: { collapse: boolean }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      {collapse ? (
        <path
          d="M6 3v3H3M10 6h3V3M10 13v-3h3M6 10H3v3"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      ) : (
        <path
          d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      )}
    </svg>
  );
}

function ExpandButton({
  panel,
  expanded,
  label,
  className,
  onClick,
}: {
  panel: PanelId;
  expanded: boolean;
  label: string;
  className?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={className ? `icon-button ${className}` : "icon-button"}
      aria-expanded={expanded}
      aria-controls={`panel-${panel}`}
      aria-label={expanded ? `Collapse ${label}` : `Expand ${label}`}
      onClick={onClick}
    >
      <ExpandIcon collapse={expanded} />
    </button>
  );
}

function replayFileName(path: string | undefined): string | null {
  if (!path) return null;
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

export function App() {
  const engine = useEngine(WS_URL);
  const {
    state,
    held,
    setHeld,
    setNetworks,
    setSource,
    sendDisposition,
    selectAlert,
    selectTx,
    selectNode,
  } = engine;
  const [query, setQuery] = useState("");
  const [reflow, setReflow] = useState(0);
  const [expanded, setExpanded] = useState<PanelId | null>(null);
  const [graphView, setGraphView] = useState<GraphViewMode>("trail");
  const narrow = useNarrowScreen();
  const graphControls = useRef<GraphControls>(null);
  const graphPanelRef = useRef<HTMLElement>(null);
  const availableSources = state.hello?.sources ?? ["sim"];
  const activeSource = state.hello?.source ?? "sim";
  const replayLabel = replayFileName(state.hello?.replay);
  const replayAvailable = availableSources.includes("replay");

  const feed = visibleFeed(state);
  const alerts = visibleAlerts(state);
  const threshold = thresholdOf(state);
  const elements = useMemo(() => buildGraph(state, query), [state, query]);
  const globeModel = useMemo(() => buildGlobe(state, query), [state, query]);
  const graphOpen = expanded === "graph";
  const layoutKey = `${state.graphMode}|${(state.networkFilter ?? ["all"]).join(",")}|${reflow}|${graphOpen ? "full" : "dock"}`;
  const selectedAlert = alerts.find((alert) => alert.alert_id === state.selectedAlertId) ?? null;
  const showCase = state.graphMode === "case" && selectedAlert !== null;
  const showGlobe = graphView === "globe";
  const emptyGraphLabel = showCase
    ? "This alert has no neighborhood to draw."
    : "Transactions will draw this corridor as they arrive.";
  const emptyGlobeLabel = showCase
    ? "This alert has no locations to plot."
    : "Transactions with locations will arc across the globe.";
  const selectedTx = feed.find((tx) => tx.tx_id === state.selectedTxId) ?? null;
  const statusLabel =
    state.status === "live" ? "Live" : state.status === "connecting" ? "Connecting" : "Reconnecting";
  const grow =
    !narrow && expanded && expanded !== "graph" ? `grow-${expanded}` : "";

  useEffect(() => {
    if (narrow && expanded && expanded !== "graph") setExpanded(null);
  }, [narrow, expanded]);

  useEffect(() => {
    document.body.classList.toggle("graph-lock", graphOpen);
    return () => document.body.classList.remove("graph-lock");
  }, [graphOpen]);

  useEffect(() => {
    if (!graphOpen) return;
    const panel = graphPanelRef.current;
    const previous = document.activeElement as HTMLElement | null;
    panel?.focus();

    const focusable = () =>
      [...(panel?.querySelectorAll<HTMLElement>("button, input, [href], [tabindex]:not([tabindex='-1'])") ?? [])].filter(
        (node) => !node.hasAttribute("disabled"),
      );

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setExpanded(null);
        return;
      }
      const typing = event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;
      if (!typing && (event.key === " " || event.code === "Space")) {
        event.preventDefault();
        setHeld(!held);
        return;
      }
      if (!showGlobe && !typing && (event.key === "+" || event.key === "=")) {
        event.preventDefault();
        graphControls.current?.zoomIn();
      } else if (!showGlobe && !typing && (event.key === "-" || event.key === "_")) {
        event.preventDefault();
        graphControls.current?.zoomOut();
      } else if (!showGlobe && !typing && event.key === "0") {
        event.preventDefault();
        graphControls.current?.fit();
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (items.length === 0) return;
      const index = items.indexOf(document.activeElement as HTMLElement);
      if (event.shiftKey && index <= 0) {
        event.preventDefault();
        items[items.length - 1]?.focus();
      } else if (!event.shiftKey && (index === -1 || index === items.length - 1)) {
        event.preventDefault();
        items[0]?.focus();
      }
    };

    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus();
    };
  }, [graphOpen, showGlobe, held, setHeld]);

  const toggle = (panel: PanelId) => setExpanded((current) => (current === panel ? null : panel));

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="kicker">Financial crime</p>
          <h1>Command center</h1>
        </div>
        <div className="topbar-switchers">
          <div className="switcher" role="group" aria-label="Source">
            <button
              type="button"
              aria-pressed={activeSource === "sim"}
              disabled={!availableSources.includes("sim")}
              onClick={() => setSource("sim")}
            >
              Simulator
            </button>
            <button
              type="button"
              aria-pressed={activeSource === "replay"}
              aria-label={replayLabel ? `Replay ${replayLabel}` : "Replay"}
              disabled={!replayAvailable}
              title={
                replayAvailable
                  ? replayLabel ?? "Replay"
                  : "Start aml-serve with --replay PATH to enable Replay"
              }
              onClick={() => setSource("replay")}
            >
              Replay
            </button>
          </div>
          <div className="switcher" role="group" aria-label="Corridor">
            {CORRIDORS.map((corridor) => (
              <button
                key={corridor.id}
                type="button"
                aria-pressed={sameFilter(state.networkFilter, corridor.value)}
                onClick={() => setNetworks(corridor.value)}
              >
                {corridor.label}
              </button>
            ))}
          </div>
        </div>
        <div className="top-actions">
          <span className={`status ${state.status}`}>
            <span className="dot" />
            {statusLabel}
          </span>
          <button type="button" className="ghost" aria-pressed={held} onClick={() => setHeld(!held)}>
            {held ? "Release" : "Hold"}
          </button>
        </div>
      </header>

      <section className="telemetry" aria-label="Engine telemetry">
        <span>
          Source{" "}
          {activeSource === "replay"
            ? replayLabel
              ? `replay · ${replayLabel}`
              : "replay"
            : "simulator"}
        </span>
        <span>Sim {formatSimTime(state.simTime)}</span>
        <span>Infer {state.inferMs == null ? "—" : `${state.inferMs.toFixed(0)} ms`}</span>
        <span>{openAlertCount(alerts)} open</span>
        <span>Threshold {threshold.toFixed(2)}</span>
        <span>{state.hello ? `${state.hello.rate.toFixed(0)} tx/s` : "— tx/s"}</span>
        {state.stats && (
          <span>
            {state.stats.nodes_by_network.banking ?? 0} banking · {state.stats.nodes_by_network.crypto ?? 0} crypto ·{" "}
            {state.stats.nodes_by_network.cross_border ?? 0} cross-border
          </span>
        )}
        {held && <span className="held-note">Held · {state.buffered} buffered</span>}
        {state.status !== "live" && <span className="held-note">Waiting for the engine on {WS_URL}</span>}
      </section>

      <main className={grow ? `workspace ${grow}` : "workspace"}>
        <section className="panel feed" id="panel-feed" aria-label="Live transactions">
          <header className="panel-head">
            <h2>Live feed</h2>
            <ExpandButton
              panel="feed"
              label="live feed"
              className="expand-desktop"
              expanded={expanded === "feed"}
              onClick={() => toggle("feed")}
            />
          </header>
          <div className="panel-body">
            <Feed
              rows={feed}
              selectedTxId={state.selectedTxId}
              threshold={threshold}
              showTypology={Boolean(state.hello?.reveal_labels)}
              onSelect={selectTx}
            />
          </div>
        </section>

        <section
          ref={graphPanelRef}
          id="panel-graph"
          className={graphOpen ? "panel graph fullscreen" : "panel graph"}
          aria-label="Money flow graph"
          role={graphOpen ? "dialog" : undefined}
          aria-modal={graphOpen ? true : undefined}
          aria-labelledby="graph-heading"
          aria-describedby={graphOpen ? "graph-help" : undefined}
          tabIndex={graphOpen ? -1 : undefined}
        >
          <header className="panel-head">
            <h2 id="graph-heading">
              {showGlobe ? (showCase ? "Alert globe" : "Live globe") : showCase ? "Alert trail" : "Live flow"}
            </h2>
            <div className="head-actions">
              <div className="graph-tools" role="toolbar" aria-label="Graph controls">
                <div className="view-toggle" role="group" aria-label="Graph view">
                  <button
                    type="button"
                    className="ghost"
                    aria-pressed={!showGlobe}
                    onClick={() => setGraphView("trail")}
                  >
                    Trail
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    aria-pressed={showGlobe}
                    onClick={() => setGraphView("globe")}
                  >
                    Globe
                  </button>
                </div>
                <input
                  type="search"
                  value={query}
                  placeholder="Find account"
                  aria-label="Find account"
                  onChange={(event) => setQuery(event.target.value)}
                />
                {graphOpen && (
                  <button
                    type="button"
                    className="ghost"
                    aria-pressed={held}
                    aria-keyshortcuts="Space"
                    aria-label={held ? "Release live stream" : "Hold live stream"}
                    onClick={() => setHeld(!held)}
                  >
                    {held ? "Release" : "Hold"}
                    {held && state.buffered > 0 ? ` · ${state.buffered}` : ""}
                  </button>
                )}
                {!showGlobe && graphOpen && (
                  <>
                    <button type="button" className="ghost" aria-label="Zoom in" aria-keyshortcuts="+" onClick={() => graphControls.current?.zoomIn()}>
                      Zoom in
                    </button>
                    <button type="button" className="ghost" aria-label="Zoom out" aria-keyshortcuts="-" onClick={() => graphControls.current?.zoomOut()}>
                      Zoom out
                    </button>
                    <button type="button" className="ghost" aria-label="Fit trail in view" aria-keyshortcuts="0" onClick={() => graphControls.current?.fit()}>
                      Fit
                    </button>
                  </>
                )}
                {!showGlobe && (
                  <button type="button" className="ghost" onClick={() => setReflow((n) => n + 1)}>
                    Reflow
                  </button>
                )}
              </div>
              <ExpandButton
                panel="graph"
                label="graph"
                expanded={graphOpen}
                onClick={() => toggle("graph")}
              />
            </div>
          </header>
          {graphOpen && (
            <p id="graph-help" className="graph-help">
              {showGlobe
                ? "Space holds the stream. Drag to orbit. Scroll to zoom. Escape closes."
                : "Space holds the stream. Plus and minus zoom. Zero fits the trail. Escape closes."}
              {held ? ` Held · ${state.buffered} buffered.` : ""}
            </p>
          )}
          {showGlobe ? (
            <GlobeView
              model={globeModel}
              emptyLabel={emptyGlobeLabel}
              onSelectNode={selectNode}
              onSelectEdge={selectTx}
            />
          ) : (
            <GraphCanvas
              ref={graphControls}
              elements={elements}
              layoutKey={layoutKey}
              layoutName={showCase ? "dagre" : "cose"}
              emptyLabel={emptyGraphLabel}
              onSelectNode={selectNode}
              onSelectEdge={selectTx}
            />
          )}
          <footer className="legend">
            <span><i className="swatch low" /> Clear</span>
            <span><i className="swatch mid" /> Watch</span>
            <span><i className="swatch high" /> High risk</span>
            <span><i className="swatch unknown" /> Scoring</span>
          </footer>
        </section>

        <section className="panel alerts" id="panel-alerts" aria-label="Alert queue">
          <header className="panel-head">
            <h2>Alert queue</h2>
            <ExpandButton
              panel="alerts"
              label="alert queue"
              className="expand-desktop"
              expanded={expanded === "alerts"}
              onClick={() => toggle("alerts")}
            />
          </header>
          <div className="panel-body">
            <AlertQueue
              alerts={alerts}
              selectedAlertId={state.selectedAlertId}
              threshold={threshold}
              onSelect={selectAlert}
            />
          </div>
        </section>

        <section className="panel investigator" id="panel-investigator" aria-label="Investigator">
          <header className="panel-head">
            <h2>Investigator</h2>
            <ExpandButton
              panel="investigator"
              label="investigator"
              className="expand-desktop"
              expanded={expanded === "investigator"}
              onClick={() => toggle("investigator")}
            />
          </header>
          <div className="panel-body">
            <Investigator
              alert={showCase ? selectedAlert : null}
              tx={showCase ? null : selectedTx}
              nodeId={showCase || selectedTx ? null : state.selectedNodeId}
              nodeScore={
                state.selectedNodeId ? (state.partyRisk[state.selectedNodeId] ?? null) : null
              }
              threshold={threshold}
              onDisposition={sendDisposition}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
