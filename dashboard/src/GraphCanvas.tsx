import cytoscape, { type Core, type ElementDefinition, type LayoutOptions } from "cytoscape";
import dagre from "cytoscape-dagre";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

import { chooseFrameZoom, clampedFitZoom, FIT_PADDING, panToCenter } from "./graphView";

cytoscape.use(dagre);

const STYLES: cytoscape.StylesheetStyle[] = [
  {
    selector: "node",
    style: {
      "background-color": "#6e6a60",
      label: "data(label)",
      color: "#efece2",
      "font-size": 10,
      "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
      "text-valign": "bottom",
      "text-margin-y": 6,
      "min-zoomed-font-size": 3,
      width: 20,
      height: 20,
      opacity: 1,
      "transition-property": "opacity, width, height, background-color",
      "transition-duration": 0.4,
      "transition-timing-function": "ease-out",
      "border-width": 1,
      "border-color": "#14160f",
    },
  },
  { selector: "node.risk-low", style: { "background-color": "#8fbf73" } },
  { selector: "node.risk-mid", style: { "background-color": "#e0a34a" } },
  {
    selector: "node.risk-high",
    style: { "background-color": "#e15d4a", width: 26, height: 26 },
  },
  { selector: "node.risk-unknown", style: { "background-color": "#6e6a60" } },
  {
    selector: "node.center, node.picked, node.hit",
    style: { "border-width": 3, "border-color": "#f4f0e4", width: 28, height: 28 },
  },
  { selector: "node.kind-mixer", style: { shape: "diamond" } },
  { selector: "node.kind-shell, node.kind-business", style: { shape: "round-rectangle" } },
  { selector: "node.kind-smart_contract", style: { shape: "hexagon" } },
  { selector: "node.dim", style: { opacity: 0.15 } },
  { selector: "node.entering, edge.entering", style: { opacity: 0 } },
  {
    selector: "edge",
    style: {
      width: 1.4,
      "line-color": "#8d8878",
      "target-arrow-color": "#8d8878",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      "arrow-scale": 0.8,
      label: "data(label)",
      "font-size": 9,
      color: "#c8c2b2",
      "text-background-color": "#1a1d14",
      "text-background-opacity": 0.85,
      "text-background-padding": "2px",
      opacity: 0.9,
      "transition-property": "opacity, width, line-color",
      "transition-duration": 0.45,
      "transition-timing-function": "ease-out",
      "control-point-step-size": 56,
    },
  },
  {
    selector: "edge.risk-low",
    style: { "line-color": "#8fbf73", "target-arrow-color": "#8fbf73" },
  },
  {
    selector: "edge.risk-mid",
    style: { "line-color": "#e0a34a", "target-arrow-color": "#e0a34a", width: 2.2 },
  },
  {
    selector: "edge.risk-high",
    style: { "line-color": "#e15d4a", "target-arrow-color": "#e15d4a", width: 2.8 },
  },
  {
    selector: "edge.selected",
    style: { width: 3.4, "line-color": "#f4f0e4", "target-arrow-color": "#f4f0e4" },
  },
  { selector: "edge.dim", style: { opacity: 0.08 } },
];

function ringPosition(origin: { x: number; y: number }, slot: number): { x: number; y: number } {
  const angle = (slot % 12) * ((Math.PI * 2) / 12);
  const dist = 160 + Math.floor(slot / 12) * 36;
  return {
    x: origin.x + Math.cos(angle) * dist,
    y: origin.y + Math.sin(angle) * dist,
  };
}

function unstack(cy: Core) {
  const groups = new Map<string, cytoscape.NodeSingular[]>();
  cy.nodes().forEach((node) => {
    const p = node.position();
    const key = `${Math.round(p.x / 8)}:${Math.round(p.y / 8)}`;
    const list = groups.get(key) ?? [];
    list.push(node);
    groups.set(key, list);
  });
  groups.forEach((list) => {
    list.forEach((node, index) => {
      if (index === 0) return;
      const p = node.position();
      const angle = index * 2.39996;
      node.position({
        x: p.x + Math.cos(angle) * (160 + index * 6),
        y: p.y + Math.sin(angle) * (160 + index * 6),
      });
    });
  });
}

function focusElements(cy: Core) {
  const marked = cy.nodes(".center, .picked, .hit");
  return marked.nonempty() ? marked : cy.nodes().slice(-6);
}

function frameGraph(cy: Core, animate: boolean, preferAll = false) {
  if (cy.nodes().empty() || cy.width() < 40 || cy.height() < 40) return;
  const viewport = { w: cy.width(), h: cy.height() };
  const all = cy.elements();
  const focus = focusElements(cy);
  const fullZoom = clampedFitZoom(all.boundingBox({ includeLabels: true }), viewport);
  const focusZoom = focus.nonempty()
    ? clampedFitZoom(focus.boundingBox({ includeLabels: true }), viewport)
    : fullZoom;
  const choice = chooseFrameZoom(fullZoom, focusZoom, preferAll);
  const framed = choice.useFocus && focus.nonempty() ? focus : all;
  const box = framed.boundingBox({ includeLabels: true });
  const zoom = choice.zoom;
  const pan = panToCenter(box, viewport, zoom);
  const current = cy.pan();
  const settled =
    Math.abs(cy.zoom() - zoom) < 0.02 &&
    Math.hypot(current.x - pan.x, current.y - pan.y) < 16;
  if (settled) return;
  if (!animate) {
    cy.viewport({ zoom, pan });
    return;
  }
  cy.stop(true, false);
  cy.animate({
    zoom,
    pan,
    duration: 320,
    easing: "ease-out-cubic",
  });
}

function runLayout(cy: Core, name: "cose" | "dagre", animate: boolean) {
  cy.resize();
  if (cy.nodes().empty()) return;
  if (name === "cose") unstack(cy);
  const motion = animate
    ? { animate: true, animationDuration: 560, animationEasing: "ease-out-cubic" }
    : { animate: false };
  const options = (
    name === "cose"
      ? {
          name: "cose",
          fit: false,
          padding: FIT_PADDING,
          nodeRepulsion: 90000,
          idealEdgeLength: 190,
          nodeOverlap: 64,
          gravity: 0.12,
          numIter: 420,
          componentSpacing: 200,
          randomize: !animate,
          ...motion,
        }
      : {
          name: "dagre",
          rankDir: "LR",
          fit: false,
          padding: FIT_PADDING,
          nodeSep: 70,
          rankSep: 150,
          edgeSep: 32,
          ...motion,
        }
  ) as unknown as LayoutOptions;
  const layout = cy.layout(options);
  const focus = () => {
    cy.nodes().forEach((node) => {
      node.data("placed", true);
    });
    frameGraph(cy, false);
  };
  if (animate) layout.one("layoutstop", focus);
  layout.run();
  if (!animate) focus();
}

export type GraphControls = {
  zoomIn: () => void;
  zoomOut: () => void;
  fit: () => void;
};

function zoomBy(cy: Core, factor: number) {
  cy.stop(true, false);
  const level = Math.min(cy.maxZoom(), Math.max(cy.minZoom(), cy.zoom() * factor));
  cy.animate({
    zoom: {
      level,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
    },
    duration: 220,
    easing: "ease-out-cubic",
  });
}

function freshPosition(cy: Core, id: string, slot: number): { x: number; y: number } {
  const others = cy.nodes().filter((node) => node.id() !== id);
  if (others.empty()) return { x: 0, y: 0 };
  const box = others.boundingBox();
  return ringPosition({ x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 }, slot);
}

type Props = {
  elements: ElementDefinition[];
  layoutKey: string;
  layoutName: "cose" | "dagre";
  emptyLabel: string;
  onSelectNode: (id: string) => void;
  onSelectEdge: (txId: string) => void;
};

export const GraphCanvas = forwardRef<GraphControls, Props>(function GraphCanvas(
  { elements, layoutKey, layoutName, emptyLabel, onSelectNode, onSelectEdge },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const onNode = useRef(onSelectNode);
  const onEdge = useRef(onSelectEdge);
  const prevCount = useRef(0);
  const seenLayout = useRef<string | null>(null);
  const seenIds = useRef("");
  const userControlled = useRef(false);
  const fitTimer = useRef<number | null>(null);
  const fitWaitStarted = useRef<number | null>(null);
  const panning = useRef(false);
  onNode.current = onSelectNode;
  onEdge.current = onSelectEdge;

  const holdView = () => {
    userControlled.current = true;
    fitWaitStarted.current = null;
    if (fitTimer.current !== null) {
      window.clearTimeout(fitTimer.current);
      fitTimer.current = null;
    }
  };

  const scheduleFit = (cy: Core) => {
    if (userControlled.current) return;
    const now = Date.now();
    if (fitWaitStarted.current === null) fitWaitStarted.current = now;
    const delay = now - fitWaitStarted.current > 700 ? 0 : 520;
    if (fitTimer.current !== null) window.clearTimeout(fitTimer.current);
    fitTimer.current = window.setTimeout(() => {
      fitTimer.current = null;
      fitWaitStarted.current = null;
      if (!userControlled.current) frameGraph(cy, true);
    }, delay);
  };

  useImperativeHandle(ref, () => ({
    zoomIn: () => {
      const cy = cyRef.current;
      if (!cy) return;
      holdView();
      zoomBy(cy, 1.25);
    },
    zoomOut: () => {
      const cy = cyRef.current;
      if (!cy) return;
      holdView();
      zoomBy(cy, 0.8);
    },
    fit: () => {
      const cy = cyRef.current;
      if (!cy || cy.nodes().empty()) return;
      userControlled.current = false;
      fitWaitStarted.current = null;
      if (fitTimer.current !== null) {
        window.clearTimeout(fitTimer.current);
        fitTimer.current = null;
      }
      frameGraph(cy, true, true);
    },
  }));

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const cy = cytoscape({
      container: host,
      elements: [],
      style: STYLES,
      wheelSensitivity: 0.2,
      minZoom: 0.08,
      maxZoom: 2.2,
      pixelRatio: "auto",
      autoungrabify: true,
    });
    cy.on("tap", "node", (event) => {
      onNode.current(event.target.id());
    });
    cy.on("tap", "edge", (event) => {
      const txId = event.target.data("txId");
      if (typeof txId === "string" && txId) onEdge.current(txId);
    });
    cy.on("mousedown", (event) => {
      if (event.target === cy) panning.current = true;
    });
    cy.on("mouseup", () => {
      panning.current = false;
    });
    cy.on("pan", () => {
      if (panning.current) holdView();
    });
    const onWheel = () => holdView();
    host.addEventListener("wheel", onWheel, { passive: true });
    cyRef.current = cy;
    const observer = new ResizeObserver(() => {
      cy.resize();
      if (!userControlled.current) frameGraph(cy, false);
    });
    observer.observe(host);
    return () => {
      observer.disconnect();
      host.removeEventListener("wheel", onWheel);
      if (fitTimer.current !== null) window.clearTimeout(fitTimer.current);
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const nodes = elements.filter((el) => !el.data.source);
    const edges = elements.filter((el) => el.data.source);
    const keep = new Set(elements.map((el) => String(el.data.id)));
    const fresh: string[] = [];
    const slides: { node: cytoscape.NodeSingular; from: { x: number; y: number }; to: { x: number; y: number } }[] = [];
    let slot = cy.nodes().length;
    cy.batch(() => {
      cy.elements().forEach((el) => {
        if (!keep.has(el.id())) el.remove();
      });
      for (const el of nodes) {
        const id = String(el.data.id);
        const existing = cy.getElementById(id);
        if (existing.nonempty()) {
          existing.data({ ...el.data, placed: existing.data("placed") });
          existing.classes(String(el.classes ?? ""));
          continue;
        }
        fresh.push(id);
        cy.add({
          group: "nodes",
          data: { ...el.data, placed: false },
          classes: `${el.classes ?? ""} entering`.trim(),
          position: freshPosition(cy, id, slot),
        });
        slot += 1;
      }
      for (const el of edges) {
        const id = String(el.data.id);
        const existing = cy.getElementById(id);
        if (existing.nonempty()) {
          existing.data(el.data);
          existing.classes(String(el.classes ?? ""));
          continue;
        }
        if (!el.data.source || !el.data.target) continue;
        if (cy.getElementById(String(el.data.source)).empty()) continue;
        if (cy.getElementById(String(el.data.target)).empty()) continue;
        fresh.push(id);
        cy.add({
          group: "edges",
          data: el.data,
          classes: `${el.classes ?? ""} entering`.trim(),
        });
      }
      const used = new Map<string, number>();
      cy.nodes().forEach((node) => {
        if (node.data("placed")) return;
        const neighbor = node.neighborhood("node").first();
        if (!neighbor.empty() && neighbor.isNode() && neighbor.data("placed")) {
          const origin = neighbor.position();
          const nextSlot = used.get(neighbor.id()) ?? neighbor.neighborhood("node").length;
          used.set(neighbor.id(), nextSlot + 1);
          const next = ringPosition(origin, nextSlot);
          slides.push({ node, from: origin, to: next });
          node.position(next);
        }
        node.data("placed", true);
      });
    });
    for (const slide of slides) {
      slide.node.position(slide.from);
      slide.node.animate({ position: slide.to, duration: 420, easing: "ease-out-cubic" });
    }
    if (fresh.length > 0) {
      requestAnimationFrame(() => {
        for (const id of fresh) {
          const el = cy.getElementById(id);
          if (el.nonempty()) el.removeClass("entering");
        }
      });
    }
    const count = cy.elements().length;
    const ids = elements.map((el) => String(el.data.id)).join("|");
    const structureChanged = ids !== seenIds.current;
    seenIds.current = ids;
    if (prevCount.current === 0 && count > 0) runLayout(cy, layoutName, false);
    else if (structureChanged) scheduleFit(cy);
    prevCount.current = count;
  }, [elements, layoutName]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const key = `${layoutKey}|${layoutName}`;
    if (seenLayout.current === null) {
      seenLayout.current = key;
      return;
    }
    if (seenLayout.current === key) return;
    seenLayout.current = key;
    userControlled.current = false;
    fitWaitStarted.current = null;
    if (fitTimer.current !== null) {
      window.clearTimeout(fitTimer.current);
      fitTimer.current = null;
    }
    runLayout(cy, layoutName, true);
  }, [layoutKey, layoutName]);

  return (
    <div className="graph-stage">
      <div className="graph-host" ref={hostRef} />
      {elements.length === 0 && <div className="empty-overlay">{emptyLabel}</div>}
    </div>
  );
});
