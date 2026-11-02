import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import Globe, { type GlobeMethods } from "react-globe.gl";

import {
  arcAltitude,
  bandColor,
  bandMutedColor,
  bandPulseColor,
  formatDistanceKm,
  type GlobeArc,
  type GlobeModel,
  type GlobePoint,
} from "./globeData";

type Props = {
  model: GlobeModel;
  emptyLabel: string;
  onSelectNode: (nodeId: string) => void;
  onSelectEdge: (txId: string) => void;
};

type PointDatum = GlobePoint & {
  color: string;
  altitude: number;
  radius: number;
};

type ArcLayer = "base" | "pulse";

type ArcDatum = GlobeArc & {
  layerId: string;
  layer: ArcLayer;
  color: string;
  altitude: number;
  stroke: number;
  dashLength: number;
  dashGap: number;
  dashAnimateTime: number;
  dashInitialGap: number;
};

const DATA_THROTTLE_MS = 450;
const AUTOROTATE_SPEED = 0.18;
const RESUME_ROTATE_MS = 4500;
const FOCUS_ALTITUDE = 1.85;
const DEFAULT_POV = { lat: 18, lng: 8, altitude: 2.15 };

// Bundled NASA Blue Marble (public domain) + topology + starfield — no CDN.
const EARTH_TEXTURE = "/globe/earth-blue-marble.jpg";
const EARTH_BUMP = "/globe/earth-topology.png";
const NIGHT_SKY = "/globe/night-sky.png";

function pointLabel(point: GlobePoint): string {
  const place = [point.city, point.country].filter(Boolean).join(", ");
  const score = point.score == null ? "scoring" : point.score.toFixed(2);
  return `${point.id}${place ? ` · ${place}` : ""}\nRisk ${score}`;
}

function arcLabel(arc: GlobeArc): string {
  const places = [arc.sourceCity, arc.destCity].filter(Boolean).join(" → ");
  const score = arc.score == null ? "scoring" : arc.score.toFixed(2);
  return `${arc.sourceId} → ${arc.destId}${places ? `\n${places}` : ""}\n${formatDistanceKm(arc.distanceKm)} · risk ${score}`;
}

function txHash(txId: string): number {
  let hash = 0;
  for (let i = 0; i < txId.length; i += 1) hash = (hash * 31 + txId.charCodeAt(i)) >>> 0;
  return hash;
}

function toPointDatum(point: GlobePoint): PointDatum {
  return {
    ...point,
    color: bandColor(point.band),
    altitude: point.band === "high" ? 0.018 : 0.008,
    radius: point.selected || point.center ? 0.52 : point.band === "high" ? 0.42 : 0.3,
  };
}

function toBaseArc(arc: GlobeArc): ArcDatum {
  const stroke = arc.selected ? 0.85 : arc.band === "high" ? 0.7 : 0.45;
  return {
    ...arc,
    layerId: `${arc.txId}:base`,
    layer: "base",
    color: bandMutedColor(arc.band),
    altitude: arcAltitude(arc.band),
    stroke,
    // Full solid corridor (no dash animation).
    dashLength: 1,
    dashGap: 0,
    dashAnimateTime: 0,
    dashInitialGap: 0,
  };
}

function toPulseArc(arc: GlobeArc): ArcDatum {
  const hash = txHash(arc.txId);
  const speed =
    arc.band === "high" ? 1800 : arc.band === "mid" ? 2400 : 3200;
  return {
    ...arc,
    layerId: `${arc.txId}:pulse`,
    layer: "pulse",
    color: bandPulseColor(arc.band),
    altitude: arcAltitude(arc.band),
    stroke: arc.selected ? 1.6 : arc.band === "high" ? 1.35 : 0.95,
    // Short bright segment that travels source → destination.
    dashLength: 0.12,
    dashGap: 1,
    dashAnimateTime: speed,
    dashInitialGap: (hash % 100) / 100,
  };
}

function expandArcs(arcs: GlobeArc[]): ArcDatum[] {
  const out: ArcDatum[] = [];
  for (const arc of arcs) {
    out.push(toBaseArc(arc));
    out.push(toPulseArc(arc));
  }
  return out;
}

function pointVisualKey(point: GlobePoint): string {
  return [
    point.lat.toFixed(4),
    point.lng.toFixed(4),
    point.band,
    point.dim ? "1" : "0",
    point.selected ? "1" : "0",
    point.center ? "1" : "0",
    point.score ?? "",
  ].join("|");
}

function arcVisualKey(arc: GlobeArc & { layer?: ArcLayer }): string {
  return [
    arc.layer ?? "",
    arc.startLat.toFixed(4),
    arc.startLng.toFixed(4),
    arc.endLat.toFixed(4),
    arc.endLng.toFixed(4),
    arc.band,
    arc.dim ? "1" : "0",
    arc.selected ? "1" : "0",
    arc.score ?? "",
  ].join("|");
}

function stabilizePoints(prev: PointDatum[], next: GlobePoint[]): PointDatum[] {
  const prior = new Map(prev.map((p) => [p.id, p]));
  return next.map((point) => {
    const old = prior.get(point.id);
    if (old && pointVisualKey(old) === pointVisualKey(point)) return old;
    return toPointDatum(point);
  });
}

function stabilizeArcs(prev: ArcDatum[], next: GlobeArc[]): ArcDatum[] {
  const prior = new Map(prev.map((a) => [a.layerId, a]));
  return expandArcs(next).map((arc) => {
    const old = prior.get(arc.layerId);
    if (old && arcVisualKey(old) === arcVisualKey(arc)) return old;
    return arc;
  });
}

function useContainerSize(ref: RefObject<HTMLDivElement | null>) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        setSize((prev) => {
          const next = {
            width: Math.round(node.clientWidth),
            height: Math.round(node.clientHeight),
          };
          if (next.width === prev.width && next.height === prev.height) return prev;
          if (next.width < 40 || next.height < 40) return prev;
          return next;
        });
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [ref]);
  return size;
}

function useThrottledModel(model: GlobeModel, ms: number): GlobeModel {
  const [shown, setShown] = useState(model);
  const pending = useRef(model);
  const lastFlush = useRef(0);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    pending.current = model;
    const now = performance.now();
    const elapsed = now - lastFlush.current;
    if (elapsed >= ms) {
      lastFlush.current = now;
      setShown(model);
      return;
    }
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      lastFlush.current = performance.now();
      setShown(pending.current);
    }, ms - elapsed);
    return () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
        timer.current = null;
      }
    };
  }, [model, ms]);

  return shown;
}

function focusKey(model: GlobeModel): string | null {
  const selected = model.points.find((p) => p.selected);
  if (selected) return `node:${selected.id}`;
  const center = model.points.find((p) => p.center);
  if (center) return `center:${center.id}`;
  const selectedArc = model.arcs.find((a) => a.selected);
  if (selectedArc) return `arc:${selectedArc.txId}`;
  return null;
}

function focusTarget(model: GlobeModel): { lat: number; lng: number } | null {
  const selected = model.points.find((p) => p.selected);
  if (selected) return selected;
  const center = model.points.find((p) => p.center);
  if (center) return center;
  const selectedArc = model.arcs.find((a) => a.selected);
  if (selectedArc) {
    return {
      lat: (selectedArc.startLat + selectedArc.endLat) / 2,
      lng: (selectedArc.startLng + selectedArc.endLng) / 2,
    };
  }
  return null;
}

export function GlobeView({ model, emptyLabel, onSelectNode, onSelectEdge }: Props) {
  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const hostRef = useRef<HTMLDivElement>(null);
  const size = useContainerSize(hostRef);
  const throttled = useThrottledModel(model, DATA_THROTTLE_MS);
  const empty = throttled.points.length === 0 && throttled.arcs.length === 0;

  const pointsRef = useRef<PointDatum[]>([]);
  const arcsRef = useRef<ArcDatum[]>([]);
  const focusSeen = useRef<string | null>(null);
  const resumeTimer = useRef<number | null>(null);
  const ready = useRef(false);

  const points = useMemo(() => {
    const next = stabilizePoints(pointsRef.current, throttled.points);
    pointsRef.current = next;
    return next;
  }, [throttled.points]);

  const arcs = useMemo(() => {
    const next = stabilizeArcs(arcsRef.current, throttled.arcs);
    arcsRef.current = next;
    return next;
  }, [throttled.arcs]);

  const pauseRotate = useCallback(() => {
    const globe = globeRef.current;
    if (!globe) return;
    const controls = globe.controls();
    controls.autoRotate = false;
    if (resumeTimer.current !== null) window.clearTimeout(resumeTimer.current);
    resumeTimer.current = window.setTimeout(() => {
      resumeTimer.current = null;
      const live = globeRef.current;
      if (!live) return;
      live.controls().autoRotate = true;
    }, RESUME_ROTATE_MS);
  }, []);

  const configureControls = useCallback(() => {
    const globe = globeRef.current;
    if (!globe) return;
    const controls = globe.controls();
    controls.autoRotate = true;
    controls.autoRotateSpeed = AUTOROTATE_SPEED;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 120;
    controls.maxDistance = 500;
    controls.addEventListener("start", pauseRotate);
  }, [pauseRotate]);

  useEffect(() => {
    return () => {
      if (resumeTimer.current !== null) window.clearTimeout(resumeTimer.current);
      const globe = globeRef.current;
      if (!globe) return;
      try {
        globe.controls().removeEventListener("start", pauseRotate);
      } catch {
        // Globe already disposed.
      }
    };
  }, [pauseRotate]);

  useEffect(() => {
    const globe = globeRef.current;
    if (!globe || !ready.current || empty) return;
    const key = focusKey(throttled);
    if (!key) {
      focusSeen.current = null;
      return;
    }
    if (key === focusSeen.current) return;
    const target = focusTarget(throttled);
    if (!target) return;
    focusSeen.current = key;
    pauseRotate();
    globe.pointOfView({ lat: target.lat, lng: target.lng, altitude: FOCUS_ALTITUDE }, 1200);
  }, [throttled, empty, pauseRotate]);

  const onReady = useCallback(() => {
    ready.current = true;
    configureControls();
    const globe = globeRef.current;
    if (!globe) return;
    globe.pointOfView(DEFAULT_POV, 0);
  }, [configureControls]);

  const handlePointClick = useCallback(
    (d: object) => {
      const point = d as GlobePoint;
      if (point?.id) onSelectNode(point.id);
    },
    [onSelectNode],
  );

  const handleArcClick = useCallback(
    (d: object) => {
      const arc = d as ArcDatum;
      if (arc?.txId) onSelectEdge(arc.txId);
    },
    [onSelectEdge],
  );

  const pointColor = useCallback((d: object) => {
    const point = d as PointDatum;
    return point.dim ? "#6e6a6040" : point.color;
  }, []);

  const arcColor = useCallback((d: object) => {
    const arc = d as ArcDatum;
    if (arc.dim) return arc.layer === "pulse" ? "#6e6a6018" : "#6e6a6028";
    return arc.color;
  }, []);

  const pointLabelFn = useCallback((d: object) => pointLabel(d as GlobePoint), []);
  const arcLabelFn = useCallback((d: object) => arcLabel(d as GlobeArc), []);

  const readyToDraw = size.width > 0 && size.height > 0;

  return (
    <div className="graph-stage globe-stage" ref={hostRef}>
      {readyToDraw && (
        <Globe
          ref={globeRef}
          width={size.width}
          height={size.height}
          backgroundColor="#010208"
          backgroundImageUrl={NIGHT_SKY}
          globeImageUrl={EARTH_TEXTURE}
          bumpImageUrl={EARTH_BUMP}
          showAtmosphere
          atmosphereColor="#6eb6ff"
          atmosphereAltitude={0.18}
          animateIn={false}
          waitForGlobeReady
          onGlobeReady={onReady}
          rendererConfig={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
          pointsData={empty ? [] : points}
          pointLat="lat"
          pointLng="lng"
          pointAltitude="altitude"
          pointRadius="radius"
          pointResolution={10}
          pointsTransitionDuration={0}
          pointColor={pointColor}
          pointLabel={pointLabelFn}
          onPointClick={handlePointClick}
          arcsData={empty ? [] : arcs}
          arcStartLat="startLat"
          arcStartLng="startLng"
          arcEndLat="endLat"
          arcEndLng="endLng"
          arcAltitude="altitude"
          arcStroke="stroke"
          arcCurveResolution={64}
          arcsTransitionDuration={0}
          arcColor={arcColor}
          arcDashLength="dashLength"
          arcDashGap="dashGap"
          arcDashInitialGap="dashInitialGap"
          arcDashAnimateTime="dashAnimateTime"
          arcLabel={arcLabelFn}
          onArcClick={handleArcClick}
        />
      )}
      {empty && <div className="empty-overlay">{emptyLabel}</div>}
    </div>
  );
}
