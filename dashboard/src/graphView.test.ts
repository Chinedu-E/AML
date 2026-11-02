import { describe, expect, it } from "vitest";

import { AUTO_ZOOM_MAX, chooseFrameZoom, clampedFitZoom, panToCenter } from "./graphView";

describe("clampedFitZoom", () => {
  it("does not zoom a small graph past native size", () => {
    const zoom = clampedFitZoom({ w: 40, h: 40 }, { w: 900, h: 600 });
    expect(zoom).toBe(AUTO_ZOOM_MAX);
  });

  it("zooms a wide graph out until it fits", () => {
    const zoom = clampedFitZoom({ w: 2000, h: 400 }, { w: 800, h: 600 });
    expect(zoom).toBeLessThan(1);
    expect(zoom).toBeCloseTo((800 - 96) / 2000, 5);
  });

  it("stays finite when the box is empty", () => {
    const zoom = clampedFitZoom({ w: 0, h: 0 }, { w: 400, h: 300 });
    expect(zoom).toBe(AUTO_ZOOM_MAX);
  });
});

describe("chooseFrameZoom", () => {
  it("keeps a small graph at native size", () => {
    expect(chooseFrameZoom(4, 4)).toEqual({ zoom: AUTO_ZOOM_MAX, useFocus: false });
  });

  it("zooms a medium graph out until every node fits", () => {
    expect(chooseFrameZoom(0.72, 1)).toEqual({ zoom: 0.72, useFocus: false });
  });

  it("follows the active account at a readable size when the whole graph would collapse", () => {
    expect(chooseFrameZoom(0.09, 4)).toEqual({ zoom: AUTO_ZOOM_MAX, useFocus: true });
    expect(chooseFrameZoom(0.09, 0.4)).toEqual({ zoom: 0.4, useFocus: true });
  });

  it("fits the whole trail on request without zooming past native size", () => {
    expect(chooseFrameZoom(0.09, 4, true)).toEqual({ zoom: 0.18, useFocus: false });
    expect(chooseFrameZoom(3, 3, true)).toEqual({ zoom: AUTO_ZOOM_MAX, useFocus: false });
  });
});

describe("panToCenter", () => {
  it("places the box center in the middle of the viewport", () => {
    const pan = panToCenter({ x1: 0, y1: 0, x2: 100, y2: 40 }, { w: 200, h: 100 }, 1);
    expect(pan).toEqual({ x: 50, y: 30 });
  });
});
