/** Automatic framing never zooms past the nodes' native size. */
export const AUTO_ZOOM_MAX = 1;
/** Below this the nodes collapse into specks, so framing follows the active account instead. */
export const AUTO_ZOOM_MIN = 0.18;
export const FIT_PADDING = 48;

export function chooseFrameZoom(
  fullZoom: number,
  focusZoom: number,
  preferAll = false,
  maxZoom = AUTO_ZOOM_MAX,
  minZoom = AUTO_ZOOM_MIN,
): { zoom: number; useFocus: boolean } {
  if (preferAll || fullZoom >= minZoom) {
    return { zoom: Math.min(maxZoom, Math.max(minZoom, fullZoom)), useFocus: false };
  }
  return { zoom: Math.min(maxZoom, Math.max(minZoom, focusZoom)), useFocus: true };
}

export function clampedFitZoom(
  box: { w: number; h: number },
  viewport: { w: number; h: number },
  padding = FIT_PADDING,
  maxZoom = AUTO_ZOOM_MAX,
): number {
  const availW = Math.max(viewport.w - padding * 2, 1);
  const availH = Math.max(viewport.h - padding * 2, 1);
  const spanW = Math.max(box.w, 1);
  const spanH = Math.max(box.h, 1);
  return Math.min(maxZoom, availW / spanW, availH / spanH);
}

export function panToCenter(
  box: { x1: number; y1: number; x2: number; y2: number },
  viewport: { w: number; h: number },
  zoom: number,
): { x: number; y: number } {
  const centerX = (box.x1 + box.x2) / 2;
  const centerY = (box.y1 + box.y2) / 2;
  return {
    x: viewport.w / 2 - centerX * zoom,
    y: viewport.h / 2 - centerY * zoom,
  };
}
