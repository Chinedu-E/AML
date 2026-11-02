import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import {
  ALL_NETWORKS,
  parseServerMessage,
  type DispositionAction,
  type NetworkId,
  type ServerMessage,
  type SourceId,
} from "./protocol";
import { initialState, reduce } from "./reduce";

const BUFFER_CAP = 400;

export function useEngine(url: string) {
  const [state, dispatch] = useReducer(reduce, initialState);
  const [held, setHeldState] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const heldRef = useRef(false);
  const bufferRef = useRef<ServerMessage[]>([]);
  const filterRef = useRef<NetworkId[] | null>(null);
  const retryTimer = useRef<number | null>(null);

  const ingest = useCallback((message: ServerMessage) => {
    if (message.type === "reset") {
      bufferRef.current = [];
      dispatch({ type: "buffered", count: 0 });
      dispatch({ type: "server", message });
      return;
    }
    if (heldRef.current && message.type !== "hello") {
      bufferRef.current.push(message);
      if (bufferRef.current.length > BUFFER_CAP) bufferRef.current.shift();
      dispatch({ type: "buffered", count: bufferRef.current.length });
      return;
    }
    dispatch({ type: "server", message });
  }, []);

  useEffect(() => {
    let closed = false;
    let retryMs = 1000;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(url);
      socket = ws;
      wsRef.current = ws;
      ws.onopen = () => {
        retryMs = 1000;
        dispatch({ type: "status", status: "live" });
        ws.send(
          JSON.stringify({
            op: "subscribe",
            networks: filterRef.current ?? [...ALL_NETWORKS],
          }),
        );
      };
      ws.onmessage = (event) => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(String(event.data));
        } catch {
          return;
        }
        const message = parseServerMessage(parsed);
        if (message) ingest(message);
      };
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
        if (closed) return;
        dispatch({ type: "status", status: "reconnecting" });
        retryTimer.current = window.setTimeout(connect, retryMs);
        retryMs = Math.min(retryMs * 2, 8000);
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
      socket?.close();
    };
  }, [url, ingest]);

  const setHeld = useCallback((next: boolean) => {
    heldRef.current = next;
    setHeldState(next);
    if (!next) {
      const queued = bufferRef.current.splice(0);
      dispatch({ type: "flush", messages: queued });
    }
  }, []);

  const setNetworks = useCallback((networks: NetworkId[] | null) => {
    filterRef.current = networks;
    dispatch({ type: "filter", networks });
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          op: "subscribe",
          networks: networks ?? [...ALL_NETWORKS],
        }),
      );
    }
  }, []);

  const sendDisposition = useCallback((alertId: string, action: DispositionAction) => {
    dispatch({
      type: "server",
      message: { type: "disposition", alert_id: alertId, action },
    });
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ op: "disposition", alert_id: alertId, action }));
    }
  }, []);

  const selectAlert = useCallback((alertId: string | null) => {
    dispatch({ type: "select-alert", alertId });
  }, []);

  const selectTx = useCallback((txId: string | null) => {
    dispatch({ type: "select-tx", txId });
  }, []);

  const selectNode = useCallback((nodeId: string) => {
    dispatch({ type: "select-node", nodeId });
  }, []);

  const setSource = useCallback((name: SourceId) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ op: "source", name }));
    }
  }, []);

  return {
    state,
    held,
    setHeld,
    setNetworks,
    setSource,
    sendDisposition,
    selectAlert,
    selectTx,
    selectNode,
  };
}
