import { useState, useEffect, useRef, useCallback } from 'react';

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: any) => void;
  enabled?: boolean;
}

interface UseWebSocketResult {
  connectionState: ConnectionState;
}

const INITIAL_BACKOFF = 1000;
const MAX_BACKOFF = 30000;
const BACKOFF_MULTIPLIER = 1.5;

/**
 * Robust WebSocket hook with exponential backoff reconnect.
 */
export function useWebSocket({ url, onMessage, enabled = true }: UseWebSocketOptions): UseWebSocketResult {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!enabled) return;

    setConnectionState(prev => (prev === 'connected' ? 'connected' : prev === 'disconnected' ? 'connecting' : 'reconnecting'));

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState('connected');
      reconnectAttemptRef.current = 0; // Reset backoff on success
      console.log(`[WS] Connected to ${url}`);
    };

    ws.onmessage = (event) => {
      if (onMessage) {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (e) {
          console.error('[WS] Failed to parse message', event.data);
        }
      }
    };

    ws.onclose = (event) => {
      console.log(`[WS] Disconnected (code: ${event.code}, reason: ${event.reason})`);
      setConnectionState('disconnected');
      scheduleReconnect();
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      // onclose will typically fire right after onerror, so we rely on onclose for reconnects
    };
  }, [url, enabled, onMessage]);

  const scheduleReconnect = useCallback(() => {
    if (!enabled) return;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    // Exponential backoff with jitter
    let backoff = INITIAL_BACKOFF * Math.pow(BACKOFF_MULTIPLIER, reconnectAttemptRef.current);
    backoff = Math.min(backoff, MAX_BACKOFF);
    const jitter = backoff * 0.2 * (Math.random() - 0.5); // ±10% jitter
    const timeout = backoff + jitter;

    console.log(`[WS] Scheduling reconnect in ${Math.round(timeout)}ms (attempt ${reconnectAttemptRef.current + 1})`);
    
    setConnectionState('reconnecting');
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectAttemptRef.current += 1;
      connect();
    }, timeout);
  }, [connect, enabled]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { connectionState };
}
