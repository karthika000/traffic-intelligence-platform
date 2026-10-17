// The single connection to /ws/live. It reconnects with exponential backoff and jitter,
// treats a silent socket as dead (the API sends heartbeats every 15 s when idle), and hands
// data messages over in batches so a burst of messages costs one store update, not hundreds.
import type { ControlMessage, LiveEnvelope } from "../api/types";

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "closed";

export type StatusInfo = {
  status: ConnectionStatus;
  attempt: number;
  /** When the next reconnect attempt is due (ms epoch), while reconnecting. */
  retryAt?: number;
  reason?: string;
};

export type LiveSocketOptions = {
  url: string;
  onMessages: (batch: LiveEnvelope[]) => void;
  onStatus?: (info: StatusInfo) => void;
  onControl?: (msg: ControlMessage) => void;
  /** Called on every (re)connect after the first, so callers can refetch what they missed. */
  onReconnect?: () => void;
  WebSocketImpl?: typeof WebSocket;
  baseDelayMs?: number;
  maxDelayMs?: number;
  /** No message (data or heartbeat) for this long: assume the connection is dead. */
  staleMs?: number;
  /** Batch window for data messages; 0 delivers each message on its own. */
  flushMs?: number;
  random?: () => number;
};

const DATA_TYPES = new Set(["incidents", "summaries", "anomalies", "traffic.metrics", "nlp.results"]);

/** Full-jitter-ish backoff: the capped exponential delay, scaled into [50%, 100%]. */
export function backoffDelay(attempt: number, base = 500, max = 30_000, random = Math.random): number {
  const capped = Math.min(max, base * 2 ** Math.max(0, attempt));
  return Math.round(capped * (0.5 + random() * 0.5));
}

export class LiveSocket {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private everOpened = false;
  private stopped = true;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private staleTimer: ReturnType<typeof setTimeout> | null = null;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private buffer: LiveEnvelope[] = [];
  private readonly o: Required<Omit<LiveSocketOptions, "onStatus" | "onControl" | "onReconnect">> &
    Pick<LiveSocketOptions, "onStatus" | "onControl" | "onReconnect">;

  constructor(opts: LiveSocketOptions) {
    this.o = {
      WebSocketImpl: globalThis.WebSocket,
      baseDelayMs: 500,
      maxDelayMs: 30_000,
      staleMs: 45_000,
      flushMs: 250,
      random: Math.random,
      ...opts,
    };
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    this.flush();
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = ws.onclose = ws.onerror = ws.onmessage = null;
      ws.close(1000, "client stopped");
    }
    this.emit({ status: "closed", attempt: this.attempt });
  }

  /** Skip the backoff wait (e.g. the browser came back online or the tab became visible). */
  reconnectNow(): void {
    if (this.stopped || this.ws?.readyState === 1 || this.ws?.readyState === 0) return;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.connect();
  }

  private connect(): void {
    this.emit({ status: this.everOpened || this.attempt > 0 ? "reconnecting" : "connecting", attempt: this.attempt });
    let ws: WebSocket;
    try {
      ws = new this.o.WebSocketImpl(this.o.url);
    } catch (err) {
      this.scheduleRetry(String(err));
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      const reconnected = this.everOpened;
      this.everOpened = true;
      this.attempt = 0;
      this.touch();
      this.emit({ status: "open", attempt: 0 });
      if (reconnected) this.o.onReconnect?.();
    };
    ws.onmessage = (ev: MessageEvent) => {
      this.touch();
      this.handle(ev.data);
    };
    ws.onerror = () => {
      /* onclose follows and handles the retry */
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.flush();
      this.scheduleRetry(ev.reason || `closed (${ev.code})`);
    };
  }

  private handle(raw: unknown): void {
    let msg: { type?: string };
    try {
      msg = JSON.parse(String(raw));
    } catch {
      return;
    }
    if (!msg || typeof msg.type !== "string") return;
    if (DATA_TYPES.has(msg.type)) {
      this.buffer.push(msg as LiveEnvelope);
      if (this.o.flushMs <= 0) this.flush();
      else if (!this.flushTimer) this.flushTimer = setTimeout(() => this.flush(), this.o.flushMs);
    } else {
      this.o.onControl?.(msg as ControlMessage);
    }
  }

  private flush(): void {
    if (this.flushTimer) clearTimeout(this.flushTimer);
    this.flushTimer = null;
    if (!this.buffer.length) return;
    const batch = this.buffer;
    this.buffer = [];
    this.o.onMessages(batch);
  }

  private touch(): void {
    if (this.staleTimer) clearTimeout(this.staleTimer);
    this.staleTimer = setTimeout(() => {
      // Silent for too long: drop it and reconnect.
      const ws = this.ws;
      if (!ws) return;
      this.ws = null;
      ws.onopen = ws.onclose = ws.onerror = ws.onmessage = null;
      ws.close(4000, "stale");
      this.scheduleRetry("no messages or heartbeats");
    }, this.o.staleMs);
  }

  private scheduleRetry(reason: string): void {
    if (this.stopped) return;
    if (this.staleTimer) clearTimeout(this.staleTimer);
    const delay = backoffDelay(this.attempt, this.o.baseDelayMs, this.o.maxDelayMs, this.o.random);
    this.attempt += 1;
    this.emit({ status: "reconnecting", attempt: this.attempt, retryAt: Date.now() + delay, reason });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private clearTimers(): void {
    for (const t of [this.retryTimer, this.staleTimer, this.flushTimer]) if (t) clearTimeout(t);
    this.retryTimer = this.staleTimer = this.flushTimer = null;
  }

  private emit(info: StatusInfo): void {
    this.o.onStatus?.(info);
  }
}
