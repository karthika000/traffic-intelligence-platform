import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeWebSocket } from "../test/fixtures";
import { backoffDelay, LiveSocket, type StatusInfo } from "./liveSocket";

describe("backoffDelay", () => {
  it("doubles per attempt, is capped, and jitters into [50%, 100%]", () => {
    expect(backoffDelay(0, 500, 30_000, () => 1)).toBe(500);
    expect(backoffDelay(3, 500, 30_000, () => 1)).toBe(4000);
    expect(backoffDelay(20, 500, 30_000, () => 1)).toBe(30_000);
    expect(backoffDelay(3, 500, 30_000, () => 0)).toBe(2000);
  });
});

describe("LiveSocket", () => {
  let statuses: StatusInfo[];
  let batches: unknown[][];
  let reconnects: number;
  let sock: LiveSocket;
  const Impl = FakeWebSocket as unknown as typeof WebSocket;

  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    statuses = [];
    batches = [];
    reconnects = 0;
    sock = new LiveSocket({
      url: "ws://api/ws/live",
      WebSocketImpl: Impl,
      onMessages: (b) => batches.push(b),
      onStatus: (s) => statuses.push(s),
      onReconnect: () => reconnects++,
      baseDelayMs: 1000,
      maxDelayMs: 8000,
      staleMs: 45_000,
      flushMs: 250,
      random: () => 1,
    });
  });
  afterEach(() => {
    sock.stop();
    vi.useRealTimers();
  });

  it("batches data messages and passes control messages separately", () => {
    const control: unknown[] = [];
    sock = new LiveSocket({ url: "ws://x", WebSocketImpl: Impl, onMessages: (b) => batches.push(b), onControl: (m) => control.push(m), flushMs: 250 });
    sock.start();
    FakeWebSocket.last.open();
    FakeWebSocket.last.message({ type: "hello", client_id: "abc" });
    FakeWebSocket.last.message({ type: "incidents", camera_ids: [], ts: "", data: {} });
    FakeWebSocket.last.message({ type: "traffic.metrics", camera_ids: [], ts: "", data: {} });
    FakeWebSocket.last.message("not json");
    expect(batches).toHaveLength(0);
    vi.advanceTimersByTime(250);
    expect(batches).toHaveLength(1);
    expect(batches[0]).toHaveLength(2);
    expect(control).toEqual([{ type: "hello", client_id: "abc" }]);
  });

  it("reconnects with growing backoff and resets it after a successful open", () => {
    sock.start();
    expect(statuses.at(-1)?.status).toBe("connecting");
    FakeWebSocket.last.serverClose();
    expect(statuses.at(-1)).toMatchObject({ status: "reconnecting", attempt: 1 });
    vi.advanceTimersByTime(999);
    expect(FakeWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.last.serverClose(); // second failure waits 2 s
    vi.advanceTimersByTime(1999);
    expect(FakeWebSocket.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances).toHaveLength(3);

    FakeWebSocket.last.open();
    expect(statuses.at(-1)).toMatchObject({ status: "open", attempt: 0 });
    FakeWebSocket.last.serverClose(); // back to the base delay
    vi.advanceTimersByTime(1000);
    expect(FakeWebSocket.instances).toHaveLength(4);
  });

  it("caps the backoff", () => {
    sock.start();
    for (let n = 0; n < 6; n++) {
      FakeWebSocket.last.serverClose();
      vi.advanceTimersByTime(8000);
    }
    const before = FakeWebSocket.instances.length;
    FakeWebSocket.last.serverClose();
    vi.advanceTimersByTime(8000);
    expect(FakeWebSocket.instances.length).toBe(before + 1);
  });

  it("calls onReconnect on every open after the first", () => {
    sock.start();
    FakeWebSocket.last.open();
    expect(reconnects).toBe(0);
    FakeWebSocket.last.serverClose();
    vi.advanceTimersByTime(1000);
    FakeWebSocket.last.open();
    expect(reconnects).toBe(1);
  });

  it("treats a silent connection as dead", () => {
    sock.start();
    const first = FakeWebSocket.last;
    first.open();
    vi.advanceTimersByTime(44_000);
    first.message({ type: "heartbeat", ts: 1 }); // heartbeats keep it alive
    vi.advanceTimersByTime(44_000);
    expect(first.closed).toBeNull();
    vi.advanceTimersByTime(1000);
    expect(first.closed?.code).toBe(4000);
    expect(statuses.at(-1)?.status).toBe("reconnecting");
    vi.advanceTimersByTime(1000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("stops cleanly and does not reconnect", () => {
    sock.start();
    FakeWebSocket.last.open();
    sock.stop();
    expect(FakeWebSocket.last.closed?.code).toBe(1000);
    expect(statuses.at(-1)?.status).toBe("closed");
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("reconnectNow skips the remaining backoff", () => {
    sock.start();
    FakeWebSocket.last.serverClose();
    expect(FakeWebSocket.instances).toHaveLength(1);
    sock.reconnectNow();
    expect(FakeWebSocket.instances).toHaveLength(2);
  });
});
