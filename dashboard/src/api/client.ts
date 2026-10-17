import { API_BASE_URL } from "../lib/config";

/** An API failure: `status` 0 means the API could not be reached at all. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
  get unreachable() {
    return this.status === 0;
  }
}

type Query = Record<string, string | number | boolean | null | undefined>;

export function buildUrl(path: string, query?: Query): string {
  const url = new URL(`${API_BASE_URL}${path}`);
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
  }
  return url.toString();
}

const TIMEOUT_MS = 10_000;

export async function api<T>(
  path: string,
  opts: { query?: Query; method?: string; body?: unknown; signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<T> {
  // Combine React Query's cancellation with our own timeout, so a slow API never hangs a view.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException("timeout", "TimeoutError")), opts.timeoutMs ?? TIMEOUT_MS);
  opts.signal?.addEventListener("abort", () => ctrl.abort(opts.signal?.reason), { once: true });
  let res: Response;
  try {
    res = await fetch(buildUrl(path, opts.query), {
      method: opts.method ?? "GET",
      headers: opts.body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: ctrl.signal,
    });
  } catch (err) {
    if (opts.signal?.aborted) throw err;
    const timedOut = ctrl.signal.aborted;
    throw new ApiError(timedOut ? "The API took too long to answer" : "Can't reach the API", 0, timedOut ? "timeout" : "unreachable");
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let code: string | undefined;
    try {
      const body = await res.json();
      message = body?.error?.message ?? message;
      code = body?.error?.code;
    } catch {
      /* not the API's error shape */
    }
    throw new ApiError(message, res.status, code);
  }
  return (await res.json()) as T;
}
