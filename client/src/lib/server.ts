// client/src/lib/server.ts — reading what the server says about itself.
//
// Everything the client shows about the far end comes from llama-server's own
// endpoints, parsed here so the cell does I/O and nothing else:
//
//   /props    what model is loaded, at what context, with how many slots
//   /health   is it up and past loading
//   /metrics  Prometheus counters — only when started with `--metrics`
//   /slots    per-slot occupancy — only when started with `--slots`
//
// Two of those four are optional, which is the whole reason this file is
// careful: a client that says "0% busy" because `/metrics` was disabled is
// lying, and a client that shows nothing because one endpoint is missing is
// useless. Every reading is therefore three-valued — a number, or "not
// reported", never a fabricated zero.

/** What `/props` tells us that a person would want to read. */
export type ServerInfo = {
  /** The model file the server actually loaded, basename only for display. */
  model: string;
  /** Full path, for the tooltip — it is the ground truth about what is running. */
  modelPath: string;
  /** Context the server ALLOCATED, which is not always the one requested. */
  ctx: number;
  /** Parallel slots: how many conversations it can hold at once. */
  slots: number;
  /** Does it carry a chat template? Without one, /v1/chat/completions is a
   *  guess and the plain /completion endpoint is the honest route. */
  chatTemplate: boolean;
};

const asRecord = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v)
    ? v as Record<string, unknown>
    : null;

const num = (v: unknown): number => {
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : 0;
};

/** The file name a person recognises, out of a path they did not choose. */
export function modelName(path: string): string {
  const base = path.split(/[/\\]/).pop() ?? "";
  return base || path;
}

/**
 * Parse `/props`.
 *
 * The context lives in one of two places depending on the build — top-level
 * `n_ctx` on older servers, inside `default_generation_settings` on current
 * ones — and reading only one of them showed "—" against a server that had
 * plainly told us. Both are read, newest first.
 */
export function parseProps(raw: unknown): ServerInfo | null {
  const p = asRecord(raw);
  if (!p) return null;
  const gen = asRecord(p.default_generation_settings);
  const modelPath = String(p.model_path ?? p.model ?? "");
  const ctx = num(gen?.n_ctx) || num(p.n_ctx);
  const slots = num(p.total_slots) || num(gen?.n_parallel) || 1;
  return {
    model: modelName(modelPath),
    modelPath,
    ctx,
    slots,
    chatTemplate: Boolean(p.chat_template),
  };
}

/** How busy the far end is. Every field is optional because every source of it
 *  is optional; `null` means "the server does not report this", which is a
 *  different thing from zero and is displayed differently. */
export type Occupancy = {
  /** Requests being generated right now. */
  processing: number | null;
  /** Requests queued behind them — the reason a reply might not start at once. */
  queued: number | null;
  /** 0..1 of the KV cache in use: how full the server's memory of conversations is. */
  kvUsed: number | null;
  /** Tokens/second the SERVER has been achieving, across everyone. */
  tps: number | null;
  /** Where these numbers came from, so the UI can say. */
  source: "metrics" | "slots" | "none";
};

export const NO_OCCUPANCY: Occupancy = {
  processing: null,
  queued: null,
  kvUsed: null,
  tps: null,
  source: "none",
};

/**
 * Parse llama-server's Prometheus text.
 *
 * Deliberately a line reader rather than a Prometheus client: four metrics are
 * wanted, the format is `name value` after a `#`-prefixed preamble, and a
 * dependency to read four numbers would be the kind of thing this project
 * writes tests to avoid.
 */
export function parseMetrics(text: string): Occupancy {
  if (!text.trim()) return NO_OCCUPANCY;
  const seen = new Map<string, number>();
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s || s.startsWith("#")) continue;
    // `llamacpp:requests_processing 1` — a name, whitespace, a number. Labels
    // are not used by llama.cpp's exporter, so anything with a `{` is skipped
    // rather than half-parsed.
    const m = /^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([-\d.eE+]+)$/.exec(s);
    if (!m) continue;
    const v = Number(m[2]);
    if (Number.isFinite(v)) seen.set(m[1] as string, v);
  }
  if (seen.size === 0) return NO_OCCUPANCY;
  const get = (k: string): number | null => seen.has(k) ? seen.get(k)! : null;
  return {
    processing: get("llamacpp:requests_processing"),
    queued: get("llamacpp:requests_deferred"),
    kvUsed: get("llamacpp:kv_cache_usage_ratio"),
    tps: get("llamacpp:predicted_tokens_seconds"),
    source: "metrics",
  };
}

/**
 * Parse `/slots`.
 *
 * The fallback when `--metrics` is off but `--slots` is on: each entry carries
 * an `is_processing` flag, which is enough to say how many of the server's
 * conversations are busy — the one occupancy number that matters to someone
 * deciding whether to press Send.
 */
export function parseSlots(raw: unknown): Occupancy {
  if (!Array.isArray(raw)) return NO_OCCUPANCY;
  const busy = raw.filter((s) => asRecord(s)?.is_processing === true).length;
  return {
    processing: busy,
    queued: null,
    kvUsed: null,
    tps: null,
    source: "slots",
  };
}

/** Slots busy out of slots available, as a fraction — the bar the UI draws.
 *  Null when the server does not report occupancy at all. */
export function busyFraction(o: Occupancy, slots: number): number | null {
  if (o.processing === null) return null;
  const total = Math.max(1, slots);
  return Math.min(1, o.processing / total);
}

/** A short, honest sentence about how busy the far end is. */
export function busyLabel(o: Occupancy, slots: number): string {
  if (o.processing === null) {
    return "not reported";
  }
  const total = Math.max(1, slots);
  const q = o.queued && o.queued > 0 ? `, ${o.queued} waiting` : "";
  if (o.processing === 0) return `idle · ${total} slot${total > 1 ? "s" : ""}`;
  return `${o.processing} of ${total} busy${q}`;
}

/** The length of a reply this estimate is quoted for. A number had to be
 *  chosen; 256 tokens is a paragraph or two, which is what a chat answer
 *  usually is, and it is stated in the UI rather than hidden here. */
export const TYPICAL_REPLY_TOKENS = 256;

/**
 * Seconds a typical reply should take, at the rate we have.
 *
 * `measured` is a generation this client has actually seen; the server's own
 * average is used only when there is none, and the caller says which it is.
 * Waiting requests are added at the same rate, because a queue is the honest
 * difference between "3 tok/s" and "3 tok/s once four people ahead of you are
 * done".
 */
export function replySeconds(
  tps: number,
  queued = 0,
  tokens = TYPICAL_REPLY_TOKENS,
): number {
  if (!Number.isFinite(tps) || tps <= 0) return 0;
  const one = tokens / tps;
  return one * (1 + Math.max(0, queued));
}

/** `12 s` / `1 m 30 s` / `—`. Seconds, in the words a person uses for them. */
export function seconds(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return "—";
  if (s < 1) return "<1 s";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  const rest = Math.round(s - m * 60);
  return rest ? `${m} m ${rest} s` : `${m} m`;
}
