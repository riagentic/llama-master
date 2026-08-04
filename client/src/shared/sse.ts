// GENERATED — do not edit. Copied from ../../../src/lib/sse.ts by
// `deno task sync` (client/sync-shared.ts), because aio serves the browser
// bundle only from inside the app's own root. Edit the original.
// src/lib/sse.ts — the two lines of server-sent-events parsing a streaming
// chat response needs, as a pure function over a buffer.
//
// Written as (buffer → events + remainder) rather than a stream transform so a
// test can feed it a chunk boundary in the middle of a token — which is exactly
// where a naive `split("\n\n")` implementation loses data.

export type SseEvent = { data: string };

/** Split off every complete event; return the incomplete tail for next time. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  // Tolerate both LF and CRLF framing.
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const block of parts) {
    const data = block
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trimStart())
      .join("\n");
    if (data) events.push({ data });
  }
  return { events, rest };
}

/** Pull the token out of one OpenAI-style streaming chunk. Returns "" for the
 *  keep-alive and role-only frames, which carry no text. */
export function deltaText(json: string): string {
  try {
    const o = JSON.parse(json) as {
      choices?: { delta?: { content?: string }; text?: string }[];
      content?: string;
    };
    const c = o.choices?.[0];
    return c?.delta?.content ?? c?.text ?? o.content ?? "";
  } catch {
    return "";
  }
}

/**
 * The REASONING token in a streaming chunk, when the model thinks first.
 *
 * llama.cpp parses a reasoning model's `<think>` block out of the text and
 * streams it as `delta.reasoning_content`, with `content` empty the whole
 * while — measured on DeepSeek-V4: the entire first half of every reply
 * arrives here. A client that only reads `content` shows NOTHING while the
 * model thinks and, if generation stops mid-think, appends no message at all;
 * a user watched exactly that and reasonably concluded chat was broken.
 */
export function deltaReasoning(json: string): string {
  try {
    const o = JSON.parse(json) as {
      choices?: { delta?: { reasoning_content?: string } }[];
    };
    return o.choices?.[0]?.delta?.reasoning_content ?? "";
  } catch {
    return "";
  }
}

/** llama.cpp reports timings on the final chunk; surface tokens/second. */
export function timingsTps(json: string): number | null {
  try {
    const o = JSON.parse(json) as {
      timings?: { predicted_per_second?: number };
    };
    const v = o.timings?.predicted_per_second;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

/** The wire budget one streaming reply may spend, per connected client. */
const FLUSH_BUDGET_B_PER_S = 65_536;
/** Never faster than this — 16 Hz already reads as continuous text. */
const FLUSH_MIN_MS = 60;
/** Never slower than this — past ~½ s the reply stops feeling live. */
const FLUSH_MAX_MS = 500;

/**
 * How long to wait before publishing an in-flight reply of `bytes` again.
 *
 * Publishing is a state write, and a state write of a string sends the WHOLE
 * string to every connected client — there is no append patch. So a fixed 60 ms
 * cadence costs `length × 16.7/s`, i.e. quadratic in the reply: a 40 KB answer
 * that takes a minute pushes ~20 MB per client, and a second window doubles it.
 * That is the `PRESSURE — 33 broadcasts/sec` a long chat logged, against a
 * threshold of 30.
 *
 * Holding the RATE of bytes flat instead keeps a long reply as cheap as a short
 * one: flush every `bytes / budget` seconds, clamped so short replies stay
 * smooth and long ones stay live.
 */
export function flushDelayMs(bytes: number): number {
  const b = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  const want = Math.ceil((b * 1000) / FLUSH_BUDGET_B_PER_S);
  return Math.min(FLUSH_MAX_MS, Math.max(FLUSH_MIN_MS, want));
}
