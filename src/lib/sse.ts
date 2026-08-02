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
