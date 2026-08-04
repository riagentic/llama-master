// client/src/cell/chat.ts — the conversation, streamed from somebody else's GPU.
//
// The same shape as the server app's chat cell, and deliberately so: it parses
// the same SSE with the same functions (`../../src/lib/sse.ts` — one
// implementation of "what did this token event contain", not two that drift —
// `src/shared` is a symlink to the server app's `src/lib`, so the dev server
// can serve it to the browser like any other file under the app root),
// and it keeps the same hard-won behaviours. What differs is where the reply
// comes from: over a LAN, from a server this app does not own.
//
// Browser-safe — plain `fetch`, no Deno API — so there is no `.server.ts` half.
// The method runs on the client's own host, which is the machine holding the
// window, and the request goes out from there.

import { cell } from "aio";
import type { MethodDraftMeta } from "aio";
import {
  deltaReasoning,
  deltaText,
  flushDelayMs,
  parseSse,
  timingsTps,
} from "../shared/sse.ts";
import type { ChatMessage } from "../shared/types.ts";

export type ChatState = {
  messages: ChatMessage[];
  input: string;
  system: string;
  streaming: boolean;
  /** The in-flight reply, before it is committed. */
  partial: string;
  /** The in-flight reasoning: a thinking model's whole first act arrives on
   *  this channel with `content` empty, and leaving it out renders "thinking"
   *  as a spinner over nothing. */
  partialThink: string;
  /** Tokens/second the server reported for the last completion. */
  lastTps: number;
  /** Seconds from Send to the first token of the last reply — the number a
   *  person actually feels, and one no server-side metric reports. */
  lastLatencyMs: number;
  lastError: string;
};

export const chat = cell("chat", {
  // A conversation survives a restart; nothing in flight does.
  persist: { include: ["messages", "system"] },
  // `lastLatencyMs` trips aio's "looks like a secret" heuristic on its name
  // alone; it is a measurement of the last reply and belongs on screen, so it
  // is declared public rather than hidden (dep/aio/docs/state/cells.md).
  ui: { publicFields: ["lastLatencyMs"] },
  state: {
    messages: [] as ChatMessage[],
    input: "",
    system: "",
    streaming: false,
    partial: "",
    partialThink: "",
    lastTps: 0,
    lastLatencyMs: 0,
    lastError: "",
  } as ChatState,
  cancelOn: { send: ["chat:stop"] },
  methods: {
    setInput(s, input: string) {
      s.input = input;
    },
    setSystem(s, system: string) {
      s.system = system;
    },
    clear(s) {
      s.messages = [];
      s.partial = "";
      s.partialThink = "";
      s.lastError = "";
      s.lastTps = 0;
      s.lastLatencyMs = 0;
    },
    /** Cancels a running `send` via `cancelOn`. */
    stop(s) {
      s.streaming = false;
    },

    /**
     * Send the current input to `${url}/v1/chat/completions` and stream back.
     *
     * `url` is passed in rather than read from `conn`: this cell has no
     * business knowing how the connection was made, and passing it keeps the
     * method callable in a test against any endpoint.
     */
    async send(
      s: ChatState & Partial<MethodDraftMeta>,
      url: string,
      opts: { temp?: number; topP?: number; maxTokens?: number } = {},
    ) {
      const text = s.input.trim();
      if (!text || s.streaming) return;

      const history: ChatMessage[] = [
        ...(s.system.trim()
          ? [{ role: "system" as const, content: s.system.trim() }]
          : []),
        ...s.messages.map((m) => ({ role: m.role, content: m.content })),
        { role: "user" as const, content: text },
      ];

      s.messages.push({ role: "user", content: text });
      s.input = "";
      s.streaming = true;
      s.partial = "";
      s.partialThink = "";
      s.lastError = "";
      s.lastLatencyMs = 0;

      // Outside the try: what arrived so far is the only thing worth keeping
      // when the stream is cut, and a cut can come from the user (Stop), from
      // shutdown, or — on a LAN — from the network. `s.partial` is only as
      // fresh as the last flush, so what gets written down is this.
      let acc = "";
      let think = "";
      let tps = 0;
      const startedAt = Date.now();
      let firstTokenAt = 0;

      try {
        const res = await fetch(`${url}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: history,
            stream: true,
            temperature: opts.temp,
            top_p: opts.topP,
            max_tokens: opts.maxTokens ?? -1,
          }),
          signal: s.$signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(
            `${res.status} ${res.statusText}: ${
              (await res.text()).slice(0, 200)
            }`,
          );
        }

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        let lastFlush = 0;

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          if (s.$signal?.aborted) break;
          buf += dec.decode(value, { stream: true });
          const { events, rest } = parseSse(buf);
          buf = rest;
          for (const e of events) {
            if (e.data === "[DONE]") continue;
            const t = deltaText(e.data);
            const r = deltaReasoning(e.data);
            if (!firstTokenAt && (t || r)) firstTokenAt = Date.now();
            acc += t;
            think += r;
            tps = timingsTps(e.data) ?? tps;
          }
          // The byte rate, not a fixed cadence: publishing a partial reply is a
          // full re-send of the string, so a fixed 60 ms is quadratic in the
          // length of the answer (`sse.ts:flushDelayMs`).
          const now = Date.now();
          if (now - lastFlush >= flushDelayMs(acc.length + think.length)) {
            lastFlush = now;
            s.partial = acc;
            s.partialThink = think;
          }
        }

        if (acc || think) {
          s.messages.push({
            role: "assistant",
            content: acc,
            ...(think ? { thinking: think } : {}),
            tps,
          });
        }
        s.lastTps = tps;
        s.lastLatencyMs = firstTokenAt ? firstTokenAt - startedAt : 0;
      } catch (e) {
        if (s.$signal?.aborted) {
          if (acc || think) {
            s.messages.push({
              role: "assistant",
              content: acc,
              ...(think ? { thinking: think } : {}),
              ...(tps ? { tps } : {}),
            });
          }
        } else {
          // Never a raw error: what the user can do about it is part of it.
          const msg = String(e);
          s.lastError = /Failed to fetch|error sending request|connection/i
              .test(msg)
            ? `The server stopped answering mid-reply (${msg}). It may have been stopped, or the network dropped — press Discover or Connect to check.`
            : msg;
        }
      } finally {
        s.partial = "";
        s.partialThink = "";
        s.streaming = false;
      }
    },
  },
  selectors: {
    turns: (s) => s.messages.length,
    canSend: (s) => s.input.trim().length > 0 && !s.streaming,
  },
});
