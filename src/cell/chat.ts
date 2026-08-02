// src/cell/chat.ts — the built-in test chat.
//
// Browser-safe and server-only-free: it talks to llama-server over plain
// `fetch`, which needs no Deno API, so there is no `.server.ts` half here. The
// method runs on the server (like every cell method), which is also where the
// llama-server process is, so the request never leaves the machine.
//
// Tokens stream in and are written to state as they arrive — that is what makes
// the answer appear word by word in every connected window at once.

import { cell } from "aio";
import type { MethodDraftMeta } from "aio";
import { deltaReasoning, deltaText, parseSse, timingsTps } from "../lib/sse.ts";
import type { ChatMessage } from "../lib/types.ts";

export type ChatState = {
  messages: ChatMessage[];
  input: string;
  system: string;
  streaming: boolean;
  /** Text of the in-flight assistant reply, before it is committed. */
  partial: string;
  /** The in-flight reasoning, for models that think before answering. A
   *  reasoning model's whole first act arrives on this channel with `content`
   *  empty — leaving it out rendered "thinking" as a spinner over nothing. */
  partialThink: string;
  /** Tokens/second reported by llama-server for the last completion. */
  lastTps: number;
  lastError: string;
};

/** Flush the partial reply to state at most this often. One dispatch per token
 *  would put a network round trip per token on every connected client. */
const FLUSH_MS = 60;

export const chat = cell("chat", {
  // A conversation is worth keeping across restarts; the in-flight fields are
  // not, and restoring `streaming: true` would show a spinner forever.
  persist: { include: ["messages", "system"] },
  state: {
    messages: [] as ChatMessage[],
    input: "",
    system: "",
    streaming: false,
    partial: "",
    partialThink: "",
    lastTps: 0,
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
    },
    /** Cancels a running `send` via `cancelOn`. */
    stop(s) {
      s.streaming = false;
    },

    /**
     * Send the current input to `${url}/v1/chat/completions` and stream back.
     *
     * `url` is passed in rather than read from the server cell: this cell has
     * no business knowing how the server was configured, and passing it keeps
     * the method callable in a test against any endpoint.
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
        let acc = "";
        let think = "";
        let tps = 0;
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
            acc += deltaText(e.data);
            // A reasoning model's whole first act arrives on this channel
            // with `content` empty (measured on DeepSeek-V4). Reading only
            // `content` showed a spinner over nothing for the entire think,
            // and a reply that stopped mid-think appended no message at all.
            think += deltaReasoning(e.data);
            tps = timingsTps(e.data) ?? tps;
          }
          const now = Date.now();
          if (now - lastFlush >= FLUSH_MS) {
            lastFlush = now;
            s.partial = acc;
            s.partialThink = think;
          }
        }

        // Anything arrived — answer, reasoning, or both — is a message. A
        // reply that ran out of tokens mid-think must say so on screen, not
        // vanish.
        if (acc || think) {
          s.messages.push({
            role: "assistant",
            content: acc,
            ...(think ? { thinking: think } : {}),
            tps,
          });
        }
        s.lastTps = tps;
      } catch (e) {
        if (s.$signal?.aborted) {
          // A user-cancelled stream is not an error; keep whatever arrived.
          if (s.partial || s.partialThink) {
            s.messages.push({
              role: "assistant",
              content: s.partial,
              ...(s.partialThink ? { thinking: s.partialThink } : {}),
            });
          }
        } else {
          s.lastError = String(e);
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
