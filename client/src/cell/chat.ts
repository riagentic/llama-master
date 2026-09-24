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
import {
  advances,
  queueAdd,
  queueRemove,
  type SendOutcome,
} from "../shared/queue.ts";
import type { ChatMessage } from "../shared/types.ts";

/** Sampling passed through to the far end. Named because `send` and `submit`
 *  must take the same shape — the drain hands one straight to the other. */
export type SendOpts = { temp?: number; topP?: number; maxTokens?: number };

export type ChatState = {
  messages: ChatMessage[];
  system: string;
  /** Messages written while a reply was streaming, oldest first. Drained one
   *  per completed reply — see `advances` for why only a CLEAN one. */
  queue: string[];
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
  // aiol: pre-alpha52 behavior pinned — remove to adopt transactions (s.$commit/s.$live)
  transaction: false,
  // A conversation survives a restart; nothing in flight does. The queue does
  // too: it is text the user typed and has not got an answer to, and losing it
  // is the one thing this feature exists to prevent. It comes back HOLDING —
  // nothing streams at boot, so nothing drains it until the user says go.
  persist: { include: ["messages", "system", "queue"] },
  // `lastLatencyMs` trips aio's "looks like a secret" heuristic on its name
  // alone; it is a measurement of the last reply and belongs on screen, so it
  // is declared public rather than hidden (dep/aio/docs/state/cells.md).
  visible: { publicFields: ["lastLatencyMs"] },
  state: {
    messages: [] as ChatMessage[],
    system: "",
    queue: [] as string[],
    streaming: false,
    partial: "",
    partialThink: "",
    lastTps: 0,
    lastLatencyMs: 0,
    lastError: "",
  } as ChatState,
  cancelOn: { send: ["chat:stop"], submit: ["chat:stop"] },
  // A reply is as long as the model needs; that is the product — and BOTH
  // entry points run the drain, which then does it again for every message
  // queued behind it. Declared here rather than as a
  // `perfBudget.methods["chat:send"]` string in app.ts: that string named one
  // of the two, and a method list checked at cell() time cannot silently
  // orphan itself the way the string can (aiol says so, and it was right —
  // it is what caught `submit` inheriting the 30-second default).
  long: ["send", "submit"],
  methods: {
    setSystem(s, system: string) {
      s.system = system;
    },
    clear(s) {
      s.messages = [];
      s.queue = [];
      s.partial = "";
      s.partialThink = "";
      s.lastError = "";
      s.lastTps = 0;
      s.lastLatencyMs = 0;
    },
    /**
     * Cancels a running `send` via `cancelOn`.
     *
     * The queue is deliberately left standing. Stop means "not this reply", and
     * the drain reads the outcome (`advances`), so a cancelled run never pulls
     * the next message — but the messages are the user's, and discarding them
     * would make Stop a destructive button.
     */
    stop(s) {
      s.streaming = false;
    },

    /** The ✕ on the error note — which shows this cell's error as well as
     *  `conn`'s, so dismissing it must clear both. */
    clearError(s) {
      s.lastError = "";
    },

    /**
     * The composer's one gesture: hold this message, or send it.
     *
     * One dispatch rather than an enqueue followed by a send — the decision is
     * made against the cell's own state, so nothing can read `streaming` a
     * moment stale and put a second request into a live stream.
     *
     * The text arrives as an ARGUMENT: the draft is browser-local (see
     * `Composer`), never a cell field — a keystroke is not shared state.
     * Returns whether `text` was taken; `false` is the one refusal, a full
     * queue, and the composer puts the text back rather than losing it.
     */
    async submit(
      s: ChatState & Partial<MethodDraftMeta>,
      url: string,
      text: string,
      opts: SendOpts = {},
    ): Promise<boolean> {
      const before = s.queue.length;
      s.queue = queueAdd(s.queue, text);
      if (s.queue.length === before && text.trim()) return false;
      // Idle → this becomes the request. Busy → `drain` returns at once and the
      // message stays where it was just put.
      await drain(s, url, "", opts);
      return true;
    },

    /** Drop one waiting message — the ✕ on a queue chip. */
    unqueue(s, i: number) {
      s.queue = queueRemove(s.queue, i);
    },

    /** Drop every waiting message, leaving the conversation alone. */
    clearQueue(s) {
      s.queue = [];
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
      text = "",
      opts: SendOpts = {},
    ) {
      await drain(s, url, text, opts);
    },
  },
  selectors: {
    turns: (s) => s.messages.length,
    waiting: (s) => s.queue.length,
  },
});

/**
 * Every turn the queue holds, one after another, until something says stop.
 *
 * A plain function rather than a cell method calling a cell method: that call
 * would run as its own transaction against committed state, which is the trap
 * `aiol` lints for. It is also why the drain is a LOOP rather than a scheduled
 * re-dispatch — one invocation owns the whole run, so `s.$signal` (which
 * `cancelOn` fires on Stop) cancels the reply AND ends the drain.
 */
async function drain(
  s: ChatState & Partial<MethodDraftMeta>,
  url: string,
  text: string,
  opts: SendOpts,
): Promise<void> {
  // One drain at a time, and `streaming` is the flag that says so — which is
  // why it is owned HERE and stays true between turns. Published per turn
  // instead, it would read false in the gap between one reply landing and the
  // next going out, and a submit arriving in that gap would start a second
  // drain: two live streams appending to one conversation.
  if (s.streaming) return;
  if (s.queue.length === 0 && !text.trim()) return;
  s.streaming = true;
  try {
    for (;;) {
      const outcome = await turn(s, url, text, opts);
      // The direct text is one message; every later turn is the queue's.
      text = "";
      // `advances` is the whole policy: only a reply that ended on its own
      // pulls the next message. Stop and errors leave the queue standing —
      // and on a LAN, an error is usually the far end going away, where
      // throwing the rest of the queue after it would lose all of it.
      if (!advances(outcome) || s.queue.length === 0) break;
    }
  } finally {
    s.streaming = false;
  }
}

/** One question and one answer. Returns how it ended, which is the only thing
 *  the drain needs to know. */
async function turn(
  s: ChatState & Partial<MethodDraftMeta>,
  url: string,
  direct: string,
  opts: SendOpts,
): Promise<SendOutcome> {
  // The queue is the front of the line whenever it has anything in it, so a
  // turn means the same thing however it was reached — a keystroke, or the
  // drain coming back for the next one. The text handed to `send` is the
  // fallback, which keeps every direct caller (and every test older than the
  // queue) working.
  const queued = s.queue.length > 0;
  const text = (queued ? s.queue[0] ?? "" : direct).trim();
  if (!text) return "error";

  {
    const history: ChatMessage[] = [
      ...(s.system.trim()
        ? [{ role: "system" as const, content: s.system.trim() }]
        : []),
      ...s.messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user" as const, content: text },
    ];

    s.messages.push({ role: "user", content: text });
    if (queued) s.queue.shift();
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
    // How this ended decides whether the queue moves on, so it is recorded
    // rather than inferred: by the time the drain reads it, `streaming` has
    // been republished and `lastError` may belong to something else.
    let outcome: SendOutcome = "error";

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
      // The read loop `break`s on an abort that lands between two reads
      // rather than throwing, so reaching here is not proof of a clean end.
      // The signal is the authority either way.
      outcome = s.$signal?.aborted ? "cancelled" : "done";
    } catch (e) {
      // What arrived is kept however the stream was cut — Stop, shutdown, or
      // the far end going away. On a LAN the last is the common one, and
      // throwing away half an answer because the network dropped under it
      // makes the user ask again for text they had already been shown.
      if (acc || think) {
        s.messages.push({
          role: "assistant",
          content: acc,
          ...(think ? { thinking: think } : {}),
          ...(tps ? { tps } : {}),
        });
      }
      if (s.$signal?.aborted) {
        outcome = "cancelled";
      } else {
        // Never a raw error: what the user can do about it is part of it.
        const msg = String(e);
        // `fetch failed` is Deno 2.9's wording for the same thing; the older
        // `error sending request` is kept for the runtimes that still say it.
        s.lastError =
          /Failed to fetch|fetch failed|error sending request|connection/i
              .test(msg)
            ? `The server stopped answering mid-reply (${msg}). It may have been stopped, or the network dropped — press Discover or Connect to check.`
            : msg;
      }
    } finally {
      s.partial = "";
      s.partialThink = "";
    }
    return outcome;
  }
}
