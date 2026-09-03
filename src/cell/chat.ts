// src/cell/chat.ts — the built-in test chat.
//
// Browser-safe and server-only-free: it talks to llama-server over plain
// `fetch`, which needs no Deno API, so there is no `.server.ts` half here. The
// method runs on the server (like every cell method), which is also where the
// llama-server process is, so the request never leaves the machine.
//
// Tokens stream in and are written to state as they arrive — that is what makes
// the answer appear word by word in every connected window at once.
//
// What is NOT here is the text being typed. The draft lived in this cell once
// (`input`, written by a `setInput` dispatch on every keystroke), and that made
// the box a controlled input over a replicated server field: each key was a
// round trip, and while a reply streamed the 60–500 ms `partial` flushes
// re-rendered the box with whatever `input` the server had last acknowledged —
// so keys still in flight were wiped, and only while the model was answering.
// A draft is per-window state no other client and no restart needs
// (dep/aio/docs/state/real-time.md), so it is a browser-local signal in
// `ChatComposer` and arrives here as an ARGUMENT to `submit`/`send`.

import { cell } from "aio";
import type { MethodDraftMeta } from "aio";
import {
  deltaReasoning,
  deltaText,
  flushDelayMs,
  parseSse,
  timingsTps,
} from "../lib/sse.ts";
import {
  advances,
  queueAdd,
  queueRemove,
  type SendOutcome,
} from "../lib/queue.ts";
import type { ChatMessage } from "../lib/types.ts";

/** Sampling passed through to llama-server. Named because `send` and `submit`
 *  must take the same shape — the drain hands one straight to the other. */
export type SendOpts = { temp?: number; topP?: number; maxTokens?: number };

export type ChatState = {
  messages: ChatMessage[];
  system: string;
  /** Messages written while a reply was streaming, oldest first. Drained one
   *  per completed reply — see `advances` for why only a CLEAN one. */
  queue: string[];
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

export const chat = cell("chat", {
  // aiol: pre-alpha52 behavior pinned — remove to adopt transactions (s.$commit/s.$live)
  transaction: false,
  // A conversation is worth keeping across restarts; the in-flight fields are
  // not, and restoring `streaming: true` would show a spinner forever. The
  // queue IS kept: it is text the user typed and has not got an answer to, and
  // losing it on a restart would be the one thing this feature exists to
  // prevent. It restores as a queue that is holding — nothing streams at boot,
  // so nothing drains it until the user says so.
  persist: { include: ["messages", "system", "queue"] },
  state: {
    messages: [] as ChatMessage[],
    system: "",
    queue: [] as string[],
    streaming: false,
    partial: "",
    partialThink: "",
    lastTps: 0,
    lastError: "",
  } as ChatState,
  cancelOn: { send: ["chat:stop"], submit: ["chat:stop"] },
  // Both entry points run the drain, and a drain is a conversation: it can hold
  // a stream open for as long as the model takes, and then do it again for
  // every message the user queued behind it. Declared HERE rather than as a
  // `perfBudget.methods["chat:send"]` string in app.ts — that string named one
  // of the two, and a method list checked at cell() time cannot silently
  // orphan itself the way the string can (aiol says so, and it was right: it
  // is what caught `submit` inheriting the 30-second default).
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
    },
    /**
     * Cancels a running `send` via `cancelOn`.
     *
     * The queue is deliberately left standing. Stop means "not this reply", and
     * the drain reads the outcome (`advances`), so a cancelled run never pulls
     * the next message — but the messages themselves are the user's, and
     * throwing them away would make Stop a destructive button.
     */
    stop(s) {
      s.streaming = false;
    },

    /**
     * The composer's one gesture: hold this message, or send it.
     *
     * Both surfaces dispatch THIS rather than deciding for themselves, and it
     * is a single dispatch rather than an enqueue followed by a send — the
     * decision is made against the cell's own state on the server, so a browser
     * client cannot read `streaming` a moment stale and send a second request
     * into a live stream.
     *
     * Returns whether `text` was taken. `false` means the queue was full — the
     * one refusal — and the composer puts the text back rather than losing it;
     * the note above the input says why. A blank `text` with messages waiting
     * is the "go on then" gesture after a Stop, and is taken.
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
     * Send `text` (or, with none, what is waiting) to
     * `${url}/v1/chat/completions` and stream it back — then the next thing,
     * and the next, until the queue is empty.
     *
     * `url` is passed in rather than read from the server cell: this cell has
     * no business knowing how the server was configured, and passing it keeps
     * the method callable in a test against any endpoint.
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
 * `cancelOn` fires on Stop) cancels the reply AND ends the drain, and the
 * in-flight guard below cannot race with itself between turns.
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
  // next request going out, and a submit arriving in that gap would start a
  // second drain: two live streams appending to one conversation.
  if (s.streaming) return;
  if (s.queue.length === 0 && !text.trim()) return;
  s.streaming = true;
  try {
    for (;;) {
      const outcome = await turn(s, url, text, opts);
      // The direct text is one message; every later turn is the queue's.
      text = "";
      // `advances` is the whole policy: only a reply that ended on its own
      // pulls the next message. Stop and errors leave the queue standing.
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
  // fallback, which keeps every direct caller (and every test written before
  // the queue) working unchanged.
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

    // Outside the try: what arrived so far is the only thing worth keeping
    // when the stream is cut, and a cut can come from the user (Stop) or
    // from shutdown (the app aborts every in-flight method before it
    // persists). `s.partial` is only as fresh as the last flush — up to half
    // a second stale — so the reply that gets written down is this one.
    let acc = "";
    let think = "";
    let tps = 0;
    // How this ended decides whether the queue moves on, so it is recorded
    // rather than inferred afterwards: by the time the drain runs, `streaming`
    // is false and `lastError` may have been written by something else.
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
          acc += deltaText(e.data);
          // A reasoning model's whole first act arrives on this channel
          // with `content` empty (measured on DeepSeek-V4). Reading only
          // `content` showed a spinner over nothing for the entire think,
          // and a reply that stopped mid-think appended no message at all.
          think += deltaReasoning(e.data);
          tps = timingsTps(e.data) ?? tps;
        }
        // One dispatch per token would put a network round trip per token on
        // every connected client; a fixed cadence would instead re-send the
        // whole (growing) reply 16 times a second. `flushDelayMs` holds the
        // BYTE rate flat, so a long answer costs no more per second than a
        // short one — and the two writes below are one commit, one push.
        const now = Date.now();
        if (now - lastFlush >= flushDelayMs(acc.length + think.length)) {
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
      // The read loop `break`s on an abort that lands between two reads
      // rather than throwing, so reaching here is not proof of a clean end —
      // and a Stop that arrived in that window must not pull the next
      // message. The signal is the authority either way.
      outcome = s.$signal?.aborted ? "cancelled" : "done";
    } catch (e) {
      if (s.$signal?.aborted) {
        outcome = "cancelled";
        // A cancelled stream is not an error — neither the user's Stop nor
        // the app closing under it. Keep whatever arrived, and keep it from
        // `acc`/`think`, which are always ahead of the last flush.
        if (acc || think) {
          s.messages.push({
            role: "assistant",
            content: acc,
            ...(think ? { thinking: think } : {}),
            ...(tps ? { tps } : {}),
          });
        }
      } else {
        // Never a raw error: a dead socket is the server going away, and what
        // to do about it is part of the message. `fetch failed` is Deno 2.9's
        // wording, `error sending request` the older one.
        const msg = String(e);
        s.lastError =
          /Failed to fetch|fetch failed|error sending request|connection/i
              .test(msg)
            ? `The server stopped answering (${msg}). It may have crashed under the request — the server log says why.`
            : msg;
      }
    } finally {
      s.partial = "";
      s.partialThink = "";
    }
    return outcome;
  }
}
