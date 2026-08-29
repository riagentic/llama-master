// GENERATED — do not edit. Copied from ../../../src/lib/queue.ts by
// `deno task sync` (client/sync-shared.ts), because aio serves the browser
// bundle only from inside the app's own root. Edit the original.
// src/lib/queue.ts — the rules for messages written while the model is busy.
//
// A local model answers at reading speed, and the thought you have WHILE it
// answers is the one worth keeping: the follow-up, the correction, the "and in
// TypeScript". Before this, the input was disabled for the whole reply, so that
// thought had to be held in the user's head until the tokens stopped.
//
// The queue is that holding place, and the rules it needs are decisions rather
// than mechanics — which is why they are here, pure and tested, rather than in
// the cell that stores the array or the component that draws it.
//
// The load-bearing rule is `advances`: the queue moves on only after a reply
// that ENDED ON ITS OWN. A user who pressed Stop, or a server that fell over,
// has said something about the next message too — firing it a millisecond later
// would be the app arguing with them.

/** How many messages may wait. Not a technical limit — the array is persisted
 *  and a paste-into-a-loop should not grow it without end, and a queue longer
 *  than this stopped being a queue and became a script. */
export const QUEUE_MAX = 20;

/** How a `send` ended, which is the only input `advances` needs. */
export type SendOutcome =
  /** The stream closed on its own. */
  | "done"
  /** The user pressed Stop, or the app closed under the reply. */
  | "cancelled"
  /** The server refused, died, or was never there. */
  | "error";

/**
 * May the queue move on after a reply that ended this way?
 *
 * Only a clean ending. Both other outcomes are the run telling us something
 * about the NEXT message as well: Stop usually means "not like that", and an
 * error means the next send would fail the same way — six queued messages
 * against a dead server is six identical errors and a lost queue.
 *
 * The queue is kept in both cases, so nothing the user typed is thrown away;
 * it simply waits for them to say go.
 */
export function advances(outcome: SendOutcome): boolean {
  return outcome === "done";
}

/**
 * Add a message to the back of the queue.
 *
 * Returns the queue unchanged when the text is blank or the queue is full —
 * both are refusals the caller can SEE (the length did not change), which is
 * what lets the UI say why instead of appearing to lose a message.
 */
export function queueAdd(queue: readonly string[], text: string): string[] {
  const t = text.trim();
  if (!t || queue.length >= QUEUE_MAX) return queue.slice();
  return [...queue, t];
}

/** Drop one waiting message. An out-of-range index changes nothing. */
export function queueRemove(queue: readonly string[], i: number): string[] {
  if (i < 0 || i >= queue.length) return queue.slice();
  return queue.filter((_, n) => n !== i);
}

/**
 * What the composer does with a submit, given what is on screen.
 *
 * One decision with three answers, so the two chat surfaces cannot disagree
 * about what pressing Enter means:
 *
 * - `"queue"` — hold it; something is already streaming, so the send that
 *   would carry it is not free. It goes to the back and the drain picks it up.
 * - `"send"` — nothing is running, so this becomes the request itself.
 * - `"full"` — the queue is at `QUEUE_MAX` and the message would be dropped.
 *   Named rather than silently ignored: a message that vanishes on Enter is
 *   the worst outcome available here.
 * - `null` — nothing to do (empty box, empty queue).
 */
export function submitKind(
  input: string,
  queueLength: number,
  streaming: boolean,
): "queue" | "send" | "full" | null {
  const has = input.trim().length > 0;
  if (has && queueLength >= QUEUE_MAX) return "full";
  if (has) return streaming ? "queue" : "send";
  // An empty box with messages waiting is the "go on then" gesture — what the
  // user is left with after a Stop, when the queue is holding and the thing
  // that would have drained it is not coming.
  if (!streaming && queueLength > 0) return "send";
  return null;
}

/**
 * The sentence above the input, or "" when there is nothing to say.
 *
 * It has to distinguish two states that look identical (a queue of two, with
 * and without a reply in flight) because only one of them is going to empty
 * itself. A queue that is waiting on the user, described as though it were
 * waiting on the model, is a user watching a spinner that will never stop.
 */
export function queueNote(queueLength: number, streaming: boolean): string {
  if (queueLength === 0) return "";
  const n = `${queueLength} message${queueLength === 1 ? "" : "s"}`;
  if (streaming) {
    return queueLength >= QUEUE_MAX
      ? `${n} waiting — the queue is full. They send as replies complete.`
      : `${n} waiting — sent as soon as this reply finishes.`;
  }
  return `${n} waiting — nothing is running, so press Send to go on.`;
}

/** A queued message shortened for a chip, with the whole of it kept for the
 *  tooltip. Cut on a word where there is one nearby, because a hard cut mid-word
 *  reads as corruption rather than as elision. */
export function queueLabel(text: string, max = 42): string {
  const one = text.replace(/\s+/g, " ").trim();
  if (one.length <= max) return one;
  const cut = one.slice(0, max);
  const space = cut.lastIndexOf(" ");
  return `${space > max * 0.6 ? cut.slice(0, space) : cut}…`;
}
