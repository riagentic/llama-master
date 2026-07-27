// src/ui/sticky.ts — the "keep the newest line in view" hook.
//
// One implementation, used by every append-at-the-bottom box in the app: the
// chat log, the all-in-one mini chat, and the build/server log views. Three
// copies of this would drift, and a chat surface that forgot it would leave the
// reply the user is waiting for below the fold.
//
// The policy itself is pure and lives in src/lib/scroll.ts; this is only the
// wiring to AIR's render cycle.

import { afterRender, useRef } from "aio/air";
import { stickToBottom } from "../lib/scroll.ts";

/**
 * Follow the bottom of a scrollable box.
 *
 * Returns the ref to put on the box. `revision` is any number that changes when
 * NEW content arrives (a message count, a line count): when it changes the
 * scroll is forced, because an arrival must be seen. Re-renders that do not
 * change it — a streamed token extending the last message — only follow when
 * the reader is already at the bottom, so scrolling up to re-read something is
 * not undone by the next token.
 *
 * `afterRender` runs once the render has committed to the DOM, on the initial
 * mount and on every signal-driven re-render alike
 * (dep/aio/docs/ui/air-signals.md). `onMount` fires once and would miss every
 * streamed delta.
 */
export function useStickyBottom(
  revision: number,
): { current: HTMLElement | null } {
  const box = useRef<HTMLElement | null>(null);
  // -1 so the first commit always counts as an arrival: a box mounted with
  // history already in it must open at the newest line, not the oldest.
  const seen = useRef(-1);
  afterRender(() => {
    const arrived = revision !== seen.current;
    seen.current = revision;
    stickToBottom(box.current, arrived);
  });
  return box;
}
