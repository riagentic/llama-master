// client/src/ui/sticky.ts — keep the newest line in view.
//
// The wiring; the RULE is `../shared/scroll.ts`, which is the server app's
// `src/lib/scroll.ts` through the `shared` symlink — one policy for both apps.
// An arrival forces the scroll; a streamed token only follows a reader who is
// already at the bottom, so scrolling up to re-read something is not undone by
// the next token.
//
// Wired rather than imported whole: the server app's hook lives in its `src/ui`,
// which the client's dev server does not serve. The twelve lines below are the
// wiring to AIR's render cycle, and the part worth sharing is shared.

import { afterRender, useRef } from "aio/air";
import { stickToBottom } from "../shared/scroll.ts";

export function useStickyBottom(
  revision: number,
): { current: HTMLElement | null } {
  const box = useRef<HTMLElement | null>(null);
  // -1 so the first commit counts as an arrival: a box mounted with a
  // conversation already in it opens at the newest message, not the oldest.
  const seen = useRef(-1);
  afterRender(() => {
    const arrived = revision !== seen.current;
    seen.current = revision;
    stickToBottom(box.current, arrived);
  });
  return box;
}
