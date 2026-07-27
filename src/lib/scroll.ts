// src/lib/scroll.ts — keeping the newest line in view.
//
// Every log and every chat in this app appends at the bottom and lives in a
// capped, scrollable box. Without this, a reply that overflows the box stays
// below the fold until the user scrolls — which is exactly the case where they
// were waiting for it.
//
// The rule, and why it is not simply "always jump to the bottom": a user who has
// deliberately scrolled up to read something must not be yanked away by the next
// streamed token. So a token only sticks when the view is ALREADY at the bottom,
// while a whole new message (or a forced write, like a log line) always sticks.
// That is the behaviour every chat client has, and it satisfies "the last
// message is visible when it arrives" without fighting the reader.
//
// Pure: it takes anything with the three scroll numbers, so it is unit-testable
// without a DOM and works for a `div` or a `pre` alike.

/** The three numbers any scrollable box exposes. */
export type Scrollable = {
  scrollTop: number;
  readonly scrollHeight: number;
  readonly clientHeight: number;
};

/** How far from the bottom still counts as "at the bottom", in CSS pixels.
 *  A couple of lines: enough to survive rounding and a partly-visible line. */
export const BOTTOM_SLACK_PX = 48;

export function isNearBottom(
  el: Scrollable,
  slack: number = BOTTOM_SLACK_PX,
): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slack;
}

/**
 * Scroll to the bottom, and report whether it happened.
 *
 * `force` is for arrivals that must always be seen — a new message, a new log
 * line. Without it the box only follows when the reader was already at the
 * bottom.
 */
export function stickToBottom(
  el: Scrollable | null | undefined,
  force = false,
  slack: number = BOTTOM_SLACK_PX,
): boolean {
  if (!el) return false;
  if (!force && !isNearBottom(el, slack)) return false;
  el.scrollTop = el.scrollHeight;
  return true;
}
