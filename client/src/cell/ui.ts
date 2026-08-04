// client/src/cell/ui.ts — how the client looks, as opposed to what it says.
//
// Two settings, both the reader's rather than the app's: which theme, and how
// big the text is. They persist, because a person who has chosen light at 18px
// has chosen it — asking again every launch is the app disagreeing with them.
//
// Separate from `conn` on purpose: nothing here is a fact about a server, and a
// cell that mixes "what is out there" with "how I like to look at it" makes
// both harder to reason about.

import { cell } from "aio";

export type Theme = "dark" | "light";

/** The legible range. Below 12 the mono blocks stop being readable at a
 *  glance; above 20 the connect bar wraps to three rows on a small window. */
export const MIN_FONT_PX = 12;
export const MAX_FONT_PX = 20;
export const DEFAULT_FONT_PX = 14;

export const ui = cell("ui", {
  persist: "all",
  state: {
    theme: "dark" as Theme,
    /** Root text size in px. Every size in the stylesheet is a ratio of this,
     *  so one control resizes the whole app — which is the honest answer to
     *  "big enough to read comfortably", a property of the reader and the
     *  screen rather than of the design. */
    fontPx: DEFAULT_FONT_PX,
  },
  methods: {
    toggleTheme(s) {
      s.theme = s.theme === "dark" ? "light" : "dark";
    },
    setTheme(s, theme: Theme) {
      s.theme = theme === "light" ? "light" : "dark";
    },
    /** Step the root size, clamped. Coerces, because a method reachable from
     *  outside the UI cannot trust its argument — and a NaN here would size the
     *  whole app to nothing. */
    zoom(s, delta: number) {
      const d = typeof delta === "number" && Number.isFinite(delta) ? delta : 0;
      s.fontPx = Math.max(
        MIN_FONT_PX,
        Math.min(MAX_FONT_PX, Math.round(s.fontPx + d)),
      );
    },
  },
  selectors: {
    /** Room to grow / shrink, so a button that would do nothing is disabled
     *  rather than dead under the finger. */
    canGrow: (s) => s.fontPx < MAX_FONT_PX,
    canShrink: (s) => s.fontPx > MIN_FONT_PX,
  },
});
