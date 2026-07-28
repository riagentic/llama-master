// src/cell/ui.ts — which panel is open, and how it looks.
//
// A shared (not per-tab) cell on purpose: this is a single-user desktop app, so
// a second window showing the same panel is the expected behaviour, and the
// choice surviving a restart is what a desktop app does.

import { cell } from "aio";

export type Tab =
  | "one"
  | "dashboard"
  | "prereq"
  | "cpu"
  | "gpu"
  | "memory"
  | "storage"
  | "build"
  | "models"
  | "settings"
  | "server"
  | "chat"
  | "about";

export const TABS: readonly { id: Tab; label: string; icon: string }[] = [
  // First and default: most sessions never need to leave it.
  { id: "one", label: "All-in-one", icon: "◎" },
  { id: "dashboard", label: "Machine", icon: "▦" },
  // Prerequisites is a task, not a reading: a list with an action on most rows.
  // The Machine page summarises it in one line and links here.
  { id: "prereq", label: "Prerequisites", icon: "✓" },
  // The "tell me about one pool" pages. Machine summarises all of them; these
  // answer each on its own terms — including storage, which is the third thing
  // this app fills and the one that fails a build minutes in.
  { id: "cpu", label: "CPU", icon: "⚙" },
  { id: "gpu", label: "GPU", icon: "◈" },
  { id: "memory", label: "Memory", icon: "▤" },
  { id: "storage", label: "Storage", icon: "▣" },
  { id: "build", label: "Build", icon: "⚒" },
  { id: "models", label: "Models", icon: "◈" },
  { id: "settings", label: "Tune", icon: "⚙" },
  { id: "server", label: "Server", icon: "⏻" },
  { id: "chat", label: "Chat", icon: "✉" },
  { id: "about", label: "About", icon: "ⓘ" },
];

export const ui = cell("ui", {
  state: {
    tab: "one" as Tab,
    theme: "dark" as "dark" | "light",
    /** Root text size in px. Every size in the stylesheet is a ratio of this,
     *  so one control resizes the whole app — the honest answer to "big enough
     *  to read comfortably", which depends on the eyes and the screen. */
    fontPx: 14,
    /** Collapses the command preview strip at the bottom of the window. */
    showCommand: true,
    /** Is the full flag catalog unfolded on the all-in-one page? Closed by
     *  default, and the controls are not rendered at all while it is — 49 of
     *  them on the page most sessions never leave is a cost for nothing. */
    showAllSettings: false,
  },
  methods: {
    go(s, tab: Tab) {
      s.tab = tab;
    },
    toggleTheme(s) {
      s.theme = s.theme === "dark" ? "light" : "dark";
    },
    /** Step the root size. Clamped to a range that stays legible at the bottom
     *  and does not break the layout at the top. */
    zoom(s, delta: number) {
      s.fontPx = Math.max(12, Math.min(20, s.fontPx + delta));
    },
    toggleCommand(s) {
      s.showCommand = !s.showCommand;
    },
    toggleAllSettings(s) {
      s.showAllSettings = !s.showAllSettings;
    },
  },
});
