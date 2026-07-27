// src/cell/models.ts — the model library. Browser-safe (see the note in hw.ts).

import { cell } from "aio";
import type { Model } from "../lib/types.ts";

export type ModelsState = {
  /** Roots to search. Seeded from the platform's usual places on first scan. */
  dirs: string[];
  items: Model[];
  scanning: boolean;
  /** Live scan progress — GGUF headers take a moment each. */
  progress: { done: number; total: number; current: string } | null;
  lastScan: number;
  /** Path of the model every other panel plans and runs against. */
  selected: string;
  filter: string;
  lastError: string;
};

export const models = cell("models", {
  // The library itself is re-derived by a scan; the user's directory list and
  // their selection are the parts worth keeping across restarts.
  persist: { include: ["dirs", "selected"] },
  state: {
    dirs: [] as string[],
    items: [] as Model[],
    scanning: false,
    progress: null as ModelsState["progress"],
    lastScan: 0,
    selected: "",
    filter: "",
    lastError: "",
  } as ModelsState,
  methods: {
    addDir(s, dir: string) {
      const d = dir.trim().replace(/\/+$/, "");
      if (d && !s.dirs.includes(d)) s.dirs.push(d);
    },
    removeDir(s, dir: string) {
      s.dirs = s.dirs.filter((d) => d !== dir);
    },
    select(s, path: string) {
      s.selected = path;
    },
    setFilter(s, filter: string) {
      s.filter = filter;
    },

    async scan(s) {
      if (s.scanning) return;
      s.scanning = true;
      s.progress = { done: 0, total: 0, current: "" };
      try {
        const io = await import("./models.server.ts");
        // First run: search the conventional locations so the button does
        // something useful before the user has configured anything.
        // aiol-ok: reading dirs after the import is the point — the user may
        // have added one, and the freshest list is the right one to scan.
        if (s.dirs.length === 0) s.dirs = io.defaultDirs();
        const found = await io.scan(s.dirs.slice(), (done, total, current) => { // aiol-ok
          s.progress = { done, total, current };
        });
        s.items = found;
        s.lastScan = Date.now();
        s.lastError = "";
        // Keep a valid selection: an unselected library makes every other
        // panel show nothing, which reads as a broken app.
        // aiol-ok: keeping the user's selection only makes sense against the
        // selection as it is NOW, which is what this reads.
        if (!found.some((m) => m.path === s.selected)) { // aiol-ok
          s.selected = found[0]?.path ?? "";
        }
      } catch (e) {
        s.lastError = String(e);
      } finally {
        s.scanning = false;
        s.progress = null;
      }
    },
  },
  selectors: {
    current: (s) => s.items.find((m) => m.path === s.selected) ?? null,
    /** Case-insensitive substring match over file name and architecture. */
    visible: (s): Model[] => {
      const q = s.filter.trim().toLowerCase();
      if (!q) return s.items;
      return s.items.filter((m) =>
        m.file.toLowerCase().includes(q) ||
        (m.meta?.arch ?? "").toLowerCase().includes(q) ||
        (m.meta?.quant ?? "").toLowerCase().includes(q)
      );
    },
    totalSizeB: (s) => s.items.reduce((a, m) => a + m.sizeB, 0),
  },
});
