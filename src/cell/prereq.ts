// src/cell/prereq.ts — what the app needs, what it found, what it can fetch.
// Browser-safe (see the note in hw.ts).

import { cell } from "aio";
import type { Prereq } from "../lib/types.ts";
import type { FixPlan } from "../lib/fixplan.ts";
import { appendLog } from "../lib/buildlog.ts";

export type PrereqState = {
  items: Prereq[];
  scanning: boolean;
  lastScan: number;
  /** Non-null while CMake is downloading — drives the progress bar. */
  install: { label: string; received: number; total: number | null } | null;
  /** What fixing each prerequisite would do — computed once per scan so the
   *  buttons can show the exact command before anything runs. */
  plans: Record<string, FixPlan>;
  /** Id of the prerequisite currently being installed, "" when idle. */
  fixing: string;
  /** Ids still queued behind it during "Fix all". */
  fixQueue: string[];
  fixLog: string[];
  lastError: string;
};

export const prereq = cell("prereq", {
  // Tool paths and versions change outside the app (a package upgrade), so a
  // persisted list would be confidently wrong; re-detect on every boot.
  persist: "none",
  state: {
    items: [] as Prereq[],
    scanning: false,
    lastScan: 0,
    install: null as PrereqState["install"],
    plans: {} as Record<string, FixPlan>,
    fixing: "",
    fixQueue: [] as string[],
    fixLog: [] as string[],
    lastError: "",
  } as PrereqState,
  methods: {
    async scan(s) {
      if (s.scanning) return;
      s.scanning = true;
      try {
        const io = await import("./prereq.server.ts");
        // One await for the whole picture: nine sequential ones are nine
        // commit points, and the panel rendered half-populated between them.
        const items = await io.detect();
        const plans = await io.plansFor(items.map((i) => i.id));
        s.items = items;
        s.plans = plans;
        s.lastScan = Date.now();
        s.lastError = "";
      } catch (e) {
        s.lastError = String(e);
      } finally {
        s.scanning = false;
      }
    },
    /** Install one missing prerequisite. Streams the installer's own output —
     *  a privileged command that shows nothing is not something to trust — and
     *  drives the progress bar for the parts this app downloads itself. */
    async fix(s, id: string) {
      if (s.fixing) return;
      s.fixing = id;
      s.fixLog = [`Fixing ${id}…`];
      s.install = null;
      s.lastError = "";
      try {
        const io = await import("./prereq.server.ts");
        const result = await io.fix(
          id,
          (line) => {
            s.fixLog = appendLog(s.fixLog.slice(), [line], 200);
          },
          (received, total, note) => {
            s.install = { label: note, received, total };
          },
        );
        s.fixLog = appendLog(s.fixLog.slice(), [result.message], 200);
        if (!result.ok) s.lastError = `${id}: ${result.message}`;
        // Re-detect either way: a partial install still changes the answer.
        const items = await io.detect();
        const plans = await io.plansFor(items.map((i) => i.id));
        s.items = items;
        s.plans = plans;
      } catch (e) {
        s.lastError = `${id}: ${e}`;
      } finally {
        s.fixing = "";
        s.install = null;
      }
    },

    /** Fix everything that can be fixed, one at a time so the log stays
     *  readable and a failure stops the queue instead of hiding in it. */
    async fixAll(s) {
      if (s.fixing || s.fixQueue.length > 0) return;
      const queue = s.items
        .filter((i) => !i.found && s.plans[i.id]?.kind !== "manual")
        .map((i) => i.id);
      if (queue.length === 0) return;
      s.fixQueue = queue;
      for (const id of queue) {
        s.fixQueue = s.fixQueue.filter((q) => q !== id);
        await prereq.fix(id);
        // aiol-ok: `fix` writes lastError, and reading it back is how this
        // queue knows to stop — the post-await read IS the mechanism.
        if (s.lastError) break; // aiol-ok — stop on the first real failure
      }
      s.fixQueue = [];
    },

    clearFixLog(s) {
      s.fixLog = [];
      s.lastError = "";
    },
  },
  selectors: {
    byId: (s, id: string) => s.items.find((i) => i.id === id) ?? null,
    /** Can we compile from source right now? */
    canBuild: (s) => {
      const ok = (id: string) =>
        s.items.find((i) => i.id === id)?.found === true;
      return ok("cmake") && ok("compiler");
    },
    /** Tools that are missing AND that the app cannot obtain itself. */
    blocking: (s) =>
      s.items.filter((i) => !i.found && i.systemOnly && i.id === "compiler"),
    /** Missing prerequisites the app can actually do something about. */
    fixable: (s) =>
      s.items.filter((i) => !i.found && s.plans[i.id]?.kind !== "manual"),
  },
});
