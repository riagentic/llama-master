// src/cell/prereq.ts — what the app needs, what it found, what it can fetch.
// Browser-safe (see the note in hw.ts).

import { cell } from "aio";
import type { Prereq } from "../lib/types.ts";
import type { FixPlan } from "../lib/fixplan.ts";
import { appendLog } from "../lib/buildlog.ts";

export type PrereqState = {
  items: Prereq[];
  scanning: boolean;
  /** Monotonic scan generation — newest scan wins (see `scan`). */
  scanEpoch: number;
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
  // aiol: pre-alpha52 behavior pinned — remove to adopt transactions (s.$commit/s.$live)
  transaction: false,
  // A fix runs a package manager or AMD's own ROCm install script: it downloads,
  // it may wait on `pkexec` for a human to type a password, and it has no
  // ceiling anyone can name. `scan` keeps its 120 s bound — detection that
  // takes two minutes is a wedged tool, not a slow one.
  long: ["fix", "fixAll"],
  // Tool paths and versions change outside the app (a package upgrade), so a
  // persisted list would be confidently wrong; re-detect on every boot.
  persist: "none",
  state: {
    items: [] as Prereq[],
    scanning: false,
    scanEpoch: 0,
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
      // Newest wins, never first-wins — same rule as `builds.scan`: a
      // concurrent `await scan()` must not be a silent no-op, and an older
      // scan must not overwrite a newer one's result.
      const epoch = ++s.scanEpoch;
      s.scanning = true;
      try {
        const io = await import("./prereq.server.ts");
        // One await for the whole picture: nine sequential ones are nine
        // commit points, and the panel rendered half-populated between them.
        const items = await io.detect();
        const plans = await io.plansFor(items.map((i) => i.id));
        if (epoch !== s.scanEpoch) return; // aiol-ok — superseded, discard
        s.items = items;
        s.plans = plans;
        s.lastScan = Date.now();
        s.lastError = "";
      } catch (e) {
        s.lastError = String(e);
      } finally {
        if (epoch === s.scanEpoch) s.scanning = false; // aiol-ok — see above
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
            // aiol-ok — a streaming progress callback: reading the log back to
            // append to it IS the mechanism, and it must see what earlier lines
            // of the same run wrote.
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
        // aiol-ok — a SECOND dispatch on purpose, not an oversight. `fix` reads
        // nothing this method has written (its only guard is `s.fixing`, which
        // it owns), and the separate commit is the point: `s.$call.fix(id)`
        // would fold the whole queue into one action, so the fix log and the
        // progress bar would not move until every item had finished. One
        // dispatch per item is what makes a queue watchable.
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
