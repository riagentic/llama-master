// src/cell/hw.ts — live hardware state.
//
// Browser-safe: imported by App.tsx as well as the server, so it must never
// touch `Deno.*`. Every read happens behind the dynamic import of
// `hw.server.ts` inside `refresh` (dep/aio/docs/build/imports.md).
//
// Polled on a schedule declared in src/app.ts, not from a component effect:
// the sampler must keep running while the UI is on another tab, and it must be
// one timer for every connected client, not one per open window.

import { cell } from "aio";
import type { Cpu, Disk, Gpu, Hw, Mem } from "../lib/types.ts";
import { coresUtilPct, pushHistory, utilPct } from "../lib/procstat.ts";

export type HwState = {
  cpu: Cpu | null;
  mem: Mem | null;
  gpus: Gpu[];
  /** Filesystems this app writes to. Refreshed on its own slow schedule: it
   *  shells out to `df`, and free space does not move second to second. */
  disks: Disk[];
  os: string;
  arch: string;
  /** Previous `/proc/stat` samples — the other half of the utilization delta. */
  prevStat: string;
  prevCoreStats: string[];
  /** Last 60 samples, for the sparklines. */
  cpuHistory: number[];
  gpuHistory: number[];
  vramUsedHistory: number[];
  ramUsedHistory: number[];
  lastRefresh: number;
  refreshing: boolean;
  /** Pauses the scheduled poll; a manual refresh still works. */
  paused: boolean;
  lastError: string;
};

export const hw = cell("hw", {
  // Live telemetry is worthless after a restart, and persisting it would write
  // to SQLite every second for no reader.
  persist: "none",
  state: {
    cpu: null as Cpu | null,
    mem: null as Mem | null,
    gpus: [] as Gpu[],
    disks: [] as Disk[],
    os: "",
    arch: "",
    prevStat: "",
    prevCoreStats: [] as string[],
    cpuHistory: [] as number[],
    gpuHistory: [] as number[],
    /** Bytes held by everything INCLUDING our own server, sampled per poll.
     *  Device-wide, so it is not a measure of anyone else's demand. */
    vramUsedHistory: [] as number[],
    ramUsedHistory: [] as number[],
    lastRefresh: 0,
    refreshing: false,
    paused: false,
    lastError: "",
  } as HwState,
  methods: {
    /** Free space on the filesystems this app writes to. Its own method, on a
     *  30 s schedule (src/app.ts) — `df` is a subprocess, and the 1 s poll has
     *  no business spawning one. */
    async refreshDisks(s) {
      const io = await import("./hw.server.ts");
      const paths = await import("./host.server.ts").then((h) => {
        const p = h.paths();
        return [p.home, p.builds, p.cache];
      });
      const models = await import("./models.server.ts").then((m) =>
        m.defaultDirs()
      );
      try {
        const found = await io.disks([...paths, ...models]);
        // Compare before writing: this runs on a schedule and an equal value
        // would re-render the page for nothing. Reading `s.disks` after the
        // await is deliberate — the comparison is against whatever is current
        // now, not against a snapshot taken before the `df`. // aiol-ok
        const same = found.length === s.disks.length &&
          found.every((d, i) =>
            // aiol-ok
            d.mount === s.disks[i]?.mount && d.availB === s.disks[i]?.availB
          );
        if (!same) s.disks = found;
      } catch (e) {
        s.lastError = `disk usage: ${e}`;
      }
    },
    async refresh(s, force?: boolean) {
      // Re-entrancy guard, still needed even though the 1 s schedule now carries
      // `skipIfRunning`: that de-duplicates the SCHEDULE, and this method is also
      // called directly — at boot, by Resume, by the Refresh button, and by
      // tests. Those can still land on top of an in-flight tick.
      if (s.refreshing) return;
      if (s.paused && !force) return;
      s.refreshing = true;
      try {
        const io = await import("./hw.server.ts");
        const snap = await io.snapshot();

        if (snap.cpu) {
          // aiol-ok: deliberate post-await read. /proc/stat counters are
          // cumulative, so a delta against whatever the previous sample is now
          // stays correct even if another poll landed while we were suspended.
          // Cumulative counters: comparing against whatever the previous sample
          // was stays correct even if another refresh landed mid-await.
          snap.cpu.utilPct = utilPct(s.prevStat, snap.cpu.stat); // aiol-ok
          snap.cpu.coresUtil = coresUtilPct(
            s.prevCoreStats, // aiol-ok — see the note above
            snap.cpu.coreStats,
          );
          s.prevStat = snap.cpu.stat;
          s.prevCoreStats = snap.cpu.coreStats;
          s.cpuHistory = pushHistory(s.cpuHistory.slice(), snap.cpu.utilPct);
        }
        const gpuUtil = snap.gpus.length
          ? Math.max(...snap.gpus.map((g) => g.utilPct))
          : 0;
        s.gpuHistory = pushHistory(s.gpuHistory.slice(), gpuUtil);

        // Device-wide memory use, sampled over the last minute. Note "device
        // wide": our own llama-server is inside these numbers, which is why they
        // must never drive a fit decision (see `src/lib/adapt.ts`). They exist so
        // the UI can show what the machine has been doing.
        const vramUsed = snap.gpus.reduce((a, g) => a + g.vramUsedB, 0);
        const ramUsed = snap.mem ? snap.mem.totalB - snap.mem.availableB : 0;
        // Appending to whatever the series is NOW is the intent, exactly as for
        // the two sparkline histories above: if another poll landed while this
        // one was suspended, its sample belongs in the series too. // aiol-ok
        s.vramUsedHistory = pushHistory(s.vramUsedHistory.slice(), vramUsed);
        // aiol-ok — same reasoning as the line above
        s.ramUsedHistory = pushHistory(s.ramUsedHistory.slice(), ramUsed);

        s.cpu = snap.cpu;
        s.mem = snap.mem;
        s.gpus = snap.gpus;
        s.os = io.PLATFORM;
        s.arch = io.ARCH;
        s.lastRefresh = Date.now();
        s.lastError = "";
      } catch (e) {
        // Surfaced in the header, never swallowed — a telemetry read that
        // starts failing silently is how a dashboard becomes a liar.
        s.lastError = String(e);
      } finally {
        s.refreshing = false;
      }
    },
    togglePause(s) {
      s.paused = !s.paused;
      if (!s.paused) hw.refresh(true);
    },
  },
  selectors: {
    /** The planner's view of this machine. */
    snapshot: (s): Hw => ({
      cpu: s.cpu,
      mem: s.mem,
      gpus: s.gpus,
      os: s.os,
      arch: s.arch,
    }),
    vramTotalB: (s) => s.gpus.reduce((a, g) => a + g.vramTotalB, 0),
    vramUsedB: (s) => s.gpus.reduce((a, g) => a + g.vramUsedB, 0),
    hasGpu: (s) => s.gpus.length > 0,
  },
});
