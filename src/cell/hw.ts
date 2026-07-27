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
import type { Cpu, Gpu, Hw, Mem } from "../lib/types.ts";
import { coresUtilPct, pushHistory, utilPct } from "../lib/procstat.ts";

export type HwState = {
  cpu: Cpu | null;
  mem: Mem | null;
  gpus: Gpu[];
  os: string;
  arch: string;
  /** Previous `/proc/stat` samples — the other half of the utilization delta. */
  prevStat: string;
  prevCoreStats: string[];
  /** Last 60 samples, for the sparklines. */
  cpuHistory: number[];
  gpuHistory: number[];
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
    os: "",
    arch: "",
    prevStat: "",
    prevCoreStats: [] as string[],
    cpuHistory: [] as number[],
    gpuHistory: [] as number[],
    lastRefresh: 0,
    refreshing: false,
    paused: false,
    lastError: "",
  } as HwState,
  methods: {
    async refresh(s, force?: boolean) {
      if (s.refreshing) return; // re-entrancy guard: polls can overlap on load
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
