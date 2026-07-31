// src/app.ts — boot.
//
// Everything long-running is a declared schedule rather than a timer inside a
// component: the pollers must keep sampling while the user is on another tab,
// and there must be exactly one of each no matter how many windows are open.
//
// The perf budgets below are per-method, not global. This app's slow methods are
// slow by nature — spawning cmake, reading a 2 MB GGUF header, draining a
// subprocess pipe — and raising the budget for all of them to accommodate the
// worst would blind every tight reducer at once, which is what this app used to
// do. Each one that legitimately takes minutes says so by name.

import { aio } from "aio";
import { ABOUT } from "./lib/about.ts";
import { builds } from "./cell/builds.ts";
import { cfg } from "./cell/cfg.ts";
import { chat } from "./cell/chat.ts";
import { hw } from "./cell/hw.ts";
import { models } from "./cell/models.ts";
import { prereq } from "./cell/prereq.ts";
import { srv } from "./cell/srv.ts";
import { ui } from "./cell/ui.ts";
// A cross-cell gesture, so it lives with the others in `ui/actions.ts` rather
// than being reimplemented here; boot is just another caller.
import { seedBackend } from "./ui/actions.ts";

await aio.run({
  appId: "llama-master",
  appVersion: ABOUT.version,
  cells: [ui, hw, prereq, builds, models, cfg, srv, chat],
  // Boot fails loudly if a cell was defined but not listed above — a cell that
  // is imported and unregistered dispatches into the void.
  strictCells: true,
  // Everything stays strict EXCEPT the handful of methods whose job is genuinely
  // slow. This app used to raise `effect` to 1000 and `effectTimeoutMs` to 30 s
  // globally, because a cmake build and a 1 s poll shared one budget — which
  // silenced one poller by blinding every tight reducer at once. Per-method
  // budgets mean a four-minute compile no longer costs the rest of the app its
  // signal. (dep/aio/docs/debugging/performance.md)
  perfBudget: {
    reduce: 100,
    methods: {
      // Spawns cmake/make and streams their output for minutes — and on the
      // release route downloads and extracts hundreds of MB instead.
      "builds:start": { effect: 600_000, timeout: 3_600_000 },
      // Rebuild/reinstall in place when llama.cpp moves on, then restart.
      "builds:update": { effect: 600_000, timeout: 3_600_000 },
      // Two HTTP calls to GitHub, plus the HTML fallback when the API quota is
      // gone.
      "builds:loadAssets": { effect: 30_000, timeout: 60_000 },
      "builds:loadRefs": { effect: 30_000, timeout: 60_000 },
      "builds:checkUpdates": { effect: 30_000, timeout: 60_000 },
      // Reads a 2 MB GGUF header per model across every search path.
      "models:scan": { effect: 30_000, timeout: 120_000 },
      // Shells out to nvidia-smi and reads /proc every second.
      "hw:refresh": { effect: 2_000, timeout: 10_000 },
      // Spawns `df`.
      "hw:refreshDisks": { effect: 5_000, timeout: 30_000 },
      // Detects toolchains, and may run an installer script.
      "prereq:scan": { effect: 30_000, timeout: 120_000 },
      "prereq:fix": { effect: 600_000, timeout: 1_800_000 },
      "prereq:fixAll": { effect: 600_000, timeout: 1_800_000 },
      // Owns a child process: spawn, health-poll, drain its pipes, SIGTERM.
      "srv:start": { effect: 120_000, timeout: 600_000 },
      "srv:stop": { effect: 30_000, timeout: 60_000 },
      "srv:poll": { effect: 2_000, timeout: 10_000 },
      "srv:scanOrphans": { effect: 2_000, timeout: 10_000 },
      "srv:freeMemory": { effect: 30_000, timeout: 60_000 },
      // Streams a completion for as long as the model takes.
      "chat:send": { effect: 600_000, timeout: 1_800_000 },
    },
  },
  ui: {
    title: "llama.master",
    width: 1440,
    height: 940,
    showStatus: false,
    // The browser client's tab icon: the same mark as src/icon.svg, inline
    // because nothing serves the .svg itself. The Electron and Android
    // packagers read src/icon.png, which is generated from that same file.
    head:
      `<link rel="icon" href="data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="%230c0f14"/><rect x="20" y="20" width="24" height="24" rx="4" transform="rotate(45 32 32)" fill="%23f0a92e"/></svg>">`,
  },
  schedules: [
    // `skipIfRunning` on all of them: each shells out, and a tick landing while
    // the previous is still draining a subprocess is pure pile-up. The scheduler
    // owns the guard — hand-rolled it needs a state flag reset in a `finally`,
    // and a throw in between leaves the flag stuck and the poll dead until
    // restart.
    //
    // 1 s: CPU/GPU utilization is only meaningful as a per-second curve.
    {
      id: "hw.poll",
      every: 1000,
      action: hw.refresh.action(),
      skipIfRunning: true,
    },
    // 1 s: server liveness. `poll` returns immediately when nothing is running.
    {
      id: "srv.poll",
      every: 1000,
      action: srv.poll.action(),
      skipIfRunning: true,
    },
    // 5 s: a llama-server left behind by a crash keeps its VRAM, and that is
    // the usual reason the next Start fails. Cheap /proc scan.
    {
      id: "srv.orphans",
      every: 5000,
      action: srv.scanOrphans.action(),
      skipIfRunning: true,
    },
    // 30 s: free disk space. `df` is a subprocess and space does not move
    // second to second, but a build that runs out part-way wastes minutes.
    {
      id: "hw.disks",
      every: 30_000,
      action: hw.refreshDisks.action(),
      skipIfRunning: true,
    },
    // 5 min: has llama.cpp moved on? Two HTTP calls, and the answer drives the
    // Update button. Cheap enough to run forever, slow enough to be polite.
    {
      id: "builds.update",
      every: 300_000,
      action: builds.checkUpdates.action(),
      skipIfRunning: true,
    },
  ],
  onStart: () => {
    // Fill the window before the first scheduled tick, and find out what this
    // machine already has — the app should be useful on the first frame. These
    // are deliberately not awaited in sequence: each panel renders as its own
    // scan lands, rather than the window staying empty until the slowest one is
    // done.
    models.scan();
    builds.checkUpdates();
    hw.refreshDisks();
    // Before anything else: is memory still held by a previous run?
    srv.scanOrphans();

    // The release route is the default, and it cannot answer "will this
    // produce a build?" without the asset list — `targetReadiness` reports
    // `pending` until it lands, which leaves Install disabled. `assets` is
    // deliberately not persisted (a stale list is worse than none), so the
    // only way it is there on the first frame is to fetch it here.
    //
    // These four are the ones the backend seed reads, so it waits for them.
    // "Build with one click" and "build what is optimal for this PC" have to be
    // the same click: the stored default backend is `cpu` — the only value that
    // is always installable — so on a machine with a GPU it has to be corrected
    // once the hardware is known. `seedBackend` does nothing if the user has
    // ever chosen a backend, and nothing once a build is installed.
    // `allSettled`, not `all`: a cell call rejects when its caller-side
    // timeout fires (a stalled `loadAssets` socket is enough), and one
    // rejection must not keep the seed from running — the failed scan
    // already surfaced its own error in cell state.
    Promise.allSettled([
      hw.refresh(true),
      prereq.scan(),
      builds.scan(),
      builds.loadAssets(),
    ]).then(() => seedBackend());
  },
});
