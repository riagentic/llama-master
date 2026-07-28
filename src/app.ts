// src/app.ts — boot.
//
// Everything long-running is a declared schedule rather than a timer inside a
// component: the pollers must keep sampling while the user is on another tab,
// and there must be exactly one of each no matter how many windows are open.
//
// The budgets below are raised deliberately. This app's methods do real I/O —
// spawning cmake, reading a 2 MB GGUF header, draining a subprocess pipe — and
// the defaults are tuned for sync CPU-bound effects. The work is awaited off the
// reduce loop, so the window stays responsive; what would otherwise happen is a
// spurious perf violation logged on every poll.

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
  // A cmake build is a legitimately long effect, and the poll cells run once a
  // second, so both budgets are raised from their defaults on purpose —
  // otherwise every tick logs a spurious violation.
  // (dep/aio/docs/debugging/performance.md for perfBudget,
  //  dep/aio/docs/debugging/troubleshooting.md for EFFECT_TIMEOUT.)
  effectTimeoutMs: 30_000,
  perfBudget: { reduce: 100, effect: 1000 },
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
    // 1 s: CPU/GPU utilization is only meaningful as a per-second curve.
    { id: "hw.poll", every: 1000, action: hw.refresh.action() },
    // 1 s: server liveness. `poll` returns immediately when nothing is running.
    { id: "srv.poll", every: 1000, action: srv.poll.action() },
    // 5 s: a llama-server left behind by a crash keeps its VRAM, and that is
    // the usual reason the next Start fails. Cheap /proc scan.
    { id: "srv.orphans", every: 5000, action: srv.scanOrphans.action() },
    // 30 s: free disk space. `df` is a subprocess and space does not move
    // second to second, but a build that runs out part-way wastes minutes.
    { id: "hw.disks", every: 30_000, action: hw.refreshDisks.action() },
    // 5 min: has llama.cpp moved on? Two HTTP calls, and the answer drives the
    // Update button. Cheap enough to run forever, slow enough to be polite.
    {
      id: "builds.update",
      every: 300_000,
      action: builds.checkUpdates.action(),
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
    Promise.all([
      hw.refresh(true),
      prereq.scan(),
      builds.scan(),
      builds.loadAssets(),
    ]).then(() => seedBackend());
  },
});
