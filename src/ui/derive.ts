// src/ui/derive.ts — the values the UI derives from cell state.
//
// WHY THIS FILE EXISTS: one list of everything the UI derives from cell state.
//
// It began as a workaround. Calling a cell SELECTOR (`models.current()`) used to
// return a correct, fresh value while registering NO reactive dependency, so a
// component whose only read was a selector rendered once and then never updated
// — the data right, the screen stale, and nothing warning you. It cost an
// afternoon (the all-in-one page kept showing "no model" while the dropdown
// beside it showed the model). Reported upstream, and FIXED in aio
// 1.0.0-alpha38: selector calls are reactive in a render now, verified here.
//
// The file stays, as a convention rather than a workaround. Every value the UI
// derives is a plain function over cell PROPERTIES, in one place, so "where does
// this number come from" has one answer — and one convention beats two that both
// work. `tests/guards.test.ts` still enforces it, and says the same thing.

import { builds } from "../cell/builds.ts";
import { cfg } from "../cell/cfg.ts";
import { chat } from "../cell/chat.ts";
import { hw } from "../cell/hw.ts";
import { models } from "../cell/models.ts";
import { prereq } from "../cell/prereq.ts";
import { srv } from "../cell/srv.ts";
import { enabledGpus } from "../lib/gpu.ts";
import { drift, headroomKey } from "../lib/adapt.ts";
import { optimalCtx, pinnedCtx, trainedCtx } from "../lib/tune.ts";
import { loadProgress } from "../lib/loadprogress.ts";
import type { LoadProgress } from "../lib/loadprogress.ts";
import {
  bytesPerToken,
  calibrate,
  estimateTps,
  speedIsMeasured,
} from "../lib/speed.ts";
import type { Drift } from "../lib/adapt.ts";
import { NO_MODEL, plan as computePlan, withoutOurUsage } from "../lib/plan.ts";
import type { Plan } from "../lib/plan.ts";
import { str } from "../lib/params.ts";
import { updateFor } from "../lib/update.ts";
import type { UpdateCheck } from "../lib/update.ts";
import type { FixPlan } from "../lib/fixplan.ts";
import type { Build, Hw, Model, Prereq, Settings } from "../lib/types.ts";

// ── models ─────────────────────────────────────────────────────────────────

export function currentModel(): Model | null {
  return models.items.find((m) => m.path === models.selected) ?? null;
}

export function visibleModels(): Model[] {
  const q = models.filter.trim().toLowerCase();
  if (!q) return models.items;
  return models.items.filter((m) =>
    m.file.toLowerCase().includes(q) ||
    (m.meta?.arch ?? "").toLowerCase().includes(q) ||
    (m.meta?.quant ?? "").toLowerCase().includes(q)
  );
}

export function modelsSizeB(): number {
  return models.items.reduce((a, m) => a + m.sizeB, 0);
}

// ── builds ─────────────────────────────────────────────────────────────────

export function activeBuild(): Build | null {
  return builds.installed.find((b) => b.id === builds.activeId) ?? null;
}

export function buildBusy(): boolean {
  return builds.job?.status === "running";
}

export function buildsSizeB(): number {
  return builds.installed.reduce((a, b) => a + b.sizeB, 0);
}

export function updateInfo(): UpdateCheck {
  return updateFor(activeBuild(), builds.upstream);
}

// ── hardware ───────────────────────────────────────────────────────────────

/**
 * The machine as the ACTIVE BUILD can see it, minus the GPUs switched off.
 *
 * Two filters, for the same reason: a plan drawn against devices that will not
 * be used is a picture of something that will not happen. A CPU build cannot put
 * a byte on a GPU and a CUDA build cannot use the AMD iGPU (`usableGpus`); and a
 * card the user unticked is not going to hold any of the model either
 * (`enabledGpus`, which reads the same `-dev` value that goes on the command
 * line). `plan`, `tune` and `stability` are all pure over this snapshot, so
 * filtering here is the whole fix.
 */
export function hwSnapshot(): Hw {
  const backend = activeBuild()?.backend;
  return {
    cpu: hw.cpu,
    mem: hw.mem,
    gpus: enabledGpus(backend, hw.gpus, str(cfg.settings, "device")),
    os: hw.os,
    arch: hw.arch,
    // The tuner needs it too: which flags are even loadable depends on the
    // backend, not just how much VRAM it can see.
    backend,
  };
}

/**
 * "The machine's memory is materially as it was."
 *
 * Coarse on purpose — see `src/lib/adapt.ts`. This is a cache key for the
 * auto-tune, so it has to change when a game takes 20 GB or a compile finishes
 * and gives 8 GB back, and NOT change when the number wobbles by 200 MB. On a
 * workstation the wobble is constant and the difference is the whole design.
 */
export function headroomNow(): string {
  const vramCapacityB = vramTotalB();
  const ramCapacityB = hw.mem?.totalB ?? 0;
  return headroomKey({
    vramFreeB: vramCapacityB - vramUsedB(),
    vramCapacityB,
    ramFreeB: hw.mem?.availableB ?? 0,
    ramCapacityB,
  });
}

/**
 * Has the machine moved under a model that is already running?
 *
 * A loaded model cannot be re-placed, so this never re-tunes — it decides what to
 * TELL the user: something else is now competing for memory this server depends
 * on, or enough has come back that a restart would buy a real improvement.
 */
export function driftNow(): Drift {
  if (!memoryIsLive()) return { kind: "none" };
  const p = currentStatePlan();
  const d = drift({
    vramOverB: p.vram.overB,
    ramOverB: p.ram.overB,
    vramFreeB: p.vram.freeB,
    ramFreeB: p.ram.freeB,
    startedVramB: p.vram.usedB,
    startedRamB: p.ram.usedB,
    vramFreeAtStartB: srv.startFreeVramB,
    ramFreeAtStartB: srv.startFreeRamB,
  });
  // "Roomier" is only news when a restart could actually spend the room: a
  // run that already has every layer resident and its full trained context
  // gains nothing from one, however much came free.
  if (d.kind === "roomier") {
    const m = shownModel()?.meta;
    const maxed = m !== undefined && m !== null &&
      p.layersOnGpu >= p.nLayer && p.moeOnCpu === 0 &&
      p.ctx >= optimalCtx(m);
    if (maxed) return { kind: "none" };
  }
  return d;
}

/**
 * The load in progress, measured — or null when nothing is loading.
 *
 * A big model takes minutes to come up and "LOADING MODEL" alone reads as a
 * hang. The poll already measures everything an honest bar needs: the
 * device-wide VRAM drop since the spawn plus the process RSS, against the
 * plan's total for the running command (`src/lib/loadprogress.ts`).
 */
export function loadingNow():
  | (LoadProgress & { startedAt: number; note: string })
  | null {
  if (srv.status !== "starting" || srv.pid === 0) return null;
  const p = currentStatePlan();
  return {
    ...loadProgress({
      lines: srv.log,
      startFreeVramB: srv.startFreeVramB,
      freeVramB: vramTotalB() - vramUsedB(),
      rssB: srv.rssB,
      plannedB: p.vram.usedB + p.ram.usedB,
    }),
    startedAt: srv.startedAt,
    // The ladder's step-down note belongs with the progress it restarted.
    note: srv.fitNote,
  };
}

/**
 * RAM the mapped model holds RIGHT NOW — the bytes every "used" meter hides.
 *
 * The kernel books a memory-mapped model as reclaimable page cache, so with
 * 138 GB of DeepSeek-V4 resident, `free` said 22 GB used and the app's own
 * RAM bars agreed — the model read as simply not there. Measured per process
 * (`RssFile`), so it is attributable to our server rather than guessed from
 * the system-wide cache figure.
 */
export function mappedModelB(): number {
  return serverRunning() ? srv.rssFileB : 0;
}

export function vramTotalB(): number {
  return hw.gpus.reduce((a, g) => a + g.vramTotalB, 0);
}

export function vramUsedB(): number {
  return hw.gpus.reduce((a, g) => a + g.vramUsedB, 0);
}

// ── prerequisites ──────────────────────────────────────────────────────────

export function prereqById(id: string): Prereq | null {
  return prereq.items.find((i) => i.id === id) ?? null;
}

/** Ids of every prerequisite that was detected — the input `canCompile` wants. */
export function foundPrereqs(): Set<string> {
  return new Set(prereq.items.filter((i) => i.found).map((i) => i.id));
}

export function fixPlanFor(id: string): FixPlan | null {
  return prereq.plans[id] ?? null;
}

/** Missing prerequisites the app can actually act on. */
export function fixablePrereqs(): Prereq[] {
  return prereq.items.filter(
    (i) => !i.found && prereq.plans[i.id]?.kind !== "manual",
  );
}

// ── settings ───────────────────────────────────────────────────────────────

export function isTouched(key: string): boolean {
  return cfg.touched.includes(key);
}

export function changedCount(): number {
  return cfg.touched.length;
}

// ── server & chat ──────────────────────────────────────────────────────────

/**
 * The settings the memory view should describe.
 *
 * While a server is up that is what it was STARTED with, not what the panel now
 * holds: the moment the user edits a field the two diverge, and a diagram
 * labelled "current" that shows an unstarted configuration is a lie. When
 * nothing is running it is the working settings, which is a projection and is
 * labelled as one.
 */
export function shownSettings(): Settings {
  return srv.runSettings ?? cfg.settings;
}

/** The model the memory view should describe — the running one while it runs. */
export function shownModel(): Model | null {
  const path = srv.runModel;
  if (path) {
    return models.items.find((m) => m.path === path) ?? currentModel();
  }
  return currentModel();
}

/**
 * The pinned context, or 0.
 *
 * An override belongs to the model it was typed for; for any other model it is
 * void. Without this, a 128k pin chosen for a 262k model silently capped a 32k
 * model, and nothing on screen said why.
 */
export function ctxOverride(): number {
  return cfg.ctxOverrideFor === models.selected ? cfg.ctxOverride : 0;
}

/** Is the memory view describing a live process rather than a plan? */
export function memoryIsLive(): boolean {
  return srv.runSettings !== null && serverRunning();
}

/**
 * The machine as it is RIGHT NOW.
 *
 * When a server is up this is the plan of the command it was actually started
 * with, so llama.cpp's own share is itemised rather than lumped in with everyone
 * else's; when nothing is running every llama.cpp bucket is zero and the pools
 * show only what other processes hold and what is free. Same `plan` either way.
 */
export function currentStatePlan(): Plan {
  const m = shownModel()?.meta;
  if (memoryIsLive() && m && srv.runSettings) {
    // `plan` reads "in use" from device-wide telemetry — the driver's VRAM
    // figure and MemAvailable — and our own llama-server is already inside both.
    // Itemising our buckets on top of that would count our bytes twice and can
    // paint the over-capacity hatch on a machine that comfortably fits, so take
    // our share out of "everyone else" first. Our buckets do not depend on what
    // anyone else holds, so the first pass is only there to size us.
    const raw = computePlan(m, hwSnapshot(), srv.runSettings);
    const base = withoutOurUsage(
      hwSnapshot(),
      raw.vram.usedB,
      srv.rssB || raw.ram.usedB,
    );
    return computePlan(m, base, srv.runSettings);
  }
  return computePlan(NO_MODEL, hwSnapshot(), { ...cfg.settings, ngl: 0 });
}

/** What llama.master itself is holding right now, so a projection can take it
 *  back out instead of counting it as somebody else's memory. */
export function ourUsageB(): { vramB: number; ramB: number } {
  if (!memoryIsLive()) return { vramB: 0, ramB: 0 };
  const p = currentStatePlan();
  // RSS is measured; the VRAM figure is this app's own exact accounting for the
  // command that is running, which is the best available — the telemetry does
  // not attribute VRAM per process.
  return { vramB: p.vram.usedB, ramB: srv.rssB || p.ram.usedB };
}

/**
 * The machine to PLAN against: everything except our own running model.
 *
 * This is the base for every "what would happen if we started this" question —
 * the placement picker, the tuner, the stability check, every projected memory
 * plan. It must not be raw telemetry, and getting that wrong produced the worst
 * class of bug this app can have: a message that is false.
 *
 * The driver reports device-wide VRAM, so while our own llama-server is up its
 * 39 GB is inside that number. Planning against it asks "could we start this on a
 * machine that has 6 GB free" and answers, correctly for the question asked and
 * absurdly for the user, **"VRAM only: does not fit"** — while VRAM only is
 * exactly what is running. Same for `stability`, which then warns about an
 * overflow that is its own model, and for the Tune and Models memory plans.
 *
 * Attributing our bytes to us turns that back into the real question: what could
 * we start if we swapped what is loaded now for this. One model runs at a time,
 * so that is always the right question.
 */
export function planningHw(): Hw {
  const ours = ourUsageB();
  if (ours.vramB === 0 && ours.ramB === 0) return hwSnapshot();
  return withoutOurUsage(hwSnapshot(), ours.vramB, ours.ramB);
}

/**
 * The machine as it WILL look once the selected model runs.
 *
 * Current state, minus whatever llama.master is holding now, plus the selected
 * model under the selected settings — which is the definition that stops a
 * running model being counted twice. Null when no model with a readable header
 * is selected.
 */
export function projectedStatePlan(): Plan | null {
  const m = currentModel()?.meta;
  if (!m) return null;
  // A pinned context is IN the projection, always. With auto-optimal on the
  // tuner writes it into the settings anyway; with it off, projecting from
  // `cfg.settings` alone showed the previous tune's context — a user pinned
  // 1M, the map did not move, and "so what memory is missing?" had no answer
  // anywhere on the page. The clamp is the pin's own (`pinnedCtx`), so the
  // number projected is the number that would run.
  const pin = ctxOverride();
  const settings = pin > 0
    ? { ...cfg.settings, ctxSize: pinnedCtx(pin, trainedCtx(m)) }
    : cfg.settings;
  return computePlan(m, planningHw(), settings);
}

/**
 * Bytes read per generated token, for what the settings would run.
 *
 * The context term uses the CONFIGURED size, i.e. the cache full — the honest
 * pessimistic end, because a conversation gets slower as it fills and the number
 * that matters to someone choosing settings is what it degrades to.
 */
export function perTokenBytes(): { gpuB: number; ramB: number } | null {
  const m = currentModel()?.meta;
  if (!m) return null;
  const p = projectedStatePlan();
  if (!p) return null;
  return bytesPerToken(m, p, cfg.settings, p.ctx);
}

/** Tokens per second these settings should reach, and whether it is measured. */
export function projectedSpeed(): { tps: number; measured: boolean } | null {
  const b = perTokenBytes();
  if (!b) return null;
  // "Measured" only when the pools carrying this projection's time are the
  // calibrated ones — a GPU-only calibration says nothing about a CPU-heavy
  // run whose time is spent at the default RAM bandwidth.
  const measured = speedIsMeasured(b, cfg.gpuBps, cfg.ramBps);
  return {
    tps: estimateTps({
      gpuB: b.gpuB,
      ramB: b.ramB,
      gpuBps: cfg.gpuBps,
      ramBps: cfg.ramBps,
    }),
    measured,
  };
}

/**
 * What this machine actually achieved on the last reply, if that reply can teach
 * us anything about bandwidth.
 *
 * Only meaningful while the server that produced it is still up — the bytes have
 * to be the ones that were running, not whatever the form now holds.
 */
export function speedCalFromLastReply(): { gpuBps?: number; ramBps?: number } {
  if (!memoryIsLive() || chat.lastTps <= 0) return {};
  const m = shownModel()?.meta;
  const run = srv.runSettings;
  if (!m || !run) return {};
  const p = currentStatePlan();
  const b = bytesPerToken(m, p, run, p.ctx);
  return calibrate(chat.lastTps, b);
}

/**
 * Why this parameter cannot be used for the model that is selected, or "".
 *
 * The catalog describes what llama.cpp accepts; it cannot know what THIS model
 * supports. `--spec-type draft-mtp` against a model with no multi-token
 * prediction block is not a slow server — llama.cpp asserts on
 * `n_layer_nextn > 0` and refuses to load. A control that offers it anyway is a
 * raw error waiting to happen, which is the one thing this app promises not to
 * do. Lives here rather than in the catalog because it depends on cell state.
 */
export function paramBlocker(key: string): string {
  if (key === "specType") {
    const m = currentModel()?.meta;
    if (!m) return "Select a model first.";
    if (m.nextnLayers === 0) {
      return `${
        m.name || "This model"
      } ships no multi-token-prediction block, and llama.cpp refuses to load when one is asked for. Speculative decoding needs a model built with it.`;
    }
  }
  return "";
}

export function serverRunning(): boolean {
  return srv.status === "starting" || srv.status === "ready";
}

export function canSend(): boolean {
  return chat.input.trim().length > 0 && !chat.streaming;
}
