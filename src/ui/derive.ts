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
  return drift({
    vramOverB: p.vram.overB,
    ramOverB: p.ram.overB,
    vramFreeB: p.vram.freeB,
    ramFreeB: p.ram.freeB,
    startedVramB: p.vram.usedB,
    startedRamB: p.ram.usedB,
  });
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
  return computePlan(m, planningHw(), cfg.settings);
}

export function serverRunning(): boolean {
  return srv.status === "starting" || srv.status === "ready";
}

export function canSend(): boolean {
  return chat.input.trim().length > 0 && !chat.streaming;
}
