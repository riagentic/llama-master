// src/ui/derive.ts — the values the UI derives from cell state.
//
// WHY THIS FILE EXISTS, and why components must not call cell selectors:
//
// In AIR, reading a cell PROPERTY (`models.items`) subscribes the component to
// that slice, so it re-renders when the slice changes. Calling a cell SELECTOR
// (`models.current()`) returns a correct, fresh value but registers NO
// dependency. A component whose only read is a selector call therefore renders
// once and then never updates — the data is right, the screen is stale, and
// nothing warns you. It cost an afternoon here (the all-in-one page kept
// showing "no model" while the model dropdown beside it showed the model),
// and it is reported upstream in dep/aio/feedback/llama-master.md.
//
// So: every derived value the UI needs is a plain function over cell
// PROPERTIES, and `tests/guards.test.ts` fails the build if a component calls a
// selector instead. The cells keep their selectors — they are correct and
// useful on the server and in tests; they are just not reactive in a render.

import { builds } from "../cell/builds.ts";
import { cfg } from "../cell/cfg.ts";
import { chat } from "../cell/chat.ts";
import { hw } from "../cell/hw.ts";
import { models } from "../cell/models.ts";
import { prereq } from "../cell/prereq.ts";
import { srv } from "../cell/srv.ts";
import { enabledGpus } from "../lib/gpu.ts";
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

export function serverRunning(): boolean {
  return srv.status === "starting" || srv.status === "ready";
}

export function canSend(): boolean {
  return chat.input.trim().length > 0 && !chat.streaming;
}
