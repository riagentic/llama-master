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

import { computed } from "aio/air";
import { builds } from "../cell/builds.ts";
import { cfg } from "../cell/cfg.ts";
import { chat } from "../cell/chat.ts";
import { hw } from "../cell/hw.ts";
import { models } from "../cell/models.ts";
import { prereq } from "../cell/prereq.ts";
import { srv } from "../cell/srv.ts";
import { enabledGpus } from "../lib/gpu.ts";
import { drift, headroomKey } from "../lib/adapt.ts";
import {
  bestPlacement,
  optimalCtx,
  pinnedCtx,
  trainedCtx,
  tune,
  tuneAll,
} from "../lib/tune.ts";
import type { Placement, Tuning } from "../lib/tune.ts";
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
import { setupRows } from "../lib/setup.ts";
import {
  bandwidthNote,
  benchApplies,
  benchIsSound,
  specVerdict,
} from "../lib/bench.ts";
import type { BandwidthNote, BenchResult, SpecVerdict } from "../lib/bench.ts";
import { findMtpSibling } from "../lib/mtp.ts";
import type { MtpSibling } from "../lib/mtp.ts";
import { parseEnvVars } from "../lib/envvars.ts";
import type { EnvVar } from "../lib/envvars.ts";
import { quantAdvice, quantOptions } from "../lib/quant.ts";
import type { QuantOption } from "../lib/quant.ts";
import type { SetupRow } from "../lib/setup.ts";
import { num, str } from "../lib/params.ts";
import { queueNote, submitKind } from "../lib/queue.ts";
import { transcript } from "../lib/richtext.ts";
import { updateFor } from "../lib/update.ts";
import { runtimeMismatch } from "../lib/runtime.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { UpdateCheck } from "../lib/update.ts";
import type { FixPlan } from "../lib/fixplan.ts";
import type {
  Build,
  Hw,
  Model,
  Prereq,
  Reserve,
  Settings,
} from "../lib/types.ts";

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

/**
 * The flags the active build declares it accepts (`src/lib/caps.ts`).
 *
 * `null` until it has been probed, which the tuner reads as "switch nothing
 * extra on" — a lever left on the table, never a command the binary cannot
 * parse.
 */
export function activeCaps(): readonly string[] | null {
  return builds.caps[builds.activeId] ?? null;
}

/** Newest CUDA runtime every NVIDIA driver here can run; 0 = unknown. The
 *  asset picker caps to it, so the "auto" it shows is the one the install
 *  step will choose (`src/lib/assets.ts:scoreAsset`). */
export function cudaMax(): number {
  const v = hw.gpus.map((g) => g.cudaDriver ?? 0).filter((x) => x > 0);
  return v.length ? Math.min(...v) : 0;
}

export function activeBuild(): Build | null {
  return builds.installed.find((b) => b.id === builds.activeId) ?? null;
}

/**
 * Why the ACTIVE build cannot run the SELECTED model correctly, or null
 * (`src/lib/runtime.ts`). The case it exists for is the silent one: a
 * Hadamard-folded PrismML file loads on upstream llama.cpp and answers in
 * garbage, so this has to be said before Start, not diagnosed after.
 */
export function modelRuntime(): Diagnosis | null {
  const m = currentModel();
  const b = activeBuild();
  if (!m || !b) return null;
  return runtimeMismatch(m.meta?.vendor, b.ref, builds.installed);
}

/**
 * What the ACTIVE build has already proven it cannot do for the SELECTED model.
 *
 * Keyed on the pair, because that is what the fact is about: a newer build is a
 * different key and starts with a clean sheet, so "not implemented yet" cannot
 * outlive the build that said it (`cfg.unsupported`).
 */
export function unsupportedHere(): readonly string[] {
  const b = builds.activeId;
  const m = models.selected;
  if (!b || !m) return [];
  return cfg.unsupported[`${b}\n${m}`] ?? [];
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

/**
 * The environment variables the command preview should show.
 *
 * The same rule as `shownSettings`: while a server is up, the prefix on the
 * command is the one the RUNNING process carries — not whatever the input has
 * drifted to since, because the preview's whole job is to describe what exists.
 */
export function shownEnv(): readonly EnvVar[] {
  return srv.runEnv ?? parseEnvVars(cfg.envVars).vars;
}

/**
 * The tokens of the env input that are not a `NAME=value` this app can honour,
 * with the reason. Empty when the line is clean — a clean line stays quiet.
 */
export function envProblems(): { token: string; why: string }[] {
  return parseEnvVars(cfg.envVars).bad.map((token) => ({
    token,
    why: token.includes("=")
      ? token.endsWith("=")
        ? "empty value — unset the variable instead of setting it to nothing"
        : "the value is empty or carries a $, which a shell would expand and this input cannot"
      : "not a NAME=value assignment",
  }));
}

/**
 * The shown configuration in words — the command's flags, humanly readable.
 * Same sources as the command view, so the two cannot disagree: a running
 * server's rows describe what it was STARTED with.
 */
export function shownSetup(): SetupRow[] {
  return setupRows(shownModel()?.meta ?? null, shownSettings(), hwSnapshot());
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

// ── The planner chain is MEMOISED ─────────────────────────────────────────
//
// Everything from here to `projectedSpeed` is `computed()`: cached until a cell
// it read changes, and read through a plain function so call sites stay
// property reads (`tests/guards.test.ts`). It was not, and the all-in-one page
// root — which reads chat, hw and srv — re-ran `tuneAll` three times per render
// (placements → projectedSettings → projectedStatePlan, then perTokenBytes did
// it all again), at ~14 ms each on a 94-layer MoE: ~100 ms of JS on every
// streamed-token flush (2-16/s), every hw tick and every srv tick. That is what
// made typing lag while a reply was streaming. A chat flush now touches none
// of these, and a 1 s hw tick recomputes each once.
//
// `computed`, not a hand-rolled cache: a cache hit that skips the read skips
// the subscription (dep/aio/docs/ui/reactivity-tracking.md), and a computed
// replays it.

/**
 * The machine as it is RIGHT NOW.
 *
 * When a server is up this is the plan of the command it was actually started
 * with, so llama.cpp's own share is itemised rather than lumped in with everyone
 * else's; when nothing is running every llama.cpp bucket is zero and the pools
 * show only what other processes hold and what is free. Same `plan` either way.
 */
const currentStatePlanC = computed(() => {
  const m = shownModel()?.meta;
  if (memoryIsLive() && m && srv.runSettings) {
    // `plan` reads "in use" from device-wide telemetry — the driver's VRAM
    // figure and MemAvailable — and our own llama-server is already inside both.
    // Itemising our buckets on top of that would count our bytes twice and can
    // paint the over-capacity hatch on a machine that comfortably fits, so take
    // our share out of "everyone else" first. Our buckets do not depend on what
    // anyone else holds, so the first pass is only there to size us.
    const raw = computePlan(m, hwSnapshot(), srv.runSettings, "running");
    // VRAM: only when there is nothing better. `withoutOurUsage` divides our
    // footprint across the cards BY PROPORTION of what each already holds,
    // which is a guess about the one thing the machine can be asked directly —
    // free-at-spawn against free-now, per card (`srv.runCardFreeB`). When that
    // baseline is present `plan` does the subtraction itself, exactly, and
    // taking a proportional share out here first would remove those bytes
    // twice. RAM has no per-card question and keeps the measured RSS either way.
    const cardFreeB = srv.runCardFreeB;
    const perCard = cardFreeB.length === hwSnapshot().gpus.length &&
      cardFreeB.some((b) => b > 0);
    const base = withoutOurUsage(
      hwSnapshot(),
      perCard ? 0 : raw.vram.usedB,
      srv.rssB || raw.ram.usedB,
    );
    // "running", so the fitter does not re-litigate a placement llama.cpp has
    // already made. It used to: the per-card budgets hold back a safety
    // reserve and our own footprint is re-derived by proportion, so re-packing
    // a loaded model came up short and this panel announced "1010 MB of layers
    // have nowhere to go" while `vram.overB` read 0 beside it and the model
    // answered prompts (`src/lib/plan.ts:PlanQuestion`).
    return computePlan(m, base, srv.runSettings, "running", cardFreeB);
  }
  return computePlan(NO_MODEL, hwSnapshot(), { ...cfg.settings, ngl: 0 });
});
export function currentStatePlan(): Plan {
  return currentStatePlanC.value;
}

/** What llama.master itself is holding right now, so a projection can take it
 *  back out instead of counting it as somebody else's memory. */
export function ourUsageB(): { vramB: number; ramB: number } {
  if (!memoryIsLive()) return { vramB: 0, ramB: 0 };
  const p = currentStatePlan();
  // RSS is measured; the VRAM figure is this app's own exact accounting for the
  // command that is running, which is the best available — the telemetry does
  // not attribute VRAM per process.
  //
  // Only the ANONYMOUS share of the RSS. The file-backed share (`rssFileB` —
  // a mapped model's weights, 138 GB of the 139 on the run that motivated
  // this) is reclaimable page cache that `MemAvailable` already counts as
  // free, so handing it to `withoutOurUsage` adds those bytes to a figure
  // that already contains them — and the clamp to `totalB` then declared the
  // whole of RAM free, everyone else's memory included. The tuner planned
  // against that.
  return {
    vramB: p.vram.usedB,
    ramB: Math.max(0, srv.rssB - srv.rssFileB) || p.ram.usedB,
  };
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
const planningHwC = computed(() => {
  const ours = ourUsageB();
  const base = ours.vramB === 0 && ours.ramB === 0
    ? hwSnapshot()
    : withoutOurUsage(hwSnapshot(), ours.vramB, ours.ramB);
  // The user's own claim on the machine, attached HERE and nowhere else: this
  // is the one function every "what would happen if we started this" question
  // goes through, and the reserve has to bind all of them — the tuner, the
  // placement picker, the stability check, the projected bars — or it would be
  // a number that changes one screen and not the run.
  //
  // Deliberately NOT on `hwSnapshot()`: that is the machine as it is, and the
  // current-state view must keep reporting real free memory. Reserved bytes are
  // free until something takes them; what they must never be is SPENDABLE.
  return { ...base, reserve: reserveNow() };
});
export function planningHw(): Hw {
  return planningHwC.value;
}

/** What the user has told the app to keep for themselves. */
export function reserveNow(): Reserve {
  return {
    perGpuB: cfg.reservePerGpuVramB,
    connectedB: cfg.reserveConnectedVramB,
    ramB: cfg.reserveRamB,
  };
}

/**
 * What the user's reserve is costing them, in the units the tuner works in.
 *
 * A reserve is honoured by planning as if the memory were absent, which is the
 * right thing and also an invisible one: the tuner simply returns a smaller
 * answer, and nothing on screen connects "my context is 16k" to the 8 GB the
 * user asked to keep. On a MoE model the connection is direct and steep — a
 * layer of routed experts is 2.5 GB on this class of model, so 8 GB is three
 * layers that could have been on the GPU, and each one measured about 2% of the
 * generation rate.
 *
 * Deliberately reported as FACTS the tuner produced (layers, context) rather
 * than as a predicted speed: the 2%-a-layer figure is measured on one model and
 * one machine, and dressing it up as a general rate would be the kind of
 * confident guess this app refuses everywhere else. Null when the reserve costs
 * nothing, which is the common case and should say nothing at all.
 */
export type ReserveCost =
  | {
    blocks: true;
    layers?: undefined;
    ctxLost?: undefined;
    ctxWith?: undefined;
  }
  | { blocks?: false; layers: number; ctxLost: number; ctxWith: number }
  | null;

const reserveCostC = computed((): ReserveCost => {
  const m = currentModel();
  const r = reserveNow();
  if (!m?.meta) return null;
  if (r.perGpuB <= 0 && r.connectedB <= 0 && r.ramB <= 0) return null;
  const base = planningHw();
  const withReserve = tune(m.meta, base, cfg.settings);
  const without = tune(m.meta, { ...base, reserve: undefined }, cfg.settings);
  // The loudest case, and the one the first version of this returned `null`
  // for: the reserve is what makes the model impossible. A refusal caused by
  // the user's own setting has to name the control that gives the memory back,
  // or it reads as "this machine cannot run this model".
  if (!withReserve.possible) return without.possible ? { blocks: true } : null;
  if (!without.possible) return null;
  // `--n-cpu-moe` counts layers held BACK, so more is worse. A dense model
  // moves `ngl` instead, and the difference there is layers too.
  const held = (t: typeof withReserve) =>
    m.meta && m.meta.nExpert > 0
      ? num(t.settings, "nCpuMoe")
      : Math.max(0, m.meta ? m.meta.nLayer - num(t.settings, "ngl") : 0);
  const layers = Math.max(0, held(withReserve) - held(without));
  const ctxLost = Math.max(0, without.ctx - withReserve.ctx);
  if (layers === 0 && ctxLost === 0) return null;
  return { layers, ctxLost, ctxWith: withReserve.ctx };
});
export function reserveCost(): ReserveCost {
  return reserveCostC.value;
}

/**
 * Every placement for the current model, so the UI can compare them without
 * three separate calls. Null when no model with a readable header is selected.
 */
/**
 * The multi-token-prediction drafter published beside the selected model.
 *
 * A derived value rather than something the tuner works out, because finding it
 * needs the model SCAN and `src/lib/` is pure. Gemma 4 ships its MTP heads as a
 * separate GGUF, so the main file's header honestly says zero blocks — and the
 * tuner honestly concluded there was nothing to draft with, over a 2-3x
 * speed-up sitting in the same directory (`src/lib/mtp.ts`).
 */
const mtpSiblingC = computed<MtpSibling | null>(() => {
  const m = currentModel();
  if (!m) return null;
  return findMtpSibling(m.path, models.items);
});
export function mtpSibling(): MtpSibling | null {
  return mtpSiblingC.value;
}

const placementsC = computed(() => {
  const m = currentModel();
  if (!m?.meta) return null;
  return tuneAll(
    m.meta,
    // NOT raw telemetry: while our own server is up its VRAM is inside the
    // driver's device-wide figure, and planning against that reported "VRAM only:
    // does not fit" for the model that was running in VRAM only at the time.
    planningHw(),
    cfg.settings,
    // Two different things, and passing them as one number conflated an
    // instruction with a hint: a context the user typed is EXACT (the tuner
    // holds it and reports the shortfall when it cannot), while the measured
    // fit is only a search ceiling for the automatic path — aiming past it
    // would just walk the retry ladder back down (`src/lib/fitladder.ts`).
    ctxOverride() || undefined,
    measuredCtx(m.path) || undefined,
    { mtpSibling: mtpSibling(), caps: activeCaps() },
  );
});
export function placements(): Record<Placement, Tuning> | null {
  return placementsC.value;
}

/**
 * The largest context each of the two GPU placements can hold, hunted to the
 * model's advertised maximum — the "Max·VRAM" / "Max·Hybrid" buttons. Two
 * full hunts (~15 ms each on a big model), memoised for the same reason as the
 * chain above: `CtxControls` computed them in its body on every render.
 */
const maxTuningsC = computed(
  (): { vram: Tuning | null; hybrid: Tuning | null } => {
    const m = currentModel();
    if (!m?.meta) return { vram: null, hybrid: null };
    const meta = m.meta;
    const h = planningHw();
    const hunt = (pl: Placement) =>
      tune(meta, h, cfg.settings, pl, undefined, undefined, true);
    return { vram: hunt("vram"), hybrid: hunt("hybrid") };
  },
);
export function maxTunings(): { vram: Tuning | null; hybrid: Tuning | null } {
  return maxTuningsC.value;
}

/** The largest context this model has been observed to actually start at on
 *  this machine, or 0 if it has never run. */
export function measuredCtx(path: string): number {
  return cfg.fitCtx[path] ?? 0;
}

/**
 * The settings a Start pressed NOW would actually run.
 *
 * Not always `cfg.settings`. With auto-optimal on, Start re-tunes first
 * (`actions.ts:applyOptimal`), and the tuner is suspended while a server is up —
 * a loaded model cannot be re-placed, so the app deliberately stops rewriting
 * settings under it. The consequence was that the panel titled "after starting"
 * projected a settings map tuned at some earlier moment, and on a machine whose
 * memory had moved since, that stale map does not fit: the projection reported
 * gigabytes of layers with nowhere to go for a command nobody was ever going to
 * issue. What a restart would get is the TUNER's answer for the machine as it is
 * now, which is what this returns.
 *
 * With auto-optimal off the user's own settings are what Start runs, so those
 * are what gets projected — including a pinned context, which the tuner would
 * otherwise be the only thing writing in.
 */
const projectedSettingsC = computed(() => {
  const m = currentModel()?.meta;
  const pin = m ? ctxOverride() : 0;
  // The clamp is the pin's own (`pinnedCtx`), so the number projected is the
  // number that would run. A user pinned 1M, the map did not move, and "so what
  // memory is missing?" had no answer anywhere on the page.
  const own: Settings = pin > 0 && m
    ? { ...cfg.settings, ctxSize: pinnedCtx(pin, trainedCtx(m)) }
    : cfg.settings;
  if (!cfg.autoOptimal) return own;
  const all = placements();
  if (!all) return own;
  // Start's own fallback rule: the chosen placement, or the best one that can
  // run this model when the chosen one cannot (`actions.ts:tunedForStart`). A
  // refusal keeps the user's settings so the projection still SHOWS what is
  // missing rather than drawing a fitting plan of something else.
  const chosen = all[cfg.placement].possible
    ? cfg.placement
    : bestPlacement(all);
  return all[chosen].possible ? all[chosen].settings : own;
});
export function projectedSettings(): Settings {
  return projectedSettingsC.value;
}

/**
 * The machine as it WILL look once the selected model runs.
 *
 * Current state, minus whatever llama.master is holding now, plus the selected
 * model under the settings a Start would use — which is the definition that
 * stops a running model being counted twice. Null when no model with a readable
 * header is selected.
 */
const projectedStatePlanC = computed(() => {
  const m = currentModel()?.meta;
  if (!m) return null;
  return computePlan(m, planningHw(), projectedSettings());
});
export function projectedStatePlan(): Plan | null {
  return projectedStatePlanC.value;
}

/**
 * Bytes read per generated token, for what the settings would run.
 *
 * The context term uses the CONFIGURED size, i.e. the cache full — the honest
 * pessimistic end, because a conversation gets slower as it fills and the number
 * that matters to someone choosing settings is what it degrades to.
 */
const perTokenBytesC = computed(() => {
  const m = currentModel()?.meta;
  if (!m) return null;
  const p = projectedStatePlan();
  if (!p) return null;
  // The bytes must come from the SAME settings the plan was drawn from — which,
  // with auto-optimal on, is the tuner's answer, not the stale map the user's
  // panel still shows. Feeding `cfg.settings` here made the two halves of the
  // page disagree about the same model: the memory bars drew the tuned
  // placement (say 29 layers on the GPU) while the projected tokens/second was
  // computed from the user's older `-ngl` (say 0, all in RAM), reporting a far
  // slower speed than the placement actually shown would reach. Same bug the
  // speed.ts fix addresses — plan and speed must describe one placement.
  return bytesPerToken(m, p, projectedSettings(), p.ctx);
});
export function perTokenBytes(): { gpuB: number; ramB: number } | null {
  return perTokenBytesC.value;
}

/** Tokens per second these settings should reach, and whether it is measured. */
const projectedSpeedC = computed(() => {
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
});
export function projectedSpeed(): { tps: number; measured: boolean } | null {
  return projectedSpeedC.value;
}

/**
 * What this machine actually achieved, if the observation can teach us anything
 * about bandwidth.
 *
 * Only meaningful while the server that produced it is still up — the bytes have
 * to be the ones that were running, not whatever the form now holds.
 *
 * The BENCH wins over the chat whenever there is one. Both are real rates, but
 * a chat reply is a rate about an unknown prompt at an unknown fill with an
 * unknown amount of thinking in it, and a bench is the same measurement with
 * every one of those held still (`src/lib/bench.ts`). Preferring the noisier
 * number because it arrived later would throw away the only reason to run a
 * bench at all.
 */
export function speedCalFromLastReply(): { gpuBps?: number; ramBps?: number } {
  if (!memoryIsLive()) return {};
  const m = shownModel()?.meta;
  const run = srv.runSettings;
  if (!m || !run) return {};
  const b = srv.lastBench;
  const useBench = benchApplies(b, srv.runModel, Number(run.ctxSize ?? 0)) &&
    benchIsSound(b, m);
  const tps = useBench ? b.genTps : chat.lastTps;
  if (tps <= 0) return {};
  const p = currentStatePlan();
  // The context ACTUALLY filled, not the configured maximum. A bench runs at a
  // near-empty cache, so billing it for a 262,144-token cache it never read
  // would attribute those phantom bytes to the machine and calibrate a
  // bandwidth several times too high.
  const filled = useBench ? Math.max(1, b.promptTokens + b.genTokens) : p.ctx;
  const bytes = bytesPerToken(m, p, run, filled);
  return calibrate(tps, bytes);
}

/** The last PROSE speed measurement, when it describes the run on screen. */
export function benchNow(): BenchResult | null {
  const run = srv.runSettings;
  if (!run) return null;
  const b = srv.lastBench;
  return benchApplies(b, srv.runModel, Number(run.ctxSize ?? 0)) ? b : null;
}

/** The last CODE speed measurement, when it describes the run on screen. */
export function benchCodeNow(): BenchResult | null {
  const run = srv.runSettings;
  if (!run) return null;
  const b = srv.lastBenchCode;
  return benchApplies(b, srv.runModel, Number(run.ctxSize ?? 0)) ? b : null;
}

/**
 * What the two measurements say about speculative decoding, together.
 *
 * Read against the settings the RUN was started with, never the panel's current
 * contents: speculative decoding is a load-time flag, so a user who has since
 * flipped the switch has changed the next run and not this one, and judging a
 * measurement by a setting that was not in force when it was taken is how a
 * verdict comes to contradict its own numbers.
 */
export function specVerdictNow(): SpecVerdict | null {
  const run = srv.runSettings;
  if (!run) return null;
  return specVerdict(benchNow(), benchCodeNow(), str(run, "specType"));
}

/**
 * What to say about this machine's memory bandwidth, or null for nothing.
 *
 * Only when the running placement actually reads host RAM per token — on a
 * VRAM-only run the figure is true and irrelevant, and the commonest cause of
 * a low one (a BIOS memory profile left off) would be advice about nothing.
 */
export function bandwidthNow(): BandwidthNote | null {
  if (!memoryIsLive()) return null;
  const m = shownModel()?.meta;
  const run = srv.runSettings;
  if (!m || !run) return null;
  const p = currentStatePlan();
  const b = bytesPerToken(m, p, run, p.ctx);
  // "Reads host RAM" means enough of it to matter: a rounding-error share is
  // not what makes a model slow, and warning about it would be noise.
  const readsHostRam = b.totalB > 0 && b.ramB / b.totalB > 0.05;
  return bandwidthNote(cfg.ramBps, readsHostRam);
}

/**
 * Every quantisation of this model worth comparing, as it would run here.
 *
 * Memoised with the rest of the planner chain because it is the most expensive
 * thing in the app: one `tune` per placement per candidate, so up to eighteen
 * on a model with six offers. It is only ever read by a panel the user opened.
 */
const quantRowsC = computed((): QuantOption[] => {
  const m = currentModel()?.meta;
  if (!m) return [];
  return quantOptions(m, planningHw(), cfg.settings, {
    gpuBps: cfg.gpuBps,
    ramBps: cfg.ramBps,
  });
});
export function quantRows(): QuantOption[] {
  return quantRowsC.value;
}

/** The one line worth showing about a smaller quantisation, or "". */
export function quantAdviceNow(): string {
  return quantAdvice(quantRows());
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

/**
 * What the composer's submit button says — and "" when it can do nothing.
 *
 * The word has to change, because the two gestures have different
 * consequences: "Send" starts a request, "Queue" writes a note for later. A
 * button that said Send while the model was mid-reply would be promising
 * something it cannot deliver for another thirty seconds.
 *
 * `submitKind` (pure, `src/lib/queue.ts`) decides; this only names the answer,
 * so both surfaces and the tests agree on when the button is live. `draft` is
 * the text in the box — browser-local, so it is an argument, not a cell read.
 */
export function submitLabel(draft: string): string {
  switch (submitKind(draft, chat.queue.length, chat.streaming)) {
    case "queue":
      return "Queue";
    case "send":
      return "Send";
    // Full is a live button that would drop the message, which is worse than a
    // dead one — the note above the input carries the reason.
    case "full":
    case null:
      return "";
  }
}

/** The messages waiting, oldest first. `.slice()` — never a spread — because
 *  this array is written by a live async method. */
export function chatQueue(): string[] {
  return chat.queue.slice();
}

/** The sentence above the input: how many are waiting, and whether anything is
 *  coming to collect them. */
export function chatQueueNote(): string {
  return queueNote(chat.queue.length, chat.streaming);
}

/**
 * The whole conversation as markdown, for the copy-chat button.
 *
 * Here rather than in the two chat surfaces because it is a value derived from
 * cell state, and both must copy the same thing — including the reply still
 * arriving, which is on screen and therefore part of what "copy the chat"
 * means (`src/lib/richtext.ts:transcript`).
 */
export function chatTranscript(): string {
  return transcript({
    system: chat.system,
    messages: chat.messages,
    partial: chat.partial,
    partialThink: chat.partialThink,
  });
}

/** Is there anything to copy or clear? A button that copies an empty string
 *  should be disabled rather than silently emptying the clipboard. */
export function chatHasContent(): boolean {
  return chat.messages.length > 0 || chat.partial.length > 0 ||
    chat.partialThink.length > 0;
}
