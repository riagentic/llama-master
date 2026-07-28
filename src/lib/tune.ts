// src/lib/tune.ts — "Optimal settings", as one pure function.
//
// There is ONE set of optimal settings, not three quality levels. What the user
// chooses is WHERE the model runs:
//
//   VRAM only — every layer on the GPU. Several times faster than any split,
//               and bounded by the card: a model that does not fit cannot use
//               this placement at all, and says so.
//   Hybrid    — GPU for what fits, RAM for the rest. On a mixture-of-experts
//               model the routed experts move first: they are most of the bytes
//               and least of the latency.
//   CPU only  — no GPU. Always available, always slowest.
//
// Everything else follows from one goal: reach the OPTIMAL CONTEXT, which is
// the largest context at which the model still performs the best — its trained
// context. Past `nCtxTrain` llama.cpp has to extrapolate the positional
// encoding and answers degrade, so a larger number would be a worse model, not
// a better one. Each placement gets as close to that ceiling as its memory
// allows and never crosses it.
//
// The result carries the context actually reached and the reasoning, so the UI
// can show all three placements side by side and the user can see what each one
// costs before choosing.

import { plan } from "./plan.ts";
import { defaults, num, str } from "./params.ts";
import type { Hw, ModelMeta, Settings } from "./types.ts";

/** Where the model runs. The only placement choice the user makes. */
export type Placement = "vram" | "hybrid" | "cpu";

export const PLACEMENTS: readonly {
  id: Placement;
  label: string;
  tip: string;
}[] = [
  {
    id: "vram",
    label: "VRAM only",
    tip:
      "Every layer on the GPU. Several times faster than any split; the context is limited by what the card can hold.",
  },
  {
    id: "hybrid",
    label: "Hybrid",
    tip:
      "As much on the GPU as fits, the rest in system RAM. Runs models far larger than the card, paying for the parts that live in RAM.",
  },
  {
    id: "cpu",
    label: "CPU only",
    tip:
      "No GPU at all. Works on any machine and needs nothing installed; expect a few tokens per second on a large model.",
  },
];

export type Tuning = {
  settings: Settings;
  reasons: string[];
  /** The context this placement actually reached. 0 when it cannot run. */
  ctx: number;
  /** The context it aimed at — the model's trained maximum. */
  optimalCtx: number;
  /** False when this placement cannot run this model on this machine. */
  possible: boolean;
  /** Why not, in the user's terms, when `possible` is false. */
  blocker: string;
  /** One line for the picker: what choosing this actually gets you. */
  summary: string;
};

/**
 * The largest context at which the model still performs the best.
 *
 * Its trained context, and nothing else. Beyond it the positional encoding is
 * extrapolated and answers get worse while the number looks better — so this is
 * the ceiling every placement aims at and none exceeds.
 */
export function optimalCtx(meta: ModelMeta): number {
  return meta.nCtxTrain > 0 ? meta.nCtxTrain : 4096;
}

/** Below this a context is too short to hold a conversation, so a placement
 *  that cannot reach it is reported impossible rather than proposed. */
export const MIN_CTX = 2048;

/**
 * The context sizes worth one click.
 *
 * Powers of two because that is how every model and every published benchmark
 * describes its context, and because the KV cache doubles with each step — so
 * the ladder is also the cost ladder. A preset above what the model was trained
 * for is offered but disabled rather than hidden: "1M is possible, not for THIS
 * model" is information, and a row that changes length per model is harder to
 * use than one that does not.
 */
export const CTX_PRESETS: readonly number[] = [
  16_384,
  32_768,
  65_536,
  131_072,
  262_144,
  524_288,
  1_048_576,
];

/** `131072` → `128k`, `1048576` → `1M`. The label users recognise. */
export function ctxLabel(tokens: number): string {
  if (tokens >= 1_048_576) return `${Math.round(tokens / 1_048_576)}M`;
  if (tokens >= 1024) return `${Math.round(tokens / 1024)}k`;
  return String(tokens);
}

/** Contexts are searched on this grid: llama.cpp allocates the cache in blocks,
 *  and 8,192 is a nicer thing to read than 8,137. */
const CTX_STEP = 256;

/**
 * Backends on which a quantised KV cache is a safe thing to propose.
 *
 * Where the backend has no quantised-KV kernel the server refuses to load, so
 * proposing one would mean "optimal settings" that do not start. CUDA and Metal
 * have carried it reliably for years; Vulkan and ROCm are uneven across driver
 * and card. An unknown backend is treated as "do not risk it".
 */
const QUANT_KV_BACKENDS: readonly string[] = ["cuda", "metal"];

/** VRAM we refuse to plan into: drivers, the desktop, and fragmentation. */
function marginB(vramTotalB: number): number {
  return Math.max(512 * 1024 * 1024, vramTotalB * 0.05);
}

/**
 * RAM we refuse to plan into.
 *
 * Weights and KV cache are anonymous pages the kernel cannot reclaim, so
 * consuming all of `MemAvailable` does not mean "slow", it means the OOM killer
 * — and llama-server is the biggest process on the machine.
 */
function ramMarginB(availB: number): number {
  return Math.max(1024 ** 3, availB * 0.10);
}

function fitsVram(meta: ModelMeta, hw: Hw, s: Settings): boolean {
  const p = plan(meta, hw, s);
  const total = hw.gpus.reduce((a, g) => a + g.vramTotalB, 0);
  return p.vram.overB === 0 && p.vram.freeB >= marginB(total);
}

/** Does the host side of this placement leave the OS room to breathe? */
function fitsRam(meta: ModelMeta, hw: Hw, s: Settings): boolean {
  const availB = hw.mem?.availableB ?? 0;
  if (availB === 0) return true; // nothing known about RAM: invent no limit
  const p = plan(meta, hw, s);
  return p.ram.overB === 0 && p.ram.freeB >= ramMarginB(availB);
}

/** Largest `-ngl` that still fits in VRAM (nLayer ≤ 1000, so a scan is fine). */
function maxLayers(meta: ModelMeta, hw: Hw, s: Settings): number {
  for (let n = meta.nLayer; n >= 0; n--) {
    if (fitsVram(meta, hw, { ...s, ngl: n })) return n;
  }
  return 0;
}

/** Fewest expert-layers in RAM that let the rest fit at this `-ngl`, or null.
 *
 *  Ascending, so the answer is the SMALLEST N: every expert left on the GPU is
 *  one that does not have to cross the PCIe bus. */
function minCpuMoe(
  meta: ModelMeta,
  hw: Hw,
  s: Settings,
  ngl = 999,
): number | null {
  for (let n = 1; n <= meta.nLayer; n++) {
    if (fitsVram(meta, hw, { ...s, ngl, nCpuMoe: n })) return n;
  }
  return null;
}

/** The CPU budget, which no longer varies by mode: two physical cores stay with
 *  the OS so the desktop keeps repainting, for generation and prefill alike. */
function cpuBudget(cores: number) {
  return {
    threads: Math.max(1, cores - 2),
    threadsBatch: Math.max(1, cores - 2),
    ubatch: 512,
    batch: 2048,
  };
}

/**
 * Place the model at a GIVEN context, or report that it cannot be placed.
 *
 * This is the whole difference between the three placements. It never chooses
 * the context — `bestCtx` does that.
 */
function place(
  meta: ModelMeta,
  hw: Hw,
  base: Settings,
  placement: Placement,
  ctx: number,
): { settings: Settings; note: string } | null {
  const s: Settings = { ...base, ctxSize: ctx };

  if (placement === "cpu") {
    const cpu = { ...s, ngl: 0, nCpuMoe: 0 };
    return fitsRam(meta, hw, cpu)
      ? { settings: cpu, note: "every layer in system RAM" }
      : null;
  }

  if (hw.gpus.length === 0) return null;

  // Both GPU placements start the same way: try to put everything on the card.
  const full = { ...s, ngl: 999, nCpuMoe: 0 };
  if (fitsVram(meta, hw, full) && fitsRam(meta, hw, full)) {
    return { settings: full, note: "every layer on the GPU" };
  }
  if (placement === "vram") return null; // by definition nothing else is allowed

  // Hybrid. Routed experts leave first: on a MoE model they are most of the
  // bytes and least of the latency, so moving them buys far more room than
  // dropping whole layers.
  if (meta.nExpert > 0) {
    const n = minCpuMoe(meta, hw, s);
    if (n !== null) {
      const moe = { ...s, ngl: 999, nCpuMoe: n };
      if (fitsRam(meta, hw, moe)) {
        return {
          settings: moe,
          note:
            `attention on the GPU, routed experts of ${n} of ${meta.nLayer} layers in RAM`,
        };
      }
    }
  }

  // Then whole layers, with every expert already in RAM so a layer that stays
  // on the GPU costs only its attention.
  const moeMax = meta.nExpert > 0 ? meta.nLayer : 0;
  const withExperts = { ...s, nCpuMoe: moeMax };
  let n = maxLayers(meta, hw, withExperts);
  let ubatch = num(s, "ubatchSize");
  // A smaller micro-batch buys back the last few hundred MB of compute buffer.
  if (n < meta.nLayer && ubatch > 256) {
    const n2 = maxLayers(meta, hw, { ...withExperts, ubatchSize: 256 });
    if (n2 > n) {
      n = n2;
      ubatch = 256;
    }
  }
  if (n === 0) return null; // nothing on the GPU is not a hybrid placement
  const split: Settings = {
    ...s,
    ngl: n,
    ubatchSize: ubatch,
    // Hand experts back to the GPU wherever they still fit at this -ngl.
    nCpuMoe: moeMax === 0
      ? 0
      : minCpuMoe(meta, hw, { ...s, ngl: n, ubatchSize: ubatch }, n) ?? moeMax,
  };
  if (!fitsRam(meta, hw, split)) return null;
  const movedExperts = Number(split.nCpuMoe) || 0;
  return {
    settings: split,
    // Name the experts when they moved: on a MoE model that is the decision
    // doing the work, and "30 of 32 layers" alone hides it.
    note: movedExperts > 0
      ? `${n} of ${meta.nLayer} layers on the GPU, and the routed experts of ${movedExperts} of them in RAM`
      : `${n} of ${meta.nLayer} layers on the GPU, the rest in RAM`,
  };
}

/**
 * The largest context on the grid, at most `ceiling`, this placement can hold —
 * or 0 when even the shortest usable context does not fit.
 *
 * Binary search: memory demand rises monotonically with the context, so the
 * highest fitting value is well defined and reached in ~10 probes instead of
 * hundreds.
 */
function bestCtx(
  meta: ModelMeta,
  hw: Hw,
  base: Settings,
  placement: Placement,
  ceiling: number,
): number {
  const top = Math.max(MIN_CTX, Math.floor(ceiling / CTX_STEP) * CTX_STEP);
  if (place(meta, hw, base, placement, top)) return top;
  if (!place(meta, hw, base, placement, MIN_CTX)) return 0;
  // Invariant: lo fits, hi does not.
  let lo = MIN_CTX;
  let hi = top;
  while (hi - lo > CTX_STEP) {
    const mid = Math.floor((lo + hi) / 2 / CTX_STEP) * CTX_STEP;
    if (mid <= lo || mid >= hi) break;
    if (place(meta, hw, base, placement, mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * Optimal settings for this model, on this machine, in this placement.
 *
 * `base` supplies everything the tuner does not decide (port, sampling). Pass
 * `ctxOverride` to hold the context at a value the user typed — an edited
 * context is an instruction, not a suggestion to be re-optimised.
 */
export function tune(
  meta: ModelMeta,
  hw: Hw,
  base: Settings,
  placement: Placement = "vram",
  ctxOverride?: number,
): Tuning {
  const d = defaults();
  const s: Settings = { ...d, ...base };
  const reasons: string[] = [];
  const target = optimalCtx(meta);

  // The cache type is the tuner's to decide, so start from llama.cpp's default
  // rather than from whatever the last model happened to need.
  s.cacheTypeK = str(d, "cacheTypeK");
  s.cacheTypeV = str(d, "cacheTypeV");

  const cores = hw.cpu?.cores ?? 0;
  if (cores > 0) {
    const b = cpuBudget(cores);
    s.threads = b.threads;
    s.threadsBatch = b.threadsBatch;
    s.ubatchSize = b.ubatch;
    s.batchSize = b.batch;
    reasons.push(
      `${b.threads} of ${cores} physical cores — two are held back so the desktop keeps moving while the model runs.`,
    );
  }

  const quantKvOk = hw.backend !== undefined &&
    QUANT_KV_BACKENDS.includes(hw.backend);
  s.flashAttn = quantKvOk ? "on" : "auto";
  if (quantKvOk) {
    reasons.push(
      "Flash attention on — less KV memory, faster long contexts, and a prerequisite for a quantised cache.",
    );
  }
  if (!quantKvOk && placement !== "cpu" && hw.gpus.length > 0) {
    reasons.push(
      `A quantised KV cache is not offered for the ${
        hw.backend ?? "selected"
      } backend — it would halve the cache and buy context, but a cache the backend cannot read is a server that does not start.`,
    );
  }

  const ceiling = ctxOverride !== undefined
    ? Math.max(MIN_CTX, Math.min(ctxOverride, target))
    : target;

  // f16 first: a quantised cache costs a little quality, so it is only worth
  // taking when it buys context the full-precision cache could not reach.
  const f16 = bestCtx(meta, hw, s, placement, ceiling);
  let ctx = f16;
  if (quantKvOk && f16 < ceiling) {
    const q8s: Settings = { ...s, cacheTypeK: "q8_0", cacheTypeV: "q8_0" };
    const q8 = bestCtx(meta, hw, q8s, placement, ceiling);
    if (q8 > f16) {
      s.cacheTypeK = "q8_0";
      s.cacheTypeV = "q8_0";
      ctx = q8;
      reasons.push(
        `KV cache quantised to q8_0 — it halves the cache, and here that is what lifts the context from ${f16.toLocaleString()} to ${q8.toLocaleString()}.`,
      );
    }
  }

  const label = PLACEMENTS.find((p) => p.id === placement)?.label ?? placement;

  if (ctx === 0) {
    // Nothing fits. Return an honest attempt so the UI still has numbers to
    // show, and say plainly that this placement is not available here.
    const fallback = place(meta, hw, s, placement, MIN_CTX)?.settings ?? {
      ...s,
      ctxSize: MIN_CTX,
      ngl: placement === "cpu" ? 0 : 999,
    };
    const blocker = blockerFor(hw, placement);
    return {
      settings: fallback,
      reasons: [...reasons, `${label} is not possible here: ${blocker}`],
      ctx: 0,
      optimalCtx: target,
      possible: false,
      blocker,
      summary: `${label} — not possible for this model here`,
    };
  }

  const placed = place(meta, hw, s, placement, ctx);
  const settings = placed?.settings ?? { ...s, ctxSize: ctx };
  if (placed?.note) reasons.push(`${label}: ${placed.note}.`);
  reasons.push(
    ctx >= target
      ? `Context ${ctx.toLocaleString()} — the full length this model was trained for. More would need RoPE scaling and answer worse, not better.`
      : `Context ${ctx.toLocaleString()} of the ${target.toLocaleString()} this model was trained for — the most this placement can hold.`,
  );

  return {
    settings,
    reasons: finish(meta, hw, settings, reasons),
    ctx,
    optimalCtx: target,
    possible: true,
    blocker: "",
    summary: ctx >= target
      ? `${label} · full ${ctx.toLocaleString()} context`
      : `${label} · ${ctx.toLocaleString()} of ${target.toLocaleString()} context`,
  };
}

/** Why a placement is unavailable, in the user's terms. */
function blockerFor(hw: Hw, placement: Placement): string {
  const floor = MIN_CTX.toLocaleString();
  if (placement === "cpu") {
    return `Needs more RAM than is free, even at a ${floor}-token context.`;
  }
  if (hw.gpus.length === 0) return "No GPU was detected on this machine.";
  const total = hw.gpus.reduce((a, g) => a + g.vramTotalB, 0);
  if (placement === "vram") {
    return `The whole model does not fit in ${
      (total / 1024 ** 3).toFixed(1)
    } GB of VRAM, even at a ${floor}-token context. Hybrid will run it.`;
  }
  return `Even split across GPU and RAM this does not fit at a ${floor}-token context. Try a smaller quantisation.`;
}

/** Everything that follows from the final placement rather than driving it. */
function finish(
  meta: ModelMeta,
  hw: Hw,
  s: Settings,
  reasons: string[],
): string[] {
  const p = plan(meta, hw, s);
  const availB = hw.mem?.availableB ?? 0;
  const hostNeed = p.ram.usedB;
  const margin = ramMarginB(availB);
  if (availB > 0 && hostNeed > 0 && hostNeed < availB * 0.7) {
    s.mlock = true;
    reasons.push(
      "--mlock on: the host-side weights fit in free RAM with room to spare, so pinning them stops the OS paging the model out mid-generation.",
    );
  } else if (hostNeed > 0) {
    s.mlock = false;
  }
  if (p.ram.overB > 0) {
    reasons.push(
      `Warning: this still needs ${
        (p.ram.overB / 1024 ** 3).toFixed(1)
      } GB more RAM than is free, and the OS needs ${
        (margin / 1024 ** 3).toFixed(1)
      } GB of that. Close something, or pick a smaller quant.`,
    );
  } else if (availB > 0 && hostNeed + margin > availB) {
    reasons.push(
      `Warning: this leaves the OS under the ${
        (margin / 1024 ** 3).toFixed(1)
      } GB of RAM it needs — ${
        ((hostNeed + margin - availB) / 1024 ** 3).toFixed(1)
      } GB short. Close something before you start.`,
    );
  }
  return reasons;
}

/**
 * All three placements at once, so the UI can show what each would give.
 *
 * This is the "no need to look elsewhere" answer: one call, three comparable
 * outcomes, each with the context it reaches or the reason it cannot run.
 */
export function tuneAll(
  meta: ModelMeta,
  hw: Hw,
  base: Settings,
  ctxOverride?: number,
): Record<Placement, Tuning> {
  return {
    vram: tune(meta, hw, base, "vram", ctxOverride),
    hybrid: tune(meta, hw, base, "hybrid", ctxOverride),
    cpu: tune(meta, hw, base, "cpu", ctxOverride),
  };
}

/** The placement to offer when the user has expressed no preference: the
 *  fastest one that can actually run this model. */
export function bestPlacement(all: Record<Placement, Tuning>): Placement {
  if (all.vram.possible) return "vram";
  if (all.hybrid.possible) return "hybrid";
  return "cpu";
}
