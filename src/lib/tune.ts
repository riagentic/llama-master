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
 * The full advertised context — the hard ceiling for a pin and the Max band.
 *
 * This is what the header declares and what llama.cpp will accept as `-c`
 * without extrapolating. For a YaRN-stretched model it INCLUDES the stretch:
 * DeepSeek-V4-Flash advertises 1,048,576 and genuinely runs there (given the
 * memory). Showing anything smaller as "Max" contradicted the Models page,
 * which reads `nCtxTrain` directly — the same model must not have two maxima.
 */
export function trainedCtx(meta: ModelMeta): number {
  return meta.nCtxTrain > 0 ? meta.nCtxTrain : 4096;
}

/**
 * The largest context at which the model still performs the best — the
 * TUNER'S aim, not the user's ceiling (`trainedCtx` is that).
 *
 * `nCtxTrain` is what the file ADVERTISES, and for a RoPE-scaled model that is
 * an extrapolation, not a measurement: DeepSeek-V4-Flash declares 1,048,576
 * over an `original_context_length` of 65,536 — a 16x YaRN stretch. Aiming the
 * AUTO-tuner at the stretched figure is how it ended up proposing a context
 * whose compute buffer alone was 68 GiB. The native length is a fact in the
 * same header, and it is the honest automatic target; the user can still pin
 * anything up to `trainedCtx` themselves.
 */
export function optimalCtx(meta: ModelMeta): number {
  const advertised = trainedCtx(meta);
  const native = meta.nCtxOrig;
  return native > 0 && native < advertised ? native : advertised;
}

/** Below this a context is too short to hold a conversation, so a placement
 *  that cannot reach it is reported impossible rather than proposed. */
export const MIN_CTX = 2048;

/** The four context sizes worth naming, in order. */
export type CtxBandId = "min" | "opt" | "big" | "max";

export type CtxBands = Record<CtxBandId, number>;

/**
 * The named bands of a model's usable context range.
 *
 * ONLY `max` is read from the model. `nCtxTrain` is the length it was actually
 * trained for; past that RoPE extrapolates and the output degrades hard, so it is
 * the real outer edge and the honest answer to "the most it can handle".
 *
 * `opt` and `big` are ESTIMATES and the UI says so. A GGUF header carries no
 * quality signal — no eval scores, nothing about where attention starts to
 * thin — while published long-context measurements (needle-in-a-haystack and
 * RULER-style suites) consistently find effective length well under the
 * advertised one. So `big` sits at half the trained length and `opt` at a
 * quarter: defensible places to look, not measurements. The only way to turn
 * them into facts is to probe THIS model at THIS quantisation, which the app is
 * equipped to do (it owns a server and a chat client) and does not do yet.
 *
 * `min` is a usability floor rather than a model property: any autoregressive
 * model "works" at one token, and what a user needs is room for a system prompt
 * and a few turns.
 *
 * Monotonic and clamped by construction, so a 512-token model still yields four
 * ordered values rather than a scrambled range.
 */
export function ctxBands(meta: ModelMeta): CtxBands {
  const max = trainedCtx(meta);
  const native = optimalCtx(meta);
  const step = (n: number) =>
    Math.max(CTX_STEP, Math.floor(n / CTX_STEP) * CTX_STEP);
  const min = Math.min(step(Math.min(4096, max)), max);
  // For a YaRN-stretched model the native pre-stretch length is the one
  // quality fact the header carries, so it anchors the bands: Big IS the
  // native length and Opt sits at half of it. Everything between Big and Max
  // is the stretched range — real, but bought with retrieval quality.
  const stretched = native < max;
  const big = Math.min(Math.max(step(stretched ? native : max / 2), min), max);
  const opt = Math.min(
    Math.max(step(stretched ? native / 2 : max / 4), min),
    big,
  );
  return { min, opt, big, max };
}

/** Label and rationale per band, so the buttons and the range visual cannot
 *  describe the same number two different ways. `estimated` drives the honesty
 *  marker in the UI. */
export const CTX_BANDS: readonly {
  id: CtxBandId;
  label: string;
  estimated: boolean;
  tip: string;
}[] = [
  {
    id: "min",
    label: "Min",
    estimated: true,
    tip:
      "The smallest context worth running: room for a system prompt and a few turns. A usability floor, not a limit of the model.",
  },
  {
    id: "opt",
    label: "Opt",
    estimated: true,
    tip:
      "Where the model should still answer at full quality. ESTIMATED at a quarter of its trained length — published long-context measurements consistently find effective length well below the advertised one, but nothing in a GGUF header says where it is for this model.",
  },
  {
    id: "big",
    label: "Big",
    estimated: true,
    tip:
      "Long, with some quality given up. ESTIMATED at half the trained length — or, for a YaRN-stretched model, its native pre-stretch length, the one quality fact the header carries.",
  },
  {
    id: "max",
    label: "Max",
    estimated: false,
    tip:
      "The full advertised context, read from the model, not estimated. For a YaRN-stretched model this includes the stretch: it genuinely runs there, at a real cost in memory and some retrieval quality past its native length.",
  },
];

/**
 * What a user-pinned context actually becomes.
 *
 * One rule, one home: the tuner clamps a pin to `[MIN_CTX, target]`, and the UI
 * has to display the same number or it is promising something that will not run.
 * It lived in both places once, and the UI copy had dropped the floor — a pin of
 * 512 rendered as 512 and ran as 2048.
 *
 * The target wins over the floor: on a model trained for 512 tokens the floor
 * must not round a pin up past what the model can actually attend over — the
 * one rule `bestCtx` exists to keep.
 */
export function pinnedCtx(override: number, target: number): number {
  return Math.min(Math.max(MIN_CTX, override), target || Infinity);
}

/**
 * The context sizes worth one click.
 *
 * Powers of two up to 128k, because that is how every model and every
 * published benchmark describes its context — then 128k steps to the top,
 * because on a machine that genuinely holds long contexts the gap from 256k
 * straight to 512k skips exactly the sizes worth trying (a user watched 256k
 * work and 512k die, with nothing offered between). The KV cache grows with
 * every rung, so the ladder is also the cost ladder. A preset above what the
 * model was trained for is offered but disabled rather than hidden: "1M is
 * possible, not for THIS model" is information, and a row that changes length
 * per model is harder to use than one that does not.
 */
export const CTX_PRESETS: readonly number[] = [
  16_384,
  32_768,
  65_536,
  131_072,
  262_144,
  393_216,
  524_288,
  655_360,
  786_432,
  917_504,
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

/**
 * VRAM we refuse to plan into: drivers, the desktop, and fragmentation.
 *
 * This was briefly widened by the machine's observed memory "churn", to reserve
 * more on a busy workstation. That is removed, and the reason is worth keeping:
 * the only churn signal available is the DEVICE-WIDE usage series, and our own
 * llama-server is inside it — so starting a 39 GB model registered as 39 GB of
 * volatility, inflated the reserve, and made the app report "will not fit" for
 * models that fit comfortably. A reserve driven by a signal that cannot separate
 * our own allocation from everyone else's produces false refusals, which is worse
 * than a reserve that is merely fixed.
 *
 * Adapting to memory that moves is still done, by the two mechanisms that CAN be
 * measured honestly: the auto-tune re-runs when real headroom changes
 * (`src/lib/adapt.ts:headroomKey`), and a running model that gets squeezed or
 * given room says so (`drift`).
 */
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

/** Bytes as GB, for the sentences the tuner writes. `src/lib/format.ts` is the
 *  UI's formatter; this stays here so `src/lib` needs no cross-import. */
function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/**
 * Does this placement fit the GPUs — all of them, and each of them?
 *
 * Two questions, and only the first used to be asked. The aggregate says the
 * machine has the VRAM; `p.devices.fits` says there is a way to cut the layers
 * so that no single card is asked for more than it holds. On one GPU they are
 * the same question. On two they are not, and the difference is a plan that
 * passes every check here and dies with `cudaMalloc failed` on device 1 —
 * because llama.cpp divides the offloaded layers by COUNT while `--n-cpu-moe`
 * makes the last ones an order of magnitude heavier (`src/lib/devsplit.ts`).
 */
function fitsVram(meta: ModelMeta, hw: Hw, s: Settings): boolean {
  const p = plan(meta, hw, s);
  const total = hw.gpus.reduce((a, g) => a + g.vramTotalB, 0);
  return p.vram.overB === 0 && p.vram.freeB >= marginB(total) &&
    p.devices.fits;
}

/**
 * Does the host side of this placement leave the OS room to breathe?
 *
 * With no memory reading at all the answer is NO, not "sure, why not". This used
 * to return true — inventing no limit — which sounds humble and is the opposite:
 * during the boot window, before the first `hw.refresh` lands, it let the tuner
 * hand back a 78 GB plan as optimal on a machine whose size it had not read yet.
 * Refusing is recoverable (the next poll is a second away and re-tunes);
 * proposing a plan the machine cannot hold is an OOM kill.
 */
function fitsRam(meta: ModelMeta, hw: Hw, s: Settings): boolean {
  const availB = hw.mem?.availableB ?? 0;
  if (availB === 0) return false;
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
  /** Drop the residency anchor: grow the context even when that pushes
   *  weights back to the host. The "Max on Hybrid" contract — the user has
   *  said context outranks speed, which is the one trade the automatic
   *  search exists to refuse. A pinned 262,144 ran fine on hybrid while the
   *  button offered 17,920, because the pin skipped the anchor and the
   *  button did not: same machine, two answers, and the button's was the
   *  wrong one for the question it claims to answer. */
  keepResidency = true,
): number {
  // RESIDENCY FIRST. The old search asked "what is the largest context that
  // fits?" and answered it by moving work to the host — on a model whose
  // compute scratch grows with the context that is a trade it will always take,
  // and it settled on one layer on the GPU at a 654,848 context: fits, and runs
  // at CPU speed. Speed is the reason a GPU is here at all, so the placement is
  // fixed first, at the shortest useful context, and the context then grows only
  // as far as it can WITHOUT pushing another byte of weight back to the host.
  //
  // Host WEIGHTS, not host bytes: the KV cache grows with the context by
  // definition and on a CPU placement everything is host-side, so anchoring on
  // total host bytes would pin every placement to its floor.
  const anchorAt = Math.min(MIN_CTX, ceiling);
  const hostWeightsB = (c: number): number => {
    const p = place(meta, hw, base, placement, c);
    if (!p) return Infinity;
    return plan(meta, hw, p.settings).ram.buckets
      .filter((b) => b.key === "weights")
      .reduce((a, b) => a + b.bytes, 0);
  };
  const anchor = hostWeightsB(anchorAt);
  const keepsResidency = (c: number): boolean =>
    place(meta, hw, base, placement, c) !== null &&
    hostWeightsB(c) <= anchor;
  const placeable = (c: number): boolean =>
    place(meta, hw, base, placement, c) !== null;
  return searchCtx(ceiling, keepResidency ? keepsResidency : placeable);
}

/** Largest value on the grid, at most `ceiling`, that satisfies `ok`. */
function searchCtx(ceiling: number, ok: (c: number) => boolean): number {
  // The model's ceiling wins over our usability floor. A model trained for 128
  // tokens cannot be handed 2,048 just because that is the shortest context we
  // would normally propose — that would break the one rule this whole search
  // exists to keep, which is never to exceed what the model was trained for.
  const floor = Math.min(MIN_CTX, ceiling);
  const rounded = Math.floor(ceiling / CTX_STEP) * CTX_STEP;
  const top = Math.max(floor, rounded);
  if (ok(top)) return top;
  if (!ok(floor)) return 0;
  // Invariant: lo fits, hi does not.
  let lo = floor;
  let hi = top;
  while (hi - lo > CTX_STEP) {
    const mid = Math.floor((lo + hi) / 2 / CTX_STEP) * CTX_STEP;
    if (mid <= lo || mid >= hi) break;
    if (ok(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * Optimal settings for this model, on this machine, in this placement.
 *
 * `base` supplies everything the tuner does not decide (port, sampling).
 *
 * `ctxOverride` is a value the user typed, and it is an INSTRUCTION: the
 * context is held exactly there (clamped only to the advertised maximum) and
 * everything else is arranged around it. When it does not fit, the answer is
 * `possible: false` with the shortfall named and a plan that SHOWS the
 * overflow — never a silently smaller number. It used to be a search ceiling
 * instead, so pinning 1,048,576 quietly started 17,920, and the projection
 * never showed what the megabyte question actually cost.
 *
 * `measuredCtx` is different in kind: the largest context this model has
 * actually run at on this machine (`cfg.fitCtx`). It is a SEARCH ceiling for
 * the automatic path — the machine may hold less today than it did then, so
 * the tuner may settle lower, it just never aims higher only to walk the
 * retry ladder back down.
 */
export function tune(
  meta: ModelMeta,
  hw: Hw,
  base: Settings,
  placement: Placement = "vram",
  ctxOverride?: number,
  measuredCtx?: number,
  /** Hunt the largest context this placement can hold, all the way to the
   *  ADVERTISED maximum — past the native-first aim and past the measured
   *  ceiling. The "Max on VRAM / Max on Hybrid" gesture: the user has said
   *  context is the priority, so the memory search goes to the model's edge
   *  and the measured-boundary warning covers what arithmetic cannot see. */
  aimFull = false,
): Tuning {
  const d = defaults();
  const s: Settings = { ...d, ...base };
  const reasons: string[] = [];
  const target = optimalCtx(meta);

  // The cache type is the tuner's to decide, so start from llama.cpp's default
  // rather than from whatever the last model happened to need. Same for the
  // tensor split: it is derived from THIS model's layer sizes, and one carried
  // over from another model is a wrong answer that looks deliberate.
  s.cacheTypeK = str(d, "cacheTypeK");
  s.cacheTypeV = str(d, "cacheTypeV");
  s.tensorSplit = str(d, "tensorSplit");
  s.noMmap = false;

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

  // Speculative decoding off the model's own MTP block. Taken whenever the model
  // ships one, because it is the rare optimisation with no trade to weigh: the
  // full model verifies every drafted token, so a rejected draft is discarded and
  // the output is exactly what it would have been. Only the speed changes. The
  // weights are in the file and loaded either way — leaving it off pays for them
  // and gets nothing.
  //
  // Never set for a model without the block: llama.cpp asserts on
  // `n_layer_nextn > 0` and refuses to load, which would be "optimal settings"
  // that do not start.
  s.specType = meta.nextnLayers > 0 ? "draft-mtp" : "";
  if (meta.nextnLayers > 0) {
    reasons.push(
      `Speculative decoding on — this model ships ${meta.nextnLayers} multi-token-prediction block${
        meta.nextnLayers === 1 ? "" : "s"
      }, so it drafts ahead and verifies against itself. Output is identical; only the speed changes.`,
    );
  }

  const quantKvOk = hw.backend !== undefined &&
    QUANT_KV_BACKENDS.includes(hw.backend);
  s.flashAttn = quantKvOk ? "on" : "auto";
  if (quantKvOk) {
    reasons.push(
      // Deliberately does NOT claim "less KV memory": the cache is the same size
      // either way, and `plan.ts` has no flash-attention term, so a bar on screen
      // would not move. What it actually buys is smaller attention scratch and
      // the ability to quantise the cache at all.
      "Flash attention on — smaller attention buffers, faster long contexts, and a prerequisite for a quantised cache.",
    );
  }
  if (!quantKvOk && placement !== "cpu" && hw.gpus.length > 0) {
    reasons.push(
      `A quantised KV cache is not offered for the ${
        hw.backend ?? "selected"
      } backend — it would halve the cache and buy context, but a cache the backend cannot read is a server that does not start.`,
    );
  }

  const pinned = ctxOverride !== undefined && ctxOverride > 0;
  // A pin clamps to the ADVERTISED maximum, not the tuner's native-first aim:
  // the UI offers Max = trainedCtx, and a pin of 1,048,576 silently searched
  // under a 65,536 ceiling was two lies at once.
  const ceiling = pinned
    ? pinnedCtx(ctxOverride, trainedCtx(meta))
    : aimFull
    ? trainedCtx(meta)
    : Math.min(
      target,
      measuredCtx !== undefined && measuredCtx > 0 ? measuredCtx : target,
    );
  if (!pinned && !aimFull && ceiling < target) {
    reasons.push(
      `Context search capped at ${ceiling.toLocaleString()} — the largest this model has actually started at on this machine. Pin a size to try beyond it.`,
    );
  }
  /** A pinned context either places at exactly the pin, or not at all. */
  const fitAt = (t: Settings, c: number): number =>
    place(meta, hw, t, placement, c) !== null ? c : 0;

  // Host-side bytes for a candidate. Every one of them crosses the PCIe bus on
  // every token, so at equal context the candidate with fewer is the faster one.
  // This is the honest measure because it covers all three ways a full-precision
  // cache pushes work off the card — whole layers, routed experts, and the cache
  // itself — where a layer count sees only the first. Weights and KV only: the
  // compute bucket has a ≥32 MB floor on every plan, and counting it made
  // "spilling" vacuously true for any GPU placement.
  const hostB = (t: Settings, c: number): number => {
    const p = place(meta, hw, t, placement, c);
    if (!p) return Infinity;
    return plan(meta, hw, p.settings).ram.buckets
      .filter((b) => b.key !== "compute")
      .reduce((a, b) => a + b.bytes, 0);
  };
  const usesGpu = placement !== "cpu" && hw.gpus.length > 0;

  // f16 first: a quantised cache costs a little quality, so it has to buy
  // something. It buys one of two things — context the full cache could not
  // reach, or residency the full cache spent on the host. The second used not to
  // count, and it can be the bigger prize: a long-context MoE that reaches its
  // full trained length either way can still be paying for it with several GB of
  // experts and cache in system RAM, which q8_0 brings back onto the card.
  const f16 = pinned
    ? fitAt(s, ceiling)
    : bestCtx(meta, hw, s, placement, ceiling, !aimFull);
  let ctx = f16;
  const f16Host = f16 > 0 ? hostB(s, f16) : 0;
  const spilling = usesGpu && f16 > 0 && f16Host > 0;
  if (quantKvOk && (f16 < ceiling || spilling)) {
    const q8s: Settings = { ...s, cacheTypeK: "q8_0", cacheTypeV: "q8_0" };
    const q8 = pinned
      ? fitAt(q8s, ceiling)
      : bestCtx(meta, hw, q8s, placement, ceiling, !aimFull);
    const q8Host = q8 > 0 ? hostB(q8s, q8) : Infinity;
    const buysContext = q8 > f16;
    // The gain has to be worth the quality cost — and worth a sentence. A
    // sub-256 MB delta would flip the cache to q8_0 with a reason that reads
    // "brings 0.0 GB back onto the GPU".
    const buysResidency = spilling && q8 === f16 &&
      f16Host - q8Host >= 256 * 1024 * 1024;
    if (buysContext || buysResidency) {
      s.cacheTypeK = "q8_0";
      s.cacheTypeV = "q8_0";
      ctx = q8;
      reasons.push(
        buysContext
          ? `KV cache quantised to q8_0 — it halves the cache, and here that is what lifts the context from ${f16.toLocaleString()} to ${q8.toLocaleString()}.`
          : `KV cache quantised to q8_0 — it halves the cache, and here that is what brings ${
            gb(f16Host - q8Host)
          } back onto the GPU that would otherwise have run from system RAM, at the same ${q8.toLocaleString()} context.`,
      );
    }
  }

  const label = PLACEMENTS.find((p) => p.id === placement)?.label ?? placement;

  if (ctx === 0) {
    // Nothing fits. Return an honest attempt so the UI still has numbers to
    // show, and say plainly that this placement is not available here.
    //
    // A PINNED context keeps the pin in the returned settings: the projection
    // is computed from them, and the whole point of pinning 1M on a machine
    // that cannot hold it is to SEE what is missing — a fallback quietly reset
    // to 2,048 drew a fitting plan and left "so what memory is missing?"
    // unanswerable.
    const floor = pinned ? ceiling : Math.min(MIN_CTX, ceiling);
    // For a pinned refusal, keep the PLACEMENT's shape: take the arrangement
    // this placement reaches at a small context and hold the pin in it. The
    // old fallback (`ngl: 999`, experts on GPU) priced a hybrid refusal as if
    // the whole model sat in VRAM and told the user "needs 142 GB more VRAM"
    // for a placement whose weights live mostly in RAM — a shortfall nobody
    // could act on.
    const shape = pinned
      ? place(meta, hw, s, placement, Math.min(MIN_CTX, ceiling))?.settings
      : place(meta, hw, s, placement, floor)?.settings;
    const fallback = shape ? { ...shape, ctxSize: floor } : {
      ...s,
      ctxSize: floor,
      ngl: placement === "cpu" ? 0 : 999,
    };
    const blocker = pinned
      ? pinBlocker(meta, hw, fallback, ceiling)
      : blockerFor(meta, hw, placement);
    return {
      settings: fallback,
      reasons: [...reasons, `${label} is not possible here: ${blocker}`],
      ctx: 0,
      optimalCtx: target,
      possible: false,
      blocker,
      summary: pinned
        ? `${label} — does not fit at the pinned ${ceiling.toLocaleString()}`
        : `${label} — not possible for this model here`,
    };
  }

  const placed = place(meta, hw, s, placement, ctx);
  const settings = placed?.settings ?? { ...s, ctxSize: ctx };
  if (placed?.note) reasons.push(`${label}: ${placed.note}.`);
  if (aimFull) {
    reasons.push(
      "Context first, by request: the search was allowed to move weights into system RAM to buy length, which the automatic path never does. The projected speed shows the price.",
    );
  }
  reasons.push(
    pinned
      ? `Context ${ctx.toLocaleString()} — pinned by you; the tuner arranges everything else around it.`
      : ctx >= target
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
    summary: pinned
      ? `${label} · pinned ${ctx.toLocaleString()} context`
      : ctx >= target
      ? `${label} · full ${ctx.toLocaleString()} context`
      : `${label} · ${ctx.toLocaleString()} of ${target.toLocaleString()} context`,
  };
}

/** Why a placement is unavailable, in the user's terms. */
/**
 * Why a PINNED context cannot run, with the missing bytes named.
 *
 * The pin is an instruction, so the answer is not "pick a smaller model" — it
 * is exactly how far short this machine falls at the size the user asked for,
 * and the two ways out: a smaller pin, or Auto.
 */
function pinBlocker(
  meta: ModelMeta,
  hw: Hw,
  settings: Settings,
  pin: number,
): string {
  const p = plan(meta, hw, settings);
  const missing = [
    p.vram.overB > 0 ? `${gb(p.vram.overB)} more VRAM` : "",
    p.ram.overB > 0 ? `${gb(p.ram.overB)} more RAM` : "",
  ].filter(Boolean).join(" and ");
  const detail = missing ||
    "no arrangement of the cards can hold its layers";
  return `does not fit at the pinned ${pin.toLocaleString()} tokens — needs ${detail}. Lower the pin, or press Auto to let the tuner find the largest that fits.`;
}

function blockerFor(
  meta: ModelMeta,
  hw: Hw,
  placement: Placement,
): string {
  const floor = MIN_CTX.toLocaleString();
  // Distinguish "does not fit" from "nothing has been measured yet". The second
  // is a half-second condition at boot, and reporting it as the first sent the
  // user off to close browser tabs over a number that had not been read.
  if ((hw.mem?.availableB ?? 0) === 0) {
    return "The memory reading has not arrived yet — this will settle on the next sample.";
  }
  if (hw.gpus.length === 0 && placement !== "cpu") {
    return "No GPU was detected on this machine.";
  }

  // Name the constraint that actually binds, with the numbers behind it.
  //
  // This used to say "does not fit in 47.8 GB of VRAM" — the card's CAPACITY —
  // when the real reason was that only 6 GB of it was free. That reads as a claim
  // about the model, sends the user looking for a smaller quantisation, and is
  // simply not what happened. What they need to know is how much room there is
  // and who has the rest.
  // Plan the placement being explained: a CPU blocker computed from a
  // full-offload plan puts every weight in the VRAM pool and then reports
  // "needs 0.0 GB of RAM" while refusing — a self-contradiction.
  const at = plan(meta, hw, {
    ...defaults(),
    ctxSize: MIN_CTX,
    ngl: placement === "cpu" ? 0 : 999,
    nCpuMoe: 0,
  });
  const gb = (n: number) => `${(n / 1024 ** 3).toFixed(1)} GB`;
  const need = (pool: "vram" | "ram") =>
    `${gb(at[pool].usedB)} at a ${floor}-token context`;
  // Room available to US: capacity minus what everything else holds. NOT
  // `freeB + usedB`, which is incoherent once the plan overflows.
  const have = (pool: "vram" | "ram") => {
    const room = Math.max(0, at[pool].capacityB - at[pool].otherB);
    return at[pool].otherB > 0
      ? `${gb(room)} available of ${
        gb(at[pool].capacityB)
      } (other processes hold ${gb(at[pool].otherB)})`
      : `${gb(room)}`;
  };

  if (placement === "cpu") {
    return `Needs ${need("ram")}, and there is ${have("ram")} of system RAM.`;
  }
  if (placement === "vram") {
    return `Every layer on the GPU needs ${need("vram")}, and there is ${
      have("vram")
    }. Hybrid will run it.`;
  }
  return `Even split across GPU and RAM this does not fit at a ${floor}-token context: ${
    have("vram")
  } of VRAM and ${have("ram")} of RAM. Try a smaller quantisation.`;
}

/** Everything that follows from the final placement rather than driving it. */
function finish(
  meta: ModelMeta,
  hw: Hw,
  s: Settings,
  reasons: string[],
): string[] {
  const p = plan(meta, hw, s);

  // Pin the layer-to-card division rather than leaving it to llama.cpp's
  // default. The default divides by each card's free memory but applies the
  // result to the layer COUNT, which is only the same thing when every layer
  // weighs the same — and with `--n-cpu-moe` holding the first N layers'
  // experts in RAM, the last layers are ~20x the first. That put 34 GB of a
  // 38 GB plan on one 24 GB card. This is the split that was actually planned,
  // so it is also the one the memory bars are drawing.
  if (p.devices.tensorSplit) {
    s.tensorSplit = p.devices.tensorSplit;
    reasons.push(
      `Layers split ${
        p.devices.bytesB.map((b) => gb(b)).join(" / ")
      } across the cards (-ts ${p.devices.tensorSplit}) — llama.cpp divides them by count, and with the experts held back the last layers are far heavier, so left to itself it would overfill one card.`,
    );
  }

  // llama.cpp's own advice for this configuration: "tensor overrides to CPU are
  // used with mmap enabled - consider using --no-mmap for better performance".
  // How the weights get into memory. ONE choice, not two flags.
  //
  // `--mlock` and `--no-mmap` are the same setting in llama.cpp: both assign
  // `params.load_mode`, so emitting them together is not "locked and unmapped",
  // it is whichever came last silently winning (`common/arg.cpp`). The app would
  // have printed a reason claiming --mlock while shipping an argv that cancelled
  // it — precisely the "what you see is what runs" promise, broken.
  //
  // So they are ranked. `mlock` mode still memory-maps (`use_mmap = MMAP ||
  // MLOCK`, llama-model-loader.cpp:545), which is what llama.cpp warns about the
  // moment any tensor is overridden to the CPU: "tensor overrides to CPU are
  // used with mmap enabled - consider using --no-mmap for better performance".
  // With the routed experts on the host that warning is about the bytes crossing
  // the bus on every token, so it wins — and it is only taken when the host side
  // has real room, because unmapped weights are anonymous pages with no file to
  // fall back to.
  const availB = hw.mem?.availableB ?? 0;
  const hostNeed = p.ram.usedB;
  const margin = ramMarginB(availB);
  const roomToSpare = availB > 0 && hostNeed > 0 && hostNeed < availB * 0.7;

  if (roomToSpare && num(s, "nCpuMoe") > 0 && p.ram.overB === 0) {
    // MEASURED, not reasoned from folklore, on the 145 GB DeepSeek-V4 with 33
    // layers of experts on the host: `--no-mmap` copies the whole file on
    // every start — 148 s cold and 160 s even with a warm page cache, because
    // its own anonymous copy evicts the cache it would have used. Mapped, the
    // same start is 73 s cold and 6 s warm, and generation is no slower —
    // faster, in fact: 9.6 tok/s mapped against 8.9 with `--no-mmap`, same
    // build, same card — and every fit-ladder rung reloads the model, so
    // this is the difference between a retry that stings and one that does
    // not. `--mlock` is not the answer either: it would try to pin the whole
    // expert set, stock Linux caps RLIMIT_MEMLOCK far below a model this size
    // (23 GB on the machine that motivated this), and llama.cpp would warn
    // and continue unpinned — a flag whose stated effect does not happen.
    // With `roomToSpare` guarding this branch, the page cache keeps the hot
    // expert pages by itself. So: llama.cpp's default, and no flag at all.
    s.noMmap = false;
    s.mlock = false;
    reasons.push(
      "Memory-mapped (llama.cpp's default): the routed experts run from the page cache, so a warm restart re-reads nothing — measured 6 s instead of 160 s on a 145 GB model, and every automatic retry reloads the model. --no-mmap would copy the whole file on every start, and --mlock would ask to pin more than stock memlock limits allow, so llama.cpp would warn and run unpinned anyway.",
    );
  } else if (roomToSpare) {
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
  measuredCtx?: number,
): Record<Placement, Tuning> {
  return {
    vram: tune(meta, hw, base, "vram", ctxOverride, measuredCtx),
    hybrid: tune(meta, hw, base, "hybrid", ctxOverride, measuredCtx),
    cpu: tune(meta, hw, base, "cpu", ctxOverride, measuredCtx),
  };
}

/** The placement to offer when the user has expressed no preference: the
 *  fastest one that can actually run this model. */
export function bestPlacement(all: Record<Placement, Tuning>): Placement {
  if (all.vram.possible) return "vram";
  if (all.hybrid.possible) return "hybrid";
  return "cpu";
}
