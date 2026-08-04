// src/lib/plan.ts — where does this model actually go?
//
// The one function behind the VRAM/RAM bars. Pure, synchronous and cheap, so it
// re-runs on every keystroke in the settings panel and the bars move while you
// drag a slider — no round trip, no "apply" button.
//
// It is arithmetic over facts, not a guess: weights come from the exact tensor
// bytes in the GGUF header (per layer, experts separated), and the KV cache from
// the model's own head geometry. The only estimated term is the compute buffer,
// which is labelled as such everywhere it is shown.

import type { Gpu, Hw, ModelMeta, Settings } from "./types.ts";
import { bool, num, str } from "./params.ts";
import {
  deviceBudgets,
  loadPerDevice,
  offloadRange,
  packSlots,
  slotOnGpu,
  tensorSplitValue,
} from "./devsplit.ts";
import { reserveLabel, reserveOf, vramReserveShares } from "./reserve.ts";

/** Bytes per cached element, by `-ctk`/`-ctv` value. Block quants carry their
 *  scales, hence the fractional sizes (q8_0 = 34 bytes per 32 elements). */
const CACHE_BYTES: Record<string, number> = {
  f32: 4,
  f16: 2,
  bf16: 2,
  q8_0: 34 / 32,
  q5_1: 24 / 32,
  q5_0: 22 / 32,
  q4_1: 20 / 32,
  q4_0: 18 / 32,
};

/** Backend context + kernels a GPU pays for merely by being used. Measured
 *  around 250-450 MB for CUDA; the mid-point is the honest planning number. */
const BACKEND_CONTEXT_B = 350 * 1024 * 1024;

/**
 * Is speculative decoding actually going to run?
 *
 * Both halves matter: the flag has to be set AND the model has to ship the block
 * it names. `--spec-type draft-mtp` against a model with no MTP block is not a
 * slow server, it is `GGML_ASSERT(hparams.n_layer_nextn > 0)` and a refusal to
 * load — so nothing in this app may emit it on a guess.
 */
export function specMtpActive(meta: ModelMeta, s: Settings): boolean {
  return meta.nextnLayers > 0 && str(s, "specType") === "draft-mtp";
}

export type BucketKey = "weights" | "experts" | "kv" | "compute" | "other";

export type Bucket = {
  key: BucketKey;
  label: string;
  bytes: number;
};

export type Pool = {
  label: string;
  capacityB: number;
  buckets: Bucket[];
  /** Everything llama.cpp will claim in this pool. */
  usedB: number;
  /** Already spoken for by other processes (idle VRAM use, other apps' RAM). */
  otherB: number;
  /** Held back by the user for their own work (`src/lib/reserve.ts`). Kept
   *  apart from `otherB` because it is a CHOICE, not a measurement: when a plan
   *  does not fit, "you reserved 4 GB" is actionable and "something else holds
   *  4 GB" sends the user hunting for a process that does not exist. */
  reservedB: number;
  freeB: number;
  /** Bytes past capacity — the amount the user has to claw back. 0 when it fits. */
  overB: number;
};

export type Plan = {
  nLayer: number;
  layersOnGpu: number;
  /** Layer indices whose experts stay in RAM (`--n-cpu-moe`). */
  moeOnCpu: number;
  ctx: number;
  vram: Pool;
  ram: Pool;
  /** KV bytes for one token across every layer — the "cost per 1k tokens" line. */
  kvPerTokenB: number;
  kvTotalB: number;
  fits: boolean;
  /** How the offloaded slots divide across the cards — the answer to "will one
   *  of them be asked for more than it has", which the aggregate cannot give.
   *  Empty on a single-GPU or CPU-only plan. */
  devices: DevicePlan;
  /** Plain-language observations, ordered most important first. */
  notes: string[];
};

/** Per-card placement, and the `-ts` that pins it. */
export type DevicePlan = {
  /** Bytes each card holds, in `hw.gpus` order. */
  bytesB: number[];
  /** Room each card had for them. */
  budgetsB: number[];
  /** The `-ts` value that produces this placement, or "" when none is needed. */
  tensorSplit: string;
  /** False when no contiguous division of the layers fits the cards, however
   *  well the totals add up. */
  fits: boolean;
  /**
   * What each CARD is asked to hold, for the picture. A second GPU is not a
   * bigger GPU, and one pooled VRAM bar hid exactly the question a two-card
   * machine asks: which card is full, with what. Weights come from the slot
   * packing; the KV cache follows its layers (apportioned by each card's
   * share of the offloaded slots); compute scratch is per device. When the
   * packing fails, this is the best-effort fill — cards to their budgets, in
   * order — so the picture still shows how far the model got.
   */
  cards: {
    name: string;
    capacityB: number;
    /** Already in use by everything else, measured now. */
    otherB: number;
    /** This card's share of the user's VRAM reserve — the per-GPU figure, plus
     *  the connected figure when a display hangs off this card
     *  (`src/lib/reserve.ts:vramReserveShares`). */
    reservedB: number;
    weightsB: number;
    kvB: number;
    computeB: number;
    /** Past this card's capacity, after everything above. */
    overB: number;
  }[];
  /** Layer bytes no card could take at all — the part of the model with
   *  nowhere to go, distinct from any single card running over. */
  unplacedB: number;
};

const MB = 1024 * 1024;

/**
 * A byte count that can be reasoned about: finite and not negative.
 *
 * Header fields come from a file this app did not write. A truncated or hostile
 * GGUF can yield NaN or a negative, and one such value poisons every total it
 * touches — silently, because arithmetic does not complain and NaN comparisons
 * are all false. Clamping here keeps "we could not read this model" looking like
 * zero rather than like a plan.
 */
export function whole(n: number): number {
  return Number.isFinite(n) && n > 0 ? n : 0;
}

function cacheBytes(type: string): number {
  return CACHE_BYTES[type] ?? 2;
}

/**
 * KV bytes per token for ONE full-attention layer.
 *
 * Exported for the "per 1k tokens" figure in the UI, which is a per-token rate
 * and so cannot express a sliding window (a windowed layer stops growing). Use
 * `kvTotal` for the number that decides whether a model fits.
 */
export function kvPerToken(meta: ModelMeta, s: Settings): number {
  const bk = cacheBytes(str(s, "cacheTypeK"));
  const bv = cacheBytes(str(s, "cacheTypeV"));
  // head_dim falls back to n_embd / n_head when the model omits key_length —
  // and `0 / 0` is NaN, which is why every value below is passed through
  // `whole`. A header this app cannot make sense of must produce a plan that
  // says "nothing", not one that says "NaN GB" and defeats every fit check
  // downstream (NaN comparisons are all false, so `overB === 0` and
  // `freeB >= margin` both quietly stop meaning anything).
  const headDim = meta.nHead > 0 ? whole(meta.nEmbd) / meta.nHead : 0;
  const kLen = whole(meta.keyLength) || headDim;
  const vLen = whole(meta.valueLength) || headDim;
  const heads = whole(meta.nHeadKv) || whole(meta.nHead);
  // MLA (DeepSeek-V2/V3) caches one compressed latent per token per layer
  // instead of one entry per head: the rank plus the 64-wide RoPE part. Billing
  // it as 128 heads x (192 + 128) overstates V3's cache by about seventy times.
  if (meta.kvLoraRank > 0) {
    return whole(meta.nLayer * (meta.kvLoraRank + MLA_ROPE_DIM) * bk);
  }
  return whole(meta.nLayer * heads * (kLen * bk + vLen * bv));
}

/** The RoPE-carrying part of an MLA cache entry, fixed by the architecture. */
const MLA_ROPE_DIM = 64;

/**
 * How many of a model's layers are windowed, and how many see the whole
 * context.
 *
 * Gemma-3 declares a 1024-token window with a pattern of 6: five local layers
 * then one global, repeating. A local layer's cache stops growing at the window,
 * so at a 32k context it holds 1/32 of what the formula above assumes.
 */
export function swaSplit(meta: ModelMeta): { full: number; windowed: number } {
  if (meta.swaWindow <= 0 || meta.nLayer <= 0) {
    return { full: meta.nLayer, windowed: 0 };
  }
  const period = Math.max(1, meta.swaPattern || 1);
  // One full-attention layer per period; with period 1 every layer is local.
  const full = period <= 1 ? 0 : Math.ceil(meta.nLayer / period);
  return { full, windowed: meta.nLayer - full };
}

/**
 * Total KV-cache bytes at this context — the number that decides the fit.
 *
 * Uniform for most models; for a sliding-window model the windowed layers are
 * capped at the window, which is the difference between "this fits" and "this
 * needs 3.7x the VRAM it actually does".
 */
export function kvTotal(meta: ModelMeta, s: Settings, ctx: number): number {
  const perLayer = meta.nLayer > 0 ? kvPerToken(meta, s) / meta.nLayer : 0;
  const { full, windowed } = swaSplit(meta);
  if (windowed === 0) return perLayer * meta.nLayer * ctx;
  // A windowed layer still has to hold the current batch alongside its window.
  const windowTokens = Math.min(
    ctx,
    meta.swaWindow + Math.min(ctx, num(s, "batchSize")),
  );
  return perLayer * (full * ctx + windowed * windowTokens);
}

/** The context llama.cpp will actually allocate: `-c 0` means "the model's". */
export function effectiveCtx(meta: ModelMeta, s: Settings): number {
  const c = num(s, "ctxSize");
  return c > 0 ? c : meta.nCtxTrain || 4096;
}

function sum(ns: number[]): number {
  return ns.reduce((a, b) => a + b, 0);
}

function pool(
  label: string,
  capacityB: number,
  otherB: number,
  reservedB: number,
  buckets: Bucket[],
): Pool {
  const usedB = sum(buckets.map((b) => b.bytes));
  // The reserve is spent memory as far as every fit test is concerned — that is
  // the whole point of it — so it counts towards the total exactly like another
  // process's allocation, and only its LABEL differs.
  const total = usedB + otherB + reservedB;
  return {
    label,
    capacityB,
    buckets: buckets.filter((b) => b.bytes > 0),
    usedB,
    otherB,
    reservedB,
    freeB: Math.max(0, capacityB - total),
    overB: Math.max(0, total - capacityB),
  };
}

/**
 * A model-shaped nothing.
 *
 * Lets the "what does the machine look like right now" view go through the same
 * `plan` as everything else: every llama.cpp bucket comes out zero, and the
 * pools still report what other processes hold and what is free. One code path
 * for both states beats a second, subtly different one.
 */
export const NO_MODEL: ModelMeta = {
  version: 0,
  arch: "",
  name: "",
  quant: "",
  nLayer: 0,
  nCtxTrain: 0,
  nEmbd: 0,
  nHead: 0,
  nHeadKv: 0,
  keyLength: 0,
  valueLength: 0,
  swaWindow: 0,
  swaPattern: 1,
  kvLoraRank: 0,
  nextnLayers: 0,
  nExpert: 0,
  nExpertUsed: 0,
  ropeFreqBase: 0,
  nTensors: 0,
  tensorBytes: 0,
  embdBytes: 0,
  outputBytes: 0,
  unknownTypes: 0,
  nCtxOrig: 0,
  indexerTopK: 0,
  splitNo: 0,
  splitCount: 0,
  splitTensors: 0,
  layers: [],
};

/**
 * The machine with llama.master's own current usage taken back out.
 *
 * Needed for an honest projection. `plan` reads "in use by others" straight off
 * the telemetry, which includes a llama-server this app is running — so
 * projecting a model while one is already loaded counted the running one TWICE:
 * once as other people's memory, once as the new plan. Removing our share first
 * makes the projection what it claims to be: the machine as it will look once
 * this model replaces whatever is loaded now.
 *
 * The VRAM subtraction is spread across cards in proportion to what each is
 * holding. Per-process VRAM attribution is not available from the telemetry this
 * app collects, and proportional is the honest approximation — the total is
 * exact, only its split across cards is inferred.
 */
export function withoutOurUsage(hw: Hw, ourVramB: number, ourRamB: number): Hw {
  const usedTotal = sum(hw.gpus.map((g) => g.vramUsedB));
  const takeVram = Math.max(0, Math.min(ourVramB, usedTotal));
  const gpus = hw.gpus.map((g) => ({
    ...g,
    vramUsedB: usedTotal > 0
      ? Math.max(0, g.vramUsedB - takeVram * (g.vramUsedB / usedTotal))
      : g.vramUsedB,
  }));
  const mem = hw.mem
    ? {
      ...hw.mem,
      usedB: Math.max(0, hw.mem.usedB - Math.max(0, ourRamB)),
      availableB: Math.min(
        hw.mem.totalB,
        hw.mem.availableB + Math.max(0, ourRamB),
      ),
    }
    : hw.mem;
  return { ...hw, gpus, mem };
}

/** VRAM currently held by anything other than the run we are planning. */
function vramInUse(gpus: Gpu[]): number {
  return sum(gpus.map((g) => g.vramUsedB));
}

/**
 * Is this plan a PROPOSAL or a description of a run that is already up?
 *
 * Every number below is the same either way. What differs is the one output
 * that is a PREDICTION rather than an accounting: whether llama.cpp can cut the
 * offloaded layers so that each card holds its share (`devices.fits`,
 * `unplacedB`, and the two notes about them).
 *
 * For a proposal that prediction is the most valuable thing here — it is what
 * stops a plan dying with `cudaMalloc failed` on device 1. For a run that is
 * ALREADY LOADED it is a contradiction waiting to happen, and it happened: the
 * fitter's per-card budgets subtract the planning safety reserve (5% of each
 * card) and re-derive our own footprint by proportion, so re-packing a live run
 * came up ~1 GB short and the machine panel announced "1010 MB of layers have
 * nowhere to go — no card has room for them, however the cut is made" about a
 * model that was answering prompts at the time, with `vram.overB` reading 0 on
 * the same screen. The layers ARE placed; llama.cpp placed them. A prediction
 * the evidence has already settled is not a warning, it is a false statement.
 *
 * Real pressure on a live run is still reported, by the measurements rather
 * than the fitter: `vram.overB`/`ram.overB` come from what the machine says is
 * in use, and `src/lib/adapt.ts:drift` reads those.
 */
export type PlanQuestion = "proposed" | "running";

/**
 * Place a model under one settings map on one machine.
 *
 * Placement rules mirror llama.cpp:
 * - `-ngl N` offloads the last N **slots**, and there are `nLayer + 1` of them —
 *   the output head counts as one. So `-ngl 43` on a 43-layer model offloads
 *   layers 1..42 AND the output, leaving layer 0 on the host; only
 *   `-ngl > nLayer` offloads every layer (`src/lib/devsplit.ts:offloadRange`).
 * - The token embedding table is NEVER offloaded. llama.cpp classifies it as an
 *   input tensor and pins those to the CPU regardless of `-ngl` — "there is very
 *   little benefit to offloading the input layer" (`llama-model.cpp`,
 *   `dev_input`). Billing it to VRAM cost ~1 GB of a card's budget that was
 *   always going to be spent on the host.
 * - `--n-cpu-moe N` keeps the routed experts of the **first** N layers in RAM,
 *   even when those layers are otherwise on the GPU.
 * - `-nkvo` moves the whole KV cache to RAM regardless of layer placement.
 *
 * `asked` is which QUESTION this plan answers, and it changes one thing: whether
 * the placement is still open to doubt. See `PlanQuestion`.
 */
export function plan(
  meta: ModelMeta,
  hw: Hw,
  s: Settings,
  asked: PlanQuestion = "proposed",
): Plan {
  // What the user has kept for themselves. Clamped to the machine here, once,
  // so every figure below — the pools, the per-card picture and the packing
  // budgets — is working from the same number (`src/lib/reserve.ts`).
  const reserve = reserveOf(hw);
  // Per card, and the machine-wide total is whatever those add up to — the VRAM
  // reserve is stated per card now (every card, plus extra on the ones with a
  // display), so the pool figure is derived from the shares and cannot disagree
  // with the per-card picture.
  const reserveShares = vramReserveShares(hw.gpus, reserve);
  const reservedVramB = reserveShares.reduce((a, b) => a + b, 0);
  const nLayer = meta.nLayer;
  const off = offloadRange(nLayer, s);
  const moeOnCpu = Math.max(0, Math.min(num(s, "nCpuMoe"), nLayer));
  const ctx = effectiveCtx(meta, s);
  const kvPerTokenB = kvPerToken(meta, s);
  const kvTotalB = kvTotal(meta, s, ctx);
  const kvOnCpu = bool(s, "noKvOffload");
  const outputOnGpu = slotOnGpu(nLayer, off);

  // Per-slot GPU bytes, in slot order — the shape `devsplit` needs to cut into
  // per-card ranges, and the sums the pools need. Slot `nLayer` is the output.
  const kvPerLayerB = kvOnCpu || nLayer <= 0 ? 0 : kvTotalB / nLayer;
  const slotCostsB: number[] = [];
  let gpuDense = 0;
  let gpuExperts = 0;
  let cpuWeights = 0;
  let layersOnGpu = 0;
  for (let i = 0; i < nLayer; i++) {
    const l = meta.layers[i];
    // Same reasoning as `kvPerToken`: these come out of the file, and a layer
    // whose experts are larger than the layer itself would otherwise make the
    // dense figure negative and every total after it wrong.
    const bytes = whole(l?.bytes ?? 0);
    const expert = Math.min(whole(l?.expert ?? 0), bytes);
    const dense = bytes - expert;
    const onGpu = slotOnGpu(i, off);
    const expertsHere = i < moeOnCpu ? 0 : expert;
    if (onGpu) {
      layersOnGpu++;
      gpuDense += dense;
      gpuExperts += expertsHere;
      cpuWeights += expert - expertsHere;
      slotCostsB.push(dense + expertsHere + kvPerLayerB);
    } else {
      cpuWeights += bytes;
    }
  }
  // The output head moves with `-ngl`; the embedding table never does.
  if (outputOnGpu) {
    gpuDense += whole(meta.outputBytes);
    slotCostsB.push(whole(meta.outputBytes));
  } else {
    cpuWeights += whole(meta.outputBytes);
  }
  cpuWeights += whole(meta.embdBytes);

  // KV follows its layer, unless -nkvo pins all of it to the host.
  const kvGpuShare = nLayer > 0 ? layersOnGpu / nLayer : 0;
  const kvOnGpu = kvOnCpu ? 0 : kvTotalB * kvGpuShare;
  const kvOnRam = kvTotalB - kvOnGpu;

  // Compute buffers scale with the micro-batch, not the batch: llama.cpp runs
  // `-ub` tokens at a time. Four activation-sized tensors is the empirical
  // shape of the graph; the backend context is a flat per-process cost.
  const ubatch = Math.max(1, num(s, "ubatchSize"));
  const activation = ubatch * whole(meta.nEmbd) * 4;
  const usingGpu = off.count > 0 && hw.gpus.length > 0;

  // Speculative decoding with the model's own MTP block costs a SECOND context —
  // llama.cpp measures "only context+compute are new", because the drafting
  // block lives on the target model and its weights are already counted above.
  // That second context is one block's KV over the same window, so it is small,
  // but it is real and a plan that ignored it could hand back settings that no
  // longer fit the moment the flag is emitted (server-context.cpp:1085).
  const mtpKvB = specMtpActive(meta, s)
    ? whole(kvPerTokenB / Math.max(1, nLayer) * ctx)
    : 0;
  const mtpDraftB = mtpKvB > 0 ? mtpKvB + BACKEND_CONTEXT_B / 2 : 0;

  const gpuCompute = usingGpu
    ? activation * 4 + BACKEND_CONTEXT_B * Math.max(1, hw.gpus.length) +
      mtpDraftB + computeScratch(meta, ubatch, ctx)
    : 0;
  // On a CPU-only run the draft context is just as real, minus the GPU backend
  // half — it lands in RAM, where a tight MTP run is exactly the case that
  // cannot afford an unbilled block of KV.
  const cpuCompute =
    (layersOnGpu < nLayer || !usingGpu ? activation * 2 : 32 * MB) +
    (usingGpu ? 0 : mtpKvB);

  // Where each slot actually lands. The aggregate above says whether the model
  // fits the machine; this says whether any single card is being asked for more
  // than it has, which is a different question and the one that OOMs.
  // The scratch lands ON a device, so it comes out of that device's budget too.
  const perDeviceOverheadB = usingGpu
    ? BACKEND_CONTEXT_B + activation * 4 + computeScratch(meta, ubatch, ctx)
    : 0;
  // A proposal is budgeted defensively — the fixed per-card safety reserve, the
  // user's reserve and this device's scratch all come off before a slot may be
  // placed. A run that is ALREADY UP gets the physical budget instead: whatever
  // the card holds beside everybody else's bytes. Those margins exist to keep a
  // future allocation from failing, and re-imposing them on an allocation that
  // has already succeeded is how the picture came to strand layers that the
  // driver was plainly holding.
  const budgetsB = asked === "running"
    ? hw.gpus.map((g) => Math.max(0, g.vramTotalB - g.vramUsedB))
    : deviceBudgets(hw.gpus, perDeviceOverheadB, reserveShares);
  const counts = usingGpu ? packSlots(slotCostsB, budgetsB) : [];

  // The per-card picture — including when the packing FAILS. That is the
  // moment the user most needs to see it, so the display falls back to a
  // best-effort fill (cards to their budgets, in order) and reports the
  // remainder as bytes with nowhere to go.
  let displayCounts = counts;
  let unplacedB = 0;
  if (usingGpu && counts === null) {
    const dc = budgetsB.map(() => 0);
    let dev = 0;
    let used = 0;
    for (const cost of slotCostsB) {
      while (dev < budgetsB.length && used + cost > (budgetsB[dev] ?? 0)) {
        dev++;
        used = 0;
      }
      if (dev >= budgetsB.length) {
        unplacedB += cost;
        continue;
      }
      dc[dev] = (dc[dev] ?? 0) + 1;
      used += cost;
    }
    displayCounts = dc;
  }
  const perCardBytes = displayCounts
    ? loadPerDevice(slotCostsB, displayCounts)
    : [];
  const slotsPlaced = displayCounts
    ? displayCounts.reduce((a, c) => a + c, 0)
    : 0;
  const cards = usingGpu
    ? hw.gpus.map((g, i) => {
      const n = displayCounts?.[i] ?? 0;
      const weightsB = perCardBytes[i] ?? 0;
      const kvB = slotsPlaced > 0 && !kvOnCpu
        ? whole(kvOnGpu * (n / slotsPlaced))
        : 0;
      const computeB = n > 0 || i === 0 ? perDeviceOverheadB : 0;
      const otherB = g.vramUsedB;
      const reservedB = reserveShares[i] ?? 0;
      const overB = Math.max(
        0,
        otherB + reservedB + weightsB + kvB + computeB - g.vramTotalB,
      );
      return {
        name: g.name || `GPU ${i}`,
        capacityB: g.vramTotalB,
        otherB,
        reservedB,
        weightsB,
        kvB,
        computeB,
        overB,
      };
    })
    : [];

  const vramCapacity = sum(hw.gpus.map((g) => g.vramTotalB));
  const ramCapacity = hw.mem?.totalB ?? 0;
  const ramOther = hw.mem ? hw.mem.totalB - hw.mem.availableB : 0;

  const vram = pool("VRAM", vramCapacity, vramInUse(hw.gpus), reservedVramB, [
    { key: "weights", label: "Weights", bytes: gpuDense },
    { key: "experts", label: "Experts", bytes: gpuExperts },
    { key: "kv", label: "KV cache", bytes: kvOnGpu },
    { key: "compute", label: "Compute (est.)", bytes: gpuCompute },
  ]);
  const ram = pool("RAM", ramCapacity, ramOther, reserve.ramB, [
    { key: "weights", label: "Weights", bytes: cpuWeights },
    { key: "kv", label: "KV cache", bytes: kvOnRam },
    { key: "compute", label: "Compute (est.)", bytes: cpuCompute },
  ]);

  // For a live run the question "can these layers be divided across the cards"
  // has already been answered — by llama.cpp, when it loaded them. So the
  // MEASUREMENT decides: the machine is over VRAM, or it is not. Only a
  // proposal is subject to the packer's verdict.
  const placementSettled = asked === "running" && vram.overB === 0;
  const devices: DevicePlan = {
    bytesB: counts ? loadPerDevice(slotCostsB, counts) : [],
    budgetsB,
    tensorSplit: counts ? tensorSplitValue(counts) : "",
    fits: counts !== null || placementSettled,
    cards,
    unplacedB: placementSettled ? 0 : unplacedB,
  };

  const notes: string[] = [];
  if (meta.unknownTypes > 0) {
    notes.push(
      `${meta.unknownTypes} tensor(s) use a ggml type this build does not know — sizes below exclude them.`,
    );
  }
  if (vram.overB > 0) {
    notes.push(
      `Over VRAM by ${
        fmtGb(vram.overB)
      } — lower GPU layers, shrink the context, or quantise the KV cache.`,
    );
  }
  if (ram.overB > 0) {
    notes.push(
      `Over RAM by ${
        fmtGb(ram.overB)
      } — the OS will swap or the load will be killed.`,
    );
  }
  // Never let the reserve be the invisible reason. It is memory the user chose
  // to hold back, so when it is what turns a fit into an overflow the way out
  // has to be on screen next to the shortfall — otherwise the app reports a
  // machine that is too small for a model that would in fact load.
  const blockedByReserve =
    (vram.overB > 0 && reservedVramB > 0 && vram.overB <= reservedVramB) ||
    (ram.overB > 0 && reserve.ramB > 0 && ram.overB <= reserve.ramB);
  if (blockedByReserve) {
    notes.push(
      `You are reserving ${
        reserveLabel(reserve, hw.gpus)
      } for your own work, and this plan needs it. Lower the reserved memory to spend it, or leave it and take the smaller plan.`,
    );
  }
  // The failure the totals cannot see: enough VRAM across the machine, and no
  // way to cut the layers so that each card holds its share. llama.cpp divides
  // the offloaded run by COUNT, and `--n-cpu-moe` makes the last layers many
  // times heavier than the first, so they pile onto the last card.
  if (vram.overB === 0 && !devices.fits && hw.gpus.length > 1) {
    notes.push(
      `This fits across the cards but not on them: llama.cpp splits the layers by count, and with the experts held back the last layers are far heavier than the first. Lower GPU layers or move more experts to RAM.`,
    );
  }
  if (devices.unplacedB > 0) {
    notes.push(
      `${
        fmtGb(devices.unplacedB)
      } of layers have no card that can hold them — the map shows each card filled as far as it goes, and this remainder is what does not fit anywhere.`,
    );
  }
  // Advice needs something to advise about. With no model (the "what is the
  // machine doing right now" view goes through here as NO_MODEL) a note telling
  // the user to raise their GPU layers is noise attached to an idle machine.
  const haveModel = nLayer > 0;
  if (hw.gpus.length === 0) {
    notes.push("No GPU detected — everything runs on the CPU.");
  } else if (haveModel && off.count === 0) {
    notes.push(
      "GPU layers is 0, so the GPU is idle. Raise it to use the card.",
    );
  }
  if (moeOnCpu > 0) {
    notes.push(
      `Experts of the first ${moeOnCpu} layer(s) stay in RAM; attention runs on the GPU.`,
    );
  }
  if (kvOnCpu) notes.push("KV cache is pinned to system RAM (-nkvo).");
  if (bool(s, "mlock") && ram.usedB + ram.otherB > ramCapacity * 0.9) {
    notes.push("--mlock with this little free RAM risks the OOM killer.");
  }

  return {
    nLayer,
    layersOnGpu,
    moeOnCpu,
    ctx,
    vram,
    ram,
    kvPerTokenB,
    kvTotalB,
    fits: vram.overB === 0 && ram.overB === 0 && devices.fits,
    devices,
    notes,
  };
}

/**
 * Attention scratch that grows with the CONTEXT — the term that was missing.
 *
 * For ordinary attention the compute buffer is a flat per-process cost plus a
 * few micro-batch-sized activations, which is what the rest of `plan` assumes
 * and what holds for almost every model. A sparse-attention model breaks it:
 * DeepSeek-V4's "lightning indexer" scores the WHOLE context for every token in
 * the micro-batch, so the graph holds tensors of `ubatch x ctx`, and the cost
 * rises linearly with the context instead of not at all.
 *
 * How wrong that was: at a 1,048,576 context llama.cpp asked for a **68.5 GiB**
 * compute buffer where this function's absence left the estimate at 730 MB. The
 * app therefore proposed a context that could not be allocated on any machine
 * and any split — 68 GiB of scratch does not fit two 24 GB cards however the
 * layers are divided. Every "it still crashed" came back to this number.
 *
 * `SCRATCH_TENSORS` is MEASURED, not derived: 68.5 GiB / (512 x 1,048,576 x 4 B)
 * = 34.3 score-sized tensors live in the graph at once. A GGUF header does not
 * say how many, so this is calibrated against a real allocation and rounded up.
 * Being a little pessimistic here is the right direction: it costs some context,
 * where being optimistic costs a failed load after a two-minute wait.
 *
 * Gated on the model DECLARING the indexer, so nothing changes for the models
 * where the flat estimate was already correct.
 *
 * Wiring this in was only safe once the tuner stopped maximising the context:
 * priced honestly against the old objective, the search bought context by
 * evicting layers and settled on `-ngl 1` at a 654,848 context — which fits,
 * and runs at CPU speed. `tune.ts` now fixes residency first, so a scratch that
 * grows with the context costs CONTEXT, which is the correct thing for it to
 * cost.
 */
export function computeScratch(
  meta: ModelMeta,
  ubatch: number,
  ctx: number,
): number {
  if (whole(meta.indexerTopK) <= 0) return 0;
  const SCRATCH_TENSORS = 34;
  return whole(Math.max(1, ubatch) * whole(ctx) * 4 * SCRATCH_TENSORS);
}

function fmtGb(b: number): string {
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}
