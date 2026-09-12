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

import type { Hw, ModelMeta, Settings } from "./types.ts";
import { bool, num, str } from "./params.ts";
import {
  countsFromSplit,
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
 * How many layers actually hold a per-token KV cache.
 *
 * Every layer, for almost every model. A hybrid linear-attention model
 * (Qwen3-Next / Qwen3.5-class) breaks it: `full_attention_interval` says only
 * every Nth trunk layer is full attention, and the rest are recurrent — their
 * state is constant-size (`recurrentStateB`), not per-token. The MTP block is
 * dense attention (llama.cpp flags it non-recurrent, `models/qwen35.cpp`), so
 * it keeps its cache. Qwen3.8-27B: 65 declared layers, 17 with a KV cache —
 * billing all 65 said "128k does not fit in 48 GB of VRAM" about a model that
 * runs there at its full 262,144.
 *
 * qwen4exp (Qwen3.8-Flash-Next) follows the same rule — trunk layer i is full
 * attention iff (i + 1) % interval == 0, `models/qwen4exp.cpp` — with the
 * interval defaulted to 4 by the READER when the header omits it, because for
 * that architecture llama.cpp hardcodes the default (`rust/src/gguf.rs`).
 * Verified against the real 111 GB model: 48 layers, 12 with a KV cache.
 */
export function kvLayers(meta: ModelMeta): number {
  const nLayer = whole(meta.nLayer);
  const interval = whole(meta.fullAttnInterval);
  if (interval < 2 || nLayer <= 0) return nLayer;
  const nextn = Math.min(whole(meta.nextnLayers), nLayer);
  const trunk = nLayer - nextn;
  // llama.cpp: trunk layer i is full attention iff (i + 1) % interval == 0.
  return Math.floor(trunk / interval) + nextn;
}

/**
 * Constant per-sequence state of the recurrent (linear-attention) layers.
 *
 * llama.cpp's `n_embd_r` (conv state) + `n_embd_s` (delta/ssm state), f32 per
 * element, per recurrent layer, per server slot. Does not grow with the
 * context and is never quantised by `-ctk` — which is why it is its own term
 * rather than part of the per-token rate. ~157 MB on Qwen3.8-27B at one slot.
 *
 * qwen4exp's single PLE layer keeps one extra conv-history row per sequence
 * (`ple_conv_state`, llama-hparams.cpp) that is NOT billed here: on
 * Qwen3.8-Flash-Next it is (4−1) × 3 × 4 × 2560 × 4 B ≈ 0.35 MB — under the
 * rounding of every figure this feeds. Worth revisiting only if an arch with
 * many PLE layers appears; llama.cpp itself asserts exactly one today.
 */
export function recurrentStateB(meta: ModelMeta, seqs = 1): number {
  const nLayer = whole(meta.nLayer);
  const interval = whole(meta.fullAttnInterval);
  if (interval < 2 || nLayer <= 0) return 0;
  const recurrent = nLayer - kvLayers(meta);
  const convB = Math.max(0, whole(meta.ssmDConv) - 1) *
    (whole(meta.ssmDInner) + 2 * whole(meta.ssmNGroup) * whole(meta.ssmDState));
  const stateB = whole(meta.ssmDState) * whole(meta.ssmDInner);
  return whole(recurrent * (convB + stateB) * 4 * Math.max(1, whole(seqs)));
}

/**
 * KV bytes per token across every layer that HAS a per-token cache.
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
  // Only the layers that hold a per-token cache pay the rate — every layer on
  // an ordinary model, one in four (plus the MTP block) on a hybrid one.
  const nKv = kvLayers(meta);
  if (meta.kvLoraRank > 0) {
    return whole(nKv * (meta.kvLoraRank + MLA_ROPE_DIM) * bk);
  }
  return whole(nKv * heads * (kLen * bk + vLen * bv));
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
  const nKv = kvLayers(meta);
  const perLayer = nKv > 0 ? kvPerToken(meta, s) / nKv : 0;
  // The recurrent layers' constant state rides along here because it is
  // allocated by the same memory module and lives on the same devices as the
  // cache — every consumer (the pools, the placement, `-nkvo`) treats it the
  // way llama.cpp does. Zero for every non-hybrid model.
  const stateB = recurrentStateB(meta, Math.max(1, num(s, "parallel")));
  const { full, windowed } = swaSplit(meta);
  // A hybrid model's KV-bearing layers all see the whole context; no known
  // model interleaves a sliding window on top of them.
  if (windowed === 0 || nKv < meta.nLayer) return perLayer * nKv * ctx + stateB;
  // A windowed layer still has to hold the current batch alongside its window.
  const windowTokens = Math.min(
    ctx,
    meta.swaWindow + Math.min(ctx, num(s, "batchSize")),
  );
  return perLayer * (full * ctx + windowed * windowTokens) + stateB;
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
  fullAttnInterval: 0,
  ssmDConv: 0,
  ssmDInner: 0,
  ssmDState: 0,
  ssmNGroup: 0,
  nextnLayers: 0,
  nExpert: 0,
  nFf: 0,
  nFfExp: 0,
  nExpertUsed: 0,
  ropeFreqBase: 0,
  nTensors: 0,
  tensorBytes: 0,
  params: 0,
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
  /**
   * Free VRAM per card at the moment this run was spawned (`srv.runCardFreeB`),
   * in the SAME order as `hw.gpus` — which means the list the planner sees, so
   * the recorder must filter its cards the way `hwSnapshot` does.
   *
   * With it, a `"running"` plan stops guessing which card holds what. Without
   * it, the per-card picture is `packSlots`' first-fit proposal, and on a live
   * run that proposal is routinely wrong in a way the user can see: with
   * `--n-cpu-moe 42` every offloaded slot is attention-only, the whole 9.2 GB
   * fits inside card 0's budget, so the packer answers `[44, 0]` — and the
   * panel drew GPU 1 holding nothing of ours while nvidia-smi showed 12 GB on
   * it. It is not even a split we ASKED for: the packer emits `-ts` only when
   * it needs one, so llama.cpp divided the layers by its own free-VRAM rule and
   * used both cards. The measurement settles it; a difference of a card is not
   * a rounding error, and our own bytes were being drawn as somebody else's.
   *
   * Ignored unless it names every card — a length that disagrees with `hw.gpus`
   * is an index mismatch, and attributing card 2's memory to card 1 would be a
   * worse answer than the packer's.
   */
  cardFreeAtStartB: readonly number[] = [],
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
  // Two parallel per-slot arrays: `slotCostsB` carries weights PLUS the KV so
  // the packer places the cache with its layers; `slotWeightsB` is the weights
  // alone, so the per-card picture can draw them separately without ever
  // double-counting the KV (see the note at `perCardBytes`).
  const slotCostsB: number[] = [];
  const slotWeightsB: number[] = [];
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
      slotWeightsB.push(dense + expertsHere);
    } else {
      cpuWeights += bytes;
    }
  }
  // The output head moves with `-ngl`; the embedding table never does.
  if (outputOnGpu) {
    gpuDense += whole(meta.outputBytes);
    slotCostsB.push(whole(meta.outputBytes));
    slotWeightsB.push(whole(meta.outputBytes));
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
  // Server slots multiply the compute scratch — each runs its own graph. Read
  // from the settings the argv is built from, never assumed: llama.cpp's own
  // default is `auto`, which chose four and quadrupled the buffer while the
  // panel said one (`params.ts:parallel`, `types.ts:Param.llamaDef`).
  const slots = Math.max(1, num(s, "parallel"));
  const activation = ubatch * whole(meta.nEmbd) * 4;
  // The widest matmul in the prefill graph is the FFN, not the embedding: a
  // gate and an up projection at `n_ff` wide (the fired experts' combined
  // width on a MoE), f32. At `-ub 512` this is noise next to the flat backend
  // cost; at `-ub 4096` it dominates, and an estimate built on `n_embd` alone
  // under-billed it ~4x — which matters now that the tuner may SPEND headroom
  // on a bigger micro-batch (`tune.ts`). Gated off for sparse-attention
  // models: their scratch is measured end to end (`computeScratch`),
  // micro-batch term included, and stacking an estimate on a measurement
  // would refuse placements the machine runs.
  const declaredFf = Math.max(
    whole(meta.nFf),
    whole(meta.nFfExp) * whole(meta.nExpertUsed),
  );
  const ffWidth = declaredFf > 0 ? declaredFf : whole(meta.nEmbd) * 4;
  const ffActivation = whole(meta.indexerTopK) > 0
    ? 0
    : ubatch * ffWidth * 4 * 2;
  // The attention mask, which is the term that makes a LARGE micro-batch
  // expensive at a LONG context — and the one this file was blind to.
  const maskB = attnMaskB(meta, s, ctx, ubatch);
  // A hybrid linear-attention model carries working set per context token on
  // top of all of the above (`HYBRID_SCRATCH_B_PER_CTX`).
  const hybridB = hybridScratch(meta, ctx);
  const usingGpu = off.count > 0 && hw.gpus.length > 0;

  // Speculative decoding with the model's own MTP block costs a SECOND CONTEXT,
  // and that is the whole point of `mtpDraft` — this file used to call it "one
  // block's KV over the same window, so it is small". It is not small, and the
  // measurement is in that function.
  const mtp = mtpDraft(meta, s, ctx, ubatch, slots);
  const mtpKvB = mtp.onceB;

  // The pool total is the sum of what the cards below are charged, and it has
  // to stay that way: one number for the bar and a different one for the packer
  // is how a plan came to say "fits" on one line and "nowhere to go" on the
  // next. Per-device costs times the devices, plus the scratch once.
  const gpuCompute = usingGpu
    ? (activation * 4 + ffActivation + BACKEND_CONTEXT_B + scratchFloor(meta) +
          maskB + hybridB + mtp.perDeviceB) * Math.max(1, hw.gpus.length) +
      mtp.onceB + computeScratch(meta, ubatch, ctx, slots)
    : 0;
  // On a CPU-only run the draft context is just as real, minus the GPU backend
  // half — it lands in RAM, where a tight MTP run is exactly the case that
  // cannot afford an unbilled block of KV.
  const cpuCompute =
    (layersOnGpu < nLayer || !usingGpu
      ? activation * 2 + ffActivation / 2
      : 32 * MB) +
    // The mask is built in PINNED HOST memory on a GPU run (llama.cpp's
    // `CUDA_Host compute buffer`) and in the CPU compute buffer otherwise, so
    // it is charged to RAM either way — once, not per device. Measured to the
    // byte on two architectures; see `attnMaskB`.
    maskB +
    (usingGpu ? 0 : hybridB) +
    (usingGpu ? 0 : mtpKvB);

  // Where each slot actually lands. The aggregate above says whether the model
  // fits the machine; this says whether any single card is being asked for more
  // than it has, which is a different question and the one that OOMs.
  //
  // How much of the scratch each CARD must be able to give.
  //
  // The total is divided between the cards, not repeated on each — measured on
  // the real model at a 1,048,576 context, the same placement costs 8,110 MiB
  // of scratch on one card and 8,567 MiB spread over two. So the POOL above
  // counts it once, and that is what the machine actually uses.
  //
  // But WHERE the division falls is llama.cpp's to choose, and it does not
  // follow the layer count. With `--n-cpu-moe 32 -ts 33.5,10.5`, card 1 held
  // 10 slots of 44 — 23% of the layers — and wanted about 60% of the scratch:
  //
  //   graph_reserve: failed to allocate compute buffers
  //   allocating 1676.89 MiB on device 1: cudaMalloc failed: out of memory
  //
  // on a plan that had budgeted card 1 its proportional 23% and called it a
  // fit. So a card is budgeted for the WHOLE scratch. That is pessimistic by
  // construction and it is the pessimism that makes a proposal start: the
  // question here is not "what will this cost" (the pool answers that) but
  // "can this card hold whatever share it is handed", and the only share that
  // is safe to promise is all of it.
  const scratchB = computeScratch(meta, ubatch, ctx, slots);
  const fixedPerDeviceB = usingGpu
    ? BACKEND_CONTEXT_B + activation * 4 + ffActivation + scratchFloor(meta) +
      maskB + hybridB + mtp.perDeviceB
    : 0;
  const perDeviceOverheadB = usingGpu ? fixedPerDeviceB + scratchB : 0;

  // OUR OWN VRAM, PER CARD, MEASURED — free-at-spawn minus free-now, which is
  // the same baseline `adapt.ts:drift` reads for "memory came back". Only for a
  // run that is up, only when the baseline names every card, and only when it
  // says we took something: a run that recorded nothing (an adopted server, a
  // start before this was kept) falls back to the packer rather than drawing an
  // empty machine.
  //
  // Anything else that allocated since the spawn lands in this figure too. That
  // is the honest error bar of a device-wide counter — and it is bounded by the
  // thing it replaces, which was not measurement at all but a first-fit guess
  // about a placement llama.cpp had already made differently.
  const measuredOursB: number[] | null = (() => {
    if (asked !== "running" || !usingGpu) return null;
    if (cardFreeAtStartB.length !== hw.gpus.length) return null;
    const ours = hw.gpus.map((g, i) =>
      Math.max(
        0,
        (cardFreeAtStartB[i] ?? 0) - Math.max(0, g.vramTotalB - g.vramUsedB),
      )
    );
    return ours.some((b) => b > 0) ? ours : null;
  })();
  /** What is on each card that is NOT this run. Our bytes are itemised as our
   *  buckets, so leaving them in "in use elsewhere" would draw them twice. */
  const othersUsedB = hw.gpus.map((g, i) =>
    Math.max(0, g.vramUsedB - (measuredOursB?.[i] ?? 0))
  );

  const budgetsB = asked === "running"
    ? hw.gpus.map((g, i) => Math.max(0, g.vramTotalB - (othersUsedB[i] ?? 0)))
    : deviceBudgets(hw.gpus, perDeviceOverheadB, reserveShares);
  // A `-ts` on the command line is not advice, it is the placement. Until this
  // read it, the plan drew the PACKER's cuts however the argv disagreed — so a
  // user who typed a split saw per-card bars for a run that was not going to
  // happen, on the one page whose whole promise is that it is not a form.
  // (The tuner writes its own `-ts` here too, and this reading it back is the
  // round trip: `tensorSplitValue` and `countsFromSplit` must agree, or the
  // number we emit is not the placement we drew.)
  const asked_ts = str(s, "tensorSplit").trim();
  const pinnedCounts = usingGpu && asked_ts
    ? countsFromSplit(asked_ts, off.count, hw.gpus.length)
    : null;
  const counts = usingGpu
    ? (pinnedCounts ?? packSlots(slotCostsB, budgetsB))
    : [];

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
  const slotsPlaced = displayCounts
    ? displayCounts.reduce((a, c) => a + c, 0)
    : 0;
  // The per-card picture splits each slot's cost back into weights and KV.
  // `slotCostsB` includes `kvPerLayerB` so the PACKER places the cache with its
  // layers, but `loadPerDevice` over it would give weights that ALREADY contain
  // the cache — and adding `kvB` on top double-counted it, so card0 + card1
  // exceeded the pool's own "VRAM used" for the same plan. Weights come from a
  // KV-free array, so they reconcile with the pool to the byte.
  const weightsNoKvB = (counts: readonly number[]): number[] =>
    loadPerDevice(slotWeightsB, counts);
  // Our machine-wide GPU total, as the buckets add it up — the denominator that
  // turns a card's measured bytes back into the three bands the map draws.
  const gpuOursB = gpuDense + gpuExperts + kvOnGpu + gpuCompute;
  const cards = usingGpu
    ? hw.gpus.map((g, i) => {
      const n = displayCounts?.[i] ?? 0;
      // Measured: the card's own total is a fact, and the three bands inside it
      // are that fact cut by the proportions the buckets already carry. The
      // split between bands stays derived — the driver reports bytes, not what
      // they are for — but which CARD they are on stops being a guess, and that
      // is the part that was visibly wrong.
      const mine = measuredOursB?.[i] ?? 0;
      const share = measuredOursB && gpuOursB > 0 ? mine / gpuOursB : 0;
      const weightsB = measuredOursB
        ? whole((gpuDense + gpuExperts) * share)
        : displayCounts
        ? weightsNoKvB(displayCounts)[i] ?? 0
        : 0;
      const kvB = measuredOursB
        ? whole(kvOnGpu * share)
        : slotsPlaced > 0 && !kvOnCpu
        ? whole(kvOnGpu * (n / slotsPlaced))
        : 0;
      // Drawn as it is budgeted, so the picture and the packer cannot disagree
      // about the same card. See the note at `perDeviceOverheadB`: the pool
      // counts the scratch once because that is what the machine uses; a card
      // is charged all of it because llama.cpp decides the split and does not
      // divide it by layer count. That pessimism is a PROPOSAL's job — for a
      // measured run the scratch is already inside the bytes on the card.
      const computeB = measuredOursB
        ? whole(gpuCompute * share)
        : n > 0 || i === 0
        ? perDeviceOverheadB
        : 0;
      const otherB = othersUsedB[i] ?? g.vramUsedB;
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

  // "In use" here means BY EVERYONE ELSE — our own buckets are itemised below,
  // so a figure that still contained them would draw this run twice. When the
  // per-card measurement is available it has already been taken out per card
  // (`othersUsedB`); otherwise this is the device-wide reading and the caller
  // is the one that subtracts us (`derive.ts:withoutOurUsage`).
  const vram = pool("VRAM", vramCapacity, sum(othersUsedB), reservedVramB, [
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
    // Measured first: on a live run this is what each card holds of ours, not
    // what a packer would have given it.
    bytesB: measuredOursB ??
      (counts ? loadPerDevice(slotCostsB, counts) : []),
    budgetsB,
    // The split that produces this placement — which, when the user pinned
    // one, is theirs. Re-deriving it from the counts would answer "" for a
    // pinned `1,0` (one card doing all the work needs no split of OURS), and
    // "" does not produce that placement: llama.cpp would fall back to
    // splitting by free VRAM and put half the model on the other card.
    tensorSplit: pinnedCounts
      ? asked_ts
      : counts
      ? tensorSplitValue(counts)
      : "",
    // A PINNED split always produces counts — it is an instruction, not a
    // search — so "the packer found an arrangement" stops being the question.
    // What is left is whether the arrangement it names actually holds, which
    // is what the cards say. Reading `counts !== null` here made every hand-
    // typed `-ts` fit by definition, including one that asks a 12 GB card for
    // 12.2 GB.
    fits: pinnedCounts
      ? cards.every((c) => c.overB === 0)
      : counts !== null || placementSettled,
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
 * the micro-batch, so the graph holds tensors sized by the context and the cost
 * rises linearly with it instead of not at all. Without this term the app
 * proposed contexts that could not be allocated on any machine and any split.
 *
 * MEASURED, twice, and the second time changed both the constant AND the shape.
 *
 * The first calibration divided one observation — llama.cpp asking for 68.5 GiB
 * at a 1,048,576 context — by `ubatch x ctx x 4` and got 34 "score-sized
 * tensors". That made the scratch scale with the MICRO-BATCH, which turns out
 * not to be how it behaves, and it made the estimate ~3.4x too large at every
 * context anyone actually runs. On a 2x24 GB machine the app offered 27,648
 * tokens on a model that runs at 262,144.
 *
 * The second calibration was a sweep at the default slot count, and it missed
 * the largest term of all. The THIRD found it: the scratch is multiplied by the
 * number of server SLOTS, and llama.cpp's `-np` default is `-1 = auto`, which
 * chose FOUR. Same model, same placement, a 1,048,576 context:
 *
 *     -np auto (4 slots)   43,517 MiB of VRAM
 *     -np 1                21,467 MiB
 *
 * So `parallel` is a term here, not a server detail. Measured at one slot
 * (`--n-cpu-moe 43`, both cards, ub 512, VRAM read after a real generation so
 * CUDA's lazy allocations have happened; the desktop's 2,762 MiB subtracted):
 *
 *     ctx        slots   ours        above the 4,096 baseline
 *     4,096      1       10,158 MiB  —
 *     262,144    1       12,203 MiB  +2,045 MiB   (8.1 KB/token)
 *     1,048,576  1       18,705 MiB  +8,547 MiB   (8.4 KB/token)
 *     4,096      4       10,381 MiB  —
 *     262,144    4       17,687 MiB  +7,306 MiB   (29.0 KB/token)
 *     524,288    4       25,123 MiB  +14,742 MiB  (29.0 KB/token)
 *     1,048,576  4       40,749 MiB  +30,368 MiB  (29.8 KB/token)
 *
 * Straight lines both times, and the ratio between them is the slot count. The
 * micro-batch term is taken from the earlier sweep, which varied `-ub` at a
 * fixed context and found it small: 256/512/1024 moved the per-token cost by
 * -12%/+17%, where a term proportional to `-ub` would have moved it 4x.
 *
 * The constants below round the measurement up by about a fifth. Pessimism here
 * costs context; optimism costs a failed load, and the fit ladder
 * (`src/lib/fitladder.ts`) now recovers from a miss in either direction — but
 * being 3x out, as the first calibration was, is not something a ladder should
 * have to fix.
 *
 * Gated on the model DECLARING the indexer, so nothing changes for the models
 * where the flat estimate was already correct.
 */
/** Per context token, per slot, independent of the micro-batch. */
const SCRATCH_B_PER_CTX = 8 * 1024;
/** Per context token per micro-batch token, per slot — the small term. */
const SCRATCH_B_PER_CTX_UBATCH = 4;
/**
 * The part that does NOT scale with the context, per device.
 *
 * Found by measuring at 4,096 tokens, where the context term is worth about
 * 30 MB and everything left over is this: 2,017 MiB on one card, 3,271 MiB
 * spread over two — flat per device, and barely moved by the slot count
 * (+223 MiB going from one slot to four). The app allowed 384 MB, so a plan at
 * a 65,536 context budgeted 1.0 GB per card and llama.cpp asked for 1,985 MiB:
 *
 *   graph_reserve: failed to allocate compute buffers
 *   allocating 1984.76 MiB on device 0: cudaMalloc failed: out of memory
 *
 * Gated on the indexer like the rest of this, because it is the indexer's
 * non-context-scaled working set; an ordinary model's compute buffer really is
 * the few hundred MB `BACKEND_CONTEXT_B` allows.
 */
const SCRATCH_FLOOR_PER_DEVICE_B = 2 * 1024 ** 3;

/**
 * What `--spec-type draft-mtp` costs — a SECOND CONTEXT, not a rounding error.
 *
 * This file used to bill it as "one block's KV over the same window, so it is
 * small", plus half the flat backend figure, charged ONCE to the pool. The
 * server log says otherwise, and it says it in the most expensive way: the main
 * context allocates and succeeds, then llama.cpp builds the draft context and
 * runs out, and the error is `failed to create MTP context` on a plan that said
 * the model fitted with room to spare.
 *
 * MEASURED by A/B on Ornith-1.5-35B-A3B, one card, `--n-cpu-moe 41`, everything
 * else held still — the only difference between the two runs is the flag:
 *
 *                        no MTP            with MTP          the draft's share
 *   ctx 32,768 ub 512
 *     KV buffer          340 MiB           340 + 64 MiB      64 MiB
 *     recurrent state     62.81 MiB        251.25 MiB        +188.44 (= 3x)
 *     compute (device)   492.02 MiB        492.02 + 259.00   259 MiB
 *     compute (host)      40.28 MiB         40.28 + 40.02    40 MiB (its mask)
 *   ctx 131,072 ub 512   draft KV 256 MiB, draft compute 259 MiB (unchanged)
 *   ctx 32,768 ub 2,048  draft compute 406 MiB
 *
 * Three rules come out of that, and each lands where it belongs:
 *
 * - The draft KV is EXACTLY one dense block at f16 — 2,048 bytes per token on
 *   this model, which is `(keyLength + valueLength) x nHeadKv x 2`. It ignores
 *   `-ctk`/`-ctv`, so it is computed from the geometry rather than from the
 *   main cache's rate. Once, not per device.
 * - The recurrent state TRIPLES on top of itself — the draft needs its own
 *   copies to roll back — and 3x is exact, not a fit. Zero for a model with no
 *   recurrent layers. Once.
 * - The draft graph is a second compute buffer on EVERY device, flat in the
 *   context and linear in the micro-batch, plus its own copy of the mask. The
 *   constants below sit 13-34% above every point measured, which is the right
 *   direction: being short here is a load that fails after the main context has
 *   already succeeded.
 */
const MTP_DRAFT_FLAT_B = 320 * 1024 * 1024;
const MTP_DRAFT_B_PER_UBATCH = 96 * 1024;

export function mtpDraft(
  meta: ModelMeta,
  s: Settings,
  ctx: number,
  ubatch: number,
  slots = 1,
): { perDeviceB: number; onceB: number } {
  if (!specMtpActive(meta, s)) return { perDeviceB: 0, onceB: 0 };
  const draftKvB = whole(ctx) *
    (whole(meta.keyLength) + whole(meta.valueLength)) * whole(meta.nHeadKv) * 2;
  return {
    perDeviceB: whole(
      MTP_DRAFT_FLAT_B + Math.max(1, ubatch) * MTP_DRAFT_B_PER_UBATCH +
        attnMaskB(meta, s, ctx, ubatch),
    ),
    onceB: whole(draftKvB + 3 * recurrentStateB(meta, slots)),
  };
}

/**
 * The attention mask: `n_kv × n_ubatch`, f16 with flash attention and f32
 * without (`llama-graph.cpp`, `llm_graph_input_attn_kv`).
 *
 * This is not a calibration, it is llama.cpp's own tensor read off its own
 * source — and it is the term that made the app's compute estimate wrong by a
 * factor of four on a long-context run. It is invisible while `-ub` is 512 (a
 * 262,144-token context then costs 256 MiB, inside the flat backend figure) and
 * dominant the moment something spends headroom on the micro-batch: at
 * `-ub 4096` the SAME context costs 2 GiB. The tuner does spend that headroom
 * (`tune.ts`, "leftover VRAM becomes micro-batch"), so it was growing `-ub` to
 * 4096 against a budget that did not know what `-ub` costs, and the start died
 * in `graph_reserve` with no flag named.
 *
 * MEASURED, on two architectures that share nothing but llama.cpp:
 *
 *   Ornith-1.5-35B-A3B (qwen35moe, hybrid linear attention), 14 points from
 *   4,096×512 to 262,144×4,096: 2.00 bytes per (context token × micro-batch
 *   token), fitting the reported `CUDA_Host compute buffer` to ±0.0 MiB.
 *   Gemma-4-26B-A4B (dense attention + sliding window, MoE), 6 points:
 *   1.99 bytes, ±1.5 MiB.
 *
 * WHERE it lands differs by backend and is why it is charged twice. On a CUDA
 * run the mask itself is built in PINNED HOST memory (`CUDA_Host compute
 * buffer` — that is the buffer the two fits above are of), and each DEVICE
 * additionally carries a working copy: 1.4 bytes per pair on Ornith, 3.0 on
 * Gemma-4, so the same 2 bytes is the honest middle and lands inside both. A
 * CPU-only run pays it once, in the CPU compute buffer.
 *
 * Not multiplied by `-np`: the tensor is `[n_kv, n_tokens/n_stream, 1,
 * n_stream]`, so more streams divide it rather than repeat it — unlike
 * `computeScratch`, whose per-slot graphs really are copies.
 *
 * Zero for a sparse-attention model: `computeScratch` measures that family end
 * to end, micro-batch term included, and stacking an estimate on a measurement
 * refuses placements the machine runs.
 */
export function attnMaskB(
  meta: ModelMeta,
  s: Settings,
  ctx: number,
  ubatch: number,
): number {
  if (whole(meta.indexerTopK) > 0) return 0;
  // `auto` is llama.cpp's default and it turns flash attention ON wherever the
  // backend has the kernel, so the f32 mask is only the answer when the user
  // has switched it OFF. Guessing the other way would double this term for
  // almost every run.
  const fp32 = str(s, "flashAttn") === "off";
  return whole(ctx) * Math.max(1, ubatch) * (fp32 ? 4 : 2);
}

/**
 * Per context token, per device, for a hybrid linear-attention model.
 *
 * The delta-net trunk keeps a working set that grows with the context and is
 * not the KV cache (that is `kvTotal`) and not the recurrent state (that is
 * `recurrentStateB`, constant). Measured on Ornith-1.5-35B-A3B at `-ub 512`,
 * one card, everything else held still, as the residual once the flat backend
 * figure, the activations and the mask above are taken out:
 *
 *     ctx       measured CUDA0   residual over the rest
 *     32,768      492 MiB         78 MiB
 *     65,536      524 MiB         78 MiB
 *     131,072     642 MiB        132 MiB
 *     262,144   1,026 MiB        388 MiB
 *
 * 2 KiB per context token covers the top of that with about a tenth to spare,
 * and over-bills the short contexts by ~100 MiB — which the flat backend figure
 * already dwarfs. Pessimism here costs context; optimism costs a failed load.
 *
 * Keyed on the model declaring recurrent geometry at all, so nothing changes
 * for the models the flat estimate already fitted.
 */
const HYBRID_SCRATCH_B_PER_CTX = 2 * 1024;

export function hybridScratch(meta: ModelMeta, ctx: number): number {
  if (whole(meta.indexerTopK) > 0) return 0;
  if (whole(meta.ssmDInner) <= 0 || whole(meta.fullAttnInterval) < 2) return 0;
  return whole(ctx) * HYBRID_SCRATCH_B_PER_CTX;
}

/** The flat, per-device part of a sparse-attention model's compute buffer. */
export function scratchFloor(meta: ModelMeta): number {
  return whole(meta.indexerTopK) > 0 ? SCRATCH_FLOOR_PER_DEVICE_B : 0;
}

/**
 * @param slots `-np`. Each server slot runs its own graph, so each one costs
 * another copy of the context-sized tensors. Defaulting this to 1 would be the
 * old bug in a new place, so the caller passes what the argv will say.
 */
export function computeScratch(
  meta: ModelMeta,
  ubatch: number,
  ctx: number,
  slots = 1,
): number {
  if (whole(meta.indexerTopK) <= 0) return 0;
  const ub = Math.max(1, ubatch);
  const np = Math.max(1, whole(slots));
  return whole(
    whole(ctx) * (SCRATCH_B_PER_CTX + SCRATCH_B_PER_CTX_UBATCH * ub) * np,
  );
}

function fmtGb(b: number): string {
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}
