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
  /** Plain-language observations, ordered most important first. */
  notes: string[];
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
function whole(n: number): number {
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
  buckets: Bucket[],
): Pool {
  const usedB = sum(buckets.map((b) => b.bytes));
  const total = usedB + otherB;
  return {
    label,
    capacityB,
    buckets: buckets.filter((b) => b.bytes > 0),
    usedB,
    otherB,
    freeB: Math.max(0, capacityB - total),
    overB: Math.max(0, total - capacityB),
  };
}

/** VRAM currently held by anything other than the run we are planning. */
function vramInUse(gpus: Gpu[]): number {
  return sum(gpus.map((g) => g.vramUsedB));
}

/**
 * Place a model under one settings map on one machine.
 *
 * Placement rules mirror llama.cpp:
 * - `-ngl N` offloads the **last** N transformer layers.
 * - `-ngl > nLayer` also offloads the output head and the embedding table.
 * - `--n-cpu-moe N` keeps the routed experts of the **first** N layers in RAM,
 *   even when those layers are otherwise on the GPU.
 * - `-nkvo` moves the whole KV cache to RAM regardless of layer placement.
 */
export function plan(meta: ModelMeta, hw: Hw, s: Settings): Plan {
  const nLayer = meta.nLayer;
  const ngl = num(s, "ngl");
  const layersOnGpu = Math.max(0, Math.min(ngl, nLayer));
  const fullOffload = ngl > nLayer;
  const moeOnCpu = Math.max(0, Math.min(num(s, "nCpuMoe"), nLayer));
  const ctx = effectiveCtx(meta, s);
  const kvPerTokenB = kvPerToken(meta, s);
  const kvTotalB = kvTotal(meta, s, ctx);
  const kvOnCpu = bool(s, "noKvOffload");

  // Layer placement: the last `layersOnGpu` indices go to the GPU.
  const firstGpuLayer = nLayer - layersOnGpu;
  let gpuDense = 0;
  let gpuExperts = 0;
  let cpuWeights = 0;
  for (const l of meta.layers) {
    // Same reasoning as `kvPerToken`: these come out of the file, and a layer
    // whose experts are larger than the layer itself would otherwise make the
    // dense figure negative and every total after it wrong.
    const bytes = whole(l.bytes);
    const expert = Math.min(whole(l.expert), bytes);
    const dense = bytes - expert;
    const onGpu = l.i >= firstGpuLayer;
    const expertsHere = l.i < moeOnCpu ? 0 : expert;
    if (onGpu) {
      gpuDense += dense;
      gpuExperts += expertsHere;
      cpuWeights += expert - expertsHere;
    } else {
      cpuWeights += bytes;
    }
  }
  if (fullOffload) {
    gpuDense += whole(meta.outputBytes) + whole(meta.embdBytes);
  } else {
    cpuWeights += meta.outputBytes + meta.embdBytes;
  }

  // KV follows its layer, unless -nkvo pins all of it to the host.
  const kvGpuShare = nLayer > 0 ? layersOnGpu / nLayer : 0;
  const kvOnGpu = kvOnCpu ? 0 : kvTotalB * kvGpuShare;
  const kvOnRam = kvTotalB - kvOnGpu;

  // Compute buffers scale with the micro-batch, not the batch: llama.cpp runs
  // `-ub` tokens at a time. Four activation-sized tensors is the empirical
  // shape of the graph; the backend context is a flat per-process cost.
  const ubatch = Math.max(1, num(s, "ubatchSize"));
  const activation = ubatch * whole(meta.nEmbd) * 4;
  const usingGpu = layersOnGpu > 0 || fullOffload;
  const gpuCompute = usingGpu
    ? activation * 4 + BACKEND_CONTEXT_B * Math.max(1, hw.gpus.length)
    : 0;
  const cpuCompute = layersOnGpu < nLayer || !usingGpu
    ? activation * 2
    : 32 * MB;

  const vramCapacity = sum(hw.gpus.map((g) => g.vramTotalB));
  const ramCapacity = hw.mem?.totalB ?? 0;
  const ramOther = hw.mem ? hw.mem.totalB - hw.mem.availableB : 0;

  const vram = pool("VRAM", vramCapacity, vramInUse(hw.gpus), [
    { key: "weights", label: "Weights", bytes: gpuDense },
    { key: "experts", label: "Experts", bytes: gpuExperts },
    { key: "kv", label: "KV cache", bytes: kvOnGpu },
    { key: "compute", label: "Compute (est.)", bytes: gpuCompute },
  ]);
  const ram = pool("RAM", ramCapacity, ramOther, [
    { key: "weights", label: "Weights", bytes: cpuWeights },
    { key: "kv", label: "KV cache", bytes: kvOnRam },
    { key: "compute", label: "Compute (est.)", bytes: cpuCompute },
  ]);

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
  if (hw.gpus.length === 0) {
    notes.push("No GPU detected — everything runs on the CPU.");
  } else if (layersOnGpu === 0 && !fullOffload) {
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
    layersOnGpu: fullOffload ? nLayer : layersOnGpu,
    moeOnCpu,
    ctx,
    vram,
    ram,
    kvPerTokenB,
    kvTotalB,
    fits: vram.overB === 0 && ram.overB === 0,
    notes,
  };
}

function fmtGb(b: number): string {
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}
