// src/lib/speed.ts — how fast this is going to be, and whether that is usable.
//
// Token generation is MEMORY-BANDWIDTH bound, not compute bound. To emit one
// token llama.cpp reads every active weight and the whole KV cache once, so the
// rate is very close to
//
//     tokens/second  =  effective bandwidth  ÷  bytes read per token
//
// and the bytes are a number this app already knows exactly: the GGUF header is
// walked tensor by tensor, and `plan.ts` places every one of them. That is the
// half of the estimate that is arithmetic.
//
// The other half — bandwidth — is the honest weak point. Nothing in
// `nvidia-smi` reports memory bus width, so a card's peak bandwidth cannot be
// read from the machine, and even if it could, the achieved fraction depends on
// the kernel. So this module does the same thing the rest of the app does with
// numbers it cannot measure: it uses a labelled default until it has a real
// observation, and prefers the observation the moment one exists. Every run
// through the chat produces one (`chat.lastTps`), so a machine calibrates itself
// after a single reply.
//
// Pure: bytes and bandwidths in, tokens per second out.

import type { ModelMeta, Settings } from "./types.ts";
import type { Plan } from "./plan.ts";
import { num, str } from "./params.ts";

/**
 * Bytes that have to be read to produce ONE token, split by where they live.
 *
 * Not the model's size: a mixture-of-experts model reads only the experts the
 * router picked (`nExpertUsed` of `nExpert`), which is the entire reason a 35B
 * A3B model generates at the speed of something far smaller. Dense layers,
 * embeddings and the output head are read whole, and so is the KV cache — every
 * token attends over everything already in the context, which is why a long
 * conversation gets slower as it goes.
 *
 * `ctx` is the context ACTUALLY filled, not the configured maximum: the cache is
 * allocated up front but only the occupied part is read. Callers that want the
 * worst case pass the configured size.
 */
export function bytesPerToken(
  meta: ModelMeta,
  hwPlan: Plan,
  s: Settings,
  ctxFilled: number,
): { gpuB: number; ramB: number; totalB: number } {
  const nLayer = Math.max(0, meta.nLayer);
  const onGpu = Math.max(0, Math.min(hwPlan.layersOnGpu, nLayer));
  const moeOnCpu = Math.max(0, Math.min(hwPlan.moeOnCpu, nLayer));
  // Fraction of each layer's expert bytes actually read. Dense models have no
  // experts and this term is zero.
  const used = meta.nExpert > 0
    ? Math.min(1, Math.max(0, meta.nExpertUsed / meta.nExpert))
    : 0;

  let gpuB = 0;
  let ramB = 0;
  for (let i = 0; i < nLayer; i++) {
    const layer = meta.layers[i];
    if (!layer) continue;
    const expert = Math.max(0, layer.expert);
    // `bytes` includes the expert bytes; the rest is attention and MLP.
    const dense = Math.max(0, layer.bytes - expert);
    const activeExpert = expert * used;
    // Layers are placed from the END of the model (see plan.ts), so the last
    // `onGpu` indices are the resident ones.
    const layerOnGpu = i >= nLayer - onGpu;
    // `--n-cpu-moe` holds the FIRST N layers' experts in RAM.
    const expertOnGpu = layerOnGpu && i >= moeOnCpu;
    if (layerOnGpu) gpuB += dense;
    else ramB += dense;
    if (expertOnGpu) gpuB += activeExpert;
    else ramB += activeExpert;
  }

  // Embeddings and the output head follow the layers onto the GPU.
  const ends = Math.max(0, meta.embdBytes) + Math.max(0, meta.outputBytes);
  if (onGpu >= nLayer && nLayer > 0) gpuB += ends;
  else ramB += ends;

  // The KV cache is read in whole every token. `-nkvo` forces it to host RAM.
  const kv = Math.max(0, hwPlan.kvPerTokenB) * Math.max(0, ctxFilled);
  if (str(s, "noKvOffload") === "true" || num(s, "ngl") === 0 || onGpu === 0) {
    ramB += kv;
  } else {
    // The cache follows the layers: the share on the GPU is the share of layers.
    const gpuShare = nLayer > 0 ? onGpu / nLayer : 0;
    gpuB += kv * gpuShare;
    ramB += kv * (1 - gpuShare);
  }

  return { gpuB, ramB, totalB: gpuB + ramB };
}

/**
 * Effective bandwidth to assume when nothing has been measured yet.
 *
 * Deliberately conservative and deliberately round: these are not this machine's
 * numbers, they are "a modern card" and "a modern desktop", and the UI labels
 * anything derived from them as an estimate. A single chat reply replaces them
 * with the truth for this machine (`calibrate`).
 *
 * VRAM: a current mid-to-high discrete card sustains a few hundred GB/s.
 * System RAM: dual-channel DDR5 peaks near 80 GB/s and llama.cpp achieves well
 * under that, which is exactly why hybrid placement is so much slower than it
 * looks from the layer count alone.
 */
export const DEFAULT_GPU_BPS = 400 * 1024 ** 3;
export const DEFAULT_RAM_BPS = 40 * 1024 ** 3;

/** Tokens per second for a placement, or 0 when there is nothing to go on. */
export function estimateTps(args: {
  gpuB: number;
  ramB: number;
  gpuBps?: number;
  ramBps?: number;
}): number {
  const gpuBps = args.gpuBps && args.gpuBps > 0 ? args.gpuBps : DEFAULT_GPU_BPS;
  const ramBps = args.ramBps && args.ramBps > 0 ? args.ramBps : DEFAULT_RAM_BPS;
  const gpuB = Math.max(0, args.gpuB);
  const ramB = Math.max(0, args.ramB);
  if (gpuB + ramB <= 0) return 0;
  // The two reads are sequential per token, so the seconds add. This is why one
  // layer left in RAM costs far more than 1/nLayer of the speed.
  const seconds = gpuB / gpuBps + ramB / ramBps;
  if (!Number.isFinite(seconds) || seconds <= 0) return 0;
  return 1 / seconds;
}

/**
 * Turn one real measurement into this machine's effective bandwidth.
 *
 * Called with a rate the app actually observed and the bytes that produced it.
 * Attributing the whole result to whichever pool holds most of the bytes is
 * crude, but it is a MEASUREMENT of this machine rather than a guess about it,
 * and it converges: a VRAM-only run calibrates the GPU figure, a CPU-only run
 * calibrates RAM, and a hybrid run is ignored because it cannot separate them.
 */
export function calibrate(
  observedTps: number,
  bytes: { gpuB: number; ramB: number },
): { gpuBps?: number; ramBps?: number } {
  if (!Number.isFinite(observedTps) || observedTps <= 0) return {};
  const { gpuB, ramB } = bytes;
  const total = gpuB + ramB;
  if (total <= 0) return {};
  // Only a run that lives almost entirely in one pool tells us about that pool.
  const PURE = 0.98;
  if (gpuB / total >= PURE) return { gpuBps: total * observedTps };
  if (ramB / total >= PURE) return { ramBps: total * observedTps };
  return {};
}

/** How usable a generation rate actually is. */
export type TpsBand = "poor" | "ok" | "great";

/**
 * The thresholds, and why they are where they are.
 *
 * Anchored on reading speed rather than on hardware, because the question the
 * user is asking is "will this be pleasant to use". Silent reading runs about
 * 250 words per minute, and a token averages rather less than a word — so
 * roughly **5 tokens/second is the pace you read at**. Below that you are
 * waiting for the model; at 20 and above the text arrives faster than you can
 * take it in, and more speed stops being something you notice.
 */
export const TPS_POOR_BELOW = 5;
export const TPS_GREAT_AT = 20;

export function tpsBand(tps: number): TpsBand {
  if (!Number.isFinite(tps) || tps < TPS_POOR_BELOW) return "poor";
  if (tps >= TPS_GREAT_AT) return "great";
  return "ok";
}

export function tpsLabel(band: TpsBand): string {
  return band === "poor" ? "slow" : band === "great" ? "fast" : "usable";
}

export function tpsWhy(band: TpsBand): string {
  return band === "poor"
    ? `Under ${TPS_POOR_BELOW} tokens/s — slower than you read, so you will be waiting for it.`
    : band === "great"
    ? `${TPS_GREAT_AT} tokens/s and up — faster than you can read, which is as fast as it needs to be.`
    : `Between ${TPS_POOR_BELOW} and ${TPS_GREAT_AT} tokens/s — keeps up with reading, and you will notice it working.`;
}
