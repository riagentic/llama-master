// src/lib/devsplit.ts — which CARD holds which layer.
//
// A second GPU is not a bigger GPU, and this is where that stops being a
// slogan. llama.cpp offloads a contiguous run of layers and cuts it into
// per-device ranges BY COUNT: `--tensor-split` (or, by default, each device's
// free memory) is normalised into cumulative fractions, and layer `i` goes to
// the first device whose fraction exceeds `i / n_offloaded`
// (`llama-model.cpp:load_tensors`, the `get_layer_buft_list` lambda).
//
// By count. Not by bytes. On a dense model those are the same thing and nobody
// notices. On a mixture-of-experts model with `--n-cpu-moe N` they are wildly
// different: that flag holds the experts of the FIRST N layers in RAM, so every
// layer that still owns its experts is at the END of the model — and the end of
// the model is the LAST device. A plan that fits in 47.8 GB of aggregate VRAM
// asked one 24 GB card for 34 GB:
//
//   ggml_backend_cuda_buffer_type_alloc_buffer: allocating 34020.32 MiB on
//   device 1: cudaMalloc failed: out of memory
//
// The aggregate was right and the placement was impossible. So the sizes are
// worked out per slot, packed into the cards in order, and the result is emitted
// as an explicit `-ts` — which also makes the placement deterministic instead of
// depending on whatever the driver reported as free at load time.

import type { Gpu, Settings } from "./types.ts";
import { num } from "./params.ts";

/**
 * The run of slots llama.cpp will offload.
 *
 * "Slot" rather than "layer" because `-ngl` counts the OUTPUT head as one:
 * there are `n_layer + 1` of them, and `-ngl 43` on a 43-layer model offloads
 * layers 1..42 plus the output, leaving layer 0 on the host. Only `-ngl > n_layer`
 * offloads everything. This is llama.cpp's arithmetic verbatim
 * (`i_gpu_start`, `act_gpu_layers`).
 */
export type Offload = {
  /** First slot on the GPU. */
  start: number;
  /** How many slots are offloaded. */
  count: number;
  /** `n_layer + 1` — the layers plus the output head. */
  slots: number;
};

export function offloadRange(nLayer: number, s: Settings): Offload {
  const slots = Math.max(0, nLayer) + 1;
  const ngl = Math.max(0, num(s, "ngl"));
  return {
    start: Math.max(slots - ngl, 0),
    count: Math.min(ngl, slots),
    slots,
  };
}

/** Is this slot on a GPU at all? Slot `nLayer` is the output head. */
export function slotOnGpu(slot: number, o: Offload): boolean {
  return slot >= o.start && slot - o.start < o.count;
}

/**
 * VRAM each card can be asked for, after its own reserve and what already
 * lives on it.
 *
 * Per card, not per machine: a reserve pooled across devices would let a plan
 * spend card 1's headroom on card 0, which is the same mistake one step down.
 *
 * `userReservesB` is the second reserve and a different thing: the fixed one
 * below exists so the allocator does not fail, this one is what the user asked
 * to keep for their desktop and their own work, already divided per card
 * (`src/lib/reserve.ts:vramReserveShares`). They add, because they defend
 * against different things.
 */
export function deviceBudgets(
  gpus: readonly Gpu[],
  overheadB = 0,
  userReservesB: readonly number[] = [],
): number[] {
  return gpus.map((g, i) => {
    const reserve = Math.max(512 * 1024 * 1024, g.vramTotalB * 0.05);
    const mine = Math.max(0, userReservesB[i] ?? 0);
    return Math.max(0, g.vramTotalB - g.vramUsedB - reserve - mine - overheadB);
  });
}

/**
 * Pack the offloaded slots into the cards, in order, and say where the cuts go.
 *
 * The ranges have to be contiguous and in order — that is llama.cpp's rule, not
 * a simplification — so this is a first-fit walk rather than a bin-packing
 * search: fill card 0 until the next slot would not fit, move on. With the
 * expensive slots at the end (which is exactly what `--n-cpu-moe` produces) that
 * gives the early cards the cheap layers and spreads the costly tail, which is
 * the shape that fits.
 *
 * `null` when no arrangement fits — one slot larger than every remaining card,
 * or simply too much model. The caller's answer to that is a different
 * placement, not a different split.
 */
export function packSlots(
  costsB: readonly number[],
  budgetsB: readonly number[],
): number[] | null {
  if (budgetsB.length === 0) return costsB.length === 0 ? [] : null;
  const counts = budgetsB.map(() => 0);
  let dev = 0;
  let used = 0;
  for (const cost of costsB) {
    while (dev < budgetsB.length && used + cost > (budgetsB[dev] ?? 0)) {
      dev++;
      used = 0;
    }
    if (dev >= budgetsB.length) return null;
    counts[dev] = (counts[dev] ?? 0) + 1;
    used += cost;
  }
  return counts;
}

/** Bytes each card ends up holding, given the cuts `packSlots` chose. */
export function loadPerDevice(
  costsB: readonly number[],
  counts: readonly number[],
): number[] {
  const out = counts.map(() => 0);
  let i = 0;
  for (let d = 0; d < counts.length; d++) {
    for (let n = 0; n < (counts[d] ?? 0); n++) {
      out[d] = (out[d] ?? 0) + (costsB[i++] ?? 0);
    }
  }
  return out;
}

/**
 * The `-ts` value that produces exactly these cuts.
 *
 * Half a slot is shaved off the first share and added to the last so every
 * boundary lands STRICTLY between two slot fractions. Emitting the counts
 * themselves puts the boundary exactly on `k / n_offloaded`, where llama.cpp's
 * `upper_bound` is deciding on an equality between two floats it computed
 * separately — correct today, and one refactor upstream from being off by one
 * card. Midpoints cannot be off by one.
 *
 * Empty when there is nothing to say: one card, or one card doing all the work.
 */
export function tensorSplitValue(counts: readonly number[]): string {
  const used = counts.filter((c) => c > 0).length;
  if (counts.length < 2 || used < 2) return "";
  const last = counts.length - 1;
  return counts
    .map((c, i) => (i === 0 ? c - 0.5 : i === last ? c + 0.5 : c))
    .map((v) => (Number.isInteger(v) ? String(v) : v.toFixed(1)))
    .join(",");
}
