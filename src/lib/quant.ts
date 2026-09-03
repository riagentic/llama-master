// src/lib/quant.ts — "would a smaller quantisation be faster, and by how much?"
//
// The largest single speed lever this app has, and the one it used to be silent
// about: generation is memory-bandwidth bound, so halving the bytes of every
// weight very nearly doubles the tokens per second. The app knew the quant
// (it draws it as a pill) and mentioned a smaller one only when the model did
// NOT fit — so a user running a Q8 that fits comfortably was never told that a
// Q4 of the same model would run at close to twice the rate.
//
// Two halves, and only one of them is an estimate.
//
// **Exact:** what THIS file weighs per weight. `rust/src/gguf.rs` walks the
// tensor table and reports `tensorBytes` and `params`, so
// `tensorBytes * 8 / params` is this file's real bits per weight — not a
// lookup against its label. That matters because the label is a MIX: a
// "Q4_K_M" is mostly Q4_K with some Q6_K, and two files wearing that label are
// not the same size.
//
// **Estimated:** what a file of some OTHER quant would weigh. Nothing in this
// machine can know that — the file does not exist here — so `TYPICAL_BPW`
// below is what a real file of that quant usually weighs, and everything
// derived from it is labelled an estimate everywhere it is shown.
//
// What is NOT estimated is the consequence. Once the bytes are scaled, the
// candidate goes through the same `tune` → `plan` → `bytesPerToken` →
// `estimateTps` the app uses for the model it is actually running, on the same
// machine, with the same reserve and the same measured bandwidth. That is what
// makes the answer worth reading: the interesting case is not "half the bytes,
// twice the speed", it is the model that stops spilling into system RAM
// altogether and goes four times faster, or the one that suddenly holds its
// full context. A ratio could not find either.

import type { LayerBytes, ModelMeta, Settings } from "./types.ts";
import type { Hw } from "./types.ts";
import { plan as computePlan } from "./plan.ts";
import { bytesPerToken, estimateTps } from "./speed.ts";
import { bestPlacement, type Placement, tune, type Tuning } from "./tune.ts";

/**
 * What a real file of each quantisation weighs, in bits per weight.
 *
 * Every one of these is a MIX, which is why they are not the block sizes in
 * `rust/src/gguf.rs`: `Q4_K_M` means the bulk of the tensors at Q4_K (4.5
 * bits) with the attention `v` and the feed-forward `down` at Q6_K, and the
 * embedding and output kept larger still. The figures below are what such
 * files actually come out at, and they vary by a few hundredths between one
 * publisher and the next — which is exactly why the CURRENT file's figure is
 * measured rather than looked up here, and why everything computed from this
 * table says "about".
 *
 * Ordered big to small; that is also the order they are offered in.
 */
export const TYPICAL_BPW: ReadonlyArray<{ quant: string; bpw: number }> = [
  { quant: "F32", bpw: 32 },
  { quant: "BF16", bpw: 16 },
  { quant: "F16", bpw: 16 },
  { quant: "Q8_0", bpw: 8.5 },
  { quant: "Q6_K", bpw: 6.56 },
  { quant: "Q5_K_M", bpw: 5.69 },
  { quant: "Q5_K_S", bpw: 5.52 },
  { quant: "Q4_K_M", bpw: 4.85 },
  { quant: "Q4_K_S", bpw: 4.58 },
  { quant: "IQ4_NL", bpw: 4.5 },
  { quant: "IQ4_XS", bpw: 4.25 },
  { quant: "Q3_K_L", bpw: 4.03 },
  { quant: "Q3_K_M", bpw: 3.74 },
  { quant: "IQ3_M", bpw: 3.66 },
  { quant: "Q3_K_S", bpw: 3.44 },
  { quant: "IQ3_XXS", bpw: 3.06 },
  { quant: "Q2_K", bpw: 2.63 },
  { quant: "IQ2_M", bpw: 2.7 },
  { quant: "IQ2_XS", bpw: 2.31 },
];

/**
 * The quants worth OFFERING, which is a shorter list than the one above.
 *
 * Two rules. Nothing above Q8_0: a 16-bit file is twice the bytes of a Q8 that
 * is indistinguishable from it, so "go bigger" is never the advice this panel
 * should give. And nothing below Q3_K_M by default, because past there the
 * model stops being the model — the app will still SIZE an IQ2 if the user is
 * looking at one, it just does not recommend the trip.
 *
 * `Q4_K_M` first among equals: it is the one the ecosystem publishes for
 * everything, so it is the one a link will actually exist for.
 */
export const OFFERED = [
  "Q8_0",
  "Q6_K",
  "Q5_K_M",
  "Q4_K_M",
  "IQ4_XS",
  "Q3_K_M",
] as const;

/** Bits per weight for a quant LABEL, or 0 when it is not one we know. */
export function typicalBpw(quant: string): number {
  const q = quant.trim().toUpperCase();
  return TYPICAL_BPW.find((t) => t.quant === q)?.bpw ?? 0;
}

/**
 * This file's REAL bits per weight — measured, not looked up.
 *
 * `params` arrived with this module; a model read by an older build reports 0,
 * and then the label's typical figure is the only thing available. That is a
 * fallback and the caller is told (`exact: false`), because the whole point of
 * the number is that it is a fact about the file in front of us.
 */
export function fileBpw(meta: ModelMeta): { bpw: number; exact: boolean } {
  const bytes = meta.tensorBytes;
  const params = meta.params;
  if (
    Number.isFinite(bytes) && Number.isFinite(params) && bytes > 0 && params > 0
  ) {
    const bpw = (bytes * 8) / params;
    // A sane file lands between 1 and 32 bits. Anything outside that is a
    // truncated read or a type we could not size, and a wrong bpw would
    // mis-scale every row of the table below.
    if (bpw >= 1 && bpw <= 32) return { bpw, exact: true };
  }
  const typical = typicalBpw(meta.quant);
  return { bpw: typical, exact: false };
}

/**
 * The same model at a different weight, for the planner to place.
 *
 * Every byte field scales by one factor, because that is what re-quantising
 * does to a file: the tensor table keeps its shape and each entry gets
 * smaller. It is an approximation in one respect — the norms are F32 in every
 * quant and do not shrink — and that is already inside the factor, which comes
 * from whole-file bits per weight rather than from one tensor's type.
 *
 * `nExpert`, the layer count and every geometry field are untouched: this is
 * the same model, so the KV cache, the context and the compute buffers are the
 * same. Only the weights move.
 */
export function rescale(meta: ModelMeta, toBpw: number): ModelMeta {
  const from = fileBpw(meta).bpw;
  if (!(from > 0) || !(toBpw > 0)) return meta;
  const k = toBpw / from;
  const layers: LayerBytes[] = meta.layers.map((l) => ({
    ...l,
    bytes: Math.round(l.bytes * k),
    expert: Math.round(l.expert * k),
  }));
  return {
    ...meta,
    quant: meta.quant,
    tensorBytes: Math.round(meta.tensorBytes * k),
    embdBytes: Math.round(meta.embdBytes * k),
    outputBytes: Math.round(meta.outputBytes * k),
    layers,
  };
}

/** One row of the comparison: a quant, what it would weigh, and what it would do. */
export type QuantOption = {
  quant: string;
  bpw: number;
  /** Weights on disk, estimated (the current row is the real file size). */
  bytesB: number;
  /** This is the file that is loaded. */
  current: boolean;
  /** Can it run at all on this machine? */
  possible: boolean;
  /** Where the tuner would put it. */
  placement: Placement;
  /** Context the tuner reached. */
  ctx: number;
  /** Estimated generation rate, tokens/second. 0 when it cannot run. */
  tps: number;
  /** `tps` against the current file's `tps`. 1 = the same. */
  speedup: number;
  /** Layers the tuner had to leave in system RAM — the thing that actually
   *  decides the speed when a model is near the edge of the cards. */
  layersInRam: number;
};

/** How much quality a step down from `fromBpw` to `toBpw` costs, in words.
 *
 *  A rule of thumb and labelled as one: measuring it needs the model, both
 *  files and a benchmark suite, none of which are here. What IS defensible is
 *  the shape of the curve, which every published perplexity table agrees on —
 *  flat from 8 bits to about 5, a gentle bend through 4, and a cliff below 3.
 */
export function qualityNote(toBpw: number): string {
  if (toBpw >= 6.5) return "No practical quality loss.";
  if (toBpw >= 5) return "Quality loss is not measurable in normal use.";
  if (toBpw >= 4.5) {
    return "A small quality loss — the usual choice, and the one most people run.";
  }
  if (toBpw >= 4) return "A small but real quality loss.";
  if (toBpw >= 3.5) {
    return "A noticeable quality loss, especially on code and long reasoning.";
  }
  return "A large quality loss — the model starts making mistakes it would not make otherwise.";
}

/**
 * Every quant worth considering for this model on this machine, ranked as they
 * would actually run.
 *
 * The current file is always a row, marked `current`, and always sized from the
 * REAL header rather than from the table — so the number the rest of the app
 * shows and the number in this table are the same number.
 *
 * `cal` is the machine's measured bandwidth when it has any. Without it every
 * row uses the same labelled defaults, so the ratios between rows stay honest
 * even when the absolute figures are not: this table's job is "which of these
 * is faster, and by roughly how much", and that survives an uncalibrated
 * machine intact.
 */
export function quantOptions(
  meta: ModelMeta,
  hw: Hw,
  settings: Settings,
  cal: { gpuBps?: number; ramBps?: number } = {},
): QuantOption[] {
  const here = fileBpw(meta);
  if (!(here.bpw > 0)) return [];

  const rowFor = (
    quant: string,
    bpw: number,
    m: ModelMeta,
    current: boolean,
  ): QuantOption => {
    // The same decision the app makes for the model it is running: try the
    // placement the user chose, fall back to the best one that can run it.
    // Anything else would compare a Q8 the tuner refused against a Q4 it
    // placed, and call the difference "quantisation".
    const all = {
      vram: tune(m, hw, settings, "vram"),
      hybrid: tune(m, hw, settings, "hybrid"),
      cpu: tune(m, hw, settings, "cpu"),
    } as Record<Placement, Tuning>;
    const best = bestPlacement(all);
    const t = all[best];
    if (!t.possible) {
      return {
        quant,
        bpw,
        bytesB: m.tensorBytes,
        current,
        possible: false,
        placement: best,
        ctx: 0,
        tps: 0,
        speedup: 0,
        layersInRam: m.nLayer,
      };
    }
    const p = computePlan(m, hw, t.settings);
    const b = bytesPerToken(m, p, t.settings, t.ctx);
    return {
      quant,
      bpw,
      bytesB: m.tensorBytes,
      current,
      possible: true,
      placement: best,
      ctx: t.ctx,
      tps: estimateTps({ ...b, gpuBps: cal.gpuBps, ramBps: cal.ramBps }),
      speedup: 1,
      layersInRam: Math.max(0, m.nLayer - p.layersOnGpu),
    };
  };

  const mine = rowFor(meta.quant || "this file", here.bpw, meta, true);

  const rows: QuantOption[] = [mine];
  for (const quant of OFFERED) {
    const bpw = typicalBpw(quant);
    if (!(bpw > 0)) continue;
    // Within 2% of the file we already have is the same file: offering
    // "Q4_K_S" beside a "Q4_K_M" that weighs the same teaches nothing and
    // makes the table longer than the decision it supports.
    if (Math.abs(bpw - here.bpw) / here.bpw < 0.02) continue;
    // Bigger than what is loaded is never the advice — see `OFFERED`.
    if (bpw > here.bpw) continue;
    rows.push(rowFor(quant, bpw, rescale(meta, bpw), false));
  }

  const baseline = mine.tps;
  for (const r of rows) {
    r.speedup = baseline > 0 && r.tps > 0 ? r.tps / baseline : 0;
  }
  // Biggest first, so the table reads down from what is loaded to what is
  // smaller and faster — the direction the decision runs in.
  rows.sort((a, b) => b.bpw - a.bpw);
  return rows;
}

/**
 * The bits per weight below which quality stops being free.
 *
 * Every published perplexity table has the same shape: flat from 8 bits down
 * to about 5, a gentle bend through 4, then a cliff. `Q4_K_M` (4.85) sits just
 * above the bend and is what the ecosystem publishes for everything, which is
 * why it is the file a link will actually exist for.
 */
const SAFE_BPW = 4.5;

/**
 * Which quant to actually NAME, out of the ones that would be faster.
 *
 * Not simply the fastest. The fastest row is always the smallest file, so
 * recommending it means recommending the most damaged version of the model
 * every single time — on a 70B that was "Q3_K_M, 4.3× faster" where Q4_K_M was
 * already 2.3× at a quality cost most people would not notice. Advice that
 * always points at the bottom of the table is not advice, it is a slider.
 *
 * Nor is it the largest that clears the bar — that picks the row nearest the
 * top, which on the same 70B was "Q6_K, 1.3× faster": true, free, and not
 * worth a download.
 *
 * So it aims AT the knee: of the files that are worth naming at all, the
 * smallest one still above `SAFE_BPW`. That is the most speed available before
 * quality starts costing something, which is the trade almost everyone wants
 * made for them. The table underneath carries every other row with its price
 * on it, so someone who wants the last 2× can read down and take it knowingly.
 */
function recommend(rows: readonly QuantOption[]): QuantOption | undefined {
  const mine = rows.find((r) => r.current);
  const usable = rows.filter((r) => !r.current && r.possible && r.tps > 0);
  if (usable.length === 0) return undefined;
  const worthIt = (r: QuantOption) => {
    // Against a file that does not run, ANY file that runs is worth naming.
    if (!mine || !mine.possible || mine.tps <= 0) return true;
    return r.tps / mine.tps >= 1.25 ||
      (mine.layersInRam > 0 && r.layersInRam === 0);
  };
  // Smallest first: the first one at or above the knee IS the knee.
  const bySize = usable.slice().sort((a, b) => a.bpw - b.bpw);
  return bySize.find((r) => r.bpw >= SAFE_BPW && worthIt(r)) ??
    // Nothing at the knee helps, so the only useful answers are below it.
    // Fastest wins there: past the bend, quality is already being paid for.
    usable.slice().sort((a, b) => b.tps - a.tps).find(worthIt);
}

/**
 * The one sentence worth putting on screen when a smaller quant is clearly
 * better, or "" when it is not.
 *
 * Deliberately quiet. A line that appears for every model is a line nobody
 * reads, so this speaks only when the gain is one a person would feel: a
 * quarter faster, or a model that stops spilling into system RAM. And it never
 * speaks about quality without saying which way — the gain and its price are
 * one sentence, or the advice is only half given.
 */
export function quantAdvice(rows: readonly QuantOption[]): string {
  const mine = rows.find((r) => r.current);
  if (!mine) return "";
  const better = recommend(rows);
  if (!better) return "";

  // A rate below ten needs its decimal: rounding 0.9 and 3.8 to whole numbers
  // printed "1 → 4 tokens/s" for a fourfold difference, which reads as a
  // rounding artefact rather than as the headline it is.
  const rate = (t: number) =>
    t >= 10 ? Math.round(t).toLocaleString() : t.toFixed(1);

  // The loudest case, and the one that used to come out as nonsense: the file
  // on disk does not fit at all, so there is no rate for the other one to be a
  // multiple OF — `mine.tps` is 0 and the ratio printed "about 0.0× faster".
  // A refusal is not a slow run, and it gets its own sentence.
  if (!mine.possible || mine.tps <= 0) {
    return `This file does not fit on this machine, but a ${better.quant} of the same model would — at about ${
      rate(better.tps)
    } tokens/s. ${qualityNote(better.bpw)}`;
  }

  const gain = better.tps / mine.tps;
  const spillFixed = mine.layersInRam > 0 && better.layersInRam === 0;
  if (gain < 1.25 && !spillFixed) return "";
  const speed = `about ${gain.toFixed(1)}× faster (${rate(mine.tps)} → ${
    rate(better.tps)
  } tokens/s, estimated)`;
  const why = spillFixed
    ? `, because it fits entirely in VRAM and this one does not`
    : "";
  return `A ${better.quant} of this model would be ${speed}${why}. ${
    qualityNote(better.bpw)
  }`;
}
