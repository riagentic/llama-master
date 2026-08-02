// src/lib/fitladder.ts — when the plan cannot be predicted, measure it.
//
// Everything else in `src/lib` computes what a model will need. This file exists
// because for some models that computation is not possible from the file alone,
// and pretending otherwise produces a server that does not start.
//
// The case that forced it: DeepSeek-V4-Flash declares a 1,048,576-token trained
// context, and at that context llama.cpp asked for a 68.5 GiB COMPUTE buffer on
// one card — against the ~730 MB `plan.ts` estimates, because the buffer for a
// sparse-attention model scales with the context and nothing in a GGUF header
// says by how much. The KV cache was out by a similar factor. No division of the
// layers across the cards can rescue a 68 GiB scratch buffer, so this was not a
// placement problem and no placement search could have found it.
//
// llama.cpp has its own fitter for exactly this (`-fit`, which measures instead
// of predicting) and it segfaulted on this model, so it cannot be delegated to
// either.
//
// What is left is the honest thing: START, and if it fails for want of memory,
// halve the context and start again — then REMEMBER what worked, so the second
// run of that model goes straight there. The estimate stays the opening bid; the
// machine has the final say.
//
// Pure: log lines and numbers in, a decision out.

import { MIN_CTX } from "./tune.ts";

/**
 * How many times a start may be retried at a smaller context before the app
 * stops and asks.
 *
 * Six, because five is what this actually took. DeepSeek-V4-Flash declares
 * 1,048,576 and needs 32,768 on a 48 GB machine — 1M → 512k → 256k → 128k →
 * 64k → 32k. Four rungs would have stopped at 64k, which was MEASURED to fail,
 * and reported defeat one step from the answer.
 *
 * It is bounded at all because each rung loads the whole model: on a 145 GB file
 * that is ~2 minutes, so six is about twelve. That is the price of a header that
 * cannot be trusted, paid ONCE — the working context is written down and the
 * next start opens there (`cfg.fitCtx`).
 */
export const MAX_FIT_RETRIES = 6;

/** Contexts are searched on the same grid the tuner uses. */
const CTX_STEP = 256;

/**
 * Did this run die because something did not fit in memory?
 *
 * Deliberately narrow. A smaller context is the answer to an allocation that
 * failed; it is not the answer to a missing file, a rejected flag or a segfault,
 * and retrying those four times over would just be four slow ways to show the
 * same error. The shapes below are the ones that a shorter context actually
 * fixes: the weights buffer, the KV cache, the compute graph — and the CUDA
 * pool.
 *
 * The pool shape matters because it is the one that fires AFTER the server
 * looked fine: CUDA allocates scratch (activation-quantise buffers, cuBLAS
 * workspace, graphs) lazily at the first real forward pass, and when that
 * allocation fails the log says `CUDA error: out of memory` from whatever CUDA
 * call happened to observe it — captured live: `cudaFuncGetAttributes` inside
 * `quantize_row_q8_1_cuda`, with the word "buffer" nowhere in sight. Requiring
 * "buffer" here made the ladder blind to exactly the failure it exists for; a
 * DeepSeek-V4 run passed /health, OOM'd on its first prompt, and was reported
 * as a driver mismatch instead of retried.
 */
export function isFitFailure(lines: readonly string[]): boolean {
  const text = lines.join("\n");
  if (
    !/cudaMalloc failed|failed to allocate|out of memory|ggml_gallocr/i.test(
      text,
    )
  ) {
    return false;
  }
  return /buffer|kv cache|compute buffers|tensor_range|CUDA error|ggml-cuda|vk_|vulkan/i
    .test(text);
}

/**
 * The next context to try, or 0 when there is no smaller one worth trying.
 *
 * Halving rather than stepping: the terms that overflow — the KV cache and the
 * attention scratch — are proportional to the context, so halving halves the
 * thing that is too big. Walking down in small steps would spend several
 * whole-model loads to discover what one halving establishes.
 */
export function retryCtx(current: number): number {
  const half = Math.floor(current / 2 / CTX_STEP) * CTX_STEP;
  return half >= MIN_CTX ? half : 0;
}

/** What the app should do about a run that has just ended. */
export type FitDecision =
  | { kind: "none" }
  | { kind: "retry"; ctx: number; attempt: number; note: string };

/**
 * Should the app try again, and at what context?
 *
 * `tries` is how many automatic retries have already happened for this start —
 * reset whenever the user presses Start themselves, so a manual attempt always
 * gets the full ladder and never inherits an exhausted one.
 */
export function fitDecision(args: {
  lines: readonly string[];
  ctx: number;
  tries: number;
  /** Off when the user is driving the settings by hand: an automatic retry
   *  would silently overwrite a context they chose on purpose. */
  auto: boolean;
}): FitDecision {
  if (!args.auto || args.tries >= MAX_FIT_RETRIES) return { kind: "none" };
  if (!isFitFailure(args.lines)) return { kind: "none" };
  const next = retryCtx(args.ctx);
  if (next <= 0) return { kind: "none" };
  return {
    kind: "retry",
    ctx: next,
    attempt: args.tries + 1,
    note:
      `${args.ctx.toLocaleString()} tokens did not fit in memory — retrying at ${next.toLocaleString()}. ` +
      `What a model of this shape needs cannot be read from its header, so the app measures it instead, and remembers.`,
  };
}

/**
 * The context in an argv, or 0 when it does not say one.
 *
 * The retry rewrites the command that was actually run rather than re-composing
 * one from settings: re-composing would quietly pick up anything the user has
 * changed since pressing Start, so the "same command, one number smaller"
 * promise would not hold.
 */
export function ctxOf(argv: readonly string[]): number {
  for (let i = 0; i < argv.length - 1; i++) {
    if (argv[i] === "-c" || argv[i] === "--ctx-size") {
      const n = Number(argv[i + 1]);
      if (Number.isFinite(n) && n > 0) return n;
    }
  }
  return 0;
}

/** The same argv with its context replaced. Appends `-c` when there was none —
 *  llama.cpp's default is the model's full trained length, which is exactly the
 *  value that just failed. */
export function withCtx(argv: readonly string[], ctx: number): string[] {
  const out = argv.slice();
  for (let i = 0; i < out.length - 1; i++) {
    if (out[i] === "-c" || out[i] === "--ctx-size") {
      out[i + 1] = String(ctx);
      return out;
    }
  }
  out.push("-c", String(ctx));
  return out;
}

/**
 * The context to OPEN with for a model that has been measured before.
 *
 * `known` is the largest context that has actually started on this machine. It
 * is a ceiling, not a target: the tuner still has to fit it in whatever memory
 * is free right now, and a machine with less free VRAM than last time gets less.
 * Absent (0), the tuner's own estimate opens the bidding.
 */
export function openingCtx(known: number, wanted: number): number {
  return known > 0 ? Math.min(known, wanted) : wanted;
}
