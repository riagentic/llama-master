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
import type { Settings } from "./types.ts";

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
 * WHICH memory ran out — because the two have opposite answers.
 *
 * `context` is everything that scales with `-c`: the KV cache, the attention
 * scratch, the compute graph, and CUDA's lazily-allocated pool. Halving the
 * context halves it, which is what the rest of this file does.
 *
 * `weights` is the model itself — a device buffer that failed while the tensors
 * were being loaded. **A smaller context does not move it by one byte.** That
 * distinction was missing and it is the difference between recovering and not:
 *
 *   ggml_backend_cuda_buffer_type_alloc_buffer: allocating 34679.64 MiB on
 *   device 1: cudaMalloc failed: out of memory
 *   alloc_tensor_range: failed to allocate CUDA1 buffer of size 36364237312
 *
 * That run was retried six times at 8,192 → 4,096 → … → 256 tokens, reloading a
 * 145 GB model each time, and every rung failed at the same 34,679.64 MiB. The
 * answer was to hold one more layer's experts on the host, which the ladder had
 * no way to say. It happens for the most ordinary reason there is: the plan was
 * made when the desktop held 2 GB of VRAM and the start happened when it held
 * 5.5 GB — a browser opened in between.
 */
export type FitFault = "weights" | "context";

/** The load-time shapes. A buffer that failed while TENSORS were being placed is
 *  the model, whatever else the same log also says. */
const WEIGHTS_SIGNS =
  /alloc_tensor_range|unable to allocate \w+ buffer|error loading model|failed to load model|create_memory: failed/i;

/**
 * What llama.cpp asked for and did not get, in bytes.
 *
 * From its own message, which prints MiB with two decimals and, on the next
 * line, the exact byte count. The byte count is preferred when it is there.
 * 0 when the log does not say — the caller then has no shortfall to size a step
 * from and falls back to the context rung.
 */
export function requestedB(lines: readonly string[]): number {
  const text = lines.join("\n");
  const exact = text.match(/failed to allocate \w+ buffer of size (\d+)/i);
  if (exact) {
    const n = Number(exact[1]);
    if (Number.isFinite(n) && n > 0) return n;
  }
  const mib = text.match(/allocating ([\d.]+) MiB on device/i);
  if (mib) {
    const n = Number(mib[1]);
    if (Number.isFinite(n) && n > 0) return Math.round(n * 1024 * 1024);
  }
  return 0;
}

/** Which device the failed allocation was for, or -1 when the log does not say. */
export function faultDevice(lines: readonly string[]): number {
  const m = lines.join("\n").match(/allocating [\d.]+ MiB on device (\d+)/i);
  const n = m ? Number(m[1]) : NaN;
  return Number.isInteger(n) ? n : -1;
}

/** The two faults, told apart. `null` when nothing ran out of memory at all. */
export function fitFault(lines: readonly string[]): FitFault | null {
  if (!isFitFailure(lines)) return null;
  return WEIGHTS_SIGNS.test(lines.join("\n")) ? "weights" : "context";
}

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
/** llama.cpp's own `-ub` default, and the floor this ladder will not go under. */
export const MIN_UBATCH = 512;

/** The next micro-batch down, or 0 when there is no step left to take. */
export function retryUbatch(current: number): number {
  const half = Math.floor(current / 2);
  return half >= MIN_UBATCH ? half : 0;
}

/** `-ub` in an argv, or 0 when it does not say one. */
export function ubatchOf(argv: readonly string[]): number {
  for (let i = 0; i < argv.length - 1; i++) {
    if (argv[i] === "-ub" || argv[i] === "--ubatch-size") {
      const n = Number(argv[i + 1]);
      if (Number.isFinite(n) && n > 0) return n;
    }
  }
  return 0;
}

/**
 * The same argv with a smaller micro-batch — and a logical batch that still
 * covers it.
 *
 * `-b` below `-ub` is not a configuration llama.cpp accepts as written: it
 * clamps, and the command would then describe a run that is not happening. The
 * tuner keeps them in step (`-b = max(2048, ub)`) and so does this.
 */
export function withUbatch(argv: readonly string[], ub: number): string[] {
  const out = argv.slice();
  let sawUb = false;
  for (let i = 0; i < out.length - 1; i++) {
    if (out[i] === "-ub" || out[i] === "--ubatch-size") {
      out[i + 1] = String(ub);
      sawUb = true;
    }
    if (out[i] === "-b" || out[i] === "--batch-size") {
      const b = Number(out[i + 1]);
      if (Number.isFinite(b) && b < ub) out[i + 1] = String(ub);
    }
  }
  if (!sawUb) out.push("-ub", String(ub));
  return out;
}

export function retryCtx(current: number): number {
  const half = Math.floor(current / 2 / CTX_STEP) * CTX_STEP;
  return half >= MIN_CTX ? half : 0;
}

/**
 * How many more layers' experts to hold on the host.
 *
 * Sized from the shortfall rather than stepped blindly: the app knows each
 * layer's routed experts EXACTLY (the Rust tensor walk), and llama.cpp says
 * exactly how many bytes it asked for, so this is arithmetic and not a search.
 * One rung should therefore be enough — a geometric ladder would overshoot by
 * up to 2x, and every layer moved to the host costs about 2% of the generation
 * rate, so overshooting is not free.
 *
 * `+1` because the failed allocation must not merely become possible, it must
 * leave room for the scratch the same device still needs.
 */
export function movedLayers(shortB: number, perLayerB: number): number {
  if (!(shortB > 0) || !(perLayerB > 0)) return 0;
  return Math.ceil(shortB / perLayerB) + 1;
}

/** What the app should do about a run that has just ended. */
/**
 * A capability the run asked for and this build cannot provide.
 *
 * Not a memory failure, and the ladder's other two rungs are useless against
 * it — but it has the same shape as one: the run told us something specific
 * that no reading of the header could have, and the answer is one flag away.
 *
 * Found the expensive way on GLM-5.3-Flash. The header declares one NextN
 * block, so the tuner turned multi-token prediction on — correctly, by every
 * fact available to it: the weights ARE in the file, and llama.cpp asserts the
 * other way (`n_layer_nextn > 0`) for a model without them. But the loader for
 * this architecture had not implemented the graph yet:
 *
 *   glm5next.cpp:712: GGML_ASSERT(params.gtype != LLM_GRAPH_TYPE_DECODER_MTP
 *   && "glm5next NextN graph not implemented yet") failed
 *
 * That is a fact about the BUILD, not the model, and it can only be learned by
 * asking. So the ladder asks once, drops the flag, and the run continues
 * without the optimisation instead of aborting with a stack trace.
 *
 * The table is deliberately tiny: only asserts that name a feature a SETTING
 * turns off belong here. An assert with no flag behind it is a crash, and
 * `serverlog.ts` explains it rather than this file retrying it.
 */
export type Unsupported = {
  /** The `Settings` key to clear, so the next start does not ask again. */
  setting: "specType";
  /** The flag to strip from the argv, and how many words it takes with it. */
  flag: string;
  arity: number;
  /** What the user loses, in their terms. */
  feature: string;
};

const UNSUPPORTED: { match: RegExp; found: Unsupported }[] = [
  {
    // llama.cpp's own words, and the quoted half is what makes this safe to
    // act on: "not implemented yet" is a promise about the BUILD, not a
    // complaint about the model or the machine.
    match: /GRAPH_TYPE_DECODER_MTP|NextN graph not implemented/i,
    found: {
      setting: "specType",
      flag: "--spec-type",
      arity: 1,
      feature: "speculative decoding off the model's own MTP block",
    },
  },
];

/**
 * Did this run die because the build cannot do something it was asked to?
 *
 * Requires the abort as well as the signature: llama.cpp prints the assert text
 * on the way down, and matching the text alone would strip a flag from a run
 * that merely MENTIONED it.
 */
export function unsupportedFeature(
  lines: readonly string[],
): Unsupported | null {
  const text = lines.join("\n");
  if (!/GGML_ASSERT|ggml_abort/.test(text)) return null;
  return UNSUPPORTED.find((u) => u.match.test(text))?.found ?? null;
}

/**
 * The tuner's answer, with anything this build has already refused taken back
 * out — and a sentence saying so.
 *
 * The tuner is right to propose it: `meta.nextnLayers > 0` means the weights
 * are in the file, and that is every fact a header can offer. What it cannot
 * know is whether the loader implements the graph, which is why this is applied
 * AFTER tuning rather than folded into it — the rule is not "this model has no
 * MTP block", it is "this build could not use the one it has".
 *
 * The reason is worth printing because the alternative is a silent difference
 * between what the tuner said and what the command shows.
 */
export function vetoUnsupported(
  settings: Settings,
  unsupported: readonly string[],
): { settings: Settings; reasons: string[] } {
  if (unsupported.length === 0) return { settings, reasons: [] };
  const out = { ...settings };
  const reasons: string[] = [];
  if (unsupported.includes("specType") && out.specType) {
    out.specType = "";
    reasons.push(
      "Speculative decoding off — this llama.cpp build aborted on this model's multi-token-prediction block, saying the graph is not implemented yet. The model still runs; only the speed differs. A newer build starts with a clean sheet.",
    );
  }
  return { settings: out, reasons };
}

export type FitDecision =
  | { kind: "none" }
  | { kind: "retry"; ctx: number; attempt: number; note: string }
  | { kind: "ubatch"; ubatch: number; attempt: number; note: string }
  | { kind: "nospec"; attempt: number; note: string }
  | { kind: "offload"; nCpuMoe: number; attempt: number; note: string }
  | {
    kind: "drop";
    unsupported: Unsupported;
    attempt: number;
    note: string;
  };

/**
 * Should the app try again, and with what changed?
 *
 * `tries` is how many automatic retries have already happened for this start —
 * reset whenever the user presses Start themselves, so a manual attempt always
 * gets the full ladder and never inherits an exhausted one.
 *
 * The fault picks the rung. A `weights` fault is answered by moving experts to
 * the host, and ONLY by that: it is the one thing that makes a model buffer
 * smaller. When there is nothing left to move — a dense model, or every layer
 * already on the host — the ladder falls through to the context rung, which at
 * least shrinks the KV cache sharing the same card.
 */
export function fitDecision(args: {
  lines: readonly string[];
  ctx: number;
  tries: number;
  /** Off when the user is driving the settings by hand: an automatic retry
   *  would silently overwrite a context they chose on purpose. */
  auto: boolean;
  /**
   * May the ladder take the MICRO-BATCH back?
   *
   * A separate permission from `auto`, because they are separate choices. `auto`
   * is off whenever the user pinned a context — "Max on Hybrid", or a number
   * they typed — and shrinking that context would be the app overruling them.
   * But `-ub` is not what they pinned: the tuner raised it from 512 to spend
   * leftover VRAM on prefill speed (`tune.ts`), so the tuner may give it back,
   * and giving it back costs prompt-ingestion speed and NOT one token of the
   * length that was asked for. Off when the tuner is off, for the same reason
   * `auto` is: then the micro-batch is the user's own.
   */
  autoUbatch?: boolean;
  /** `-ub` as it ran. */
  ubatch?: number;
  /** `--n-cpu-moe` as it ran. */
  nCpuMoe?: number;
  /** Layers this model has — the cap on the above. */
  nLayer?: number;
  /** One layer's routed experts, exact, from the header. 0 for a dense model. */
  expertPerLayerB?: number;
  /** VRAM the failing device had to give, so the shortfall is a real number
   *  rather than the whole allocation. Card total minus what else is on it. */
  deviceFreeB?: readonly number[];
}): FitDecision {
  if (args.tries >= MAX_FIT_RETRIES) return { kind: "none" };
  if (!args.auto && !args.autoUbatch) return { kind: "none" };

  // BEFORE the memory rungs, because an abort is not an allocation failure and
  // neither of them would move it: the run would shrink its context five times
  // and hit the same assert at every size.
  const cannot = unsupportedFeature(args.lines);
  if (cannot) {
    return {
      kind: "drop",
      unsupported: cannot,
      attempt: args.tries + 1,
      note:
        `This llama.cpp build cannot do ${cannot.feature} for this model — it said so itself, in an assert naming the graph it has not implemented yet. ` +
        `Retrying without ${cannot.flag}. Nothing else about the plan changes, and the answer is the same either way: only the speed differs.`,
    };
  }

  const fault = fitFault(args.lines);
  if (!fault) return { kind: "none" };

  // Speculative decoding off the model's own MTP block builds a SECOND FULL
  // CONTEXT — its own KV cache, its own compute buffer on every device, and (on
  // a recurrent model) three more copies of the recurrent state
  // (`plan.ts:mtpDraft`). When THAT is what ran out, llama.cpp says so by name,
  // and the cheapest possible answer is to stop drafting: it costs generation
  // speed and nothing else — not one token of context, not one layer of
  // residency — and the answer the model gives is identical either way.
  //
  // Deliberately NOT recorded as a capability the build lacks, unlike the
  // `drop` rung above: this build can do it, this machine did not have room for
  // it today. Writing it down would disable drafting for that build and model
  // for ever, including on the day the desktop is not holding 5 GB of VRAM.
  if (
    fault === "context" &&
    /failed to create MTP context|common_speculative_init/i.test(
      args.lines.join("\n"),
    )
  ) {
    return {
      kind: "nospec",
      attempt: args.tries + 1,
      note:
        "The main context loaded and then the draft context for speculative decoding did not fit — it is a second context, with its own cache and its own compute buffer on every card. " +
        "Retrying without it. Nothing else about the plan changes: same context, same placement, same answers, only slower to produce them.",
    };
  }

  // The micro-batch rung, and it comes FIRST among the memory rungs for a
  // context fault — it is the cheapest thing on the table.
  //
  // The compute graph is sized `context x micro-batch` (`plan.ts:attnMaskB`),
  // so halving `-ub` and halving `-c` free exactly the same bytes. They do not
  // cost the same thing: halving `-c` takes away the length the user asked for,
  // halving `-ub` takes away prompt-ingestion speed and nothing else — and the
  // tuner is what raised `-ub` above 512 in the first place, precisely because
  // it thought there was VRAM going spare. There was not, so it takes it back.
  //
  // Only down to 512 (llama.cpp's own default). Below that the placement itself
  // may have chosen 256 to buy layers, and undoing that trade is the ladder
  // working against the plan.
  if (fault === "context" && args.autoUbatch) {
    const now = args.ubatch ?? 0;
    const next = retryUbatch(now);
    if (next > 0) {
      return {
        kind: "ubatch",
        ubatch: next,
        attempt: args.tries + 1,
        note:
          `The compute buffer did not fit — retrying with a micro-batch of ${next.toLocaleString()} instead of ${now.toLocaleString()}. ` +
          `It is sized context x micro-batch, so this frees as much as halving the context would, and it costs prompt-ingestion speed rather than the context you asked for.`,
      };
    }
  }

  // Everything below shrinks something the USER asked for — the residency, then
  // the context — so it needs the full permission, not the micro-batch one.
  if (!args.auto) return { kind: "none" };

  if (fault === "weights") {
    const perLayerB = args.expertPerLayerB ?? 0;
    const nLayer = args.nLayer ?? 0;
    const now = Math.max(0, args.nCpuMoe ?? 0);
    const dev = faultDevice(args.lines);
    const free = args.deviceFreeB?.[dev] ?? 0;
    // The shortfall, not the request: asking for 34.7 GB of a card holding
    // 22 GB of headroom is 12.7 GB short, and moving 34.7 GB of experts to the
    // host would give away most of the GPU for no reason.
    const shortB = Math.max(0, requestedB(args.lines) - free);
    const move = movedLayers(shortB, perLayerB);
    const next = Math.min(nLayer, now + move);
    if (move > 0 && nLayer > 0 && next > now) {
      return {
        kind: "offload",
        nCpuMoe: next,
        attempt: args.tries + 1,
        note:
          `The weights did not fit on GPU ${dev < 0 ? "?" : dev} — short by ${
            (shortB / 1024 ** 3).toFixed(1)
          } GB. Retrying with ${next} layers' experts held in RAM instead of ${now}. ` +
          `A smaller context would not have moved this: it is the model itself, not the cache.`,
      };
    }
  }

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

/** `--n-cpu-moe` in an argv, or 0 when it does not say one. */
export function nCpuMoeOf(argv: readonly string[]): number {
  for (let i = 0; i < argv.length - 1; i++) {
    if (argv[i] === "--n-cpu-moe" || argv[i] === "-ncmoe") {
      const n = Number(argv[i + 1]);
      if (Number.isFinite(n) && n >= 0) return n;
    }
  }
  return 0;
}

/**
 * The same argv holding more layers' experts on the host.
 *
 * The `-ts` that went with it is DROPPED, and that is the whole subtlety: the
 * split was computed for a placement that has just been proven wrong, and it
 * pins the layers to exactly the cards that could not hold them. Removing it
 * lets llama.cpp fall back to splitting by each device's free VRAM, which for
 * the retry is the better information — it is measured at load time, on the
 * machine as it is now, which is precisely what the stale plan was not.
 */
export function withNCpuMoe(argv: readonly string[], n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-ts" || a === "--tensor-split") {
      i++; // and its value
      continue;
    }
    out.push(a as string);
  }
  for (let i = 0; i < out.length - 1; i++) {
    if (out[i] === "--n-cpu-moe" || out[i] === "-ncmoe") {
      out[i + 1] = String(n);
      return out;
    }
  }
  out.push("--n-cpu-moe", String(n));
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
/**
 * The same argv without one flag and the words it takes with it.
 *
 * Editing the command that RAN rather than re-composing one, for the reason
 * every other rung does: re-composing would pick up whatever the user has
 * changed since they pressed Start, and "what you see is what runs" would stop
 * being true across a retry.
 */
export function withoutFlag(
  argv: readonly string[],
  flag: string,
  arity = 1,
): string[] {
  const out: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === flag) {
      i += arity;
      continue;
    }
    out.push(argv[i] as string);
  }
  return out;
}

export function openingCtx(known: number, wanted: number): number {
  return known > 0 ? Math.min(known, wanted) : wanted;
}
