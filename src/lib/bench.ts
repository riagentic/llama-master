// src/lib/bench.ts — measure this machine instead of estimating it.
//
// Everything the app says about speed is arithmetic over one uncertain number:
// effective memory bandwidth. `speed.ts` explains why it cannot be read from
// the machine — nothing reports a card's bus width, and the achieved fraction
// depends on the kernel — so it ships labelled defaults and prefers a real
// observation whenever one exists.
//
// The observation used to come from whatever the user last said in the chat,
// which makes it a measurement of an unknown prompt at an unknown context with
// an unknown amount of thinking in it. That is enough to beat a default and not
// enough to trust. A bench is the same measurement with the variables held
// still: a fixed prompt, a fixed number of tokens, the cache disabled so the
// prefill is real work every time, and temperature 0 so the model cannot wander
// into a different amount of it.
//
// It runs against the SERVER THAT IS ALREADY UP. That is the whole design
// constraint and it is what makes this cheap enough to offer as a button: no
// second process, no reload, no extra memory, no risk to a run that took two
// minutes to place. llama.cpp reports its own `timings` on the response, so the
// numbers are the server's, not ours — we contribute the request and the
// arithmetic.
//
// Pure: a request body out, a parsed result in, verdicts over both.

import type { ModelMeta } from "./types.ts";

/**
 * Which KIND of work a bench measured, and why there are two.
 *
 * Speculative decoding is the app's biggest un-measured lever, and its whole
 * character is that the gain DEPENDS ON THE TEXT. A drafter is only paid for
 * the tokens the full model then accepts, so output that repeats itself — code,
 * lists, edits of earlier text — is where it wins, and flowing prose is where it
 * very nearly does not. One prompt therefore cannot answer "is this worth it";
 * it can only answer it for whichever kind of work the prompt happened to be.
 *
 * Ours was prose, deliberately and for good reasons (it runs to a steady state
 * rather than stopping early). The consequence went unnoticed: every speculative
 * setting in this app was being measured at its WORST case, every time, and the
 * tuner then told the user to go and measure it themselves.
 *
 * So there are two, run against the same loaded server, and the pair is the
 * result — one number is a rate, two numbers are an answer.
 */
export type BenchKind = "prose" | "code";

/** What one bench run measured. Every field is llama.cpp's own number. */
export type BenchResult = {
  /** Which prompt produced this. Part of the result because a code rate and a
   *  prose rate are not comparable to each other by accident — only on purpose. */
  kind: BenchKind;
  /** Prompt tokens processed, and how fast — the wait before the first word. */
  promptTokens: number;
  promptTps: number;
  /** Tokens generated, and how fast — the pace the answer arrives at. */
  genTokens: number;
  genTps: number;
  /** Seconds from the request leaving to the first token, measured HERE: no
   *  server-side metric reports it, and it is the number a person feels. */
  latencyMs: number;
  /** When it ran, so a stale number can say so. */
  at: number;
  /** The context the server was running at, for the record: a rate measured at
   *  4k says little about the same model at 200k, and a bench that does not
   *  carry its conditions invites exactly that comparison. */
  ctx: number;
  /** The model and settings this describes, so a bench cannot be shown beside
   *  a different run. */
  modelPath: string;
};

export const EMPTY_BENCH: BenchResult = {
  kind: "prose",
  promptTokens: 0,
  promptTps: 0,
  genTokens: 0,
  genTps: 0,
  latencyMs: 0,
  at: 0,
  ctx: 0,
  modelPath: "",
};

/**
 * How many tokens to generate.
 *
 * Long enough that the rate is the steady-state rate rather than the first
 * batch's warm-up, short enough that pressing the button is not a commitment:
 * at the 9 tok/s a big MoE with experts in host RAM reaches, 128 tokens is
 * about fourteen seconds. Below ~64 the first-token cost distorts the average;
 * above ~256 nobody waits for it twice.
 */
export const BENCH_TOKENS = 128;

/**
 * The two prompts, and why they are these.
 *
 * Both are long enough to make the prefill measurable (a two-word prompt
 * measures nothing but overhead), fixed so two runs are comparable, and
 * deliberately dull: a prompt that invites reasoning gets a thinking model to
 * spend the whole budget thinking, and then the "generation rate" is measured
 * on a different kind of work than the one being compared. Both ask for
 * open-ended output that keeps flowing for the whole budget, where a question
 * an instruct model can answer in a sentence would stop early and measure the
 * first batch instead of the steady state.
 *
 * What differs is the ONE property speculative decoding is sensitive to:
 * whether the next token is guessable from the tokens before it.
 *
 * - `prose` is the original, and it stays the calibration prompt. Bandwidth is
 *   what `speed.ts` needs and prose is the honest floor for it — no drafter can
 *   flatter a rate that has nothing to predict.
 * - `code` is written to be the opposite: repeated declaration shapes, repeated
 *   punctuation, a doc comment before each function. That is not a trick, it is
 *   what a local model is mostly asked for, and it is the case a drafter was
 *   built to win. Anything a drafter can do, it does here.
 *
 * Neither is "the" speed. The PAIR is the measurement, and the gap between them
 * is the only honest answer to "should I turn speculative decoding on".
 */
export const BENCH_PROMPTS: Record<BenchKind, string> = {
  prose: "You are a text generator used for a speed measurement. " +
    "Do not think, explain, or comment. Write plain prose about the sea, " +
    "continuously, until you are stopped. Begin now: The sea at dawn is",
  code: "You are a code generator used for a speed measurement. " +
    "Do not think, explain, or comment. Emit only TypeScript. " +
    "Write a long series of very small pure functions, one after another, " +
    "each preceded by a one-line JSDoc comment, continuously, until you are " +
    "stopped. Begin now:\n\n/** Adds two numbers. */\n" +
    "export function add(a: number, b: number): number {\n  return a + b;\n}\n",
};

/** The calibration prompt: what `speed.ts` divides bytes by. Prose, because a
 *  drafted token is not a token the memory bus paid for, and a bandwidth
 *  calibrated off accepted drafts would be several times too high. */
export const BENCH_PROMPT = BENCH_PROMPTS.prose;

/**
 * The request body for one bench run.
 *
 * `cache_prompt: false` matters more than it looks: with the prompt cache on,
 * the second run of the same prompt skips the prefill entirely and reports an
 * enormous prompt rate that is really a cache hit. `temperature: 0` for
 * repeatability. No `stop`, because a stop sequence would end the run early
 * and measure fewer tokens than asked for.
 */
export function benchRequest(
  kind: BenchKind = "prose",
  tokens = BENCH_TOKENS,
): Record<string, unknown> {
  return {
    prompt: BENCH_PROMPTS[kind] ?? BENCH_PROMPTS.prose,
    n_predict: Math.max(16, Math.round(tokens)),
    temperature: 0,
    cache_prompt: false,
    // A reasoning model asked for 128 tokens can spend all of them thinking
    // and emit nothing, which measures the right rate on the wrong work — and
    // on a build that supports it, this is the one place we WANT thinking off.
    // Unknown fields are ignored by older servers, so it costs nothing there.
    //
    // It has one side effect worth knowing, because it is invisible: a
    // reasoning budget builds llama.cpp's `rbudget` sampler, and that sampler
    // turns BACKEND SAMPLING off for the request with a warning in the server
    // log (`common/sampling.cpp`). So on a thinking model this bench measures
    // the CPU sampling path even when `-bs` is set. It is the right trade —
    // measuring generation while the model thinks would be far more wrong —
    // but it is why `-bs` is never judged from a bench alone.
    reasoning_budget: 0,
  };
}

/** llama.cpp's `timings` block, as much of it as we rely on. */
type Timings = {
  prompt_n?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_per_second?: number;
};

/**
 * Read one `/completion` response into a result, or `null` when it does not
 * carry timings.
 *
 * Null rather than zeros: a build too old to report `timings` has not measured
 * a slow machine, it has measured nothing, and a 0 tok/s on screen would be a
 * lie about the hardware. The caller says so instead.
 */
export function parseBench(
  json: unknown,
  ctx: {
    kind: BenchKind;
    latencyMs: number;
    ctx: number;
    modelPath: string;
    at: number;
  },
): BenchResult | null {
  const t = (json as { timings?: Timings } | null)?.timings;
  if (!t) return null;
  const n = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0;
  const genTps = n(t.predicted_per_second);
  const genTokens = n(t.predicted_n);
  // A run that generated nothing measured nothing. Reporting 0 tok/s would
  // read as "this machine is broken" when the truth is "the model stopped".
  if (genTokens <= 0 || genTps <= 0) return null;
  return {
    kind: ctx.kind,
    promptTokens: n(t.prompt_n),
    promptTps: n(t.prompt_per_second),
    genTokens,
    genTps,
    latencyMs: Math.max(0, Math.round(ctx.latencyMs)),
    at: ctx.at,
    ctx: Math.max(0, Math.round(ctx.ctx)),
    modelPath: ctx.modelPath,
  };
}

/** Does this bench describe the run that is on screen? */
export function benchApplies(
  b: BenchResult,
  modelPath: string,
  ctx: number,
): boolean {
  if (b.at === 0 || !b.modelPath) return false;
  if (b.modelPath !== modelPath) return false;
  // The context is part of the conditions: the KV cache is read every token,
  // so the same model at 8k and at 256k are different measurements. Equal is
  // the only safe test — a bench taken at another length is shown with its
  // length rather than silently reused.
  return b.ctx === ctx;
}

// ── What the bandwidth number means ────────────────────────────────────────
//
// A calibrated `ramBps` is the one machine number a user can actually act on,
// because the commonest reason it is low is a BIOS setting rather than a
// hardware limit. DDR5 sold as 6000 MT/s boots at 4800 (its JEDEC default)
// unless EXPO or XMP is switched on, and every model with experts in host RAM
// then runs about a fifth slower for no reason at all.
//
// What this must NOT do is diagnose. The app cannot read the installed DIMMs
// without root (`dmidecode` and `lshw` both need it), so it does not know what
// this RAM is rated for, and llama.cpp never reaches theoretical peak anyway.
// So the bands below are wide, they are stated as "what machines like this
// usually reach", and the advice names something the user can go and look at
// rather than asserting what they will find.

/** Rough effective read bandwidth, in bytes/second, that a modern desktop
 *  reaches through llama.cpp's CPU kernels. Dual-channel DDR5 at JEDEC 4800
 *  lands near the low end, the same sticks at their rated 6000+ near the high
 *  end, and a workstation with four or eight channels well above it. */
export const RAM_BPS_LOW = 30 * 1024 ** 3;
export const RAM_BPS_TYPICAL = 45 * 1024 ** 3;

export type BandwidthNote = {
  tone: "info" | "caution";
  message: string;
};

/**
 * What to say about a measured host-RAM bandwidth, or `null` for nothing.
 *
 * Silent unless it matters: a model living entirely in VRAM does not read host
 * RAM per token, so the figure is true and irrelevant, and a line that appears
 * anyway is a line that trains people to skip this panel.
 */
export function bandwidthNote(
  ramBps: number,
  readsHostRam: boolean,
): BandwidthNote | null {
  if (!readsHostRam) return null;
  if (!Number.isFinite(ramBps) || ramBps <= 0) return null;
  const gb = ramBps / 1024 ** 3;
  if (ramBps < RAM_BPS_LOW) {
    return {
      tone: "caution",
      message:
        `This machine reads model weights from system RAM at about ${
          gb.toFixed(0)
        } GB/s, which is at the low end of what a modern desktop reaches. ` +
        `Most of this model's speed is that number, so it is worth checking: ` +
        `memory sold as DDR5-6000 runs at 4800 until EXPO (AMD) or XMP (Intel) ` +
        `is switched on in the BIOS, which is about a fifth of the speed of ` +
        `every model that keeps its experts in RAM. Populating all the memory ` +
        `channels the board has is the other half of it.`,
    };
  }
  if (ramBps < RAM_BPS_TYPICAL) {
    return {
      tone: "info",
      message:
        `System RAM is being read at about ${
          gb.toFixed(0)
        } GB/s, which is ordinary for dual-channel memory. ` +
        `This model's speed is mostly that number — if the BIOS memory profile ` +
        `(EXPO or XMP) is off, switching it on is the cheapest speed available.`,
    };
  }
  return {
    tone: "info",
    message: `System RAM is being read at about ${
      gb.toFixed(0)
    } GB/s, which is healthy — the memory is not what is holding this back.`,
  };
}

/**
 * A sentence about prefill, or "" when there is nothing worth saying.
 *
 * Prompt processing is the wait before the first word, and unlike generation
 * it is compute-bound, so it responds to `-ub` rather than to bandwidth. Worth
 * naming when it is slow enough that a long prompt becomes a coffee break.
 */
export function prefillNote(b: BenchResult, ubatch: number): string {
  if (b.promptTps <= 0 || b.promptTokens <= 0) return "";
  const secondsFor = (tokens: number) => tokens / b.promptTps;
  const long = 32_768;
  const mins = secondsFor(long) / 60;
  const at = `${Math.round(b.promptTps).toLocaleString()} tokens/s`;
  if (mins >= 1) {
    return `Prompt processing runs at ${at}, so a ${long.toLocaleString()}-token prompt takes about ${
      mins.toFixed(1)
    } minutes before the first word. ${
      ubatch < 2048
        ? "A larger micro-batch is the lever here, and it needs spare VRAM."
        : "The micro-batch is already large; this is the machine's compute limit."
    }`;
  }
  return `Prompt processing runs at ${at} — a ${long.toLocaleString()}-token prompt is about ${
    Math.round(secondsFor(long))
  } seconds before the first word.`;
}

/** Is a bench worth trusting as a calibration? */
export function benchIsSound(b: BenchResult, meta: ModelMeta | null): boolean {
  if (b.genTokens < 16 || b.genTps <= 0) return false;
  // A model whose header we could not read cannot be turned into bytes, so the
  // rate is a fact with nothing to divide it by.
  if (!meta || meta.tensorBytes <= 0) return false;
  return true;
}

// ── The pair, and what it settles ──────────────────────────────────────────
//
// Speculative decoding is the one setting this app names and refuses to switch
// on (`tune.ts`): it is lossless, it is free of memory cost, and it is
// nonetheless a bad idea about half the time, because a drafted token that the
// full model rejects is work thrown away. Which half you are in depends on what
// you ask the model to write, and nothing in a GGUF header knows that.
//
// So the tuner said "measure it" and pointed at the Speed panel — which then
// measured PROSE, the case where a drafter has least to offer, and reported one
// number with no baseline to read it against. The advice was right and the
// instrument could not carry it out.
//
// Two runs against the same loaded server can. Generation is bandwidth-bound,
// and a byte of weights costs the same to read whatever the text is about, so
// with drafting OFF the two rates land within a few percent of each other. That
// makes the pair self-calibrating: the gap that appears when drafting is ON is
// the drafter's acceptance rate, in tokens per second, on this machine.

/** How far apart two rates must be before the difference is a finding rather
 *  than noise. Two runs of the same prompt on an idle machine land within a
 *  couple of percent; 15% is comfortably outside that and well inside the
 *  1.5–3x a working drafter produces on code. */
const SPEC_MARGIN = 0.15;

export type SpecVerdict = {
  tone: "ok" | "info" | "caution";
  /** What the two runs measured, in one sentence. */
  headline: string;
  /** What to do about it — "" when the answer is "nothing". */
  advice: string;
  /** code ÷ prose. 1 means the text made no difference. */
  ratio: number;
};

/**
 * Read a prose bench and a code bench together.
 *
 * `null` until both exist: half a comparison is not a weaker finding, it is a
 * different and much worse one — a single code rate invites exactly the
 * "2.9x faster!" reading that has no baseline under it.
 */
export function specVerdict(
  prose: BenchResult | null,
  code: BenchResult | null,
  specType: string,
): SpecVerdict | null {
  if (!prose || !code) return null;
  if (prose.genTps <= 0 || code.genTps <= 0) return null;
  const ratio = code.genTps / prose.genTps;
  const on = specType !== "";
  const pct = Math.round(Math.abs(ratio - 1) * 100);
  const rates = `${fmt(code.genTps)} tok/s on code against ${
    fmt(prose.genTps)
  } on prose`;

  if (!on) {
    // Drafting off: the two SHOULD agree, because the model reads the same
    // weights either way. Agreement is therefore not a null result — it is the
    // baseline that makes the next measurement mean something.
    if (ratio < 1 + SPEC_MARGIN && ratio > 1 - SPEC_MARGIN) {
      return {
        tone: "info",
        ratio,
        headline:
          `Speculative decoding is off, and code and prose run at the same rate — ${rates}. That is the expected result and it is the baseline.`,
        advice:
          "Switch Speculative decoding to `n-gram map (k4v)` under Performance, restart, and measure again. The code number is the one that should move; if it does, the setting is worth keeping for the work you do.",
      };
    }
    return {
      tone: "info",
      ratio,
      headline: `Speculative decoding is off, and code still runs ${pct}% ${
        ratio > 1 ? "faster" : "slower"
      } than prose — ${rates}. With no drafter involved that is the tokenizer, not the machine: the two texts pack a different number of characters into a token.`,
      advice:
        "Switch Speculative decoding to `n-gram map (k4v)` under Performance, restart, and measure again — the gap that opens beyond this one is the drafter's.",
    };
  }

  if (ratio >= 1 + SPEC_MARGIN) {
    return {
      tone: "ok",
      ratio,
      headline: `Speculative decoding is paying: ${
        ratio.toFixed(2)
      }x on code — ${rates}.`,
      advice:
        "Drafting is lossless, so this is free speed on code, edits and structured output. Prose is close to the undrafted rate, which is the honest shape of this setting rather than a fault in it.",
    };
  }
  if (ratio > 1 - SPEC_MARGIN) {
    return {
      tone: "info",
      ratio,
      headline:
        `Speculative decoding is on and is not making much difference here — ${rates}.`,
      advice: specType === "draft-mtp"
        ? "The model's own MTP block costs nothing to keep on: its weights are in the file and loaded either way, and a rejected draft is discarded rather than wrong. Leave it."
        : "The n-gram kinds are worth trying against each other — `n-gram map (k4v)` and `n-gram (cache)` accept at quite different rates on different work. Or switch it off: what it is not doing, it is not doing for free.",
    };
  }
  return {
    tone: "caution",
    ratio,
    headline:
      `Speculative decoding is on and code is ${pct}% SLOWER than prose — ${rates}.`,
    advice:
      "That is drafts being rejected: every one is a batch of work thrown away, and here they are costing more than the accepted ones save. Switch Speculative decoding off under Performance, or try a different n-gram kind, and measure again.",
  };
}

/** One decimal below 100, none above — a rate of 1043.2 tok/s is false
 *  precision, and 9.6 rounded to 10 loses a real difference. */
function fmt(v: number): string {
  return v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(1);
}
