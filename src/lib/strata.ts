// src/lib/strata.ts — Strata, a second engine behind the same Start button.
//
// Strata (github.com/Niko1221/Strata, MIT) runs the Qwen3.8-Flash-Next family
// (ISTA-DASLab's GSQ-RCO sizes, Swift 1.5, the Coder, Unsloth's UD/IQ sizes,
// and Gyro-S / Gyro-M) and Qwen3.6-35B-A3B: a per-expert VRAM cache, drafting
// and its own kernels,
// where llama.cpp streams every routed expert from RAM. Measured 2026-10-08 on
// two 24 GB Blackwell cards, the same UD-Q4_K_XL file, the app's own bench
// prompts, both warm, a 4 GB reserve on each card: 53-64 tok/s with its MTP
// draft layer (41-54 without) against llama.cpp's 11-15, and a 28,863-token
// prompt read at 2,690 tok/s against 159. Load 105 s warm.
//
// It is NOT ported into llama.cpp and it does not replace it. It is a build
// like any other (`fork:Niko1221/Strata`, the ref grammar `srcref.ts` already
// has), and this file is everything the app decides about it. Pure: names and
// numbers in, argv and a diagnosis out.
//
// Three things Strata's own installer would do that this app must not allow,
// and where each is closed:
//   - `sudo apt-get install` for Python or CUDA — the launcher puts a refusing
//     `sudo` first on PATH and points `STRATA_NVCC` at the app's toolkit
//     (`src/cell/strata.server.ts`);
//   - downloading model WEIGHTS — `--gguf-dir` hands it the files already on
//     disk; the only model fetch left is the Flash-Next packs' ~6.5 GB MTP
//     draft layer, which the offer below states before the button is pressed;
//   - filling the display card to the last byte — `reserveMib` carries the
//     user's reserve (`src/lib/reserve.ts`).

import type { Diagnosis } from "./diagnose.ts";
import type { Settings } from "./types.ts";
import type { SetupRow } from "./setup.ts";
import { formatRef, parseRef } from "./srcref.ts";
import { bytes } from "./format.ts";

const REPO = "Niko1221/Strata";
export const STRATA_REF = formatRef({ kind: "fork", repo: REPO, ref: null });
export const STRATA_DOCS = `https://github.com/${REPO}`;

/** The file the app writes into a Strata build and spawns — the build's
 *  `serverBin`. It lives under the builds root, so the sandbox rule in
 *  `srv.server.ts` holds for it exactly as for `llama-server`. */
export const STRATA_LAUNCHER = "llama-master-strata";

export function isStrataRef(ref: string): boolean {
  const r = parseRef(ref);
  return r.kind === "fork" && r.repo.toLowerCase() === REPO.toLowerCase();
}

/** What Strata's `setup.py` calls a file it can run. */
export type StrataModel = {
  family: string;
  model: string;
  tag: string;
  /** The family in words, when the file name does not say it. */
  title?: string;
  /** Present when Strata's path for this family runs on one card. */
  cards?: "one";
  /** Share of the FILE kept in host RAM; absent = the measured 0.75. */
  ramShare?: number;
  /** Total weights memory (VRAM + RAM) for a family whose file says so
   *  (Gyro's `mem_total`); absent = the `ramShare` estimate. */
  memTotalGB?: number;
  /** The RAM below which Strata's low-RAM budget cannot run this file: the
   *  rest is read from the SSD. Absent = the whole `ramShare` must be free. */
  ramFloorGB?: number;
  /** Trained context when this family's path has no rope scaling. */
  maxContext?: number;
  /** Strata's own pack, which setup DOWNLOADS (no `--gguf-dir`): the file on
   *  disk only says which model it is. The size is stated before the button. */
  packGB?: number;
  /** The context steps this family's setup offers, when they are its own. */
  contexts?: readonly number[];
  /** Host RAM the engine takes whatever the file (Basalt's Inkling: the
   *  pinned token embedding and host buffers, measured 1.0 GiB). Absent =
   *  `ramShare` / `memTotalGB`. */
  ramGB?: number;
  /** How long a start takes, in words. Absent = Strata's 2–3 minutes. */
  load?: string;
};

/** A rule from setup.py's own `FAMILIES`/`MODELS` tables (checked against
 *  Strata 0.2.10, 2026-11-02; Inkling against 0.2.21, 2026-11-24). Matched on
 *  the FIRST shard's name, anchored at both ends: Strata refuses every other
 *  GGUF of the model (K-quants, other Unsloth sizes), and a loose match would
 *  offer an engine that then says no. */
type FileRule = {
  family: string;
  re: RegExp;
  prefix: string;
  title?: string;
  cards?: "one";
  ramShare?: number;
  memTotalGB?: number;
  ramFloorGB?: number;
  maxContext?: number;
  packGB?: number;
  /** Fixed model id, when the file name does not carry it (a pack). */
  model?: string;
  contexts?: readonly number[];
};

/** Inkling-2-Mini's whole footprint per build (setup's `inkling_fit.BUILDS`,
 *  Strata 0.2.21): VRAM and RAM together are the file, nothing more. */
const INKLING_GB: Readonly<Record<string, number>> = {
  "UD-Q8_K_XL": 22.87,
  "UD-Q4_K_XL": 17.63,
  "UD-Q4_K_M": 16.61,
  "UD-IQ4_XS": 15.73,
  "UD-Q3_K_XL": 13.63,
  "UD-IQ3_XXS": 11.66,
  "UD-Q2_K_XL": 9.14,
  "UD-IQ2_XXS": 8.49,
  "UD-IQ1_M": 8.02,
  "UD-IQ1_S": 7.76,
  "SQ-20": 19.35,
};

/** Inkling's own engine (`strata-inkling`) on one card: its DSpark drafter
 *  does not run yet when the layers are split across NVIDIA cards (#6741). */
const INKLING = {
  family: "inkling-2-mini",
  prefix: "inkling-2-mini-",
  title: "Inkling 2 Mini",
  cards: "one",
  maxContext: 262_144,
  // setup's `inkling_fit.CONTEXTS`
  contexts: [8192, 32768, 65536, 131072, 262144],
} as const;

const FILES: readonly FileRule[] = [
  {
    family: "qwen",
    re:
      /^Qwen3\.8-Flash-Next-GSQ-RCO-(Q2_0|IQ2_XS|IQ3_XXS|IQ3_S)-00001-of-00002\.gguf$/,
    prefix: "",
  },
  {
    family: "coder",
    re: /^Qwen3\.8-Flash-Next-GSQ-RCO-(IQ1_M)-00001-of-00002\.gguf$/,
    prefix: "coder-",
  },
  {
    family: "swift",
    re:
      /^Swift-Qwen3\.8-Flash-Next-GSQ-RCO-(IQ2_XS|IQ3_S|IQ3_XXS)-00001-of-00002\.gguf$/,
    prefix: "swift-",
  },
  {
    family: "unsloth",
    re: /^Qwen3\.8-Flash-Next-(UD-Q4_K_XL)-00001-of-00004\.gguf$/,
    prefix: "unsloth-",
    // setup.py's UNSLOTH_MIN_RAM_GB / MODELS.ram_gb: below this the low-RAM
    // budget cannot run the file, although 0.75 of it is what it uses when
    // the RAM is there.
    ramFloorGB: 48,
  },
  {
    family: "unsloth",
    re: /^Qwen3\.8-Flash-Next-(UD-IQ4_XS)-00001-of-00003\.gguf$/,
    prefix: "unsloth-",
    ramFloorGB: 48,
  },
  {
    family: "unsloth",
    re: /^Qwen3\.8-Flash-Next-(UD-Q5_K_XL)-00001-of-00006\.gguf$/,
    prefix: "unsloth-",
    ramFloorGB: 108,
  },
  {
    family: "unsloth",
    re: /^Qwen3\.8-Flash-Next-(UD-Q6_K_XL)-00001-of-00006\.gguf$/,
    prefix: "unsloth-",
    ramFloorGB: 118,
  },
  {
    family: "unsloth",
    re: /^Qwen3\.8-Flash-Next-(Q8_0)-00001-of-00006\.gguf$/,
    prefix: "unsloth-",
    ramFloorGB: 136,
  },
  {
    // Gyro: one file, the n-gram table inside it on the SSD. MODELS' ram_gb is
    // what the WEIGHTS take (VRAM and RAM together), not a RAM share.
    family: "gyro",
    re: /^Qwen3\.8-Flash-Next-(Gyro-S)-TQ1_0\.gguf$/,
    prefix: "gyro-",
    title: "Qwen3.8-Flash-Next (Gyro)",
    memTotalGB: 27.6,
  },
  {
    family: "gyro",
    re: /^Qwen3\.8-Flash-Next-(Gyro-M)-TQ2_0\.gguf$/,
    prefix: "gyro-",
    title: "Qwen3.8-Flash-Next (Gyro)",
    memTotalGB: 35.0,
  },
  {
    // Qwen3.6-35B-A3B: its own engine path (one card, no pack, its MTP block
    // is in the GGUF). The hybrid mode copies nearly the whole file into RAM.
    family: "qwen36",
    re: /^Qwen3\.6-35B-A3B-(UD-Q4_K_XL|UD-Q6_K_XL|Q8_0)\.gguf$/,
    prefix: "qwen36-",
    title: "Qwen3.6-35B-A3B",
    cards: "one",
    ramShare: 0.9,
    maxContext: 262_144,
  },
  // Unsloth's files, pinned by size and SHA-256 in setup: read in place.
  {
    ...INKLING,
    re: new RegExp(
      `^Inkling-2-Mini-(${
        Object.keys(INKLING_GB).filter((k) => k.startsWith("UD-")).join("|")
      })\\.gguf$`,
    ),
  },
  // Strata's own SQ-20 pack, listed from where its setup downloaded it: the
  // model itself rather than a file standing for it.
  {
    ...INKLING,
    re: /^Inkling-2-Mini-(SQ-20)\.strata$/,
    packGB: INKLING_GB["SQ-20"],
  },
  // Any other Inkling GGUF (LM Studio's, a self-made quant): setup refuses the
  // file itself, so Strata runs its own SQ-20 pack — made for a 24 GB card —
  // and downloads it once into the data folder.
  {
    ...INKLING,
    // A quantisation label only (`Q4_K_M`, `IQ4_XS`, `BF16`): a fine-tune
    // (`-Uncensored-…`) is other weights, and running SQ-20 in its place
    // would be a silent substitution.
    re: /^Inkling-2-Mini-(?!UD-)(?:I?Q\d\w*|BF16|F16|F32)\.gguf$/i,
    model: "SQ-20",
    packGB: INKLING_GB["SQ-20"],
  },
];

/** The Strata model a file IS, or null when Strata cannot run it. */
export function strataModel(file: string): StrataModel | null {
  const name = file.split(/[\\/]/).pop() ?? "";
  for (const rule of FILES) {
    const m = rule.re.exec(name);
    if (m) {
      const id = rule.model ?? m[1]!;
      const model: StrataModel = {
        family: rule.family,
        model: id,
        tag: (rule.prefix + id).toLowerCase(),
      };
      if (rule.title) model.title = rule.title;
      if (rule.cards) model.cards = rule.cards;
      if (rule.ramShare !== undefined) model.ramShare = rule.ramShare;
      if (rule.memTotalGB !== undefined) model.memTotalGB = rule.memTotalGB;
      if (rule.ramFloorGB !== undefined) model.ramFloorGB = rule.ramFloorGB;
      if (rule.maxContext !== undefined) model.maxContext = rule.maxContext;
      if (rule.packGB !== undefined) model.packGB = rule.packGB;
      if (rule.contexts) model.contexts = rule.contexts;
      if (rule.family === INKLING.family) model.memTotalGB = INKLING_GB[id];
      return model;
    }
  }
  return null;
}

/** The contexts Strata's setup offers. Every GB of KV cache is a GB less of
 *  cached experts, so the request is rounded DOWN to a step it was measured
 *  at, never up past what was asked for. 384K and 512K are past the trained
 *  262,144 positions: setup adds rope scaling for them on the Flash-Next
 *  path; Qwen3.6's own path has no rope scaling yet (`maxContext`). */
export const STRATA_CONTEXTS: readonly number[] = [
  8192,
  32768,
  65536,
  131072,
  204800,
  262144,
  393216,
  524288,
];

/** The steps this model's Strata path can actually take. */
export function strataContexts(
  model?: StrataModel | null,
): readonly number[] {
  if (model?.contexts) return model.contexts;
  const max = model?.maxContext;
  return max === undefined
    ? STRATA_CONTEXTS
    : STRATA_CONTEXTS.filter((c) => c <= max);
}

export function strataContext(ctx: number, model?: StrataModel | null): number {
  const list = strataContexts(model);
  const fit = list.filter((c) => c <= ctx);
  return fit.length ? fit[fit.length - 1]! : list[0]!;
}

/** The card a one-card path should run on: the most free VRAM now, the
 *  lower number on a tie — so a card another program fills is passed over. */
export function strataCard(freeB: readonly number[]): number {
  let best = 0;
  freeB.forEach((b, i) => {
    if (b > (freeB[best] ?? -Infinity)) best = i;
  });
  return best;
}

/** Strata's own floor for `--vram-reserve-mib`. */
const RESERVE_FLOOR_MIB = 700;

/** One number for every card — Strata has no per-card reserve — so it is the
 *  DISPLAY card's figure: over-reserving a headless card costs a little
 *  speed, under-reserving the one drawing the desktop is a driver reset. */
export function strataReserveMib(perGpuB: number, connectedB: number): number {
  const mib = Math.round((perGpuB + connectedB) / 2 ** 20);
  return Math.max(RESERVE_FLOOR_MIB, Number.isFinite(mib) ? mib : 0);
}

export type StrataRun = {
  /** Path of the model's FIRST shard. */
  modelPath: string;
  model: StrataModel;
  ctx: number;
  reserveMib: number;
  host: string;
  port: number;
  /** The card a one-card path runs on, as nvidia-smi numbers it. */
  gpu?: number;
};

/**
 * The launcher's arguments: the config's tag, then exactly what `setup.py`
 * is given. The launcher runs setup (a no-op when nothing changed) and then
 * execs Strata's server with the config setup wrote, so this one argv is both
 * the settings and the command — what is shown is what runs.
 */
export function strataArgs(r: StrataRun): string[] {
  const dir = r.modelPath.replace(/[\\/][^\\/]*$/, "");
  return [
    r.model.tag,
    "--family",
    r.model.family,
    "--model",
    r.model.model,
    ...(r.model.packGB ? [] : ["--gguf-dir", dir]),
    // setup.py: `--gpus` takes two or more cards; one card is `--gpu N`.
    ...(r.model.cards === "one"
      ? ["--gpu", String(r.gpu ?? 0)]
      : ["--gpus", "all"]),
    "--context",
    String(strataContext(r.ctx, r.model)),
    "--vram-reserve-mib",
    String(r.reserveMib),
    // The app's chat sends text only: leave the image encoder out, so setup
    // does not fetch its ~0.9 GB file or keep ~1.4 GB of VRAM from the experts.
    "--vision",
    "no",
    "--host",
    r.host,
    "--port",
    String(r.port),
  ];
}

/** The settings a Strata or Basalt run actually has: the context its ARGV
 *  carries — rounded to a step, and for Basalt capped to the card. Everything
 *  that describes a live run reads `runSettings` (the bench keys on it), and
 *  a figure the engine never used would describe some other server. */
export function strataSettings(
  settings: Settings,
  argv: readonly string[],
): Settings {
  return {
    ...settings,
    ctxSize: strataRunOf(argv)?.ctx ?? (Number(settings.ctxSize) || 0),
  };
}

type Installed = { id: string; ref: string };

/**
 * Why the ACTIVE build cannot run this model, when the reason is Strata —
 * a Strata build asked for a file it does not know. Blocks Start.
 */
export function strataMismatch(
  file: string,
  buildRef: string,
  installed: readonly Installed[],
): Diagnosis | null {
  if (!isStrataRef(buildRef) || strataModel(file)) return null;
  const other = installed.find((b) => !isStrataRef(b.ref));
  return {
    reason:
      "The active build is Strata, and Strata runs only the files its setup has kernels for: Qwen3.8-Flash-Next (ISTA-DASLab's GSQ-RCO sizes, Swift 1.5, the Coder, Unsloth's UD-IQ4_XS / UD-Q4 / Q5 / Q6_K_XL / Q8_0, and Gyro-S / Gyro-M), Qwen3.6-35B-A3B and Inkling 2 Mini. This model needs llama.cpp.",
    steps: [
      other
        ? {
          text: `Switch to your llama.cpp build (${other.id}).`,
          action: { kind: "use-build", id: other.id },
        }
        : {
          text: "Install a llama.cpp build.",
          action: { kind: "open-tab", tab: "build" },
        },
    ],
  };
}

/**
 * "This model has a faster engine" — an OFFER, never a block: llama.cpp runs
 * the file correctly, only slower. Null when Strata is already active, the
 * file is not one of its, or the machine has no NVIDIA card for it.
 */
export function strataOffer(
  file: string,
  buildRef: string,
  installed: readonly Installed[],
  hasNvidia: boolean,
): Diagnosis | null {
  const model = strataModel(file);
  if (!hasNvidia || isStrataRef(buildRef) || !model) return null;
  const ready = installed.find((b) => isStrataRef(b.ref));
  return {
    reason:
      "A faster engine exists for this model. Strata keeps the most-used experts on the GPU and drafts ahead; on Qwen3.8-Flash-Next it measured about four times llama.cpp's generation rate and read a long prompt about seventeen times faster (two 24 GB cards, 2026-10-08). Qwen3.6-35B-A3B is newer and runs on the same engine. The price: a load of two to three minutes, one request at a time by default, and a context rounded down to a size it was tuned for.",
    steps: [
      ready
        ? {
          text: `Switch to the Strata build you already have (${ready.id}).`,
          action: { kind: "use-build", id: ready.id },
        }
        : {
          text: model.packGB
            ? `Build Strata for this model — its engine and setup run here with no root. Strata does not read this file: it downloads its own ${model.model} pack (${model.packGB} GB, made for a 24 GB card) and the model's own engine.`
            : model.family === "inkling-2-mini"
            ? "Build Strata for this model — its engine and setup run here with no root, using the model file you already have. It downloads the model's own engine and its 0.4 GB DSpark drafter."
            : model.family === "qwen36"
            ? "Build Strata for this model — its engine and setup run here with no root, using the model file you already have. It needs no separate draft layer: the file carries its own."
            : "Build Strata for this model — its engine and setup run here with no root, using the model files you already have. It downloads its engine and the model's ~6.5 GB draft layer.",
          action: { kind: "use-ref", ref: STRATA_REF },
        },
      {
        text: "What Strata is and how it works.",
        action: { kind: "open-url", url: STRATA_DOCS },
      },
    ],
  };
}

/** What a Strata argv (`strataArgs`) says about its run — read back from the
 *  command that is on screen or running, so nothing describes Strata from
 *  settings it never saw. Null for any other argv. */
export function strataRunOf(
  argv: readonly string[],
): {
  ctx: number;
  reserveMib: number;
  /** `--gpu`'s cards, main first; empty when the argv names none. */
  gpus: number[];
  /** Basalt's `--kv`, or null (Strata has none). */
  kv: string | null;
} | null {
  const after = (flag: string): number => {
    const i = argv.indexOf(flag);
    return i < 0 ? NaN : Number(argv[i + 1]);
  };
  // Strata's setup takes `--context`; Basalt's server `--max-context`.
  const c = after("--context");
  const ctx = Number.isFinite(c) ? c : after("--max-context");
  const reserveMib = after("--vram-reserve-mib");
  if (!Number.isFinite(ctx) || !Number.isFinite(reserveMib)) return null;
  const i = argv.indexOf("--gpu");
  const gpus = i < 0 ? [] : String(argv[i + 1] ?? "").split(",").map(Number);
  const k = argv.indexOf("--kv");
  return {
    ctx,
    reserveMib,
    gpus: gpus.every((g) => Number.isInteger(g) && g >= 0) ? gpus : [],
    kv: k < 0 ? null : argv[k + 1] ?? null,
  };
}

/** Share of the model FILE Strata holds in host RAM. Measured 2026-10-08 on
 *  UD-Q4_K_XL: 111.3 GB on disk, 82.7 GB resident in the engine process, 79.6
 *  of it ANONYMOUS — copied and partly pinned, not a mapping the kernel can
 *  drop. One file on one machine, so every use of it says "about". */
const RAM_SHARE = 0.75;

/**
 * What a Strata run takes — an ESTIMATE, and a different kind of answer from
 * `plan.ts`: Strata places nothing by layers. It fills every card up to the
 * reserve with the most-used experts and keeps the rest in RAM, so VRAM is
 * "whatever is free, less the reserve" (measured: 35.6 GB taken against the
 * 34.7 this predicts) and RAM is a share of the file.
 */
export function strataNeeds(a: {
  fileB: number;
  freeVramB: number;
  cards: number;
  reserveMib: number;
  /** `StrataModel.ramShare`: the measured 0.75 for the Flash-Next packs, 0.9
   *  for Qwen3.6's hybrid path (nearly the whole file is copied). */
  ramShare?: number;
  /** `StrataModel.memTotalGB` (Gyro: VRAM and RAM together): the part the
   *  cards do not take is the part RAM must hold. */
  memTotalB?: number;
  /** `StrataModel.ramGB`: a fixed host share, whatever the cards take. */
  fixedRamB?: number;
}): { vramB: number; ramB: number } {
  const n = (x: number | undefined) =>
    x !== undefined && Number.isFinite(x) && x > 0 ? x : 0;
  const share = Number.isFinite(a.ramShare) && a.ramShare! > 0
    ? a.ramShare!
    : RAM_SHARE;
  const vramB = Math.max(
    0,
    n(a.freeVramB) - n(a.cards) * n(a.reserveMib) * 2 ** 20,
  );
  return {
    vramB,
    ramB: n(a.fixedRamB) > 0
      ? n(a.fixedRamB)
      : n(a.memTotalB) > 0
      ? Math.max(0, n(a.memTotalB) - vramB)
      : Math.round(n(a.fileB) * share),
  };
}

/**
 * Why Start would end in the OOM killer, or "". Strata's experts are
 * anonymous memory: with less free than it copies, the kernel does not slow
 * down, it kills something — and not necessarily Strata. `ownB` is the
 * anonymous memory a RUNNING engine gives back when a restart stops it;
 * without it a running Strata blocks its own restart.
 */
export function strataRamShort(
  needB: number,
  availableB: number,
  ownB = 0,
  floorB = 0,
  name = "Strata",
): string {
  const floor = Number.isFinite(floorB) && floorB > 0 ? floorB : 0;
  const need = floor > 0 ? Math.min(needB, floor) : needB;
  const freeB = availableB + (Number.isFinite(ownB) && ownB > 0 ? ownB : 0);
  if (!(need > 0) || !(availableB > 0) || freeB >= need) return "";
  const words = floor > 0 && needB > floor
    ? `${name} needs at least ${
      bytes(need)
    } of this model in RAM (the rest is read from the SSD), and ${
      bytes(freeB)
    } is free.`
    : `${name} keeps about ${bytes(need)} of this model in RAM, and ${
      bytes(freeB)
    } is free.`;
  return `${words} Starting now would run the machine out of memory — close something large first, or run this model on llama.cpp, which can leave it on disk.`;
}

/** The Setup rows of a Strata run. llama.cpp's rows (`setup.ts`) describe
 *  flags this engine does not have. */
export function strataSetupRows(a: {
  model: StrataModel;
  ctx: number;
  reserveMib: number;
  cards: number;
  ramB: number;
  /** The engine's name; absent = Strata. */
  name?: string;
  /** The cache format (`--kv`), when the engine takes one. */
  kv?: string | null;
}): SetupRow[] {
  const name = a.name ?? "Strata";
  const ctx = a.ctx.toLocaleString("en-US");
  const reserve = bytes(a.reserveMib * 2 ** 20);
  // A one-card path given several cards is a layer split (Basalt).
  const split = a.model.cards === "one" && a.cards > 1;
  const one = a.model.cards === "one" && !split;
  return [
    {
      label: "Engine",
      value: a.model.title
        ? `${name} · ${a.model.title} · ${a.model.model}`
        : `${name} · ${a.model.model}`,
      short: name,
      tip:
        "A second engine behind the same Start button: it keeps the most-used experts on the GPU and drafts ahead, and the target checks every draft. llama.cpp's flags do not apply to it.",
    },
    {
      label: "Runs on",
      value: split
        ? `${a.cards} GPUs, the layers split between them`
        : one
        ? "1 GPU, the rest computed on the CPU"
        : `${a.cards} GPU${a.cards === 1 ? "" : "s"}, filled to the reserve`,
      short: split
        ? `${a.cards} GPUs split`
        : one
        ? "1 GPU"
        : `${a.cards} GPU${a.cards === 1 ? "" : "s"} full`,
      tip: split
        ? `Each card runs a part of the layers and holds their cache, so the context can be longer than one card allows. Every expert stays on a GPU.`
        : one
        ? `${name}'s path for this model runs on one card; every expert that does not fit there is computed on the CPU.`
        : "Strata decides the placement itself and uses every card it is given. A full card is the design, not a leak.",
    },
    {
      label: "Kept free per GPU",
      value: reserve,
      short: `${reserve} free`,
      tip: split
        ? `${name} takes one number for every card: your per-GPU reserve plus its own scratch. The display card's extra reserve cannot be held in a split.`
        : one
        ? `Your reserve for the card it runs on, plus what ${name} keeps for its own scratch. More reserve means fewer experts on the GPU, so less speed.`
        : "Your reserve for the display card, applied to every card — Strata takes one number. More reserve means fewer experts on the GPU, so less speed.",
    },
    ...(a.kv
      ? [{
        label: "Cache",
        value: a.kv === "fp16" ? "fp16 (exact)" : a.kv,
        short: a.kv === "fp16" ? "cache exact" : `cache ${a.kv}`,
        tip: a.kv === "fp16"
          ? "The context is kept at full precision."
          : `The context is kept in ${a.kv}: about ${
            a.kv === "int8" ? "60" : "55"
          }% of the memory, so a longer context fits, at a small cost in precision.`,
      }]
      : []),
    {
      label: "Context",
      value: `${ctx} tokens`,
      short: `ctx ${ctx}`,
      tip:
        `Rounded down to a size ${name} offers. A longer context leaves less GPU memory for experts.`,
    },
    {
      label: "RAM",
      value: `about ${bytes(a.ramB)}`,
      short: `~${bytes(a.ramB)} RAM`,
      tip: a.model.ramGB
        ? "The token embedding and the engine's host buffers, pinned in RAM. Measured on this model."
        : a.model.memTotalGB && a.model.family === "gyro"
        ? `Gyro's weights take about ${a.model.memTotalGB} GB of VRAM and RAM together; the rest of the file (the n-gram table) stays on the SSD.`
        : "The experts that do not fit on the GPU, copied into RAM (or read from the SSD when RAM is short). Measured on one file; an estimate for others.",
    },
    {
      label: "Requests",
      value: "one at a time",
      short: "1 at a time",
      tip: "By default a second request waits for the first.",
    },
  ];
}
