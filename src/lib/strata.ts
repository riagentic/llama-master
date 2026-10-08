// src/lib/strata.ts — Strata, a second engine behind the same Start button.
//
// Strata (github.com/Niko1221/Strata, MIT) runs ONE model family —
// Qwen3.8-Flash-Next — and nothing else: a per-expert VRAM cache, drafting
// and its own kernels, where llama.cpp streams every routed expert from RAM.
// Measured 2026-10-08 on two 24 GB Blackwell cards, the same UD-Q4_K_XL file,
// the app's own bench prompts, both warm, a 4 GB reserve on each card: 53-64
// tok/s with its MTP draft layer (41-54 without) against llama.cpp's 11-15,
// and a 28,863-token prompt read at 2,690 tok/s against 159. Load 105 s warm.
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
//     disk; the only fetch left is its ~6.5 GB MTP draft layer, which the offer
//     below states before the button is pressed;
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
export type StrataModel = { family: string; model: string; tag: string };

/** `[family, filename pattern, tag prefix]` — setup.py's own `FAMILIES` and
 *  `MODELS` tables. Matched on the FIRST shard's name, anchored at both ends:
 *  Strata refuses every other GGUF of this model (K-quants, other Unsloth
 *  sizes), and a loose match would offer an engine that then says no. */
const FILES: readonly (readonly [string, RegExp, string])[] = [
  [
    "qwen",
    /^Qwen3\.8-Flash-Next-GSQ-RCO-(Q2_0|IQ2_XS|IQ3_XXS|IQ3_S)-00001-of-00002\.gguf$/,
    "",
  ],
  [
    "coder",
    /^Qwen3\.8-Flash-Next-GSQ-RCO-(IQ1_M)-00001-of-00002\.gguf$/,
    "coder-",
  ],
  [
    "swift",
    /^Swift-Qwen3\.8-Flash-Next-GSQ-RCO-(IQ2_XS|IQ3_XXS)-00001-of-00002\.gguf$/,
    "swift-",
  ],
  [
    "unsloth",
    /^Qwen3\.8-Flash-Next-(UD-Q4_K_XL)-00001-of-00004\.gguf$/,
    "unsloth-",
  ],
  [
    "unsloth",
    /^Qwen3\.8-Flash-Next-(UD-IQ4_XS)-00001-of-00003\.gguf$/,
    "unsloth-",
  ],
];

/** The Strata model a file IS, or null when Strata cannot run it. */
export function strataModel(file: string): StrataModel | null {
  const name = file.split(/[\\/]/).pop() ?? "";
  for (const [family, re, prefix] of FILES) {
    const m = re.exec(name);
    if (m) {
      return { family, model: m[1]!, tag: (prefix + m[1]!).toLowerCase() };
    }
  }
  return null;
}

/** The contexts Strata's setup offers. Every GB of KV cache is a GB less of
 *  cached experts, so the request is rounded DOWN to a step it was measured
 *  at, never up past what was asked for. */
export const STRATA_CONTEXTS: readonly number[] = [
  8192,
  32768,
  65536,
  131072,
  204800,
  262144,
];

export function strataContext(ctx: number): number {
  const fit = STRATA_CONTEXTS.filter((c) => c <= ctx);
  return fit.length ? fit[fit.length - 1]! : STRATA_CONTEXTS[0]!;
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
    "--gguf-dir",
    dir,
    "--gpus",
    "all",
    "--context",
    String(strataContext(r.ctx)),
    "--vram-reserve-mib",
    String(r.reserveMib),
    "--host",
    r.host,
    "--port",
    String(r.port),
  ];
}

/** The settings a Strata run actually has: the context it was rounded to.
 *  Everything that describes a live run reads `runSettings`, and a figure the
 *  engine never used would be a description of some other server. */
export function strataSettings(settings: Settings): Settings {
  return {
    ...settings,
    ctxSize: strataContext(Number(settings.ctxSize) || 0),
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
      "The active build is Strata, and Strata runs only the Qwen3.8-Flash-Next files it has kernels for (ISTA-DASLab's GSQ-RCO sizes, Unsloth's UD-Q4_K_XL and UD-IQ4_XS). This model needs llama.cpp.",
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
  if (!hasNvidia || isStrataRef(buildRef) || !strataModel(file)) return null;
  const ready = installed.find((b) => isStrataRef(b.ref));
  return {
    reason:
      "A faster engine exists for this model. Strata keeps the most-used experts on the GPU and drafts ahead; on the same file it measured about four times llama.cpp's generation rate and read a long prompt about seventeen times faster (two 24 GB cards, 2026-10-08). The price: a load of two to three minutes, one request at a time, and a context rounded down to a size it was tuned for.",
    steps: [
      ready
        ? {
          text: `Switch to the Strata build you already have (${ready.id}).`,
          action: { kind: "use-build", id: ready.id },
        }
        : {
          text:
            "Build Strata for this model — compiled here in 10-20 minutes with no root, using the model files you already have. It downloads one thing: the model's ~6.5 GB draft layer.",
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
): { ctx: number; reserveMib: number } | null {
  const after = (flag: string): number => {
    const i = argv.indexOf(flag);
    return i < 0 ? NaN : Number(argv[i + 1]);
  };
  const ctx = after("--context");
  const reserveMib = after("--vram-reserve-mib");
  if (!Number.isFinite(ctx) || !Number.isFinite(reserveMib)) return null;
  return { ctx, reserveMib };
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
}): { vramB: number; ramB: number } {
  const n = (x: number) => (Number.isFinite(x) && x > 0 ? x : 0);
  return {
    vramB: Math.max(
      0,
      n(a.freeVramB) - n(a.cards) * n(a.reserveMib) * 2 ** 20,
    ),
    ramB: Math.round(n(a.fileB) * RAM_SHARE),
  };
}

/**
 * Why Start would end in the OOM killer, or "". Strata's experts are
 * anonymous memory: with less free than it copies, the kernel does not slow
 * down, it kills something — and not necessarily Strata.
 */
export function strataRamShort(needB: number, availableB: number): string {
  if (!(needB > 0) || !(availableB > 0) || availableB >= needB) return "";
  return `Strata keeps about ${bytes(needB)} of this model in RAM, and ${
    bytes(availableB)
  } is free. Starting now would run the machine out of memory — close something large first, or run this model on llama.cpp, which can leave it on disk.`;
}

/** The Setup rows of a Strata run. llama.cpp's rows (`setup.ts`) describe
 *  flags this engine does not have. */
export function strataSetupRows(a: {
  model: StrataModel;
  ctx: number;
  reserveMib: number;
  cards: number;
  ramB: number;
}): SetupRow[] {
  const ctx = a.ctx.toLocaleString("en-US");
  const reserve = bytes(a.reserveMib * 2 ** 20);
  return [
    {
      label: "Engine",
      value: `Strata · ${a.model.model}`,
      short: "Strata",
      tip:
        "A second engine for Qwen3.8-Flash-Next: it keeps the most-used experts on the GPU and drafts ahead. llama.cpp's flags do not apply to it.",
    },
    {
      label: "Runs on",
      value: `${a.cards} GPU${a.cards === 1 ? "" : "s"}, filled to the reserve`,
      short: `${a.cards} GPU${a.cards === 1 ? "" : "s"} full`,
      tip:
        "Strata decides the placement itself and uses every card it is given. A full card is the design, not a leak.",
    },
    {
      label: "Kept free per GPU",
      value: reserve,
      short: `${reserve} free`,
      tip:
        "Your reserve for the display card, applied to every card — Strata takes one number. More reserve means fewer experts on the GPU, so less speed.",
    },
    {
      label: "Context",
      value: `${ctx} tokens`,
      short: `ctx ${ctx}`,
      tip:
        "Rounded down to a size Strata was tuned for. A longer context leaves less GPU memory for experts.",
    },
    {
      label: "RAM",
      value: `about ${bytes(a.ramB)}`,
      short: `~${bytes(a.ramB)} RAM`,
      tip:
        "The experts that do not fit on the GPU, copied into RAM. Measured on one file; an estimate for others.",
    },
    {
      label: "Requests",
      value: "one at a time",
      short: "1 at a time",
      tip: "A second request waits for the first.",
    },
  ];
}
