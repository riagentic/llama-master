// src/lib/basalt.ts — Basalt, a third engine behind the same Start button.
//
// Basalt (github.com/jesdga95/basalt, MIT) is a Strata fork rebuilt for
// Blackwell (sm_120): the official NVFP4 experts on the FP4 tensor cores, a
// DSpark drafter whose every draft the target checks (output is plain
// decoding's), and one compiled engine under a Python server. Measured
// 2026-11-25 on one RTX PRO 4000 Blackwell (24 GB, the headless card), the
// official Inkling 2 Mini checkpoint, the app's own bench prompts, 512
// tokens, median of 3: 126.5 tok/s prose and 166.9 code, against Strata's
// 48.9 / 43.4 (SQ-20 pack) and llama.cpp's 35.7 / 27.4 (Q4_K_M) — and vLLM,
// which cannot hold the model on one 24 GB card at all, 106.7 / 191.1 across
// BOTH cards. Load 4 s warm, 25 s cold.
//
// It shares Strata's vocabulary (a pack, one card, a reserve, context steps),
// so its model is a `StrataModel` and its run is drawn by the same views.
// Pure: names and numbers in, argv and a diagnosis out.
//
// Two things this app holds it to:
//   - the WEIGHTS are the official release. Its pack is built from Thinking
//     Machines' checkpoint (experts bit for bit, dense projections Q8_0, the
//     drafter whole); the app never runs a file it was not shown — the offer
//     says the pack is downloaded and that it replaces the GGUF on disk;
//   - nothing lossy is switched on by the app. `--route-prior` and the
//     expert-dropping flags trade accuracy for speed (Basalt's
//     docs/precision/quality.md) and the argv never carries them; the KV
//     cache is fp16 unless the USER picks int8 or fp8 for a longer context.

import type { Diagnosis } from "./diagnose.ts";
import type { EngineParts } from "./plan.ts";
import type { StrataModel } from "./strata.ts";
import { formatRef, parseRef } from "./srcref.ts";

const REPO = "jesdga95/basalt";
export const BASALT_REF = formatRef({ kind: "fork", repo: REPO, ref: null });
export const BASALT_DOCS = `https://github.com/${REPO}`;

/** The engine binary, relative to the build. `basalt serve` runs Basalt's
 *  Python server from the same tree, which starts this binary again as its
 *  engine — both under the builds root, so the sandbox holds for each. */
export const BASALT_BIN = "bin/basalt";

/** Where a build finds its packs: a link into the app's cache, so an engine
 *  update or rebuild never fetches 20 GB again. */
export const BASALT_PACKS = "packs";

export function isBasaltRef(ref: string): boolean {
  const r = parseRef(ref);
  return r.kind === "fork" && r.repo.toLowerCase() === REPO.toLowerCase();
}

/** A published pack: what the build downloads when it is not on disk. */
export type BasaltPack = {
  file: string;
  url: string;
  sizeB: number;
  sha256: string;
};

/** Inkling 2 Mini, `jesdga/Inkling-2-Mini-Basalt` at c2afac3 (2026-11-23).
 *  Built from the official release; every weight section was compared with
 *  a pack converted here from the checkpoint and is bit-identical. */
export const INKLING_PACK: BasaltPack = {
  file: "Inkling-2-Mini-NVFP4-MX8.basalt",
  url:
    "https://huggingface.co/jesdga/Inkling-2-Mini-Basalt/resolve/c2afac3adb772e14d77b2ea2ac7068ad4460e932/Inkling-2-Mini-NVFP4-MX8.basalt",
  sizeB: 19_606_374_189,
  sha256: "28ec70723feaa037942a02f55b9d5e898276f3e62a3b1fa083745f932a041b99",
};

/** Context steps. One card holds every expert to ~64K at fp16 here; past it
 *  Basalt moves the least-routed experts to RAM by itself, slower but whole —
 *  until the cache alone outgrows the card (`basaltMaxContext`). A bigger
 *  step costs nothing on a short chat: the experts come back to the card
 *  while the cache is empty (fp16 448K 127.0 / 172.2 tok/s against 262K's
 *  127.0 / 173.4). */
export const BASALT_CONTEXTS: readonly number[] = [
  8192,
  32768,
  65536,
  131072,
  262144,
  393216,
  524288,
  786432,
  1048576,
];

/** An official Inkling 2 Mini GGUF: the name goes straight from the model to
 *  a quantisation label. A fine-tune (`-Uncensored-`, `-Heretic-`) is other
 *  weights, and Basalt would silently run the original in its place. */
const INKLING_FILE =
  /^Inkling-2-Mini-(?:UD-)?(?:I?Q\d\w*|BF16|F16|F32|MXFP\d\w*|NVFP4\w*)(?:-\d{5}-of-\d{5})?\.gguf$/i;

/** The Basalt model a file stands for, or null. */
export function basaltModel(file: string): StrataModel | null {
  const name = file.split(/[\\/]/).pop() ?? "";
  // The GGUF names the model; the pack IS it.
  const pack = name.toLowerCase() === INKLING_PACK.file.toLowerCase();
  if (!pack && !INKLING_FILE.test(name)) return null;
  return {
    family: "inkling2",
    model: "NVFP4-MX8",
    tag: "inkling-2-mini",
    title: "Inkling 2 Mini",
    cards: "one",
    maxContext: 1_048_576,
    contexts: BASALT_CONTEXTS,
    packGB: Math.round(INKLING_PACK.sizeB / 1e8) / 10,
    // `[server] loaded 17.0GiB cuda1=16.0GiB ram=1.0GiB`, at 32K and 64K.
    ramGB: 1.1,
    load: "under a minute",
  };
}

/** The pack a model runs from. One family today; the lookup is the place a
 *  second one goes. */
export function basaltPack(_model: StrataModel): BasaltPack {
  return INKLING_PACK;
}

/** Basalt's own floor for its graphs, scratch and head on the main card. */
const SCRATCH_MIB = 700;
/** Basalt's default RAM kept free for the system. */
const RAM_FLOOR_MIB = 2048;

/** The cache formats Basalt takes for Inkling (it refuses its 4-bit ones).
 *  fp16 is exact and the default; int8 keeps an fp16 scale per 64 values;
 *  fp8 is the smallest and the roughest. */
export type BasaltKv = "fp16" | "int8" | "fp8";
export const BASALT_KVS: readonly BasaltKv[] = ["fp16", "int8", "fp8"];

/** Cache bytes per context token. Measured 2026-11-25 on one 24 GB card,
 *  24,450 MiB free as this app counts it (total − used), 700 MiB reserved, at
 *  1,048,576 (`spill plan: even with every expert in RAM…`): fp16 19,084 MiB
 *  short, int8 4,677, fp8 2,619 — and fp16 524,288 652 short. Over 2^20
 *  tokens a MiB is a byte per token: fp16 36,864 (36 KiB exactly), int8
 *  22,457, fp8 20,399. Besides the cache the card holds 5,970 MiB with every
 *  expert in RAM (dense weights, drafter, graphs). */
const KV_B_PER_TOKEN: Record<BasaltKv, number> = {
  fp16: 36_864,
  int8: 22_457,
  fp8: 20_399,
};
const FIXED_B = 5970 * 2 ** 20;

/** The largest step Basalt will accept on ONE card with `freeB` free and its
 *  `--vram-reserve-mib` held back. A bigger request is not slower, it is a
 *  refusal: past this the cache alone does not fit beside the dense weights. */
export function basaltMaxContext(
  freeB: number,
  reserveMib: number,
  kv: BasaltKv = "fp16",
): number {
  const roomB = freeB - reserveMib * 2 ** 20 - FIXED_B;
  const fit = BASALT_CONTEXTS.filter((c) => c * KV_B_PER_TOKEN[kv] <= roomB);
  return fit.length ? fit[fit.length - 1]! : BASALT_CONTEXTS[0]!;
}

/** A layer split keeps every expert on the cards: what both cards held at
 *  fp8 1M (40,422 MiB, measured free-before less free-after) less its cache. */
const SPLIT_FIXED_B = 20_023 * 2 ** 20;
/** The cut lands on a whole layer, and a layer of Inkling's looped core holds
 *  3 of the 48 caches plus ~725 MiB of weights — so the cards can be that much
 *  out of balance. With it, int8 1M is excluded (Basalt refused it 1 GB short
 *  on the main card) and fp8 1M is kept (it ran, 2.9 + 1.4 GB left). */
const CORE_LAYER_CACHE = 3 / 48;
const CORE_LAYER_WEIGHTS_B = 725 * 2 ** 20;

/** The largest step for a layer split over `freeB` (one entry per card). */
export function basaltSplitMaxContext(
  freeB: readonly number[],
  reserveMib: number,
  kv: BasaltKv = "fp16",
): number {
  const roomB = freeB.reduce(
    (a, f) => a + Math.max(0, f - reserveMib * 2 ** 20),
    0,
  );
  const fit = BASALT_CONTEXTS.filter((c) => {
    const cacheB = c * KV_B_PER_TOKEN[kv];
    return SPLIT_FIXED_B + cacheB + cacheB * CORE_LAYER_CACHE +
        CORE_LAYER_WEIGHTS_B <= roomB;
  });
  return fit.length ? fit[fit.length - 1]! : BASALT_CONTEXTS[0]!;
}

/** The cap for the cards a run is given: one, or a split over several. */
export function basaltCap(
  freeB: readonly number[],
  reserveMib: number,
  kv: BasaltKv,
): number {
  return freeB.length > 1
    ? basaltSplitMaxContext(freeB, reserveMib, kv)
    : basaltMaxContext(freeB[0] ?? 0, reserveMib, kv);
}

export type BasaltRun = {
  /** Absolute path of the build (its `dir`). */
  buildDir: string;
  /** The selected model file. A `.basalt` pack runs from where it is; a GGUF
   *  only names the model, which runs from the build's `packs/`. */
  modelPath: string;
  model: StrataModel;
  ctx: number;
  /** The user's reserve, in bytes: for one card, that card's; for a split,
   *  the per-GPU figure — Basalt takes one number for every card. */
  reserveB: number;
  /** The user's RAM reserve, in bytes. */
  ramReserveB: number;
  host: string;
  port: number;
  /** The cards, as nvidia-smi numbers them, main card first. More than one
   *  is a layer split. */
  gpus: readonly number[];
  /** Free VRAM on each of them (total − used). Absent: no cap is applied. */
  freeB?: readonly number[];
  kv: BasaltKv;
};

const mib = (b: number) =>
  Number.isFinite(b) && b > 0 ? Math.round(b / 2 ** 20) : 0;

/** The context a run gets: the request rounded DOWN to a step. */
export function basaltContext(ctx: number): number {
  const fit = BASALT_CONTEXTS.filter((c) => c <= ctx);
  return fit.length ? fit[fit.length - 1]! : BASALT_CONTEXTS[0]!;
}

/**
 * Everything after the binary: `basalt serve PACK` and the engine flags the
 * server passes on. The one argv — Start spawns it, the Command panel draws
 * it, and `strataRunOf` reads the run back from it.
 */
export function basaltArgs(r: BasaltRun): string[] {
  const reserveMib = SCRATCH_MIB + mib(r.reserveB);
  return [
    "serve",
    /\.basalt$/i.test(r.modelPath)
      ? r.modelPath
      : `${r.buildDir}/${BASALT_PACKS}/${basaltPack(r.model).file}`,
    "--gpu",
    r.gpus.join(","),
    // An even share, not Basalt's default (by free VRAM): Inkling's looped
    // core holds 36 of the 48 caches, and only a cut inside it balances the
    // cards — the default cut at layer 15 left the main card 1.1 GB short at
    // 1M, 1:1 cuts at 14 and runs.
    ...(r.gpus.length > 1
      ? ["--gpu-mode", "split", "--gpu-share", r.gpus.map(() => "1").join(",")]
      : []),
    "--kv",
    r.kv,
    "--max-context",
    String(basaltContext(
      r.freeB === undefined
        ? r.ctx
        : Math.min(r.ctx, basaltCap(r.freeB, reserveMib, r.kv)),
    )),
    "--vram-reserve-mib",
    String(reserveMib),
    "--ram-reserve-mib",
    String(Math.max(RAM_FLOOR_MIB, mib(r.ramReserveB))),
    "--host",
    r.host,
    "--port",
    String(r.port),
  ];
}

/** Basalt's kernels are sm_120: compute capability 12.x (RTX 50, RTX PRO
 *  Blackwell). A Hopper or Ada card cannot load them. */
export function basaltCard(computeCaps: readonly number[]): boolean {
  return computeCaps.some((c) => c >= 12 && c < 13);
}

type Installed = { id: string; ref: string };

/** Why a Basalt build cannot run this model. Blocks Start. */
export function basaltMismatch(
  file: string,
  buildRef: string,
  installed: readonly Installed[],
): Diagnosis | null {
  if (!isBasaltRef(buildRef) || basaltModel(file)) return null;
  const other = installed.find((b) => !isBasaltRef(b.ref));
  return {
    reason:
      "The active build is Basalt, and this app runs Basalt for Inkling 2 Mini only (an official quantisation of it, not a fine-tune). This model needs llama.cpp.",
    steps: [
      other
        ? {
          text: `Switch to another build (${other.id}).`,
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
 * "This model has a faster engine" — an offer, never a block. Null when
 * Basalt is active, the file is not one of its, or no card is Blackwell.
 */
export function basaltOffer(
  file: string,
  buildRef: string,
  installed: readonly Installed[],
  computeCaps: readonly number[],
): Diagnosis | null {
  const model = basaltModel(file);
  if (!model || isBasaltRef(buildRef) || !basaltCard(computeCaps)) return null;
  const ready = installed.find((b) => isBasaltRef(b.ref));
  return {
    reason:
      "A faster engine exists for this model. Basalt runs Inkling 2 Mini from the official NVFP4 release on the Blackwell tensor cores and drafts ahead; every draft is checked, so the answer is the same. Measured here on one 24 GB card: 126 tok/s on prose and 167 on code, against Strata's 49 / 43 and llama.cpp's 36 / 27. It runs on one card, loads in under a minute, and its context comes in fixed steps.",
    steps: [
      ready
        ? {
          text: `Switch to the Basalt build you already have (${ready.id}).`,
          action: { kind: "use-build", id: ready.id },
        }
        : {
          text:
            `Build Basalt — about ten minutes of compiling here, no root. It does not read this file: it downloads the official weights as its own pack (${model.packGB} GB, verified), once.`,
          action: { kind: "use-ref", ref: BASALT_REF },
        },
      {
        text: "What Basalt is and how it works.",
        action: { kind: "open-url", url: BASALT_DOCS },
      },
    ],
  };
}

const rec = (v: unknown): Record<string, unknown> =>
  v !== null && typeof v === "object" ? v as Record<string, unknown> : {};
const bytesOf = (v: unknown): number =>
  typeof v === "number" && Number.isFinite(v) && v > 0 ? v : 0;

/**
 * What a running Basalt holds, from its own `/metrics` (JSON, not
 * Prometheus): `hardware.gpus[].vram_parts` per card by nvidia-smi `index`,
 * and `hardware.ram_parts.experts`. Null when the answer has no card figures
 * — an older engine, or a server that is not Basalt.
 */
export function basaltParts(json: unknown): EngineParts | null {
  const hw = rec(rec(json).hardware);
  const gpus = Array.isArray(hw.gpus) ? hw.gpus : [];
  const cards = gpus.flatMap((g) => {
    const c = rec(g);
    const parts = rec(c.vram_parts);
    return typeof c.index === "number" && c.vram_parts
      ? [{
        index: c.index,
        expertsB: bytesOf(parts.experts),
        kvB: bytesOf(parts.kv),
        otherB: bytesOf(parts.other),
      }]
      : [];
  });
  return cards.length > 0
    ? { cards, ramExpertsB: bytesOf(rec(hw.ram_parts).experts) }
    : null;
}
