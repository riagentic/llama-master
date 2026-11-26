// src/lib/packs.ts — models that are not GGUF: an engine's own pack.
//
// Basalt runs `.basalt` packs and Strata its `.strata` packs. They are models
// like any other — the Models list shows them — but only their engine can
// read them: llama.cpp cannot open one, and the planner has no header to read,
// so the engine places it itself. Pure: a file name in, words and a block out.

import type { Diagnosis } from "./diagnose.ts";
import {
  BASALT_DOCS,
  BASALT_REF,
  basaltCard,
  basaltModel,
  isBasaltRef,
} from "./basalt.ts";
import { isStrataRef, STRATA_DOCS, STRATA_REF, strataModel } from "./strata.ts";

export type PackEngine = "basalt" | "strata";

const ENGINES = {
  basalt: {
    name: "Basalt",
    ref: BASALT_REF,
    docs: BASALT_DOCS,
    isRef: isBasaltRef,
    model: basaltModel,
  },
  strata: {
    name: "Strata",
    ref: STRATA_REF,
    docs: STRATA_DOCS,
    isRef: isStrataRef,
    model: strataModel,
  },
} as const;

/** The engine whose pack `file` is, by its extension; null for a GGUF. */
export function packEngine(file: string): PackEngine | null {
  const f = file.toLowerCase();
  return f.endsWith(".basalt")
    ? "basalt"
    : f.endsWith(".strata")
    ? "strata"
    : null;
}

export type PackInfo = {
  /** The engine that reads it, in words. */
  engine: string;
  title: string;
  /** The quantisation, as the engine names it (`NVFP4-MX8`, `SQ-20`). */
  quant: string;
  maxContext: number;
};

/** What a pack is, or null for a GGUF or a pack this app does not run. */
export function packInfo(file: string): PackInfo | null {
  const e = packEngine(file);
  const m = e ? ENGINES[e].model(file) : null;
  if (!e || !m) return null;
  return {
    engine: ENGINES[e].name,
    title: m.title ?? m.family,
    quant: m.model,
    maxContext: m.maxContext ?? 0,
  };
}

type Installed = { id: string; ref: string };

/** Why the active build cannot read this pack — any build but its own
 *  engine. Blocks Start; on its own engine that engine's rules judge it. */
export function packMismatch(
  file: string,
  buildRef: string,
  installed: readonly Installed[],
  computeCaps: readonly number[],
): Diagnosis | null {
  const e = packEngine(file);
  if (!e || ENGINES[e].isRef(buildRef)) return null;
  const { name, ref, docs, isRef } = ENGINES[e];
  const info = packInfo(file);
  const about = {
    text: `What ${name} is.`,
    action: { kind: "open-url", url: docs },
  } as const;
  if (!info) {
    return {
      reason:
        `This is a ${name} pack of a model llama.master does not run with ${name} yet, and only ${name} reads a pack.`,
      steps: [about],
    };
  }
  if (e === "basalt" && !basaltCard(computeCaps)) {
    return {
      reason:
        `This is a ${name} pack, and ${name} needs an NVIDIA Blackwell card (RTX 50 or RTX PRO Blackwell) — none is in this machine.`,
      steps: [about],
    };
  }
  const ready = installed.find((b) => isRef(b.ref));
  return {
    reason:
      `This is a ${name} pack of ${info.title} (${info.quant}), not a GGUF: only ${name} reads it.`,
    steps: [
      ready
        ? {
          text: `Switch to the ${name} build you already have (${ready.id}).`,
          action: { kind: "use-build", id: ready.id },
        }
        : {
          text: `Build ${name} — it compiles here with no root.`,
          action: { kind: "use-ref", ref },
        },
      about,
    ],
  };
}
