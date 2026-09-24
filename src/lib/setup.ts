// src/lib/setup.ts — the running configuration, in words.
//
// The command view answers "what exactly will be spawned"; this answers "what
// does that mean". Same inputs as the command — the settings a running server
// was STARTED with, or the working settings as a projection — so the two can
// never disagree, and a flag someone cannot read (`-ctk q8_0`, `-np`,
// `--spec-type draft-mtp`) has a sentence next to it.
//
// Pure: settings + header facts in, rows out. Every value is derived through
// the same helpers the planner uses (`offloadRange`, `effectiveCtx`,
// `specMtpActive`), because a summary that recomputes its own answers is a
// second code path waiting to drift.

import type { Hw, ModelMeta, Settings } from "./types.ts";
import { bool, num, str } from "./params.ts";
import { offloadRange, slotOnGpu } from "./devsplit.ts";
import { effectiveCtx, specMtpActive } from "./plan.ts";

export type SetupRow = {
  label: string;
  value: string;
  /** The chip form for the all-in-one page — same fact, fewest characters.
   *  The full sentence stays reachable as the chip's hover. */
  short: string;
  /** One sentence of why/what, for the hover. */
  tip?: string;
};

const fmt = (n: number): string => n.toLocaleString("en-US");

/**
 * The setup as a human would describe it, most consequential first.
 *
 * Rows that merely restate a default the user never touched are still shown
 * when they decide something (offload, threads, cache types, context) — the
 * point is a complete picture in one glance. Rows that only matter when set
 * (`-ts`, `-nkvo`, experts on CPU) appear only then.
 */
export function setupRows(
  meta: ModelMeta | null,
  s: Settings,
  hw: Hw,
): SetupRow[] {
  const rows: SetupRow[] = [];
  const nLayer = meta?.nLayer ?? 0;
  const off = offloadRange(nLayer, s);
  const gpus = hw.gpus.length;

  // GPU offload — the placement in one line.
  if (gpus === 0) {
    rows.push({
      label: "GPU offload",
      value: "no GPU — everything on the CPU",
      short: "no GPU",
    });
  } else if (off.count === 0) {
    rows.push({
      label: "GPU offload",
      value: "off — CPU only",
      short: "GPU off",
      tip: "GPU layers is 0, so the card is idle.",
    });
  } else if (nLayer > 0) {
    const head = slotOnGpu(nLayer, off);
    const layersOn = off.count - (head ? 1 : 0);
    rows.push({
      label: "GPU offload",
      value: layersOn >= nLayer
        ? `all ${fmt(nLayer)} layers + output head`
        : `${fmt(layersOn)} of ${fmt(nLayer)} layers${
          head ? " + output head" : ""
        }`,
      short: layersOn >= nLayer
        ? `GPU all ${fmt(nLayer)}+head`
        : `GPU ${fmt(layersOn)}/${fmt(nLayer)}${head ? "+head" : ""}`,
      tip:
        "llama.cpp offloads the LAST N slots; the token-embedding table always stays in system RAM (input tensors are pinned to the CPU).",
    });
  } else {
    rows.push({
      label: "GPU offload",
      value: `${fmt(num(s, "ngl"))} layers requested`,
      short: `GPU ${fmt(num(s, "ngl"))}`,
    });
  }

  const moe = Math.max(0, Math.min(num(s, "nCpuMoe"), nLayer || Infinity));
  if (moe > 0) {
    rows.push({
      label: "MoE experts on CPU",
      value: `first ${fmt(moe)} layers' experts in system RAM`,
      short: `MoE→RAM ${fmt(moe)}`,
      tip:
        "--n-cpu-moe: attention stays on the GPU; the routed experts of these layers run from RAM.",
    });
  }

  // Threads — the pool the generation actually runs on.
  const t = num(s, "threads");
  const tb = num(s, "threadsBatch");
  const cores = hw.cpu?.cores ?? 0;
  const th = (n: number) => n > 0 ? fmt(n) : "auto";
  rows.push({
    label: "CPU threads",
    value: `${th(t)} generate · ${th(tb > 0 ? tb : t)} batch` +
      (cores > 0 ? ` (${fmt(cores)} physical cores)` : ""),
    short: `t ${th(t)}·${th(tb > 0 ? tb : t)}`,
    tip:
      "One thread per physical core is the measured optimum here — SMT siblings share a memory port and collapse throughput.",
  });

  // The cache, both halves — quantisation is per K and per V.
  const ctk = str(s, "cacheTypeK");
  const ctv = str(s, "cacheTypeV");
  const cache = (v: string) =>
    v === "f16"
      ? "f16 — full precision (default)"
      : v.startsWith("q")
      ? `${v} — quantised, ~${
        v === "q8_0" ? "half" : "a quarter of"
      } the f16 size`
      : v;
  rows.push({
    label: "K cache type",
    value: cache(ctk),
    short: `K ${ctk}`,
    tip: "-ctk: the precision the attention KEYS are cached at.",
  });
  rows.push({
    label: "V cache type",
    value: cache(ctv),
    short: `V ${ctv}`,
    tip: "-ctv: the precision the attention VALUES are cached at.",
  });
  if (bool(s, "noKvOffload")) {
    rows.push({
      label: "KV cache placement",
      value: "system RAM (-nkvo)",
      short: "KV→RAM",
      tip: "The whole cache stays on the host, whatever the layer placement.",
    });
  }

  const fa = str(s, "flashAttn");
  rows.push({
    label: "Flash attention",
    value: fa === "on"
      ? "on"
      : fa === "off"
      ? "off"
      : "auto — llama.cpp decides",
    short: `FA ${fa || "auto"}`,
    tip:
      "Smaller attention buffers and faster long contexts; a prerequisite for a quantised cache.",
  });

  // Context — the number, and whether it is the model's own edge.
  const ctx = meta ? effectiveCtx(meta, s) : num(s, "ctxSize");
  const train = meta?.nCtxTrain ?? 0;
  rows.push({
    label: "Context",
    value: `${fmt(ctx)} tokens` +
      (train > 0
        ? ctx >= train ? " — the trained maximum" : ` of ${fmt(train)} trained`
        : ""),
    short: `ctx ${fmt(ctx)}${train > 0 && ctx >= train ? " max" : ""}`,
    tip: train > 0 && ctx >= train
      ? "The length the model was trained for; past it RoPE extrapolates and answers degrade."
      : undefined,
  });

  const np = Math.max(1, num(s, "parallel"));
  rows.push({
    label: "Parallel slots",
    value: np === 1
      ? "1 — the whole context serves one conversation"
      : `${fmt(np)} — the context is shared between ${fmt(np)} conversations`,
    short: `np ${fmt(np)}`,
    tip:
      "-np: each slot runs its own graph and, on some models, its own copy of the context-sized scratch.",
  });

  rows.push({
    label: "Speculative decoding",
    value: meta && specMtpActive(meta, s)
      ? "on — the model's own MTP draft block"
      : str(s, "specType") !== ""
      ? str(s, "specType")
      : "off",
    short: meta && specMtpActive(meta, s)
      ? "MTP"
      : str(s, "specType") !== ""
      ? `spec ${str(s, "specType")}`
      : "no spec",
    tip:
      "Lossless: the full model verifies every drafted token, so only the speed changes.",
  });

  rows.push({
    label: "Batch sizes",
    value: `${fmt(num(s, "batchSize"))} batch · ${
      fmt(num(s, "ubatchSize"))
    } micro-batch`,
    short: `b ${fmt(num(s, "batchSize"))}·ub ${fmt(num(s, "ubatchSize"))}`,
    tip: "-b / -ub: how many tokens are processed per step during prompt work.",
  });

  // How the weights get into memory — ONE setting, `-lm/--load-mode`. It was
  // read here as the two booleans that `cfg` version 4 folded into it, which no
  // longer exist, so every run was described as memory-mapped whatever it did.
  const mode = str(s, "loadMode") || "auto";
  const load = LOAD_MODES[mode] ?? LOAD_MODES.auto!;
  rows.push({
    label: "Weights loading",
    value: load.value,
    short: load.short,
    tip:
      "-lm / --load-mode: one llama.cpp setting. --mlock and --no-mmap were two spellings of it, which is why only one is ever shown.",
  });

  const ts = str(s, "tensorSplit").trim();
  if (ts && gpus > 1) {
    rows.push({
      label: "Tensor split",
      value: `-ts ${ts} — pins how the layers divide across the cards`,
      short: `ts ${ts}`,
      tip:
        "Without it llama.cpp divides by each card's free VRAM at load time.",
    });
  }

  const host = str(s, "host") || "127.0.0.1";
  const port = num(s, "port");
  rows.push({
    label: "Listening on",
    // A port is an identifier, not a quantity — no thousands separator.
    value: `${host}:${port}` +
      (host === "0.0.0.0" || host === "::"
        ? " — reachable from the LAN"
        : " — this machine only"),
    short: `${host === "0.0.0.0" || host === "::" ? "LAN" : "local"} :${port}`,
  });

  return rows;
}

/** What each `--load-mode` value does, in the words the setup table uses. */
const LOAD_MODES: Record<string, { value: string; short: string }> = {
  auto: {
    value: "memory-mapped — loaded on demand, near-instant warm starts",
    short: "mmap",
  },
  mmap: {
    value: "memory-mapped — loaded on demand, near-instant warm starts",
    short: "mmap",
  },
  none: { value: "copied up front (-lm none)", short: "no-mmap" },
  mlock: {
    value: "locked in RAM (-lm mlock) — the OS may not page them out",
    short: "mlock",
  },
  "mmap+mlock": {
    value: "memory-mapped and locked (-lm mmap+mlock) — never paged out",
    short: "mmap+mlock",
  },
  dio: {
    value: "read with direct I/O (-lm dio) — bypasses the page cache",
    short: "dio",
  },
};
