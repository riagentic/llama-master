// src/lib/stability.ts — is this configuration going to hurt?
//
// The app lets you change anything. That is the point: the tuner's answer is a
// starting position, not a cage. But a setting that will crash the load, swap
// the machine to a standstill, or freeze the desktop should say so BEFORE the
// server is started, not in a log afterwards.
//
// Pure, so the warning is computed from the same numbers the bars are drawn
// from and cannot disagree with them.

import { plan } from "./plan.ts";
import { bool, num, str } from "./params.ts";
import type { Hw, ModelMeta, Settings } from "./types.ts";

export type Severity = "risk" | "caution";

export type Warning = {
  severity: Severity;
  /** The parameter this is about, so the UI can point at the control. */
  key: string;
  message: string;
};

export type Stability = {
  /** `risk` = this will probably fail. `caution` = it will work, but badly. */
  level: "ok" | Severity;
  warnings: Warning[];
};

const GB = 1024 ** 3;

/**
 * Check a settings map against a model and a machine.
 *
 * Only checks that can be decided from measured numbers are here. "This quant
 * is bad for this model" is an opinion; "this needs 6 GB more VRAM than the
 * card has" is arithmetic, and arithmetic is what a warning should be made of.
 */
export function stability(
  meta: ModelMeta | null,
  hw: Hw,
  s: Settings,
  opts: { lowPriority?: boolean } = {},
): Stability {
  const warnings: Warning[] = [];
  const cores = hw.cpu?.cores ?? 0;
  const threads = hw.cpu?.threads ?? 0;

  if (meta) {
    const p = plan(meta, hw, s);

    if (p.vram.overB > 0) {
      warnings.push({
        severity: "risk",
        key: "ngl",
        message:
          `Needs ${
            (p.vram.overB / GB).toFixed(1)
          } GB more VRAM than the GPUs have. ` +
          `The load will fail or fall back to the CPU mid-model.`,
      });
    } else if (p.vram.capacityB > 0 && p.vram.freeB < 256 * 1024 * 1024) {
      warnings.push({
        severity: "caution",
        key: "ngl",
        message:
          "Under 256 MB of VRAM headroom. Anything else that uses the GPU — a browser, a screen recorder — can push this over.",
      });
    }

    if (p.ram.overB > 0) {
      warnings.push({
        severity: "risk",
        key: "ctxSize",
        message:
          `Needs ${(p.ram.overB / GB).toFixed(1)} GB more RAM than is free. ` +
          `Expect heavy swapping, or the OOM killer.`,
      });
    } else if (p.ram.capacityB > 0 && p.ram.freeB < GB) {
      // The mirror of the 256 MB VRAM caution above, and it was missing: model
      // weights and the KV cache are anonymous pages, so the kernel cannot
      // reclaim them under pressure. "It exactly fits" is not a safe answer.
      warnings.push({
        severity: "caution",
        key: "ctxSize",
        message: `Under ${
          (p.ram.freeB / GB).toFixed(1)
        } GB of RAM headroom. The weights and KV cache cannot be paged out, so anything you open next comes out of swap.`,
      });
    }

    if (
      str(s, "loadMode").includes("mlock") &&
      p.ram.usedB > (hw.mem?.availableB ?? 0) * 0.85
    ) {
      warnings.push({
        severity: "risk",
        key: "mlock",
        message:
          "--mlock pins the weights in physical memory. At this size the kernel has nothing left to reclaim, and the OOM killer picks a victim.",
      });
    }

    const ctx = num(s, "ctxSize");
    if (meta.nCtxTrain > 0 && ctx > meta.nCtxTrain) {
      warnings.push({
        severity: "caution",
        key: "ctxSize",
        message:
          `Context ${ctx} exceeds the ${meta.nCtxTrain} this model was trained for. ` +
          `Quality degrades past that point unless RoPE scaling is set.`,
      });
    }
  }

  // ── the machine has to stay usable ───────────────────────────────────────
  const t = num(s, "threads");
  if (cores > 0 && t > 0) {
    if (t > threads) {
      warnings.push({
        severity: "risk",
        key: "threads",
        message:
          `${t} threads on ${threads} logical CPUs. Oversubscription makes generation slower, not faster.`,
      });
    } else if (t > cores) {
      // Measured, not folklore: the same DeepSeek-V4 placement runs 15.91 tok/s
      // at 16 threads (= physical cores) and 0.94 tok/s at 32. Generation with
      // experts on the host is memory-bandwidth-bound, and two threads sharing
      // one core's memory port thrash rather than pipeline.
      warnings.push({
        severity: "risk",
        key: "threads",
        message:
          `${t} threads on ${cores} physical cores. SMT siblings share one memory port and generation is bandwidth-bound — measured 15x slower than one thread per core.`,
      });
    } else if (t >= cores && cores > 2 && !opts.lowPriority) {
      warnings.push({
        severity: "caution",
        key: "threads",
        message:
          'Every physical core is claimed and llama-server runs at normal priority, so the desktop competes with it. Turn on "Low priority" — it keeps the throughput and yields the CPU the moment anything else asks.',
      });
    }
  }

  // `-tb` was never checked at all, and prompt processing is the densest phase
  // there is: an oversubscribed batch thread count slows the phase the user
  // waits on most. Claiming every PHYSICAL core is not oversubscription — it is
  // the fastest setting measured (`tune.ts:cpuBudget`), and what keeps the
  // desktop responsive under it is the priority switch, not idle cores.
  const tb = num(s, "threadsBatch");
  if (cores > 0 && tb > 0) {
    if (tb > threads) {
      warnings.push({
        severity: "risk",
        key: "threadsBatch",
        message:
          `${tb} batch threads on ${threads} logical CPUs. Oversubscription slows prompt processing down.`,
      });
    } else if (tb > cores) {
      warnings.push({
        severity: "caution",
        key: "threadsBatch",
        message:
          `${tb} batch threads on ${cores} physical cores. SMT siblings share one memory port, and prompt processing is memory-bound — measured 15x slower at two threads per core.`,
      });
    }
  }

  // A draft model is a second set of weights in VRAM and the memory plan does
  // not know about it: `plan.ts` sizes the model at `-m`, and nothing reads
  // `-md`. Left unsaid, the bars would show room that the run is about to
  // spend, which is the one thing this app's memory picture must never do.
  if (str(s, "draftModel")) {
    warnings.push({
      severity: "caution",
      key: "draftModel",
      message:
        "A draft model loads its own weights into VRAM, and the memory plan below does not count them — leave headroom for the size of that file, or the run fails to allocate.",
    });
  }

  if (num(s, "ubatchSize") > num(s, "batchSize")) {
    warnings.push({
      severity: "risk",
      key: "ubatchSize",
      message:
        "Micro-batch is larger than the batch — llama.cpp will reject this.",
    });
  }

  if (num(s, "parallel") > 1 && meta) {
    const perSlot = num(s, "ctxSize") / num(s, "parallel");
    if (perSlot < 512) {
      warnings.push({
        severity: "caution",
        key: "parallel",
        message: `${num(s, "parallel")} slots split the context into ${
          Math.floor(perSlot)
        } tokens each — too little for a real conversation.`,
      });
    }
  }

  // llama-server drops backend sampling on its own, with a warning in the log,
  // when the sampler chain contains something it cannot run on the device
  // (`common/sampling.cpp`). A grammar or JSON schema is set per REQUEST rather
  // than in the argv, so that case cannot be decided here — what can be is the
  // one that IS in the argv, so the flag never sits on a command line looking
  // effective while doing nothing.
  //
  // There was a fourth case and it is deliberately NOT checked: until some
  // point between 2026-07-27 and 2026-09-07, backend sampling was also
  // disabled whenever a drafter was attached. That is gone from master, the two
  // now stack, and a warning written from a stale source cache would talk a
  // user out of the better configuration.
  if (bool(s, "backendSampling") && str(s, "reasoning") === "on") {
    warnings.push({
      severity: "caution",
      key: "backendSampling",
      message:
        "Sample on the GPU is on together with forced reasoning. A reasoning budget builds a sampler that llama.cpp cannot run on the backend, so it logs a warning and falls back to CPU sampling for those requests.",
    });
  }

  const host = String(s.host ?? "");
  const open = (host === "0.0.0.0" || host === "::") && !String(s.apiKey ?? "");
  if (open) {
    warnings.push({
      severity: "risk",
      key: "host",
      message:
        "Bound to every interface with no API key: anyone on the network can use this model, and read every prompt.",
    });
  }
  // /slots is on by default upstream and shows every prompt in flight — on a
  // private bind that is a debugging aid, on an open one it is a second copy
  // of the risk above with its own switch.
  if (open && s.slots !== false) {
    warnings.push({
      severity: "caution",
      key: "slots",
      message:
        "The /slots endpoint is on, and it shows every prompt in flight to anyone who can reach the server. Turn Expose slots off, or set an API key.",
    });
  }

  const level = warnings.some((w) => w.severity === "risk")
    ? "risk"
    : warnings.length > 0
    ? "caution"
    : "ok";
  return { level, warnings };
}
