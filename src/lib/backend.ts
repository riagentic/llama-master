// src/lib/backend.ts — can this machine actually compile that backend?
//
// A source build with `-DGGML_CUDA=ON` and no `nvcc` does not fail fast: cmake
// configures for a while, then stops with a message about a missing CUDA
// toolkit — minutes in, several screens down a log. The information needed to
// prevent that is already on the Prerequisites panel, so the Build tab should
// use it and refuse up front, naming the tool.
//
// A prebuilt release needs none of this: the binary is already compiled, so
// only the driver matters, and that is a run-time concern the server log will
// report honestly if it is missing.
//
// Pure: a backend and what was detected in, a verdict out.

import type { Backend } from "./types.ts";
import type { Diagnosis } from "./diagnose.ts";
import { diagnoseNoAsset } from "./diagnose.ts";

/** Prerequisite ids (as detected in prereq.server.ts) each backend needs to
 *  COMPILE, beyond cmake and a C++ compiler which every source build needs. */
const COMPILE_NEEDS: Record<Backend, string[]> = {
  cpu: [],
  cuda: ["cuda"], // nvcc — and see src/lib/cuda.ts for the architecture check
  // glslc AND SPIRV-Headers. The headers are the one that actually stops a
  // build on a machine that looks equipped (ggml-vulkan/CMakeLists.txt:14).
  vulkan: ["vulkan", "spirv"],
  hip: ["hip"], // hipcc
  metal: [], // the Apple toolchain provides it
};

export type BuildReadiness = {
  ok: boolean;
  /** Prerequisite ids that are missing, in the order they should be fixed. */
  missing: string[];
  /** One line naming what is wrong, or "" when it is fine. */
  reason: string;
};

const LABELS: Record<string, string> = {
  cmake: "CMake",
  compiler: "a C++ compiler",
  cuda: "the CUDA toolkit (nvcc)",
  vulkan: "the Vulkan shader compiler (glslc)",
  spirv: "SPIRV-Headers",
  hip: "the ROCm toolchain (hipcc)",
};

function label(id: string): string {
  return LABELS[id] ?? id;
}

/**
 * Whether a source build of `backend` can even start here.
 *
 * `found` is the set of prerequisite ids that were detected.
 */
export function canCompile(
  backend: Backend,
  found: ReadonlySet<string>,
  os: string,
): BuildReadiness {
  const missing: string[] = [];
  for (const id of ["cmake", "compiler"]) {
    if (!found.has(id)) missing.push(id);
  }
  if (backend === "metal" && os !== "darwin") {
    return {
      ok: false,
      missing,
      reason: "Metal only exists on Apple hardware.",
    };
  }
  for (const id of COMPILE_NEEDS[backend]) {
    if (!found.has(id)) missing.push(id);
  }
  if (missing.length === 0) return { ok: true, missing: [], reason: "" };
  return {
    ok: false,
    missing,
    reason: `A ${backend.toUpperCase()} source build needs ${
      missing.map(label).join(" and ")
    }. Install it from the Machine tab, or download a prebuilt release instead.`,
  };
}

/**
 * Can the CURRENT selection — route AND backend — actually produce a build?
 *
 * The prerequisites panel answers "is this machine equipped to compile", which
 * is a different question from "will the thing I just selected work". Green
 * checkmarks next to a Build button that then fails is the exact experience
 * this function exists to prevent: it is asked before the button is enabled,
 * for both routes, and its answer is what the button's state and the banner
 * above it are made of.
 */
export function targetReadiness(
  origin: "source" | "release",
  backend: Backend,
  ctx: {
    platform: string;
    arch: string;
    found: ReadonlySet<string>;
    /** Backends with a prebuilt asset; null while the list is not yet known. */
    availableBackends: readonly Backend[] | null;
    assetCount: number;
    /** The precise reason a prerequisite is unmet, when detection knows one.
     *  "hipcc is installed but the HIP development package is not" is far more
     *  use than "needs the ROCm toolchain", and detection already worked it
     *  out — this is how that reaches the banner. */
    explain?: (id: string) => string | undefined;
  },
): { ok: boolean; diagnosis: Diagnosis | null; pending: boolean } {
  if (origin === "source") {
    const r = canCompile(backend, ctx.found, ctx.platform);
    if (r.ok) return { ok: true, diagnosis: null, pending: false };
    // Prefer detection's own words over the generic label.
    const detail = r.missing
      .map((id) => ctx.explain?.(id))
      .filter((x): x is string => Boolean(x));
    return {
      ok: false,
      pending: false,
      diagnosis: {
        reason: detail.length > 0 ? detail.join(" ") : r.reason,
        steps: [
          ...r.missing.map((id) => ({
            text: ctx.explain?.(id) ?? `Install ${label(id)}.`,
            action: { kind: "fix-prereq" as const, id },
          })),
          {
            text: "Or install a prebuilt release, which needs no toolchain.",
            action: { kind: "switch-origin" as const, to: "release" as const },
          },
        ],
      },
    };
  }

  // Release route: unknown until the asset list has been fetched. Saying
  // "ready" before we know would be the same lie in a different place.
  if (ctx.availableBackends === null) {
    return { ok: false, diagnosis: null, pending: true };
  }
  if (ctx.availableBackends.includes(backend)) {
    return { ok: true, diagnosis: null, pending: false };
  }
  return {
    ok: false,
    pending: false,
    diagnosis: diagnoseNoAsset(
      {
        origin,
        backend,
        platform: ctx.platform,
        arch: ctx.arch,
        availableBackends: ctx.availableBackends,
        found: ctx.found,
      },
      ctx.assetCount,
    ),
  };
}

/** Backends this machine could compile right now — used to pick a default that
 *  is not a dead end. */
export function compilableBackends(
  found: ReadonlySet<string>,
  os: string,
): Backend[] {
  const all: Backend[] = ["cpu", "cuda", "vulkan", "hip", "metal"];
  return all.filter((b) => canCompile(b, found, os).ok);
}

/**
 * The GPUs a build with this backend can actually use.
 *
 * llama.cpp's backends are vendor-specific: a CUDA build addresses NVIDIA
 * devices only, a HIP build AMD only, and a CPU build none at all. Vulkan and
 * Metal are the exceptions — Vulkan runs on anything with a driver, and on
 * Apple every GPU is a Metal GPU. Planning against devices the build cannot
 * reach produces a memory map of something that will never happen.
 */
export function usableGpus<G extends { vendor: string }>(
  backend: Backend | undefined,
  gpus: readonly G[],
): G[] {
  switch (backend) {
    case "cpu":
      return [];
    case "cuda":
      return gpus.filter((g) => g.vendor === "nvidia");
    case "hip":
      return gpus.filter((g) => g.vendor === "amd");
    // Vulkan is vendor-neutral; Metal only exists where every GPU is Apple's.
    // `undefined` means no build is selected yet — show the whole machine.
    default:
      return gpus.slice();
  }
}

/**
 * Backends worth suggesting on this machine, best first.
 *
 * Preference, not availability: the caller intersects this with what can
 * actually be had (assets for the release route, toolchains for source). CPU is
 * always last and always present, because it always works.
 */
export function preferredBackends(
  vendors: ReadonlySet<string>,
  os: string,
): Backend[] {
  if (os === "darwin") return ["metal", "cpu"];
  if (vendors.has("nvidia")) return ["cuda", "vulkan", "cpu"];
  if (vendors.has("amd")) {
    // ROCm is Linux-only; elsewhere an AMD card is reached through Vulkan.
    return os === "linux" ? ["hip", "vulkan", "cpu"] : ["vulkan", "cpu"];
  }
  if (vendors.has("intel")) return ["vulkan", "cpu"];
  return ["cpu"];
}

/**
 * The raised graph-split input cap, for builds meant to run extreme contexts.
 *
 * `GGML_SCHED_MAX_SPLIT_INPUTS` is 30 by default, behind an `#ifndef` — a
 * compile-time cap on how many tensors one graph split may pull across a
 * device boundary. With routed experts in RAM and attention across GPUs, this
 * model class needs more of them as the context grows: measured on
 * DeepSeek-V4 on 2×24 GB, a 262,144 context generates and 524,288 dies at
 * `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)` during load — memory
 * to spare, the constant was the wall. If ~30 inputs carry ~256k, the model's
 * full 1M needs roughly 4×; 1,024 is that with an 8× margin, and costs only
 * kilobytes of scheduler bookkeeping per split.
 */
export const SCHED_SPLIT_CAP = 1024;

/** The compiler define that raises the cap, or nothing for a stock build. */
export function schedCapFlags(bypass: boolean): string[] {
  return bypass ? [`-DGGML_SCHED_MAX_SPLIT_INPUTS=${SCHED_SPLIT_CAP}`] : [];
}
