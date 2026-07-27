// src/lib/diagnose.ts — turn a failure into something a person can act on.
//
// The rule this file exists for: never show a raw error. Every way a build or
// an install can fail here has a cause that is knowable and a next step that is
// concrete, and a list of twenty-seven asset filenames is neither.
//
// A step can carry an ACTION, so the UI renders a button that performs it
// rather than asking the user to go and find the control. "Use the source
// route" is a click, not an instruction.
//
// Pure: a failure description in, an explanation out.

import type { Backend } from "./types.ts";

/** Something the UI can do on the user's behalf, named so it can be a button. */
export type FixAction =
  | { kind: "switch-origin"; to: "source" | "release" }
  | { kind: "switch-backend"; to: Backend }
  | { kind: "fix-prereq"; id: string }
  | { kind: "open-tab"; tab: "dashboard" | "build" }
  | { kind: "open-url"; url: string };

export type Step = {
  text: string;
  /** Present when the app can just do it. */
  action?: FixAction;
};

export type Diagnosis = {
  /** One sentence: what went wrong, in the user's terms. */
  reason: string;
  steps: Step[];
};

export type FailureContext = {
  /** Which route was being used. */
  origin: "source" | "release";
  backend: Backend;
  platform: string;
  arch: string;
  /** Backends that DO have a prebuilt asset here, when known. */
  availableBackends?: readonly Backend[];
  /** Prerequisite ids detected as present. */
  found?: ReadonlySet<string>;
};

const hasNvcc = (c: FailureContext) => c.found?.has("cuda") === true;

/** The route that will actually work for this backend, as a step. */
function alternatives(c: FailureContext): Step[] {
  const steps: Step[] = [];
  if (c.origin === "release") {
    if (c.backend === "cuda" && c.platform === "linux") {
      steps.push({
        text: hasNvcc(c)
          ? "Build CUDA from source instead — you already have the CUDA toolkit, and llama.master picks the right GPU architecture for your card."
          : "Build CUDA from source instead. That needs the CUDA toolkit (nvcc), which the Machine tab can install.",
        action: { kind: "switch-origin", to: "source" },
      });
      if (!hasNvcc(c)) {
        steps.push({
          text: "Install the CUDA toolkit (nvcc).",
          action: { kind: "fix-prereq", id: "cuda" },
        });
      }
    }
    if (
      (c.availableBackends ?? []).includes("vulkan") && c.backend !== "vulkan"
    ) {
      steps.push({
        text:
          "Or use the prebuilt Vulkan build — it runs on NVIDIA, AMD and Intel alike and needs nothing installed.",
        action: { kind: "switch-backend", to: "vulkan" },
      });
    }
  }
  return steps;
}

/**
 * No prebuilt asset exists for this combination.
 *
 * Distinguishes the three real cases, because "not found" covers a permanent
 * upstream policy (no Linux CUDA), a platform impossibility (Metal off macOS),
 * and a transient one (a release whose assets are still uploading) — and the
 * right next step differs for each.
 */
export function diagnoseNoAsset(
  c: FailureContext,
  assetCount: number,
): Diagnosis {
  // A published release carries ~25 assets. A handful means CI is still
  // uploading them, which resolves itself in minutes.
  if (assetCount > 0 && assetCount < 6) {
    return {
      reason:
        `This release only has ${assetCount} file(s) published so far — llama.cpp's CI uploads them over several minutes after tagging, so the one you want is probably still on its way.`,
      steps: [
        { text: "Wait a few minutes and press the button again." },
        {
          text:
            "Or pick the previous release from the Version list, which is complete.",
        },
      ],
    };
  }

  if (c.backend === "metal" && c.platform !== "darwin") {
    return {
      reason:
        "Metal is Apple's GPU API and exists only on macOS — no build of it can run on this machine.",
      steps: alternatives(c).concat({
        text: "Use CPU, Vulkan or CUDA on this machine.",
        action: { kind: "switch-backend", to: "vulkan" },
      }),
    };
  }

  if (c.backend === "cuda" && c.platform === "linux") {
    return {
      reason:
        "llama.cpp publishes prebuilt CUDA binaries for Windows only. There is no Linux CUDA download — this is upstream's choice, not a problem with your machine.",
      steps: alternatives(c),
    };
  }

  const avail = c.availableBackends ?? [];
  return {
    reason:
      `No prebuilt ${c.backend.toUpperCase()} binary is published for ${c.platform}/${c.arch}.`,
    steps: [
      ...(avail.length > 0
        ? [{
          text: `Prebuilt and ready here: ${avail.join(", ")}.`,
          action: { kind: "switch-backend" as const, to: avail[0] as Backend },
        }]
        : []),
      {
        text: "Or build this backend from source.",
        action: { kind: "switch-origin", to: "source" },
      },
    ],
  };
}

/** Signatures of build failures we have actually seen, and what to do. */
const BUILD_SIGNATURES: {
  match: RegExp;
  reason: string;
  steps: (c: FailureContext) => Step[];
}[] = [
  {
    match: /Unsupported gpu architecture|compute_\d+a?'/i,
    reason:
      "Your CUDA toolkit is older than your GPU, so nvcc does not recognise the architecture cmake asked it to build for.",
    steps: () => [
      {
        text:
          "llama.master normally caps this automatically (it builds PTX the driver can JIT). Seeing it means the cap did not apply — re-run the build so the architecture is re-detected.",
      },
      {
        text: "Installing a newer CUDA toolkit removes the need for the cap.",
        action: { kind: "fix-prereq", id: "cuda" },
      },
      {
        text: "Or use Vulkan, which has no toolkit version to match.",
        action: { kind: "switch-backend", to: "vulkan" },
      },
    ],
  },
  {
    match: /'spv' has not been declared|SPIRV-Headers|spirv\/unified1/i,
    reason:
      "The Vulkan backend needs the SPIRV-Headers package, which is not on this machine's include path.",
    steps: () => [
      {
        text:
          "Install SPIRV-Headers — headers only, no root needed, llama.master downloads it.",
        action: { kind: "fix-prereq", id: "spirv" },
      },
    ],
  },
  {
    match:
      /HIP runtime CMake package|hip-lang-config\.cmake|Could NOT find hip\b/i,
    reason:
      "ROCm's runtime is installed but its development package is not — cmake needs hip-lang-config.cmake, which only the -dev packages ship. hipcc being on PATH is not enough.",
    steps: () => [
      {
        text: "Install the HIP development package.",
        action: { kind: "fix-prereq", id: "hip" },
      },
      {
        text:
          "Or use the prebuilt ROCm release, which bundles everything and needs no toolchain.",
        action: { kind: "switch-origin", to: "release" },
      },
    ],
  },
  {
    match: /Could NOT find Vulkan|glslc.*not found|COMPONENTS glslc/i,
    reason: "The Vulkan shader compiler (glslc) is missing.",
    steps: () => [
      {
        text: "Install the Vulkan shader tools.",
        action: { kind: "fix-prereq", id: "vulkan" },
      },
      {
        text: "Or use the prebuilt Vulkan release, which needs no toolchain.",
        action: { kind: "switch-origin", to: "release" },
      },
    ],
  },
  {
    match:
      /No CMAKE_CXX_COMPILER|c\+\+.*not able to compile|CMAKE_CXX_COMPILER not set/i,
    reason: "cmake cannot find a working C++ compiler.",
    steps: () => [
      {
        text: "Install a C++ toolchain.",
        action: { kind: "fix-prereq", id: "compiler" },
      },
      {
        text: "Or install a prebuilt release, which needs no compiler at all.",
        action: { kind: "switch-origin", to: "release" },
      },
    ],
  },
  {
    match: /cmake.*not found|CMake not found/i,
    reason: "CMake is missing.",
    steps: () => [
      {
        text: "llama.master can download CMake into its own directory.",
        action: { kind: "fix-prereq", id: "cmake" },
      },
    ],
  },
  {
    match:
      /Killed|out of memory|cannot allocate memory|c\+\+: fatal error: Killed/i,
    reason:
      "The compiler was killed — the machine ran out of memory. Parallel compile jobs each need up to ~2 GB, and a CUDA build needs more.",
    steps: () => [
      {
        text:
          "Lower the job count in the Compile row (try half), then build again. It is slower but it finishes.",
      },
      { text: "Closing other memory-heavy applications also helps." },
    ],
  },
  {
    match: /No space left on device|ENOSPC/i,
    reason: "The disk filled up. A llama.cpp build tree needs several GB.",
    steps: () => [
      {
        text:
          "Free space, then build again. Old build trees live in ~/.llama-master/cache and are safe to delete.",
      },
      {
        text: "Deleting builds you no longer use also frees space.",
        action: { kind: "open-tab", tab: "build" },
      },
    ],
  },
  {
    match: /rate limit/i,
    reason: "GitHub's anonymous API quota is exhausted.",
    steps: () => [
      {
        text:
          "llama.master falls back to GitHub's plain pages for this, so retrying usually works.",
      },
      {
        text:
          "Setting GITHUB_TOKEN in the environment raises the quota from 60 to 5000 requests/hour.",
        action: {
          kind: "open-url",
          url: "https://github.com/settings/tokens",
        },
      },
    ],
  },
];

/**
 * Explain a build or install failure.
 *
 * Falls back to a generic-but-useful answer rather than echoing the raw text:
 * an unrecognised failure still gets the log pointer and the other route.
 */
export function diagnoseFailure(
  message: string,
  c: FailureContext,
): Diagnosis {
  for (const sig of BUILD_SIGNATURES) {
    if (sig.match.test(message)) {
      return { reason: sig.reason, steps: sig.steps(c) };
    }
  }
  return {
    reason: c.origin === "source"
      ? "The build stopped. The last lines of the log below name the file and the error."
      : "The install did not finish. The log below has the details.",
    steps: [
      ...(c.origin === "source"
        ? [{
          text:
            "A prebuilt release needs no toolchain and takes seconds — often the fastest way past a build problem.",
          action: { kind: "switch-origin" as const, to: "release" as const },
        }]
        : [{
          text: "Building from source avoids whatever is missing upstream.",
          action: { kind: "switch-origin" as const, to: "source" as const },
        }]),
      {
        text: "Re-check the prerequisites — one of them may have changed.",
        action: { kind: "open-tab", tab: "dashboard" },
      },
    ],
  };
}
