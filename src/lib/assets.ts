// src/lib/assets.ts — choose the right prebuilt llama.cpp asset.
//
// Upstream asset names are conventional, not contractual
// (`llama-b6234-bin-ubuntu-vulkan-x64.zip`, `llama-b6234-bin-macos-arm64.zip`,
// `llama-b6234-bin-win-cuda-12.4-x64.zip`), and they have changed shape more
// than once. So this scores candidates instead of pattern-matching a fixed
// template, and the UI always shows the full list with the auto-pick
// highlighted — when the heuristic is wrong the user is one click from right.
//
// Pure: name strings in, a choice out. Every rule below has a test.

import type { Backend } from "./types.ts";

export type Asset = {
  name: string;
  url: string;
  sizeB: number;
};

const PLATFORM_TOKENS: Record<string, string[]> = {
  linux: ["ubuntu", "linux"],
  darwin: ["macos", "osx"],
  windows: ["win"],
};

const ARCH_TOKENS: Record<string, string[]> = {
  x86_64: ["x64", "x86_64", "amd64"],
  aarch64: ["arm64", "aarch64"],
};

/** Backend markers that appear in asset names. `cpu` is the absence of all. */
const BACKEND_TOKENS: Record<Backend, string[]> = {
  cuda: ["cuda"],
  vulkan: ["vulkan"],
  hip: ["hip", "rocm"],
  metal: ["macos", "metal"],
  cpu: [],
};

/** Accelerator markers we must NOT accept when the user asked for plain CPU.
 *  Verified against a real llama.cpp release manifest (b10144), which ships
 *  openvino and sycl variants alongside the plain build. */
const ALL_ACCEL = [
  "cuda",
  "vulkan",
  "hip",
  "rocm",
  "sycl",
  "musa",
  "cann",
  "opencl",
  "openvino",
  "hexagon",
];

/** Architectures that are neither of the two we know how to name. A build for
 *  one of these carries no x64/arm64 token, so without this list it would look
 *  like an architecture-neutral asset and be treated as usable. */
const FOREIGN_ARCH = ["s390x", "ppc64", "riscv", "loongarch", "android"];

function has(name: string, token: string): boolean {
  return name.includes(token);
}

/** Is this a binary bundle at all (rather than source, debug info, or a hash)? */
export function isBinaryAsset(name: string): boolean {
  const n = name.toLowerCase();
  if (!n.endsWith(".zip") && !n.endsWith(".tar.gz")) return false;
  if (n.includes("source")) return false;
  if (n.includes("dsym") || n.includes("-debug")) return false;
  // `cudart-llama-bin-*` is NVIDIA's redistributable runtime, not llama.cpp —
  // it contains no llama-server and would install a directory of DLLs.
  if (n.startsWith("cudart")) return false;
  // `*-xcframework.zip` / `*-ui.tar.gz` are libraries and web assets.
  if (n.includes("xcframework") || n.endsWith("-ui.tar.gz")) return false;
  return n.includes("-bin-");
}

/**
 * Score an asset for (platform, arch, backend). Higher is better; `null` means
 * "not usable here" — a wrong-platform binary must never be a fallback.
 */
export function scoreAsset(
  name: string,
  platform: string,
  arch: string,
  backend: Backend,
): number | null {
  const n = name.toLowerCase();
  if (!isBinaryAsset(n)) return null;

  // Metal exists only on Apple hardware. Without this, the plain Linux build
  // would score as a valid "metal" asset, because a macOS binary carries no
  // accelerator token to distinguish it from a CPU one.
  if (backend === "metal" && platform !== "darwin") return null;

  const plat = PLATFORM_TOKENS[platform] ?? [platform];
  if (!plat.some((t) => has(n, t))) return null;
  if (FOREIGN_ARCH.some((t) => has(n, t))) return null;

  const archTokens = ARCH_TOKENS[arch] ?? [arch];
  const otherArch = Object.entries(ARCH_TOKENS)
    .filter(([a]) => a !== arch)
    .flatMap(([, t]) => t);
  const archHit = archTokens.some((t) => has(n, t));
  // A name carrying the *other* architecture is disqualifying; a name carrying
  // none (macOS universal builds) is acceptable but scores lower.
  if (!archHit && otherArch.some((t) => has(n, t))) return null;

  const accel = ALL_ACCEL.filter((t) => has(n, t));
  if (backend === "cpu" || backend === "metal") {
    if (accel.length > 0) return null;
  } else {
    const want = BACKEND_TOKENS[backend];
    if (!want.some((t) => has(n, t))) return null;
  }

  let score = 100;
  if (archHit) score += 20;
  // Prefer the shortest name among equals: extra tokens mean extra specificity
  // (a CUDA version, a distro release) that we did not ask for.
  score -= Math.min(30, Math.floor(n.length / 4));
  return score;
}

/** The asset to download by default, or null when nothing here fits. */
export function pickAsset(
  assets: readonly Asset[],
  platform: string,
  arch: string,
  backend: Backend,
): Asset | null {
  let best: { a: Asset; s: number } | null = null;
  for (const a of assets) {
    const s = scoreAsset(a.name, platform, arch, backend);
    if (s === null) continue;
    if (!best || s > best.s) best = { a, s };
  }
  return best?.a ?? null;
}

const BACKENDS: readonly Backend[] = ["cpu", "cuda", "vulkan", "hip", "metal"];

/** Assets usable on this machine at all, for the "or choose another" list. */
export function usableAssets(
  assets: readonly Asset[],
  platform: string,
  arch: string,
): Asset[] {
  return assets.filter((a) =>
    BACKENDS.some((b) => scoreAsset(a.name, platform, arch, b) !== null)
  );
}

/**
 * Why there is no prebuilt binary for this combination, and what to do instead.
 *
 * The kata's rule: when something cannot be fixed automatically, explain it
 * precisely and give the steps. "No asset found" is not an explanation — these
 * are the three real reasons, each with the route that does work.
 */
export function noAssetExplanation(
  backend: Backend,
  platform: string,
  arch: string,
  available: readonly Backend[],
): { reason: string; steps: string[] } {
  if (backend === "metal" && platform !== "darwin") {
    return {
      reason:
        "Metal is Apple's GPU API and exists only on macOS, so no build of it can run on this machine.",
      steps: [
        platform === "linux"
          ? "Use Vulkan (works on NVIDIA, AMD and Intel) or CUDA on this machine."
          : "Use CPU or Vulkan on this machine.",
      ],
    };
  }
  if (backend === "cuda" && platform === "linux") {
    return {
      reason:
        "llama.cpp publishes prebuilt CUDA binaries for Windows only — there is no Linux CUDA release to download, upstream.",
      steps: [
        'Switch the route to "Build from source" and pick CUDA — that works on Linux and llama.master will handle the CUDA architecture for your GPU.',
        "Or use the prebuilt Vulkan release, which runs on NVIDIA cards too and needs no toolchain.",
      ],
    };
  }
  if (backend === "hip" && platform !== "linux") {
    return {
      reason: "ROCm binaries are published for Linux and Windows only.",
      steps: ["Use CPU, Vulkan, or Metal on this platform."],
    };
  }
  return {
    reason:
      `No prebuilt ${backend.toUpperCase()} binary is published for ${platform}/${arch} in this release.`,
    steps: available.length > 0
      ? [
        `Prebuilt here: ${available.join(", ")}.`,
        "Or build this backend from source.",
      ]
      : ["Build from source instead — no prebuilt binary here fits."],
  };
}

/**
 * Backends that actually have a prebuilt asset here.
 *
 * This matters more than it looks: upstream ships CUDA binaries for Windows
 * only, so on Linux + NVIDIA the obvious choice silently has no download. The
 * UI uses this to say so before the user presses the button.
 */
export function availableBackends(
  assets: readonly Asset[],
  platform: string,
  arch: string,
): Backend[] {
  return BACKENDS.filter((b) => pickAsset(assets, platform, arch, b) !== null);
}
