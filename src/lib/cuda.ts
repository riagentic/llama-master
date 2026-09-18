// src/lib/cuda.ts — can this CUDA toolkit build for this GPU?
//
// The failure this exists to prevent, measured on the machine this was written
// on: an RTX PRO 4000 Blackwell (compute capability 12.0) with Ubuntu's
// `nvidia-cuda-toolkit` (CUDA 12.0). llama.cpp auto-detects the native
// architecture, hands nvcc `compute_120a`, and nvcc — which has never heard of
// Blackwell — stops with `nvcc fatal: Unsupported gpu architecture`. Four
// minutes of compiling, then a wall.
//
// Three things follow from that, and all three are decided here:
//
//   1. The mismatch is knowable UP FRONT, from two version numbers.
//   2. There is usually a way through anyway: a PTX-only build for an
//      architecture nvcc DOES know still runs on a newer GPU, because the
//      driver JIT-compiles PTX forward. Verified on the Blackwell card above —
//      a `compute_90` binary runs correctly on `sm_120`.
//   3. When there is no way through, the app should say which CUDA release
//      would fix it rather than "build failed".
//
// Pure: version numbers in, a build decision out.

/** Highest compute capability each CUDA release can emit code for.
 *
 *  Only the boundaries matter, so this is a table of "from this release, this
 *  is the newest architecture" — extend it as NVIDIA ships. Sources: the CUDA
 *  release notes' supported `--gpu-architecture` values. */
const MAX_ARCH: [number, number][] = [
  // [CUDA minor-precision version, max compute capability × 10]
  [11.0, 80],
  [11.1, 86],
  [11.4, 87],
  [11.8, 90],
  [12.0, 90],
  [12.6, 90],
  [12.8, 120],
  [13.0, 121],
];

/** Compute capability an architecture number refers to: 90 → 9.0. */
const archToCap = (arch: number): number => arch / 10;

/** `12.0.140` / `12.0` → 12.0. NaN-free: unparseable → 0. */
export function parseCudaVersion(version: string): number {
  const m = /(\d+)\.(\d+)/.exec(version);
  if (!m) return 0;
  return Number(`${m[1]}.${m[2]}`);
}

/** The newest architecture this CUDA release can target, or 0 if unknown. */
export function maxArchFor(cudaVersion: number): number {
  if (!(cudaVersion > 0)) return 0;
  let best = 0;
  for (const [ver, arch] of MAX_ARCH) {
    if (cudaVersion >= ver) best = Math.max(best, arch);
  }
  // A release newer than the table knows: assume it handles the newest entry.
  return best || (MAX_ARCH.at(-1) as [number, number])[1];
}

/** The oldest CUDA release that can target `cap`, for the "install this" hint. */
export function cudaVersionForCap(cap: number): number | null {
  const arch = Math.round(cap * 10);
  for (const [ver, max] of MAX_ARCH) {
    if (max >= arch) return ver;
  }
  return null;
}

export type CudaMode =
  /** nvcc can emit native code for every GPU here. */
  | "native"
  /** nvcc is older than the GPU; emit PTX it does know and let the driver JIT.
   *  Works, at the cost of a slower first load. */
  | "ptx"
  /** nvcc is too old even to emit usable PTX, or there is no GPU. */
  | "impossible";

export type CudaPlan = {
  mode: CudaMode;
  /** The value for `-DCMAKE_CUDA_ARCHITECTURES`. Empty = let cmake decide. */
  architectures: string;
  /** One line for the UI: what was decided and why. */
  reason: string;
  /** What the user could install to get a native build, when that applies. */
  remedy: string;
};

/**
 * Decide what to build CUDA for.
 *
 * `caps` are the compute capabilities reported by the driver (12.0 for
 * Blackwell, 8.9 for Ada, …). `nvccVersion` is whatever `nvcc --version` said.
 */
export function cudaPlan(
  nvccVersion: string,
  caps: readonly number[],
): CudaPlan {
  const cuda = parseCudaVersion(nvccVersion);
  const max = maxArchFor(cuda);

  if (caps.length === 0) {
    return {
      mode: "impossible",
      architectures: "",
      reason: "No NVIDIA GPU was detected, so there is nothing to target.",
      remedy: "Use the CPU or Vulkan backend.",
    };
  }
  if (!(cuda > 0) || max === 0) {
    return {
      mode: "impossible",
      architectures: "",
      reason: `Could not read a CUDA version from "${nvccVersion}".`,
      remedy: "Install the CUDA toolkit, or use the Vulkan backend.",
    };
  }

  const maxCap = archToCap(max);
  const tooNew = caps.filter((c) => c > maxCap);

  if (tooNew.length === 0) {
    // Everything is in range: name the architectures explicitly rather than
    // leaving it to auto-detection, so the build is reproducible.
    const archs = [...new Set(caps.map((c) => Math.round(c * 10)))]
      .sort((a, b) => a - b)
      .join(";");
    return {
      mode: "native",
      architectures: archs,
      reason: `CUDA ${cuda} builds native code for ${
        caps.map((c) => `sm_${Math.round(c * 10)}`).join(", ")
      }.`,
      remedy: "",
    };
  }

  const needed = cudaVersionForCap(Math.max(...tooNew));
  const newest = Math.max(...tooNew);
  return {
    mode: "ptx",
    // PTX only ("-virtual"): the driver JIT-compiles it for the real GPU.
    architectures: `${max}-virtual`,
    reason: `CUDA ${cuda} cannot emit code for sm_${
      Math.round(newest * 10)
    } (its newest is sm_${max}). Building PTX for sm_${max} instead — the driver compiles it for your GPU on first load, which works but makes that first load slower.`,
    remedy: needed
      ? `Install CUDA ${needed} or newer for a native build.`
      : "A newer CUDA release than any known here would be needed for a native build.",
  };
}

/**
 * Which CUDA release to INSTALL so every one of these cards gets native code,
 * or "" when the toolkit already covers them.
 *
 * `cudaPlan` decides what to do with the toolkit that is here; this decides
 * whether a better one is worth fetching, which is a different question and
 * the one the prerequisites panel asks. Deliberately the OLDEST release that
 * covers the newest card rather than the newest release that exists: a bigger
 * jump is a bigger download and a bigger change for no gain, and NVIDIA's own
 * table is what says which is enough.
 *
 * The DRIVER is not consulted here and does not need to be — a driver runs any
 * toolkit up to its own CUDA version, and the caller checks that separately
 * (`driverCudaVersion`). Measured on the machine this was written for: driver
 * 610.43 reported CUDA 13.3 while nvcc was 12.0, so the toolkit was the only
 * thing between it and native Blackwell code.
 */
export function cudaUpgradeFor(
  nvccVersion: string,
  caps: readonly number[],
): { need: number; have: number; newestCap: number } | null {
  if (caps.length === 0) return null;
  const have = parseCudaVersion(nvccVersion);
  const max = maxArchFor(have);
  // An unreadable version is a different problem — "the toolkit is missing" —
  // and the existing `cuda` prerequisite already covers it.
  if (!(have > 0) || max === 0) return null;
  const newestCap = Math.max(...caps);
  if (newestCap <= archToCap(max)) return null;
  const need = cudaVersionForCap(newestCap);
  return need === null ? null : { need, have, newestCap };
}

/**
 * The CUDA version the installed DRIVER can run, from `nvidia-smi`.
 *
 * The one hard ceiling on this whole idea: a toolkit newer than the driver
 * produces binaries the machine cannot load. Everything else about installing
 * a toolkit into a private directory is safe and reversible; this is the check
 * that keeps it that way, because the app must never propose a fix whose
 * outcome is "now nothing runs".
 *
 * `nvidia-smi` prints it in the header ("CUDA Version: 13.3" on older drivers,
 * "CUDA UMD Version: 13.3" on newer ones). 0 when it cannot be read, which the
 * caller treats as "do not offer an upgrade" rather than as "any version will
 * do".
 */
export function driverCudaVersion(smiText: string): number {
  // `\s*:?\s*`: the banner writes "CUDA UMD Version: 13.3", while
  // `nvidia-smi --version` pads the label ("CUDA UMD version    : 13.3") and
  // prints a "CUDA version : Deprecated" line first, which must not match.
  const m = /CUDA(?:\s+UMD)?\s+Version\s*:?\s*([0-9]+\.[0-9]+)/i.exec(smiText);
  return m ? Number(m[1]) : 0;
}

/**
 * The best toolkit to offer: new enough for the cards, not newer than the
 * driver can run, and "" when there is nothing to gain or nothing safe to do.
 */
export function cudaOffer(
  nvccVersion: string,
  caps: readonly number[],
  driverCuda: number,
): { need: number; have: number; newestCap: number } | null {
  const up = cudaUpgradeFor(nvccVersion, caps);
  if (!up) return null;
  // A driver that cannot run what the cards need is not something a toolkit
  // download can fix, and pretending otherwise would install two gigabytes to
  // no effect. Silence here is the honest answer; the build still falls back
  // to PTX and says why.
  if (!(driverCuda > 0) || driverCuda < up.need) return null;
  return up;
}

/** `-DCMAKE_CUDA_ARCHITECTURES=…` for a plan, or nothing when cmake should
 *  decide for itself. */
export function cudaCmakeFlags(plan: CudaPlan): string[] {
  return plan.architectures
    ? [`-DCMAKE_CUDA_ARCHITECTURES=${plan.architectures}`]
    : [];
}
