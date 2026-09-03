// src/lib/cudaredist.ts — which pieces of CUDA to fetch, and why each one.
//
// `cuda.ts` answers "can this toolkit build for this GPU". When the answer is
// no, the app used to stop at a sentence: install a newer CUDA yourself. On the
// machine this was written for that sentence is the difference between 19 and
// (potentially) far more tokens a second, because everything CUDA 12.0 can emit
// for a Blackwell card is PTX that the driver re-compiles at load.
//
// So the app installs it — and the whole design here is about doing that
// SAFELY, because a CUDA install is the most dangerous thing a tool can offer
// to do to a working Linux desktop. The usual routes are both hazardous:
//
//   - `apt install cuda` adds NVIDIA's apt repository and pulls a new DRIVER.
//     A driver swap under a running X/Wayland session is how a machine ends up
//     at a black screen, and it can take unrelated packages with it.
//   - the .run installer defaults to writing across /usr/local and can install
//     a driver too.
//
// Neither is acceptable for a thing behind a "Fix" button. What IS acceptable
// is what NVIDIA publishes for exactly this purpose: the toolkit as ordinary
// component tarballs, each with a SHA-256 in a signed-over-HTTPS manifest.
// Unpacking those into a directory this app owns:
//
//   - needs no root, and touches nothing outside `~/.llama-master`
//   - installs no driver, no kernel module, no udev rule, no apt repository
//   - changes no system package, no PATH, no ld.so config
//   - is undone completely by deleting one directory
//
// The one thing it CANNOT do is make the driver newer, and it does not need
// to: a driver runs any toolkit up to its own CUDA version, so the check that
// matters is toolkit ≤ driver, not the other way round. On the machine that
// prompted this, the driver already reported CUDA 13.3 while nvcc was 12.0 —
// the toolkit was the only thing standing between it and native Blackwell code.
//
// Pure: a manifest in, a list of files and a size out.

/** Where NVIDIA publishes the component manifests. */
const REDIST = "https://developer.download.nvidia.com/compute/cuda/redist";

/** The manifest for one CUDA release. */
export function manifestUrl(version: string): string {
  return `${REDIST}/redistrib_${version}.json`;
}

/** Absolute URL of one component archive. */
export function componentUrl(relativePath: string): string {
  return `${REDIST}/${relativePath}`;
}

/**
 * The components llama.cpp's CUDA backend actually needs, and why.
 *
 * Deliberately a SHORT list. The full toolkit is tens of gigabytes and most of
 * it is for things this app never does — profilers, debuggers, image and
 * signal libraries, Visual Studio integration. Every entry below is here
 * because something in a llama.cpp CUDA build reaches for it, and the comment
 * says what. Read off `ggml/src/ggml-cuda/CMakeLists.txt`, which links
 * `CUDA::cudart`, `CUDA::cublas` (and the `_static` forms) and `CUDA::cuda_driver`.
 */
export const CUDA_COMPONENTS: ReadonlyArray<{ key: string; why: string }> = [
  { key: "cuda_nvcc", why: "the compiler itself" },
  { key: "cuda_crt", why: "headers nvcc includes on every compile" },
  { key: "libnvvm", why: "nvcc's code generator" },
  { key: "libnvptxcompiler", why: "nvcc's PTX assembler" },
  { key: "cuda_cudart", why: "the CUDA runtime and its headers" },
  { key: "libcublas", why: "the matrix multiply llama.cpp calls for prefill" },
  { key: "cuda_culibos", why: "needed when cuBLAS is linked statically" },
  {
    key: "cccl",
    why: "CUB and Thrust headers — ggml-cuda includes cub/cub.cuh",
  },
  { key: "libnvjitlink", why: "cuBLAS links its kernels through this" },
  { key: "cuda_profiler_api", why: "a header ggml includes" },
  { key: "libnvfatbin", why: "reads the fat binaries nvcc emits" },
  { key: "cuda_cuobjdump", why: "nvcc calls it while linking device code" },
  { key: "cuda_nvdisasm", why: "the same, for the disassembly step" },
  { key: "cuda_nvtx", why: "range annotations ggml compiles against" },
];

/** One file to fetch: where it is, what it must hash to, how big it is. */
export type CudaFile = {
  key: string;
  why: string;
  url: string;
  /** SHA-256 from NVIDIA's manifest. Every download is checked against it. */
  sha256: string;
  sizeB: number;
  name: string;
};

type Platformed = { relative_path?: string; sha256?: string; size?: string };
type ManifestEntry = Record<string, Platformed | undefined>;

/**
 * Turn NVIDIA's manifest into the list of files to fetch.
 *
 * A component the manifest does not carry is SKIPPED rather than fatal: the
 * set above spans several CUDA releases and NVIDIA renames things between
 * them (`cuda_cccl` became `cccl`). A missing optional header is a build that
 * still works; a hard failure here would be the app refusing to install
 * because of a name change in a release it was never asked about.
 *
 * What is NOT optional is the compiler. Without `cuda_nvcc` there is nothing
 * to install and the caller is told so rather than handed a partial toolkit.
 */
export function cudaFiles(
  manifest: unknown,
  platform = "linux-x86_64",
): { files: CudaFile[]; totalB: number; missing: string[] } {
  const m = (manifest ?? {}) as Record<string, ManifestEntry>;
  const files: CudaFile[] = [];
  const missing: string[] = [];
  for (const { key, why } of CUDA_COMPONENTS) {
    const component = m[key];
    const entry = typeof component === "object" && component !== null
      ? component[platform]
      : undefined;
    const path = entry?.relative_path;
    const sha = entry?.sha256;
    if (!path || !sha) {
      missing.push(key);
      continue;
    }
    const sizeB = Number(entry?.size ?? 0);
    files.push({
      key,
      why,
      url: componentUrl(path),
      sha256: sha,
      sizeB: Number.isFinite(sizeB) && sizeB > 0 ? sizeB : 0,
      name: path.split("/").pop() ?? key,
    });
  }
  return {
    files,
    totalB: files.reduce((a, f) => a + f.sizeB, 0),
    missing,
  };
}

/** Is this set usable at all? Only the compiler is truly required. */
export function cudaFilesUsable(files: readonly CudaFile[]): boolean {
  return files.some((f) => f.key === "cuda_nvcc");
}

/**
 * Room needed on disk, including the unpacking.
 *
 * The archives are xz and the toolkit lands about twice their size (measured:
 * 929 MB of downloads became 2.0 GB installed), and both exist at once while
 * unpacking. Three times the download, rounded up, with a gigabyte of slack —
 * running a disk to zero is a worse outcome than refusing politely, especially
 * on the machine this was written for, which sits at 99% full.
 */
export function cudaDiskNeededB(totalDownloadB: number): number {
  return Math.round(totalDownloadB * 3) + 1024 ** 3;
}

/** The directory name for an installed toolkit. */
export function cudaDirName(version: string): string {
  return `cuda-${version.replace(/[^0-9A-Za-z._-]/g, "_")}`;
}

/**
 * What this fix is about to do, in the user's terms, before it does it.
 *
 * The app's rule is that nothing privileged runs unexplained; this runs
 * nothing privileged at all, and saying THAT is the point. A person who has
 * ever broken a machine installing CUDA needs to know which kind of install
 * this is before they press anything.
 */
export function cudaFixSummary(
  version: string,
  totalB: number,
  root: string,
): string[] {
  const gb = (totalB / 1024 ** 3).toFixed(1);
  return [
    `Download CUDA ${version} from NVIDIA (${gb} GB) and unpack it into ${root}.`,
    "No administrator rights are used and nothing outside that folder changes.",
    "No driver, no kernel module and no system package is installed or altered — your graphics driver is left exactly as it is.",
    "Every file is checked against the SHA-256 NVIDIA publishes for it before it is unpacked.",
    "To undo it, delete that folder. Nothing else will have moved.",
  ];
}
