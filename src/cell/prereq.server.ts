// src/cell/prereq.server.ts — detect (and where possible, obtain) the tools a
// llama.cpp build needs. SERVER ONLY.
//
// The kata's promise is "nothing but a running OS". Two of the four tools can
// be honoured literally — CMake is a self-contained tarball Kitware publishes,
// and llama.cpp source is a tarball GitHub publishes, so neither cmake nor git
// has to pre-exist. A C++ compiler cannot be conjured, so when one is missing
// the app says so plainly and points at the prebuilt-release path, which needs
// no toolchain at all.

import type { Prereq } from "../lib/types.ts";
import type { Distro, FixPlan, PackageManager } from "../lib/fixplan.ts";
import { elevate, fixPlan } from "../lib/fixplan.ts";
import {
  ARCH,
  download,
  ensureDir,
  exec,
  exists,
  extract,
  fetchJson,
  fetchText,
  makeExecutable,
  paths,
  PLATFORM,
  which,
} from "./host.server.ts";
import { cudaOffer, driverCudaVersion, parseCudaVersion } from "../lib/cuda.ts";
import {
  cudaDirName,
  cudaDiskNeededB,
  cudaFiles,
  cudaFilesUsable,
  cudaFixSummary,
  manifestUrl,
} from "../lib/cudaredist.ts";
import { join } from "@std/path";
import { disks, nvidiaSmi } from "./hw.server.ts";
import { quote } from "../lib/command.ts";

/** First line of `--version` output, trimmed — every tool here prints one. */
function firstLine(s: string): string {
  return s.split("\n")[0]?.trim() ?? "";
}

async function probe(
  bin: string,
  args: string[] = ["--version"],
): Promise<{ path: string; version: string } | null> {
  const path = await which(bin);
  if (!path) return null;
  // A ceiling, because a wedged driver tool (nvidia-smi after a suspend is
  // the known one) would otherwise hang the whole prerequisite scan.
  const r = await exec(path, args, { timeoutMs: 10_000 });
  if (r.code !== 0 && !r.stdout && !r.stderr) return null;
  return { path, version: firstLine(r.stdout || r.stderr) };
}

/** The CMake this app will actually use: its own download wins over PATH, so a
 *  system cmake that is too old cannot silently break a build. */
export async function resolveCmake(): Promise<
  { path: string; version: string; managed: boolean } | null
> {
  const managed = await managedCmakePath();
  if (managed) {
    const r = await exec(managed, ["--version"]);
    if (r.code === 0) {
      return { path: managed, version: firstLine(r.stdout), managed: true };
    }
  }
  const p = await probe("cmake");
  return p ? { ...p, managed: false } : null;
}

async function managedCmakePath(): Promise<string | null> {
  const base = join(paths().toolchain, "cmake");
  const candidates = [
    join(base, "bin", PLATFORM === "windows" ? "cmake.exe" : "cmake"),
    // macOS ships CMake inside an app bundle.
    join(base, "CMake.app", "Contents", "bin", "cmake"),
  ];
  for (const c of candidates) if (await exists(c)) return c;
  return null;
}

/** Where the app keeps a CUDA toolkit it installed itself. */
export function cudaPrefix(version: string): string {
  return join(paths().toolchain, cudaDirName(version));
}

/**
 * The newest CUDA toolkit this app has installed, if any.
 *
 * Directories only, and never a `.partial` staging directory — an install is
 * renamed to its final name only when every file is in, so anything named
 * `cuda-<version>` with an `nvcc` in it is complete.
 */
export async function managedCuda(): Promise<
  { path: string; nvcc: string; version: string } | null
> {
  let best: { path: string; nvcc: string; version: string } | null = null;
  try {
    for await (const e of Deno.readDir(paths().toolchain)) {
      // `cuda-<v>.partial` is an install in progress (or one a crash cut
      // short): it has an `nvcc` long before it has the rest of the toolkit.
      if (!e.isDirectory || !e.name.startsWith("cuda-")) continue;
      if (e.name.endsWith(".partial")) continue;
      const path = join(paths().toolchain, e.name);
      const nvcc = join(path, "bin", "nvcc");
      if (!(await exists(nvcc))) continue;
      const version = e.name.slice("cuda-".length);
      if (!best || parseCudaVersion(version) > parseCudaVersion(best.version)) {
        best = { path, nvcc, version };
      }
    }
  } catch {
    // No toolchain directory yet.
  }
  return best;
}

/**
 * Install a CUDA toolkit into a directory this app owns.
 *
 * The safe half of the "your CUDA is too old for your GPU" fix, and every
 * choice here is about that word (`src/lib/cudaredist.ts` has the long form):
 * component tarballs from NVIDIA's redistributable manifest, each verified
 * against the SHA-256 published with it, unpacked under `~/.llama-master`. No
 * root, no apt repository, no driver, no kernel module, no system package —
 * and undone completely by deleting one directory.
 *
 * Disk is checked first, because the machine this was written for sits at 99%
 * full and running it to zero would be a far worse outcome than a refusal.
 */
export async function installCudaToolkit(
  version: string,
  onProgress: (received: number, total: number | null, note: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  onProgress(0, null, `Reading NVIDIA's file list for CUDA ${version}`);
  const manifest = await fetchJson<unknown>(manifestUrl(version));
  const { files, totalB, missing } = cudaFiles(manifest);
  if (!cudaFilesUsable(files)) {
    throw new Error(
      `NVIDIA's manifest for CUDA ${version} does not list a compiler for this platform, so there is nothing to install.`,
    );
  }
  if (missing.length > 0) {
    onProgress(0, null, `Not in this release, skipped: ${missing.join(", ")}`);
  }

  const tar = await which("tar");
  if (!tar) {
    throw new Error(
      "Unpacking NVIDIA's CUDA archives needs `tar`, which is not installed. Every Linux and macOS ships it; install it and try again.",
    );
  }

  const need = cudaDiskNeededB(totalB);
  // The same `df -kP` reading the rest of the app uses, so "not enough room"
  // means here what it means on the Storage page.
  await ensureDir(paths().toolchain);
  const free = (await disks([paths().toolchain]))[0]?.availB ?? 0;
  if (free > 0 && free < need) {
    throw new Error(
      `Not enough room: CUDA ${version} needs about ${
        (need / 1024 ** 3).toFixed(1)
      } GB free while unpacking and there is ${
        (free / 1024 ** 3).toFixed(1)
      } GB. Free some space and try again — nothing has been downloaded.`,
    );
  }

  // Into a temporary directory first, renamed only once every file is in.
  // A half-unpacked toolkit that looks installed is worse than none: the next
  // build would pick it up and fail deep inside cmake.
  const prefix = cudaPrefix(version);
  const staging = `${prefix}.partial`;
  await Deno.remove(staging, { recursive: true }).catch(() => {});
  await ensureDir(staging);
  try {
    let done = 0;
    for (const f of files) {
      if (signal?.aborted) throw new Error("cancelled");
      const note = `Downloading ${f.key} — ${f.why}`;
      const bytes = await download(
        f.url,
        (r, t) =>
          onProgress(
            done + r,
            totalB || (t ? done + t : null),
            note,
          ),
        signal,
      );
      // Checked BEFORE anything is written. NVIDIA publishes the hash beside
      // the file; a mismatch means a corrupted download or a tampered mirror,
      // and either way it must not reach the disk.
      const got = await sha256Hex(bytes);
      if (got !== f.sha256) {
        throw new Error(
          `${f.name} did not match the checksum NVIDIA published for it. Nothing was installed.`,
        );
      }
      onProgress(done, totalB, `Unpacking ${f.key}`);
      // System `tar`, not the app's own extractor: these archives are xz and
      // Deno's DecompressionStream has gzip and deflate only. `tar` reads xz
      // everywhere this app runs, and the alternative is shipping an xz
      // decoder to save a dependency that is already present.
      //
      // Every archive wraps its contents in one versioned directory
      // (`cuda_nvcc-linux-x86_64-13.3.73-archive/`), so it is stripped and the
      // components merge into a single toolkit layout.
      const tmp = join(staging, `.${f.key}.tar.xz`);
      await Deno.writeFile(tmp, bytes);
      const un = await exec(tar, [
        "xf",
        tmp,
        "-C",
        staging,
        "--strip-components=1",
      ]);
      await Deno.remove(tmp).catch(() => {});
      if (un.code !== 0) {
        throw new Error(
          `Could not unpack ${f.name}: ${un.stderr.trim().slice(0, 200)}`,
        );
      }
      done += f.sizeB || bytes.length;
    }
    await Deno.remove(prefix, { recursive: true }).catch(() => {});
    await Deno.rename(staging, prefix);
  } catch (e) {
    await Deno.remove(staging, { recursive: true }).catch(() => {});
    throw e;
  }

  const nvcc = join(prefix, "bin", "nvcc");
  if (!(await exists(nvcc))) {
    await Deno.remove(prefix, { recursive: true }).catch(() => {});
    throw new Error(
      "The download finished but no compiler appeared, so it was removed rather than left half-installed.",
    );
  }
  await makeExecutable(nvcc);
  return nvcc;
}

/**
 * The newest published release of a CUDA major.minor line, e.g. 13.0 → 13.3.1.
 *
 * NVIDIA's redistributable index is a directory listing of
 * `redistrib_<version>.json`, so the available releases can be read without an
 * API or a key. Patch releases of the same line are the same compiler
 * generation with fixes, so taking the newest of the line the machine NEEDS is
 * strictly better than taking the oldest — while still not jumping to a major
 * version nobody asked for.
 */
export async function newestRedistFor(line: number): Promise<string | null> {
  const prefix = `${line}`.includes(".") ? `${line}.` : `${line}.0.`;
  try {
    const html = await fetchText(
      "https://developer.download.nvidia.com/compute/cuda/redist/",
    );
    const all = [...html.matchAll(/redistrib_(\d+\.\d+\.\d+)\.json/g)]
      .map((m) => m[1] as string)
      .filter((v) => v.startsWith(prefix));
    if (all.length === 0) return null;
    const key = (v: string) =>
      v.split(".").map((n) => Number(n).toString().padStart(4, "0")).join(".");
    all.sort((a, b) => key(a) < key(b) ? 1 : -1);
    return all[0] ?? null;
  } catch {
    return null;
  }
}

/** SHA-256 of a buffer, as lowercase hex. */
async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest(
    "SHA-256",
    bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer,
  );
  return Array.from(new Uint8Array(d))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Where the app keeps its own SPIRV-Headers install. */
export function spirvPrefix(): string {
  return join(paths().toolchain, "spirv-headers");
}

/** SPIRV-Headers, ours or the system's.
 *
 *  A pure-headers CMake package, and the real reason a Vulkan source build
 *  fails on a machine that has glslc: `find_package(SPIRV-Headers CONFIG
 *  REQUIRED)` at ggml-vulkan/CMakeLists.txt:14. cmake's "missing components:
 *  glslangValidator" line on the same run is informational — llama.cpp asks for
 *  glslc only — and following it leads to installing the wrong package. */
export async function resolveSpirvHeaders(): Promise<
  { path: string; version: string; managed: boolean } | null
> {
  const own = join(
    spirvPrefix(),
    "share",
    "cmake",
    "SPIRV-Headers",
    "SPIRV-HeadersConfig.cmake",
  );
  if (await exists(own)) {
    return { path: spirvPrefix(), version: "app-managed", managed: true };
  }
  for (
    const dir of [
      "/usr/share/cmake/SPIRV-Headers",
      "/usr/lib/cmake/SPIRV-Headers",
      "/usr/local/share/cmake/SPIRV-Headers",
      "/usr/lib/x86_64-linux-gnu/cmake/SPIRV-Headers",
    ]
  ) {
    if (await exists(join(dir, "SPIRV-HeadersConfig.cmake"))) {
      return { path: dir, version: "system", managed: false };
    }
  }
  return null;
}

/** Download and install SPIRV-Headers into the app's toolchain directory.
 *
 *  No root, no package manager: it is a header tree plus a cmake config, and
 *  the app already guarantees a cmake to install it with. */
export async function installSpirvHeaders(
  onProgress: (received: number, total: number | null, note: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const cmake = await resolveCmake();
  if (!cmake) {
    throw new Error(
      "CMake is needed to install SPIRV-Headers — install CMake first (the app can download it).",
    );
  }

  const url =
    "https://codeload.github.com/KhronosGroup/SPIRV-Headers/tar.gz/refs/heads/main";
  onProgress(0, null, "Downloading SPIRV-Headers");
  const bytes = await download(
    url,
    (r, t) => onProgress(r, t, "Downloading SPIRV-Headers"),
    signal,
  );

  const src = join(paths().sources, "spirv-headers");
  await Deno.remove(src, { recursive: true }).catch(() => {});
  await ensureDir(src);
  onProgress(0, null, "Extracting");
  await extract(bytes, src, "tar.gz");

  const prefix = spirvPrefix();
  await Deno.remove(prefix, { recursive: true }).catch(() => {});
  onProgress(0, null, "Installing headers");
  const build = join(src, "build");
  const cfg = await exec(cmake.path, [
    "-S",
    src,
    "-B",
    build,
    `-DCMAKE_INSTALL_PREFIX=${prefix}`,
    "-DCMAKE_BUILD_TYPE=Release",
  ]);
  if (cfg.code !== 0) {
    throw new Error(`SPIRV-Headers configure failed: ${cfg.stderr.trim()}`);
  }
  const ins = await exec(cmake.path, ["--install", build]);
  if (ins.code !== 0) {
    throw new Error(`SPIRV-Headers install failed: ${ins.stderr.trim()}`);
  }

  const found = await resolveSpirvHeaders();
  if (!found?.managed) {
    throw new Error(
      `SPIRV-Headers installed to ${prefix} but its cmake config was not found`,
    );
  }
  onProgress(1, 1, `Installed to ${prefix}`);
  return prefix;
}

/** ROCm's HIP **development** package, which is what a build actually needs.
 *
 *  `hipcc` on PATH is not enough and saying otherwise is the bug this exists to
 *  fix: a runtime-only ROCm install has hipcc, hipconfig and rocminfo, and then
 *  cmake stops with "does not contain the HIP runtime CMake package, expected
 *  at .../cmake/hip-lang/hip-lang-config.cmake". The file is the only honest
 *  test — ROCm 7.x moved its layout (/opt/rocm/core-7.14), so a hard-coded path
 *  would go stale. */
export async function resolveHipDev(): Promise<
  { path: string; version: string } | null
> {
  const roots = new Set<string>(["/opt/rocm", "/usr"]);
  const cfg = await exec("hipconfig", ["--rocmpath"]);
  if (cfg.code === 0 && cfg.stdout.trim()) roots.add(cfg.stdout.trim());
  const env = Deno.env.get("ROCM_PATH");
  if (env) roots.add(env);

  for (const root of roots) {
    for (const libdir of ["lib", "lib64", "lib/x86_64-unknown-linux-gnu"]) {
      const p = join(
        root,
        libdir,
        "cmake",
        "hip-lang",
        "hip-lang-config.cmake",
      );
      if (await exists(p)) {
        return { path: join(root, libdir, "cmake"), version: "hip-lang" };
      }
    }
  }
  return null;
}

/** The C++ compiler cmake would pick, and its version. */
export async function resolveCompiler(): Promise<
  { path: string; version: string } | null
> {
  for (const bin of ["c++", "g++", "clang++"]) {
    const p = await probe(bin);
    if (p) return p;
  }
  if (PLATFORM === "windows") {
    const cl = await probe("cl", []);
    if (cl) return cl;
  }
  return null;
}

/**
 * Whether the CUDA toolkit here can build NATIVE code for the cards here.
 *
 * A prerequisite in its own right, separate from "is nvcc installed", because
 * the failure is different and so is the fix: nvcc is present, cmake succeeds,
 * the build finishes, and the binary is slower than it should be for ever.
 * That is the shape of problem this app exists to surface before it costs
 * somebody an afternoon.
 *
 * `found: true` means "nothing to do here" — either the toolkit already covers
 * every card, or there is no NVIDIA card, or the DRIVER is too old to run a
 * toolkit that would help. The last one matters: installing a toolkit newer
 * than the driver produces binaries the machine cannot load, so when the
 * driver is the limit the app says nothing rather than offering a fix that
 * makes things worse (`cudaOffer`).
 */
async function resolveCudaFit(
  nvcc: { path: string; version: string } | null,
): Promise<
  | { ok: true; note: string; managed: boolean; path: string; version: string }
  | { ok: false; need: number; have: number; newestCap: number }
> {
  const none = {
    ok: true as const,
    note: "",
    managed: false,
    path: "",
    version: "",
  };
  // A toolkit this app installed wins: it is the one the build will use.
  const own = await managedCuda();
  const version = own?.version ?? nvcc?.version ?? "";
  if (!version) return none;

  const smi = await nvidiaSmi([]);
  if (smi.code !== 0) return none; // no NVIDIA card: nothing to fit
  const capsOut = await nvidiaSmi([
    "--query-gpu=compute_cap",
    "--format=csv,noheader",
  ]);
  const caps = capsOut.code === 0
    ? capsOut.stdout.split("\n").map((l) => Number(l.trim())).filter((n) =>
      n > 0
    )
    : [];
  if (caps.length === 0) return none;

  const offer = cudaOffer(version, caps, driverCudaVersion(smi.stdout));
  if (!offer) {
    return {
      ok: true,
      note: own
        ? `CUDA ${own.version}, installed by llama.master`
        : `CUDA ${parseCudaVersion(version)}`,
      managed: Boolean(own),
      path: own?.path ?? nvcc?.path ?? "",
      version: own?.version ?? version,
    };
  }
  return { ok: false, ...offer };
}

/** Everything the Prerequisites panel lists, in the order it lists them. */
export async function detect(): Promise<Prereq[]> {
  const [
    cmake,
    compiler,
    git,
    ccache,
    nvcc,
    nvidiaSmi,
    glslc,
    hipcc,
    spirv,
    hipDev,
  ] = await Promise.all([
    resolveCmake(),
    resolveCompiler(),
    probe("git"),
    probe("ccache"),
    probe("nvcc"),
    probe("nvidia-smi", ["--version"]),
    probe("glslc"),
    probe("hipcc"),
    resolveSpirvHeaders(),
    resolveHipDev(),
  ]);

  // Is the toolkit new enough for the cards that are actually in this machine?
  // A separate question from "is nvcc installed", and the one that costs real
  // speed: a toolkit older than the GPU can only emit PTX, which the driver
  // re-compiles at every load and which runs slower than native code. Measured
  // on this machine — CUDA 12.0 against Blackwell (sm_120).
  const cudaFit = await resolveCudaFit(nvcc);

  const mk = (
    id: string,
    label: string,
    why: string,
    found: { path: string; version: string; managed?: boolean } | null,
    opts: { managed?: boolean; systemOnly?: boolean } = {},
  ): Prereq => ({
    id,
    label,
    why,
    found: found !== null,
    version: found?.version ?? "",
    path: found?.path ?? "",
    managed: opts.managed ?? false,
    systemOnly: opts.systemOnly ?? false,
  });

  return [
    mk("deno", "Deno", "Runs llama.master itself.", {
      path: Deno.execPath(),
      version: `deno ${Deno.version.deno}`,
    }),
    mk(
      "cmake",
      "CMake",
      "Configures and drives the llama.cpp build. Downloaded on demand if absent.",
      cmake,
      { managed: cmake?.managed },
    ),
    mk(
      "compiler",
      "C++ compiler",
      "Compiles llama.cpp. Cannot be downloaded — install build tools, or use a prebuilt release.",
      compiler,
      { systemOnly: true },
    ),
    mk(
      "git",
      "git",
      "Optional. Source is fetched as a tarball, so a build works without it.",
      git,
      { systemOnly: true },
    ),
    mk(
      "ccache",
      "ccache",
      "Optional. Makes a rebuild of the same ref several times faster.",
      ccache,
      { systemOnly: true },
    ),
    mk(
      "nvidia",
      "NVIDIA driver",
      "Required for the CUDA backend at run time.",
      nvidiaSmi,
      { systemOnly: true },
    ),
    mk(
      "cuda",
      "CUDA toolkit (nvcc)",
      "Required to COMPILE the CUDA backend. Prebuilt CUDA releases need only the driver.",
      nvcc,
      { systemOnly: true },
    ),
    mk(
      "cuda-arch",
      "CUDA new enough for your GPU",
      cudaFit.ok
        ? "The CUDA toolkit here can build native code for the cards here."
        : `CUDA ${cudaFit.have} cannot build native code for sm_${
          Math.round(cudaFit.newestCap * 10)
        } — it can only emit PTX, which the driver re-compiles on every load and which runs slower. CUDA ${cudaFit.need} or newer fixes it. llama.master can install one into its own folder: no administrator rights, no driver change, nothing outside ~/.llama-master.`,
      cudaFit.ok && cudaFit.note
        ? { path: cudaFit.path, version: cudaFit.note }
        : cudaFit.ok
        ? { path: "", version: "not applicable" }
        : null,
      { managed: cudaFit.ok ? cudaFit.managed : false },
    ),
    mk(
      "vulkan",
      "Vulkan (glslc)",
      "Required to compile the Vulkan backend — the portable GPU option. llama.cpp needs glslc only (`find_package(Vulkan COMPONENTS glslc)`); cmake's note about glslangValidator is informational.",
      glslc,
      { systemOnly: true },
    ),
    mk(
      "spirv",
      "SPIRV-Headers",
      "Required by the Vulkan backend (`find_package(SPIRV-Headers CONFIG REQUIRED)`). Headers only — downloaded on demand if absent.",
      spirv,
      { managed: spirv?.managed },
    ),
    mk(
      "hip",
      "ROCm (HIP dev)",
      hipcc && !hipDev
        ? "hipcc is installed but the HIP development package is not — cmake needs hip-lang-config.cmake, which only the -dev packages ship."
        : "Required to compile the ROCm backend for AMD cards on Linux. Needs hipcc AND the HIP development package.",
      // Both, or it is not buildable — hipcc alone lets cmake fail later.
      hipcc && hipDev ? hipcc : null,
      { systemOnly: true },
    ),
  ];
}

// ── CMake installation ─────────────────────────────────────────────────────

type GhAsset = { name: string; browser_download_url: string; size: number };
type GhRelease = { tag_name: string; assets: GhAsset[] };

/** Pick the Kitware asset for this platform. Names are stable across releases:
 *  `cmake-<ver>-<os>-<arch>.<tar.gz|zip>`. */
function pickCmakeAsset(assets: GhAsset[]): GhAsset | null {
  const os = PLATFORM === "darwin"
    ? "macos"
    : PLATFORM === "windows"
    ? "windows"
    : "linux";
  const arch = ARCH === "aarch64" ? "aarch64" : "x86_64";
  const wanted = os === "macos"
    ? ["macos-universal"]
    : [`${os}-${arch}`, `${os}-${arch === "aarch64" ? "arm64" : arch}`];
  const ext = os === "windows" ? ".zip" : ".tar.gz";
  return (
    assets.find(
      (a) =>
        a.name.startsWith("cmake-") &&
        a.name.endsWith(ext) &&
        wanted.some((w) => a.name.includes(w)),
    ) ?? null
  );
}

/** Download the latest CMake into the app's toolchain directory.
 *  Returns the path to the binary it installed. */
export async function installCmake(
  onProgress: (received: number, total: number | null, note: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  onProgress(0, null, "Looking up the latest CMake release");
  const rel = await fetchJson<GhRelease>(
    "https://api.github.com/repos/Kitware/CMake/releases/latest",
  );
  const asset = pickCmakeAsset(rel.assets);
  if (!asset) {
    throw new Error(
      `no CMake build published for ${PLATFORM}/${ARCH} in ${rel.tag_name}`,
    );
  }

  onProgress(0, asset.size, `Downloading ${asset.name}`);
  const bytes = await download(
    asset.browser_download_url,
    (r, t) => onProgress(r, t ?? asset.size, `Downloading ${asset.name}`),
    signal,
  );

  const dest = join(paths().toolchain, "cmake");
  await Deno.remove(dest, { recursive: true }).catch(() => {});
  await ensureDir(dest);
  onProgress(asset.size, asset.size, "Extracting");
  await extract(bytes, dest, asset.name.endsWith(".zip") ? "zip" : "tar.gz");

  const bin = await managedCmakePath();
  if (!bin) {
    throw new Error(
      `CMake extracted to ${dest} but no cmake binary was found inside it`,
    );
  }
  const check = await exec(bin, ["--version"]);
  if (check.code !== 0) {
    throw new Error(`installed CMake is not runnable: ${check.stderr.trim()}`);
  }
  onProgress(asset.size, asset.size, firstLine(check.stdout));
  return bin;
}

// ── fixing what is missing ─────────────────────────────────────────────────

const MANAGERS: [PackageManager, string][] = [
  ["apt", "apt-get"],
  ["dnf", "dnf"],
  ["pacman", "pacman"],
  ["zypper", "zypper"],
  ["brew", "brew"],
];

/** `/etc/os-release` → `{ id, version }`. Null off Linux or when unreadable —
 *  which is a real answer: it means "do not guess a repository URL". */
export async function detectDistro(): Promise<Distro | null> {
  if (PLATFORM !== "linux") return null;
  try {
    const text = await Deno.readTextFile("/etc/os-release");
    const field = (key: string) =>
      new RegExp(`^${key}=\"?([^\"\n]*)\"?`, "m").exec(text)?.[1] ?? "";
    const id = field("ID");
    return id
      ? {
        id,
        version: field("VERSION_ID"),
        ubuntuCodename: field("UBUNTU_CODENAME"),
      }
      : null;
  } catch {
    return null;
  }
}

/** The first package manager on PATH, or null on a system we do not know. */
export async function detectPackageManager(): Promise<PackageManager | null> {
  for (const [name, bin] of MANAGERS) {
    if (await which(bin)) return name;
  }
  return null;
}

/** What `fix()` would do for each id, without doing it — the UI shows this on
 *  the button so nothing privileged ever runs unexplained.
 *
 *  Every plan in ONE call on purpose: `fixPlan` is pure, so the only I/O is
 *  finding the package manager, and doing that once per scan instead of once
 *  per prerequisite turns ~45 PATH probes into ~5 and the cell's nine sequential
 *  awaits into one. Nine awaits in a method are also nine commit points, which
 *  made the panel flicker and the plans occasionally land after a test settled. */
export async function plansFor(
  ids: readonly string[],
): Promise<Record<string, FixPlan>> {
  const [manager, distro, hipcc] = await Promise.all([
    detectPackageManager(),
    detectDistro(),
    which("hipcc"),
  ]);
  const out: Record<string, FixPlan> = {};
  for (const id of ids) {
    out[id] = fixPlan(id, PLATFORM, manager, distro, hipcc !== null);
  }
  return out;
}

export type FixResult = { ok: boolean; message: string };

/**
 * Install one missing prerequisite.
 *
 * Downloads happen in-process. Package installs need root, so the command is
 * elevated through `pkexec` (the desktop's own auth agent) or a passwordless
 * `sudo`, and if neither can work the failure names the exact command to run by
 * hand rather than pretending to have tried.
 */
export async function fix(
  id: string,
  onLine: (line: string) => void,
  /** Byte progress for the downloads this app performs itself. The cell turns
   *  it into the progress bar; a long download with no bar reads as a hang. */
  onProgress: (received: number, total: number | null, note: string) => void =
    () => {},
): Promise<FixResult> {
  const plan = (await plansFor([id]))[id] as FixPlan;

  if (plan.kind === "manual") {
    return { ok: false, message: plan.reason };
  }

  if (plan.kind === "script") return await runScript(plan, onLine);

  if (plan.kind === "download") {
    onLine(plan.label);
    const report = (received: number, total: number | null, note: string) => {
      onProgress(received, total, note);
      onLine(
        total ? `${note} — ${Math.round((received / total) * 100)}%` : note,
      );
    };
    if (id === "spirv") {
      const prefix = await installSpirvHeaders(report);
      return { ok: true, message: `Installed ${prefix}` };
    }
    if (id === "cuda-arch") {
      // Which release to fetch is decided from the machine, not typed in: the
      // OLDEST one that covers the newest card, capped by what the driver can
      // run. A bigger jump is a bigger download for no gain.
      const fit = await resolveCudaFit(await probe("nvcc"));
      if (fit.ok) {
        return {
          ok: true,
          message: "The CUDA toolkit here already covers every card.",
        };
      }
      const version = await newestRedistFor(fit.need);
      if (!version) {
        return {
          ok: false,
          message:
            `NVIDIA publishes no downloadable CUDA ${fit.need} for this platform, so there is nothing safe to install automatically.`,
        };
      }
      for (const line of cudaFixSummary(version, 0, cudaPrefix(version))) {
        onLine(line);
      }
      const nvcc = await installCudaToolkit(version, report);
      return { ok: true, message: `Installed ${nvcc}` };
    }
    const bin = await installCmake(report);
    return { ok: true, message: `Installed ${bin}` };
  }

  const isRoot = (await exec("id", ["-u"])).stdout.trim() === "0";
  const hasPkexec = (await which("pkexec")) !== null;
  // `sudo -n` only counts if it actually works without a password: a prompt
  // would hang a GUI app forever with nothing on screen.
  const sudoWorks = !isRoot && !hasPkexec &&
    (await exec("sudo", ["-n", "true"])).code === 0;

  const argv = elevate(plan.command, {
    isRoot,
    pkexec: hasPkexec,
    sudo: sudoWorks,
  });
  if (!argv) {
    return {
      ok: false,
      message: `Cannot elevate: run this yourself — sudo ${
        plan.command.join(" ")
      }`,
    };
  }

  onLine(`$ ${argv.join(" ")}`);
  const { execStream } = await import("./host.server.ts");
  const code = await execStream(
    argv[0] as string,
    argv.slice(1),
    {},
    (line) => onLine(line),
  );
  if (code !== 0) {
    return {
      ok: false,
      message:
        `${plan.manager} exited with code ${code} — the log above says why`,
    };
  }
  return { ok: true, message: `Installed ${plan.packages.join(", ")}` };
}

/** How a script plan's steps are elevated, resolved once for the whole run. */
async function elevation(): Promise<
  { ok: true; wrap: (sh: string) => string[] } | { ok: false; why: string }
> {
  const isRoot = (await exec("id", ["-u"])).stdout.trim() === "0";
  if (isRoot) return { ok: true, wrap: (sh) => ["bash", "-c", sh] };
  if (await which("pkexec")) {
    return { ok: true, wrap: (sh) => ["pkexec", "bash", "-c", sh] };
  }
  if ((await exec("sudo", ["-n", "true"])).code === 0) {
    return { ok: true, wrap: (sh) => ["sudo", "-n", "bash", "-c", sh] };
  }
  return {
    ok: false,
    why:
      "no way to run a privileged command (no pkexec, and sudo needs a password)",
  };
}

/**
 * Run a documented multi-step install, stopping at the first failure.
 *
 * Each command is echoed before it runs, so the log is a transcript of exactly
 * what was done to this machine — which matters when the steps add a
 * third-party repository and a GPU driver.
 */
async function runScript(
  plan: Extract<FixPlan, { kind: "script" }>,
  onLine: (line: string) => void,
): Promise<FixResult> {
  const elev = await elevation();
  if (!elev.ok) {
    onLine(`Cannot run these steps: ${elev.why}. Run them yourself:`);
    // POSIX-quoted: a step that itself contains a single quote (the ROCm
    // repository line does) would otherwise end the string early and print
    // a command that silently does something else.
    for (const st of plan.steps) onLine(`  sudo bash -c ${quote(st.sh)}`);
    return { ok: false, message: elev.why };
  }

  const { execStream } = await import("./host.server.ts");
  for (const [i, st] of plan.steps.entries()) {
    onLine(`[${i + 1}/${plan.steps.length}] ${st.label}`);
    onLine(`$ ${st.sh}`);
    const argv = elev.wrap(st.sh);
    const code = await execStream(
      argv[0] as string,
      argv.slice(1),
      {},
      (line) => onLine(line),
    );
    if (code !== 0) {
      return {
        ok: false,
        message: `Step ${
          i + 1
        } ("${st.label}") exited with code ${code}. Nothing after it ran — see ${plan.docsUrl}`,
      };
    }
  }
  return {
    ok: true,
    message: plan.rebootAfter
      ? "Installed. Reboot before using it — the driver and your new group membership only take effect then."
      : "Installed.",
  };
}
