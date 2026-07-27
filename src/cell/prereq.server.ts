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
  paths,
  PLATFORM,
  which,
} from "./host.server.ts";
import { join } from "@std/path";

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
  const r = await exec(path, args);
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
    for (const st of plan.steps) onLine(`  sudo bash -c '${st.sh}'`);
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
