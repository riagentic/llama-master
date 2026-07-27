// src/cell/builds.server.ts — get llama.cpp onto this machine. SERVER ONLY.
//
// Two routes to the same result, a directory with `llama-server` and
// `llama-cli` in it:
//
//   source  — fetch the ref's tarball, cmake configure, cmake build.
//             Needs a C++ compiler; CMake is downloaded if missing.
//   release — fetch the official prebuilt asset for this platform/backend.
//             Needs nothing at all, which is what makes the app work on a
//             bare OS (kata: "no prerequisites except a running OS").
//
// Source is fetched as a tarball rather than cloned: it removes git from the
// prerequisite list, downloads ~20x less, and a specific tag is exactly what a
// user asking for "b6234" means.

import { join } from "@std/path";
import type { Asset } from "../lib/assets.ts";
import { availableBackends, pickAsset } from "../lib/assets.ts";
import { progressOf } from "../lib/buildlog.ts";
import { cudaCmakeFlags, cudaPlan } from "../lib/cuda.ts";
import { diagnoseNoAsset } from "../lib/diagnose.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { CudaPlan } from "../lib/cuda.ts";
import type { Backend, Build } from "../lib/types.ts";
import {
  ARCH,
  dirSize,
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
  RateLimited,
  resolveRedirect,
} from "./host.server.ts";
import {
  assetsFromHtml,
  assetUrl,
  shaFromCommitsAtom,
  tagFromReleaseUrl,
} from "../lib/github.ts";
import { resolveCmake } from "./prereq.server.ts";
import { DEMO_ENV, demoBuilds } from "../lib/demo.ts";

const REPO = "ggml-org/llama.cpp";
const API = `https://api.github.com/repos/${REPO}`;

export const BIN_SERVER = PLATFORM === "windows"
  ? "llama-server.exe"
  : "llama-server";
export const BIN_CLI = PLATFORM === "windows" ? "llama-cli.exe" : "llama-cli";

/** Progress reporting shared by both routes. `progress` is null while a step's
 *  total is genuinely unknown (cmake configure), never a fake animation. */
export type Progress = {
  step: number;
  steps: string[];
  progress: number | null;
  lines?: string[];
};
export type OnProgress = (p: Progress) => void;

/** A failure that already knows how to explain itself. The cell stores the
 *  diagnosis alongside the job so the UI can render steps and buttons rather
 *  than a wall of text. */
export class BuildFailure extends Error {
  constructor(readonly diagnosis: Diagnosis) {
    super(diagnosis.reason);
    this.name = "BuildFailure";
  }
}

// ── upstream metadata ──────────────────────────────────────────────────────

type GhTag = { name: string };
type GhCommit = { sha: string };
type GhAsset = { name: string; browser_download_url: string; size: number };
type GhRelease = { tag_name: string; assets: GhAsset[]; published_at: string };

/** Release tags, newest first. llama.cpp tags builds as `b<number>`. */
export async function listRefs(): Promise<string[]> {
  try {
    const tags = await fetchJson<GhTag[]>(`${API}/tags?per_page=100`);
    return tags.map((t) => t.name);
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    // The releases atom feed is not rate limited and carries the same tags.
    const xml = await fetchText(`https://github.com/${REPO}/releases.atom`);
    const tags = [...xml.matchAll(/\/releases\/tag\/([^"<]+)/g)]
      .map((m) => decodeURIComponent(m[1] as string));
    return [...new Set(tags)];
  }
}

/** The newest published release tag. Falls back to the release page when the
 *  API is rate limited, so the Update button keeps working. */
export async function latestTag(): Promise<string> {
  try {
    const rel = await fetchJson<GhRelease>(`${API}/releases/latest`);
    return rel.tag_name;
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    return tagFromReleaseUrl(
      await resolveRedirect(`https://github.com/${REPO}/releases/latest`),
    ) ?? "";
  }
}

/** The commit `master` currently points at — the only thing that distinguishes
 *  one "master" build from the next. */
export async function masterSha(): Promise<string> {
  try {
    const c = await fetchJson<GhCommit>(`${API}/commits/master`);
    return c.sha;
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    // Same treatment as listRefs/latestTag: without this, an exhausted quota
    // hides the Update button on every `master` build — the one case where the
    // sha IS the version. Atom feeds are not rate limited.
    const xml = await fetchText(
      `https://github.com/${REPO}/commits/master.atom`,
    );
    return shaFromCommitsAtom(xml) ?? "";
  }
}

export async function listAssets(
  ref: string,
): Promise<{ tag: string; assets: Asset[] }> {
  try {
    const rel = await fetchJson<GhRelease>(
      ref === "master"
        ? `${API}/releases/latest`
        : `${API}/releases/tags/${ref}`,
    );
    return {
      tag: rel.tag_name,
      assets: rel.assets.map((a) => ({
        name: a.name,
        url: a.browser_download_url,
        sizeB: a.size,
      })),
    };
  } catch (e) {
    // A rate-limited API must not stop an install: downloads were never
    // limited, and github.com's own pages carry the same facts.
    if (!(e instanceof RateLimited)) throw e;
    return await listAssetsWithoutApi(ref);
  }
}

/**
 * The same answer from plain github.com pages, which have no rate limit.
 *
 * Sizes are unknown this way (the HTML rounds them), so they come back as 0 —
 * the download reports real progress from `content-length` regardless, and the
 * picker only ever needed names.
 */
export async function listAssetsWithoutApi(
  ref: string,
): Promise<{ tag: string; assets: Asset[] }> {
  const tag = ref === "master"
    ? tagFromReleaseUrl(
      await resolveRedirect(`https://github.com/${REPO}/releases/latest`),
    )
    : ref;
  if (!tag) {
    throw new Error(
      "GitHub's API is rate limited and the latest release tag could not be resolved from the release page either.",
    );
  }
  const html = await fetchText(
    `https://github.com/${REPO}/releases/expanded_assets/${tag}`,
  );
  const names = assetsFromHtml(html, REPO);
  if (names.length === 0) {
    throw new Error(
      `GitHub's API is rate limited and no assets were listed for ${tag}.`,
    );
  }
  return {
    tag,
    assets: names.map((name) => ({
      name,
      url: assetUrl(REPO, tag, name),
      sizeB: 0,
    })),
  };
}

// ── the build registry ─────────────────────────────────────────────────────

const META = "llama-master.json";

export function buildId(
  origin: Build["origin"],
  ref: string,
  backend: Backend,
): string {
  return `${origin}-${ref}-${backend}`;
}

async function writeMeta(dir: string, b: Build): Promise<void> {
  await Deno.writeTextFile(join(dir, META), JSON.stringify(b, null, 2));
}

/** Every build on disk. The directory IS the registry — no index file to drift
 *  out of sync with reality, and deleting a directory is a valid uninstall. */
export async function listBuilds(): Promise<Build[]> {
  // Demo mode: a build that does not exist, so the app can be shown working
  // without a llama.cpp install (src/lib/demo.ts).
  if (Deno.env.get(DEMO_ENV) === "1") return demoBuilds();
  const root = paths().builds;
  const out: Build[] = [];
  try {
    for await (const e of Deno.readDir(root)) {
      if (!e.isDirectory) continue;
      const dir = join(root, e.name);
      try {
        const meta = JSON.parse(
          await Deno.readTextFile(join(dir, META)),
        ) as Build;
        // Trust the directory over the metadata: a moved app home must not
        // leave every build pointing at paths that no longer exist.
        const serverBin = await findBinary(dir, BIN_SERVER);
        const cliBin = await findBinary(dir, BIN_CLI);
        if (!serverBin) continue;
        out.push({
          ...meta,
          dir,
          serverBin,
          cliBin: cliBin ?? "",
          sizeB: await dirSize(dir),
        });
      } catch {
        // A directory without readable metadata is a half-finished install;
        // it is skipped here and overwritten by the next install of that id.
      }
    }
  } catch {
    // No builds directory yet.
  }
  return out.sort((a, b) => b.createdAt - a.createdAt);
}

export async function removeBuild(id: string): Promise<void> {
  const dir = join(paths().builds, id);
  // Refuse anything that is not a direct child of the builds root.
  if (!dir.startsWith(paths().builds + "/") || id.includes("..")) {
    throw new Error(`refusing to remove ${dir}`);
  }
  await Deno.remove(dir, { recursive: true });
}

/** Depth-limited search for a named binary inside an extracted tree. */
async function findBinary(
  dir: string,
  name: string,
  depth = 3,
): Promise<string | null> {
  const direct = join(dir, name);
  if (await exists(direct)) return direct;
  const inBin = join(dir, "bin", name);
  if (await exists(inBin)) return inBin;
  if (depth <= 0) return null;
  try {
    for await (const e of Deno.readDir(dir)) {
      if (!e.isDirectory) continue;
      const found = await findBinary(join(dir, e.name), name, depth - 1);
      if (found) return found;
    }
  } catch {
    // Unreadable subtree — nothing to find here.
  }
  return null;
}

// ── route 1: prebuilt release ──────────────────────────────────────────────

export async function installRelease(
  opts: {
    ref: string;
    backend: Backend;
    assetName?: string;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = ["Find release", "Download", "Extract", "Verify"];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  p(0, null, [`Looking up ${opts.ref} on ${REPO}`]);
  const { tag, assets } = await listAssets(opts.ref);
  const asset = opts.assetName
    ? assets.find((a) => a.name === opts.assetName) ?? null
    : pickAsset(assets, PLATFORM, ARCH, opts.backend);
  if (!asset) {
    // Never a filename dump: say why, and give the route that works. The
    // prerequisite state is read here so the advice is accurate — "you already
    // have nvcc, just switch route" reads very differently from "install nvcc".
    const { detect } = await import("./prereq.server.ts");
    const found = new Set(
      (await detect()).filter((i) => i.found).map((i) => i.id),
    );
    throw new BuildFailure(
      diagnoseNoAsset(
        {
          origin: "release",
          backend: opts.backend,
          platform: PLATFORM,
          arch: ARCH,
          availableBackends: availableBackends(assets, PLATFORM, ARCH),
          found,
        },
        assets.length,
      ),
    );
  }

  p(1, 0, [
    asset.sizeB > 0
      ? `Downloading ${asset.name} (${(asset.sizeB / 1e6).toFixed(0)} MB)`
      : `Downloading ${asset.name}`,
  ]);
  const bytes = await download(
    asset.url,
    (received, total) => p(1, total ? received / total : null),
    opts.signal,
  );

  const id = buildId("release", tag, opts.backend);
  const dir = join(paths().builds, id);
  await Deno.remove(dir, { recursive: true }).catch(() => {});
  await ensureDir(dir);
  p(2, null, [`Extracting into ${dir}`]);
  const n = await extract(
    bytes,
    dir,
    asset.name.endsWith(".zip") ? "zip" : "tar.gz",
  );

  p(3, null, [`${n} files extracted, checking binaries`]);
  const build = await finalize({
    id,
    ref: tag,
    origin: "release",
    backend: opts.backend,
    dir,
  });
  p(3, 1, [`${BIN_SERVER} ready at ${build.serverBin}`]);
  return build;
}

// ── route 2: build from source ─────────────────────────────────────────────

const BACKEND_FLAGS: Record<Backend, string[]> = {
  cpu: [],
  cuda: ["-DGGML_CUDA=ON"],
  vulkan: ["-DGGML_VULKAN=ON"],
  hip: ["-DGGML_HIP=ON"],
  metal: ["-DGGML_METAL=ON"],
};

/** A tarball has no git metadata, so a source build reports
 *  `version: 0 (unknown)`. When the ref is a release tag we already know the
 *  number, so hand it to cmake and let the binary tell the truth about itself. */
export function buildNumberFlags(ref: string): string[] {
  const m = /^b(\d+)$/.exec(ref);
  return m ? [`-DLLAMA_BUILD_NUMBER=${m[1]}`] : [];
}

/** What CUDA can target here — nvcc's version against the driver's report of
 *  each GPU. Read at build time so the answer is never stale. */
export async function detectCudaPlan(): Promise<CudaPlan> {
  const [nvcc, smi] = await Promise.all([
    exec("nvcc", ["--version"]),
    exec("nvidia-smi", ["--query-gpu=compute_cap", "--format=csv,noheader"]),
  ]);
  const caps = smi.code === 0
    ? smi.stdout.split("\n").map((l) => Number(l.trim())).filter((n) => n > 0)
    : [];
  return cudaPlan(nvcc.stdout || nvcc.stderr, caps);
}

export async function buildFromSource(
  opts: {
    ref: string;
    backend: Backend;
    jobs: number;
    native: boolean;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = ["Fetch source", "Configure", "Compile", "Install"];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  const cmake = await resolveCmake();
  if (!cmake) {
    throw new Error(
      "CMake not found. Install it from the Prerequisites panel — llama.master can download it.",
    );
  }

  // Vulkan: point cmake at the app's own SPIRV-Headers when the system has
  // none. `find_package` is satisfied by CMAKE_PREFIX_PATH, but llama.cpp does
  // not link the imported target — it expects the headers on the default
  // include path — so the include directory has to be added too, or the compile
  // fails later with "'spv' has not been declared".
  let vulkanFlags: string[] = [];
  if (opts.backend === "vulkan") {
    const { resolveSpirvHeaders } = await import("./prereq.server.ts");
    const spirv = await resolveSpirvHeaders();
    if (!spirv) {
      throw new Error(
        "SPIRV-Headers not found. Install it from the Prerequisites panel — llama.master can download it (headers only, no root needed).",
      );
    }
    if (spirv.managed) {
      vulkanFlags = [
        `-DCMAKE_PREFIX_PATH=${spirv.path}`,
        `-DCMAKE_CXX_FLAGS=-isystem ${join(spirv.path, "include")}`,
      ];
      p(0, null, [`Using the app's SPIRV-Headers at ${spirv.path}`]);
    }
  }

  // CUDA: name the architectures explicitly. Left to auto-detection, cmake asks
  // nvcc for the GPU's native arch, and an nvcc older than the card dies with
  // `Unsupported gpu architecture` several minutes into the compile.
  let cudaFlags: string[] = [];
  if (opts.backend === "cuda") {
    const plan = await detectCudaPlan();
    if (plan.mode === "impossible") {
      throw new Error(`${plan.reason} ${plan.remedy}`);
    }
    cudaFlags = cudaCmakeFlags(plan);
    p(0, null, [plan.reason, ...(plan.remedy ? [plan.remedy] : [])]);
  }

  // 1 — source tarball (cached: rebuilding the same ref must not re-download).
  const srcDir = join(paths().sources, opts.ref);
  if (!(await exists(join(srcDir, "CMakeLists.txt")))) {
    const url = opts.ref === "master"
      ? `https://codeload.github.com/${REPO}/tar.gz/refs/heads/master`
      : `https://codeload.github.com/${REPO}/tar.gz/refs/tags/${opts.ref}`;
    p(0, 0, [`Fetching ${url}`]);
    const bytes = await download(
      url,
      (received, total) => p(0, total ? received / total : null),
      opts.signal,
    );
    await Deno.remove(srcDir, { recursive: true }).catch(() => {});
    await ensureDir(srcDir);
    const n = await extract(bytes, srcDir, "tar.gz");
    p(0, 1, [`${n} source files extracted to ${srcDir}`]);
  } else {
    p(0, 1, [`Reusing cached source at ${srcDir}`]);
  }

  // 2 — configure.
  const buildDir = join(srcDir, `build-${opts.backend}`);
  await ensureDir(buildDir);
  const configureArgs = [
    "-S",
    srcDir,
    "-B",
    buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    // No libcurl dependency: model downloading is this app's job, not
    // llama.cpp's, and requiring libcurl-dev would break the "bare OS" promise.
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    `-DGGML_NATIVE=${opts.native ? "ON" : "OFF"}`,
    ...BACKEND_FLAGS[opts.backend],
    ...buildNumberFlags(opts.ref),
    ...cudaFlags,
    ...vulkanFlags,
  ];
  p(1, null, [`${cmake.path} ${configureArgs.join(" ")}`]);
  const cfgCode = await runStreaming(
    cmake.path,
    configureArgs,
    srcDir,
    opts.signal,
    (lines) => p(1, null, lines),
  );
  if (cfgCode !== 0) {
    throw new Error(
      `cmake configure failed (exit ${cfgCode}). The log tail above names the missing dependency.`,
    );
  }

  // 3 — compile. cmake prints `[ nn%]`, so the bar is real.
  const buildArgs = [
    "--build",
    buildDir,
    "--config",
    "Release",
    "-j",
    String(Math.max(1, opts.jobs)),
    "--target",
    "llama-server",
    "llama-cli",
  ];
  p(2, 0, [`${cmake.path} ${buildArgs.join(" ")}`]);
  let last = 0;
  const code = await runStreaming(
    cmake.path,
    buildArgs,
    srcDir,
    opts.signal,
    (lines) => {
      for (const l of lines) {
        const pr = progressOf(l);
        if (pr !== null) last = pr;
      }
      p(2, last, lines);
    },
  );
  if (code !== 0) throw new Error(`build failed (exit ${code})`);

  // 4 — install into the durable builds directory.
  const id = buildId("source", opts.ref, opts.backend);
  const dest = join(paths().builds, id);
  await Deno.remove(dest, { recursive: true }).catch(() => {});
  await ensureDir(join(dest, "bin"));
  p(3, null, [`Installing to ${dest}`]);
  const binSrc = join(buildDir, "bin");
  let copied = 0;
  for await (const e of Deno.readDir(binSrc)) {
    if (!e.isFile) continue;
    // The whole bin/ directory: the binaries need the ggml/llama shared
    // objects that sit beside them.
    await Deno.copyFile(join(binSrc, e.name), join(dest, "bin", e.name));
    copied++;
  }
  p(3, 0.9, [`${copied} files installed`]);

  const build = await finalize({
    id,
    ref: opts.ref,
    origin: "source",
    backend: opts.backend,
    dir: dest,
    // A tag identifies itself; "master" does not, so record what it was.
    sourceSha: opts.ref === "master" ? await masterSha().catch(() => "") : "",
  });
  p(3, 1, [`${BIN_SERVER} ready at ${build.serverBin}`]);
  return build;
}

/** Stream a child process, batching lines so the UI gets one dispatch per tick
 *  instead of one per line — a cmake build emits thousands. */
async function runStreaming(
  bin: string,
  args: string[],
  cwd: string,
  signal: AbortSignal | undefined,
  emit: (lines: string[]) => void,
): Promise<number> {
  const { execStream } = await import("./host.server.ts");
  let pending: string[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    if (pending.length === 0) return;
    emit(pending);
    pending = [];
  };
  const code = await execStream(bin, args, { cwd, signal }, (line) => {
    pending.push(line);
    timer ??= setTimeout(flush, 100);
  });
  if (timer !== null) clearTimeout(timer);
  flush();
  return code;
}

/** Locate the binaries, make them executable, write the metadata file. */
async function finalize(
  base: Omit<Build, "serverBin" | "cliBin" | "createdAt" | "sizeB">,
): Promise<Build> {
  const serverBin = await findBinary(base.dir, BIN_SERVER);
  const cliBin = await findBinary(base.dir, BIN_CLI);
  if (!serverBin) {
    throw new Error(
      `${BIN_SERVER} not found under ${base.dir} — the archive layout was not what we expected`,
    );
  }
  for (const b of [serverBin, cliBin]) if (b) await makeExecutable(b);

  const build: Build = {
    ...base,
    serverBin,
    cliBin: cliBin ?? "",
    createdAt: Date.now(),
    sizeB: await dirSize(base.dir),
  };
  await writeMeta(base.dir, build);

  // Prove it runs. A binary that cannot start (missing CUDA runtime, wrong
  // glibc) must fail here, not two clicks later when the user hits Start.
  const check = await exec(serverBin, ["--version"]);
  if (check.code === 127) {
    throw new Error(
      `installed ${BIN_SERVER} will not execute: ${
        check.stderr.trim() || "not runnable"
      }`,
    );
  }
  return build;
}
