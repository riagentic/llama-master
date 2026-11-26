// src/cell/basalt.server.ts — install the Basalt engine as a build. SERVER ONLY.
//
// Basalt is a compiled engine (`bin/basalt`) under a Python server in the
// same tree: `basalt serve PACK [flags]` starts the server from `.venv`, and
// the server starts the binary as its engine. So a build is the whole
// checkout under the builds root, compiled IN PLACE (a venv carries absolute
// paths, as for Strata), with the metadata written last so a build that
// failed half-way is not listed.
//
// The model is a `.basalt` pack built from the official release. Packs live in
// `cache/basalt/packs` (an update or a rebuild must not fetch 20 GB again) and
// the build links them in as `packs/`. A pack is used only once it is VERIFIED:
// a download against the published SHA-256, after which `<pack>.ok` records it.

import { join } from "@std/path";
import type { Build } from "../lib/types.ts";
import { parseRef, tarballUrl } from "../lib/srcref.ts";
import { parseCudaVersion } from "../lib/cuda.ts";
import { bytes } from "../lib/format.ts";
import {
  BASALT_BIN,
  BASALT_DOCS,
  BASALT_PACKS,
  basaltModel,
  basaltPack,
} from "../lib/basalt.ts";
import {
  dirSize,
  download,
  downloadToFile,
  ensureDir,
  exec,
  execStream,
  exists,
  extract,
  paths,
  PLATFORM,
} from "./host.server.ts";
import {
  BuildFailure,
  buildId,
  movingSha,
  writeMeta,
} from "./builds.server.ts";

type OnProgress = (p: {
  step: number;
  steps: string[];
  progress: number | null;
  lines?: string[];
}) => void;

/** Basalt's kernels need CUDA 13 (its README: "an NVIDIA Blackwell GPU with
 *  CUDA 13"). */
const MIN_CUDA = 13;

export function basaltPacksDir(): string {
  return join(paths().cache, "basalt", "packs");
}

const fail = (reason: string, steps: BuildFailure["diagnosis"]["steps"]) =>
  new BuildFailure({ reason, steps });

export async function buildBasalt(
  opts: {
    ref: string;
    model?: string;
    backend?: string;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = [
    "Fetch source",
    "Compile",
    "Python environment",
    "Model pack",
    "Verify",
  ];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  if (PLATFORM !== "linux") {
    throw fail(
      "llama.master builds Basalt on Linux only for now. Basalt itself also ships Windows archives.",
      [{
        text: "Basalt's releases.",
        action: { kind: "open-url", url: BASALT_DOCS },
      }],
    );
  }
  if (opts.backend && opts.backend !== "cuda") {
    throw fail(
      `Basalt is an NVIDIA Blackwell engine; the Build page has ${opts.backend} selected.`,
      [{
        text: "Choose the CUDA backend on the Build page, then build again.",
        action: { kind: "open-tab", tab: "build" },
      }],
    );
  }
  const model = basaltModel(opts.model ?? "");
  if (!model) {
    throw fail(
      "Basalt is built here for one model, and it is not the one selected: select an official Inkling 2 Mini GGUF on the Models page (it only names the model — Basalt runs its own pack of the official weights).",
      [{
        text: "Which models Basalt runs.",
        action: { kind: "open-url", url: `${BASALT_DOCS}#models` },
      }],
    );
  }
  const { managedCuda, resolveCmake } = await import("./prereq.server.ts");
  const cuda = await managedCuda();
  if (!cuda || parseCudaVersion(cuda.version) < MIN_CUDA) {
    throw fail(
      `Basalt needs CUDA ${MIN_CUDA} to compile its Blackwell kernels, and ${
        cuda
          ? `the app's toolkit is ${cuda.version}`
          : "the app has no CUDA toolkit of its own"
      }.`,
      [{
        text:
          "Install a newer toolkit from the prerequisites on the Build page (no root, nothing system-wide).",
        action: { kind: "open-tab", tab: "build" },
      }],
    );
  }
  const cmake = await resolveCmake();
  if (!cmake) {
    throw fail("Basalt builds with CMake, and none is installed.", [{
      text: "Install CMake from the prerequisites on the Build page.",
      action: { kind: "open-tab", tab: "build" },
    }]);
  }

  const id = buildId("source", opts.ref, "cuda");
  const dir = join(paths().builds, id);
  await ensureDir(dir);
  // Unlisted until it verifies again: a half-updated tree must not be offered
  // as the build that worked.
  await Deno.remove(join(dir, "llama-master.json")).catch(() => {});

  const url = tarballUrl(parseRef(opts.ref));
  p(0, 0, ["Fetching Basalt", url]);
  const tarball = await download(
    url,
    (received, total) => p(0, total ? received / total : null),
    opts.signal,
  );
  const n = await extract(tarball, dir, "tar.gz");
  p(0, 1, [`${n} source files extracted to ${dir}`]);
  opts.signal?.throwIfAborted();

  // The app's toolkit first on PATH, and named for CMake: the same nvcc and
  // runtime a llama.cpp source build gets, found again at run time through
  // the RUNPATH the build records.
  const env = {
    CUDACXX: cuda.nvcc,
    PATH: `${join(cuda.path, "bin")}:${Deno.env.get("PATH") ?? ""}`,
  };
  const cmakeStep = async (args: string[], what: string) => {
    p(1, null, [`$ cmake ${args.join(" ")}`]);
    const code = await execStream(
      cmake.path,
      args,
      { cwd: dir, env, signal: opts.signal },
      (line) => {
        // `[ 42%] Building CUDA object …` is the only honest progress here.
        const m = /^\[\s*(\d+)%\]/.exec(line);
        p(1, m ? Number(m[1]) / 100 : null, [line]);
      },
    );
    opts.signal?.throwIfAborted();
    if (code !== 0) {
      throw new Error(
        `Basalt's ${what} stopped with code ${code} — the reason is in the log above.`,
      );
    }
  };
  // A cache from an earlier configure keeps the libraries it found then,
  // whatever this one is told; the objects stay, so a rebuild is still quick.
  await Deno.remove(join(dir, "build", "CMakeCache.txt")).catch(() => {});
  await cmakeStep(
    [
      "-S",
      ".",
      "-B",
      "build",
      "-DCMAKE_BUILD_TYPE=Release",
      // Every model: one build serves every pack a later version adds.
      "-DBASALT_MODELS=all",
      // Pin the toolkit for the LIBRARIES too, as the llama.cpp route does:
      // CUDACXX names only the compiler, and a newer CMake then took the
      // system's CUDA 12.0 libcudart, which lacks
      // `cudaGetDriverEntryPointByVersion` — a link failure 40% in.
      `-DCMAKE_CUDA_COMPILER=${cuda.nvcc}`,
      `-DCUDAToolkit_ROOT=${cuda.path}`,
    ],
    "configure",
  );
  await cmakeStep(
    ["--build", "build", "-j", String(navigator.hardwareConcurrency || 8)],
    "compile",
  );

  p(2, null);
  const { makeVenv } = await import("./strata.server.ts");
  await makeVenv(dir, (lines) => p(2, null, lines), "Basalt");
  const pip = await execStream(
    join(dir, ".venv", "bin", "python"),
    [
      "-m",
      "pip",
      "install",
      "--disable-pip-version-check",
      "-r",
      "requirements.txt",
    ],
    { cwd: dir, signal: opts.signal },
    (line) => p(2, null, [line]),
  );
  opts.signal?.throwIfAborted();
  if (pip !== 0) {
    throw new Error(
      `Installing Basalt's server dependencies stopped with code ${pip} — pip's reason is in the log above.`,
    );
  }
  p(2, 1);

  // A selected `.basalt` pack runs from where it is (`basaltArgs`): nothing
  // to download for it.
  const own = /\.basalt$/i.test(opts.model ?? "") &&
      await exists(opts.model ?? "")
    ? opts.model ?? null
    : null;
  await ensurePack(
    model,
    dir,
    own,
    opts.signal,
    (progress, lines) => p(3, progress, lines),
  );

  p(4, null);
  const bin = join(dir, BASALT_BIN);
  const help = await exec(bin, ["--help"], { timeoutMs: 10_000 });
  if (help.code !== 0) {
    throw new Error(
      `Basalt compiled, but ${bin} does not start (exit ${help.code}): ${
        help.stderr.trim().split("\n").slice(-3).join(" ")
      }`,
    );
  }
  const build: Build = {
    id,
    ref: opts.ref,
    origin: "source",
    backend: "cuda",
    engine: "basalt",
    dir,
    serverBin: bin,
    cliBin: "",
    createdAt: Date.now(),
    sizeB: await dirSize(dir),
    sourceSha: await movingSha(opts.ref).catch(() => ""),
  };
  await writeMeta(dir, build);
  p(4, 1, [`Basalt is ready for ${model.title} at ${dir}`]);
  return build;
}

/** Link the shared packs folder into the build and make sure this model's
 *  pack is in it, verified. */
async function ensurePack(
  model: NonNullable<ReturnType<typeof basaltModel>>,
  dir: string,
  own: string | null,
  signal: AbortSignal | undefined,
  say: (progress: number | null, lines?: string[]) => void,
): Promise<void> {
  const packs = basaltPacksDir();
  await ensureDir(packs);
  const link = join(dir, BASALT_PACKS);
  await Deno.remove(link).catch(() => {});
  await Deno.symlink(packs, link);
  if (own) {
    say(1, [`Runs your pack ${own}: nothing to download.`]);
    return;
  }

  const pack = basaltPack(model);
  const file = join(packs, pack.file);
  const ok = `${file}.ok`;
  if (await exists(ok) && await exists(file)) {
    say(1, [`Pack ${pack.file} is already here, verified.`]);
    return;
  }
  const { disks } = await import("./hw.server.ts");
  const part = `${file}.part`;
  const have = await Deno.stat(part).then((s) => s.size, () => 0);
  const free = (await disks([packs]))[0]?.availB ?? 0;
  if (free > 0 && free < pack.sizeB - have) {
    throw fail(
      `The ${pack.file} pack is ${bytes(pack.sizeB)} and ${
        bytes(free)
      } is free where it goes (${packs}).`,
      [{
        text:
          "Free some disk space, then build again — the download resumes where it stopped.",
      }],
    );
  }
  say(0, [
    `Downloading the official weights as ${pack.file} (${bytes(pack.sizeB)})`,
    pack.url,
  ]);
  await downloadToFile(
    pack.url,
    part,
    (received, total) => say(received / (total ?? pack.sizeB)),
    signal,
  );
  say(null, ["Checking the SHA-256 of the pack"]);
  const sum = await exec("sha256sum", [part]);
  const got = sum.stdout.split(/\s/)[0] ?? "";
  if (got !== pack.sha256) {
    await Deno.remove(part).catch(() => {});
    throw new Error(
      `The downloaded pack does not match its published SHA-256 (got ${
        got || sum.stderr.trim()
      }, expected ${pack.sha256}); it was deleted.`,
    );
  }
  await Deno.rename(part, file);
  await Deno.writeTextFile(ok, `${pack.sha256}  ${pack.url}\n`);
  say(1, [`Pack verified: ${pack.sha256}`]);
}
