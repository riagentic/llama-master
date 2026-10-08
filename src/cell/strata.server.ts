// src/cell/strata.server.ts — install the Strata engine as a build. SERVER ONLY.
//
// Strata is not one binary: it is a Python server, a compiled CUDA engine and
// a setup script that prepares ONE model (an index of the GGUF, the draft
// layer, a config). So a Strata build is its whole checkout under the builds
// root, and what the app spawns is a launcher it writes there itself
// (`STRATA_LAUNCHER`): run Strata's `setup.py` for the model — a no-op once
// nothing changed — then `exec` its server. One process, one pid, one log, and
// the argv on screen is the argv that ran.
//
// What Strata's installer would do on its own, and must not here:
//   - `sudo apt-get install` Python or the CUDA toolkit. The launcher puts a
//     `sudo` that REFUSES first on PATH, so no path through setup.py can reach
//     the real one, and `STRATA_NVCC` names the toolkit this app manages
//     (`prereq.server.ts:managedCuda`) — the same nvcc a llama.cpp build uses.
//   - download the model. `--gguf-dir` (`src/lib/strata.ts`) gives it the
//     files on disk; its ~6.5 GB draft layer is the one fetch left.
//
// Built IN PLACE, unlike a llama.cpp build: a Python venv carries absolute
// paths in every script it installs, so the stage-then-rename the other route
// uses would hand setup.py a `cmake` whose shebang names a directory that no
// longer exists. The metadata is still written last, so a build that failed
// half-way is not listed.

import { join } from "@std/path";
import type { Build } from "../lib/types.ts";
import { parseRef, tarballUrl } from "../lib/srcref.ts";
import {
  STRATA_DOCS,
  STRATA_LAUNCHER,
  strataArgs,
  strataModel,
} from "../lib/strata.ts";
import {
  dirSize,
  download,
  ensureDir,
  exec,
  execStream,
  exists,
  extract,
  makeExecutable,
  paths,
  PLATFORM,
  which,
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

/** Packs and the draft layer: ~6.5 GB that an engine update must not refetch. */
export function strataDataDir(): string {
  return join(paths().cache, "strata");
}

function launcherScript(nvcc: string, dataDir: string): string {
  return `#!/bin/sh
# Written by llama.master (src/cell/strata.server.ts) — do not edit; a rebuild
# replaces it. Usage: ${STRATA_LAUNCHER} <config-tag> <setup.py arguments…>
d=$(dirname "$(readlink -f "$0")"); cd "$d" || exit 1
tag=$1; shift
# No sudo, ever: setup.py offers to apt-get install Python and CUDA.
PATH="$d/.guard:$PATH"; export PATH
${nvcc ? `STRATA_NVCC='${nvcc}'; export STRATA_NVCC` : "# system nvcc"}
# Packs and the draft layer live outside the build, so an engine update does
# not fetch ~6.5 GB again.
.venv/bin/python setup.py --setup --no-start --yes --no-browser \\
  --data-dir '${dataDir}' "$@" </dev/null &
c=$!; trap 'kill "$c" 2>/dev/null' TERM INT; wait "$c"; rc=$?; trap - TERM INT
[ "$rc" -eq 0 ] || exit "$rc"
[ -n "$LLAMA_MASTER_PREPARE_ONLY" ] && exit 0
host=127.0.0.1; port=8080
while [ $# -gt 0 ]; do
  case "$1" in --host) host=$2 ;; --port) port=$2 ;; esac
  shift
done
# Absolute paths: an orphan is recognised by its argv (srv.server.ts).
exec "$d/.venv/bin/python" "$d/serve/server.py" --engine strata \\
  --config "$d/strata-$tag.json" --host "$host" --port "$port"
`;
}

const SUDO_GUARD = `#!/bin/sh
echo "llama.master does not run sudo for Strata (asked: sudo $*)." >&2
echo "Install the missing tool from the Prerequisites page, then build again." >&2
exit 1
`;

/** A venv with pip in it. Debian and Ubuntu ship `python3` without
 *  `ensurepip`, where `python3 -m venv` makes an environment that cannot
 *  install anything — so the check is pip, not the exit code. */
async function makeVenv(dir: string, say: (l: string[]) => void) {
  const py = join(dir, ".venv", "bin", "python");
  const hasPip = async () =>
    (await exec(py, ["-m", "pip", "--version"], { timeoutMs: 30_000 }))
      .code === 0;
  if (await exists(py) && await hasPip()) return;
  await Deno.remove(join(dir, ".venv"), { recursive: true }).catch(() => {});
  const std = await exec("python3", ["-m", "venv", ".venv"], { cwd: dir });
  if (std.code === 0 && await hasPip()) {
    say(["Python environment made with python3 -m venv"]);
    return;
  }
  await Deno.remove(join(dir, ".venv"), { recursive: true }).catch(() => {});
  const uv = await which("uv");
  if (uv) {
    const r = await exec(uv, ["venv", "--seed", ".venv"], { cwd: dir });
    if (r.code === 0 && await hasPip()) {
      say(["Python environment made with uv (python3 here has no ensurepip)"]);
      return;
    }
  }
  throw new BuildFailure({
    reason:
      "Strata's server is Python, and this machine cannot make a Python environment: python3 has no venv/ensurepip module and uv is not installed.",
    steps: [
      {
        text:
          "Install uv — one file in your home directory, no root: curl -LsSf https://astral.sh/uv/install.sh | sh",
        action: { kind: "open-url", url: "https://docs.astral.sh/uv/" },
      },
      { text: "Or, with root: sudo apt install python3-venv" },
    ],
  });
}

export async function buildStrata(
  opts: { ref: string; model?: string; signal?: AbortSignal },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = [
    "Fetch source",
    "Python environment",
    "Compile and prepare the model",
    "Verify",
  ];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  if (PLATFORM !== "linux") {
    throw new BuildFailure({
      reason:
        "llama.master builds Strata on Linux only for now. Strata itself also runs on Windows through its own installer.",
      steps: [{
        text: "Strata's own installer.",
        action: { kind: "open-url", url: STRATA_DOCS },
      }],
    });
  }
  const model = strataModel(opts.model ?? "");
  if (!opts.model || !model) {
    throw new BuildFailure({
      reason:
        "Strata is set up for one model, and no Strata-compatible model is selected. It runs Qwen3.8-Flash-Next only: ISTA-DASLab's GSQ-RCO files, or Unsloth's UD-Q4_K_XL / UD-IQ4_XS.",
      steps: [
        { text: "Select that model on the Models page, then build again." },
        {
          text: "Which files Strata runs.",
          action: { kind: "open-url", url: `${STRATA_DOCS}#readme` },
        },
      ],
    });
  }

  const id = buildId("source", opts.ref, "cuda");
  const dir = join(paths().builds, id);
  await ensureDir(dir);
  // Unlisted until it verifies again: a half-updated tree (new Python, old
  // engine) must not be offered as the build that worked.
  await Deno.remove(join(dir, "llama-master.json")).catch(() => {});

  const url = tarballUrl(parseRef(opts.ref));
  p(0, 0, [`Fetching Strata`, url]);
  const bytes = await download(
    url,
    (received, total) => p(0, total ? received / total : null),
    opts.signal,
  );
  const n = await extract(bytes, dir, "tar.gz");
  p(0, 1, [`${n} source files extracted to ${dir}`]);
  opts.signal?.throwIfAborted();

  p(1, null);
  await makeVenv(dir, (lines) => p(1, null, lines));

  const { managedCuda } = await import("./prereq.server.ts");
  const cuda = await managedCuda();
  const launcher = join(dir, STRATA_LAUNCHER);
  await ensureDir(join(dir, ".guard"));
  await Deno.writeTextFile(join(dir, ".guard", "sudo"), SUDO_GUARD);
  await makeExecutable(join(dir, ".guard", "sudo"));
  await Deno.writeTextFile(
    launcher,
    launcherScript(cuda?.nvcc ?? "", strataDataDir()),
  );
  await makeExecutable(launcher);
  p(1, 1, [
    cuda
      ? `Using the CUDA toolkit llama.master installed: ${cuda.path}`
      : "No app-managed CUDA toolkit — Strata will use the nvcc on PATH",
  ]);

  // Setup's own defaults for everything Start decides later (context, reserve,
  // address): the launcher re-runs setup with the real ones, and that re-run
  // only rewrites the config.
  const args = strataArgs({
    modelPath: opts.model,
    model,
    ctx: 65536,
    reserveMib: 700,
    host: "127.0.0.1",
    port: 8080,
  });
  p(2, null, [`$ ${STRATA_LAUNCHER} ${args.join(" ")}`]);
  const code = await execStream(
    launcher,
    args,
    { cwd: dir, env: { LLAMA_MASTER_PREPARE_ONLY: "1" }, signal: opts.signal },
    (line) => {
      // ninja's `[62/135]` is the only honest progress this step has.
      const m = /^\[(\d+)\/(\d+)\]/.exec(line);
      // The draft-layer fetch prints one line per 2%; keep every tenth.
      const pct = /\s(\d+)%$/.exec(line);
      if (pct && Number(pct[1]) % 10 !== 0) return;
      p(2, m ? Number(m[1]) / Number(m[2]) : null, [line]);
    },
  );
  opts.signal?.throwIfAborted();
  if (code !== 0) {
    throw new Error(
      `Strata's setup stopped with code ${code} — its own explanation is in the log above.`,
    );
  }

  p(3, null);
  const config = join(dir, `strata-${model.tag}.json`);
  for (const need of [join(dir, "engine", "strata"), config]) {
    if (!(await exists(need))) {
      throw new Error(
        `Strata's setup finished without producing ${need}, so there is nothing to start.`,
      );
    }
  }
  const build: Build = {
    id,
    ref: opts.ref,
    origin: "source",
    backend: "cuda",
    engine: "strata",
    dir,
    serverBin: launcher,
    cliBin: "",
    createdAt: Date.now(),
    sizeB: await dirSize(dir),
    sourceSha: await movingSha(opts.ref).catch(() => ""),
  };
  await writeMeta(dir, build);
  p(3, 1, [`Strata is ready for ${model.model} at ${dir}`]);
  return build;
}
