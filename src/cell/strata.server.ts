// src/cell/strata.server.ts — install the Strata engine as a build. SERVER ONLY.
//
// Strata is not one binary: it is a Python server, an engine (a ready-made
// CUDA package, or one compiled here) and a setup script that prepares ONE
// model (an index of the GGUF, the draft layer, a config). So a Strata build
// is its whole checkout under the builds root, and what the app spawns is a
// launcher it writes there itself (`STRATA_LAUNCHER`): run Strata's `setup.py`
// for the model — a no-op once nothing changed — then `exec` its server. One
// process, one pid, one log, and the argv on screen is the argv that ran.
//
// What Strata's installer would do on its own, and must not here:
//   - `sudo apt-get install` Python or the CUDA toolkit. The launcher puts a
//     `sudo` that REFUSES first on PATH, so no path through setup.py can reach
//     the real one, and `STRATA_NVCC` names the toolkit this app manages
//     (`prereq.server.ts:managedCuda`) — the same nvcc a llama.cpp build uses.
//   - download the model. `--gguf-dir` (`src/lib/strata.ts`) gives it the
//     files on disk; the Flash-Next packs' ~6.5 GB draft layer is the one
//     model fetch left. The image encoder is off (`--vision no`), because this
//     app's chat sends text only.
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

/** A POSIX single-quoted word, so a path with a quote or space stays one. */
function shQuote(s: string): string {
  return `'${s.replaceAll("'", "'\\''")}'`;
}

/** Where setup.py, its venv, the run configs and the server live: the
 *  checkout's `terminal/` since Strata 0.2.16, its root before. */
async function strataWorkDir(dir: string): Promise<string> {
  const t = join(dir, "terminal");
  return await exists(join(t, "setup.py")) ? t : dir;
}

function launcherScript(
  nvcc: string,
  dataDir: string,
  sub: string,
): string {
  return `#!/bin/sh
# Written by llama.master (src/cell/strata.server.ts) — do not edit; a rebuild
# replaces it. Usage: ${STRATA_LAUNCHER} <config-tag> <setup.py arguments…>
r=$(dirname "$(readlink -f "$0")"); d="$r"${sub ? `/${sub}` : ""}
cd "$d" || exit 1
tag=$1; shift
# No sudo, ever: setup.py offers to apt-get install Python and CUDA.
PATH="$r/.guard:$PATH"; export PATH
${nvcc ? `STRATA_NVCC=${shQuote(nvcc)}; export STRATA_NVCC` : "# system nvcc"}
# Ready-made engines (Inkling's strata-inkling among them) are components:
# kept with the packs, not in ~/.local/share, so one folder holds all of it.
STRATA_COMPONENTS_DIR=${shQuote(join(dataDir, "components"))}
export STRATA_COMPONENTS_DIR
# Packs and the draft layer live outside the build, so an engine update does
# not fetch ~6.5 GB again.
.venv/bin/python setup.py --setup --no-start --yes --no-browser \\
  --data-dir ${shQuote(dataDir)} "$@" </dev/null &
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
export async function makeVenv(
  dir: string,
  say: (l: string[]) => void,
  who = "Strata",
) {
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
      `${who}'s server is Python, and this machine cannot make a Python environment: python3 has no venv/ensurepip module and uv is not installed.`,
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

type StrataConfig = {
  exe?: unknown;
  args?: unknown;
  cwd?: unknown;
  model_name?: unknown;
};

/** The config setup wrote for this model. The tag is stable, but a setup
 *  update may change it; the `--native` file in the args is the identity. */
async function findConfig(
  dir: string,
  tag: string,
  modelPath: string,
): Promise<string> {
  const expected = join(dir, `strata-${tag}.json`);
  if (await exists(expected)) return expected;
  const file = modelPath.split(/[\\/]/).pop() ?? "";
  try {
    for await (const e of Deno.readDir(dir)) {
      if (
        !e.isFile || !e.name.startsWith("strata-") || !e.name.endsWith(".json")
      ) continue;
      const path = join(dir, e.name);
      try {
        const cfg = JSON.parse(
          await Deno.readTextFile(path),
        ) as StrataConfig;
        const args = Array.isArray(cfg.args) ? cfg.args.map(String) : [];
        if (file && args.some((x) => x.endsWith(file))) return path;
      } catch {
        // A file beside the configs that is not JSON: not one of ours.
      }
    }
  } catch {
    // No directory: the caller reports the missing expected config.
  }
  return expected;
}

/** What `configEngine` answers for a `component:` engine (Inkling's
 *  `component:strata-inkling`): setup fetched it from Strata's signed feed and
 *  test-started it, and the server resolves it — there is no path to check. */
const COMPONENT = "component";

/** The engine file a config names, `COMPONENT`, or "" when unreadable. */
async function configEngine(config: string, dir: string): Promise<string> {
  try {
    const cfg = JSON.parse(await Deno.readTextFile(config)) as StrataConfig;
    const exe = typeof cfg.exe === "string" ? cfg.exe : "";
    if (exe.startsWith("component:")) return COMPONENT;
    if (!exe) return "";
    if (exe.startsWith("/")) return exe;
    const cwd = typeof cfg.cwd === "string" && cfg.cwd ? cfg.cwd : dir;
    return join(cwd.startsWith("/") ? cwd : join(dir, cwd), exe);
  } catch {
    return "";
  }
}

export async function buildStrata(
  opts: {
    ref: string;
    model?: string;
    /** The Build page's backend. Strata runs on CUDA here; a HIP build would
     *  need its own launcher and toolchain and is not wired yet. */
    backend?: string;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = [
    "Fetch source",
    "Python environment",
    "Build and prepare the model",
    "Verify",
  ];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  if (PLATFORM !== "linux") {
    throw new BuildFailure({
      reason:
        "llama.master builds Strata on Linux only for now. Strata itself also runs on Windows and macOS through its own installer.",
      steps: [{
        text: "Strata's own installer.",
        action: { kind: "open-url", url: STRATA_DOCS },
      }],
    });
  }
  if (opts.backend && opts.backend !== "cuda") {
    throw new BuildFailure({
      reason:
        `Strata builds in llama.master are CUDA (NVIDIA) builds only for now; the Build page has ${opts.backend} selected. Strata itself also runs on AMD through HIP.`,
      steps: [
        {
          text: "Choose the CUDA backend on the Build page, then build again.",
          action: { kind: "open-tab", tab: "build" },
        },
        {
          text: "Or use Strata's own installer for the AMD card.",
          action: { kind: "open-url", url: STRATA_DOCS },
        },
      ],
    });
  }
  const model = strataModel(opts.model ?? "");
  if (!opts.model || !model) {
    throw new BuildFailure({
      reason:
        "Strata is set up for one model family, and no Strata-compatible model is selected. It runs Qwen3.8-Flash-Next (ISTA-DASLab's GSQ-RCO sizes, Swift 1.5, the Coder, Unsloth's UD-IQ4_XS / UD-Q4 / Q5 / Q6_K_XL / Q8_0, and Gyro-S / Gyro-M) and Qwen3.6-35B-A3B.",
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

  // 0.2.16 moved setup, its venv and the server into `terminal/`. An update
  // from an older tree carries its venv along, as Strata's own setup.sh does.
  const work = await strataWorkDir(dir);
  if (
    work !== dir && await exists(join(dir, ".venv")) &&
    !(await exists(join(work, ".venv")))
  ) {
    await Deno.rename(join(dir, ".venv"), join(work, ".venv"));
  }
  p(1, null);
  await makeVenv(work, (lines) => p(1, null, lines));

  const { managedCuda } = await import("./prereq.server.ts");
  const cuda = await managedCuda();
  const launcher = join(dir, STRATA_LAUNCHER);
  await ensureDir(join(dir, ".guard"));
  await Deno.writeTextFile(join(dir, ".guard", "sudo"), SUDO_GUARD);
  await makeExecutable(join(dir, ".guard", "sudo"));
  await Deno.writeTextFile(
    launcher,
    launcherScript(
      cuda?.nvcc ?? "",
      strataDataDir(),
      work === dir ? "" : "terminal",
    ),
  );
  await makeExecutable(launcher);
  p(1, 1, [
    cuda
      ? `CUDA toolkit for a source build: ${cuda.path}`
      : "No app-managed CUDA toolkit — a source build will use the nvcc on PATH",
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
  // Verify what setup WROTE, not a hardcoded path: a CUDA 12 card gets
  // engine-cuda12/strata, and the config is the one thing that says which
  // engine and args the server will start.
  const config = await findConfig(work, model.tag, opts.model);
  if (!(await exists(config))) {
    throw new Error(
      `Strata's setup finished without writing a run config for ${model.model} (${config}), so there is nothing to start.`,
    );
  }
  // The launcher runs `strata-<tag>.json` and nothing else: a config found
  // under another name is one Start would never read.
  const expected = join(work, `strata-${model.tag}.json`);
  if (config !== expected) {
    throw new Error(
      `Strata's setup wrote the run config for ${model.model} as ${config}, but Start runs ${expected}. This Strata names its configs differently from what this app knows; update llama.master, or build the Strata version it was made for.`,
    );
  }
  const engine = await configEngine(config, work);
  if (engine === COMPONENT) {
    p(3, null, [
      "The engine is a Strata component, fetched, verified and test-started by Strata's setup",
    ]);
  } else if (!(await exists(engine))) {
    throw new Error(
      `Strata's setup wrote ${config}, but the engine it names does not exist (${engine}), so there is nothing to start.`,
    );
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
