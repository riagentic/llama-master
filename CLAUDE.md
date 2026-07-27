# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

`llama.master` — a local Electron desktop app for
[llama.cpp](https://github.com/ggml-org/llama.cpp): acquire it (prebuilt release
or source build), find GGUF models, tune flags against a live VRAM/RAM plan, run
`llama-server`, chat with it. Product spec (source of truth): `.katana/app.md`.
Framework rules: `.katana/_aio.md`. Universal rules: `.katana/_universal.md`.

## Stack

- **Deno 2.9+ + aio `1.0.0-alpha38`**, vendored at `dep/aio` → symlink to
  `../../aio`. Never `npm`/`node`. aio internals: `dep/aio/CLAUDE.md`; docs
  index: `dep/aio/docs/content.md`.
- JSX via `jsxImportSource: "aio"` (`class=`, not `className`); state via
  `cell({ state, methods })`; persistence is automatic SQLite in
  `~/.llama-master/data/`.
- Rust crate `llama-sys` in `rust/`, compiled to `wasm32-unknown-unknown`,
  staged as the committed `src/llama-sys.wasm`.

## Commands

- `deno task dev` — Electron · `dev:browser` — browser client
- `deno task test` — all tests · single: `deno test -A tests/lib.test.ts` · one
  case: `--filter "name"`
- `deno task test:rust` — the Rust core (`cargo test`)
- `deno task wasm` — rebuild `src/llama-sys.wasm` (runs `cargo test` first).
  **Required after any `rust/**.rs` edit** — a guard test fails if the artifact
  is older than the source.
- `deno task aiol` — the aio framework linter
- `deno task am <cmd>` — the live app: `am state '<dot-path>'`, `am dispatch`,
  `am surface 0`, `am trigger 0 "<path>" click`. This is the debugging tool;
  reach for it before curl.
- `deno task verify` — the pre-merge gate: `fmt --check` → `lint` → `check` →
  `test`. Plus `deno task test:rust` when Rust changed.

## Architecture

Three layers, and the boundaries between them are load-bearing.

**`src/lib/` — pure.** The parameter catalog, the command builder, the memory
planner, the tuner, the archive readers, the SSE parser. No I/O, no DOM, no
clock. Every decision the app makes lives here and is unit-tested here. If you
are about to put a rule in a component or a cell, it probably belongs here.

**`src/cell/` — state.** One cell per concern (`hw`, `prereq`, `builds`,
`models`, `cfg`, `srv`, `chat`, `ui`). Cell modules are imported by the browser,
so they must stay browser-safe. All host access lives in a sibling
`*.server.ts`, reached **only** by `await import()` inside an async method — a
static import of one blank-screens the app with `Deno is not defined`
(`dep/aio/docs/build/imports.md`). A guard test enforces this.

**`src/ui/` — presentation.** Components take props and emit intent.
`src/ui/derive.ts` holds every value the UI derives from cell state (as property
reads — see the selector rule below), and `src/ui/actions.ts` the few gestures
that span cells.

Data flow worth knowing:

- **One catalog drives everything.** `src/lib/params.ts` is the only place a
  llama.cpp flag is declared; the settings UI renders from it, `command.ts`
  emits from it, `tune.ts` writes into it. Adding a flag anywhere else is a bug.
- **What you see is what runs.** The UI composes argv with `command.ts` and
  hands that exact array to `srv.start`. There is no second code path.
  `srv.server.ts` refuses any binary outside
  `~/.llama-master/data/files/builds/`.
- **The process lives in `srv.server.ts`; the cell is its shadow.** `srv.poll`
  (1 s schedule) is the only writer of liveness, so "running" always means the
  pid is alive and `/health` answered.
- **A source build is gated on the backend's own toolchain**
  (`src/lib/backend.ts`), not just cmake + compiler: `-DGGML_CUDA=ON` without
  `nvcc` fails minutes into cmake configure, so the Build tab refuses up front
  and names the tool.
- **A CUDA build must be told which architectures to target**
  (`src/lib/cuda.ts`). Left to cmake's auto-detection, an nvcc older than the
  GPU dies with `nvcc fatal: Unsupported gpu architecture` several minutes into
  the compile — measured with CUDA 12.0 against a Blackwell card (sm_120). The
  planner caps to PTX for the newest architecture nvcc knows (`90-virtual`),
  which the driver JIT-compiles forward; verified building AND running on that
  card.
- **The Vulkan backend needs `glslc` AND SPIRV-Headers** — and cmake lies about
  which. It prints "missing components: glslangValidator" (informational;
  llama.cpp asks for `glslc` only) on the same run that fails at
  `find_package(SPIRV-Headers CONFIG REQUIRED)`, ggml-vulkan/CMakeLists.txt:14.
  Following the visible message installs the wrong package. SPIRV-Headers is a
  header tree, so the app downloads and installs it itself with no root
  (`installSpirvHeaders`), and the build passes both `-DCMAKE_PREFIX_PATH` and
  `-isystem <prefix>/include` — `find_package` alone is not enough, because
  llama.cpp does not link the imported target and the compile then fails with
  "'spv' has not been declared".
- **GitHub's API is 60 requests/hour anonymous, and it WILL run out.** Every API
  call goes through `fetchJson`, which turns a quota 403 into a `RateLimited`
  error naming the reset time and `GITHUB_TOKEN`; `builds.server.ts` then falls
  back to plain github.com pages (`/releases/latest` redirect for the tag,
  `/releases/expanded_assets/<tag>` for the asset list, `releases.atom` for
  tags) which are not rate limited. Downloads never were. Verified against a
  genuinely exhausted quota (`src/lib/github.ts`).
- **"Fix" installs prerequisites, and never silently.** `src/lib/fixplan.ts`
  returns one of three honest outcomes — `download` (the app does it itself),
  `package` (the exact command, elevated via `pkexec`, shown on the button
  before it runs), or `manual` (with the reason). Nothing privileged runs
  unexplained, and when nothing can elevate the failure names the command to run
  by hand. ROCm is a `script` plan built from AMD's own Ubuntu-noble procedure,
  keyed on `UBUNTU_CODENAME` so derivatives (Mint, Pop!_OS) get it too; every
  step is shown before it runs and a docs link is always present.
- **Models come from three kinds of store.** Plain `.gguf` trees (including LM
  Studio's), and ollama — which keeps no `.gguf` at all: weights live in
  `blobs/sha256-<hex>` and are only reachable through the JSON manifests
  (`src/lib/ollama.ts`). A cloud-only ollama entry has `"layers": null` and must
  not be listed; there is nothing to load.
- **The tuner is constrained by RAM as well as VRAM.** `fitsVram` was the only
  fit test, so a plan that consumed every byte of `MemAvailable` was emitted
  with a warning after the fact — and weights and KV cache are anonymous pages
  the kernel cannot reclaim, so that is the OOM killer, not slowness.
  `ramMarginB` (1 GiB or 10%) is a real constraint, and when a placement would
  starve the OS the tuner halves the context and re-plans rather than narrating
  the problem (`src/lib/tune.ts`).
- **Which flags are loadable depends on the backend, so `Hw` carries it.**
  `tune` used to set `-fa on` and `-ctk/-ctv q8_0` for everyone; on a backend
  with no quantised-KV kernel that is a server that will not start, i.e.
  "optimal settings" that fail. Only CUDA and Metal are offered a quantised
  cache (`QUANT_KV_BACKENDS`); elsewhere flash attention is left on `auto` and
  the reason says why. The cache type is also reset to the default on every run,
  so a `q8_0` chosen for one model is never inherited by the next.
- **A MoE model keeps its experts in RAM even when layers must move too.** The
  partial-offload branch used to reset `--n-cpu-moe` to 0, discarding the
  strategy exactly when it is worth most: a Mixtral-shaped layer is ~40 MB of
  attention against ~720 MB of experts, so holding the experts back buys ~16x
  more layers. Measured on a 3 GB card: 2 of 32 layers before, 32 of 32 after.
- **KV-cache size is per-architecture.** One uniform formula overestimated
  Gemma-3-class sliding-window attention ~3.7x and DeepSeek MLA ~71x, while the
  UI labelled the figure exact. `rust/src/gguf.rs` reads
  `attention.sliding_window{,_pattern}` and `attention.kv_lora_rank`;
  `plan.ts:kvTotal` caps windowed layers at their window and bills MLA as one
  compressed latent per layer. `kvPerToken` remains the per-token rate the UI
  shows — a windowed layer has no constant rate, which is why the fit uses
  `kvTotal`.
- **Every append-at-the-bottom box follows its newest line.** Both chats and
  every `LogView` are capped scroll containers; with no scroll handling a reply
  stayed below the fold exactly when the user was waiting for it. One hook
  (`src/ui/sticky.ts`) over one pure policy (`src/lib/scroll.ts`): an arrival
  forces the scroll, a streamed token only follows when the reader is already at
  the bottom. `tests/guards.test.ts` fails if a new surface forgets.
- **Memory numbers are exact, not estimated.** `rust/src/gguf.rs` walks the
  tensor table and reports per-layer bytes with routed experts separated; only
  the compute buffer is an estimate, and it is labelled as one everywhere.

## Never a raw error

Two files carry this, and it is the app's main promise:

- **`src/lib/backend.ts:targetReadiness`** answers "will the route+backend the
  user has SELECTED produce a build?" — for both routes, before the button is
  enabled. The prerequisites panel answers a different question ("is this
  machine equipped to compile"), and green ticks followed by a failure was a
  real user report. The release route auto-fetches the asset list so the answer
  is never a guess; unknown is reported as _pending_, never as ready.
- **`src/lib/diagnose.ts`** turns every failure into `{ reason, steps[] }` where
  a step may carry an ACTION, so `src/ui/Guidance.tsx` renders a button that
  performs it. A list of twenty-seven asset filenames is not an error report.
  Known signatures (nvcc arch, SPIRV headers, glslc, no compiler, OOM, disk
  full, rate limit) map to their real cause; an unrecognised failure still gets
  the other route and a prerequisite re-check.

When adding a way to fail, add its signature and steps. A message the user
cannot act on is a bug.

## Rules that bite

- **Read cell properties, not selectors, from a component.** Until aio alpha38 a
  selector call registered **no** reactive dependency and the component silently
  went stale; that is fixed (verified), so this is now a convention rather than
  a correctness rule. It is still enforced by `tests/guards.test.ts`, because
  `src/ui/derive.ts` being the one list of every derived value is worth keeping.
- **Prefer mutating the draft over assigning a spread of state back.**
  `s.job = { ...s.job, step }` was once rejected wholesale at runtime
  (`preventExtensions on proxy`), silently discarding the action — it shipped,
  froze the build panel, and no in-process test caught it. aio alpha38 removed
  that restriction (verified), so this is now style: mutating in place says
  which field changed. Still guarded in `tests/guards.test.ts`, and
  `tests/runtime.test.ts` covers the callback paths.
- **Harness parity, as of aio alpha38.** The in-process harnesses used to be
  more permissive than production — they ignored `own` effects entirely and
  missed the proxy guard — so a green suite did not mean a working app. Both are
  fixed. `am` against the running app is still the fastest way to confirm
  anything involving a real process, but it is no longer compensating for the
  harness.
- **An `own` effect must name the resource it owns.** `own.set(key, …)` with a
  key already in use **disposes the previous effect** — so a teardown written as
  "stop the server" stops whatever is running _now_, not the process that effect
  was created for. This shipped: after any crash, the next Start came up and was
  SIGTERMed a moment later (`exited with code 143`), and the app looked like it
  simply could not start a server. The close must be conditional on identity
  (`io.stopOwned(pid)`). `testUI`/`bootCells` print _"own effects are ignored in
  standalone/test mode"_ and stay green — only `testServer` reproduces it, and
  only when the previous run **crashed** rather than being stopped (`stop`
  disposes the effect cleanly). Pinned by `tests/runtime.test.ts`.
- **"The log below" must actually be below.** Every diagnosis this app writes
  points at the log; a page that shows a diagnosis therefore renders
  `ServerLog`/`LogView` itself rather than pointing at another tab.
- **In a `testUI`, mount the panel — not `App` plus navigation.** `ui.tab` is
  persisted, and its rehydration on mount is async: a `ui.go(…)` (or a rail
  click) issued before it lands is silently reverted to whatever the previous
  test left in the store. That is a ~40% flake, not a rare one. Rendering
  `ServerPanel`/`OnePage` directly tests the same thing deterministically. The
  rail itself has one test, and that is where navigation belongs.
- **A test that writes app files must set `LLAMA_MASTER_HOME` first.**
  `testServer` redirects the app home; **`bootCells` and `testUI` do not**, and
  aio exports no `registerAppDirs`. So `paths()` honours `LLAMA_MASTER_HOME`
  (`src/cell/host.server.ts`) and every test that installs a fixture build sets
  it to a temp dir **before** the first `paths()` call. This shipped wrong: the
  server tests left a `test-build/` directory inside the developer's real
  `~/.llama-master`, and two UI tests silently passed only because that home had
  an app-managed CMake in it. `AIO_DATA_DIR` looks like the override and is not
  one.
- **Drain the child's output before reporting that it exited.** `child.status`
  resolves when the process is reaped, which can beat the last of its stderr
  through the pipe — and `srv.poll` diagnoses an exit from exactly those lines.
  Reporting "not running" early replaced "cudaMalloc failed: out of memory" with
  the generic fallback on precisely the failures that happen fastest.
  `srv.server.ts` awaits both pumps first.
- **Fail loud.** A missing binary, an unreadable header, a 404 — surface it in
  `lastError` and render it. Never swallow.
- **`.slice()`, not spread**, on live async state arrays.
- Tests live in `tests/`, never beside their source.

## Katas

`.katana/*.md` are the quality specs; `/use-katana` audits against them. Field
reports on the framework go to `dep/aio/feedback/llama-master.md` — that file is
required by `.katana/_aio.md` and should be updated whenever aio gets in the
way.
