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

- **Deno 2.9+ + aio `1.0.19-beta`**, vendored at `dep/aio` → symlink to the
  provisioned release (`~/.local/lib/aio-versions/v1.0.19-beta`). Never
  `npm`/`node`. aio internals: `dep/aio/CLAUDE.md`; docs index:
  `dep/aio/docs/content.md`. The pin in `deno.json` (`aioVersion`) must name the
  version the symlink actually resolves to — `deno task aiol` says so when they
  drift, and a declaration that lags the code is how the app came to be running
  alpha54 while claiming alpha44. Change it with `am pin <version>`, never by
  hand; `am pin <path>` writes a gitignored `.aio/pin.local` that follows a
  framework checkout instead, which is for developing against a WIP aio and
  overrides the committed pin silently — `deno task am pin` prints which one is
  in force.
- **Async cells carry `transaction: false`.** alpha52 made `transaction: true`
  (snapshot reads, atomic commit at return) the async default; every cell here
  was written for incremental commits — `srv.poll` is an observer that must read
  state that moved under it, `chat.send` streams — so they opt out explicitly.
  Adopting transactions is a per-cell decision (`s.$commit()` to publish
  mid-method, `s.$live` to read past a pinned snapshot), not a default to drift
  into. Effects go through `s.$do(effect)`; returning them is deprecated.
- **The surface is frozen at beta.** alpha75 → alpha76 was the last release that
  broke anything, and this app carried none of the six retired spellings (no
  returned effects, no spread selector deps, no `killExisting`, no `--zero-port`
  / `--backup-logs` / bare `--server-url`), so the move was a pin change and
  nothing else. `am pin` refuses an unfixed app; it also refused this one over
  three FALSE positives — `perfBudget: { reduce: 100 }` read as the cell key
  `reduce:` removed in alpha27, and a `const machine: Hw = {` annotation read as
  a `machine:` key — so `--force` was the right answer and the finding went to
  the framework's feedback file. Anything that breaks from here is a bug in aio,
  not a step this app missed.
  - **beta → 1.0.9-beta (2026-09-23) was the same: a pin change.** The suite was
    green before any app edit. `am pin` still refused over
    `perfBudget: { reduce: 100 }` (1.0.1-beta fixed the `machine:` case only),
    so `--force` again. `am fix` moved Electron to aio's tested `44.4.1` in BOTH
    apps — but run inside `client/` it also pinned `v1.0.8-beta` and made a
    `client/dep/aio` link nothing imports; the client resolves aio through
    `../dep/aio` and must carry neither. Both went to the feedback file.
  - **1.0.9-beta → 1.0.19-beta (2026-10-08): a pin change again, and `am pin` no
    longer refuses.** No `--force` — the `perfBudget: { reduce: 100 }` false
    positive is gone. The pin moved Electron to `44.5.1`, esbuild to `0.25.12`
    and happy-dom to `20.14.5` in the app; the CLIENT's three were matched by
    hand in `client/deno.json` (no `am fix` there — see above). Both suites
    green, `am migrations` reports no drift.
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
  `am surface 0`, `am trigger 0 "<path>" click`, `am eval '<expr>'` (geometry
  and computed styles, 1.0.0-beta), `am migrations` (declared vs stored cell
  versions + shape drift — the check after any `version:` bump), `am expect`,
  `am timeline`. This is the debugging tool; reach for it before curl.
- **The app's LIFECYCLE is `am` too: `am start` / `am stop` / `am restart`, from
  this directory.** They are scoped to THIS project and go through the lock.
  Never `pkill` an aio app: every aio app on the machine is
  `deno run -A src/app.ts`, so a pattern that looks specific matches all of them
  — `pkill -f "deno run -A src/app.ts"` killed two of the developer's unrelated
  apps mid-session. `am instances` lists what is running and is the thing to
  check first. `am kill --stale` reaps orphans without touching anything that
  still holds its lock.
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
  `~/.llama-master/data/files/builds/`. It is SHOWN by one component
  (`src/ui/CommandView.tsx`) on the three pages that can change it — all-in-one
  (under Memory), Tune, Server — rather than by a footer strip pinned under
  every tab: that strip was the one part of the app that was not a panel and it
  read as one, taking a band of height from every page for two lines that
  wrapped anyway. It draws in the chat's `.codeblock` shape because it is the
  same kind of object (a named block of text with a button that takes it), it
  composes from `shownSettings()` so a RUNNING server shows the argv it was
  started with, and the copy button copies ONE line while the page shows the
  `\`-broken form — what lands in a shell has to be a command.
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
  back to plain github.com pages (`releases.atom` for tags,
  `/releases/expanded_assets/<tag>` for the asset list) which are not rate
  limited. Downloads never were. Verified against a genuinely exhausted quota
  (`src/lib/github.ts`).
- **Neither `/tags` nor `/releases/latest` names a llama.cpp build any more.**
  In August 2026 upstream started tagging semver milestones (`v0.3.0`) and
  per-commit `master-<sha>` snapshots, and marked the nightly `b<number>` builds
  prerelease. `/tags` sorts by NAME, so its first hundred entries contain zero
  builds; `/releases/latest` skips prereleases, so it answers the milestone —
  which is days STALE and ships no binaries (one `nightly-tag.txt`), i.e. an
  "update" that moves backwards and then cannot download. `builds.server.ts`
  therefore reads the RELEASES list (publish-date order, prereleases included)
  filtered to `BUILD_TAG` (`/^b\d+$/`), and the release route's "master"
  resolves to the newest build release that actually carries assets. Measured
  live 2026-08-29; the Linux/macOS assets also changed `.zip` → `.tar.gz` around
  the same time, which `assets.ts`/`archive.ts` already handled.
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
- **Memory moves under the app, so the plan adapts — coarsely on purpose.** This
  runs on workstations where a game takes 20 GB of VRAM, another tool loads a
  model, or a compile finishes and gives 8 GB of RAM back. Every one of those
  changes the right answer, in **both** directions and in **both** pools, so
  `src/lib/adapt.ts` drives three things. (1) `headroomKey` — eighths of each
  pool — is part of the auto-tune key in `OnePage`, so a real change re-tunes
  and 200 MB of jitter does not; keying on `availableB` itself would rewrite the
  user's settings on every 1 s poll and fight their typing. (2) The SAFETY
  reserve is FIXED (`tune.ts:marginB`/`ramMarginB`) — it was briefly widened by
  observed memory "churn", and that was removed on purpose: the only churn
  signal available is the device-wide usage series, our own llama-server is
  inside it, so loading a 39 GB model registered as 39 GB of volatility and the
  app refused models that fit. A reserve driven by a signal that cannot separate
  our allocation from everyone else's produces false refusals. (3) `drift` — a
  loaded model cannot be re-placed, so while a server runs the app does not
  re-tune, it TELLS you: squeezed (someone took memory this run depends on) or
  roomier (enough came back that a restart would get more), each with the
  restart button. Roomier is measured against the free memory recorded at the
  moment the run was spawned (`srv.startFreeVramB/RamB`) — without that baseline
  it fired forever on any machine that simply had headroom.
- **The user's reserve is a second reserve, and a different kind of thing.**
  `marginB` above exists so the ALLOCATOR does not fail and the user never sees
  it; the reserve (`src/lib/reserve.ts`, `src/ui/ReserveControls.tsx`, on both
  the all-in-one and Tune pages) is the user saying "that card also draws my
  desktop", because the tuner filling a display card to the last byte is a
  driver reset mid-generation, and a host pool run to the edge is the OOM killer
  picking llama-server's neighbour. It is honoured by planning as if the memory
  were ABSENT, so it enters through `Hw` (`types.ts:Reserve`, attached only in
  `derive.ts:planningHw`) and every consumer of `plan` — the tuner, the picker,
  stability, the bars, the per-card packing budgets — inherits it without
  knowing it exists.
  - **Three numbers, because the display is on ONE card.** Reserved per GPU
    (default 0) is charged to every card; reserved on the connected GPU (default
    8 GB) only to the card(s) with a monitor attached; reserved RAM (16 GB) is
    the host pool. A single machine-wide VRAM figure divided across the cards
    was the first design and it is wrong in both directions: it took memory from
    a headless compute card to defend a desktop that is not on it, and it left
    the display card with a fraction of what a compositor + browser + game
    actually need. Applying one figure to every card is the other error — a
    two-card machine paying twice for one desktop.
  - **Which card has the display is MEASURED, and "unknown" is a third answer.**
    `Gpu.display` comes from
    `nvidia-smi --query-gpu=display_mode,display_active` (cached 30 s — it
    changes when a monitor is plugged in, not on the 1 s poll; `display_mode` is
    deprecated on current drivers and returns a sentence, hence reading both)
    and, for sysfs cards, `/sys/class/drm/<card>-*/status`. Verified against
    this machine: nvidia-smi index 0 = PCI 01:00.0 = `card2`, which is the one
    with two connected DisplayPort outputs. `undefined` (a vendor that does not
    report, no DRM connectors) falls back to card 0 and the UI says it is an
    assumption; every card answering `false` is taken at its word and reserves
    nothing (`reserve.ts:displayGpus`).
  - It is labelled apart from "in use elsewhere" everywhere it is shown
    (`Pool.reservedB`), because a refusal caused by the user's own setting has
    to name the control that gives the memory back. In the memory MAP that means
    a band with a colour of its own (teal, `--seg-reserved`, still hatched
    because a decision must not look like a measurement) and an entry in the
    legend: it was drawn all along in the machine's greys and named nowhere, so
    the one band the user put there themselves read as empty track — which is
    the one thing it is not. The region foot counts it as a third figure,
    because reserved bytes are neither used (nothing is in them) nor free
    (nothing may go in them). `hwSnapshot` never carries it — the current-state
    view reports real free memory; reserved bytes are free until something takes
    them, they are merely not SPENDABLE.
- **Of the four context bands, only Max is a fact.** `.katana/context.md` asks
  for Min / Opt / Big / Max buttons and a picture of the usable range
  (`src/lib/tune.ts:ctxBands`, `src/ui/CtxControls.tsx`, on both the all-in-one
  and Tune pages). `max` is `nCtxTrain` — the length the model was trained for,
  read from the header, and the real edge because RoPE extrapolates past it.
  `opt` (¼) and `big` (½) are **estimates**: a GGUF header carries no quality
  signal at all, while published long-context suites consistently find effective
  length well under the advertised one. So they are marked `≈` on every button
  and the range explains itself in words — the same honesty the compute-buffer
  estimate gets. The auto-tuner still aims at `max`; quietly quartering
  everyone's context on the strength of a chosen fraction would be worse than
  the gap. The only way to make Opt and Big facts is to probe THIS model at THIS
  quantisation with a needle-in-a-haystack run and cache it — the app owns a
  server, a chat client and an SSE parser, so it is equipped to, and does not
  yet.
- **KV-cache size is per-architecture.** One uniform formula overestimated
  Gemma-3-class sliding-window attention ~3.7x, DeepSeek MLA ~71x, and
  Qwen3.5-class hybrid linear attention ~3.8x, while the UI labelled the figure
  exact. `rust/src/gguf.rs` reads `attention.sliding_window{,_pattern}`,
  `attention.kv_lora_rank`, and `full_attention_interval` + `ssm.*`;
  `plan.ts:kvTotal` caps windowed layers at their window, bills MLA as one
  compressed latent per layer, and bills a hybrid model only for its KV-bearing
  layers (`kvLayers` — every Nth trunk layer plus the MTP block, llama.cpp's
  `is_recr_impl` rule) while the recurrent layers pay a constant per-sequence
  f32 state (`recurrentStateB`, ~157 MB on Qwen3.8-27B, one copy per server
  slot, untouched by `-ctk`). This was found the expensive way: the tuner
  refused Qwen3.8-27B a 131,072 context on 48 GB of VRAM over an 18.5 GB q8
  cache that is really 4.9 GB — the model runs VRAM-only at its full 262,144 (17
  of 65 declared layers hold a cache; verified against a live run's measured
  footprint). `kvPerToken` remains the per-token rate the UI shows — a windowed
  layer has no constant rate, which is why the fit uses `kvTotal`.
- **Every append-at-the-bottom box follows its newest line.** Both chats and
  every `LogView` are capped scroll containers; with no scroll handling a reply
  stayed below the fold exactly when the user was waiting for it. One hook
  (`src/ui/sticky.ts`) over one pure policy (`src/lib/scroll.ts`): an arrival
  forces the scroll, a streamed token only follows when the reader is already at
  the bottom. `tests/guards.test.ts` fails if a new surface forgets.
- **A reply is blocks, not a string — and both chats draw it the same way.**
  What a local model answers with most is code and file contents, and one
  pre-wrapped string made the fences visible, the indentation fight the prose,
  and taking a file a drag-select that caught the ``` at both ends.
  `src/lib/richtext.ts` splits a reply into text and fenced blocks (pure: the
  fence rules have edge cases — a longer fence containing a shorter one, an
  indented fence, and above all an UNCLOSED one, which is every block while it
  is still streaming and must render as a block rather than reflowing into one
  the second the model finishes). Every common spelling of the info string is
  read, because there is no standard and a model trained on all of them emits
  all of them: `ts`, `src/lib/plan.ts`, `ts src/lib/plan.ts`, `ts:src/…`,
  `title="…"`. The block header names the FILE over the language, and carries
  its own copy button — the file is the unit people want, not the message.
  `src/ui/ChatMessage.tsx` is the one renderer for both surfaces: they had
  drifted (tok/s above the answer on one, below on the other; the "ended while
  still thinking" fallback on only one), which is what two copies of a message
  renderer always do. tok/s belongs AFTER the answer — it is a measurement of
  the thing above it, and printing it in the role line put a number the user
  cannot have yet over the text they are waiting for.
- **One Memory section, and what is in it depends on whether the answer is
  already known.** Nothing running: both maps, now and next, because that IS the
  decision. A model running: only the measurement — the projection's question
  has been answered by the machine itself, and an estimate beside the
  measurement of the same thing asks the reader which one to believe. What a
  restart would cost is still on screen, in the placement picker and the fit
  line (`src/ui/OnePage.tsx`, keyed on `memoryIsLive()`).
- **Memory numbers are exact, not estimated.** `rust/src/gguf.rs` walks the
  tensor table and reports per-layer bytes with routed experts separated; only
  the compute buffer is an estimate, and it is labelled as one everywhere.
- **A split model must be read as a model, not as its first part.** Anything
  past ~40 GB ships as `-00001-of-000NN.gguf`, and the tensor table is DIVIDED
  across the parts, not repeated: part 1 of DeepSeek-V4-Flash parses perfectly
  and describes 38 tensors and 37 GB, of the 1,328 tensors and 145 GB on disk.
  Nothing downstream can tell — the planner sized it at a quarter, VRAM-only
  looked possible on 48 GB of cards, the tuner proposed it, and llama-server was
  OOM-killed loading the other 108 GB. `src/lib/shards.ts` merges every part and
  `readModel` refuses to return a merge that did not see all of them (a partial
  set is not a smaller model; llama.cpp cannot load it either). Two independent
  completeness checks, because they catch different failures: the header's own
  tensor count across all parts, and the summed on-disk size for parts that are
  present but truncated. Parts 2..N carry THREE metadata keys and no
  `block_count`, so the Rust layer table grows to fit the tensors it actually
  sees — sizing it from that header filed 107 GB of routed experts as the output
  head, where neither `-ngl` nor `--n-cpu-moe` can place them.

- **A second GPU is not a bigger GPU.** llama.cpp offloads a contiguous run of
  layers and cuts it into per-device ranges **by count** — `--tensor-split`, or
  by default each card's free VRAM, is normalised into cumulative fractions and
  layer `i` goes to the first device whose fraction exceeds `i / n_offloaded`
  (`llama-model.cpp`, `get_layer_buft_list`). By count, not by bytes. On a dense
  model those agree; with `--n-cpu-moe N` they do not, because that flag holds
  the FIRST N layers' experts in RAM, so every layer that still owns its experts
  is at the END of the model — and the end of the model is the LAST card. A
  DeepSeek-V4 plan of 38 GB against 42 GB of free VRAM asked one 24 GB card for
  34 GB and died with `cudaMalloc failed`. `src/lib/devsplit.ts` sizes each
  slot, packs them into the cards in order, and emits the `-ts` that pins the
  result; `plan.devices.fits` is a separate constraint from `vram.overB === 0`,
  and `tune` requires both.
- **A placement that has already happened is not a prediction.** That packer is
  the most valuable thing in a PROPOSAL and a liability in a description of a
  live run: its per-card budgets hold back the safety reserve, the user's
  reserve and the device's scratch, and the live path re-derives our own
  footprint by proportion (`withoutOurUsage`) — which under-counts, because
  llama.cpp's real VRAM overhead is larger than the estimate (measured 1.6 GB
  more on DeepSeek-V4 across two cards). Re-packing a LOADED model therefore
  came up ~1 GB short and the machine panel announced "1010 MB of layers have
  nowhere to go — no card has room for them, however the cut is made" about a
  server that was answering prompts, with `vram.overB` reading 0 on the same
  screen. So `plan()` takes a `PlanQuestion`: `"proposed"` (the packer decides)
  or `"running"` (the MEASUREMENT decides — `vram.overB`, which is also what
  `drift` reads, so genuine pressure is still reported). `currentStatePlan()` is
  the only caller that passes `"running"`, and it is the only one describing
  something that already exists.
  - **And its per-card BYTES are measured too, not just its verdict.** Dropping
    the packer's `fits` was half the job: `bytesB` and the three bands per card
    still came from `packSlots`, which is first-fit. With `--n-cpu-moe 42` every
    offloaded slot but one is attention-only, so the whole 9.2 GB fits inside
    card 0 and the packer answers `[44, 0]` — while nvidia-smi showed 11.6 and
    12.2 GB on the two cards. It was never even a split we ASKED for: the packer
    emits `-ts` only when it needs one, so llama.cpp divided the layers by its
    own free-VRAM rule and used both. The panel drew GPU 1 holding nothing of
    ours and filed our own 12 GB there as somebody else's memory. `plan()` now
    takes `cardFreeAtStartB` (`srv.runCardFreeB`, free per card at the spawn —
    the same baseline `drift` reads) and a live plan attributes per card by
    free-at-spawn minus free-now. The card TOTAL is then measured; the split
    between weights/KV/compute inside it stays apportioned, because the driver
    reports bytes and not what they are for. When the baseline is absent, all
    zeroes, or names a different number of cards, the packer's answer stands —
    an index mismatch would be worse than the guess it replaces.
  - **That baseline is recorded against the cards the PLANNER sees.** It was
    taken from the raw `hw.gpus`, which on this machine includes an AMD iGPU a
    CUDA build filters out (`backend.ts:usableGpus`), while both readers index
    by a position that excludes it — `plan` against `hwSnapshot().gpus`, and the
    fit ladder against the `CUDA1` in llama.cpp's own error. Harmless only while
    the NVIDIA cards happen to sort first.
- **The projection is of the command Start would issue.** The panel says "after
  starting", and with auto-optimal on what starts is the tuner's answer for the
  machine as it is NOW — while the tuner is deliberately suspended during a run,
  so `cfg.settings` drifts away from it. Projecting those stale settings drew a
  plan for a command nobody would ever issue (gigabytes of unplaceable layers,
  beside a tuner that had a fitting plan the same second).
  `derive.ts:projectedSettings` closes that: the tuner's settings when it is on,
  the user's own — pin included — when it is off. `placements()`/`measuredCtx()`
  live in `derive.ts` for this reason (re-exported from `actions.ts`): they are
  derived values, and the projection needs them without an import cycle.
- **The user's own environment variables are a PREFIX, not argv.**
  `GGML_CUDA_DISABLE_GRAPHS=1` and kin are process ENVIRONMENT — no llama.cpp
  flag exists for them — so they enter through one input in the Command panel
  (`cfg.envVars`, `src/lib/envvars.ts`), appear as the prefix on the command the
  way a shell reads it, and ride the spawn as `env` while `argv` stays
  byte-identical: everything that indexes argv (the fit ladder's `-c`/`-ts`
  surgery, the builds-root sandbox check) cannot be moved by one position.
  `commandLine`/`commandBlock` render the prefix; `argv` does not. The input is
  shell-shaped (quotes group, quotes are syntax and stripped) and REFUSES what
  it cannot honour — a bare word, `NAME=`, a `$` expansion — naming each under
  the box, because a token that disappears silently is a setting the user
  believes in that does not exist. While a server runs the preview shows the
  RUNNING process's own variables (`srv.runEnv`, the same promise `runSettings`
  makes) and the box is locked. The ladder's rungs inherit the env untouched
  (`s.runEnv`), and `updateNow` carries it across its restart. Verified against
  a real child: the variable reaches the process and the inherited environment
  survives (Deno MERGES `env` over the parent's — a replace would strand the
  loader's PATH).
- **`-ngl` counts the output head, and never moves the embeddings.** There are
  `nLayer + 1` slots, so `-ngl 43` on a 43-layer model offloads layers 1..42 AND
  the output, leaving layer 0 on the host; only `-ngl > nLayer` takes every
  layer. The token embedding table is an INPUT tensor and llama.cpp pins those
  to the CPU at any `-ngl` (`dev_input`, "very little benefit to offloading the
  input layer"), so billing it to VRAM spent ~1 GB of a card's budget on bytes
  that were never going there. Both rules live in `devsplit.ts:offloadRange` and
  `plan.ts` reads them.
- **`--mlock` and `--no-mmap` were ONE setting, and upstream has now DELETED
  them.** Both assigned `params.load_mode` (`common/arg.cpp`), so emitting both
  was never "locked and unmapped" — it was whichever came last, silently, while
  the app printed a reason claiming the other. Upstream finished the job:
  deprecated in favour of `-lm/--load-mode`, then removed, so a **master build
  answers `--mlock` with `unknown argument` and exits before it has read the
  model path**. That is what "llama.cpp master fails" looked like from the
  outside — no diagnosis, no model name, nothing on screen naming a flag. The
  catalog now holds one enum (`params.ts:loadMode`,
  `auto|none|mmap|mlock|
  mmap+mlock|dio`) and `Param.legacy` carries the old
  spelling for a build that still wants it. `cfg` is `version: 4` and CARRIES
  the old value forward, unlike the reserve in version 2 — the pair and the enum
  say the same thing about the same file. With routed experts on the host the
  tuner still chooses `auto` and emits nothing at all, and that was measured:
  `--no-mmap` copies the whole 145 GB file on every start (148 s cold, 160 s
  even warm, because its own copy evicts the page cache), while mapped the same
  start is 73 s cold and **6 s warm**, and generation is slightly FASTER mapped
  (9.6 against 8.9 tok/s, same CUDA build, same cards). Every fit-ladder rung
  reloads the model, so this is also what makes the retry ladder affordable.
  `mlock` is not chosen there either: it would ask to pin more than stock
  `RLIMIT_MEMLOCK` allows (23 GB on the machine that motivated this, against
  ~110 GB of experts), and llama.cpp would warn and run unpinned — a setting
  whose stated effect does not happen.

- **The command is composed against the build's OWN vocabulary, flags AND
  values.** `builds.probe` already asked each binary what it accepts; only the
  tuner read the answer, so the app could still hand a build a flag it had never
  heard of — and upstream removes flags, not just adds them.
  `command.ts:emitFor` now leaves out anything a PROBED build does not declare,
  `droppedFlags` names what it left out under the command
  (`src/ui/CommandView.tsx`), and `Param.legacy` maps the value to an older
  spelling where one exists. The default is the OPPOSITE of
  `caps.ts:supportsFlag` and deliberately so: there the question is "may the
  tuner switch an extra thing ON?" and silence means no; here it is "should the
  app DELETE a setting the user can see?", so an unprobed build (`null`,
  `undefined` or `[]`) is given everything.
  - **A VALUE can be missing while the flag is present, and it fails the same
    way.** `--lazy-mode` is in master; its `on-direct` value arrived in PR
    #28136 and is not, and master answers `--lazy-mode on-direct` with
    `error while handling argument: invalid value` and exits. So `caps.ts`
    records the listed values as `flag=value` entries beside the flag, reading
    all four shapes llama.cpp writes them in — `allowed values: …`,
    `- value:
    description` bullets, `{none,layer,row}` in the placeholder,
    and a bare comma list (`--spec-type none,draft-simple,…`). A placeholder
    containing `...` is a SHAPE and not a list of choices (`-ts N0,N1,N2,...`),
    or `-ts` would "allow" the literal value `N0`. A flag whose help lists no
    values is never judged: silence means "the help does not say".
  - **The test stub is now load-bearing.** `tests/stub-llama-server.ts` must
    declare the whole catalog or every test build looks like an ancient
    llama.cpp; it cannot import `params.ts` (it is COPIED into a temp builds
    directory) so it holds its own copy and `tests/guards.test.ts` fails if the
    two drift. `--old-build` still answers a llama.cpp from before `-bs` and
    `--fit`.

- **Some models cannot be sized from their header, so the app measures.**
  `plan.ts` is arithmetic over facts and that is still true for almost every
  model — but DeepSeek-V4-Flash declares a 1,048,576-token context at which
  llama.cpp asked for a **68.5 GiB compute buffer** (predicted: 730 MB) and an
  18 GiB KV cache (predicted: 183 MB). Both scale with the context for a
  sparse-attention model and nothing in the header says by how much, so no
  placement search could have found it — 68 GiB of scratch does not fit any
  division of any cards. llama.cpp's own fitter (`-fit`) exists for exactly this
  and **segfaults on this model**, so it cannot be delegated to either. What is
  left is `src/lib/fitladder.ts`: start, and if the run dies for want of memory,
  halve the context and start again (six rungs — 1M → 32k, because 64k was
  measured to fail), then write the working length into `cfg.fitCtx` so the next
  run of that model opens there. The estimate is the opening bid; the allocator
  has the final say — **at generation, not just at load**: CUDA allocates its
  compute scratch (activation-quantise buffers, cuBLAS workspace, graphs) lazily
  at the first real batch, so a too-tight plan can pass `/health` and die on the
  first prompt (measured: healthy at 17,408, then `CUDA error: out of memory`
  inside `quantize_row_q8_1_cuda` answering "Hi" — stderr that never says
  "buffer", which `isFitFailure` must still recognise). So `srv.poll` enters
  `ready` only through a one-shot generation probe (`srv.server.ts:probe`,
  `/completion`, a few dozen prompt tokens + 2 predicted); a probe that kills
  the process is the ladder's next rung, and `cfg.rememberFit` waits for
  `srv.proven`, never `/health` alone — recording at health once wrote a
  crashing 17,408 down as a fact, and `rememberFit` only ever grows. The retry
  rewrites `-c` in the argv that actually ran rather than re-composing one, so
  "what you see is what runs" survives it, and it is off whenever the user typed
  the context themselves. `tests/deepseek.e2e.test.ts` proves the whole path on
  the real 145 GB model (opt-in: `LLAMA_MASTER_E2E=1`).
- **A model is not a cache, so the ladder has two rungs.** The rung above only
  ever shrank `-c`, and the commonest overflow on a workstation is not the
  cache: the plan is made when the desktop holds 2 GB of VRAM and Start happens
  when it holds 5.5, a browser having opened in between. The weights buffer then
  fails (`alloc_tensor_range: failed to allocate CUDA1 buffer`), and halving the
  context does not move it by one byte — six full reloads, then defeat, with the
  answer one `--n-cpu-moe` step away. `fitladder.ts:fitFault` tells the two
  apart: an OOM whose signature says the TENSORS were being placed is `weights`,
  everything else is `context`. The weights rung is sized from the SHORTFALL and
  not the request — the card had 22 GB to give and llama.cpp asked for 33.9 GiB,
  which is six layers at 2.5 GB of routed experts each, not the fourteen the
  request alone implies — and it DROPS the `-ts`, because that split pins the
  layers to the cards that just proved they could not hold them, while
  llama.cpp's own free-VRAM split is measured at load time on the machine as it
  actually is.
- **Leftover VRAM becomes micro-batch, and the estimate had to learn the FFN
  first.** After placement, residency and context are settled, the tuner grows
  `-ub` 512 → 1024/2048/4096 through the same fit gates as the placement itself
  (`tune.ts`) — prefill is the wait at long context, measured 150 → 364 tok/s
  going 512 → 4096 on a MoE with experts on the host, and VRAM the placement
  left idle buys exactly that. It never grows past what the plan can bill: the
  compute estimate now charges two FFN-wide f32 activations per micro-batch
  token (`plan.ts:ffActivation`, from `feed_forward_length` /
  `expert_feed_forward_length` × `expert_used_count`, read in
  `rust/src/gguf.rs`) — the `n_embd`-only estimate under-billed `-ub 4096` about
  4x, which was harmless while nothing spent headroom and unsound the moment
  something did. Gated off for sparse-attention models, whose scratch is
  measured end to end with its own ub term. A 256 set by the placement to buy
  layers is never grown — that would undo the trade just made — and the
  behaviour is visible in the verdicts: Flash-Next pinned at 250k grows to 1024
  (the ctx×ub term prices 2048 out), aimFull at 262,144 grows nothing, a small
  dense model on a big card reaches 4096.
  - **Revised 2026-09-18: only when weights stream from RAM.** With every layer
    in VRAM across two cards, a bigger `-ub` is a LOSS: llama.cpp pipelines a
    batch through the cards in `-ub` slices, and fewer slices means less
    overlap. Measured all-VRAM, 512 → 4096: Qwen3.8-27B prefill 1622 → 1055
    tok/s, Ternary-Bonsai-2 1966 → 1212. So `tune.ts` grows `-ub` only when
    `--n-cpu-moe > 0` or a layer stays on the host (`hostWeights`) — that is
    where the 150 → 364 above was measured.
- **The attention mask is `context × micro-batch`, and that was the hole under
  the micro-batch growth above.** llama.cpp builds `kq_mask` as
  `[n_kv,
  n_tokens]` — f16 with flash attention, f32 without
  (`llama-graph.cpp`, `llm_graph_input_attn_kv`). It is invisible at `-ub 512`
  (256 MiB at a 262,144 context, inside the flat backend figure) and it
  DOMINATES at `-ub 4096`, where the same context costs 2 GiB — per card, plus a
  pinned-host copy. `plan.ts` knew nothing about it, so the tuner grew `-ub` to
  4096 against a budget that did not know what a micro-batch costs, and
  Ornith-1.5-35B-A3B at its full 262,144 died in `graph_reserve` asking for
  **9,249 MiB on a card with 1.3 GB free**, with no flag named anywhere in the
  error. Measured on two architectures that share nothing but llama.cpp: **2.00
  bytes per (context token × micro-batch token)** on Ornith over 14 points,
  fitting the reported `CUDA_Host compute buffer` to ±0.0 MiB, and 1.99 on
  Gemma-4-26B-A4B over 6. Charged to RAM once (that is where the mask is built
  on a CUDA run) and to each DEVICE once (the working copy: 1.4 bytes per pair
  on Ornith, 3.0 on Gemma-4, so the same 2 is the honest middle). Not multiplied
  by `-np` — `[n_kv, n_tokens/n_stream, 1, n_stream]` divides rather than
  repeats, unlike `computeScratch`, whose per-slot graphs really are copies.
  Zero for a sparse-attention model, whose scratch is measured end to end
  already.
  - **Re-measured 2026-09-18 on current llama.cpp: 8 bytes per pair, not 2** (16
    without flash attention), on the host AND on each device. The 2 above was a
    llama.cpp of July. `plan.ts` was refit to 18 measured compute buffers:
    `graphBytesPerToken` (FFN- and embedding-wide activations per ub token,
    dense and expert parts), `HOST_GRAPH_B_PER_TOKEN` 56 KiB,
    `BACKEND_CONTEXT_B` 420 MiB (CUDA context + cuBLAS beyond the reported
    buffers, per card), `HYBRID_SCRATCH` folded into the graph term. Host fits
    ±3%, devices 1-15% OVER (the safe side). Live check, Ternary-Bonsai-2 at
    262,144 on two cards: plan 15.80 / 9.97 GiB, nvidia-smi 15.67 / 9.92.
- **`--spec-type draft-mtp` is a SECOND CONTEXT, not a rounding error.**
  `plan.ts` billed it as "one block's KV over the same window, so it is small"
  plus half the flat backend figure, charged once to the pool. The failure mode
  is the worst kind: the main context allocates and succeeds, then llama.cpp
  builds the draft context and runs out, and the log says
  `failed to create MTP context` about a plan that said the model fitted. A/B on
  Ornith, one card, `--n-cpu-moe 41`, ctx 32,768, `-ub 512`, the flag the only
  difference: draft KV **+64 MiB** (exactly one dense block at f16 —
  `(keyLength + valueLength) × nHeadKv × 2` = 2 KiB/token, and it ignores
  `-ctk`), recurrent state **×4** (62.81 → 251.25 MiB — three more copies, for
  rollback, exact), compute **+259 MiB per device** (flat in the context, linear
  in the micro-batch) and **+40 MiB host** (its own copy of the mask).
  `plan.ts:mtpDraft` sits 13-34% above every point measured, which is the right
  direction: being short here fails a load that had already half succeeded.
- **A hybrid linear-attention model carries per-context working set beyond its
  KV cache.** Neither the cache (`kvTotal`) nor the constant recurrent state
  (`recurrentStateB`): the delta-net trunk's own scratch. Residual on Ornith at
  `-ub 512`, once the flat figure, the activations and the mask are removed — 78
  MiB at 32k and 65k, 132 at 131k, 388 at 262k. `HYBRID_SCRATCH_B_PER_CTX` is 2
  KiB per context token, which covers the top of that with a tenth to spare and
  over-bills the short contexts by ~100 MiB, where the flat backend figure
  already dwarfs it.
- **The ladder's two cheapest rungs cost nothing the user asked for.** Both fire
  on a `context` fault, both BEFORE the rung that shortens `-c`:
  - **Stop drafting** (`fitDecision` → `nospec`) when llama.cpp names it —
    `failed to create MTP context`. Speculative decoding buys SPEED and nothing
    else; the answer is identical without it. Deliberately NOT written to
    `cfg.unsupported`, unlike the `drop` rung: that rung means "this build
    cannot do it", and this means "this machine had no room for it today".
  - **Take the micro-batch back** (`ubatch`, halving to a floor of llama.cpp's
    own 512). The graph is sized `ctx × ub`, so halving either frees the same
    bytes — and they do not cost the same thing. `-c` is what the user asked
    for; `-ub` is what the TUNER raised from 512 on the theory that VRAM was
    going spare. So it has its own permission, `autoUbatch`, separate from
    `autoFit`: pressing **Max·Hybrid** pins a context and turns `autoFit` off,
    and the micro-batch is still the app's to give back. A 256 set by the
    placement to buy layers is never touched — that would undo the trade.
- **Verified end to end on the model that started this**: Ornith-1.5-35B-A3B
  Q6_K, hybrid, its full 262,144 context, `--spec-type draft-mtp`, two 24 GB
  cards — the tuner's answer now keeps `-ub 512`, starts, and generates. Before,
  the same gesture produced `-ub 4096` and `cudaMalloc failed`.
- **The sparse-attention scratch is measured, and the biggest term was SLOTS.**
  llama.cpp's `-np` default is `-1 = auto`, and auto chose **four**. Each server
  slot runs its own graph, so each one costs another copy of the context-sized
  indexer tensors: the same model, placement and context cost **43,517 MiB of
  VRAM at four slots and 21,467 at one**. That is why a 1,048,576-token context
  looked impossible on 48 GB of cards that run it comfortably. `parallel` is a
  term in `plan.ts:computeScratch` now, read from the settings the argv is built
  from. Measured at one slot, both cards, ub 512, VRAM read after a real
  generation: **8.1 KB per context token**, straight from 4,096 to 1,048,576
  (2,045 MiB of scratch at 262,144; 8,547 MiB at 1,048,576). At four slots the
  same line reads 29.0 KB/token — the ratio is the slot count. Two earlier
  calibrations missed this because both were run at the default.
  - **The scratch is divided between the cards, but not by layer count.** One
    card and two cost the same TOTAL (8,110 MiB against 8,567 at 1M), so the
    pool counts it once. Where the division falls is llama.cpp's decision and it
    is not proportional: a card holding 10 slots of 44 wanted about 60% of it
    and died in `graph_reserve` under a plan that had budgeted it 23%. So each
    CARD is budgeted for the whole thing. The two views answer different
    questions — the pool says what the machine will use, a card says what it
    must be able to give — and `plan.devices.fits` was already a separate
    constraint from `vram.overB === 0` for exactly this reason.
  - Verified end to end on the real model: the app's own answer at 1,048,576
    (`--n-cpu-moe 36 -np 1 -ts 37.5,6.5`) starts and generates at 13.1 tok/s; at
    524,288 (`--n-cpu-moe 32`) 14.9 tok/s.
- **llama-server has its own fitter now, and it defaults ON — so the command
  always pins `--fit off`.** Upstream `fit_params = true` (`common.h`) quietly
  adjusts whatever the command line left unset to fit device memory. This app IS
  that fitter — the plan, the packer, the retry ladder — and its promise is that
  the command shown is the command that runs; two fitters disagreeing about one
  process means the settings panel describes a run that never happened. It also
  segfaults on DeepSeek-V4-Flash, which is why `fitladder.ts` exists. The flag
  is a catalog entry like any other (`params.ts:fit`, default off, `llamaDef`
  on), so a user who wants upstream's behaviour flips one switch and the flag
  honestly disappears from the command. A build too old to know `--fit` lands in
  the existing unknown-argument signature in `serverlog.ts`, whose advice —
  reset the setting — is exactly right.
- **Qwen3.8-Flash-Next (qwen4exp) is ready before the llama.cpp PR merges, and
  two of its facts live in the reader.** Verified against the real 111 GB
  4-shard GGUF (headers through `readMeta` → `mergeShards` → `tune`, on the 24
  GB + 110 GB rig from the field report that motivated it — the tuner's answer
  fits at a pinned 250,000 and reaches the full 262,144 with `--n-cpu-moe 42`).
  (1) `per_layer_token_embd` — the PLE n-gram gather table, ~27 GB, a quarter of
  the file — is `LLM_TENSOR_LAYER_INPUT` in llama.cpp, CPU-pinned at any `-ngl`
  like `token_embd`, so `gguf.rs` files it with the embeddings; under "output"
  it handed a quarter of the model to the head `-ngl` offloads and every VRAM
  plan was wrong. (2) `full_attention_interval` absent means 4 on this arch, not
  dense — llama.cpp hardcodes that default (`models/qwen4exp.cpp`), and reading
  absence as "every layer has a cache" is the 4x KV overestimate the hybrid
  support exists to prevent. The official conversion writes the key; a file from
  another tool may not. Its sparse attention (QSA) declares the same
  `attention.indexer.top_k` DeepSeek-V4 does, so the measured scratch term and
  the fit ladder apply unchanged; the single PLE layer's ~0.35 MB per-sequence
  conv state is deliberately unbilled (`plan.ts:recurrentStateB` says why).
  Support needs llama.cpp PR #27742, which the app cannot build until it merges
  — source builds fetch tags and master, never a PR ref. `command.ts` omits a
  flag whose value equals `def`, on the theory that `def` IS what llama.cpp does
  without it. Upstream moved: `-ngl` now defaults to **auto**, so "CPU only"
  emitted no `-ngl` and llama.cpp offloaded to the GPU anyway; `-c` defaults to
  **0 = take it from the model**, so a plan drawn for 4,096 tokens started a
  server at this model's declared 1,048,576 and could not allocate — a start
  that cannot succeed, with an error naming none of it. `Param.llamaDef`
  (`types.ts`) carries llama.cpp's own default when it differs, and omission is
  judged against THAT. The three flags that decide the placement — `-ngl`, `-c`,
  `-np` — are always emitted, because their whole job is to pin what runs.
  Re-check after any llama.cpp bump: the failure mode is silent, and
  `llama-server --help` prints every default.
- **One thread per PHYSICAL core, and the priority switch is what protects the
  desktop.** `cpuBudget` used to leave two cores to the OS. Measured on the same
  DeepSeek placement: 16 threads 15.91 tok/s, 32 threads (SMT) **0.94** — two
  threads sharing one core's memory port thrash rather than pipeline, because
  generation with experts on the host is bandwidth-bound. Leaving cores idle
  costs throughput and buys responsiveness that nice 19 + idle I/O
  (`src/lib/priority.ts`) already provides, so `stability` now warns about the
  SWITCH being off rather than about the thread count, and SMT is a `risk`.
- **The residency anchor has 5% of slack.** One layer of routed experts moved
  back to the host costs about 2% of the generation rate (15.7 tok/s at 28 held
  back, 14.0 at 30, 13.2 at 32). A STRICT anchor spent that 2% to defend a
  context three times shorter — 9,728 tokens where 27,648 was available — and
  said nothing. The slack is bounded on purpose; `aimFull` remains the only way
  to buy context without limit.
- **A reserve that costs something says so.** Honouring it by planning as if the
  memory were absent is right and completely invisible: the answer just comes
  back smaller. `derive.ts:reserveCost` re-tunes with the reserve dropped and
  reports the difference in the tuner's own units — layers of experts, tokens of
  context, or the whole model when the reserve is what makes it impossible.
  Never a predicted tok/s: the 2%-a-layer figure is one model on one machine.
- **`--mlock` is a promise the kernel can refuse, so the limit is read.** The
  partial-offload branch already declined it; a CPU-ONLY placement has
  `--n-cpu-moe 0` and fell straight past that guard, so the app emitted
  `--mlock` for a 97 GB host-side model and printed "pinning them stops the OS
  paging the model out" — against a stock `RLIMIT_MEMLOCK` of 23.3 GB, where
  llama.cpp warns and runs unpinned. `hw.server.ts:lockable` reads
  `/proc/self/limits` (the child inherits ours, so ours is the right one) into
  `Mem.lockableB`, and `tune` promises the flag only when the limit covers the
  need. Absent — macOS, Windows — is treated as "do not promise": unknown is a
  reason to stay quiet, not to assume the best.
- **The build is already native, and it matters.** `GGML_NATIVE=ON` is passed
  for source builds and the resulting `libggml-cpu.so` on this Zen 5 box carries
  AVX-512 (`zmm`), VNNI (`vpdpbusd`) and BF16 (`vdpbf16ps`). Worth re-checking
  after any build-flag change: with experts on the host, the CPU kernels are
  half the generation path, and a generic x86-64 build would quietly halve it.

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

- **A nested same-cell call reads COMMITTED state, so it cannot see what the
  caller is halfway through writing — and that shipped as a real bug.**
  `builds.update` set `ref`, `origin` and `backend` from the active build and
  then called `builds.start()`. That is a SECOND DISPATCH with its own draft:
  `start` read the values from before, and the job came out labelled
  `Install b1111` — the version already on disk. **The Update button reinstalled
  what you had, reported success, and said it updated**, and it could change the
  backend under you the same way. Measured with a throwaway `testCell`, because
  reasoning about commit points is how it got there.
  - **`s.$commit()` does NOT fix it** — verified: the sibling still reads the
    pre-call snapshot. Only sharing the draft does. `s.$call.start()` runs the
    sibling's body against THIS draft in THIS commit
    (`dep/aio/docs/state/methods.md`).
  - **`$call` moves the cancel wiring with it.** The sibling gets the CALLER's
    signal, so `cancelOn` needs `update: ["builds:cancel"]` beside `start`'s —
    without it Cancel stopped a Build and did nothing at all to an Update.
  - **Reached through a local cast, not `MethodDraftCalls`.** That type names
    the siblings, `start`'s return type is inferred from the same object literal
    the methods live in, and TypeScript answers the circularity by widening the
    WHOLE method map to `undefined` — surfacing as
    `builds.checkUpdates is possibly undefined` in `app.ts`, one file away.
  - The other three nested calls in this repo are safe and say why in a comment:
    `prereq.fixAll` wants separate commits (one dispatch per item is what makes
    the queue watchable), `builds.setOrigin` → `loadAssets` reads only `s.ref`
    which it does not write, `hw.togglePause` → `refresh(true)` is safe
    _because_ of the `true`, and the client's `discover` → `connect(url)` passes
    the URL as an argument. A sync method cannot `$call` an async sibling at all
    — there is nowhere to await it.
- **A cell read inside `afterRender`/`onMount` subscribes to NOTHING.** A
  component re-renders only for what its render BODY touched, so a value read
  first inside the callback never triggers the render that would run the
  callback again — it fires once and the feature "works sometimes". `RunStrip`
  read `chat.lastTps` that way, and the symptom had already been seen and
  mis-diagnosed once ("pressing Measure changed nothing until the user happened
  to say something afterwards" — the bench was added to the key, which treated
  the symptom). The key is built in the body now and closed over. aio 1.0.0-beta
  names the value, the component and the consequence at dev time.
- **Every colour in this app is measured now, and one had never been.**
  `--text-faint` was #5b6577: 3.11:1 on the panel, 2.42:1 on a selected card's
  accent wash, where WCAG AA asks 4.5:1 for body text — and it IS body text
  (units, sub-lines, parameter tips), on twelve elements at once. Raising it
  alone would have collapsed it into `--text-dim` (4.53 against 4.70) and
  silently cost the design a level, so the whole ramp moved: worst case over
  every background this app paints, 11.2 / 7.0 / 4.5 in both themes and in the
  client. Found by aio 1.0.0-beta's dev-time contrast walk, which composites
  translucent layers — the accent wash is `rgba(240,169,46,0.14)` over the
  panel, and nothing that reads the stylesheet alone would have seen it.
- **A container's children are all keyed or none of them are.** Five here mixed
  a keyed `.map()` with static siblings (`nav.rail`, `.map-track`,
  `.cmd-blocks`, `.ctx-bands`, `.ctx-range-track`), which leaves the reconciler
  matching one half by key and the other by position. The static ones carry keys
  now.
- **Two components must not spell a class the same way and want opposite
  things.** `.cores`/`.core` was the CPU panel's one-row-per-core grid AND the
  dashboard's 26px strip of bare bars, 1,200 lines apart in one stylesheet — so
  the later rule won in both places and the dashboard strip was being laid out
  as a three-column grid it has no children for. Renamed to `.core-rows` /
  `.core-row`. `aiol` finds these; a shared rule followed by an override of it
  (`.log, .cmd` then `.cmd`) is the same finding and is answered by giving each
  class its own home, so the shared block is only what is genuinely shared.
- **`force?: boolean` and `force = false` are the same to TypeScript and not to
  aio.** Only the DEFAULT is visible at runtime, and aio counts a method's
  declared parameters against what a dispatch actually passed — so
  `hw.refresh.action()` on the 1 s schedule was a short dispatch reported on
  every boot. Give an optional parameter a default in the signature.

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
- **Renaming a persisted field is a migration, written or not.** `cfg` is
  `version: 2` with an `onMigrate` that drops `reserveVramB` — without it aio
  deep-merges the stored blob over the defaults, keeps the orphaned key forever
  and says so on every boot ("shape drift: 1 stored field(s) no longer match the
  declared shape"). Verified by seeding a store with the old world: the boot
  reported `{cell: "cfg", from: 0, to: 2, outcome: "migrated"}` and the key was
  gone. The old value is deliberately NOT carried forward — it meant "hold this
  much across the whole machine, divided between the cards", which is neither of
  the fields that replaced it. Pinned in `tests/cells.test.ts`.
- **An `own` effect must name the resource it owns.** `own.set(key, …)` with a
  key already in use **disposes the previous effect** — so a teardown written as
  "stop the server" stops whatever is running _now_, not the process that effect
  was created for. This shipped: after any crash, the next Start came up and was
  SIGTERMed a moment later (`exited with code 143`), and the app looked like it
  simply could not start a server. The close must be conditional on identity
  (`io.stopOwned(pid)`). Reproducing it needs `testServer` **and** a previous
  run that **crashed** rather than being stopped (`stop` disposes the effect
  cleanly). Pinned by `tests/runtime.test.ts`. (aio ≤ alpha37 also ignored `own`
  effects in the in-process harnesses, which hid it entirely; alpha38 acquires
  and disposes them for real and warns once per replaced key.) The slot is keyed
  BY PID now — `srv:process:<pid>`, which is aio's own advice for this case
  (`docs/state/methods.md`: "give each resource its own id") and what stops the
  framework warning that a live resource was displaced; the dead process's slot
  is released in the same effect, so a session of starts does not accumulate
  no-op disposers. `stop()` also ends with `if (slot === s) slot = null`: it
  awaits the child's exit, and a start landing in that window had its slot
  erased, leaving every later stop, rss reading and liveness poll working from
  "nothing is running" while llama-server held its VRAM.
- **The all-in-one page is a budget, and the machine column is where it runs
  out.** Three columns, each scrolling alone, and the left one carries the
  vitals, both memory states and the command — so anything added there has to be
  paid for. It was paid for once already: the vitals lost their sparklines and
  two sub-lines apiece (the history graphs are one click away on the pages built
  for them), the current-state table is drawn only when something IS running
  (idle, every row of it is a zero the map above already shows), the two maps
  share one legend and drop the "Memory map — 234 GB total" caption they both
  repeated, and the command shows the server line wrapped rather than one flag
  per line. Verified the way layout has to be verified — by looking:
  `chromium --headless --window-size=1600,1000 --screenshot` against the running
  dev server (`am instances` for the port). For what a picture cannot settle,
  the same chromium with `--remote-debugging-port` and a five-line CDP client
  reports `getBoundingClientRect`/`scrollHeight` for any selector — which is how
  the favicon above was found, and how "the log is below the fold" stopped being
  a matter of opinion.
- **A run of adjacent conditional strings in a fragment is not one text node.**
  `ReserveControls` built its summary as `{a}{cond ? " x" : ""}{cond2 ? …}`
  inside a `<>…</>`; re-rendered on a keystroke, the reconciler left the
  previous sentence beside the new one and the line explaining a refusal
  appeared TWICE, with two different numbers. Build the sentence in the
  component body and interpolate it once. Pinned in `tests/ui.test.ts`.
- **"The log below" must actually be below.** Every diagnosis this app writes
  points at the log; a page that shows a diagnosis therefore renders
  `ServerLog`/`LogView` itself rather than pointing at another tab.
- **In a `testUI`, mount the panel — not `App` plus navigation.** `ui.tab` is
  persisted, and its rehydration on mount is async: a `ui.go(…)` (or a rail
  click) issued before it lands is silently reverted to whatever the previous
  test left in the store. That is a ~40% flake, not a rare one. Rendering
  `ServerPanel`/`OnePage` directly tests the same thing deterministically. The
  rail itself has one test, and that is where navigation belongs.
- **A test that writes app files must set `LLAMA_MASTER_HOME` first.** This app
  owns its home directory rather than using aio's app dirs, because what lands
  there is gigabytes — llama.cpp checkouts, cmake trees, release archives — and
  a user has to be able to find it, back it up and delete it. So `paths()`
  honours `LLAMA_MASTER_HOME` (`src/cell/host.server.ts`) and every test that
  installs a fixture build sets it to a temp dir **before** the first `paths()`
  call — the static imports are hoisted, so "before" means at the top of the
  file. This shipped wrong: the server tests left a `test-build/` directory
  inside the developer's real `~/.llama-master`, and two UI tests silently
  passed only because that home had an app-managed CMake in it.
- **Drain the child's output before reporting that it exited.** `child.status`
  resolves when the process is reaped, which can beat the last of its stderr
  through the pipe — and `srv.poll` diagnoses an exit from exactly those lines.
  Reporting "not running" early replaced "cudaMalloc failed: out of memory" with
  the generic fallback on precisely the failures that happen fastest.
  `srv.server.ts` awaits both pumps first.
- **Never early-return from `stop` on a status a pending `start` has not written
  yet.** `srv.stop()` skipped its work when the status read "stopped" — which is
  exactly what it reads between a Start being dispatched and its body running.
  Stop did nothing, the spawn completed a moment later, and a server the user
  had cancelled sat there holding its VRAM while the UI said "stopped".
  Cancellation now lives in `srv.server.ts` as `stopGeneration`, which the spawn
  checks after creating the process: it is the module that owns the process, and
  it is not subject to cell-state timing. Pinned in `tests/server.test.ts`.
- **A streaming reply must survive the app closing under it.** Shutting down
  aborts every in-flight method (aio `abortAllInflight`, Phase 1) before it
  persists, so `chat.send` reaches its abort branch mid-reply — and what it
  writes there is what the user gets back. It records `acc`/`think`, never
  `s.partial`, which is only as fresh as the last flush. Before this, a window
  closed during a reply printed an `EFFECT_ASYNC_ERROR` block over
  `chat:__setSend` and lost the whole answer, because aio closed dispatch before
  draining the effect that was still writing (fixed in aio;
  `tests/cells.test.ts` pins our half against a real SSE stream).
- **Publishing a partial reply is a full re-send, so the cadence is a byte
  rate.** A state write of a string sends the whole string to every client;
  flushing every 60 ms therefore costs `length × 16.7/s` — quadratic in the
  reply, doubled per open window, and measured at a sustained
  `PRESSURE — 33 broadcasts/sec` against aio's threshold of 30.
  `sse.ts:flushDelayMs` holds the byte rate flat instead (64 KiB/s, clamped to
  60–500 ms), which makes a long answer cost the same per second as a short one.
- **Clamp everything that comes out of a GGUF header.** A truncated or hostile
  file can yield NaN or a negative, `nHead: 0` makes the head-dim fallback
  `0 / 0`, and one such value poisons every total it reaches — silently, because
  NaN comparisons are all FALSE, so `overB === 0` and `freeB >= margin` quietly
  stop meaning anything and the tuner's fit checks become coin flips.
  `plan.ts:whole()` is the one gate; a hostile-header test pins it.
- **Compare paths resolved, never as text.** `bin.startsWith(buildsRoot())`
  accepted `<buildsRoot>/../../../../usr/bin/id` — a rule that read like a
  sandbox and was not one. Archive containment had the mirror bug: it split on
  `/` only, so a Windows-spelled `..\..\evil` walked straight through. And
  `removeBuild` had a third: `join(root, "/")` is `root/`, which
  `startsWith(root + "/")` and contains no `..` — `builds.remove("/")` would
  have removed every build. It is `dirname(resolve(root, id)) === root` now.
- **Fail loud.** A missing binary, an unreadable header, a 404 — surface it in
  `lastError` and render it. Never swallow.
- **A call ceiling belongs on the cell, a reporting budget in `app.ts`.**
  `perfBudget.methods["srv:start"].timeout` is a string in another file that no
  rename follows and nothing checks; `long: ["start"]` on `cell("srv", …)` is
  checked against the method list at `cell()` time. So the six methods whose
  duration belongs to a compiler, a package manager, a disk or a 145 GB file
  (`builds.start/update`, `prereq.fix/fixAll`, `models.scan`, `srv.start`)
  declare `long:` and name no number — the ceilings they used to carry were
  guesses about somebody else's machine, and a timeout does not cancel anything,
  it only abandons a call that is still working. The `timeout:` entries that
  remain are real bounds on work that is quick by nature (a 1 s poll, a
  `nvidia-smi`, a GitHub GET), where a breach is a fault worth reporting.
  `effect:` stays per method in `app.ts` either way — that is the budget, not
  the ceiling.
- **"Your CUDA is too old for your GPU" is a PREREQUISITE with a Fix button, and
  the fix installs nothing on the system.** `cuda.ts` always knew the answer —
  an nvcc older than the card can emit only PTX, which the driver re-compiles at
  every load — but it was a sentence in a build log. Measured the difference on
  this machine, same model, same flags, same source: **19 tok/s built with CUDA
  12.0 (PTX for sm_90) against 41.6 with CUDA 13.3 (native sm_120), and 70 s to
  load against 10 s.** That is a bigger win than any of the pull requests it was
  competing with, so it is a prerequisite (`cuda-arch`) rather than a footnote.
  - **The fix is a `download` plan, never `package` or `script`, and that is the
    whole design.** `apt install cuda` adds NVIDIA's repository and pulls a new
    DRIVER; swapping a driver under a running desktop is the one failure a Fix
    button must never be able to cause, and the .run installer writes across
    `/usr/local`. What this does instead is what NVIDIA publishes for exactly
    this purpose: the toolkit as component tarballs, each with a SHA-256 in its
    manifest (`src/lib/cudaredist.ts`), verified before anything is written,
    unpacked into `~/.llama-master/cache/toolchain/cuda-<version>`. No root, no
    apt repository, no driver, no kernel module, no system package, no PATH
    change — and undone completely by deleting one directory. The list of
    components is short and every entry says which part of a llama.cpp build
    asks for it; a test fails if `nvidia_driver` ever appears in it.
  - **The DRIVER is the ceiling and is read first.** A toolkit newer than the
    driver produces binaries the machine cannot load, so `cudaOffer` returns
    nothing when the driver could not run the release that would help — better a
    slow build than a broken one. Read from `nvidia-smi`, which prints it as
    `CUDA Version` on older drivers and `CUDA UMD Version` on newer ones. On
    this machine the driver already reported 13.3 while nvcc was 12.0, so the
    toolkit was the only thing in the way.
  - Installed into a staging directory and renamed only when every file is in; a
    half-unpacked toolkit that looks installed is worse than none, because the
    next build would pick it up and fail deep inside cmake. Disk is checked at
    three times the download (measured: 929 MB of archives → 2.0 GB installed,
    both present at once) — this machine sits at 99% full. System `tar` does the
    unpacking, because Deno's `DecompressionStream` has gzip and deflate but no
    xz.
  - `detectCudaPlan` prefers the app-managed nvcc over PATH and says so in the
    build log. Without that the Fix button would install a toolkit that cmake
    then ignored.
- **A shallow clone cannot always find a merge base, and the fix is more history
  — never `--allow-unrelated-histories`.** Assembling a stack fetches at
  `--depth=300`, and a pull request whose branch point is older than that makes
  git refuse with "unrelated histories". That flag would splice two genuinely
  unrelated trees together and hand the result to a compiler, so the merge
  deepens (`--deepen=5000`) and retries instead. Also: a failed merge with NO
  conflicted files is not a conflict — reporting it as one sent a debugging
  session looking for overlapping edits that did not exist, so git's own words
  are used when the file list is empty.
- **A pull request is a source ref, and GitHub does the merge.** The interesting
  models arrive as PRs months before they merge, and the only way to run one was
  to drop its tarball into the source cache by hand — which works once and then
  rots under a name that no longer describes it. The tempting fix is "clone and
  merge"; it is wrong here, because source is fetched as a tarball precisely so
  git is not a dependency. GitHub already publishes `refs/pull/<N>/head` (the
  author's branch) AND `refs/pull/<N>/merge` (that branch merged into current
  master, kept up to date by GitHub), and codeload serves a tarball of either.
  So "latest llama.cpp with this one PR" is a plain download.
  `src/lib/srcref.ts` owns the ref grammar (`master`, `b7421`, `pr/27754`,
  `pr/27754@head`), the URLs, and the filesystem-safe directory name —
  `pr/27754` carries a slash and both the source cache and the build id become
  directories, so that is closed at the source rather than at every consumer.
  **The merge ref's 404 is a verdict, not a missing file**: GitHub withdraws it
  when the branch and master disagree, so `refNotFound` says "this PR does not
  currently merge" and offers the branch alone. Verified live 2026-09-03: PR
  27754 answered 200 on both refs, 27742 answered 200 on `head` and 404 on
  `merge`, and the 27754 merge tarball contains both the PR's code and master's
  newer CUDA kernels. The release route REFUSES a PR before the button lights
  (`targetReadiness`) — nobody publishes binaries for unmerged code. One PR at a
  time: stacking two needs a real merge, which needs git.
- **More than one PR needs a real merge, so it needs git — and only then.**
  GitHub merges ONE pull request into master for us; it will not merge two into
  each other. So `master+pr/A+pr/B` (`SrcRef.kind === "stack"`) is assembled
  here, and it is the ONLY ref that requires a tool to be installed
  (`refNeedsGit`) — saying "git is required" flatly would be false for
  everything else this app builds. Assembly uses one bare clone reused by every
  stack (`sources/_gitrepo`), merges each PR in the ORDER GIVEN (order is part
  of the ref and of the directory name, because merging A then B is not merging
  B then A), and exports the result with `git archive`, so the source tree has
  the same shape a tarball produces. A conflict names the pull request AND the
  files. A fetch that fails when the clone already holds every ref needed falls
  back to what is on disk and SAYS so — GitHub answers 401 to unauthenticated
  git when it is throttling an address, and losing a build to that would be
  worse than building from a copy that is named as possibly stale.
- **A clean merge is not a clean build, and the app must not pretend
  otherwise.** Measured the expensive way on 2026-09-03: PR #27773 (GLM-5.3-
  Flash) merged into master with zero textual conflicts and then failed to
  compile, because master had added an `n_kv_max` parameter to `build_attn_mha`
  and the PR still called the nine-argument form. Git cannot see that; only the
  compiler can. So a stack that assembles is not a stack that works, the build
  log is where that is found out, and the app must never present "merged
  cleanly" as "will build". Nothing here patches a PR to fix it: guessing a
  value for an argument in an attention kernel is how an inference engine
  produces confidently wrong output.
- **A merged pull request is the failure that looks like success.** GitHub keeps
  `refs/pull/<N>/merge` after a PR lands, still pointing at the merge computed
  back then — so the day it merges, `pr/<N>` quietly starts meaning "master as
  it was months ago, plus a change master already has". The build succeeds, the
  name looks right, and the binary is BEHIND plain master. `prState` reads the
  PR's plain page (no API — the 60/hour quota is routinely exhausted) and
  `prStateNote` says it, with the answer, which is always "build master".
- **A PR-only setting must default to llama.cpp's own behaviour.** `--lazy-mode`
  is in master with default `auto`; its `on-direct` VALUE comes from PR #28136
  only. The catalog entry carries `llamaDef: "auto"`, so at the default the flag
  is not emitted at all and a build without that PR is untouched. That is the
  rule for every setting added for an unmerged change: the app's promise that a
  plain-master build keeps working survives only if nothing PR-specific is ever
  emitted by default.
- **A fork is a source ref too, for models upstream cannot load yet.** PrismML's
  ternary `PQ2_0`/`PTQ1_0` (ggml types 142/143, parked high so they cannot
  collide) exist only in `PrismML-Eng/llama.cpp`; the one upstream PR is
  CPU-only (0.4 tok/s) and was asked to wait for PrismML.
  `fork:<owner>/<name>
  [@ref]` (`srcref.ts`, typed as the fork's URL into the
  PR box) is a codeload tarball — `HEAD` = the fork's default branch — so it
  needs no git, always re-fetches, records the FORK's commit (`movingSha`), and
  Update follows the fork, never an upstream tag (which would swap in a runtime
  that rejects the model). Source route only: the fork's Linux CUDA release
  links `libcudart.so.13`/`libcublas.so.13` and ships neither, while a source
  build gets them from the app's toolkit through RUNPATH. Verified 2026-09-18:
  built in 3 min, native `sm_120a`, Ternary-Bonsai-27B Q2_0 at 262k on two
  cards, coherent output, 51.6 tok/s. `gguf.rs` sizes all five newer types
  (NVFP4, Q1_0, Q2_0, PQ2_0, PTQ1_0) — an unknown id bills its tensors at 0
  bytes.
- **A build must not borrow from the source cache.** cmake's build-tree RUNPATH
  points into `cache/sources/<ref>/build/bin`, and the install copied
  `libllama-server-impl.so.0` but not its `.so` symlinks — so a build worked
  until the next build of that ref replaced the tree, then died with
  `error while loading shared libraries`. Builds now pass
  `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`, the install recreates symlinks, and
  `finalize` runs `ldd` and REFUSES a build that resolves anything into the
  cache or to "not found" (`borrowedLibs`). An existing build that fails its
  `--help` probe that way is `builds.broken[id]` and Start is blocked with the
  reason — rebuild it. Verified by hiding the cache and running the fork build.
- **A model can need a specific runtime, and the header says which.** `gguf.rs`
  sets `vendor` ("prism") from a `prism.*` key or a type 142/143;
  `src/lib/runtime.ts:runtimeMismatch` blocks Start on any upstream build and
  offers one button — switch to the vendor build already installed, or build it.
  Another person's fork is `unknown` and never blocked: the app cannot see into
  it. A Hadamard-folded `Q2_0` loads on upstream and answers GARBAGE, which is
  why this is a block and not a warning.
- **Strata is a second ENGINE behind the same Start button, not a port.**
  `Niko1221/Strata` runs one model family — Qwen3.8-Flash-Next — with a
  per-expert VRAM cache, its own kernels and an MTP draft layer, where llama.cpp
  streams every routed expert from RAM. Measured 2026-10-08, the same UD-Q4_K_XL
  file, two 24 GB Blackwell cards, 4 GB reserved on each, the app's own bench
  prompts, warm: **53-64 tok/s against 11-15, and a 28,863-token prompt read at
  2,690 tok/s against 159** (41-54 without the draft layer). Built and started
  THROUGH the app: 80 s to build, ~110 s to load, 51-69 tok/s at a 131,072
  context.
  - **It is a build like any other** (`fork:Niko1221/Strata`,
    `Build.engine ===
    "strata"`), and `src/lib/strata.ts` is everything the
    app decides about it: which files it takes (`strataModel`, anchored on the
    first shard's exact name — Strata refuses every other quant, and a loose
    match would offer an engine that then says no), the argv, the OFFER on a
    llama.cpp build (`strataOffer` — never a block, llama.cpp runs the file
    correctly) and the BLOCK on a Strata build with any other model
    (`strataMismatch`).
  - **What the app spawns is a launcher it wrote** (`llama-master-strata`, in
    the build directory, so the builds-root sandbox holds): it runs Strata's
    `setup.py` for the model — a rewrite of the config when nothing else changed
    — then `exec`s the Python server. One pid, one log, and the argv on screen
    is the argv that ran. `srv.server.ts` recognises the orphan by
    `/serve/server.py` because the `exe` is an interpreter outside the root.
  - **Three things its installer would do and must not here.** `sudo apt-get`
    for Python or CUDA: a `sudo` that refuses is first on the launcher's PATH,
    and `STRATA_NVCC` names the app's own toolkit. Download the model:
    `--gguf-dir` hands it the files on disk — the one fetch is the draft layer,
    **6.5 GB**, stated on the offer before the button. Fill the display card:
    `--vram-reserve-mib` carries the user's reserve (one number for every card,
    so it is the display card's).
  - **Packs and the draft layer live in `cache/strata/`, not in the build**, so
    an engine update does not fetch 6.5 GB again. Built IN PLACE, unlike a
    llama.cpp build: a venv bakes absolute paths into every script, so
    stage-then-rename breaks it. Debian's `python3` has no `ensurepip`; the venv
    falls back to `uv`.
  - **Strata is not llama-server, and three callers had to learn it.** No
    `--help` vocabulary (`builds.probe` skips an `engine` build — its launcher
    would answer by running a setup), no tuner and no fit ladder (`-c` and
    `--n-cpu-moe` are flags this command does not have), and no `/completion`:
    both `probe` and `bench` fall back to `/v1/chat/completions` on a 404, which
    returns the same `timings`. Without the probe's half a run that generated
    fine stayed "unproven" for ever.
  - **A Strata run is described from ITS OWN argv and from measurement, never
    from a llama.cpp plan of the same file.** The first version left every
    llama.cpp reading in place and each one was wrong in its own way: `drift`
    announced "something else has taken memory" about memory the run itself held
    (Strata fills every card and copies its experts into RAM by design, so
    against a llama.cpp plan it is "over" from second one); the Setup chips said
    `MoE→RAM · q8_0 · np 1`; the status tooltip said `TypeError: fetch failed`
    for the whole load, because Strata opens its port only once the model is in
    where llama-server answers 503; and `srv.rss` read the Python WRAPPER — 0.2
    GB for a run holding 83 — so the map drew the engine's memory as somebody
    else's. Now `derive.ts:strataView` is the one source (the running argv via
    `strataRunOf`, else what Start would spawn): Setup rows, the load bar's
    total, the memory forecast, the live figures and the Start check all read
    it; `driftNow` is `none` on an engine build; `rss` sums the process TREE
    (per thread — a child belongs to the thread that spawned it); the spawn
    records `cardFreeB` so each card's share is measured.
  - **Its footprint is an estimate of a different kind** (`strataNeeds`): VRAM
    is "whatever is free, less the reserve on each card" (predicted 34.7 GB,
    took 35.6) and RAM is 0.75 of the FILE (111.3 GB on disk, 82.8 GB resident,
    79.6 of it anonymous). That RAM is not a mapping the kernel can drop, so
    `strataRamShort` BLOCKS Start when less is free — the alternative is the OOM
    killer picking a neighbour. One file on one machine; every use says "about".
  - **On a Strata build the page drops what it cannot honour**: the placement
    picker, the quantisation hint, the tuner's switch, the thinking switch,
    stability and the llama.cpp catalog. Context is Strata's six sizes as
    buttons (`STRATA_CONTEXTS`), the reserve boxes stay — they are the one
    setting both engines share.
  - **Still open.** The memory MAP's bands inside a card (weights / KV /
    compute) are apportioned from a llama.cpp plan; only the card totals and the
    RAM figure are measured. The Tune and Server pages still show llama.cpp's
    controls. A failed Strata rebuild leaves it unlisted until the next good
    one.
- **Upstream ships Linux CUDA again (b11039+), in two archives.** The binary
  (`llama-bNNN-bin-ubuntu-cuda-13.3-x64.tar.gz`) and its runtime
  (`cudart-<same name>`; Windows: `cudart-llama-bin-win-cuda-X.Y-x64.zip`, no
  build number) — `assets.ts:companionAsset` pulls the second into the same
  directory, whose `$ORIGIN` RUNPATH finds it. `pickAsset` never takes a CUDA
  newer than the DRIVER (`Gpu.cudaDriver`, from `nvidia-smi --version`, whose
  "CUDA version : Deprecated" line must not match).
- **The parent's `LLAMA_ARG_*` would silently override the command.** llama.cpp
  reads `LLAMA_ARG_<FLAG>` for every flag the argv leaves out, and the app omits
  flags at their default — so a stray export changed the run while the screen
  showed another. `envvars.ts:spawnEnv` spawns with `clearEnv` and the parent's
  environment minus `LLAMA_ARG_*` (the user's own Command-panel variables still
  win) and logs what it dropped.
- **`-lv 4` is always asked for.** At llama.cpp's default verbosity 3 every
  `load_tensors`, buffer-size and `sched_reserve` line is hidden — measured on
  b10144, b10151, master and the fork (~22 startup lines at 3, ~210 at 4). Those
  are what `loadprogress.ts` reads and what anyone reads to see where memory
  went. Catalog entry `logVerbosity` (`def 4`, `llamaDef 3`).
- **A default that moved upstream is reset once, not kept for ever.** `cfg`
  persists every setting, so `-to 600` and `--jinja` off outlived the catalog
  change that fixed them and were emitted on every run.
  `params.ts:
  FORMER_DEFAULTS` lists values that were once the app's default;
  `cfg` version 6 drops a stored setting equal to one of them. Not gated on
  `touched`: the tuner writes through the same path, so `touched` cannot tell a
  choice from an old default.
- **Per-layer KV follows llama.cpp's own per-layer rules.** Gemma-4 declares
  per-layer `head_count_kv` arrays and a separate `key_length_swa`, and the
  sliding-window pattern is an ARRAY key when present; a model that states no
  pattern gets its arch's hardcoded default (`plan.ts:SWA_DEFAULTS`, from
  `load_swa_pattern` in llama-model.cpp) and otherwise none. A windowed layer
  holds `pad256(window × streams + ub)` cells, not `ctx`. Gemma was billed 7.5×
  UNDER before; `kvByLayer` is the one source and the per-card KV bars sum it
  over the layers each card actually holds.
- **Only an immutable ref may be cached, and `master` is not one.** The source
  cache was keyed on the ref NAME and reused whenever `CMakeLists.txt` existed,
  so the first `master` build pinned that machine to that day's master for ever
  — the developer's own cache held a master five weeks old, 112 files behind
  (new `fattn-swizzle`, `moe-weighted-reduction` and `mmq-config-*` CUDA kernels
  among them), while the log cheerfully said "Reusing cached source". That is a
  build compiling something other than what its name claims, which is the exact
  class this app exists to refuse. `refMoves()` decides: a tag is reused for
  ever, `master` and any PR ref are re-fetched every time — 37 MB against a
  compile measured in minutes. A moving build also records the master sha it saw
  and prints `refProvenance` into the log, because "master + PR #27754" means
  something different tomorrow.
- **Speed is MEASURED now, not only estimated.** Everything the app said about
  tokens per second divided bytes by an effective bandwidth that cannot be read
  off the machine (`speed.ts` says why), and the only observation it had was
  `chat.lastTps` — a real rate about an unknown prompt, at an unknown context
  fill, with an unknown amount of thinking in it. `src/lib/bench.ts` holds those
  variables still (fixed prompt, fixed 128 tokens, `cache_prompt: false` so the
  prefill is real work, `temperature: 0`, `reasoning_budget: 0` so a thinking
  model does not spend the whole budget thinking) and `srv.bench` runs it
  **against the server that is already up** — no reload, no second process, no
  extra memory, nothing at risk on a run that took two minutes to place. The
  numbers are llama.cpp's own `timings`; the first-token latency is measured
  here because no server metric reports it. `speedCalFromLastReply` prefers the
  bench over the chat and bills it at the context ACTUALLY filled — charging a
  near-empty run for a 262,144-token cache would calibrate a bandwidth several
  times too high. A build too old to report `timings` has measured NOTHING and
  says so: writing 0 tok/s would put "this machine is broken" on screen over a
  build's age. The result carries its model AND its context (`benchApplies`),
  because the same model at 8k and at 256k are different measurements, and it is
  cleared on every start. State key `lastBench`, not `bench` — aio refuses a
  cell whose state key shadows a method, and it is right to.
- **The bench runs TWO prompts, and the PAIR is the measurement.** It measured
  prose — chosen well, because prose runs to a steady state where a question an
  instruct model answers in a sentence measures the first batch. The consequence
  went unseen: speculative decoding pays on output that repeats itself and close
  to nothing on prose, so every speculative setting in this app was being
  measured at its WORST case, every time, while `tune.ts` told the user to go
  and measure it. The instrument could not carry out the advice. `BENCH_PROMPTS`
  is prose AND code now (`srv.benchBoth`, sequential — two generations in flight
  share the server and each would be timing a machine busy doing the other).
  Generation is bandwidth-bound and a byte of weights costs the same whatever
  the text is about, so with drafting OFF the two land within a couple of
  percent: prose is a self-calibrating baseline, and the gap that opens on code
  is the drafter's acceptance rate in tok/s, on this machine. `specVerdict`
  reads them together and refuses to speak from one — a code rate with no
  baseline under it is exactly the "2.9x faster!" claim this exists to avoid
  making. Bandwidth is still calibrated from PROSE only (`lastBench`, with
  `lastBenchCode` beside it): a drafted token is not a token the memory bus paid
  for, and calibrating off accepted drafts would report a machine several times
  faster than it is.
- **`-bs` is a real lever, and checking it against the local source cache is how
  this app got it wrong once.** `--backend-sampling` picks the next token on the
  GPU instead of copying the model's output row to the CPU every token —
  megabytes per token on a large vocabulary, and worse than its size because the
  copy is a SYNCHRONISATION POINT that stalls the device mid-loop. In the
  catalog (`params.ts:backendSampling`, `llamaDef: false`, so the default emits
  nothing). **Measured 2026-09-18 and the tuner now leaves it OFF:** on two
  Blackwell cards it added 0.2-0.8 s to EVERY request's first token (growing
  with the context) and bought no generation rate at all. The reason text says
  so; a user can still turn it on, and a build that has never heard of it is
  never handed it. A probed build that lacks it is told so, because "the same
  model is quicker on a newer build" is otherwise an unexplained difference.
  llama-server still drops it silently for a grammar, a JSON schema or a
  reasoning budget (`common/sampling.cpp`, warns in the log); `stability.ts`
  names the reasoning case, which is the one visible in the argv.
  - **It DOES stack with speculative decoding, and this file said the opposite
    for one round of work.** `tools/server/server-context.cpp` used to carry
    `backend_sampling &= !(slot.can_speculate())` — "requires multiple samples
    per batch - not supported yet" — which made the flag inert on any drafted
    run. That line is gone from master; the two now compose, and it is the
    combination worth having (drafting makes a token cheaper to produce, this
    makes it cheaper to collect).
  - **The cause is worth more than the fact: `~/.llama-master/cache/sources/` is
    whatever the last BUILD fetched, not what upstream does today.**
    `refMoves()` re-fetches `master` on every build, which is correct and is not
    the same promise — with no master build since 2026-07-27, the checkout sat
    six weeks behind while reading like the current source. It is the right
    place to check a flag's spelling, its default, and whether it exists at all;
    it is NOT evidence about behaviour that may have changed. For that, read
    `raw.githubusercontent.com/ggml-org/llama.cpp/master/<path>` and compare. A
    conclusion drawn from the cache alone was published here as a correction to
    a third party who was right.
- **A build is ASKED what it can do, and that is what lets the tuner turn
  something on.** Every default in this app is one a stale llama.cpp can safely
  ignore — until `-bs`, which an older binary refuses outright with
  `unknown
  argument`. Inferring support from the version is not available
  either: a release is a `b`-number, `master` is a day, and a PR stack has no
  version that means anything. So `llama-server --help` is run once per build
  (`builds.probe` → `builds.server.ts:probeCaps`, sandboxed to the builds root
  the same way a start is, 5 s ceiling, both pipes because usage has gone to
  stderr before now) and `src/lib/caps.ts` parses it. The parse splits each
  option line at the first run of two-plus spaces NOT followed by another flag —
  by column would be a guess about a layout upstream may change, by "two spaces"
  alone would cut `-t,    --threads` in half — so a description that mentions
  `--threads` can never be mistaken for a build that declares it. Verified
  against a real b10151 binary: 410 flags, no placeholders, and it correctly
  reported `--lazy-mode` ABSENT, which is the whole point.
  - **An unprobed build supports nothing** (`supportsFlag(null, …) === false`).
    The tuner then leaves a lever on the table, which is recoverable; the other
    reading — "assume modern" — writes a command the binary cannot parse.
  - `caps` is persisted per build id and dropped when the build leaves disk, so
    a removed-and-rebuilt id is never answered from its predecessor's flags.
    Only the ACTIVE build is probed: a process per installed build, per scan,
    would be spent on builds nothing is going to run.
  - The stub llama-server answers `--help` and exits for this reason. It did
    not, and every `setActive` in the UI suite hung until the probe timed out —
    23 s of suite became 95 s, which is exactly what a wedged binary would do to
    the app.
- **A multi-token-prediction head does not always live in the model file.**
  `tune` keys `draft-mtp` on `meta.nextnLayers > 0`, read from the header — and
  Gemma 4 publishes its heads as a SEPARATE GGUF beside the weights
  (`…-mtp-Q4_0.gguf`), passed to llama.cpp as a second model path, not as a
  header key. So the header was honest, the tuner was honest, and the sentence
  on screen was "this model ships no multi-token-prediction block, so there is
  nothing to draft with for free" — printed over a 2-3x speed-up sitting
  unopened in the same directory. Not a crash: worse, because nothing said
  anything. `src/lib/mtp.ts` finds it (pure — the scan is handed in, because a
  function that read the disk could not be tested against the naming shapes),
  matching `mtp` as a whole TOKEN and requiring the rest of the name to pair
  after quantisation labels are stripped: `…-Q8_0` beside `…-mtp-Q4_0` is the
  normal case, while "any file with mtp in it" would pair a Gemma drafter to a
  Qwen model — llama.cpp loads that, the vocabularies differ, every draft is
  rejected, and the result is a SILENT slowdown. Attached only up to
  `MTP_SIBLING_MAX_B` (2 GiB): a draft model's weights are VRAM `plan.ts` does
  not bill, and a "drafter" larger than a head is a mispairing or a second full
  model. Above it the tuner names the file and lets the user decide. `TuneOpts`
  is the trailing argument that carries it, because `src/lib/` is pure and
  finding the file is I/O.
- **"Would a smaller quantisation be faster?" is the app's biggest speed lever
  and it used to be silent about it.** A smaller quant was named only in
  refusals ("try a smaller quantisation"); a model that FITS was never told that
  half the bits per weight is close to twice the rate. `src/lib/quant.ts`
  answers it, and the answer has an exact half and an estimated half. Exact:
  this file's real bits per weight, `tensorBytes * 8 / params`, counted off its
  own tensor table — `params` was added to `rust/src/gguf.rs` for this, because
  the LABEL is a mix (a "Q4_K_M" is mostly Q4_K with some Q6_K) and two files
  wearing it are not the same size. Estimated: what a file of some other quant
  would weigh (`TYPICAL_BPW`), since that file is not on this machine. What is
  NOT estimated is the consequence — each candidate is rescaled and run through
  the real `tune` → `plan` → `bytesPerToken` → `estimateTps`, on this machine,
  with this reserve. That is the whole point: the interesting row is not "half
  the bytes, twice the speed", it is the one where the model stops spilling into
  system RAM and goes four times faster, and a ratio cannot find it (pinned in
  `tests/lib.test.ts`). Two bugs this found, both fixed and pinned: a current
  file that does not fit has no rate to be a multiple of and printed "about 0.0×
  faster"; and the fastest row is always the SMALLEST file, so recommending it
  recommended the most damaged model every time — `recommend()` aims at the
  quality knee (`SAFE_BPW`, 4.5 bits) instead, and the table carries the rest
  with the price on each row.
- **A reasoning model's thinking is a speed control, and it was invisible.**
  `--reasoning off` removes most of a thinking model's tokens, which is a bigger
  win on a short question than any kernel flag — and it shipped as an advanced
  server flag filed next to Prometheus metrics. `ThinkSwitch` (all-in-one and
  Tune, beside the LAN and priority switches) has three states because `auto` is
  the honest default: the model's own template decides, and forcing `on` for a
  model with no reasoning mode is a request llama.cpp cannot honour.
- **Speculative decoding: MTP is taken, n-gram is NAMED, a draft model is
  offered.** `draft-mtp` is switched on whenever the model ships the block — the
  rare optimisation with nothing to weigh. The `ngram-*` kinds need no second
  model and no MTP block, so they are available to every model, and they are
  deliberately NOT switched on: a rejected draft is work thrown away, they pay
  on repetitive output and cost a little on prose, and nothing here knows what
  the user is about to ask for. This app does not enable what it has not
  measured — so the tuner names the option, says what decides it, and points at
  the Speed panel, which settles it in fifteen seconds. `-md` (a small GGUF of
  the same family) is in the catalog with a `stability.ts` caution: it loads a
  second set of weights and `plan.ts` sizes only `-m`, so those bytes are real
  and unbilled.
- **Host memory bandwidth is reported, never diagnosed.** A low `ramBps` is the
  one machine number a user can act on — DDR5 sold as 6000 boots at 4800 until
  EXPO/XMP is switched on, which is about a fifth of every model with experts in
  RAM. But the DIMMs cannot be read without root (`dmidecode`, `lshw`), so
  `bandwidthNote` states wide bands as "what machines like this usually reach"
  and names something to go and look at rather than asserting what will be
  found. Silent unless the running placement actually reads host RAM per token:
  on a VRAM-only run the figure is true and irrelevant.
- **A draft is not shared state.** The chat box was `value={chat.input}` with a
  `setInput` dispatch per keystroke, and that is a controlled input over a
  REPLICATED field: every key was a round trip, and while a reply streamed the
  60–500 ms `partial` flushes re-rendered the box with whatever the server had
  last acknowledged — keys still in flight were wiped, and only while the model
  was answering, which is exactly how it was reported. aio's own rule
  (`docs/state/real-time.md`): if two clients or a restart never need to agree
  on it, it does not belong in a server cell. So the draft is a module-level
  `signal` in `ChatComposer` (survives the Chat-tab ↔ all-in-one switch) and
  reaches the cell as an ARGUMENT — `chat.submit(url, text)` and
  `chat.send(url, text)`; `chat.input` no longer exists. The box is a textarea
  now (Enter sends, Shift+Enter breaks a line, height follows `draftRows`),
  because what gets pasted into a chat with a model that writes code is code.
  Every other text and number box bound to a cell (settings, reserve, build
  jobs, system prompt) goes through `kit.tsx:DraftInput`: the draft is local
  while the box has focus and the cell sees ONE write on `change`. The reserve
  boxes are in the auto-tune key, so typing "16" used to re-tune and persist at
  "1" and again at "16". In a `testUI`, `setValue` fires only `input` — `blur()`
  is the commit. Pinned in `tests/ui.test.ts` (the chip test types while a reply
  streams and asserts the text survives a chat-cell write). The LAN client
  (`client/src/ui/Composer.tsx`) follows the same rule.
- **The planner chain is memoised, and the page root reads no chat.** Every
  derived plan in `src/ui/derive.ts` from `currentStatePlan` to `projectedSpeed`
  (and `maxTunings`, the two hunted contexts `CtxControls` draws) is a
  `computed()` behind a plain function. It was not, and the all-in-one root —
  which read chat, hw and srv — ran `tuneAll` three times per render
  (`placements` → `projectedSettings` → `projectedStatePlan`, then
  `perTokenBytes` again) at ~14 ms each on a 94-layer MoE: ~100 ms of JS on
  every streamed-token flush, plus every 1 s hw and srv tick. That was the
  second half of "typing is slow while it answers". The four things the root
  read from chat (tok/s pill, error line, copy/clear, the live table's speed)
  are their own components now (`OnePage.tsx:TpsPill` etc.), so a flush
  re-renders four small things and not the page. `computed`, never a hand-rolled
  cache: a cache hit that skips the read skips the subscription.
- **A boolean whose upstream default is ON needs `llamaDef: true` AND an
  `offFlag`.** Three switches shipped without: `--jinja`, `--slots` (both ON
  upstream — `use_jinja`, `endpoint_slots` in `common.h`) and context shift
  (whose default FLIPPED to off, with the pair now `--context-shift` /
  `--no-context-shift`). As `def: false` with no `offFlag`, "on" emitted a flag
  that changed nothing and "off" emitted nothing and changed nothing — the
  server ran the same in both positions while the panel claimed a choice, and
  `/slots` (which shows every prompt in flight) was up the whole time under a
  tip saying it leaks. `stability.ts` now names it on an open bind. Also in that
  pass: `--defrag-thold` is deprecated upstream (value ignored) and left the
  catalog; `--cache-reuse 256` is emitted (off upstream, 256 in every upstream
  preset, auto-disabled where the cache cannot shift); `-to` is upstream's 3600
  again (600 was shorter than a 250k-token prefill); `--reasoning` /
  `--reasoning-budget` and the `ngram-*` speculative kinds are in the catalog.
  `cfg` is `version: 3` and drops the two orphaned keys. Re-check after any
  llama.cpp bump — the app's own checkout is at
  `~/.llama-master/cache/sources/master/common/{arg.cpp,common.h}`.
- **A crash is processed once.** `srv.poll`'s not-running branch is guarded on
  `s.pid === 0`: the pid is zeroed at the end of that branch, and without the
  guard every later tick re-diagnosed the same dead run — a fresh `diagnosis`
  object broadcast per second for as long as the app sat on a crash, `clearLog`
  undone by the next poll — and, in the gap between a Start being dispatched and
  its spawn (status "starting", no process yet), ran the fit ladder against the
  PREVIOUS run's lines. Pinned in `tests/server.test.ts`.
- **`exec` takes a ceiling, and `nvidia-smi` absence is remembered.** A wedged
  `nvidia-smi` (after suspend, typically) blocked the 1 s hardware refresh
  forever with no error anywhere; `hw.server.ts:nvidiaSmi` is the one door, with
  a 5 s timeout (exit 124) and a 60 s memory of "not installed" so an AMD or
  Apple box does not spawn a process a second to be told so again.
- **The ROCm group step is checked by asking bash.**
  `${SUDO_USER:-$PKEXEC_UID_NAME:-$USER}` read like a fallback chain and was not
  one — bash parses the nested default as the literal word
  `$PKEXEC_UID_NAME:-$USER`, and pkexec sets no such variable (it sets
  `PKEXEC_UID`, a number) — so under the app's preferred `pkexec` path step 7 of
  7 ran `usermod … :-root` after the driver was in. It is
  `$(id -un "${PKEXEC_UID:-${SUDO_UID:-$(id -u)}}")` now, and
  `tests/lib.test.ts` runs it through bash under all three.
- **`.slice()`, not spread**, on live async state arrays.
- Tests live in `tests/`, never beside their source.

- **The desktop goes first, by default.** llama.cpp will take every core and
  every spare IOPS, and on the machine it runs on that reads as "the computer is
  broken". "Low priority" (ON by default, `cfg.lowPriority`) puts the process at
  nice 19 in the idle I/O class — it gets everything nothing else wants, which
  on an idle machine is everything. Two queues, because CPU is not the half that
  hurts: reading 145 GB of weights off an NVMe stutters a desktop whatever the
  nice value is (`src/lib/priority.ts`).
  - **Applied to the PID after the spawn, never by wrapping the command.**
    `nice <bin> …` would put `/usr/bin/nice` at the front of the argv, which
    breaks two promises at once: the command on screen would stop being the
    command that runs, and `srv.server.ts` refuses any binary outside the builds
    root — a sandbox rule, not a formality. `renice`/`ionice` by pid keep both,
    for the price of a few milliseconds at normal priority during a load that
    takes a minute. Every fit-ladder rung is a fresh process and is reniced too,
    from `srv.runLowPriority` (the RUN's choice, not the toggle's current
    position two minutes later).
  - **It degrades rather than fails.** The idle I/O class is refused on some
    kernels and in containers, so the fallback is the lowest best-effort band; a
    machine with no `ionice` at all still gets the renice. Whatever happened is
    one line in the server log, including "could not lower the priority" — a run
    left at normal priority while the switch says otherwise is the kind of
    silent disagreement this app refuses everywhere else. `tests/server.test.ts`
    reads `/proc/<pid>/stat` back and asserts the KERNEL agrees, both ways.
  - Turning it off while a server runs says "takes effect on the next start",
    because lowering a priority needs no privileges but raising one back does.
- **"Available on LAN" is one flag, one switch, two pages.** llama-server binds
  to 127.0.0.1 unless told otherwise, so it is invisible from every other
  machine — the commonest reason the client (`client/`) finds nothing. The
  switch writes `--host` through the catalog like any other setting
  (`src/ui/LanSwitch.tsx` → `cfg.set("host", …)`), so the command strip, the
  argv that is spawned and the switch cannot disagree. OFF by default: binding
  to the world is not a default. What it adds beside itself is the ADDRESS —
  `0.0.0.0` is what llama-server binds, not what anyone dials, and typing it
  into the client reaches nothing (`src/lib/lan.ts:pickLanIp` over `hw.lanIps`,
  a real LAN address ahead of a link-local one). What it does NOT repeat is the
  risk: an open bind with no API key already raises a red banner on the same
  page (`stability.ts:189`), and saying it twice makes both quieter. It lives in
  its own `run-row`, not in the actions row — there its address line wrapped
  into a narrow column and squeezed Start against it.

## The client (`client/`)

A second, standalone aio app in this repository: a LAN chat client for a
llama.master somebody else is running (`.katana/client.md`). `deno task dev`,
`deno task verify` and `deno task am` all work from inside `client/`.

- **It watches; it never operates.** Everything it shows comes from the far
  end's own endpoints — `/props` (what is loaded), `/health` (is it up),
  `/metrics` or `/slots` (how busy), `/v1/chat/completions` (the conversation).
  There is no start, no stop, no settings, and a UI test asserts that no other
  path is ever requested.
- **Discovery is a sweep, because llama-server does not announce itself.** One
  /24 (this machine's own subnets, private ranges only), the four ports
  llama.cpp is served on, 64 probes in flight, localhost first — and the
  identifying answer is `/props`, not a 200, or every router admin page on the
  subnet would be reported as a server (`client/src/lib/discover.ts`).
- **The commonest LAN failure has a sentence, not a shrug.** llama.cpp binds to
  127.0.0.1 unless told otherwise, so it is invisible from every other machine;
  both "unreachable" and "nothing found" name `--host 0.0.0.0`.
- **An absent reading is not a zero.** `--metrics` is off by default, so
  occupancy is three-valued: a number, or "not reported (server does not publish
  it)". A client that renders 0% because it could not ask is lying.
- **The shared libraries are COPIED, and a test polices the copy.** aio serves
  the browser bundle only from inside the app's own root and refuses a symlink
  out of it (`server-static.ts`), so `client/src/shared/` is a mechanical copy
  of `src/lib` made by `deno task sync`; `client/tests/shared.test.ts` fails the
  moment the two differ, naming the command that fixes it. Same arrangement as
  `src/llama-sys.wasm`: committed so nothing has to be built, guarded so it
  cannot fall behind.

## Katas

`.katana/*.md` are the quality specs; `/use-katana` audits against them. Field
reports on the framework go to `feedback/llama-master.md` in the aio CHECKOUT
(`/home/dev/code/gen/aio`) — that file is required by `.katana/_aio.md`, which
spells the path `dep/aio/feedback/…`, and that spelling stopped resolving when
this app moved from a path pin to a version pin: a provisioned release worktree
ships no `feedback/` directory. Update it whenever aio gets in the way.
