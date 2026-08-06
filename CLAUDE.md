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
  stability, the bars, the per-card packing budgets — inherits it without knowing
  it exists.
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
    `Gpu.display` comes from `nvidia-smi --query-gpu=display_mode,display_active`
    (cached 30 s — it changes when a monitor is plugged in, not on the 1 s poll;
    `display_mode` is deprecated on current drivers and returns a sentence, hence
    reading both) and, for sysfs cards, `/sys/class/drm/<card>-*/status`.
    Verified against this machine: nvidia-smi index 0 = PCI 01:00.0 = `card2`,
    which is the one with two connected DisplayPort outputs. `undefined` (a
    vendor that does not report, no DRM connectors) falls back to card 0 and the
    UI says it is an assumption; every card answering `false` is taken at its
    word and reserves nothing (`reserve.ts:displayGpus`).
  - It is labelled apart from "in use elsewhere" everywhere it is shown
    (`Pool.reservedB`), because a refusal caused by the user's own setting has to
    name the control that gives the memory back. In the memory MAP that means a
    band with a colour of its own (teal, `--seg-reserved`, still hatched because
    a decision must not look like a measurement) and an entry in the legend: it
    was drawn all along in the machine's greys and named nowhere, so the one
    band the user put there themselves read as empty track — which is the one
    thing it is not. The region foot counts it as a third figure, because
    reserved bytes are neither used (nothing is in them) nor free (nothing may
    go in them). `hwSnapshot` never carries it —
    the current-state view reports real free memory; reserved bytes are free until
    something takes them, they are merely not SPENDABLE.
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
  drifted (tok/s above the answer on one, below on the other; the
  "ended while still thinking" fallback on only one), which is what two copies
  of a message renderer always do. tok/s belongs AFTER the answer — it is a
  measurement of the thing above it, and printing it in the role line put a
  number the user cannot have yet over the text they are waiting for.
- **One Memory section, and what is in it depends on whether the answer is
  already known.** Nothing running: both maps, now and next, because that IS
  the decision. A model running: only the measurement — the projection's
  question has been answered by the machine itself, and an estimate beside the
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
- **`-ngl` counts the output head, and never moves the embeddings.** There are
  `nLayer + 1` slots, so `-ngl 43` on a 43-layer model offloads layers 1..42 AND
  the output, leaving layer 0 on the host; only `-ngl > nLayer` takes every
  layer. The token embedding table is an INPUT tensor and llama.cpp pins those
  to the CPU at any `-ngl` (`dev_input`, "very little benefit to offloading the
  input layer"), so billing it to VRAM spent ~1 GB of a card's budget on bytes
  that were never going there. Both rules live in `devsplit.ts:offloadRange` and
  `plan.ts` reads them.
- **`--mlock` and `--no-mmap` are ONE setting.** Both assign `params.load_mode`
  (`common/arg.cpp`), so emitting both is not "locked and unmapped" — it is
  whichever came last, silently, while the app prints a reason claiming the
  other. With routed experts on the host the tuner now emits NEITHER flag —
  llama.cpp's mmap default — and it was measured: `--no-mmap` copies the whole
  145 GB file on every start (148 s cold, 160 s even warm, because its own copy
  evicts the page cache), while mapped the same start is 73 s cold and **6 s
  warm**, and generation is slightly FASTER mapped (9.6 against 8.9 tok/s, same
  CUDA build, same cards). Every fit-ladder rung reloads the model, so this is
  also what makes the retry ladder affordable. `--mlock` is not emitted there
  either: it would ask to pin more than stock `RLIMIT_MEMLOCK` allows (23 GB on
  the machine that motivated this, against ~110 GB of experts), and llama.cpp
  would warn and run unpinned — a flag whose stated effect does not happen. A
  test in `tests/lib.test.ts` fails if any placement emits both flags.

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
    (`--n-cpu-moe 36 -np 1 -ts 37.5,6.5`) starts and generates at 13.1 tok/s;
    at 524,288 (`--n-cpu-moe 32`) 14.9 tok/s.
- **A catalog default is not llama.cpp's default, and assuming so shipped two
  silent lies.** `command.ts` omits a flag whose value equals `def`, on the
  theory that `def` IS what llama.cpp does without it. Upstream moved:
  `-ngl` now defaults to **auto**, so "CPU only" emitted no `-ngl` and llama.cpp
  offloaded to the GPU anyway; `-c` defaults to **0 = take it from the model**,
  so a plan drawn for 4,096 tokens started a server at this model's declared
  1,048,576 and could not allocate — a start that cannot succeed, with an error
  naming none of it. `Param.llamaDef` (`types.ts`) carries llama.cpp's own
  default when it differs, and omission is judged against THAT. The three flags
  that decide the placement — `-ngl`, `-c`, `-np` — are always emitted, because
  their whole job is to pin what runs. Re-check after any llama.cpp bump: the
  failure mode is silent, and `llama-server --help` prints every default.
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
  vitals, both memory states and the command — so anything added there has to
  be paid for. It was paid for once already: the vitals lost their sparklines
  and two sub-lines apiece (the history graphs are one click away on the pages
  built for them), the current-state table is drawn only when something IS
  running (idle, every row of it is a zero the map above already shows), the two
  maps share one legend and drop the "Memory map — 234 GB total" caption they
  both repeated, and the command shows the server line wrapped rather than one
  flag per line. Verified the way layout has to be verified — by looking:
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
  `/` only, so a Windows-spelled `..\..\evil` walked straight through.
- **Fail loud.** A missing binary, an unreadable header, a 404 — surface it in
  `lastError` and render it. Never swallow.
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
    kernels and in containers, so the fallback is the lowest best-effort band;
    a machine with no `ionice` at all still gets the renice. Whatever happened
    is one line in the server log, including "could not lower the priority" —
    a run left at normal priority while the switch says otherwise is the kind of
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
  into the client reaches nothing (`src/lib/lan.ts:pickLanIp` over
  `hw.lanIps`, a real LAN address ahead of a link-local one). What it does NOT
  repeat is the risk: an open bind with no API key already raises a red banner
  on the same page (`stability.ts:189`), and saying it twice makes both
  quieter. It lives in its own `run-row`, not in the actions row — there its
  address line wrapped into a narrow column and squeezed Start against it.

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
  /24 (this machine's own subnets, private ranges only), the four ports llama.cpp
  is served on, 64 probes in flight, localhost first — and the identifying answer
  is `/props`, not a 200, or every router admin page on the subnet would be
  reported as a server (`client/src/lib/discover.ts`).
- **The commonest LAN failure has a sentence, not a shrug.** llama.cpp binds to
  127.0.0.1 unless told otherwise, so it is invisible from every other machine;
  both "unreachable" and "nothing found" name `--host 0.0.0.0`.
- **An absent reading is not a zero.** `--metrics` is off by default, so
  occupancy is three-valued: a number, or "not reported (server does not publish
  it)". A client that renders 0% because it could not ask is lying.
- **The shared libraries are COPIED, and a test polices the copy.** aio serves
  the browser bundle only from inside the app's own root and refuses a symlink
  out of it (`server-static.ts`), so `client/src/shared/` is a mechanical copy of
  `src/lib` made by `deno task sync`; `client/tests/shared.test.ts` fails the
  moment the two differ, naming the command that fixes it. Same arrangement as
  `src/llama-sys.wasm`: committed so nothing has to be built, guarded so it
  cannot fall behind.

## Katas

`.katana/*.md` are the quality specs; `/use-katana` audits against them. Field
reports on the framework go to `dep/aio/feedback/llama-master.md` — that file is
required by `.katana/_aio.md` and should be updated whenever aio gets in the
way.
