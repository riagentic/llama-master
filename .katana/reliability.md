# Reliability kata — loading must just work

Status: **the tuner's own proposal now starts and answers.** DeepSeek-V4-Flash
(145 GB, 2x24 GB + 186 GB RAM):
`-ngl 999 --n-cpu-moe 34 -ts 36.5,7.5 -c 17920
-fa on -ctk q8_0 -ctv q8_0 --no-mmap`
→ healthy, 9.0 tok/s, cards at 20.4 and 22.5 GB. Items 1, 4, 5, 6 below are
still open. This file is the brief and the measured ground truth, so the work
does not restart from zero.

## The bar

1. **Reliable** — the model loads and answers, first time, out of the box.
2. **Workstation-friendly** — a few GB of VRAM left for the desktop; launching a
   browser or a non-demanding game must not kill the run.
3. **Fast** — as much of the hot path on the GPU(s) as possible, balanced across
   them.

In that order. **Undercut rather than fail**: a conservative setting that starts
beats an optimal one that does not.

## Measured facts (this machine, `source-master-cuda`)

Do not re-derive these; they cost hours.

| what                 | value                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| model                | 43 layers, 256 experts (6 used), MXFP4, 4 shards, 145.6 GB                                                               |
| declared context     | 1,048,576 (YaRN 16x over a native `original_context_length` of 65,536)                                                   |
| dense per layer      | ~0.15 GB · experts per layer ~3.19 GB · embd 0.99 GB · output 0.99 GB                                                    |
| compute buffer @ 1M  | **68.5 GiB** (planner predicted 730 MB)                                                                                  |
| compute buffer @ 64k | 1.25 GiB                                                                                                                 |
| KV @ 1M              | **18 GiB** (planner predicted 183 MB)                                                                                    |
| **works**            | `-ngl 999 --n-cpu-moe 38 -ts 40.5,3.5 -c 32768 -fa on -t 14 -tb 14` → 8.1 tok/s, card0 21.4 GB, card1 8.5 GB, RAM 110 GB |
| 65,536               | measured to FAIL (compute buffer)                                                                                        |
| llama.cpp `-fit`     | **segfaults** on this model (`common_params_fit_impl`) — cannot delegate                                                 |
| `-fitt MiB,MiB`      | per-device margin flag, exists and is the right knob IF `-fit` is ever fixed                                             |
| n_parallel           | defaults to **4**, `kv_unified = true` — affects SWA cache sizing                                                        |
| deepseek4 SWA        | `set_swa_pattern(0)` = ALL layers SWA at n_swa=128; base cache gets 0 layers                                             |

## Already fixed (keep — each has a regression test)

- Split GGUF read as one model (`src/lib/shards.ts`); a partial set is an error.
- Per-card byte-aware split (`src/lib/devsplit.ts`); llama.cpp divides layers by
  COUNT, so `--n-cpu-moe` piles the heavy tail on the last card.
- `-ngl` counts the output head; the embedding table is never offloaded.
- `--mlock` and `--no-mmap` are one setting — never emit both.
- Retry ladder (`src/lib/fitladder.ts`) + `cfg.fitCtx` memory of what worked.

## What is still wrong

1. **The ladder is the wrong shape.** Six whole-model loads (~2 min each) is a
   twelve-minute "loading" with no feedback. It must not be the primary
   mechanism — the OPENING BID must be right.
2. **The tuner's OBJECTIVE is wrong, not just its arithmetic.** The
   context-scaled compute term now exists and is calibrated
   (`plan.ts:computeScratch`, gated on `meta.indexerTopK`, which
   `rust/src/gguf.rs` now reads). It is deliberately NOT wired in: charging it
   makes every number honest and every proposal WORSE, because `bestCtx`
   maximises the context first — so the search buys context by evicting layers
   and settles on `-ngl 1` at a 654,848 context. That fits, and runs at CPU
   speed. Tried and reverted; do not re-attempt without item 2b. 2b. **Fix the
   objective first.** Optimise for GPU residency (speed), with the context as a
   constraint, not the reverse. Open at a defensible context — the header's own
   `rope.scaling.original_context_length` (65,536 here) is a fact, the
   YaRN-extended 1,048,576 headline is not — then climb only if it fits with the
   layers still on the card. THEN wire `computeScratch` in.
3. **No progress.** The UI shows "Loading model" for minutes with nothing else.
4. **No copy-paste** for the command or the log.
5. **The e2e test does not pass** (`tests/deepseek.e2e.test.ts`): the harness
   gets a false healthy from a stale listener on the port. Bind-check the port
   before each attempt and verify the health response comes from THIS child.

## Plan, in order

1. **Analyse once, remember.** A named phase — "Analysing optimal configuration"
   — that resolves a model to a known-good config and caches it by model+machine
   (extend `cfg.fitCtx` to a full record). Never analysed twice.
2. **Open low, climb.** Start conservative (native context, not the YaRN
   headline; experts to RAM), confirm it starts, then optionally probe upward.
   Inverts the current fail-downward ladder into a succeed-upward one.
3. **Context-scaled compute estimate** for sparse-attention models, so the
   opening bid is defensible rather than 100x optimistic.
4. **Status that tells the truth**: phase, attempt N of M, bytes loaded,
   elapsed, what is being tried and why — driven by `srv` state, not a spinner.
5. **All-in-one page**: full llama.cpp command visible, copy button on the
   command and on the log; log with select-all/copy.
6. **UI tests that prove it** — `testUI` over the real flow with a stub server
   that reproduces the measured failures (68 GiB compute at 1M, clean cudaMalloc
   at 64k, success at 32k), asserting the app converges and reports honestly.
   Plus the opt-in real-model e2e, fixed.

## Rules learned the hard way

- Never claim a fix without starting the real server. Four "verified" fixes in a
  row each ended in a different crash.
- The GGUF header cannot size a new architecture. When the arithmetic and the
  allocator disagree, the allocator is right.
- Clean up every probe process — a killed 100 GB server holds its VRAM and the
  next attempt fails for a reason that has nothing to do with the change.
