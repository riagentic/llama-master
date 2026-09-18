// src/lib/params.ts — the llama.cpp parameter catalog.
//
// ONE declaration per flag, used by three consumers: the settings UI renders
// controls from it, `command.ts` emits argv from it, `tune.ts` writes values
// into it. Adding a flag anywhere else in the app is a bug — add it here and it
// appears in the UI, in both command previews, and in the tuner's search space.
//
// `def` is llama.cpp's OWN default: a value equal to it is omitted from the
// command line, so the preview shows only what the user actually changed.

import type { Param, ParamGroup, ParamValue, Settings } from "./types.ts";

const CACHE_TYPES = [
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "q5_1",
  "q5_0",
  "q4_1",
  "q4_0",
];

/**
 * Defaults this catalog USED to have, per key.
 *
 * `cfg.settings` stores values, not intentions, so a default changed here
 * never reached anyone who had run the app before: v0.8.0 moved `-to` from
 * 600 to 3600 and turned `--jinja` and `--slots` back on (both are ON
 * upstream), and older stores kept launching `-to 600 --no-slots` — and
 * `--no-jinja`, which breaks every chat template with tool calls or
 * reasoning. `cfg`'s migration resets a stored value found here to today's
 * default.
 *
 * `cfg.touched` cannot vet this: it records every key that DIFFERS from the
 * default, including every value the tuner wrote — so it names `ngl` and
 * `ctxSize` beside the old `timeout`, and says nothing about intent. Hence the
 * list holds only values nobody would choose on purpose: a read timeout
 * shorter than a long prefill takes (~690 s at 250k tokens measured), and a
 * template engine switched off. `slots: false` is NOT here — hiding
 * `/slots` (which shows every prompt in flight) is a reasonable choice, and
 * costs nothing but the LAN client's occupancy reading.
 *
 * When a `def:` below changes, its old value goes here if keeping it would be
 * a defect, and `cfg.version` goes up by one.
 */
export const FORMER_DEFAULTS: Readonly<Record<string, readonly unknown[]>> = {
  timeout: [600],
  jinja: [false],
};

export const PARAMS: readonly Param[] = [
  // ── offload ──────────────────────────────────────────────────────────────
  {
    key: "ngl",
    flag: "-ngl",
    label: "GPU layers",
    kind: "int",
    group: "offload",
    scope: "both",
    def: 0,
    // llama.cpp's default is AUTO — it offloads what it thinks fits. -1 is
    // outside this parameter's range, so `-ngl` is always emitted: without it,
    // "CPU only" silently ran on the GPU.
    llamaDef: -1,
    min: 0,
    max: 999,
    tip:
      "Transformer layers to run on the GPU. llama.cpp offloads the LAST N layers; 999 means everything including the output layer. Each offloaded layer moves its weights and its KV cache into VRAM.",
  },
  {
    key: "nCpuMoe",
    flag: "--n-cpu-moe",
    label: "MoE layers on CPU",
    kind: "int",
    group: "offload",
    scope: "both",
    def: 0,
    min: 0,
    max: 999,
    tip:
      "Keep the routed-expert tensors of the first N layers in system RAM while attention stays on the GPU. On a mixture-of-experts model this is usually far faster than dropping whole layers: attention is bandwidth-bound and tiny, the experts are huge and only a few fire per token.",
  },
  {
    key: "fit",
    flag: "--fit",
    label: "llama.cpp auto-fit",
    kind: "enum",
    group: "offload",
    scope: "both",
    def: "off",
    // Upstream defaults to ON (common.h: fit_params = true): llama-server
    // quietly adjusts whatever the command line left unset to fit device
    // memory. This app IS that fitter — the plan, the packer and the fit
    // ladder — and its promise is that the command shown is the command that
    // runs, so the flag is always emitted. Upstream's own fitter also
    // segfaults on DeepSeek-V4-Flash, which is why the retry ladder exists.
    llamaDef: "on",
    options: ["off", "on"],
    advanced: true,
    tip:
      "Let llama.cpp resize unset options to fit free device memory. Off (recommended): this app already plans the placement, and what you see is what runs. On: llama.cpp may silently change settings the command does not pin.",
  },
  {
    key: "splitMode",
    flag: "-sm",
    label: "Split mode",
    kind: "enum",
    group: "offload",
    scope: "both",
    def: "layer",
    options: ["none", "layer", "row"],
    advanced: true,
    tip:
      "How to spread a model across multiple GPUs. layer: whole layers per GPU (default, least traffic). row: split each tensor (needs fast interconnect). none: use one GPU only.",
  },
  {
    key: "tensorSplit",
    flag: "-ts",
    label: "Tensor split",
    kind: "text",
    group: "offload",
    scope: "both",
    def: "",
    advanced: true,
    unit: "e.g. 3,1",
    tip:
      "Proportion of the model given to each GPU, comma separated. Leave empty for an even split. Use it when your cards have different VRAM.",
  },
  {
    key: "mainGpu",
    flag: "-mg",
    label: "Main GPU",
    kind: "int",
    group: "offload",
    scope: "both",
    def: 0,
    min: 0,
    max: 15,
    advanced: true,
    tip:
      "Index of the GPU that holds the small shared tensors and does scratch work.",
  },
  {
    // Not `advanced`: choosing which GPUs to use is a headline capability, and
    // it was previously reachable only by typing llama.cpp's own device names
    // ("CUDA0,CUDA1") into a free-text box hidden behind the advanced toggle.
    key: "device",
    flag: "-dev",
    label: "GPUs to use",
    kind: "devices",
    group: "offload",
    scope: "both",
    def: "",
    tip:
      "Which GPUs llama.cpp may use. All of them by default; switch one off and it is left alone — the memory plan below follows the same choice.",
  },
  {
    key: "noKvOffload",
    flag: "-nkvo",
    label: "Keep KV cache on CPU",
    kind: "bool",
    group: "offload",
    scope: "both",
    def: false,
    advanced: true,
    tip:
      "Store the KV cache in system RAM instead of VRAM. Frees a lot of VRAM for weights at a real speed cost — try quantising the cache first.",
  },
  {
    key: "specType",
    flag: "--spec-type",
    label: "Speculative decoding",
    kind: "enum",
    options: [
      "",
      "draft-mtp",
      "ngram-simple",
      "ngram-map-k",
      "ngram-map-k4v",
      "ngram-mod",
      "ngram-cache",
    ],
    optionLabels: [
      "off",
      "MTP (model's own block)",
      "n-gram (simple)",
      "n-gram map (k)",
      "n-gram map (k4v)",
      "n-gram (mod)",
      "n-gram (cache)",
    ],
    group: "performance",
    scope: "both",
    def: "",
    tip:
      "Draft several tokens ahead and let the full model verify them. LOSSLESS — a rejected draft is discarded, so the output is exactly what the model would have produced anyway; only the speed changes. `draft-mtp` uses the multi-token-prediction block the model already ships, so it needs no second model, and is only offered for models that declare one. The `ngram-*` kinds draft from the text already generated — no second model, no MTP block, any model — and pay off on repetitive output (code, lists, edits of earlier text). Not measured by the tuner; try `ngram-map-k4v` first.",
  },
  {
    // In llama.cpp master since late August 2026, defaulting to `auto`. The
    // `on-direct` VALUE is not: it comes from pull request #28136, which
    // replaces demand-paging the per-layer-embedding table with explicit reads
    // and was measured at better than twice the cold prefill on several
    // machines. Left at `auto` the flag is not emitted at all, so a build
    // without that pull request is completely unaffected — which is the rule
    // every PR-only setting has to follow.
    key: "lazyMode",
    flag: "--lazy-mode",
    label: "Lazy tensor reads",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "auto",
    llamaDef: "auto",
    options: ["auto", "on", "on-direct", "off"],
    optionLabels: [
      "auto (over 4 GiB)",
      "on",
      "on-direct (needs PR #28136)",
      "off",
    ],
    advanced: true,
    tip:
      "Read huge lookup tensors — a per-layer embedding table, which is a quarter of Qwen3.8-Flash-Next — from disk on demand instead of holding them in memory. `auto` does this above 4 GiB and is llama.cpp's own default. `on-direct` reads the rows with explicit file reads instead of page faults, which is much faster on a cold cache; it exists only in builds carrying pull request #28136, and a build without it will refuse to start and say so.",
  },
  {
    key: "draftModel",
    flag: "-md",
    label: "Draft model",
    kind: "text",
    group: "performance",
    scope: "both",
    def: "",
    tip:
      "Path to a small GGUF of the SAME family, used to draft tokens the big model then verifies. The strongest form of speculative decoding — a 0.5B drafting for a 32B is the classic pairing — and still lossless: a rejected draft is discarded, so the output is unchanged. It costs the VRAM the draft model occupies, which the memory plan does not yet bill, so leave headroom.",
  },
  {
    key: "specDraftNMax",
    flag: "--spec-draft-n-max",
    label: "Draft tokens (max)",
    kind: "int",
    group: "performance",
    scope: "both",
    // No llamaDef: 0 here means "not set", and the flag is inert unless
    // --spec-type is on, so emitting it always would be noise on a command
    // line the user reads.
    def: 0,
    min: 0,
    max: 16,
    advanced: true,
    tip:
      "How many tokens to draft per step. More is faster when the drafts are accepted and slower when they are not, because a rejected draft is work thrown away. 0 leaves llama.cpp's default (3). Model authors publishing an MTP block usually name a figure for it — 4 is the published value for the Gemma 4 heads — and it is worth measuring with the Speed panel rather than assuming either.",
  },
  {
    key: "specDraftNMin",
    flag: "--spec-draft-n-min",
    label: "Draft tokens (min)",
    kind: "int",
    group: "performance",
    scope: "both",
    def: 0,
    min: 0,
    max: 16,
    advanced: true,
    tip:
      "Stop drafting below this many tokens rather than pay the overhead for a very short run. 0 leaves llama.cpp's default.",
  },
  {
    // ── Why this is here, and why the tuner will not switch it on ──────────
    //
    // Sampling — turning the model's output row into the next token — normally
    // happens on the CPU, which means the logits are copied off the GPU every
    // single token. On a large vocabulary that is megabytes per token, and the
    // bytes are the smaller half of the cost: the copy is a SYNCHRONISATION
    // POINT, so the device stops and waits in the middle of the hot loop.
    // `-bs` keeps the whole step on the backend and removes the wait.
    //
    // It is off upstream and marked experimental, and llama-server still drops
    // it silently in three situations — the flag stays on the command line and
    // simply does nothing:
    //
    //   1. a grammar or JSON schema is in use (warns in the log)
    //   2. a reasoning budget is set (warns in the log)
    //   3. the request asked for logprobs before sampling
    //
    // A FOURTH used to be here and is worth remembering, because getting it
    // wrong cost a round of this work: until some point between 2026-07-27 and
    // 2026-09-07, `server-context.cpp` also did
    // `backend_sampling &= !(slot.can_speculate())` — "requires multiple
    // samples per batch - not supported yet" — so the flag did nothing whenever
    // a drafter was attached. That line is GONE from master, and `-bs` now
    // stacks with speculative decoding, which is the combination worth having:
    // drafting makes tokens cheaper to produce and this makes each one cheaper
    // to collect. The lesson is in CLAUDE.md — the local source cache is
    // whatever the last build fetched, not what upstream does today.
    key: "backendSampling",
    flag: "-bs",
    label: "Sample on the GPU",
    kind: "bool",
    group: "performance",
    scope: "both",
    def: false,
    llamaDef: false,
    advanced: true,
    tip:
      "Pick the next token on the GPU instead of copying the model's output row back to the CPU every token. Experimental upstream, and measured here as a LOSS: 0.2-0.8 s added before the first token of every reply (more at long contexts) for no measurable generation speed, on upstream master and on PrismML's fork alike. llama.cpp also turns it off by itself, with a warning in the log, when a grammar, a JSON schema or a reasoning budget is in use. The tuner leaves it off; measure with the Speed panel if your build differs.",
  },
  {
    key: "overrideTensor",
    flag: "-ot",
    label: "Tensor override",
    kind: "text",
    group: "offload",
    scope: "both",
    def: "",
    advanced: true,
    unit: "regex=device",
    tip:
      "Pin tensors matching a regex to a device, e.g. `.ffn_.*_exps.=CPU`. The manual form of the MoE-on-CPU trick, for when you need per-tensor control.",
  },

  // ── context ──────────────────────────────────────────────────────────────
  {
    key: "ctxSize",
    flag: "-c",
    label: "Context size",
    kind: "int",
    group: "context",
    scope: "both",
    def: 4096,
    // llama.cpp's default is 0 = take it from the model, which for a model
    // declaring 1,048,576 is 1,048,576. Always emitted, or a plan drawn for
    // 4,096 tokens starts a server that needs 256 times the memory.
    llamaDef: 0,
    min: 256,
    max: 1048576,
    step: 256,
    unit: "tokens",
    tip:
      "Tokens the model can attend to. KV-cache memory grows linearly with this number, so it is the first knob to turn when a model nearly fits. 0 = use the model's trained maximum.",
  },
  {
    key: "batchSize",
    flag: "-b",
    label: "Batch size",
    kind: "int",
    group: "context",
    scope: "both",
    def: 2048,
    min: 32,
    max: 32768,
    step: 32,
    unit: "tokens",
    tip:
      "Logical batch: how many prompt tokens are submitted per evaluation. Bigger speeds up prompt processing, costs compute-buffer VRAM.",
  },
  {
    key: "ubatchSize",
    flag: "-ub",
    label: "Micro-batch size",
    kind: "int",
    group: "context",
    scope: "both",
    def: 512,
    min: 16,
    max: 8192,
    step: 16,
    unit: "tokens",
    tip:
      "Physical batch actually run at once. This is what sizes the compute buffer — lower it first if you are a few hundred MB short of VRAM.",
  },
  {
    key: "parallel",
    flag: "-np",
    label: "Parallel slots",
    kind: "int",
    group: "context",
    scope: "server",
    def: 1,
    // llama.cpp's default is -1 = auto, and auto chose FOUR on this machine —
    // so the panel said "1 slot" while the server ran four. Always emitted.
    llamaDef: -1,
    min: 1,
    max: 64,
    tip:
      "Concurrent requests the server will serve. The context size is divided between slots, so 4 slots at -c 32768 gives each request 8192 tokens.",
  },
  {
    // Upstream's default FLIPPED: `ctx_shift = false` (common.h), and the
    // flag pair is `--context-shift` / `--no-context-shift`. This entry used
    // to be "No context shift", emitting `--no-context-shift` when ON — which
    // reproduced the default — and offering no way to turn shifting on at all:
    // a switch that did nothing in either position. Stored as
    // `noContextShift` until cfg v3, which drops it (the old default and the
    // new one run the same server).
    key: "contextShift",
    flag: "--context-shift",
    offFlag: "--no-context-shift",
    label: "Context shift",
    kind: "bool",
    group: "context",
    scope: "server",
    def: false,
    advanced: true,
    tip:
      "When the context fills, drop the oldest tokens and carry on instead of stopping. Off by default (llama.cpp's own default): a conversation that silently loses its beginning is worse than one that says it is full. llama.cpp refuses it anyway on models whose cache cannot shift (hybrid and recurrent architectures).",
  },
  {
    key: "cacheReuse",
    flag: "--cache-reuse",
    label: "Cache reuse",
    kind: "int",
    group: "context",
    scope: "server",
    def: 256,
    // llama.cpp's own default is 0 — off. Every upstream server preset sets
    // 256, and the server disables it by itself where the cache cannot shift,
    // so it is safe to emit for every model.
    llamaDef: 0,
    min: 0,
    max: 8192,
    unit: "tokens",
    advanced: true,
    tip:
      "Minimum chunk of an earlier prompt to reuse by shifting the KV cache instead of recomputing it. Turns an edit in the middle of a conversation — a changed system prompt, a trimmed message — from a full re-prefill into a shift. 0 = off.",
  },
  {
    key: "keep",
    flag: "--keep",
    label: "Keep tokens",
    kind: "int",
    group: "context",
    scope: "both",
    def: 0,
    // `-1` means "keep everything", and the tip says so — the range must allow
    // it or a user who types the documented value silently gets `0` (keep
    // nothing) instead. Same pattern as `repeatLastN`, which already ships
    // `min: -1` for exactly this reason.
    min: -1,
    max: 100000,
    advanced: true,
    tip:
      "Tokens from the start of the prompt to preserve when the context shifts. -1 keeps all of them.",
  },

  // ── performance ──────────────────────────────────────────────────────────
  {
    key: "threads",
    flag: "-t",
    label: "Threads",
    kind: "int",
    group: "performance",
    scope: "both",
    def: 0,
    min: 0,
    max: 512,
    tip:
      "CPU threads for generation. Physical cores is the sweet spot — hyper-threads usually cost throughput because they contend for the same memory bandwidth. 0 = llama.cpp decides.",
  },
  {
    key: "threadsBatch",
    flag: "-tb",
    label: "Threads (batch)",
    kind: "int",
    group: "performance",
    scope: "both",
    def: 0,
    min: 0,
    max: 512,
    advanced: true,
    tip:
      "Threads for prompt processing. 0 = same as -t, which is llama.cpp's default and what the tuner emits (one per physical core). Two threads on one core share its memory port, so SMT siblings rarely help here either — measure before raising it.",
  },
  {
    key: "flashAttn",
    flag: "-fa",
    label: "Flash attention",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "auto",
    options: ["auto", "on", "off"],
    tip:
      "Fused attention kernel: less KV-cache memory and faster long contexts. `auto` enables it wherever the backend supports it. Required before the KV cache can be quantised to q4/q5 on most backends.",
  },
  {
    key: "cacheTypeK",
    flag: "-ctk",
    label: "KV cache type (K)",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "f16",
    options: CACHE_TYPES,
    tip:
      "Precision of the key cache. q8_0 halves KV memory for a barely measurable quality cost — the standard move when a long context does not fit.",
  },
  {
    key: "cacheTypeV",
    flag: "-ctv",
    label: "KV cache type (V)",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "f16",
    options: CACHE_TYPES,
    tip:
      "Precision of the value cache. Quantising V usually needs flash attention on; keep it at least as precise as K.",
  },
  {
    // ONE setting, because llama.cpp has one: `--mlock`, `--no-mmap` and
    // `--direct-io` all assigned the same `params.load_mode`, so emitting two
    // of them was never "locked and unmapped" — it was whichever came last,
    // silently, while the app printed a reason claiming the other.
    //
    // Upstream finished the job: the three old flags were deprecated in favour
    // of `-lm/--load-mode` and then REMOVED. A master build meets `--mlock`
    // with `unknown argument` and exits before it has read the model path, so
    // the catalog speaks the new spelling and `legacy` below carries the old
    // one for a build that still wants it (`command.ts`, gated on the probe —
    // never on a version number, which a PR stack does not have).
    key: "loadMode",
    flag: "--load-mode",
    label: "Model loading",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "auto",
    llamaDef: "auto",
    options: ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"],
    optionLabels: [
      "auto (mmap where possible)",
      "none (read into RAM)",
      "mmap",
      "mlock (pin in RAM)",
      "mmap + mlock",
      "direct I/O",
    ],
    legacy: {
      // Older llama.cpp, same meanings. `auto` and `mmap` were the default
      // there too, so they emit nothing at all.
      auto: [],
      mmap: [],
      none: ["--no-mmap"],
      mlock: ["--mlock"],
      "mmap+mlock": ["--mlock"],
      dio: ["--direct-io"],
    },
    tip:
      "How the weights get into memory. auto memory-maps the file, so a warm restart re-reads almost nothing. mlock pins it so the OS can never page it out — only safe when the host-side weights fit in free RAM and under this machine's memlock limit. none reads the whole file up front: slower to start, more RAM, occasionally helps on a network filesystem.",
  },
  {
    key: "numa",
    flag: "--numa",
    label: "NUMA policy",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "",
    options: ["", "distribute", "isolate", "numactl"],
    advanced: true,
    tip:
      "Thread and memory placement on multi-socket machines. distribute spreads across nodes; isolate keeps everything on one.",
  },
  {
    key: "ropeScaling",
    flag: "--rope-scaling",
    label: "RoPE scaling",
    kind: "enum",
    group: "performance",
    scope: "both",
    def: "",
    options: ["", "none", "linear", "yarn"],
    advanced: true,
    tip:
      "Extend the context beyond what the model was trained for. yarn degrades least; expect some quality loss either way.",
  },
  {
    key: "ropeFreqBase",
    flag: "--rope-freq-base",
    label: "RoPE freq base",
    kind: "float",
    group: "performance",
    scope: "both",
    def: 0,
    min: 0,
    max: 10000000,
    advanced: true,
    tip: "Override the RoPE base frequency. 0 = take it from the model.",
  },
  {
    key: "ropeFreqScale",
    flag: "--rope-freq-scale",
    label: "RoPE freq scale",
    kind: "float",
    group: "performance",
    scope: "both",
    def: 0,
    min: 0,
    max: 8,
    step: 0.05,
    advanced: true,
    tip: "Linear RoPE scaling factor. 0 = take it from the model.",
  },

  // ── sampling ─────────────────────────────────────────────────────────────
  {
    key: "temp",
    flag: "--temp",
    label: "Temperature",
    kind: "float",
    group: "sampling",
    scope: "both",
    def: 0.8,
    min: 0,
    max: 2,
    step: 0.05,
    tip:
      "Randomness of the next-token choice. 0 is deterministic; above ~1.2 most models start to wander.",
  },
  {
    key: "topK",
    flag: "--top-k",
    label: "Top-K",
    kind: "int",
    group: "sampling",
    scope: "both",
    def: 40,
    min: 0,
    max: 1000,
    tip: "Only sample from the K most likely tokens. 0 disables the filter.",
  },
  {
    key: "topP",
    flag: "--top-p",
    label: "Top-P",
    kind: "float",
    group: "sampling",
    scope: "both",
    def: 0.95,
    min: 0,
    max: 1,
    step: 0.01,
    tip:
      "Nucleus sampling: keep the smallest set of tokens whose probabilities sum to P.",
  },
  {
    key: "minP",
    flag: "--min-p",
    label: "Min-P",
    kind: "float",
    group: "sampling",
    scope: "both",
    def: 0.05,
    min: 0,
    max: 1,
    step: 0.01,
    tip:
      "Drop tokens less likely than P times the top token. A steadier alternative to top-p.",
  },
  {
    key: "repeatPenalty",
    flag: "--repeat-penalty",
    label: "Repeat penalty",
    kind: "float",
    group: "sampling",
    scope: "both",
    def: 1,
    min: 0.5,
    max: 2,
    step: 0.01,
    tip: "Penalise tokens that already appeared. 1.0 = off.",
  },
  {
    key: "repeatLastN",
    flag: "--repeat-last-n",
    label: "Repeat window",
    kind: "int",
    group: "sampling",
    scope: "both",
    def: 64,
    min: -1,
    max: 8192,
    advanced: true,
    tip:
      "How many recent tokens the repeat penalty looks at. -1 = the whole context.",
  },
  {
    key: "seed",
    flag: "-s",
    label: "Seed",
    kind: "int",
    group: "sampling",
    scope: "both",
    def: -1,
    min: -1,
    max: 2147483647,
    advanced: true,
    tip: "Fix the RNG for reproducible output. -1 = random each run.",
  },

  // ── server ───────────────────────────────────────────────────────────────
  {
    key: "host",
    flag: "--host",
    label: "Host",
    kind: "text",
    group: "server",
    scope: "server",
    def: "127.0.0.1",
    tip:
      "Interface to bind. Keep 127.0.0.1 unless you intend to expose the server to your network — llama-server has no authentication unless you set an API key.",
  },
  {
    key: "port",
    flag: "--port",
    label: "Port",
    kind: "int",
    group: "server",
    scope: "server",
    def: 8080,
    min: 1,
    max: 65535,
    tip: "TCP port for the HTTP API and the built-in web UI.",
  },
  {
    key: "alias",
    flag: "-a",
    label: "Model alias",
    kind: "text",
    group: "server",
    scope: "server",
    def: "",
    tip:
      "Name the API reports for this model. Handy when a client expects a specific model id.",
  },
  {
    key: "apiKey",
    flag: "--api-key",
    label: "API key",
    kind: "text",
    group: "server",
    scope: "server",
    def: "",
    tip:
      "Require this bearer token on every request. Set it before binding to anything other than localhost.",
  },
  {
    key: "jinja",
    flag: "--jinja",
    // ON upstream (`use_jinja = true`, common.h). This was `def: false` with
    // no `offFlag`, so switching it on emitted a flag that changed nothing and
    // switching it off emitted nothing and changed nothing — the server ran
    // Jinja either way while the panel said "off".
    offFlag: "--no-jinja",
    label: "Jinja templates",
    kind: "bool",
    group: "server",
    scope: "server",
    def: true,
    llamaDef: true,
    tip:
      "Render the chat template embedded in the GGUF with the Jinja engine — llama.cpp's default, and what tool calling and reasoning models need. Off falls back to the built-in template table; the command then shows --no-jinja.",
  },
  {
    key: "reasoning",
    flag: "--reasoning",
    label: "Reasoning",
    kind: "enum",
    group: "server",
    scope: "server",
    def: "auto",
    llamaDef: "auto",
    options: ["auto", "on", "off"],
    optionLabels: ["auto (the model decides)", "on", "off"],
    tip:
      "Whether a thinking model thinks before it answers. `off` skips the reasoning pass on models that allow it (Qwen3, GLM) — faster, and often enough for short questions. Replaces the deprecated `--chat-template-kwargs enable_thinking`.",
  },
  {
    key: "reasoningBudget",
    flag: "--reasoning-budget",
    label: "Reasoning budget",
    kind: "int",
    group: "server",
    scope: "server",
    def: -1,
    llamaDef: -1,
    min: -1,
    max: 1_000_000,
    unit: "tokens",
    advanced: true,
    tip:
      "Cap on thinking tokens per reply; the model is nudged to answer once it is spent. -1 = unlimited, 0 = no thinking.",
  },
  {
    key: "chatTemplate",
    flag: "--chat-template",
    label: "Chat template",
    kind: "text",
    group: "server",
    scope: "server",
    def: "",
    advanced: true,
    tip:
      "Override the chat template by name (chatml, llama3, …) when the one in the file is wrong or missing.",
  },
  {
    key: "contBatching",
    flag: "--cont-batching",
    offFlag: "--no-cont-batching",
    label: "Continuous batching",
    kind: "bool",
    group: "server",
    scope: "server",
    def: true,
    advanced: true,
    tip:
      "Interleave requests instead of running them one after another. On by default; the command shows --no-cont-batching when you turn it off.",
  },
  {
    key: "metrics",
    flag: "--metrics",
    label: "Prometheus metrics",
    kind: "bool",
    group: "server",
    scope: "server",
    def: false,
    advanced: true,
    tip: "Expose /metrics for scraping.",
  },
  {
    key: "slots",
    flag: "--slots",
    // ON upstream (`endpoint_slots = true`, common.h). As `def: false` with no
    // `offFlag` the switch was inert and the endpoint was up the whole time —
    // while its own tip warned that it leaks prompts. The LAN client reads
    // /slots for occupancy, so it stays on; what changes is that OFF now
    // works, and `stability.ts` says when it should be used.
    offFlag: "--no-slots",
    label: "Expose slots",
    kind: "bool",
    group: "server",
    scope: "server",
    def: true,
    llamaDef: true,
    advanced: true,
    tip:
      "Expose /slots with live per-request state — what the LAN client reads for occupancy. It also shows every prompt in flight, so turn it off (the command then shows --no-slots) when the server is bound to the network without an API key.",
  },
  {
    key: "noWebui",
    flag: "--no-webui",
    label: "Disable web UI",
    kind: "bool",
    group: "server",
    scope: "server",
    def: false,
    advanced: true,
    tip: "Serve only the API, without llama-server's own browser UI.",
  },
  {
    key: "timeout",
    flag: "-to",
    label: "Read timeout",
    kind: "int",
    group: "server",
    scope: "server",
    // llama.cpp's own 3600. It was 600 here, which is shorter than a long
    // prompt takes to prefill: 250k tokens at the measured 364 tok/s is ~690 s
    // with nothing on the wire, and the server cut the request off.
    def: 3600,
    llamaDef: 3600,
    min: 1,
    max: 86400,
    unit: "s",
    advanced: true,
    tip:
      "Seconds the server waits on a stalled request before giving up. A long prompt sends nothing while it prefills, so keep this above the prefill time of your biggest context.",
  },
  {
    key: "verbose",
    flag: "-v",
    label: "Verbose log",
    kind: "bool",
    group: "server",
    scope: "both",
    def: false,
    advanced: true,
    tip: "Log every request and the full model load trace.",
  },
  {
    key: "logVerbosity",
    flag: "-lv",
    label: "Log verbosity",
    kind: "int",
    group: "server",
    scope: "both",
    // llama.cpp's default (3) hides every `load_tensors`, buffer-size and
    // `sched_reserve` line — measured on b10144, b10151, master and the
    // PrismML fork: ~22 startup lines at 3, ~210 at 4. Those are the lines
    // the load-progress phases read and the lines a person reads to see
    // where memory went, so the app asks for 4. Errors print at any level.
    def: 4,
    llamaDef: 3,
    min: 0,
    max: 4,
    advanced: true,
    tip:
      "How much llama-server writes to the log. 4 includes the model load trace and the per-device buffer sizes — what the load progress and a memory diagnosis read. 3 is llama.cpp's own default and hides them.",
  },

  // ── cli-only ─────────────────────────────────────────────────────────────
  {
    key: "prompt",
    flag: "-p",
    label: "Prompt",
    kind: "text",
    group: "sampling",
    scope: "cli",
    def: "",
    tip: "The prompt llama-cli starts from.",
  },
  {
    key: "nPredict",
    flag: "-n",
    label: "Tokens to predict",
    kind: "int",
    group: "sampling",
    scope: "cli",
    def: -1,
    min: -2,
    max: 1000000,
    tip: "How many tokens llama-cli generates. -1 = until the model stops.",
  },
  {
    key: "conversation",
    flag: "-cnv",
    label: "Conversation mode",
    kind: "bool",
    group: "sampling",
    scope: "cli",
    def: false,
    tip: "Run llama-cli as an interactive chat using the model's template.",
  },
  {
    // The escape hatch, and deliberately IN the catalog rather than beside it:
    // llama.cpp has far more flags than are worth a control each, and without
    // this a flag the catalog does not carry could not be passed at all. An
    // empty `flag` means "emit the value's own tokens", handled in command.ts.
    key: "extraArgs",
    flag: "",
    label: "Extra arguments",
    kind: "text",
    group: "performance",
    scope: "both",
    def: "",
    advanced: true,
    unit: "e.g. --lora adapter.gguf",
    tip:
      "Anything else to append to the command, exactly as typed. For llama.cpp flags this app has no control for. It appears in the command preview above, so what you see is still what runs.",
  },
] as const;

/** Catalog lookup by key. Built once — the catalog is immutable. */
const BY_KEY: ReadonlyMap<string, Param> = new Map(
  PARAMS.map((p) => [p.key, p]),
);

export function param(key: string): Param | undefined {
  return BY_KEY.get(key);
}

export const GROUPS: readonly { id: ParamGroup; label: string }[] = [
  { id: "offload", label: "Offload" },
  { id: "context", label: "Context" },
  { id: "performance", label: "Performance" },
  { id: "sampling", label: "Sampling" },
  { id: "server", label: "Server" },
];

/** Every default, as a settings map. The single source of "unset". */
export function defaults(): Settings {
  return Object.fromEntries(PARAMS.map((p) => [p.key, p.def]));
}

/** Read a setting with the catalog default as the fallback. */
export function get(s: Settings, key: string): ParamValueOf {
  const v = s[key];
  return v === undefined ? (param(key)?.def ?? "") : v;
}

type ParamValueOf = string | number | boolean;

export function num(s: Settings, key: string): number {
  const v = get(s, key);
  return typeof v === "number" ? v : Number(v) || 0;
}

export function str(s: Settings, key: string): string {
  const v = get(s, key);
  return typeof v === "string" ? v : String(v);
}

export function bool(s: Settings, key: string): boolean {
  return get(s, key) === true;
}

/** Coerce a raw UI input into the type the catalog declares. Invalid numeric
 *  text keeps the previous value rather than writing NaN into state. */
export function coerce(p: Param, raw: string | boolean): ParamValue {
  if (p.kind === "bool") return raw === true || raw === "true";
  if (p.kind === "int" || p.kind === "float") {
    const n = p.kind === "int"
      ? parseInt(String(raw), 10)
      : parseFloat(String(raw));
    if (Number.isNaN(n)) return p.def;
    const lo = p.min ?? -Infinity;
    const hi = p.max ?? Infinity;
    return Math.min(hi, Math.max(lo, n));
  }
  return String(raw);
}
