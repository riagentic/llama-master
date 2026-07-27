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
    min: 1,
    max: 64,
    tip:
      "Concurrent requests the server will serve. The context size is divided between slots, so 4 slots at -c 32768 gives each request 8192 tokens.",
  },
  {
    key: "noContextShift",
    flag: "--no-context-shift",
    label: "No context shift",
    kind: "bool",
    group: "context",
    scope: "server",
    def: false,
    advanced: true,
    tip:
      "Stop instead of silently dropping the oldest tokens when the context fills. Preferable when you need reproducible, complete conversations.",
  },
  {
    key: "keep",
    flag: "--keep",
    label: "Keep tokens",
    kind: "int",
    group: "context",
    scope: "both",
    def: 0,
    min: 0,
    max: 100000,
    advanced: true,
    tip:
      "Tokens from the start of the prompt to preserve when the context shifts. -1 keeps all of them.",
  },
  {
    key: "defragThold",
    flag: "--defrag-thold",
    label: "KV defrag threshold",
    kind: "float",
    group: "context",
    scope: "both",
    def: 0.1,
    min: 0,
    max: 1,
    step: 0.05,
    advanced: true,
    tip:
      "Compact the KV cache when this fraction of it is holes. Matters for long multi-slot server sessions.",
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
      "Threads for prompt processing, which is compute-bound rather than bandwidth-bound — here every logical processor can help. 0 = same as -t.",
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
    key: "mlock",
    flag: "--mlock",
    label: "Lock in RAM",
    kind: "bool",
    group: "performance",
    scope: "both",
    def: false,
    tip:
      "Pin the model in physical memory so the OS can never swap it out. Only safe when the CPU-side weights comfortably fit in free RAM.",
  },
  {
    key: "noMmap",
    flag: "--no-mmap",
    label: "Disable mmap",
    kind: "bool",
    group: "performance",
    scope: "both",
    def: false,
    tip:
      "Read the whole file into RAM up front instead of mapping it. Slower to start and uses more RAM; occasionally helps on network filesystems.",
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
    label: "Jinja templates",
    kind: "bool",
    group: "server",
    scope: "server",
    def: false,
    tip:
      "Use the chat template embedded in the GGUF. Needed for correct formatting on most instruct models, and for tool calling.",
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
    label: "Expose slots",
    kind: "bool",
    group: "server",
    scope: "server",
    def: false,
    advanced: true,
    tip:
      "Expose /slots with live per-request state. Useful for debugging, leaks prompt contents.",
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
    def: 600,
    min: 1,
    max: 86400,
    unit: "s",
    advanced: true,
    tip: "Seconds the server waits on a stalled request before giving up.",
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
