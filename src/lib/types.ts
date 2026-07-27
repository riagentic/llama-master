// src/lib/types.ts — the domain vocabulary, shared by every cell, the pure
// planners, and the UI. Types only: importing this file pulls in no runtime and
// no platform API, so it is safe on both sides of the wire.

/** One transformer block's byte footprint, as measured from the GGUF header. */
export type LayerBytes = {
  i: number;
  /** Every tensor in `blk.<i>.*`. */
  bytes: number;
  /** The `*_exps` subset — routed experts, what `--n-cpu-moe` moves to CPU. */
  expert: number;
};

/** Where a model came from. Worth showing: an ollama blob has a digest for a
 *  file name, so without this the library is a list of unreadable hashes. */
export type ModelSource = "file" | "lmstudio" | "ollama";

/** A GGUF model on disk, with everything the planner needs. */
export type Model = {
  path: string;
  /** Display name: the file name, or the ollama model name (`llama3.2:3b`). */
  file: string;
  source: ModelSource;
  /** Directory the scan found it in — the UI groups by this. */
  dir: string;
  sizeB: number;
  mtime: number;
  /** Parsed header — null while unparsed, or when the file is not readable. */
  meta: ModelMeta | null;
  /** Set when the header could not be parsed; shown instead of silent absence. */
  metaError: string | null;
};

/** GGUF header facts. Mirrors the JSON emitted by `rust/src/gguf.rs`. */
export type ModelMeta = {
  version: number;
  arch: string;
  name: string;
  quant: string;
  nLayer: number;
  nCtxTrain: number;
  nEmbd: number;
  nHead: number;
  nHeadKv: number;
  keyLength: number;
  valueLength: number;
  /** Interleaved sliding-window attention: window size in tokens, and the
   *  period at which a full-attention layer appears. 0 = every layer is full
   *  attention, which is the overwhelming majority of models. */
  swaWindow: number;
  swaPattern: number;
  /** Multi-head latent attention rank (DeepSeek-V2/V3). 0 = not MLA. */
  kvLoraRank: number;
  nExpert: number;
  nExpertUsed: number;
  ropeFreqBase: number;
  nTensors: number;
  tensorBytes: number;
  embdBytes: number;
  outputBytes: number;
  /** Tensors whose ggml type this build does not know — sizes exclude them. */
  unknownTypes: number;
  layers: LayerBytes[];
};

/** One accelerator, normalized across vendors. */
export type Gpu = {
  vendor: "nvidia" | "amd" | "intel";
  name: string;
  /** CUDA compute capability (12.0 = Blackwell), 0 when not an NVIDIA card.
   *  This is what decides whether the installed nvcc can build for it. */
  computeCap: number;
  tempC: number;
  utilPct: number;
  vramTotalB: number;
  vramUsedB: number;
  powerW: number;
  powerLimitW: number;
};

export type Cpu = {
  model: string;
  /** Physical cores — the number `-t` should usually match. */
  cores: number;
  /** Logical processors (SMT included). */
  threads: number;
  mhz: number;
  tempC: number;
  utilPct: number;
  /** Raw `/proc/stat` aggregate line; the delta source for `utilPct`. */
  stat: string;
  coreStats: string[];
  coresUtil: number[];
};

export type Mem = {
  totalB: number;
  availableB: number;
  usedB: number;
  swapTotalB: number;
  swapUsedB: number;
};

/** Everything the tuner is allowed to reason about. */
export type Hw = {
  cpu: Cpu | null;
  mem: Mem | null;
  gpus: Gpu[];
  os: string;
  arch: string;
  /** Backend of the build that will actually run this — part of the machine as
   *  far as the tuner is concerned, because a flag the backend rejects is not a
   *  setting, it is a failed load. Undefined before a build is chosen. */
  backend?: Backend;
};

/** A llama.cpp parameter value. Flat and JSON-safe so the whole settings map
 *  persists, syncs, and diffs without a custom codec. */
export type ParamValue = string | number | boolean;

/** The settings map: `param.key` → value. Missing key = the catalog default. */
export type Settings = Record<string, ParamValue>;

/** Which binary a parameter applies to. */
export type ParamScope = "both" | "server" | "cli";

export type ParamKind =
  | "bool"
  | "int"
  | "float"
  | "text"
  | "enum"
  /** A `-dev`-style comma list of llama.cpp device names, rendered as one
   *  checkbox per detected device (src/lib/gpu.ts). */
  | "devices";

export type ParamGroup =
  | "offload"
  | "context"
  | "performance"
  | "sampling"
  | "server";

/** One llama.cpp flag, described once and used everywhere: the UI renders from
 *  this, the command builder emits from this, the tuner writes into this. */
export type Param = {
  key: string;
  /** The literal CLI flag, e.g. `-ngl`. */
  flag: string;
  /** For a boolean whose default is ON: the flag that turns it off. Without
   *  this, "unchecked" would silently emit nothing and change nothing. */
  offFlag?: string;
  label: string;
  kind: ParamKind;
  group: ParamGroup;
  scope: ParamScope;
  /** Value at which the flag is omitted — llama.cpp's own default. */
  def: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Plain-language explanation shown as a tooltip. Every param has one. */
  tip: string;
  /** Hidden behind the "advanced" toggle — rarely needed, never removed. */
  advanced?: boolean;
  /** Placeholder / unit hint for text and number inputs. */
  unit?: string;
};

/** A llama.cpp build this app produced or downloaded. */
export type Build = {
  id: string;
  /** Git ref: `master` or a release tag such as `b6234`. */
  ref: string;
  /** `source` = compiled here, `release` = official prebuilt asset. */
  origin: "source" | "release";
  backend: Backend;
  dir: string;
  serverBin: string;
  cliBin: string;
  createdAt: number;
  /** Bytes on disk, for the "reclaim space" affordance. */
  sizeB: number;
  /** The upstream commit this was built from. Only a `master` build needs it:
   *  a tag is its own version, but "master" means nothing without the commit,
   *  and the update check has to compare against something. */
  sourceSha?: string;
};

export type Backend = "cpu" | "cuda" | "vulkan" | "hip" | "metal";

/** A long-running job (build, download, extract) with progress a bar can show. */
export type Job = {
  id: string;
  label: string;
  /** 0..1, or null when the total is genuinely unknown (git clone, cmake). */
  progress: number | null;
  /** Current step out of `steps.length`, for the stepper UI. */
  step: number;
  steps: string[];
  startedAt: number;
  endedAt: number | null;
  status: "running" | "done" | "failed" | "cancelled";
  error: string | null;
};

/** One detected prerequisite — what it is, whether we have it, how we got it. */
export type Prereq = {
  id: string;
  label: string;
  /** Why llama.master needs it, in one line. */
  why: string;
  found: boolean;
  version: string;
  path: string;
  /** True when this app downloaded it into its own data dir. */
  managed: boolean;
  /** False = the app can obtain it itself; true = must come from the OS. */
  systemOnly: boolean;
};

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
  /** Server-reported timings, when the response carried them. */
  tps?: number;
};
