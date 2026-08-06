// src/lib/types.ts — the domain vocabulary, shared by every cell, the pure
// planners, and the UI. Types only: importing this file pulls in no runtime and
// no platform API, so it is safe on both sides of the wire.

import type { Disk } from "./disk.ts";

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
  /** Multi-token-prediction blocks (`nextn_predict_layers`). A model declaring
   *  these ships a block that drafts the next tokens for llama.cpp to verify
   *  against the full model — speculative decoding with no second model and no
   *  change to the output. `nLayer` INCLUDES them; llama.cpp's own `n_layer()`
   *  subtracts them again. 0 = not an MTP model. */
  nextnLayers: number;
  nExpert: number;
  nExpertUsed: number;
  ropeFreqBase: number;
  nTensors: number;
  tensorBytes: number;
  embdBytes: number;
  outputBytes: number;
  /** Tensors whose ggml type this build does not know — sizes exclude them. */
  unknownTypes: number;
  /** Which part of a split set this header came from, and how many parts there
   *  are (`split.no`, `split.count`). A model over ~40 GB is always split, and
   *  each part holds only its own slice of the tensor table — so every byte
   *  count above describes ONE PART until `mergeShards` has run over all of
   *  them (`src/lib/shards.ts`). 0 = a single-file model. */
  /** Sparse-attention indexer top-k (`attention.indexer.top_k`). Non-zero means
   *  the compute buffer scales with the CONTEXT, not just the micro-batch —
   *  see `plan.ts:computeScratch`. 0 = ordinary attention. */
  /** The context the model was really trained at, before RoPE scaling stretched
   *  the advertised one (`rope.scaling.original_context_length`). 0 = unscaled. */
  nCtxOrig: number;
  indexerTopK: number;
  splitNo: number;
  splitCount: number;
  /** Tensors across every part (`split.tensors.count`) — the check that a merge
   *  actually saw the whole model. */
  splitTensors: number;
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
  /** Is a display attached to this card? `undefined` when the machine could not
   *  be asked (no `nvidia-smi` field, no DRM connectors, a vendor with no such
   *  reading) — which is a third answer, not a `false`: the connected-GPU
   *  reserve falls back to card 0 when nothing is known, and holds back nothing
   *  when the machine has answered and every card is headless
   *  (`src/lib/reserve.ts:displayGpus`). */
  display?: boolean;
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
  /**
   * RLIMIT_MEMLOCK — how much memory a process may actually PIN.
   *
   * Here because `--mlock` is a promise the kernel can refuse. Stock Linux caps
   * this far below what a large model weighs (23.3 GB on the machine that
   * motivated this, against ~100 GB of host-side weights), and llama.cpp's
   * response to being over is a warning and an unpinned run — so the app would
   * print "pinning them stops the OS paging the model out" about something that
   * did not happen. Inherited by the child, so OUR limit is the right one to
   * read. 0 means "not known" (a platform that does not report it), and the
   * tuner treats that as "do not promise" — as does its absence.
   */
  lockableB?: number;
};

/** Everything the tuner is allowed to reason about. */
export type { Disk };

/**
 * Memory the user has told the app to keep for themselves.
 *
 * Not a measurement and not a safety margin — those are `tune.ts:marginB` and
 * `devsplit.ts`, and they exist to stop the allocator failing. This is the
 * user's own claim on their machine: the desktop, the browser, the compile they
 * are going to start. It is subtracted from what any plan may spend, so a card
 * that drives the display is not filled to the last byte
 * (`src/lib/reserve.ts`).
 *
 * VRAM is two figures because the two claims are different claims. `perGpuB` is
 * "leave this much on every card" — a compute card shared with another tool.
 * `connectedB` is "leave this much on the card that draws my screen", which is
 * where the cost actually is (a compositor, a browser, a game) and is typically
 * ONE card of several, so charging it to all of them would refuse memory nobody
 * wants back.
 */
export type Reserve = { perGpuB: number; connectedB: number; ramB: number };

export type Hw = {
  cpu: Cpu | null;
  mem: Mem | null;
  gpus: Gpu[];
  /** What the user has set aside for their own work. Absent on raw telemetry —
   *  it is attached where a plan is made (`src/ui/derive.ts:planningHw`), never
   *  by the hardware reader, because it is a preference, not a reading. */
  reserve?: Reserve;
  /** Filesystems this app writes to. The third memory pool: builds and models
   *  are gigabytes, and "no space left on device" is a real build failure. */
  disks?: Disk[];
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
  /** The value this app starts from and shows in the panel. */
  def: ParamValue;
  /**
   * llama.cpp's OWN default, when it is not `def`. The flag is omitted from the
   * argv at THIS value, not at `def`.
   *
   * The two used to be assumed identical, and upstream moved: `-ngl` now
   * defaults to **auto** rather than 0, and `-c` to **0 = take it from the
   * model**. So "CPU only" emitted no `-ngl` and llama.cpp offloaded to the GPU
   * anyway, and a plan drawn for a 4,096-token context emitted no `-c` and
   * llama.cpp loaded the model's declared 1,048,576 — a plan for 4 GB of cache
   * running as a plan for 40 GB of scratch, which is a start that cannot
   * succeed and an error that names none of this.
   *
   * Where llama.cpp's default is not a value the app can hold ("auto"), use a
   * sentinel outside the parameter's own range — the flag is then ALWAYS
   * emitted, which for the three that decide the placement is the right answer
   * anyway: their whole job is to pin what runs.
   */
  llamaDef?: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  /** Human labels for `options`, positionally. Without these a select shows the
   *  raw flag values, which are llama.cpp's vocabulary and not the user's. */
  optionLabels?: string[];
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
  /** `GGML_SCHED_MAX_SPLIT_INPUTS` this build was compiled with, when raised
   *  above llama.cpp's stock 30 — the "bypass the graph-split limit" option.
   *  Absent = stock. Recorded so a build that behaves differently says why. */
  schedCap?: number;
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
  /** The model's reasoning, when it thinks before answering — llama.cpp
   *  streams a reasoning model's `<think>` block as `reasoning_content`,
   *  separate from the answer. Kept so the reply's first half is never
   *  invisible, shown folded so it never drowns the answer. */
  thinking?: string;
  /** Server-reported timings, when the response carried them. */
  tps?: number;
};
