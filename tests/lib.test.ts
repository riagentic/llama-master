// test/lib.test.ts — the pure core.
//
// Everything the app decides (what the command line is, what fits in VRAM, what
// "optimal" means) is a pure function, and this is where those decisions are
// pinned. No cells, no DOM, no I/O: if one of these fails, the app is wrong, not
// flaky.

import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertExists,
  assertNotEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";

import {
  advances,
  draftRows,
  QUEUE_MAX,
  queueAdd,
  queueLabel,
  queueNote,
  queueRemove,
  submitKind,
} from "../src/lib/queue.ts";
import {
  argv,
  commandBlock,
  commandLine,
  quote,
  serverUrl,
} from "../src/lib/command.ts";
import {
  bool,
  coerce,
  defaults,
  num,
  param,
  PARAMS,
  str,
} from "../src/lib/params.ts";
import {
  computeScratch,
  effectiveCtx,
  kvLayers,
  kvPerToken,
  kvTotal,
  NO_MODEL,
  plan,
  recurrentStateB,
  scratchFloor,
  swaSplit,
  withoutOurUsage,
} from "../src/lib/plan.ts";
import {
  countsFromSplit,
  deviceBudgets,
  loadPerDevice,
  offloadRange,
  packSlots,
  slotOnGpu,
  tensorSplitValue,
} from "../src/lib/devsplit.ts";
import {
  displayUnknown,
  MAX_RESERVE_GB,
  reserveBytes,
  reserveGb,
  reserveOf,
  vramReserveShares,
} from "../src/lib/reserve.ts";
import {
  ctxOf,
  fitDecision,
  fitFault,
  isFitFailure,
  MAX_FIT_RETRIES,
  movedLayers,
  nCpuMoeOf,
  openingCtx,
  requestedB,
  unsupportedFeature,
  vetoUnsupported,
  withCtx,
  withNCpuMoe,
  withoutFlag,
} from "../src/lib/fitladder.ts";
import type { Hw, ModelMeta, Settings } from "../src/lib/types.ts";
import {
  drift,
  HEADROOM_FRACTION,
  headroomBucket,
  headroomKey,
} from "../src/lib/adapt.ts";
import {
  bestPlacement,
  CTX_BANDS,
  CTX_PRESETS,
  ctxBands,
  ctxLabel,
  MIN_CTX,
  optimalCtx,
  pinnedCtx,
  tune,
  tuneAll,
} from "../src/lib/tune.ts";
import { stability } from "../src/lib/stability.ts";
import { autoJobs } from "../src/cell/builds.ts";
import { buildNumberFlags } from "../src/cell/builds.server.ts";
import { bytes, duration, pct, shortPath, tps } from "../src/lib/format.ts";
import { coresUtilPct, pushHistory, utilPct } from "../src/lib/procstat.ts";
import { appendLog, isError, progressOf } from "../src/lib/buildlog.ts";
import {
  deltaReasoning,
  deltaText,
  flushDelayMs,
  parseSse,
  timingsTps,
} from "../src/lib/sse.ts";
import {
  availableBackends,
  isBinaryAsset,
  noAssetExplanation,
  pickAsset,
  scoreAsset,
  usableAssets,
} from "../src/lib/assets.ts";
import {
  gunzip,
  safeEntries,
  stripRoot,
  untar,
  unzip,
} from "../src/lib/archive.ts";
import { buildNumber, updateFor, updateTarget } from "../src/lib/update.ts";
import {
  isOllamaStore,
  manifestSkipReason,
  nameFromManifestPath,
  resolveManifest,
} from "../src/lib/ollama.ts";
import type { Build } from "../src/lib/types.ts";
import { CPU_TJMAX, GPU_TJMAX, tempTone } from "../src/lib/thermal.ts";
import {
  canCompile,
  compilableBackends,
  preferredBackends,
  SCHED_SPLIT_CAP,
  schedCapFlags,
  targetReadiness,
  usableGpus,
} from "../src/lib/backend.ts";
import { diagnoseFailure, diagnoseNoAsset } from "../src/lib/diagnose.ts";
import { isNearBottom, stickToBottom } from "../src/lib/scroll.ts";
import { setupRows } from "../src/lib/setup.ts";
import { parseDf, tooFullToBuild } from "../src/lib/disk.ts";
import { demoCpu, demoGpus, demoMem, demoModels } from "../src/lib/demo.ts";
import {
  devices,
  enabledGpus,
  isEnabled,
  parseDevices,
  toggleDevice,
} from "../src/lib/gpu.ts";
import {
  diagnoseServerExit,
  extractErrors,
  signalOf,
} from "../src/lib/serverlog.ts";
import {
  cudaCmakeFlags,
  cudaOffer,
  cudaPlan,
  cudaUpgradeFor,
  cudaVersionForCap,
  driverCudaVersion,
  maxArchFor,
  parseCudaVersion,
} from "../src/lib/cuda.ts";
import {
  assetsFromHtml,
  assetUrl,
  isRateLimited,
  rateLimitMessage,
  shaFromCommitsAtom,
  tagFromReleaseUrl,
} from "../src/lib/github.ts";
import {
  describe as describeFix,
  elevate,
  fixPlan,
  isFixable,
  rocmPlan,
  scriptPreview,
} from "../src/lib/fixplan.ts";
import {
  bytesPerToken,
  calibrate,
  estimateTps,
  speedIsMeasured,
  tpsBand,
} from "../src/lib/speed.ts";
import {
  elapsedLabel,
  loadPhase,
  loadProgress,
} from "../src/lib/loadprogress.ts";
import { loadTone } from "../src/lib/thermal.ts";
import {
  ioFallback,
  niceFromProcStat,
  priorityNote,
  prioritySteps,
} from "../src/lib/priority.ts";
import {
  isLanExposed,
  lanHost,
  lanUrl,
  lanWarning,
  pickLanIp,
} from "../src/lib/lan.ts";
import {
  inlineChunks,
  parseInfo,
  replyBlocks,
  transcript,
} from "../src/lib/richtext.ts";
import { gpu, hw, layers, meta, moeMeta, NO_GPU } from "./fixtures.ts";
import {
  fileBpw,
  qualityNote,
  quantAdvice,
  quantOptions,
  rescale,
} from "../src/lib/quant.ts";
import {
  bandwidthNote,
  BENCH_PROMPTS,
  BENCH_TOKENS,
  benchApplies,
  benchRequest,
  EMPTY_BENCH,
  parseBench,
  specVerdict,
} from "../src/lib/bench.ts";
import type { BenchResult } from "../src/lib/bench.ts";
import { findMtpSibling, isMtpName, pairKey } from "../src/lib/mtp.ts";
import { parseHelpFlags, supportsFlag } from "../src/lib/caps.ts";
import {
  formatRef,
  parsePrInput,
  parsePrList,
  parseRef,
  prStateNote,
  refDirName,
  refForPrs,
  refMoves,
  refNeedsGit,
  refNotFound,
  refProvenance,
  refPrs,
  tarballUrl,
} from "../src/lib/srcref.ts";
import {
  CUDA_COMPONENTS,
  cudaDiskNeededB,
  cudaFiles,
  cudaFilesUsable,
  cudaFixSummary,
} from "../src/lib/cudaredist.ts";

const GB = 1024 ** 3;
const MB = 1024 ** 2;

// ── catalog ────────────────────────────────────────────────────────────────

Deno.test("catalog: every parameter is complete and uniquely keyed", () => {
  const keys = new Set<string>();
  for (const p of PARAMS) {
    assert(!keys.has(p.key), `duplicate key ${p.key}`);
    keys.add(p.key);
    // One entry has no flag of its own: the extra-arguments escape hatch,
    // whose value IS the argv it contributes (see emit() in command.ts).
    assert(
      p.flag.startsWith("-") || p.key === "extraArgs",
      `${p.key}: flag must start with '-'`,
    );
    assert(p.tip.length > 20, `${p.key}: tooltip is too thin to help anyone`);
    assert(p.label.length > 0, `${p.key}: needs a label`);
    if (p.kind === "enum") {
      assert(p.options && p.options.length > 1, `${p.key}: enum needs options`);
      assert(
        p.options?.includes(String(p.def)),
        `${p.key}: default ${p.def} is not one of its options`,
      );
    }
    // A boolean defaulting to ON must be able to express OFF.
    if (p.kind === "bool" && p.def === true) {
      assert(p.offFlag, `${p.key}: default-on boolean needs an offFlag`);
    }
  }
});

Deno.test("catalog: flags are unique across the catalog", () => {
  const seen = new Map<string, string>();
  for (const p of PARAMS) {
    const prev = seen.get(p.flag);
    assert(!prev, `${p.flag} used by both ${prev} and ${p.key}`);
    seen.set(p.flag, p.key);
  }
});

Deno.test("params: coerce clamps, rejects NaN, and keeps types", () => {
  const ngl = param("ngl")!;
  assertEquals(coerce(ngl, "12"), 12);
  assertEquals(coerce(ngl, "-5"), 0, "clamped to min");
  assertEquals(coerce(ngl, "99999"), 999, "clamped to max");
  assertEquals(coerce(ngl, "abc"), ngl.def, "NaN falls back to the default");
  const temp = param("temp")!;
  assertEquals(coerce(temp, "0.35"), 0.35);
  const mlock = param("mlock")!;
  assertEquals(coerce(mlock, true), true);
  assertEquals(coerce(mlock, "false"), false);
});

Deno.test("params: a range that the tip documents is actually reachable", () => {
  // The `keep` tip says "-1 keeps all of them"; a control whose `min` clamps
  // -1 to 0 would silently turn "keep everything" into "keep nothing", with no
  // error on screen. The documented value has to be inside the range.
  const keep = param("keep")!;
  assertEquals(
    coerce(keep, "-1"),
    -1,
    "-1 must survive coerce, the tip names it",
  );
  assert(
    keep.min !== undefined && keep.min <= -1,
    "min must not clamp -1 away",
  );
});

Deno.test("params: typed readers fall back to catalog defaults", () => {
  assertEquals(num({}, "ctxSize"), 4096);
  assertEquals(str({}, "host"), "127.0.0.1");
  assertEquals(bool({}, "mlock"), false);
  assertEquals(num({ ctxSize: 32768 }, "ctxSize"), 32768);
});

// ── command building ───────────────────────────────────────────────────────

Deno.test("command: a flag is omitted only when llama.cpp would agree", () => {
  // A parameter at the catalog's default used to be dropped from the argv, on
  // the assumption that the catalog's default IS llama.cpp's. Upstream moved
  // and the assumption went stale silently, in the worst possible direction:
  // `-ngl` now defaults to AUTO, so "CPU only" emitted nothing and llama.cpp
  // offloaded to the GPU; `-c` defaults to 0 = take it from the model, so a
  // plan drawn for 4,096 tokens started a server at the model's declared
  // 1,048,576 and could not allocate. Omission is judged against `llamaDef`
  // now, and the three flags that DECIDE the placement always appear.
  const cmd = argv("server", {
    bin: "/b/llama-server",
    model: "/m/x.gguf",
    settings: defaults(),
  });
  assertEquals(cmd.slice(0, 3), ["/b/llama-server", "-m", "/m/x.gguf"]);
  for (const flag of ["-ngl", "-c", "-np"]) {
    assert(cmd.includes(flag), `${flag} must be explicit: ${cmd.join(" ")}`);
  }
  // And the value that follows is the one the panel is showing.
  assertEquals(cmd[cmd.indexOf("-ngl") + 1], "0");
  assertEquals(cmd[cmd.indexOf("-c") + 1], "4096");
  assertEquals(cmd[cmd.indexOf("-np") + 1], "1");
  // Everything whose catalog default really is llama.cpp's stays absent — the
  // command is still short enough to read, which is the point of omitting.
  for (const flag of ["-fa", "-ts", "--mlock", "--no-mmap", "-ctk", "-ctv"]) {
    assertEquals(cmd.includes(flag), false, `${flag} should be absent`);
  }
});

Deno.test("command: three switches that were inert now move the command", () => {
  // Each of these was a boolean with `def: false` and no `offFlag` for a flag
  // that is ON upstream — so "on" emitted a no-op and "off" emitted nothing,
  // and the server ran the same either way while the panel claimed a choice.
  // Verified against common.h: `use_jinja = true`, `endpoint_slots = true`,
  // `ctx_shift = false`.
  const line = (s: Record<string, string | number | boolean>) =>
    argv("server", {
      bin: "/b/llama-server",
      model: "/m/x.gguf",
      settings: { ...defaults(), ...s },
    });
  const base = line({});
  for (const flag of ["--jinja", "--no-jinja", "--slots", "--no-slots"]) {
    assertEquals(base.includes(flag), false, `${flag} at the default`);
  }
  assert(line({ jinja: false }).includes("--no-jinja"), "off is sayable");
  assert(line({ slots: false }).includes("--no-slots"), "off is sayable");
  assert(line({ contextShift: true }).includes("--context-shift"));
  assertEquals(
    line({ contextShift: false }).includes("--no-context-shift"),
    false,
    "the default is llama.cpp's own — nothing to say",
  );
  // Deprecated upstream (its value is ignored): gone from the catalog rather
  // than emitted as a flag that only prints a warning.
  assertEquals(param("defragThold"), undefined);
  assertEquals(param("noContextShift"), undefined);
  // `--cache-reuse`: off upstream, 256 in every upstream preset — emitted.
  assertEquals(base[base.indexOf("--cache-reuse") + 1], "256");
  // The read timeout is upstream's own again: 600 s was shorter than a big
  // prompt takes to prefill, so the server cut the request off.
  assertEquals(base.includes("-to"), false);
  assertEquals(param("timeout")?.def, 3600);
});

Deno.test("stability: /slots on an open bind is named, with its own switch", () => {
  const open = { ...defaults(), host: "0.0.0.0" };
  const st = stability(meta(), hw(), open);
  assert(st.warnings.some((w) => w.key === "slots"), "slots is on by default");
  const closed = stability(meta(), hw(), { ...open, slots: false });
  assertEquals(closed.warnings.some((w) => w.key === "slots"), false);
  const local = stability(meta(), hw(), { ...defaults(), host: "127.0.0.1" });
  assertEquals(
    local.warnings.some((w) => w.key === "slots"),
    false,
    "private bind — a debugging aid, not a leak",
  );
});

Deno.test("command: --fit off is always emitted, because upstream's fitter defaults on", () => {
  // llama-server defaults to fit_params = true (common.h): it silently adjusts
  // whatever the command line left unset to fit device memory. This app is that
  // fitter — the plan, the packer, the fit ladder — and its promise is that the
  // command shown is the command that runs, so the flag is pinned like -ngl,
  // -c and -np. (Upstream's fitter also segfaults on DeepSeek-V4-Flash.)
  const cmd = argv("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: defaults(),
  });
  assertEquals(cmd[cmd.indexOf("--fit") + 1], "off");
  // Choosing "on" means llama.cpp's own default — the honest spelling of which
  // is no flag at all.
  const on = argv("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), fit: "on" },
  });
  assertEquals(on.includes("--fit"), false);
});

Deno.test("command: only changed values appear, in catalog order", () => {
  const cmd = argv("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), ngl: 99, ctxSize: 16384, flashAttn: "on" },
  });
  // Catalog order, and the always-emitted placement flags carry their values.
  assertEquals(cmd.slice(0, 5), [
    "llama-server",
    "-m",
    "/m/x.gguf",
    "-ngl",
    "99",
  ]);
  assertEquals(cmd[cmd.indexOf("-c") + 1], "16384");
  assertEquals(cmd[cmd.indexOf("-fa") + 1], "on");
  assert(cmd.indexOf("-fa") > cmd.indexOf("-ngl"), "catalog order holds");
});

Deno.test("command: a default-on boolean emits its negative flag when off", () => {
  const on = argv("server", {
    bin: "s",
    model: "",
    settings: { ...defaults(), contBatching: true },
  });
  assertEquals(on.includes("--no-cont-batching"), false, "on is the default");
  const off = argv("server", {
    bin: "s",
    model: "",
    settings: { ...defaults(), contBatching: false },
  });
  assert(off.includes("--no-cont-batching"));
});

Deno.test("command: scope separates server-only from cli-only flags", () => {
  const settings = { ...defaults(), port: 9090, nPredict: 128 };
  const server = argv("server", { bin: "s", model: "", settings });
  const cli = argv("cli", { bin: "c", model: "", settings });
  assert(server.includes("--port") && server.includes("9090"));
  assert(!server.includes("-n"), "cli-only flag must not reach the server");
  assert(cli.includes("-n") && cli.includes("128"));
  assert(!cli.includes("--port"), "server-only flag must not reach the cli");
});

Deno.test("command: empty text and enum values never emit a bare flag", () => {
  const cmd = argv("server", {
    bin: "s",
    model: "",
    settings: { ...defaults(), alias: "", numa: "", tensorSplit: "" },
  });
  for (const flag of ["-a", "--numa", "-ts"]) {
    assertEquals(cmd.includes(flag), false, `${flag} must not appear bare`);
  }
});

Deno.test("command: quoting survives spaces and single quotes", () => {
  assertEquals(quote("/models/a.gguf"), "/models/a.gguf");
  assertEquals(quote("/my models/a.gguf"), "'/my models/a.gguf'");
  assertEquals(quote("it's"), `'it'\\''s'`);
  assertEquals(quote(""), "''");
  const line = commandLine("server", {
    bin: "s",
    model: "/my models/a.gguf",
    settings: defaults(),
  });
  assertStringIncludes(line, "s -m '/my models/a.gguf'");
});

Deno.test("command: the home prefix compacts to $HOME, and only where a shell would expand it", () => {
  const home = "/home/dev";
  const opts = {
    bin: `${home}/.llama-master/data/files/builds/cuda/bin/llama-server`,
    model: `${home}/.lmstudio/models/q/Qwen3.8-27B-Q8_0.gguf`,
    settings: defaults(),
  };
  // Display compaction: both the one-liner and the block show `$HOME/…`,
  // which pastes back to the identical absolute path.
  const line = commandLine("server", { ...opts, home });
  assertStringIncludes(
    line,
    "$HOME/.llama-master/data/files/builds/cuda/bin/llama-server",
  );
  assertStringIncludes(
    line,
    "-m $HOME/.lmstudio/models/q/Qwen3.8-27B-Q8_0.gguf",
  );
  const block = commandBlock("server", { ...opts, home });
  assertStringIncludes(block[0] ?? "", "$HOME/");
  // No home given (or not yet read): nothing changes.
  assertStringIncludes(commandLine("server", opts), opts.bin);
  // A path that needs quoting keeps its absolute spelling — quotes would
  // silence the `$HOME` and paste a path that does not exist.
  const spaced = commandLine("server", {
    ...opts,
    model: `${home}/my models/a.gguf`,
    home,
  });
  assertStringIncludes(spaced, `-m '${home}/my models/a.gguf'`);
  // Prefix means PATH prefix: `/home/devote/…` is not under `/home/dev`.
  const sibling = commandLine("server", {
    ...opts,
    model: "/home/devote/a.gguf",
    home,
  });
  assertStringIncludes(sibling, "-m /home/devote/a.gguf");
});

Deno.test("setup: the flags come back as sentences, and they follow the settings", () => {
  // The tuned Qwen3.8-27B run: everything on the GPU, q8 cache, MTP drafting.
  const m = meta({
    nLayer: 65,
    nCtxTrain: 262144,
    nextnLayers: 1,
    layers: layers(65, 400 * 1024 * 1024),
  });
  const machine = hw({ gpus: [gpu(24), gpu(24)] });
  const s = {
    ...defaults(),
    ngl: 999,
    ctxSize: 262144,
    cacheTypeK: "q8_0",
    cacheTypeV: "f16",
    threads: 16,
    threadsBatch: 16,
    specType: "draft-mtp",
    flashAttn: "on",
    tensorSplit: "37.5,28.5",
    port: 18080,
  };
  const rows = setupRows(m, s, machine);
  const row = (label: string) =>
    rows.find((r) => r.label === label)?.value ?? "(missing)";
  assertEquals(row("GPU offload"), "all 65 layers + output head");
  assertStringIncludes(row("CPU threads"), "16 generate · 16 batch");
  assertStringIncludes(row("K cache type"), "q8_0");
  assertStringIncludes(row("V cache type"), "f16");
  assertStringIncludes(row("Context"), "262,144 tokens — the trained maximum");
  assertStringIncludes(row("Speculative decoding"), "MTP");
  assertStringIncludes(row("Tensor split"), "-ts 37.5,28.5");
  // A port is an identifier: 18080, never 18,080.
  assertStringIncludes(row("Listening on"), "127.0.0.1:18080");
  assertStringIncludes(row("Listening on"), "this machine only");
  // Rows that only matter when set stay out of the way.
  assert(!rows.some((r) => r.label === "MoE experts on CPU"));
  assert(!rows.some((r) => r.label === "KV cache placement"));

  // The chip forms — same facts, fewest characters, for the all-in-one page.
  const short = (label: string) =>
    rows.find((r) => r.label === label)?.short ?? "(missing)";
  assertEquals(short("GPU offload"), "GPU all 65+head");
  assertEquals(short("CPU threads"), "t 16·16");
  assertEquals(short("K cache type"), "K q8_0");
  assertEquals(short("V cache type"), "V f16");
  assertEquals(short("Context"), "ctx 262,144 max");
  assertEquals(short("Speculative decoding"), "MTP");
  assertEquals(short("Listening on"), "local :18080");
  // Every row has one — the compact view may not come up blank anywhere.
  for (const r of rows) assert(r.short.length > 0, r.label);

  // Partial offload with experts held back, KV pinned to host, LAN bind.
  const partial = setupRows(m, {
    ...s,
    ngl: 43,
    nCpuMoe: 30,
    noKvOffload: true,
    host: "0.0.0.0",
    tensorSplit: "",
    specType: "",
  }, machine);
  const p = (label: string) =>
    partial.find((r) => r.label === label)?.value ?? "(missing)";
  // 43 slots = 42 layers + the output head; layer 0..22 stay on the host.
  assertEquals(p("GPU offload"), "42 of 65 layers + output head");
  assertStringIncludes(p("MoE experts on CPU"), "first 30 layers");
  assertEquals(p("KV cache placement"), "system RAM (-nkvo)");
  assertStringIncludes(p("Listening on"), "reachable from the LAN");
  assertEquals(p("Speculative decoding"), "off");
  assert(!partial.some((r) => r.label === "Tensor split"));

  // CPU-only says so instead of counting to zero.
  assertEquals(
    setupRows(m, { ...s, ngl: 0 }, machine).find((r) =>
      r.label === "GPU offload"
    )?.value,
    "off — CPU only",
  );
});

Deno.test("command: the block form keeps each flag with its value", () => {
  const lines = commandBlock("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), ngl: 99, mlock: true },
  });
  assertEquals(lines.slice(0, 3), [
    "llama-server",
    "  -m /m/x.gguf",
    "  -ngl 99",
  ]);
  assert(lines.includes("  --mlock"), lines.join("\n"));
  // One flag per line, value attached to it — never a line of bare values.
  for (const l of lines.slice(1)) assert(l.startsWith("  -"), l);
});

Deno.test("command: a negative VALUE stays attached to its flag in the block", () => {
  // `-1` starts with `-`, exactly like a flag — a block builder that judged
  // "is this a flag" by that prefix orphaned the value onto its own line:
  // `--repeat-last-n` then `-1`, which is a command that cannot be read.
  // The value is distinguished by shape: `-1` is a number, every real flag is
  // not.
  const lines = commandBlock("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), repeatLastN: -1, ropeFreqScale: -0.5 },
  });
  assert(lines.includes("  --repeat-last-n -1"), lines.join("\n"));
  assert(lines.includes("  --rope-freq-scale -0.5"), lines.join("\n"));
  assert(
    lines.every((l) => l === "llama-server" || l.startsWith("  -")),
    "every value is attached to its flag",
  );
});

Deno.test("command: a wildcard bind address dials loopback", () => {
  assertEquals(
    serverUrl({ host: "0.0.0.0", port: 8080 }),
    "http://127.0.0.1:8080",
  );
  assertEquals(
    serverUrl({ host: "192.168.1.5", port: 99 }),
    "http://192.168.1.5:99",
  );
  assertEquals(serverUrl({}), "http://127.0.0.1:8080");
});

// ── the planner ────────────────────────────────────────────────────────────

Deno.test("plan: KV cache scales with context, heads, and cache precision", () => {
  const m = meta();
  const f16 = kvPerToken(m, defaults());
  // 32 layers × 8 kv heads × (128 + 128) × 2 bytes
  assertEquals(f16, 32 * 8 * (128 * 2 + 128 * 2));
  const q8 = kvPerToken(m, {
    ...defaults(),
    cacheTypeK: "q8_0",
    cacheTypeV: "q8_0",
  });
  assert(q8 < f16 * 0.6, "q8_0 must roughly halve the cache");
});

Deno.test("plan: -c 0 means the model's trained context", () => {
  assertEquals(
    effectiveCtx(meta({ nCtxTrain: 131072 }), { ctxSize: 0 }),
    131072,
  );
  assertEquals(effectiveCtx(meta(), { ctxSize: 2048 }), 2048);
});

Deno.test("plan: with no offload every byte of weight is in RAM", () => {
  const m = meta();
  const p = plan(m, hw(), { ...defaults(), ngl: 0 });
  assertEquals(p.layersOnGpu, 0);
  assertEquals(
    p.ram.buckets.find((b) => b.key === "weights")?.bytes,
    m.tensorBytes,
  );
  assertEquals(p.vram.usedB, 0, "an unused GPU claims nothing");
});

/** The embedding table is the exception, and it is llama.cpp's, not ours: it is
 *  an INPUT tensor, and those are pinned to the host at any `-ngl` ("there is
 *  very little benefit to offloading the input layer" — `llama-model.cpp`,
 *  `dev_input`). Billing it to VRAM spent ~1 GB of a card's budget on bytes that
 *  were never going to be there. */
Deno.test("plan: full offload puts every layer and the head in VRAM, but never the embeddings", () => {
  const m = meta();
  const p = plan(m, hw({ gpus: [gpu(48)] }), { ...defaults(), ngl: 999 });
  assertEquals(p.layersOnGpu, m.nLayer);
  const w = p.vram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  assertEquals(w, m.tensorBytes - m.embdBytes);
  assertEquals(
    p.ram.buckets.find((b) => b.key === "weights")?.bytes,
    m.embdBytes,
    "the embedding table stays on the host",
  );
});

/** `-ngl N` offloads the last N SLOTS, and the output head is one of them —
 *  there are `nLayer + 1`. So `-ngl 16` on a 32-layer model does not put 16
 *  layers on the card; it puts 15 layers and the head. This is llama.cpp's own
 *  `i_gpu_start` / `act_gpu_layers` arithmetic (`src/lib/devsplit.ts`). */
Deno.test("plan: partial offload splits weights and KV by llama.cpp's slot count", () => {
  const m = meta();
  const p = plan(m, hw(), { ...defaults(), ngl: 16 });
  const vw = p.vram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  const rw = p.ram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  assertEquals(vw + rw, m.tensorBytes, "no byte is lost or double-counted");
  assertEquals(
    p.layersOnGpu,
    15,
    "15 layers and the output head, not 16 layers",
  );
  const vkv = p.vram.buckets.find((b) => b.key === "kv")?.bytes ?? 0;
  assertEquals(
    Math.round(vkv),
    Math.round(p.kvTotalB * 15 / m.nLayer),
    "the KV follows the layers that are actually offloaded",
  );
});

Deno.test("plan: --n-cpu-moe keeps attention on the GPU and experts in RAM", () => {
  const m = moeMeta();
  const full = plan(m, hw({ gpus: [gpu(80)] }), { ...defaults(), ngl: 999 });
  const split = plan(m, hw({ gpus: [gpu(80)] }), {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 32,
  });
  const attentionOnly = m.layers.reduce((a, l) => a + (l.bytes - l.expert), 0);
  assertEquals(split.moeOnCpu, 32);
  assertEquals(
    split.vram.buckets.find((b) => b.key === "weights")?.bytes,
    attentionOnly + m.outputBytes,
    "the head comes too; the embedding table never does",
  );
  assert(
    split.vram.usedB < full.vram.usedB / 3,
    "moving every expert must free most of the VRAM",
  );
  assert(
    split.vram.buckets.find((b) => b.key === "kv")!.bytes > 0,
    "the KV cache still lives with attention on the GPU",
  );
});

Deno.test("plan: -nkvo moves the whole KV cache to RAM", () => {
  const m = meta();
  const p = plan(m, hw(), { ...defaults(), ngl: 999, noKvOffload: true });
  assertEquals(p.vram.buckets.find((b) => b.key === "kv"), undefined);
  assertEquals(
    Math.round(p.ram.buckets.find((b) => b.key === "kv")?.bytes ?? 0),
    Math.round(p.kvTotalB),
  );
});

Deno.test("plan: overflow is reported, not clipped", () => {
  const m = meta({
    layers: Array.from({ length: 80 }, (_, i) => ({
      i,
      bytes: 1024 ** 3,
      expert: 0,
    })),
    nLayer: 80,
  });
  const p = plan(m, hw({ gpus: [gpu(8)] }), { ...defaults(), ngl: 999 });
  assert(!p.fits);
  assert(
    p.vram.overB > 60 * GB,
    `expected a large overflow, got ${p.vram.overB}`,
  );
  assert(p.notes.some((n) => n.includes("Over VRAM")));
});

Deno.test("plan: a machine with no GPU says so and plans for the CPU", () => {
  const p = plan(meta(), NO_GPU, { ...defaults(), ngl: 999 });
  assertEquals(p.vram.capacityB, 0);
  assert(p.notes.some((n) => n.includes("No GPU")));
});

Deno.test("plan: VRAM already in use by other processes counts against capacity", () => {
  const m = meta();
  const idle = plan(m, hw({ gpus: [gpu(24, 0)] }), { ...defaults(), ngl: 8 });
  const busy = plan(m, hw({ gpus: [gpu(24, 20)] }), { ...defaults(), ngl: 8 });
  assertEquals(idle.vram.usedB, busy.vram.usedB, "our own demand is unchanged");
  assert(busy.vram.freeB < idle.vram.freeB, "someone else's 20 GB is not free");
});

// ── the tuner ──────────────────────────────────────────────────────────────

Deno.test("tune: a model that fits goes entirely on the GPU, at full context", () => {
  const m = meta({ nCtxTrain: 8192 });
  const machine = hw({ gpus: [gpu(48)] });
  const t = tune(m, machine, defaults(), "vram");
  assertEquals(t.settings.ngl, 999);
  assertEquals(t.settings.nCpuMoe, 0);
  assertEquals(t.settings.flashAttn, "on");
  assert(t.possible, "VRAM only must be available for a model this size");
  // The goal is the model's trained context, and here there is room for it.
  assertEquals(t.ctx, 8192);
  assertEquals(t.optimalCtx, 8192);
  assert(t.reasons.length >= 2, "every decision is explained");
  assert(t.summary.includes("full"), `summary should say so: ${t.summary}`);
});

Deno.test("tune: leftover VRAM is spent on the micro-batch, and the plan pays for it", () => {
  // Prefill is the wait at long context: 512 → 4096 measured 150 → 364 tok/s
  // prompt ingestion on a MoE with experts on the host. A small model on a big
  // card leaves tens of GB idle; the tuner turns some of it into `-ub`.
  const m = meta({ nCtxTrain: 8192 });
  const roomy = hw({ gpus: [gpu(48)] });
  const t = tune(m, roomy, defaults(), "vram");
  assertEquals(t.settings.ubatchSize, 4096);
  assertEquals(t.settings.batchSize, 4096);
  // The growth is billed, not assumed: the grown settings still fit with the
  // safety margins intact.
  assert(plan(m, roomy, t.settings).fits, "grown plan must still fit");
  assert(
    t.reasons.some((r) => r.includes("Micro-batch")),
    `the spend is explained: ${t.reasons.join(" | ")}`,
  );
});

Deno.test("tune: a starved placement keeps its small micro-batch", () => {
  // When the placement itself cut `-ub` to 256 to buy layers, growth would
  // undo that trade — and on a card with no headroom there is nothing to
  // spend anyway.
  const mb = 1024 ** 2;
  const m = meta({
    nLayer: 60,
    layers: layers(60, 512 * mb),
    nCtxTrain: 32768,
  });
  const tight = hw({ gpus: [gpu(8)] });
  const t = tune(m, tight, defaults(), "hybrid");
  if (t.possible) {
    assert(
      num(t.settings, "ubatchSize") <= 512,
      `no growth on a starved card: ${t.settings.ubatchSize}`,
    );
    assert(plan(m, tight, t.settings).fits);
  }
});

Deno.test("plan: the compute buffer charges the FFN width for a large micro-batch", () => {
  // Prefill activations scale with the widest matmul — the FFN. An estimate
  // built on n_embd alone under-billed `-ub 4096` about 4x, which would let
  // the tuner spend VRAM that is not there.
  const m = meta({ nFf: 25600 });
  const machine = hw({ gpus: [gpu(48)] });
  const at = (ub: number) =>
    plan(m, machine, { ...defaults(), ngl: 999, ubatchSize: ub }).vram.buckets
      .find((b) => b.key === "compute")!.bytes;
  const grown = at(4096) - at(512);
  // Two FFN-wide f32 tensors over the extra 3,584 tokens, at least.
  assert(
    grown >= (4096 - 512) * 25600 * 4 * 2,
    `compute must grow with ub x n_ff: +${grown}`,
  );
});

Deno.test("tune: the context aimed at is the model's trained maximum", () => {
  // "Optimal context" is not a number this app invents — past `nCtxTrain` the
  // positional encoding is extrapolated and answers get worse, so that is the
  // ceiling, and no placement is allowed to exceed it.
  for (const trained of [4096, 32768, 262144]) {
    const m = meta({ nCtxTrain: trained });
    assertEquals(optimalCtx(m), trained);
    const t = tune(m, hw({ gpus: [gpu(48, 0)] }), defaults(), "vram");
    assertEquals(t.optimalCtx, trained);
    assert(t.ctx <= trained, `never past trained: ${t.ctx} > ${trained}`);
  }
});

Deno.test("tune: when the optimal context does not fit, it takes the largest that does", () => {
  // The rule: get as close to optimal as memory allows, never beyond it.
  const m = meta({ nCtxTrain: 262144 });
  const roomy = tune(m, hw({ gpus: [gpu(80, 0)] }), defaults(), "vram");
  const tight = tune(m, hw({ gpus: [gpu(12, 0)] }), defaults(), "vram");
  assert(tight.possible, "12 GB can still run an 8B model");
  assert(
    tight.ctx < roomy.ctx,
    `a smaller card must mean a smaller context: ${tight.ctx} vs ${roomy.ctx}`,
  );
  assert(tight.ctx >= MIN_CTX, "and still a usable one");
  // It is the LARGEST that fits: one step more must not.
  const machine = hw({ gpus: [gpu(12, 0)] });
  const bigger = plan(m, machine, {
    ...tight.settings,
    ctxSize: tight.ctx + 256,
  });
  const total = 12 * 1024 ** 3;
  assert(
    bigger.vram.overB > 0 ||
      bigger.vram.freeB < Math.max(512 * 1024 * 1024, total * 0.05),
    `${tight.ctx + 256} also fits, so ${tight.ctx} was not the largest`,
  );
});

Deno.test("tune: a 2-core machine still gets a usable thread count", () => {
  const tiny = hw({
    cpu: {
      model: "tiny",
      cores: 2,
      threads: 2,
      mhz: 1000,
      tempC: 0,
      utilPct: 0,
      stat: "",
      coreStats: [],
      coresUtil: [],
    },
  });
  const { settings } = tune(meta(), tiny, defaults(), "vram");
  assertEquals(settings.threads, 2, "one thread per physical core");
  assertEquals(settings.threadsBatch, 2, "prefill too");
});

Deno.test("tune: a tight fit quantises the KV cache before dropping layers", () => {
  // 8B weights ≈ 4.6 GB; on a 12 GB card an 8k f16 cache is what tips it over.
  const m = meta({ nCtxTrain: 131072 });
  const { settings, reasons } = tune(m, hw({ gpus: [gpu(12, 0)] }), {
    ...defaults(),
    ctxSize: 65536,
  });
  assertEquals(settings.cacheTypeK, "q8_0");
  assertEquals(settings.ngl, 999, "the cache quantisation should be enough");
  assert(reasons.some((r) => r.includes("q8_0")));
});

Deno.test("tune: hybrid moves a MoE model's experts, not its layers", () => {
  const m = moeMeta();
  const machine = hw({ gpus: [gpu(24, 0)] });
  const t = tune(m, machine, defaults(), "hybrid");
  assert(t.possible);
  assertEquals(t.settings.ngl, 999, "attention stays on the GPU");
  assert(Number(t.settings.nCpuMoe) > 0, "experts move to RAM");
  assert(
    t.reasons.some((r) => r.includes("routed experts")),
    `the reason must name them: ${JSON.stringify(t.reasons)}`,
  );
  assertEquals(plan(m, machine, t.settings).vram.overB, 0);
});

Deno.test("tune: a dense model too big falls back to partial offload that fits", () => {
  const m = meta({
    nLayer: 80,
    layers: Array.from({ length: 80 }, (_, i) => ({
      i,
      bytes: 512 * 1024 ** 2,
      expert: 0,
    })),
  });
  const machine = hw({ gpus: [gpu(24, 0)] });
  const { settings } = tune(m, machine, defaults(), "hybrid");
  const n = Number(settings.ngl);
  assert(n > 0 && n < 80, `expected a partial offload, got ${n}`);
  assertEquals(
    plan(m, machine, settings).vram.overB,
    0,
    "the tuned plan must fit",
  );
});

Deno.test("tune: with no GPU only the CPU placement is possible", () => {
  const cpu = tune(meta(), NO_GPU, defaults(), "cpu");
  assertEquals(cpu.settings.ngl, 0);
  assert(cpu.possible, "CPU only always works");

  // And the GPU placements say so rather than pretending.
  for (const p of ["vram", "hybrid"] as const) {
    const t = tune(meta(), NO_GPU, defaults(), p);
    assertEquals(t.possible, false);
    assertStringIncludes(t.blocker, "No GPU");
  }
});

Deno.test("tune: output is idempotent — tuning a tuned config changes nothing", () => {
  const m = moeMeta();
  const machine = hw({ gpus: [gpu(24, 0)] });
  const first = tune(m, machine, defaults(), "hybrid").settings;
  const second = tune(m, machine, first, "hybrid").settings;
  assertEquals(second, first);
});

Deno.test("tune: never leaves the machine over its RAM budget silently", () => {
  const m = meta({
    nLayer: 80,
    layers: Array.from({ length: 80 }, (_, i) => ({
      i,
      bytes: 4 * 1024 ** 3,
      expert: 0,
    })),
  });
  const machine = hw({
    gpus: [],
    mem: {
      totalB: 16 * GB,
      availableB: 12 * GB,
      usedB: 4 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });
  const { settings, reasons } = tune(m, machine, defaults(), "cpu");
  assertEquals(settings.mlock, false, "never pin what does not fit");
  assert(
    reasons.some((r) => r.includes("RAM")),
    `the shortfall must be named: ${JSON.stringify(reasons)}`,
  );
});

// ── formatting ─────────────────────────────────────────────────────────────

Deno.test("format: bytes read the way a person would say them", () => {
  assertEquals(bytes(0), "0 B");
  assertEquals(bytes(-1), "0 B");
  assertEquals(bytes(512), "512 B");
  assertEquals(bytes(1536), "1.50 KB");
  assertEquals(bytes(24 * GB), "24.0 GB");
  assertEquals(bytes(1.5 * GB), "1.50 GB");
});

Deno.test("format: duration, percent, tokens/s and path shortening", () => {
  assertEquals(duration(812), "812ms");
  assertEquals(duration(45_000), "45s");
  assertEquals(duration(252_000), "4m 12s");
  assertEquals(duration(3_780_000), "1h 03m");
  assertEquals(duration(-1), "—");
  assertEquals(pct(5, 10), 50);
  assertEquals(pct(5, 0), 0, "never NaN");
  assertEquals(pct(20, 10), 100, "never past full");
  assertEquals(tps(0), "—");
  assertEquals(tps(8.234), "8.23");
  assertEquals(tps(812.4), "812");
  assertEquals(shortPath("/a/b.gguf"), "/a/b.gguf");
  assert(shortPath("/very/long/path/".repeat(6) + "m.gguf").startsWith("…/"));
});

// ── /proc/stat deltas ──────────────────────────────────────────────────────

Deno.test("procstat: utilization is the busy share between two samples", () => {
  assertEquals(utilPct("cpu 0 0 0 100", "cpu 0 0 0 200"), 0, "all idle");
  assertEquals(utilPct("cpu 0 0 0 100", "cpu 100 0 0 100"), 100, "all busy");
  assertEquals(utilPct("cpu 0 0 0 100", "cpu 50 0 0 150"), 50);
});

Deno.test("procstat: a first or malformed sample reports 0, never NaN", () => {
  assertEquals(utilPct("", "cpu 1 2 3 4"), 0);
  assertEquals(utilPct("cpu 1 2 3 4", ""), 0);
  assertEquals(utilPct("cpu 1 2 3 4", "cpu 1 2 3 4"), 0, "no time passed");
});

Deno.test("procstat: per-core pairs by position and history stays bounded", () => {
  const u = coresUtilPct(["cpu0 0 0 0 100"], [
    "cpu0 100 0 0 100",
    "cpu1 1 0 0 1",
  ]);
  assertEquals(u.length, 2);
  assertEquals(u[0], 100);
  assertEquals(u[1], 0, "a core with no previous sample reports 0");

  let h: number[] = [];
  for (let i = 0; i < 100; i++) h = pushHistory(h, i, 60);
  assertEquals(h.length, 60);
  assertEquals(h[59], 99, "newest sample is last");
  assertEquals(h[0], 40);
});

// ── build log ──────────────────────────────────────────────────────────────

Deno.test("buildlog: cmake percentages become progress", () => {
  assertEquals(
    progressOf("[ 42%] Building CXX object ggml/CMakeFiles/x.o"),
    0.42,
  );
  assertEquals(progressOf("[100%] Linking CXX executable llama-server"), 1);
  assertEquals(progressOf("-- Configuring done"), null);
  assertEquals(progressOf(""), null);
});

Deno.test("buildlog: errors are recognised, warnings are not", () => {
  assert(isError("src/x.cpp:1:2: error: no member named 'y'"));
  assert(isError("undefined reference to `cublasCreate'"));
  assert(!isError("src/x.cpp:1:2: warning: unused variable"));
});

Deno.test("buildlog: the tail is bounded and keeps the newest lines", () => {
  let log: string[] = [];
  for (let i = 0; i < 1000; i++) log = appendLog(log, [`line ${i}`], 400);
  assertEquals(log.length, 400);
  assertEquals(log[399], "line 999");
});

// ── SSE ────────────────────────────────────────────────────────────────────

Deno.test("sse: a token split across chunk boundaries is not lost", () => {
  const a = parseSse('data: {"choices":[{"delta":{"content":"Hel');
  assertEquals(a.events.length, 0);
  const b = parseSse(a.rest + 'lo"}}]}\n\ndata: [DONE]\n\n');
  assertEquals(b.events.length, 2);
  assertEquals(deltaText(b.events[0]!.data), "Hello");
  assertEquals(b.events[1]!.data, "[DONE]");
});

Deno.test("sse: a thinking model's reasoning channel is read, and kept apart", () => {
  // Verbatim shape from llama.cpp serving DeepSeek-V4: the whole first act of
  // a reply streams as `reasoning_content` with `content` absent. A client
  // that reads only `content` renders nothing for minutes — a real user
  // watched "thinking" produce no answer and reasonably called chat broken.
  const thinkChunk = JSON.stringify({
    choices: [{ delta: { reasoning_content: "Analyze the request. " } }],
  });
  assertEquals(deltaReasoning(thinkChunk), "Analyze the request. ");
  assertEquals(deltaText(thinkChunk), "", "reasoning is not the answer");

  const answerChunk = JSON.stringify({
    choices: [{ delta: { content: "Hello!" } }],
  });
  assertEquals(deltaText(answerChunk), "Hello!");
  assertEquals(deltaReasoning(answerChunk), "");
  assertEquals(deltaReasoning("not json"), "");
});

Deno.test("sse: CRLF framing and keep-alive frames are handled", () => {
  const r = parseSse('data: {"choices":[{"delta":{}}]}\r\n\r\n');
  assertEquals(r.events.length, 1);
  assertEquals(
    deltaText(r.events[0]!.data),
    "",
    "a role-only frame has no text",
  );
  assertEquals(deltaText("not json"), "");
});

Deno.test("sse: the flush cadence holds the byte rate flat, not the interval", () => {
  // The defect: a fixed 60 ms flush re-sends the WHOLE reply 16.7 times a
  // second to every client, so the wire cost of one answer is quadratic in its
  // length. A 23-minute chat session logged a sustained
  // `PRESSURE — 33 broadcasts/sec` against a threshold of 30, with two windows
  // open. The cadence must therefore depend on the size of what it re-sends.
  const rate = (bytes: number) => (bytes * 1000) / flushDelayMs(bytes);

  // Short reply: as fast as before — a fixed 60 ms is the floor, not the rule.
  assertEquals(flushDelayMs(0), 60);
  assertEquals(flushDelayMs(1_000), 60);

  // Long reply: slower, and never slower than half a second.
  assert(flushDelayMs(32_000) > 60, "a 32 KB reply must not flush at 16 Hz");
  assertEquals(flushDelayMs(1_000_000), 500);

  // The point of the whole thing: bytes/second stays inside the budget while
  // the budget is what binds, where a fixed interval grows without bound.
  for (const b of [1_000, 8_000, 32_000]) {
    assert(
      rate(b) <= 70_000,
      `a ${b}-byte reply pushes ${Math.round(rate(b))} B/s`,
    );
  }
  // Past that the ½-second liveness cap binds instead, so the rate does climb
  // again — but a long reply still costs an order of magnitude less than the
  // fixed 60 ms cadence it replaces, which is the whole complaint.
  for (const b of [64_000, 250_000]) {
    assert(
      rate(b) < (b * 1000) / 60 / 8,
      `a ${b}-byte reply must beat the fixed cadence by 8x`,
    );
  }

  // A hostile/absent length can never make the cadence a busy loop.
  assertEquals(flushDelayMs(NaN), 60);
  assertEquals(flushDelayMs(-1), 60);
});

Deno.test("sse: llama.cpp timings surface as tokens/second", () => {
  assertEquals(timingsTps('{"timings":{"predicted_per_second":42.5}}'), 42.5);
  assertEquals(timingsTps('{"choices":[]}'), null);
  assertEquals(timingsTps("nope"), null);
});

// ── release asset selection ────────────────────────────────────────────────

const ASSETS = [
  { name: "llama-b6234-bin-ubuntu-x64.zip", url: "u1", sizeB: 1 },
  { name: "llama-b6234-bin-ubuntu-vulkan-x64.zip", url: "u2", sizeB: 1 },
  { name: "llama-b6234-bin-win-cuda-12.4-x64.zip", url: "u3", sizeB: 1 },
  { name: "llama-b6234-bin-macos-arm64.zip", url: "u4", sizeB: 1 },
  { name: "llama-b6234-bin-ubuntu-arm64.zip", url: "u5", sizeB: 1 },
  { name: "Source code (zip)", url: "u6", sizeB: 1 },
];

Deno.test("assets: the CPU pick never has an accelerator in its name", () => {
  const a = pickAsset(ASSETS, "linux", "x86_64", "cpu");
  assertEquals(a?.name, "llama-b6234-bin-ubuntu-x64.zip");
});

Deno.test("assets: backend and architecture both have to match", () => {
  assertEquals(
    pickAsset(ASSETS, "linux", "x86_64", "vulkan")?.name,
    "llama-b6234-bin-ubuntu-vulkan-x64.zip",
  );
  assertEquals(
    pickAsset(ASSETS, "darwin", "aarch64", "metal")?.name,
    "llama-b6234-bin-macos-arm64.zip",
  );
  assertEquals(
    pickAsset(ASSETS, "linux", "aarch64", "cpu")?.name,
    "llama-b6234-bin-ubuntu-arm64.zip",
  );
});

Deno.test("assets: a wrong-platform binary is never a fallback", () => {
  assertEquals(pickAsset(ASSETS, "linux", "x86_64", "cuda"), null);
  assertEquals(
    scoreAsset("llama-b1-bin-win-x64.zip", "linux", "x86_64", "cpu"),
    null,
  );
  assertEquals(
    scoreAsset("llama-b1-bin-ubuntu-arm64.zip", "linux", "x86_64", "cpu"),
    null,
  );
});

Deno.test("assets: source archives and debug symbols are not binaries", () => {
  assert(!isBinaryAsset("Source code (zip)"));
  assert(!isBinaryAsset("llama-b1-bin-macos-arm64.dSYM.zip"));
  assert(isBinaryAsset("llama-b1-bin-ubuntu-x64.zip"));
  assertEquals(usableAssets(ASSETS, "linux", "x86_64").length, 2);
});

// The real manifest of llama.cpp release b10144, copied verbatim from the
// GitHub API. Hand-written fixtures agree with whatever you assumed; this one
// caught two wrong assumptions — that Linux CUDA binaries exist (they do not,
// upstream ships CUDA for Windows only) and that Linux assets are .zip.
const REAL_B10144 = [
  "cudart-llama-bin-win-cuda-12.4-x64.zip",
  "cudart-llama-bin-win-cuda-13.3-x64.zip",
  "llama-b10144-bin-android-arm64.tar.gz",
  "llama-b10144-bin-macos-arm64.tar.gz",
  "llama-b10144-bin-macos-x64.tar.gz",
  "llama-b10144-bin-ubuntu-arm64.tar.gz",
  "llama-b10144-bin-ubuntu-openvino-2026.2.1-x64.tar.gz",
  "llama-b10144-bin-ubuntu-rocm-7.2-x64.tar.gz",
  "llama-b10144-bin-ubuntu-s390x.tar.gz",
  "llama-b10144-bin-ubuntu-sycl-fp16-x64.tar.gz",
  "llama-b10144-bin-ubuntu-sycl-fp32-x64.tar.gz",
  "llama-b10144-bin-ubuntu-vulkan-arm64.tar.gz",
  "llama-b10144-bin-ubuntu-vulkan-x64.tar.gz",
  "llama-b10144-bin-ubuntu-x64.tar.gz",
  "llama-b10144-bin-win-cpu-arm64.zip",
  "llama-b10144-bin-win-cpu-x64.zip",
  "llama-b10144-bin-win-cuda-12.4-x64.zip",
  "llama-b10144-bin-win-cuda-13.3-x64.zip",
  "llama-b10144-bin-win-hip-radeon-x64.zip",
  "llama-b10144-bin-win-opencl-adreno-arm64.zip",
  "llama-b10144-bin-win-openvino-2026.2.1-x64.zip",
  "llama-b10144-bin-win-sycl-x64.zip",
  "llama-b10144-bin-win-vulkan-x64.zip",
  "llama-b10144-ui.tar.gz",
  "llama-b10144-xcframework.zip",
].map((name) => ({ name, url: `https://example/${name}`, sizeB: 1 }));

Deno.test("assets: the real b10144 manifest picks correctly per platform", () => {
  assertEquals(
    pickAsset(REAL_B10144, "linux", "x86_64", "cpu")?.name,
    "llama-b10144-bin-ubuntu-x64.tar.gz",
    "openvino and sycl are accelerators, not the plain CPU build",
  );
  assertEquals(
    pickAsset(REAL_B10144, "linux", "x86_64", "vulkan")?.name,
    "llama-b10144-bin-ubuntu-vulkan-x64.tar.gz",
  );
  assertEquals(
    pickAsset(REAL_B10144, "linux", "x86_64", "hip")?.name,
    "llama-b10144-bin-ubuntu-rocm-7.2-x64.tar.gz",
  );
  assertEquals(
    pickAsset(REAL_B10144, "linux", "aarch64", "cpu")?.name,
    "llama-b10144-bin-ubuntu-arm64.tar.gz",
  );
  assertEquals(
    pickAsset(REAL_B10144, "darwin", "aarch64", "metal")?.name,
    "llama-b10144-bin-macos-arm64.tar.gz",
  );
  assert(
    pickAsset(REAL_B10144, "windows", "x86_64", "cuda")?.name.includes(
      "win-cuda",
    ),
  );
});

Deno.test("assets: upstream ships no Linux CUDA build, and we say so", () => {
  assertEquals(pickAsset(REAL_B10144, "linux", "x86_64", "cuda"), null);
  assertEquals(
    availableBackends(REAL_B10144, "linux", "x86_64").sort(),
    ["cpu", "hip", "vulkan"],
  );
  assertEquals(
    availableBackends(REAL_B10144, "windows", "x86_64").sort(),
    ["cpu", "cuda", "hip", "vulkan"],
  );
  // On macOS the one binary IS the Metal build, so both names resolve to it.
  assertEquals(availableBackends(REAL_B10144, "darwin", "aarch64").sort(), [
    "cpu",
    "metal",
  ]);
});

Deno.test("assets: the NVIDIA runtime bundle is never mistaken for llama.cpp", () => {
  assert(!isBinaryAsset("cudart-llama-bin-win-cuda-12.4-x64.zip"));
  assert(!isBinaryAsset("llama-b10144-xcframework.zip"));
  assert(!isBinaryAsset("llama-b10144-ui.tar.gz"));
});

Deno.test("assets: a foreign architecture is never a fallback", () => {
  assertEquals(
    scoreAsset(
      "llama-b10144-bin-ubuntu-s390x.tar.gz",
      "linux",
      "x86_64",
      "cpu",
    ),
    null,
  );
  assertEquals(
    scoreAsset(
      "llama-b10144-bin-android-arm64.tar.gz",
      "linux",
      "aarch64",
      "cpu",
    ),
    null,
  );
});

// ── archives ───────────────────────────────────────────────────────────────

/** Build a tar in memory — the format IS the fixture. */
function tarOf(
  files: { name: string; body: string; mode?: number; link?: string }[],
): Uint8Array {
  const enc = new TextEncoder();
  const blocks: Uint8Array[] = [];
  for (const f of files) {
    const head = new Uint8Array(512);
    const put = (s: string, at: number, len: number) =>
      head.set(enc.encode(s.slice(0, len)), at);
    put(f.name, 0, 100);
    put((f.mode ?? 0o644).toString(8).padStart(7, "0"), 100, 8);
    put((f.link ? 0 : f.body.length).toString(8).padStart(11, "0"), 124, 12);
    head[156] = (f.link ? "2" : "0").charCodeAt(0);
    if (f.link) put(f.link, 157, 100);
    put("ustar\0", 257, 6);
    // Checksum: the header is otherwise valid, and untar() does not verify it.
    put(" ".repeat(8), 148, 8);
    blocks.push(head);
    const body = enc.encode(f.link ? "" : f.body);
    const padded = new Uint8Array(Math.ceil(body.length / 512) * 512);
    padded.set(body);
    blocks.push(padded);
  }
  blocks.push(new Uint8Array(1024));
  const total = blocks.reduce((a, b) => a + b.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of blocks) {
    out.set(b, off);
    off += b.length;
  }
  return out;
}

Deno.test("archive: untar reads names, bodies and modes", () => {
  const entries = untar(
    tarOf([
      { name: "pkg/bin/llama-server", body: "ELF...", mode: 0o755 },
      { name: "pkg/README.md", body: "hello" },
    ]),
  );
  assertEquals(entries.length, 2);
  assertEquals(entries[0]?.name, "pkg/bin/llama-server");
  assertEquals(entries[0]?.mode, 0o755);
  assertEquals(new TextDecoder().decode(entries[1]?.bytes), "hello");
});

Deno.test("archive: a truncated tar fails loudly instead of half-extracting", () => {
  const full = tarOf([{ name: "a", body: "x".repeat(2000) }]);
  assertThrows(
    () => untar(full.subarray(0, 700)),
    Error,
    "past the end",
  );
});

Deno.test("archive: the release wrapper directory is stripped", () => {
  const entries = [
    {
      name: "llama.cpp-b6234/CMakeLists.txt",
      bytes: new Uint8Array(),
      mode: 0o644,
    },
    { name: "llama.cpp-b6234/src/x.cpp", bytes: new Uint8Array(), mode: 0o644 },
  ];
  assertEquals(stripRoot(entries).map((e) => e.name), [
    "CMakeLists.txt",
    "src/x.cpp",
  ]);
  const twoRoots = [
    { name: "a/x", bytes: new Uint8Array(), mode: 0 },
    { name: "b/y", bytes: new Uint8Array(), mode: 0 },
  ];
  assertEquals(stripRoot(twoRoots).length, 2, "two roots means no wrapper");
});

Deno.test("archive: path traversal entries are dropped", () => {
  const bad = [
    { name: "../../etc/passwd", bytes: new Uint8Array(), mode: 0 },
    { name: "/etc/shadow", bytes: new Uint8Array(), mode: 0 },
    { name: "C:/windows/x", bytes: new Uint8Array(), mode: 0 },
    { name: "ok/file", bytes: new Uint8Array(), mode: 0 },
  ];
  assertEquals(safeEntries(bad).map((e) => e.name), ["ok/file"]);
});

Deno.test("archive: SONAME symlinks survive extraction", () => {
  // llama.cpp's Linux release ships every .so twice — the real file and a
  // SONAME symlink the loader resolves. Dropping the links makes every binary
  // in the archive fail at startup with "cannot open shared object file".
  const entries = untar(
    tarOf([
      { name: "pkg/libllama.so.0.0.10144", body: "ELF" },
      { name: "pkg/libllama.so.0", body: "", link: "libllama.so.0.0.10144" },
      { name: "pkg/libllama.so", body: "", link: "libllama.so.0" },
      { name: "pkg/llama-server", body: "ELF", mode: 0o755 },
    ]),
  );
  assertEquals(entries.length, 4);
  const link = entries.find((e) => e.name === "pkg/libllama.so.0");
  assertEquals(link?.link, "libllama.so.0.0.10144");
  assertEquals(link?.bytes.length, 0);
  assertEquals(
    entries.find((e) => e.name === "pkg/llama-server")?.link,
    undefined,
  );
});

Deno.test("archive: a symlink pointing outside the destination is dropped", () => {
  const bad = [
    { name: "ok", bytes: new Uint8Array(), mode: 0, link: "../../etc/passwd" },
    { name: "abs", bytes: new Uint8Array(), mode: 0, link: "/etc/shadow" },
    { name: "fine", bytes: new Uint8Array(), mode: 0, link: "real.so" },
  ];
  assertEquals(safeEntries(bad).map((e) => e.name), ["fine"]);
});

// ── stability ──────────────────────────────────────────────────────────────

Deno.test("stability: a configuration that fits raises nothing", () => {
  const machine = hw({ gpus: [gpu(48, 0)] });
  const tuned = tune(meta(), machine, defaults()).settings;
  // With the priority switch on — which is its default, and the reason the
  // tuner is free to claim every physical core.
  const st = stability(meta(), machine, tuned, { lowPriority: true });
  assertEquals(st.level, "ok", JSON.stringify(st.warnings));
});

Deno.test("stability: not enough VRAM is a risk, not a hint", () => {
  const machine = hw({ gpus: [gpu(4, 0)] });
  const st = stability(meta(), machine, { ...defaults(), ngl: 999 });
  assertEquals(st.level, "risk");
  assert(
    st.warnings.some((w) => w.key === "ngl" && w.message.includes("VRAM")),
  );
});

Deno.test("stability: claiming every core warns only when nothing yields", () => {
  // The tuner claims every PHYSICAL core on purpose — measured fastest. What
  // keeps the desktop alive under it is the priority switch, so the warning is
  // about the switch being off, not about the thread count.
  const busy = { ...defaults(), threads: 16, ngl: 0 };
  assert(
    stability(meta(), hw(), busy).warnings.some((w) =>
      w.key === "threads" && w.message.includes("Low priority")
    ),
    "normal priority + every core is worth saying",
  );
  assertEquals(
    stability(meta(), hw(), busy, { lowPriority: true }).warnings.filter((w) =>
      w.key === "threads"
    ).length,
    0,
    "at nice 19 the desktop already goes first — nothing to warn about",
  );
  // SMT is not a preference: 15x slower, measured.
  const smt = stability(meta(), hw(), {
    ...defaults(),
    threads: 32,
    ngl: 0,
  }, { lowPriority: true });
  assertEquals(smt.level, "risk", "two threads per core");
  const over = stability(meta(), hw(), { ...defaults(), threads: 64, ngl: 0 });
  assertEquals(over.level, "risk", "more threads than logical CPUs");
});

Deno.test("stability: an impossible batch pair is caught before llama.cpp sees it", () => {
  const st = stability(null, hw(), {
    ...defaults(),
    batchSize: 256,
    ubatchSize: 512,
  });
  assertEquals(st.level, "risk");
  assert(st.warnings.some((w) => w.key === "ubatchSize"));
});

Deno.test("stability: exposing the server with no API key is a risk", () => {
  const st = stability(null, hw(), { ...defaults(), host: "0.0.0.0" });
  assertEquals(st.level, "risk");
  assert(st.warnings.some((w) => w.key === "host"));
  const keyed = stability(null, hw(), {
    ...defaults(),
    host: "0.0.0.0",
    apiKey: "secret",
  });
  assert(!keyed.warnings.some((w) => w.key === "host"));
});

Deno.test("stability: a context past the trained length is a caution, not a block", () => {
  const st = stability(meta({ nCtxTrain: 4096 }), hw({ gpus: [gpu(48, 0)] }), {
    ...defaults(),
    ctxSize: 32768,
    ngl: 999,
  });
  assert(st.warnings.some((w) => w.key === "ctxSize"));
  assertEquals(st.level, "caution");
});

// ── build parallelism ──────────────────────────────────────────────────────

Deno.test("build: auto job count leaves two cores to the operating system", () => {
  assertEquals(autoJobs(32), 30);
  assertEquals(autoJobs(4), 2);
  assertEquals(autoJobs(2), 1, "never zero, never negative");
  assertEquals(autoJobs(1), 1);
});

Deno.test("build: a tagged source build stamps its real version number", () => {
  // A tarball carries no git metadata, so without this the binary reports
  // `version: 0 (unknown)` — verified against a real source build.
  assertEquals(buildNumberFlags("b10144"), ["-DLLAMA_BUILD_NUMBER=10144"]);
  assertEquals(
    buildNumberFlags("master"),
    [],
    "master has no build number yet",
  );
  assertEquals(buildNumberFlags("../evil"), []);
});

/** Build a zip in memory. Windows llama.cpp releases and the Windows CMake
 *  tarball are both .zip, so this path is not optional — and it was the least
 *  covered file in the project until this test existed. */
async function zipOf(
  files: { name: string; body: string; deflate?: boolean; mode?: number }[],
): Promise<Uint8Array> {
  const enc = new TextEncoder();
  const locals: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const f of files) {
    const raw = enc.encode(f.body);
    const data = f.deflate
      ? await new Response(
        new Blob([raw as BlobPart]).stream().pipeThrough(
          new CompressionStream("deflate-raw"),
        ),
      ).bytes()
      : raw;
    const name = enc.encode(f.name);
    const method = f.deflate ? 8 : 0;

    const lh = new Uint8Array(30 + name.length);
    const lv = new DataView(lh.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(8, method, true);
    lv.setUint32(18, data.length, true);
    lv.setUint32(22, raw.length, true);
    lv.setUint16(26, name.length, true);
    lh.set(name, 30);
    locals.push(lh, data);

    const ch = new Uint8Array(46 + name.length);
    const cv = new DataView(ch.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(10, method, true);
    cv.setUint32(20, data.length, true);
    cv.setUint32(24, raw.length, true);
    cv.setUint16(28, name.length, true);
    // External attributes: unix mode in the high 16 bits.
    cv.setUint32(38, (f.mode ?? 0o644) << 16, true);
    cv.setUint32(42, offset, true);
    ch.set(name, 46);
    central.push(ch);

    offset += lh.length + data.length;
  }

  const cenSize = central.reduce((a, b) => a + b.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, cenSize, true);
  ev.setUint32(16, offset, true);

  const parts = [...locals, ...central, eocd];
  const out = new Uint8Array(parts.reduce((a, b) => a + b.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

Deno.test("archive: unzip reads stored and deflated entries", async () => {
  const zip = await zipOf([
    { name: "pkg/README.txt", body: "plain stored bytes" },
    {
      name: "pkg/llama-server.exe",
      body: "MZ".repeat(500),
      deflate: true,
      mode: 0o755,
    },
  ]);
  const entries = await unzip(zip);
  assertEquals(entries.length, 2);
  const dec = new TextDecoder();
  assertEquals(dec.decode(entries[0]?.bytes), "plain stored bytes");
  assertEquals(dec.decode(entries[1]?.bytes), "MZ".repeat(500));
  assertEquals(entries[1]?.mode, 0o755);
});

Deno.test("archive: unzip reads a symlink out of the unix mode bits", async () => {
  // S_IFLNK (0xA000) in the high bits of the external attributes.
  const zip = await zipOf([
    { name: "lib/libllama.so.0.0.1", body: "ELF" },
    { name: "lib/libllama.so.0", body: "libllama.so.0.0.1", mode: 0xa1ff },
  ]);
  const entries = await unzip(zip);
  assertEquals(entries[1]?.link, "libllama.so.0.0.1");
  assertEquals(entries[1]?.bytes.length, 0);
  assertEquals(entries[0]?.link, undefined);
});

Deno.test("archive: a zip with no end-of-central-directory fails loudly", async () => {
  await assertRejects(
    () => unzip(new TextEncoder().encode("not a zip at all")),
    Error,
    "end-of-central-directory",
  );
});

Deno.test("archive: gunzip round-trips through the runtime's own inflate", async () => {
  const body = new TextEncoder().encode("llama.cpp source tarball".repeat(50));
  const gz = await new Response(
    new Blob([body as BlobPart]).stream().pipeThrough(
      new CompressionStream("gzip"),
    ),
  ).bytes();
  assertEquals(
    new TextDecoder().decode(await gunzip(gz)),
    new TextDecoder().decode(body),
  );
});

// ── update detection ───────────────────────────────────────────────────────

function build(over: Partial<Build> = {}): Build {
  return {
    id: "release-b10000-cpu",
    ref: "b10000",
    origin: "release",
    backend: "cpu",
    dir: "/d",
    serverBin: "/d/llama-server",
    cliBin: "/d/llama-cli",
    createdAt: 1,
    sizeB: 1,
    ...over,
  };
}

const CHECKED = {
  latestTag: "b10144",
  masterSha: "a".repeat(40),
  checkedAt: 1,
};

Deno.test("update: a tagged build is compared by build number", () => {
  assertEquals(buildNumber("b10144"), 10144);
  assertEquals(buildNumber("master"), null);
  assertEquals(buildNumber("v1.2.3"), null);

  const behind = updateFor(build({ ref: "b10000" }), CHECKED);
  assertEquals(behind.available, true);
  assertEquals(behind.to, "b10144");
  assert(behind.reason.includes("144"));

  const current = updateFor(build({ ref: "b10144" }), CHECKED);
  assertEquals(current.available, false);
  assertEquals(current.reason, "This is the newest published release.");
});

Deno.test("update: a newer local build than upstream is not an update", () => {
  const ahead = updateFor(build({ ref: "b99999" }), CHECKED);
  assertEquals(ahead.available, false);
});

Deno.test("update: a master build is compared by commit, not by name", () => {
  const same = updateFor(
    build({ ref: "master", origin: "source", sourceSha: CHECKED.masterSha }),
    CHECKED,
  );
  assertEquals(same.available, false, "same commit is not an update");
  assert(same.from.includes("aaaaaaa"), "the short sha identifies the build");

  const moved = updateFor(
    build({ ref: "master", origin: "source", sourceSha: "b".repeat(40) }),
    CHECKED,
  );
  assertEquals(moved.available, true);
  assertEquals(moved.to, "master aaaaaaa");
});

Deno.test("update: a master build with no recorded commit offers an update", () => {
  // It cannot be shown to be current, and claiming it is would be a lie.
  const u = updateFor(build({ ref: "master", origin: "source" }), CHECKED);
  assertEquals(u.available, true);
  assert(u.reason.includes("did not record"));
});

Deno.test("update: nothing is offered before upstream has been read", () => {
  const never = { latestTag: "", masterSha: "", checkedAt: 0 };
  assertEquals(updateFor(build(), never).available, false);
  assertEquals(updateFor(null, CHECKED).available, false);
});

Deno.test("update: the target keeps the build on the track it was on", () => {
  assertEquals(updateTarget(build({ ref: "b10000" }), CHECKED), "b10144");
  assertEquals(updateTarget(build({ ref: "master" }), CHECKED), "master");
  assertEquals(updateTarget(null, CHECKED), "master");
});

// ── ollama store ───────────────────────────────────────────────────────────

const MANIFEST_ROOT = "/usr/share/ollama/.ollama/models/manifests";

Deno.test("ollama: a manifest path becomes the name a user would type", () => {
  assertEquals(
    nameFromManifestPath(
      `${MANIFEST_ROOT}/registry.ollama.ai/library/llama3.2/3b`,
    ),
    "llama3.2:3b",
    "the library namespace is hidden, exactly as ollama hides it",
  );
  assertEquals(
    nameFromManifestPath(
      `${MANIFEST_ROOT}/registry.ollama.ai/hf.co/user/repo/q4`,
    ),
    "hf.co/user/repo:q4",
  );
  assertEquals(nameFromManifestPath("/tmp/not/a/manifest"), null);
  assertEquals(nameFromManifestPath(`${MANIFEST_ROOT}/registry/library`), null);
});

Deno.test("ollama: a local model resolves to its blob", () => {
  const json = JSON.stringify({
    schemaVersion: 2,
    layers: [
      {
        mediaType: "application/vnd.ollama.image.license",
        digest: "sha256:aaa",
        size: 8,
      },
      {
        mediaType: "application/vnd.ollama.image.model",
        digest: "sha256:beef",
        size: 4661211808,
      },
      {
        mediaType: "application/vnd.ollama.image.template",
        digest: "sha256:ccc",
        size: 9,
      },
    ],
  });
  const m = resolveManifest(
    `${MANIFEST_ROOT}/registry.ollama.ai/library/llama3.2/3b`,
    json,
  );
  assertEquals(m?.name, "llama3.2:3b");
  assertEquals(m?.blob, "sha256-beef", "the colon becomes a dash on disk");
  assertEquals(m?.sizeB, 4661211808);
});

Deno.test("ollama: a cloud model has no weights and is not listed", () => {
  // Verified against a real store: cloud entries carry `"layers": null`.
  const cloud = JSON.stringify({ schemaVersion: 2, config: {}, layers: null });
  assertEquals(
    resolveManifest(
      `${MANIFEST_ROOT}/registry.ollama.ai/library/glm-5.2/cloud`,
      cloud,
    ),
    null,
  );
  const noModelLayer = JSON.stringify({
    layers: [{
      mediaType: "application/vnd.ollama.image.template",
      digest: "sha256:x",
    }],
  });
  assertEquals(
    resolveManifest(
      `${MANIFEST_ROOT}/registry.ollama.ai/library/x/y`,
      noModelLayer,
    ),
    null,
  );
  assertEquals(
    resolveManifest(
      `${MANIFEST_ROOT}/registry.ollama.ai/library/x/y`,
      "not json",
    ),
    null,
  );
});

Deno.test("ollama: a store is recognised by its two directories", () => {
  assert(isOllamaStore(["blobs", "manifests"]));
  assert(!isOllamaStore(["blobs"]));
  assert(!isOllamaStore(["model-a.gguf", "model-b.gguf"]));
});

Deno.test("thermal: the bar turns red at the throttle point, not before", () => {
  assertEquals(tempTone(0, CPU_TJMAX), "idle", "no sensor is not 'cold'");
  assertEquals(tempTone(45, CPU_TJMAX), "ok");
  assertEquals(tempTone(80, CPU_TJMAX), "warn");
  assertEquals(tempTone(94, CPU_TJMAX), "bad");
  // The GPU throttles lower, so the same reading means different things.
  assertEquals(tempTone(80, GPU_TJMAX), "bad");
  assertEquals(tempTone(60, GPU_TJMAX), "ok");
});

Deno.test("tune: a context from another model never leaks into this one", () => {
  // Found on real data: a 128-token context set for a toy model survived onto a
  // 262k-context 35B. The tuner now derives the context from THIS model rather
  // than inheriting one, so a stale value cannot survive at all.
  const big = meta({ nCtxTrain: 262144 });
  const machine = hw({ gpus: [gpu(48, 0)] });

  const fromStale = tune(big, machine, { ...defaults(), ctxSize: 128 }, "vram");
  assert(
    Number(fromStale.settings.ctxSize) >= MIN_CTX,
    `a 128-token leftover must not survive: got ${fromStale.settings.ctxSize}`,
  );
  const fresh = tune(big, machine, defaults(), "vram");
  assertEquals(
    fromStale.settings.ctxSize,
    fresh.settings.ctxSize,
    "what was in `base` makes no difference to the context chosen",
  );

  // Nothing is ever proposed past what the model was trained for.
  const small = tune(
    meta({ nCtxTrain: 4096 }),
    machine,
    { ...defaults(), ctxSize: 200000 },
    "vram",
  );
  assertEquals(small.settings.ctxSize, 4096);
  assertEquals(small.optimalCtx, 4096);

  // A context the USER pins is an instruction, and is honoured up to the
  // trained ceiling.
  const pinned = tune(big, machine, defaults(), "vram", 16384);
  assertEquals(pinned.settings.ctxSize, 16384);
  const overPinned = tune(
    meta({ nCtxTrain: 4096 }),
    machine,
    defaults(),
    "vram",
    999999,
  );
  assertEquals(overPinned.settings.ctxSize, 4096, "still capped at trained");
});

Deno.test("tune: a pin that does not fit fails LOUD, with the shortfall named", () => {
  // The report, verbatim: "even if I set context size to 1M, the projection
  // looks the same — so what memory is missing? And when I run the server it
  // sets back to 17k." The pin was a search ceiling, so the tuner silently
  // settled lower and no surface ever showed what 1M would cost. A pin is an
  // instruction: hold it, and when it cannot run, say exactly how far short
  // the machine falls — in a plan that shows the overflow.
  const big = meta({ nCtxTrain: 1_048_576, nCtxOrig: 65_536 });
  const machine = hw({ gpus: [gpu(24, 0)] });

  const t = tune(big, machine, defaults(), "vram", 1_048_576);
  assertEquals(t.possible, false, "1M does not fit a 24 GB card");
  assertEquals(
    t.settings.ctxSize,
    1_048_576,
    "the returned settings CARRY the pin, so the projection shows the overflow instead of a fitting plan at some other size",
  );
  assertStringIncludes(t.blocker, "pinned 1,048,576");
  assertStringIncludes(t.blocker, "more", "the missing bytes are named");
  assertStringIncludes(t.blocker, "Auto", "and the way out is offered");

  // A pin that fits is exact, not merely an upper bound.
  const fits = tune(big, machine, defaults(), "vram", 8192);
  assertEquals(fits.possible, true);
  assertEquals(fits.settings.ctxSize, 8192);
  assert(
    fits.reasons.some((r) => r.includes("pinned")),
    `the reason says it was pinned: ${JSON.stringify(fits.reasons)}`,
  );
});

Deno.test("tune: the measured fit is a search ceiling, never a pin", () => {
  // `cfg.fitCtx` records the largest context that actually generated. The
  // automatic path must not aim past it (that is the retry ladder's slow way
  // down), but it must also keep ADAPTING below it — a game taking VRAM means
  // less fits today than fitted yesterday, and holding the old number exact
  // would flip the placement to impossible instead of settling lower.
  const m = meta({ nCtxTrain: 262_144 });
  const roomy = hw({ gpus: [gpu(48, 0)] });
  const capped = tune(m, roomy, defaults(), "vram", undefined, 16_384);
  assertEquals(capped.possible, true);
  assert(
    Number(capped.settings.ctxSize) <= 16_384,
    `never past the measured ceiling: ${capped.settings.ctxSize}`,
  );
  assert(
    capped.reasons.some((r) => r.includes("actually started")),
    "the cap is explained",
  );

  // Tight machine: the same ceiling, but less fits now — settle lower, do not
  // refuse.
  const tight = tune(
    m,
    hw({ gpus: [gpu(8, 0)] }),
    defaults(),
    "vram",
    undefined,
    262_144,
  );
  assertEquals(tight.possible, true);
  assert(
    Number(tight.settings.ctxSize) < 262_144,
    "adapts below the ceiling instead of failing at it",
  );
});

// ── build readiness ────────────────────────────────────────────────────────

const FULL = new Set(["cmake", "compiler", "cuda", "vulkan", "spirv", "hip"]);
const BARE = new Set(["cmake", "compiler"]);

Deno.test("backend: the sched-cap bypass is one define, off by default", () => {
  // `GGML_SCHED_MAX_SPLIT_INPUTS` is #ifndef-guarded upstream — the define is
  // the supported way past the stock 30 that aborts extreme contexts with
  // experts in RAM (measured: 256k generates, 512k asserts). The raised value
  // covers the model's full 1M with margin.
  assertEquals(schedCapFlags(false), []);
  assertEquals(schedCapFlags(true), [
    `-DGGML_SCHED_MAX_SPLIT_INPUTS=${SCHED_SPLIT_CAP}`,
  ]);
  assert(
    SCHED_SPLIT_CAP >= 30 * 4,
    "the raise must cover 1M where ~30 carried ~256k",
  );
});

Deno.test("backend: a CPU source build needs only cmake and a compiler", () => {
  assertEquals(canCompile("cpu", BARE, "linux").ok, true);
  const none = canCompile("cpu", new Set(), "linux");
  assertEquals(none.ok, false);
  assertEquals(none.missing, ["cmake", "compiler"]);
  assert(none.reason.includes("CMake"));
});

Deno.test("backend: an accelerator build is refused up front, naming the tool", () => {
  // Without this the build fails minutes into cmake configure, several screens
  // down a log — which is where this project actually found it.
  const cuda = canCompile("cuda", BARE, "linux");
  assertEquals(cuda.ok, false);
  assertEquals(cuda.missing, ["cuda"]);
  assert(cuda.reason.includes("nvcc"), cuda.reason);

  const vulkan = canCompile("vulkan", BARE, "linux");
  assertEquals(vulkan.ok, false);
  assert(vulkan.reason.includes("glslc"));
  // SPIRV-Headers is the one that actually stops a machine that looks equipped:
  // measured, `find_package(SPIRV-Headers CONFIG REQUIRED)` at
  // ggml-vulkan/CMakeLists.txt:14 fails while glslc is present.
  assert(vulkan.reason.includes("SPIRV-Headers"), vulkan.reason);
  assertEquals(vulkan.missing, ["vulkan", "spirv"]);

  assertEquals(canCompile("cuda", FULL, "linux").ok, true);
  assertEquals(canCompile("vulkan", FULL, "linux").ok, true);
});

Deno.test("backend: Metal is only offered on Apple hardware", () => {
  assertEquals(canCompile("metal", FULL, "linux").ok, false);
  assertEquals(canCompile("metal", BARE, "darwin").ok, true);
  assertEquals(compilableBackends(BARE, "linux"), ["cpu"]);
  assertEquals(compilableBackends(FULL, "linux").sort(), [
    "cpu",
    "cuda",
    "hip",
    "vulkan",
  ]);
  assertEquals(compilableBackends(BARE, "darwin").sort(), ["cpu", "metal"]);
});

// ── fixing prerequisites ───────────────────────────────────────────────────

Deno.test("fix: CMake is a download the app does itself", () => {
  const plan = fixPlan("cmake", "linux", null);
  assertEquals(plan.kind, "download");
  assert(isFixable(plan));
  assert(describeFix(plan).includes("CMake"));
});

Deno.test("fix: a package install names the exact command", () => {
  const apt = fixPlan("compiler", "linux", "apt");
  assertEquals(apt.kind, "package");
  if (apt.kind !== "package") throw new Error("unreachable");
  assertEquals(apt.command, ["apt-get", "install", "-y", "build-essential"]);
  assertEquals(apt.needsRoot, true);
  assert(describeFix(apt).includes("apt-get install -y build-essential"));

  const pac = fixPlan("vulkan", "linux", "pacman");
  if (pac.kind !== "package") throw new Error("unreachable");
  assertEquals(pac.command[0], "pacman");
  assert(pac.packages.includes("shaderc"));

  // Homebrew refuses to run as root, so it must not be elevated.
  const brew = fixPlan("git", "darwin", "brew");
  if (brew.kind !== "package") throw new Error("unreachable");
  assertEquals(brew.needsRoot, false);
});

Deno.test("fix: nothing is invented for an unknown system", () => {
  assertEquals(fixPlan("compiler", "linux", null).kind, "manual");
  assertEquals(fixPlan("cuda", "darwin", "brew").kind, "manual");
  assertEquals(fixPlan("compiler", "darwin", "brew").kind, "manual");
  assert(!isFixable(fixPlan("nvidia", "linux", "apt")), "drivers stay manual");
  assert(
    !isFixable(fixPlan("deno", "linux", "apt")),
    "deno is already running",
  );
});

Deno.test("fix: every manual plan says where to read the instructions", () => {
  // "We will not do this for you" is only acceptable with a link.
  const nvidia = fixPlan("nvidia", "linux", "apt");
  if (nvidia.kind !== "manual") throw new Error("unreachable");
  assert(nvidia.docsUrl?.startsWith("https://"), "no link for the driver");

  const rocmElsewhere = rocmPlan("linux", {
    id: "fedora",
    version: "42",
    ubuntuCodename: "",
  });
  if (rocmElsewhere.kind !== "manual") throw new Error("unreachable");
  assert(rocmElsewhere.docsUrl?.includes("rocm.docs.amd.com"));
  assert(rocmElsewhere.reason.includes("fedora 42"), "names what it saw");
});

Deno.test("fix: ROCm on Ubuntu 24.04 is a guided install, not a shrug", () => {
  // AMD publishes exact steps for this combination, so the app runs them —
  // every command visible first, because they add a repository and a driver.
  const plan = rocmPlan("linux", {
    id: "ubuntu",
    version: "24.04",
    ubuntuCodename: "noble",
  });
  assertEquals(plan.kind, "script");
  if (plan.kind !== "script") throw new Error("unreachable");
  assert(plan.steps.length >= 6, `only ${plan.steps.length} steps`);
  assertEquals(plan.needsRoot, true);
  assertEquals(plan.rebootAfter, true, "the driver needs a restart");
  assert(plan.docsUrl.includes("rocm.docs.amd.com"));

  const all = plan.steps.map((st) => st.sh).join("\n");
  assert(all.includes("/etc/apt/keyrings/amdrocm.gpg"), "adds AMD's key");
  assert(all.includes("repo.amd.com/rocm"), "adds AMD's repository");
  assert(all.includes("amdgpu-install"), "installs the driver");
  assert(all.includes("usermod -a -G render,video"), "grants device access");
  assert(!all.includes("sudo "), "elevation is applied once, not baked in");

  const preview = scriptPreview(plan);
  assertEquals(preview.length, plan.steps.length);
  assert(preview[0]?.startsWith("# "), "each step is labelled");
});

Deno.test("fix: ROCm is only scripted where AMD documents those exact steps", () => {
  for (
    const d of [
      { id: "ubuntu", version: "22.04", ubuntuCodename: "jammy" },
      { id: "debian", version: "12", ubuntuCodename: "" },
      { id: "arch", version: "", ubuntuCodename: "" },
    ]
  ) {
    assertEquals(rocmPlan("linux", d).kind, "manual", `${d.id} ${d.version}`);
  }
  assertEquals(rocmPlan("linux", null).kind, "manual", "unknown distro");
  assertEquals(rocmPlan("darwin", null).kind, "manual");
});

Deno.test("fixplan: the ROCm group step names the user who pressed the button, under pkexec AND sudo", async () => {
  // `${SUDO_USER:-$PKEXEC_UID_NAME:-$USER}` read like a fallback chain and was
  // not one: bash parses the nested default as the literal word
  // `$PKEXEC_UID_NAME:-$USER`, and pkexec sets no such variable (it sets
  // PKEXEC_UID, a number). Under pkexec — the app's preferred path — the step
  // ran `usermod -a -G render,video :-root`, and failed AFTER the driver and
  // the repo were installed. So the expansion is checked by asking bash.
  const plan = rocmPlan("linux", {
    id: "ubuntu",
    version: "24.04",
    ubuntuCodename: "noble",
  });
  assert(plan.kind === "script");
  const step = plan.steps.find((s) => s.sh.startsWith("usermod"));
  assert(step, "the group step exists");
  const me = new TextDecoder().decode(
    (await new Deno.Command("id", { args: ["-un"] }).output()).stdout,
  ).trim();
  const uid = String(Deno.uid());
  const run = async (env: Record<string, string>) =>
    new TextDecoder().decode(
      (await new Deno.Command("bash", {
        args: ["-c", step.sh.replace(/^usermod/, "echo usermod")],
        env,
        clearEnv: true,
      }).output()).stdout,
    ).trim();
  const want = `usermod -a -G render,video ${me}`;
  assertEquals(await run({ PKEXEC_UID: uid, USER: "root" }), want, "pkexec");
  assertEquals(await run({ SUDO_UID: uid, USER: "root" }), want, "sudo");
  assertEquals(await run({ USER: "root" }), want, "neither: whoever runs it");
});

Deno.test("fix: elevation prefers the desktop's auth agent, and can refuse", () => {
  const cmd = ["apt-get", "install", "-y", "git"];
  assertEquals(elevate(cmd, { isRoot: true, pkexec: true, sudo: true }), cmd);
  assertEquals(elevate(cmd, { isRoot: false, pkexec: true, sudo: true }), [
    "pkexec",
    ...cmd,
  ]);
  assertEquals(elevate(cmd, { isRoot: false, pkexec: false, sudo: true }), [
    "sudo",
    "-n",
    ...cmd,
  ]);
  assertEquals(
    elevate(cmd, { isRoot: false, pkexec: false, sudo: false }),
    null,
    "no silent privileged run when nothing can elevate",
  );
});

// ── surviving GitHub's rate limit ──────────────────────────────────────────

Deno.test("github: a quota 403 is told apart from a real one", () => {
  const quota = new Headers({
    "x-ratelimit-remaining": "0",
    "x-ratelimit-limit": "60",
  });
  assert(isRateLimited(403, quota));
  assert(isRateLimited(429, new Headers({ "retry-after": "60" })));
  // A 403 with quota left is a permissions problem and must not be retried as
  // if it were throttling.
  assert(!isRateLimited(403, new Headers({ "x-ratelimit-remaining": "42" })));
  assert(!isRateLimited(404, quota));
});

Deno.test("github: the rate-limit message says when and how to fix it", () => {
  const now = 1_700_000_000_000;
  const h = new Headers({
    "x-ratelimit-remaining": "0",
    "x-ratelimit-limit": "60",
    "x-ratelimit-reset": String(Math.floor(now / 1000) + 600),
  });
  const msg = rateLimitMessage(h, now);
  assertStringIncludes(msg, "60 requests/hour");
  assertStringIncludes(msg, "10 minutes");
  assertStringIncludes(msg, "GITHUB_TOKEN");
});

Deno.test("github: the release page yields the tag and the asset list", () => {
  // The fallback that keeps installs working at 0 API budget. Shape copied from
  // a real response.
  assertEquals(
    tagFromReleaseUrl(
      "https://github.com/ggml-org/llama.cpp/releases/tag/b10144",
    ),
    "b10144",
  );
  assertEquals(tagFromReleaseUrl("https://github.com/x/y/releases"), null);

  const html = `<div>
    <a href="/ggml-org/llama.cpp/releases/download/b10144/llama-b10144-bin-ubuntu-x64.tar.gz">x</a>
    <a href="/ggml-org/llama.cpp/releases/download/b10144/llama-b10144-bin-ubuntu-vulkan-x64.tar.gz">y</a>
    <a href="/ggml-org/llama.cpp/releases/download/b10144/llama-b10144-bin-ubuntu-x64.tar.gz">dup</a>
    <a href="/other/repo/releases/download/b1/nope.zip">wrong repo</a>
  </div>`;
  const names = assetsFromHtml(html, "ggml-org/llama.cpp");
  assertEquals(names.length, 2, "deduplicated, and scoped to the repo");
  assert(names.includes("llama-b10144-bin-ubuntu-vulkan-x64.tar.gz"));

  assertEquals(
    assetUrl("ggml-org/llama.cpp", "b10144", "a.tar.gz"),
    "https://github.com/ggml-org/llama.cpp/releases/download/b10144/a.tar.gz",
  );
});

// ── CUDA architecture selection ────────────────────────────────────────────

Deno.test("cuda: a version string becomes a number, safely", () => {
  assertEquals(
    parseCudaVersion("Cuda compilation tools, release 12.0, V12.0.140"),
    12,
  );
  assertEquals(parseCudaVersion("release 12.8, V12.8.61"), 12.8);
  assertEquals(parseCudaVersion("nvcc: not found"), 0);
  assertEquals(parseCudaVersion(""), 0);
});

Deno.test("cuda: each release knows how far it can target", () => {
  assertEquals(maxArchFor(11.8), 90);
  assertEquals(maxArchFor(12.0), 90);
  assertEquals(maxArchFor(12.8), 120, "Blackwell support lands in 12.8");
  assertEquals(maxArchFor(0), 0, "no version, no answer");
  assertEquals(cudaVersionForCap(12.0), 12.8);
  // sm_89 is Ada: 11.1 tops out at sm_86 (Ampere), so 11.8 is the first that
  // can target it.
  assertEquals(cudaVersionForCap(8.9), 11.8);
  assertEquals(cudaVersionForCap(8.6), 11.1);
});

Deno.test("cuda: a toolkit newer than the GPU builds native code", () => {
  const p = cudaPlan("release 12.8, V12.8.61", [8.9]);
  assertEquals(p.mode, "native");
  assertEquals(p.architectures, "89");
  assertEquals(cudaCmakeFlags(p), ["-DCMAKE_CUDA_ARCHITECTURES=89"]);
  // Two different cards: both named, deduplicated and ordered.
  assertEquals(
    cudaPlan("release 12.8", [8.9, 12.0, 8.9]).architectures,
    "89;120",
  );
});

Deno.test("cuda: an old toolkit with a new GPU builds PTX instead of failing", () => {
  // THE regression this file exists for. Measured: CUDA 12.0 + RTX PRO 4000
  // Blackwell (sm_120) → `nvcc fatal: Unsupported gpu architecture
  // 'compute_120a'` four minutes into the compile. With the cap below, the same
  // machine builds and runs on the GPU via the driver's PTX JIT.
  const p = cudaPlan("Cuda compilation tools, release 12.0, V12.0.140", [12.0]);
  assertEquals(p.mode, "ptx");
  assertEquals(
    p.architectures,
    "90-virtual",
    "PTX only, so the driver JITs it",
  );
  assertEquals(cudaCmakeFlags(p), ["-DCMAKE_CUDA_ARCHITECTURES=90-virtual"]);
  assertStringIncludes(p.reason, "sm_120");
  assertStringIncludes(p.reason, "first load");
  assertStringIncludes(p.remedy, "12.8");
});

Deno.test("cuda: no GPU and no toolkit are both said plainly", () => {
  const noGpu = cudaPlan("release 12.8", []);
  assertEquals(noGpu.mode, "impossible");
  assertStringIncludes(noGpu.reason, "No NVIDIA GPU");
  assertStringIncludes(noGpu.remedy, "Vulkan");

  const noNvcc = cudaPlan("nvcc: command not found", [12.0]);
  assertEquals(noNvcc.mode, "impossible");
  assertStringIncludes(noNvcc.remedy, "CUDA toolkit");
  assertEquals(cudaCmakeFlags(noNvcc), [], "nothing to pass cmake");
});

Deno.test("backend: Vulkan needs glslc AND SPIRV-Headers, not glslangValidator", () => {
  // cmake prints "missing components: glslangValidator" on the same run that
  // fails, which is informational and misleading — llama.cpp asks for glslc
  // only (`find_package(Vulkan COMPONENTS glslc REQUIRED)`). The real stopper is
  // the SPIRV-Headers package on the next line.
  const r = canCompile("vulkan", new Set(["cmake", "compiler"]), "linux");
  assertEquals(r.ok, false);
  assertStringIncludes(r.reason, "glslc");
  assertStringIncludes(r.reason, "SPIRV-Headers");
  assert(!r.reason.includes("glslangValidator"), "that was a red herring");

  // With glslc but no headers — the exact state of the machine this was found
  // on — it must still refuse.
  const halfway = canCompile(
    "vulkan",
    new Set(["cmake", "compiler", "vulkan"]),
    "linux",
  );
  assertEquals(halfway.ok, false);
  assertEquals(halfway.missing, ["spirv"]);
});

// ── explaining what cannot be downloaded ───────────────────────────────────

Deno.test("assets: an impossible combination is explained, with the route that works", () => {
  // The kata's rule: no bare "not found". Each of these is a real upstream fact
  // verified against release b10144.
  const cudaLinux = noAssetExplanation("cuda", "linux", "x86_64", [
    "cpu",
    "vulkan",
    "hip",
  ]);
  assertStringIncludes(cudaLinux.reason, "Windows only");
  assert(cudaLinux.steps.length >= 2);
  assertStringIncludes(cudaLinux.steps.join(" "), "Build from source");
  assertStringIncludes(cudaLinux.steps.join(" "), "Vulkan");

  const metalLinux = noAssetExplanation("metal", "linux", "x86_64", ["cpu"]);
  assertStringIncludes(metalLinux.reason, "only on macOS");
  assertStringIncludes(metalLinux.steps.join(" "), "Vulkan");

  // The generic case still names what IS available rather than shrugging.
  const other = noAssetExplanation("vulkan", "windows", "aarch64", ["cpu"]);
  assertStringIncludes(other.reason, "windows/aarch64");
  assertStringIncludes(other.steps.join(" "), "cpu");
});

Deno.test("assets: every backend has a prebuilt route somewhere", () => {
  // Guards the claim the Builds kata makes: each backend is installable from a
  // release on at least one platform this app supports.
  const real = REAL_B10144;
  const found: Record<string, string> = {};
  for (
    const [backend, platform, arch] of [
      ["cpu", "linux", "x86_64"],
      ["vulkan", "linux", "x86_64"],
      ["hip", "linux", "x86_64"],
      ["cuda", "windows", "x86_64"],
      ["metal", "darwin", "aarch64"],
    ] as const
  ) {
    const a = pickAsset(real, platform, arch, backend);
    assert(a, `no prebuilt ${backend} for ${platform}/${arch}`);
    found[backend] = a.name;
  }
  assertStringIncludes(found.hip as string, "rocm");
  assertStringIncludes(found.cuda as string, "cuda");
  assertStringIncludes(found.metal as string, "macos");
});

// ── never a raw error ──────────────────────────────────────────────────────

const LINUX = { platform: "linux", arch: "x86_64" } as const;

Deno.test("diagnose: the exact failure a user reported becomes advice", () => {
  // Was: "Error: b10145 has no prebuilt cuda binary for linux/x86_64.
  //       Available: <27 filenames>". A list of filenames is not an answer.
  const d = diagnoseNoAsset(
    {
      origin: "release",
      backend: "cuda",
      ...LINUX,
      availableBackends: ["cpu", "vulkan", "hip"],
      found: new Set(["cuda", "cmake", "compiler"]),
    },
    27,
  );
  assertStringIncludes(d.reason, "Windows only");
  assert(!d.reason.includes(".tar.gz"), "no filename dumps");
  // It knows nvcc is already installed, so it says "switch route", not "install".
  assertStringIncludes(d.steps[0]?.text ?? "", "already have");
  assertEquals(d.steps[0]?.action, { kind: "switch-origin", to: "source" });
  // And every step the app can perform is a button.
  assert(d.steps.every((st) => st.action), "each step is actionable here");
});

Deno.test("diagnose: without nvcc the advice changes to installing it", () => {
  const d = diagnoseNoAsset(
    { origin: "release", backend: "cuda", ...LINUX, found: new Set() },
    27,
  );
  assert(d.steps.some((st) => st.action?.kind === "fix-prereq"));
  assertStringIncludes(d.steps.map((st) => st.text).join(" "), "nvcc");
});

Deno.test("diagnose: a half-published release is not confused with an absent one", () => {
  // llama.cpp tags a release, then CI uploads ~25 assets over some minutes.
  const d = diagnoseNoAsset(
    { origin: "release", backend: "vulkan", ...LINUX },
    2,
  );
  assertStringIncludes(d.reason, "still on its way");
  assertStringIncludes(d.steps.map((st) => st.text).join(" "), "Wait");
});

Deno.test("diagnose: known build failures map to their real cause", () => {
  const c = { origin: "source" as const, backend: "cuda" as const, ...LINUX };
  const cuda = diagnoseFailure(
    "nvcc fatal : Unsupported gpu architecture 'compute_120a'",
    c,
  );
  assertStringIncludes(cuda.reason, "older than your GPU");

  const spirv = diagnoseFailure("error: 'spv' has not been declared", {
    ...c,
    backend: "vulkan",
  });
  assertStringIncludes(spirv.reason, "SPIRV-Headers");
  assertEquals(spirv.steps[0]?.action, { kind: "fix-prereq", id: "spirv" });

  const oom = diagnoseFailure(
    "c++: fatal error: Killed signal terminated program",
    c,
  );
  assertStringIncludes(oom.reason, "out of memory");
  assertStringIncludes(oom.steps.map((s) => s.text).join(" "), "job count");

  const disk = diagnoseFailure("fatal error: No space left on device", c);
  assertStringIncludes(disk.reason, "disk filled up");
});

Deno.test("diagnose: an unknown failure still gets a route out", () => {
  const d = diagnoseFailure("something nobody has seen before", {
    origin: "source",
    backend: "cpu",
    ...LINUX,
  });
  assert(d.reason.length > 20, "still a sentence, not the raw text");
  assert(d.steps.length >= 2, "still offers the other route and a re-check");
  assert(d.steps.some((s) => s.action?.kind === "switch-origin"));
});

// ── readiness for the exact selection ──────────────────────────────────────

Deno.test("readiness: green prerequisites do not mean the selection will work", () => {
  // The user's complaint: everything ticked, then the build failed. A fully
  // equipped machine still cannot download a Linux CUDA release.
  const equipped = new Set(["cmake", "compiler", "cuda", "vulkan", "spirv"]);
  const r = targetReadiness("release", "cuda", {
    ...LINUX,
    found: equipped,
    availableBackends: ["cpu", "vulkan", "hip"],
    assetCount: 27,
  });
  assertEquals(r.ok, false, "must be refused BEFORE the button is pressed");
  assertStringIncludes(r.diagnosis?.reason ?? "", "Windows only");

  // The same machine, same backend, other route: fine.
  assertEquals(
    targetReadiness("source", "cuda", {
      ...LINUX,
      found: equipped,
      availableBackends: null,
      assetCount: 0,
    }).ok,
    true,
  );
});

Deno.test("readiness: unknown is reported as pending, never as ready", () => {
  const r = targetReadiness("release", "vulkan", {
    ...LINUX,
    found: new Set(),
    availableBackends: null, // list not fetched yet
    assetCount: 0,
  });
  assertEquals(r.ok, false);
  assertEquals(r.pending, true, "pending, not a false 'ready'");
  assertEquals(r.diagnosis, null);
});

Deno.test("readiness: a source build missing a tool lists it with a Fix action", () => {
  const r = targetReadiness("source", "vulkan", {
    ...LINUX,
    found: new Set(["cmake", "compiler"]),
    availableBackends: null,
    assetCount: 0,
  });
  assertEquals(r.ok, false);
  const actions = (r.diagnosis?.steps ?? []).map((s) => s.action?.kind);
  assert(actions.includes("fix-prereq"), "each missing tool is fixable");
  assert(actions.includes("switch-origin"), "and the other route is offered");
});

// ── ROCm: runtime installed is not the same as buildable ───────────────────

Deno.test("fix: a runtime-only ROCm gets a one-package fix, not a reinstall", () => {
  // Measured on a real machine: hipcc, hipconfig, amdclang++ and rocminfo all
  // present, /opt/rocm populated — and cmake still stops with "does not contain
  // the HIP runtime CMake package". Telling that user to install ROCm from
  // AMD's repository again would be wrong and slow; one -dev package is the fix.
  const noble = { id: "ubuntu", version: "24.04", ubuntuCodename: "noble" };
  const withHipcc = fixPlan("hip", "linux", "apt", noble, true);
  assertEquals(withHipcc.kind, "package");
  if (withHipcc.kind !== "package") throw new Error("unreachable");
  assertEquals(withHipcc.command, [
    "apt-get",
    "install",
    "-y",
    "amdrocm-core-dev",
  ]);

  // With no hipcc at all, the full guided install is still the right answer.
  const without = fixPlan("hip", "linux", "apt", noble, false);
  assertEquals(without.kind, "script");
});

Deno.test("diagnose: cmake's HIP message is recognised", () => {
  const d = diagnoseFailure(
    "does not contain the HIP runtime CMake package, expected at one of:\n" +
      "/opt/rocm/core-7.14/lib/cmake/hip-lang/hip-lang-config.cmake",
    { origin: "source", backend: "hip", platform: "linux", arch: "x86_64" },
  );
  assertStringIncludes(d.reason, "development package");
  assertStringIncludes(d.reason, "not enough");
  assertEquals(d.steps[0]?.action, { kind: "fix-prereq", id: "hip" });
  // The prebuilt ROCm release bundles everything — always worth offering.
  assert(d.steps.some((st) => st.action?.kind === "switch-origin"));
});

// ── why llama-server stopped ───────────────────────────────────────────────

/** Verbatim tail of a real failing run on this machine. */
const OOM_LOG = [
  "0.00.990.403 W load: control-looking token: 212 '</s>' was not control-type",
  "0.02.662.706 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2406.98 MiB on device 0: cudaMalloc failed: out of memory",
  "0.02.662.710 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 2523902336",
  "0.03.191.444 E llama_model_load: error loading model: unable to allocate CUDA0 buffer",
  "0.03.209.383 E srv  llama_server: exiting due to model loading error",
];

Deno.test("serverlog: only the error lines are extracted", () => {
  const errs = extractErrors(OOM_LOG);
  assertEquals(errs.length, 4, "the W line is not an error");
  assertStringIncludes(errs[0] as string, "cudaMalloc failed");
});

Deno.test("serverlog: 'exited with code 1' becomes the actual reason", () => {
  // The user's report: the app showed the code, while the cause sat in a log
  // it had already captured.
  const d = diagnoseServerExit(1, OOM_LOG);
  assertStringIncludes(d.reason, "GPU ran out of memory");
  assert(!d.reason.includes("code 1"), "the code is not the explanation");
  assertStringIncludes(
    d.steps.map((s2) => s2.text).join(" "),
    "another llama-server is still running",
  );
});

Deno.test("serverlog: a per-card allocation failure names the card and the size", () => {
  // Verbatim from the run that exposed the split bug. "The GPU ran out of
  // memory" is true and useless here: 42 GB of VRAM was free across two cards
  // and 34 GB of it was asked of one. The number and the device index are the
  // whole diagnosis.
  const d = diagnoseServerExit(1, [
    "0.00.293.391 I srv    load_model: loading model '/models/big-00001-of-00004.gguf'",
    "1.51.022.152 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 34020.32 MiB on device 1: cudaMalloc failed: out of memory",
    "1.51.022.161 E alloc_tensor_range: failed to allocate CUDA1 buffer of size 35672889856",
    "1.51.856.158 E llama_model_load: error loading model: unable to allocate CUDA1 buffer",
  ]);
  assertStringIncludes(d.reason, "Card 1");
  assertStringIncludes(d.reason, "33.2 GB");
  assertStringIncludes(d.reason, "divided badly");
  assertStringIncludes(
    d.steps.map((s2) => s2.text).join(" "),
    "Tensor split",
  );
});

Deno.test("serverlog: a generation-time OOM is memory advice, not a driver mismatch", () => {
  // Captured live: the CUDA pool OOM at the first real batch matched the
  // generic /CUDA error/ signature and told the user "the build and the
  // installed driver do not match — use the Vulkan build". They believed it and
  // switched backends, when a smaller context was the actual fix.
  const d = diagnoseServerExit(134, [
    "/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
    "2.17.177.475 E CUDA error: out of memory",
    "2.17.177.483 E   current device: 0, in function ggml_cuda_kernel_can_use_pdl at /src/ggml-cuda/common.cuh:1622",
    "2.17.177.484 E   cudaFuncGetAttributes(&attr, kernel)",
  ]);
  assertStringIncludes(d.reason, "during generation");
  assert(!d.reason.includes("driver"), "an OOM is not a driver mismatch");

  // A genuine device rejection still gets the driver answer.
  const drv = diagnoseServerExit(1, [
    "0.00.100.000 E CUDA error: forward compatibility was attempted on non supported HW",
  ]);
  assertStringIncludes(drv.reason, "driver");
});

Deno.test("fitladder: a feature the build lacks is dropped, not retried smaller", () => {
  // Verbatim from GLM-5.3-Flash on the first build that knew the architecture.
  // The header declares one NextN block, so the tuner turned MTP on — right by
  // every fact a header can carry — and the loader aborted because the graph
  // was not written yet. Both memory rungs are useless here: the ladder would
  // have reloaded 157 GB six times, halving a context that was never the
  // problem, and hit the same assert at every size.
  const lines = [
    "1.03.849.154 I common_speculative_init_result: creating MTP draft context against the target model",
    '/src/models/glm5next.cpp:712: GGML_ASSERT(params.gtype != LLM_GRAPH_TYPE_DECODER_MTP && "glm5next NextN graph not implemented yet") failed',
    "[llama.master] llama-server exited with code 134",
  ];
  const found = unsupportedFeature(lines);
  assertEquals(found?.setting, "specType");
  assertEquals(found?.flag, "--spec-type");

  const d = fitDecision({ lines, ctx: 124416, tries: 0, auto: true });
  assertEquals(d.kind, "drop");
  if (d.kind !== "drop") throw new Error("unreachable");
  assertStringIncludes(d.note, "--spec-type");

  // And the argv it produces is the one that ran, minus that flag and its value.
  const argv = [
    "/b/llama-server",
    "-m",
    "/m/glm.gguf",
    "--spec-type",
    "draft-mtp",
    "-c",
    "124416",
  ];
  assertEquals(withoutFlag(argv, "--spec-type", 1), [
    "/b/llama-server",
    "-m",
    "/m/glm.gguf",
    "-c",
    "124416",
  ]);

  // An abort with no setting behind it is a crash, not a rung: nothing is
  // stripped and the ordinary path explains it.
  assertEquals(
    unsupportedFeature([
      "/src/ggml.c:1: GGML_ASSERT(ne0 == ne1) failed",
    ]),
    null,
  );
  // And the words alone, with no abort, must never strip a flag from a run that
  // merely mentioned the feature.
  assertEquals(
    unsupportedFeature(["I srv: LLM_GRAPH_TYPE_DECODER_MTP enabled"]),
    null,
  );
});

Deno.test("fitladder: a refused feature is taken back out of the tuner's answer", () => {
  const base = { specType: "draft-mtp", ctxSize: 4096 } as unknown as Settings;
  const veto = vetoUnsupported(base, ["specType"]);
  assertEquals(veto.settings.specType, "");
  assertEquals(veto.settings.ctxSize, 4096, "nothing else moves");
  assertStringIncludes(veto.reasons[0] ?? "", "not implemented yet");
  // Untouched when nothing has been refused — the tuner's answer stands, and
  // the object is returned as-is rather than rebuilt.
  assertEquals(vetoUnsupported(base, []).settings, base);
  assertEquals(vetoUnsupported(base, []).reasons.length, 0);
});

Deno.test("serverlog: an unknown architecture blames the build, not the file", () => {
  // Verbatim from GLM-5.3-Flash (unsloth IQ4_XS, 157 GB across 5 shards) on a
  // master build made the same morning: the load failed in 58 ms and the app
  // said "a truncated download or an unsupported quantisation are the usual
  // ones" and offered a re-scan. Both wrong. The file was intact — this app's
  // own reader had already parsed 1,412 tensors out of it — and re-downloading
  // 157 GB would have changed nothing, because `glm5next` had three OPEN pull
  // requests upstream and nothing merged.
  const lines = [
    "0.00.518.604 I srv    load_model: loading model '/m/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf'",
    "0.00.576.055 E llama_model_load: error loading model: unknown model architecture: 'glm5next'",
    "0.00.576.067 E llama_model_load_from_file_impl: failed to load model",
    "0.00.576.081 E srv    load_model: failed to load model, '/m/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf'",
  ];
  const d = diagnoseServerExit(1, lines);
  assertStringIncludes(d.reason, "glm5next");
  assertStringIncludes(d.reason, "does not know the architecture");
  assertStringIncludes(d.reason, "The file is fine");
  assert(
    !/truncated/i.test(d.reason),
    `the download is not the suspect: ${d.reason}`,
  );
  assert(
    !d.steps.some((s2) => /Re-scan/i.test(s2.text)),
    "re-scanning the model list cannot teach a build a new architecture",
  );
  assert(
    d.steps.some((s2) => s2.action?.kind === "open-tab"),
    "the first move is a newer build",
  );
  assertStringIncludes(
    d.steps.map((s2) => s2.action?.kind === "open-url" ? s2.action.url : "")
      .join(""),
    "glm5next",
    "and the second names where support would come from",
  );

  // A genuinely broken file still gets the ordinary answer.
  const plain = diagnoseServerExit(1, [
    "0.00.100.000 E error loading model: tensor 'blk.0.attn_q.weight' data is not within the file bounds",
  ]);
  assertStringIncludes(plain.reason, "could not load this model file");
});

Deno.test("serverlog: a scheduler-limit assert is named, not blamed on memory", () => {
  // Verbatim from a 524,288-token pin of DeepSeek-V4 on 2×24 GB: the assert
  // line carries no timestamp and no severity column, so extractErrors missed
  // it and the diagnosis fell through to "died on SIGABRT — crash inside
  // llama.cpp". The user's next question was "so not enough RAM?" — wrong
  // trail, and the answer was sitting in the captured log.
  const lines = [
    "0.00.440.832 I srv    load_model: loading model '/m/big.gguf'",
    "/home/dev/src/ggml/src/ggml-backend.cpp:1367: GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed",
    "[llama.master] llama-server exited with code 134",
  ];
  const errs = extractErrors(lines);
  assert(
    errs.some((e) => e.includes("GGML_SCHED_MAX_SPLIT_INPUTS")),
    `the bare assert line is extracted: ${JSON.stringify(errs)}`,
  );
  const d = diagnoseServerExit(134, lines);
  assertStringIncludes(d.reason, "not a memory shortage");
  assertStringIncludes(d.reason, "hard limit");
  assert(
    d.steps.some((s2) => s2.text.includes("Lower the pinned context")),
    "the lever is named",
  );
});

Deno.test("serverlog: signals are named, and a clean stop is not a crash", () => {
  assertEquals(signalOf(143), "SIGTERM (asked to stop)");
  assertEquals(signalOf(139), "SIGSEGV (segmentation fault)");
  assertEquals(signalOf(1), null, "a plain exit code is not a signal");

  assertStringIncludes(
    diagnoseServerExit(143, []).reason,
    "normal shutdown, not a crash",
  );
  assertEquals(diagnoseServerExit(143, []).steps.length, 0);

  // 137 is the OOM killer, and it says so rather than "signal 9".
  assertStringIncludes(diagnoseServerExit(137, []).reason, "OOM killer");

  const segv = diagnoseServerExit(139, []);
  assertStringIncludes(segv.reason, "crash inside llama.cpp");
  assert(segv.steps.some((s2) => s2.action?.kind === "open-url"));
});

Deno.test("serverlog: a port clash and a bad flag are told apart", () => {
  const port = diagnoseServerExit(1, [
    "0.00.1 E bind: Address already in use",
  ]);
  assertStringIncludes(port.reason, "port is already taken");

  const flag = diagnoseServerExit(1, [
    '0.00.1 E error while handling argument "--n-cpu-moe": unknown argument',
  ]);
  assertStringIncludes(flag.reason, "rejected one of the flags");
});

Deno.test("serverlog: an unhelpful exit still quotes what it said", () => {
  const quoted = diagnoseServerExit(1, ["0.00.1 E something specific broke"]);
  assertStringIncludes(quoted.reason, "something specific broke");

  // And with nothing at all, it still offers a way to narrow it down.
  const silent = diagnoseServerExit(1, []);
  assertStringIncludes(silent.reason, "printed nothing useful");
  assert(silent.steps.length > 0);
});

// ── backend policy: which devices, which backend ───────────────────────────

Deno.test("backend: a build only sees the GPUs its backend can address", () => {
  const gpus = [
    { vendor: "nvidia", name: "RTX PRO 4000" },
    { vendor: "nvidia", name: "RTX PRO 4000 #2" },
    { vendor: "amd", name: "iGPU" },
  ];
  // A CPU build cannot put a byte on a GPU — planning against 49 GB of VRAM
  // it can never touch is a picture of something that will not happen.
  assertEquals(usableGpus("cpu", gpus), []);
  assertEquals(usableGpus("cuda", gpus).map((g) => g.vendor), [
    "nvidia",
    "nvidia",
  ]);
  assertEquals(usableGpus("hip", gpus).map((g) => g.vendor), ["amd"]);
  // Vulkan is vendor-neutral, and "no build chosen yet" shows the machine.
  assertEquals(usableGpus("vulkan", gpus).length, 3);
  assertEquals(usableGpus(undefined, gpus).length, 3);
});

Deno.test("backend: the suggested backend follows the hardware", () => {
  const set = (...v: string[]) => new Set(v);
  assertEquals(preferredBackends(set("nvidia"), "linux")[0], "cuda");
  assertEquals(preferredBackends(set("amd"), "linux")[0], "hip");
  // ROCm is Linux-only: an AMD card on Windows is reached through Vulkan.
  assertEquals(preferredBackends(set("amd"), "windows")[0], "vulkan");
  assertEquals(preferredBackends(set("intel"), "linux")[0], "vulkan");
  // macOS is Metal regardless of what is plugged in.
  assertEquals(preferredBackends(set("nvidia"), "darwin")[0], "metal");
  assertEquals(preferredBackends(set(), "linux"), ["cpu"]);
  // CPU is always the last resort, and always present.
  for (const os of ["linux", "windows", "darwin"]) {
    for (const v of [set("nvidia"), set("amd"), set("intel"), set()]) {
      const order = preferredBackends(v, os);
      assertEquals(order[order.length - 1], "cpu", `${os} ${[...v]}`);
    }
  }
});

Deno.test("github: master's head comes out of the commits atom feed", () => {
  // Verbatim shape of GitHub's feed — the fallback used when the API quota is
  // gone, which is the only time the Update button on a master build matters.
  const sha = "7ef790f0f1e6a1b1c0d2e3f4a5b6c7d8e9f0a1b2";
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>tag:github.com,2008:Grit::Commit/${sha}</id>
  <link href="https://github.com/ggml-org/llama.cpp/commit/${sha}"/>
 </entry>
</feed>`;
  assertEquals(shaFromCommitsAtom(xml), sha);
  assertEquals(shaFromCommitsAtom("<feed></feed>"), null);
});

Deno.test("command: extra arguments reach argv exactly as typed", () => {
  const base = { bin: "llama-server", model: "/m.gguf", settings: {} };
  // The escape hatch: a flag the catalog does not carry still gets through,
  // as ordinary argv tokens rather than one quoted blob.
  const withExtra = {
    ...base,
    settings: { extraArgs: "--lora /a.gguf  --cache-reuse 256" },
  };
  const cmd = argv("server", withExtra);
  assertEquals(cmd.slice(cmd.indexOf("--lora")), [
    "--lora",
    "/a.gguf",
    "--cache-reuse",
    "256",
  ]);
  // And it is visible in the preview, so what you see is still what runs.
  assertStringIncludes(commandLine("server", withExtra), "--cache-reuse 256");
  // Empty stays empty — no stray token.
  const blank = argv("server", { ...base, settings: { extraArgs: "   " } });
  assertEquals(blank.at(-1) === "" || /^-/.test(String(blank.at(-2))), true);
  assertEquals(blank.includes("--lora"), false);
});

// ── which GPUs llama.cpp may use ───────────────────────────────────────────

Deno.test("gpu: devices are named the way llama.cpp names them", () => {
  const nv = { ...gpu(24), vendor: "nvidia" as const, name: "RTX A" };
  const nv2 = { ...gpu(24), vendor: "nvidia" as const, name: "RTX B" };
  const amd = { ...gpu(2), vendor: "amd" as const, name: "iGPU" };
  const all = [nv, nv2, amd];

  // The prefix follows the BUILD, not the card, and the index counts within the
  // backend — so on CUDA the second NVIDIA card is CUDA1 even though it is the
  // machine's second of three GPUs.
  assertEquals(devices("cuda", all).map((d) => d.id), ["CUDA0", "CUDA1"]);
  assertEquals(devices("hip", all).map((d) => d.id), ["ROCm0"]);
  assertEquals(devices("vulkan", all).map((d) => d.id), [
    "Vulkan0",
    "Vulkan1",
    "Vulkan2",
  ]);
  // A CPU build addresses no GPU at all.
  assertEquals(devices("cpu", all), []);
  // And the label is what the user recognises, not the flag.
  assertEquals(devices("cuda", all)[1]?.label, "RTX B");
});

Deno.test("gpu: an empty -dev means every device, not none", () => {
  // llama.cpp's own default. Getting this backwards would silently disable the
  // GPUs of every user who never touched the setting.
  assertEquals(isEnabled("", "CUDA0"), true);
  assertEquals(isEnabled("CUDA1", "CUDA0"), false);
  assertEquals(isEnabled("CUDA0,CUDA1", "CUDA1"), true);
  assertEquals(parseDevices(" CUDA0 , CUDA1 "), ["CUDA0", "CUDA1"]);
  assertEquals(parseDevices(""), []);
});

Deno.test("gpu: toggling writes a stable flag and returns to the default", () => {
  const all = ["CUDA0", "CUDA1", "CUDA2"];
  // Switching one off from the unrestricted state names the survivors.
  assertEquals(toggleDevice("", all, "CUDA1", false), "CUDA0,CUDA2");
  // Order follows the device list, not the clicks — the preview must not churn.
  assertEquals(toggleDevice("CUDA2", all, "CUDA0", true), "CUDA0,CUDA2");
  // Switching everything back on is the DEFAULT, not a list of all of them.
  assertEquals(toggleDevice("CUDA0,CUDA2", all, "CUDA1", true), "");
  // Idempotent.
  assertEquals(toggleDevice("CUDA0", all, "CUDA1", false), "CUDA0");
});

Deno.test("gpu: the plan sees only the GPUs that are switched on", () => {
  const a = { ...gpu(24), name: "A" };
  const b = { ...gpu(24), name: "B" };
  const all = [a, b];

  // This is the point of the whole feature: unticking a card has to change the
  // memory plan, or the picture shows a placement that will not happen.
  assertEquals(enabledGpus("cuda", all, "").length, 2);
  assertEquals(enabledGpus("cuda", all, "CUDA0").map((g) => g.name), ["A"]);
  assertEquals(enabledGpus("cuda", all, "CUDA1").map((g) => g.name), ["B"]);

  // Before a build is chosen the device names are unknown, so no restriction is
  // expressible — but the machine still has its VRAM and the plan must say so.
  assertEquals(devices(undefined, all), []);
  assertEquals(enabledGpus(undefined, all, "").length, 2);
  // A CPU build genuinely has none.
  assertEquals(enabledGpus("cpu", all, "").length, 0);

  const both = plan(meta(), hw({ gpus: all }), { ngl: 999 });
  const one = plan(meta(), hw({ gpus: [a] }), { ngl: 999 });
  assert(
    both.vram.capacityB > one.vram.capacityB,
    "two cards must plan more VRAM than one",
  );
});

Deno.test("gpu: the device restriction reaches the command line", () => {
  const base = { bin: "llama-server", model: "/m.gguf", settings: {} };
  // Default emits nothing — llama.cpp already uses every device.
  assert(!argv("server", base).includes("-dev"));
  const restricted = argv("server", {
    ...base,
    settings: { device: "CUDA0" },
  });
  assertEquals(restricted[restricted.indexOf("-dev") + 1], "CUDA0");
});

// ── keeping the newest line in view ────────────────────────────────────────

Deno.test("scroll: an arrival is always shown, a token only if already at the bottom", () => {
  // A reader parked at the bottom follows everything.
  const atBottom = () => ({
    scrollTop: 600,
    scrollHeight: 900,
    clientHeight: 300,
  });
  const el1 = atBottom();
  assertEquals(stickToBottom(el1), true);
  assertEquals(el1.scrollTop, 900);

  // A reader who scrolled up is NOT yanked down by a streamed token…
  const scrolledUp = { scrollTop: 100, scrollHeight: 900, clientHeight: 300 };
  assertEquals(stickToBottom(scrolledUp), false);
  assertEquals(scrolledUp.scrollTop, 100, "their position is untouched");

  // …but a whole new message arriving is forced into view, which is what
  // "the last message is always visible when it arrives" means.
  assertEquals(stickToBottom(scrolledUp, true), true);
  assertEquals(scrolledUp.scrollTop, 900);

  // A box with nothing to scroll is trivially at the bottom.
  assertEquals(
    isNearBottom({ scrollTop: 0, scrollHeight: 200, clientHeight: 200 }),
    true,
  );
  // And a missing element is a no-op, not a crash.
  assertEquals(stickToBottom(null, true), false);
});

Deno.test("scroll: the slack tolerates a part-visible last line", () => {
  // 40px from the bottom still counts as "at the bottom" (default slack 48).
  assertEquals(
    isNearBottom({ scrollTop: 560, scrollHeight: 900, clientHeight: 300 }),
    true,
  );
  // 60px does not.
  assertEquals(
    isNearBottom({ scrollTop: 540, scrollHeight: 900, clientHeight: 300 }),
    false,
  );
});

Deno.test("stability: oversubscribed prompt threads are caught too", () => {
  // `-tb` had no check at all, at any value — and prompt processing is the
  // phase the user actually waits on.
  const machine = hw(); // 16 physical / 32 logical
  const over = stability(meta(), machine, { ...defaults(), threadsBatch: 64 });
  assert(
    over.warnings.some((w) =>
      w.key === "threadsBatch" && w.severity === "risk"
    ),
    "more batch threads than logical CPUs is a risk",
  );
  const smt = stability(meta(), machine, { ...defaults(), threadsBatch: 32 });
  assert(
    smt.warnings.some((w) => w.key === "threadsBatch"),
    "two batch threads per physical core is worth saying",
  );
  assertEquals(
    stability(meta(), machine, { ...defaults(), threadsBatch: 16 }).warnings
      .filter((w) => w.key === "threadsBatch"),
    [],
    "one thread per physical core is the intended answer, not a warning",
  );
  // The tuner's own output must never trip its own warning.
  const tuned = tune(meta(), machine, defaults(), "vram").settings;
  assertEquals(
    stability(meta(), machine, tuned).warnings.filter((w) =>
      w.key === "threadsBatch"
    ),
    [],
  );
});

Deno.test("tune: a MoE model that cannot fit still keeps attention on the GPU", () => {
  // The bug this pins: the partial-offload branch reset `--n-cpu-moe` to 0,
  // discarding the MoE strategy at the exact moment it is worth most. A
  // Mixtral-shaped layer is ~40 MB of attention against ~720 MB of experts, so
  // keeping experts in RAM buys ~16x more layers than dropping layers does.
  // Measured before the fix: 2 of 32 layers on a 3 GB card. After: 32 of 32.
  const m = moeMeta();
  const machine = hw({ gpus: [gpu(3, 0)] });
  const { settings, reasons } = tune(m, machine, defaults(), "hybrid");

  assertEquals(Number(settings.nCpuMoe), m.nLayer, "every expert goes to RAM");
  assert(
    Number(settings.ngl) >= m.nLayer - 4,
    `nearly every layer still fits: ${settings.ngl} of ${m.nLayer}`,
  );
  const p = plan(m, machine, settings);
  assertEquals(p.vram.overB, 0, "the placement actually fits");
  assert(
    reasons.some((r) => r.includes("experts")),
    `the reason must say the experts moved; got ${JSON.stringify(reasons)}`,
  );

  // Even on a card too small for a full offload, experts leave before layers do.
  // 2 GB rather than 1: at `-ngl >= 1` llama.cpp offloads the output head as
  // well (it is the last of the `nLayer + 1` slots), so this fixture's 300 MB
  // head plus the compute buffers plus the reserve is most of a 1 GB card
  // before a single layer is placed. That is a real constraint, not a fixture
  // detail — the old plan simply did not charge the head at partial offload.
  const tiny =
    tune(m, hw({ gpus: [gpu(2, 0)] }), defaults(), "hybrid").settings;
  assert(Number(tiny.nCpuMoe) > 0, "experts still move first");
  assert(Number(tiny.ngl) > 0, "and something still runs on the GPU");
});

Deno.test("tune: --n-cpu-moe is the smallest N that fits, not just one that does", () => {
  // "Optimal" is the word in the rule: every expert left on the GPU is one that
  // does not have to cross the PCIe bus, so N must be minimal.
  const m = moeMeta();
  for (const gb of [24, 12, 8, 4]) {
    const machine = hw({ gpus: [gpu(gb, 0)] });
    const { settings } = tune(m, machine, defaults(), "hybrid");
    const n = Number(settings.nCpuMoe);
    assert(n > 0, `${gb}GB: experts should move`);
    // One fewer must NOT fit — otherwise the tuner moved an expert it did not
    // have to.
    const looser = plan(m, machine, { ...settings, nCpuMoe: n - 1 });
    const total = machine.gpus.reduce((a, g) => a + g.vramTotalB, 0);
    assert(
      looser.vram.overB > 0 ||
        looser.vram.freeB < Math.max(512 * 1024 * 1024, total * 0.05),
      `${gb}GB: nCpuMoe=${n - 1} also fits, so ${n} was not minimal`,
    );
  }
});

Deno.test("tune: an ample card keeps the KV cache at f16", () => {
  // The other direction of "quantise when reasonable": q8_0 costs quality, so a
  // machine with room must not be given it. Nothing asserted this, which is what
  // would let a future edit make it unconditional.
  for (const gb of [48, 80]) {
    const s = tune(meta(), hw({ gpus: [gpu(gb, 0)] }), defaults()).settings;
    assertEquals(s.cacheTypeK, "f16", `${gb}GB has room to spare`);
    assertEquals(s.cacheTypeV, "f16");
  }
});

Deno.test("tune: the OS keeps its RAM, or the tuner says why it cannot", () => {
  const GB = 1024 ** 3;
  const mem = (totalGb: number, availGb: number) => ({
    totalB: totalGb * GB,
    availableB: availGb * GB,
    usedB: (totalGb - availGb) * GB,
    swapTotalB: 0,
    swapUsedB: 0,
  });
  const big = meta({
    name: "70B",
    nLayer: 80,
    nCtxTrain: 262144,
    layers: layers(80, 500 * 1024 * 1024),
  });

  // Enough RAM, absurd context: the tuner must LOWER the context rather than
  // emit a plan that eats every byte of MemAvailable. Before this it returned
  // the 262144 unchanged with a warning, and the OOM killer did the rest.
  const roomy = hw({ gpus: [gpu(24, 0)], mem: mem(64, 44) });
  const t1 = tune(big, roomy, defaults(), "hybrid");
  const p1 = plan(big, roomy, t1.settings);
  assertEquals(p1.ram.overB, 0);
  assert(
    p1.ram.freeB >= Math.max(GB, 44 * GB * 0.10),
    `the OS reserve must survive; free ${p1.ram.freeB}`,
  );
  assert(
    Number(t1.settings.ctxSize) < 262144,
    "the context is the lever it owns",
  );
  assert(
    t1.ctx < t1.optimalCtx,
    "and it reports having fallen short of the optimum",
  );

  // Not enough RAM at ANY context: no setting can fix that, so it must say so
  // loudly rather than pretend. This is the honest-refusal case.
  const tight = hw({ gpus: [gpu(4, 0)], mem: mem(16, 12) });
  const t2 = tune(big, tight, defaults(), "cpu");
  // The tuner no longer proposes a placement that cannot fit — it reports the
  // refusal, which is stronger than warning about a plan it handed over anyway.
  assertEquals(t2.possible, false, "no context makes this fit");
  assertStringIncludes(t2.blocker, "RAM");
  assert(
    t2.reasons.some((r) => r.includes("not possible")),
    `the refusal must reach the reasons too; got ${JSON.stringify(t2.reasons)}`,
  );
  assert(
    stability(big, tight, t2.settings).warnings.some((w) =>
      w.severity === "risk"
    ),
    "and it must be a risk on screen, not just a sentence in the reasons",
  );
});

Deno.test("stability: near-zero RAM headroom is a caution", () => {
  const GB = 1024 ** 3;
  // Fits, but only just: the kernel cannot page these pages out.
  const m = meta({ nLayer: 40, layers: layers(40, 300 * 1024 * 1024) });
  const machine = hw({
    gpus: [],
    mem: {
      totalB: 16 * GB,
      // 12.95 GB of weights + KV against 13 GB free: it fits, with 0.05 GB
      // spare — the "exactly fits" case that used to pass without a word.
      availableB: 13 * GB,
      usedB: 3 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });
  const s = { ...defaults(), ngl: 0, ctxSize: 4096 };
  const p = plan(m, machine, s);
  assertEquals(p.ram.overB, 0, "the fixture must actually fit");
  assert(p.ram.freeB < GB, "and leave under a GB");
  assert(
    stability(m, machine, s).warnings.some((w) => w.key === "ctxSize"),
    "which has to be said",
  );
});

Deno.test("tune: a backend without a quantised-KV kernel is not given one", () => {
  // The bug: the tuner set `-ctk/-ctv q8_0` for every backend. Where there is no
  // quantised-KV kernel the server refuses to load, so "optimal settings" meant
  // settings that do not start. A f16 cache that loads beats a q8_0 that might
  // not. It also used to force `-fa on`, which fails the same way.
  // Same shape as the tight-fit test above: an 8k f16 cache is what tips a
  // 12 GB card over, so the tuner genuinely WANTS to quantise here.
  const m = meta({ nCtxTrain: 131072 });
  const tight = { gpus: [gpu(12, 0)] };
  const base = { ...defaults(), ctxSize: 65536 };

  const cuda = tune(m, hw({ ...tight, backend: "cuda" }), base).settings;
  assertEquals(cuda.cacheTypeK, "q8_0", "CUDA has carried this for years");
  assertEquals(cuda.flashAttn, "on");

  for (const backend of ["vulkan", "hip", "cpu"] as const) {
    const s = tune(m, hw({ ...tight, backend }), base).settings;
    assertEquals(s.cacheTypeK, "f16", `${backend} must keep f16`);
    assertEquals(s.cacheTypeV, "f16", `${backend} must keep f16`);
    assertEquals(
      s.flashAttn,
      "auto",
      `${backend}: let llama.cpp decide rather than force a missing kernel`,
    );
  }

  // No build selected yet: unknown is not a licence to guess.
  const unknown = tune(m, hw({ ...tight, backend: undefined }), base);
  assertEquals(unknown.settings.cacheTypeK, "f16");
  assert(
    unknown.reasons.some((r) =>
      r.includes("quantised KV cache is not offered")
    ),
    "and it says why, rather than silently costing VRAM",
  );
});

Deno.test("tune: a cache type left over from another model is not inherited", () => {
  // q8_0 is the tuner's own deliberate choice, so it must not survive as a
  // silent quality cost on a model that has room to spare — nor as a load
  // failure carried into a session with a different backend.
  const roomy = hw({ gpus: [gpu(80, 0)] });
  const stale = { ...defaults(), cacheTypeK: "q8_0", cacheTypeV: "q8_0" };
  const { settings } = tune(meta(), roomy, stale);
  assertEquals(settings.cacheTypeK, "f16", "reset, because it is not needed");
  assertEquals(settings.cacheTypeV, "f16");
});

// ── KV cache: the architecture decides, not one formula ────────────────────

Deno.test("plan: a sliding-window model does not pay full context on every layer", () => {
  // Gemma-3 shape: a 1024-token window with a pattern of 6, so five layers in
  // six stop growing at the window. Billing every layer for the whole context
  // overstated its cache several-fold — and the number was presented as exact.
  const full = meta({
    name: "dense-48L",
    nLayer: 48,
    nCtxTrain: 131072,
    layers: layers(48, 100 * 1024 * 1024),
  });
  const swa = meta({
    ...full,
    name: "gemma-3 shape",
    swaWindow: 1024,
    swaPattern: 6,
  });
  const s = { ...defaults(), ctxSize: 32768, batchSize: 2048 };

  const kvFull = kvTotal(full, s, 32768);
  const kvSwa = kvTotal(swa, s, 32768);
  assert(kvSwa < kvFull, "the windowed model must cost less");
  // 8 global layers at full context + 40 windowed at ~3072 tokens each, versus
  // 48 at full context: roughly a quarter.
  const ratio = kvSwa / kvFull;
  assert(
    ratio > 0.15 && ratio < 0.35,
    `expected roughly a quarter, got ${ratio.toFixed(3)}`,
  );
  // And the split itself is the thing being claimed.
  assertEquals(swaSplit(swa), { full: 8, windowed: 40 });
  assertEquals(swaSplit(full), { full: 48, windowed: 0 });

  // The sparse-attention scratch, against the sweep it was calibrated on. The
  // shape is what matters: the context term dominates and the micro-batch term
  // is small, which is the OPPOSITE of the first estimate — it derived one
  // ubatch-proportional constant from a single 68.5 GiB observation at a
  // 1M context, and was ~3.4x too pessimistic everywhere anyone runs.
  const sparse = meta({ indexerTopK: 2048 });
  const MB = 1024 ** 2;
  const at = (ctx: number, ub: number, np = 1) =>
    computeScratch(sparse, ub, ctx, np) / MB;
  // Measured at ONE slot, ub 512, above a 4,096 baseline (plan.ts carries the
  // table): +2,045 MB at 262,144 and +8,547 MB at 1,048,576. The estimate must
  // cover them without ballooning: never under, within about half again.
  for (const [ctx, measuredMb] of [[262144, 2045], [1048576, 8547]] as const) {
    const est = at(ctx, 512);
    assert(est >= measuredMb, `${ctx}: ${est.toFixed(0)} MB under-estimates`);
    assert(est < measuredMb * 1.6, `${ctx}: ${est.toFixed(0)} MB is wild`);
  }
  // Slots MULTIPLY it, and that is the term the app was missing entirely:
  // llama.cpp's `-np` default is auto, auto chose FOUR, and a 1,048,576 context
  // cost 43.5 GB of VRAM instead of 21.5. Measured 29.0 KB/token at four slots
  // against 8.1 at one — the ratio is the slot count.
  assert(
    Math.abs(at(262144, 512, 4) / at(262144, 512) - 4) < 0.01,
    "four slots is four times the scratch",
  );
  // Quadrupling the micro-batch moved 1.4 GB at a quarter-million tokens, not
  // 4x the whole term. Anything that scales the scratch BY ubatch fails this.
  const spread = at(262144, 1024) / at(262144, 256);
  assert(spread > 1 && spread < 1.6, `ubatch swing was ${spread.toFixed(2)}x`);
  // It keeps growing past the points it was fitted to — the shape is linear in
  // the context, so nothing plateaus and no plan is offered on a flat spot.
  assert(at(393216, 512) > at(262144, 512) * 1.4);
  // And a model that declares no indexer is untouched: this is the sparse term.
  assertEquals(computeScratch(meta({ indexerTopK: 0 }), 512, 262144), 0);

  // A window with no pattern means every layer is local (llama.cpp's default).
  assertEquals(
    swaSplit(meta({ ...full, swaWindow: 512, swaPattern: 1 })),
    { full: 0, windowed: 48 },
  );
});

Deno.test("plan: a hybrid linear-attention model caches only its full-attention layers", () => {
  // Qwen3.8-27B shape: 65 declared layers, but `full_attention_interval: 4`
  // means only every 4th trunk layer holds a per-token KV cache — 16 of 64,
  // plus the MTP block, which is dense attention. The other 48 are recurrent
  // (gated delta net) and hold a constant per-sequence state. Billing all 65
  // layers said a 131,072 context needs 18.5 GB of q8_0 cache on a model whose
  // real cache is 4.9 GB — the difference between "does not fit in 48 GB of
  // VRAM" and the 262,144 the same cards actually run.
  const qwen = meta({
    name: "Qwen3.8 27B shape",
    arch: "qwen35",
    nLayer: 65,
    nCtxTrain: 262144,
    nEmbd: 5120,
    nHead: 24,
    nHeadKv: 4,
    keyLength: 256,
    valueLength: 256,
    fullAttnInterval: 4,
    ssmDConv: 4,
    ssmDInner: 6144,
    ssmDState: 128,
    ssmNGroup: 16,
    nextnLayers: 1,
    layers: layers(65, 400 * 1024 * 1024),
  });
  const dense = meta({ ...qwen, fullAttnInterval: 0 });
  assertEquals(
    kvLayers(qwen),
    17,
    "16 full-attention trunk layers + the MTP block",
  );
  assertEquals(kvLayers(dense), 65);

  const s = { ...defaults(), ctxSize: 131072 };
  const kvHybrid = kvTotal(qwen, s, 131072);
  const kvDense = kvTotal(dense, s, 131072);
  // 17/65 of the uniform figure, plus the small constant recurrent state.
  const state = recurrentStateB(qwen, 1);
  assertEquals(kvHybrid, kvDense * (17 / 65) + state);
  // The state itself: 48 recurrent layers x (n_embd_r + n_embd_s) x f32 —
  // (3 * (6144 + 2*16*128) + 128 * 6144) * 4 bytes = ~157 MB, per slot.
  assertEquals(state, 48 * (3 * (6144 + 2 * 16 * 128) + 128 * 6144) * 4);
  assertEquals(recurrentStateB(qwen, 4), state * 4, "each slot holds its own");
  assertEquals(recurrentStateB(dense), 0, "no recurrent layers, no state");

  // The per-token rate the UI shows follows the same split — 17 layers' worth.
  const perTok = kvPerToken(qwen, s);
  assertEquals(perTok, 17 * 4 * (256 * 2 + 256 * 2));
  // At f16 and 131,072 tokens that is ~9.1 GB where uniform billing said 34.9.
  assert(
    kvHybrid < 10 * 1024 ** 3,
    `hybrid cache is ${kvHybrid / 1024 ** 3} GB`,
  );
  assert(kvDense > 30 * 1024 ** 3);

  // A quantised cache shrinks the per-token half and leaves the recurrent
  // state alone — it is f32 in llama.cpp regardless of `-ctk`.
  const q8 = { ...s, cacheTypeK: "q8_0", cacheTypeV: "q8_0" };
  assertEquals(
    kvTotal(qwen, q8, 131072),
    kvDense * (17 / 65) * (34 / 64) + state,
  );
});

Deno.test("plan: an MLA model caches a latent, not one entry per head", () => {
  // DeepSeek-V2/V3 compress the cache to a rank-512 latent plus a 64-wide RoPE
  // part, per layer. Billing 128 heads x (192 + 128) overstated V3 ~70x.
  const heads = meta({
    name: "V3 as if MHA",
    nLayer: 61,
    nHead: 128,
    nHeadKv: 128,
    keyLength: 192,
    valueLength: 128,
    layers: layers(61, 100 * 1024 * 1024),
  });
  const mla = meta({ ...heads, name: "V3 shape", kvLoraRank: 512 });
  const s = defaults();

  const perTokenMha = kvPerToken(heads, s);
  const perTokenMla = kvPerToken(mla, s);
  // 61 * (512 + 64) * 2 bytes = 70,272 against 61 * 128 * (192+128) * 2.
  assertEquals(perTokenMla, 61 * (512 + 64) * 2);
  assert(
    perTokenMha / perTokenMla > 50,
    `the difference is the point: ${(perTokenMha / perTokenMla).toFixed(0)}x`,
  );
  // And the quantised cache still applies to it.
  const q8 = kvPerToken(mla, { ...s, cacheTypeK: "q8_0", cacheTypeV: "q8_0" });
  assert(q8 < perTokenMla, "q8_0 must still shrink an MLA cache");
});

Deno.test("plan: an ordinary model is unaffected by either correction", () => {
  // The regression guard: 99% of models declare neither key, and their number
  // must not move by a byte.
  const m = meta({ nCtxTrain: 131072 });
  const s = { ...defaults(), ctxSize: 8192 };
  assertEquals(kvTotal(m, s, 8192), kvPerToken(m, s) * 8192);
  // Exactly the figure the older test pinned: 32 layers x 8 kv-heads x
  // (128 * 2 + 128 * 2).
  assertEquals(kvPerToken(m, s), 32 * 8 * (128 * 2 + 128 * 2));
});

Deno.test("demo: the demo machine and library are internally consistent", () => {
  // Demo mode exists so a screenshot or a bug report need not carry anyone's
  // real hardware, paths or model collection. The numbers still have to be
  // coherent, because the planner and tuner run against them for real — a
  // fictional model with no layers would make a fictional memory plan.
  const gpus = demoGpus();
  const mem = demoMem();
  assert(demoCpu().cores >= 2 && demoCpu().threads >= demoCpu().cores);
  assert(gpus.length > 0 && gpus[0]!.vramTotalB > 0);
  assert(mem.availableB > 0 && mem.availableB <= mem.totalB);

  const machine = {
    cpu: demoCpu(),
    mem,
    gpus,
    os: "linux",
    arch: "x86_64",
    backend: "cuda" as const,
  };
  for (const m of demoModels()) {
    assert(m.meta.nLayer > 0, `${m.file}: needs layers to be plannable`);
    assertEquals(
      m.meta.layers.length,
      m.meta.nLayer,
      `${m.file}: a layer table that matches nLayer`,
    );
    assert(m.meta.nCtxTrain > 0, `${m.file}: needs a trained context`);
    assert(!m.path.includes("home"), "no real-looking home paths");

    // The whole point: every demo model can be placed somewhere, so the demo
    // never opens on an app that cannot do anything.
    const all = tuneAll(m.meta, machine, defaults());
    assert(
      all.vram.possible || all.hybrid.possible || all.cpu.possible,
      `${m.file}: no placement can run it`,
    );
    assertEquals(bestPlacement(all) in all, true);
  }
});

Deno.test("plan: a header this app cannot read produces zeros, never NaN", () => {
  // GGUF headers come from files this app did not write. A truncated or hostile
  // one can yield NaN, a negative, or `nHead: 0` (which makes the head-dim
  // fallback `0 / 0`). One such value used to poison every total it touched —
  // silently, because arithmetic does not complain and NaN comparisons are all
  // FALSE, so `overB === 0` and `freeB >= margin` quietly stopped meaning
  // anything and the tuner's fit checks became coin flips.
  const machine = hw({ gpus: [gpu(24, 0)] });
  const broken = [
    meta({ nEmbd: Number.NaN }),
    meta({ nHead: 0, nHeadKv: 0, keyLength: 0, valueLength: 0 }),
    meta({ nLayer: 1, layers: [{ i: 0, bytes: -100, expert: -5 }] }),
    // Experts larger than the layer that holds them: dense would go negative.
    meta({ nLayer: 1, layers: [{ i: 0, bytes: 100, expert: 500 }] }),
    meta({ nLayer: 0, layers: [] }),
    meta({ embdBytes: Number.NaN, outputBytes: Number.NaN }),
  ];
  for (const m of broken) {
    const p = plan(m, machine, { ...defaults(), ngl: 999 });
    for (
      const [label, v] of [
        ["vram.usedB", p.vram.usedB],
        ["ram.usedB", p.ram.usedB],
        ["kvTotalB", p.kvTotalB],
        ["kvPerTokenB", p.kvPerTokenB],
        ["vram.freeB", p.vram.freeB],
        ["vram.overB", p.vram.overB],
      ] as const
    ) {
      assert(Number.isFinite(v), `${m.name}: ${label} = ${v}`);
      assert(v >= 0, `${m.name}: ${label} is negative (${v})`);
    }
    // And the tuner still returns a usable answer rather than throwing.
    const t = tune(m, machine, defaults(), "vram");
    assert(Number.isFinite(t.ctx), `${m.name}: ctx = ${t.ctx}`);
  }
});

Deno.test("archive: a Windows-spelled traversal does not escape either", () => {
  // `contained` split on "/" only, so an archive written on Windows spelling
  // the traversal `..\..\evil` had no ".." component and sailed through —
  // harmless on Linux (one oddly-named file), an escape on Windows.
  const e = (name: string, link?: string) => ({
    name,
    link,
    mode: 0o644,
    size: 0,
    data: new Uint8Array(),
    type: "file" as const,
  });
  const kept = safeEntries([
    e("../../evil"),
    e("..\\..\\evil"),
    e("a/..\\../evil"),
    e("/etc/passwd"),
    e("C:\\Windows\\evil"),
    e("link", "../../etc/passwd"),
    e("link2", "..\\..\\etc\\shadow"),
    e("ok/file"),
  ] as never).map((x) => x.name);
  assertEquals(kept, ["ok/file"], "only the contained entry survives");
});

Deno.test("ollama: a broken manifest is distinguished from a cloud-only one", () => {
  // Both used to return null, so the scanner could not tell them apart and
  // dropped both in silence. A cloud entry has nothing on disk and SHOULD be
  // skipped; an unparseable manifest is a model the user has that would simply
  // never appear — the exact "silent absence" this app promises not to do.
  const path = "/x/manifests/registry.ollama.ai/library/demo/latest";

  const good = JSON.stringify({
    layers: [{
      mediaType: "application/vnd.ollama.image.model",
      digest: "sha256:abc",
      size: 10,
    }],
  });
  assertEquals(manifestSkipReason(path, good), null, "a real model is no skip");

  const cloud = JSON.stringify({ layers: null });
  assertEquals(manifestSkipReason(path, cloud), "cloud-only");
  assertEquals(
    manifestSkipReason(path, JSON.stringify({ layers: [] })),
    "cloud-only",
  );

  assertEquals(manifestSkipReason(path, "{not json"), "unreadable");
  // Parses, has layers, but none of them are the model blob.
  const odd = JSON.stringify({
    layers: [{
      mediaType: "application/vnd.ollama.image.license",
      digest: "sha256:z",
    }],
  });
  assertEquals(manifestSkipReason(path, odd), "unreadable");
});

Deno.test("context: the preset ladder is the one people recognise", () => {
  // Powers of two up to 128k — how models and benchmarks describe context —
  // then 128k steps to the top: 256k straight to 512k skipped exactly the
  // sizes a long-context machine wants to try.
  assertEquals(CTX_PRESETS.map(ctxLabel), [
    "16k",
    "32k",
    "64k",
    "128k",
    "256k",
    "384k",
    "512k",
    "640k",
    "768k",
    "896k",
    "1M",
  ]);
  // Ascending: doubling below 128k, 128k steps above it.
  const STEP = 131_072;
  for (let i = 1; i < CTX_PRESETS.length; i++) {
    const prev = CTX_PRESETS[i - 1]!;
    const cur = CTX_PRESETS[i]!;
    assertEquals(cur, prev < STEP ? prev * 2 : prev + STEP);
  }
  assert(CTX_PRESETS[0]! > MIN_CTX, "every preset is a usable context");
  // The label is what a user reads, not a byte count.
  assertEquals(ctxLabel(2048), "2k");
  assertEquals(ctxLabel(1_048_576), "1M");
  assertEquals(ctxLabel(512), "512");
});

Deno.test("plan: a projection does not count our own running model twice", () => {
  // The bug rule visuals#3 names. `plan` reads "in use by others" straight off
  // the telemetry, and that includes a llama-server THIS app is running — so
  // projecting a model while one was loaded charged the running one twice: once
  // as somebody else's memory, once as the new plan. The projection has to be
  // "current state, minus what we hold, plus the new model".
  const GB = 1024 ** 3;
  const machine = hw({
    gpus: [{ ...gpu(24), vramUsedB: 9 * GB }],
    mem: {
      totalB: 64 * GB,
      availableB: 40 * GB,
      usedB: 24 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });

  // 8 of those 9 GB are ours, and 6 GB of RAM.
  const base = withoutOurUsage(machine, 8 * GB, 6 * GB);
  assertEquals(
    Math.round(base.gpus[0]!.vramUsedB / GB),
    1,
    "only the 1 GB that is not ours remains",
  );
  assertEquals(Math.round(base.mem!.availableB / GB), 46, "our RAM comes back");

  // The projection is therefore smaller than one made without the subtraction.
  const m = meta();
  const naive = plan(m, machine, { ...defaults(), ngl: 999 });
  const honest = plan(m, base, { ...defaults(), ngl: 999 });
  assert(
    honest.vram.otherB < naive.vram.otherB,
    "the running model must not appear as other people's memory",
  );
  assert(
    honest.vram.freeB > naive.vram.freeB,
    "and its VRAM is available again",
  );

  // Never removes more than is there, and never invents free memory.
  const over = withoutOurUsage(machine, 999 * GB, 999 * GB);
  assert(over.gpus[0]!.vramUsedB >= 0);
  assert(over.mem!.availableB <= over.mem!.totalB);
  // Nothing of ours running: the machine is untouched.
  const same = withoutOurUsage(machine, 0, 0);
  assertEquals(same.gpus[0]!.vramUsedB, machine.gpus[0]!.vramUsedB);
});

Deno.test("plan: the current state of an idle machine has no llama.cpp in it", () => {
  // The "nothing running" half of Current Memory State goes through the same
  // `plan` as everything else, via NO_MODEL — one code path, not a second
  // subtly-different one.
  const GB = 1024 ** 3;
  const machine = hw({ gpus: [{ ...gpu(24), vramUsedB: 2 * GB }] });
  const p = plan(NO_MODEL, machine, { ...defaults(), ngl: 0 });
  assertEquals(p.vram.usedB, 0, "we are using no VRAM");
  assertEquals(p.kvTotalB, 0);
  assertEquals(p.layersOnGpu, 0);
  // But the machine's real occupancy still shows.
  assertEquals(Math.round(p.vram.otherB / GB), 2);
  assert(p.vram.freeB > 0);
});

Deno.test("plan: an idle machine is not given advice about a model it has no", () => {
  // The "current state" view goes through `plan` as NO_MODEL, and it used to
  // come back carrying "GPU layers is 0, so the GPU is idle. Raise it to use the
  // card." — guidance attached to a machine with nothing loaded to raise.
  const idle = plan(NO_MODEL, hw({ gpus: [gpu(24, 0)] }), {
    ...defaults(),
    ngl: 0,
  });
  assert(
    !idle.notes.some((n) => n.includes("GPU layers is 0")),
    `no advice without a model; got ${JSON.stringify(idle.notes)}`,
  );
  // With a real model the advice is still there — it is useful then.
  const loaded = plan(meta(), hw({ gpus: [gpu(24, 0)] }), {
    ...defaults(),
    ngl: 0,
  });
  assert(loaded.notes.some((n) => n.includes("GPU layers is 0")));
});

// ── storage ────────────────────────────────────────────────────────────────

Deno.test("disk: df output parses, and one filesystem is counted once", () => {
  // Verbatim `df -kP` from the development machine, which is the spec: `-P` is
  // what keeps the columns aligned when a device name is long.
  const out =
    `Filesystem            1024-blocks       Used  Available Capacity Mounted on
/dev/nvme0n1p4         1572096000  187242512 1384853488      12% /
/dev/mapper/cryptHome  2353905664 2261927900   88748452      97% /home
/dev/mapper/cryptHome  2353905664 2261927900   88748452      97% /home`;
  const d = parseDf(out);
  assertEquals(d.length, 2, "the repeated mount is counted once");
  assertEquals(d[0]!.mount, "/");
  assertEquals(d[0]!.filesystem, "/dev/nvme0n1p4");
  assertEquals(d[0]!.totalB, 1572096000 * 1024);
  assertEquals(d[1]!.mount, "/home");
  // `available` is not `total - used`: the reserved blocks are root's.
  assert(d[1]!.availB < d[1]!.totalB - d[1]!.usedB + 1024 * 1024 * 1024);

  // A mount point with spaces keeps them.
  const spaced = parseDf(
    `Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/sdb1 1000 500 500 50% /media/my drive`,
  );
  assertEquals(spaced[0]!.mount, "/media/my drive");

  // Junk is skipped rather than producing NaN rows.
  assertEquals(parseDf("").length, 0);
  assertEquals(parseDf("Filesystem\nnot a row at all").length, 0);
});

Deno.test("disk: a filesystem too full for a source build is flagged", () => {
  const GB = 1024 ** 3;
  const roomy = {
    filesystem: "a",
    mount: "/",
    totalB: 500 * GB,
    usedB: 100 * GB,
    availB: 400 * GB,
  };
  const tight = { ...roomy, availB: 2 * GB };
  assertEquals(tooFullToBuild(roomy), false);
  assertEquals(tooFullToBuild(tight), true, "2 GB will not hold a cmake tree");
  assertEquals(tooFullToBuild(null), false, "unknown is not a failure");
});

Deno.test("tune: a pinned context is clamped the same way everywhere", () => {
  // This clamp used to exist twice — once in the tuner, once inline in the
  // all-in-one page — and the UI copy had dropped the floor. A pin of 512
  // therefore RENDERED as 512 and RAN as 2048, which is the one thing a "what you
  // see is what runs" app must not do.
  assertEquals(
    pinnedCtx(512, 32768),
    MIN_CTX,
    "below the floor comes up to it",
  );
  assertEquals(pinnedCtx(64000, 32768), 32768, "above the model comes down");
  assertEquals(pinnedCtx(16384, 32768), 16384, "in range is left alone");
  // No model header yet: the ceiling is unknown, so only the floor applies.
  assertEquals(pinnedCtx(16384, 0), 16384);
  assertEquals(pinnedCtx(100, 0), MIN_CTX);
  // A model trained under the floor: the model wins. The floor must not round
  // a pin up past what the model can attend over — that is the one invariant
  // the whole context search exists to keep.
  assertEquals(pinnedCtx(512, 512), 512, "the trained ceiling beats the floor");
  assertEquals(pinnedCtx(4096, 1024), 1024);
});

Deno.test("tune: with no memory reading yet, nothing is proposed", () => {
  // The boot race, which shipped: `models.scan()` and `hw.refresh()` resolve
  // within milliseconds of each other, and when models won the tuner was asked to
  // plan against a machine of unknown size. It answered "yes, 131,072 tokens" for
  // a plan needing 78 GB of host RAM, `cfg.setPlacement` PERSISTED the CPU
  // fallback, and the GPU then sat idle for good. Refusing costs one poll.
  const big = meta({
    name: "70B",
    nLayer: 80,
    nHeadKv: 8,
    nCtxTrain: 131072,
    layers: layers(80, 420 * 1024 * 1024),
  });
  for (const placement of ["vram", "hybrid", "cpu"] as const) {
    const t = tune(big, hw({ mem: null }), defaults(), placement);
    assertEquals(t.possible, false, `${placement} must not claim to fit`);
    assertStringIncludes(
      t.blocker,
      "memory reading has not arrived",
      "and it says WHY, not 'needs more RAM than is free'",
    );
  }
  // And the moment the reading lands, the same call succeeds.
  assertEquals(tune(big, hw(), defaults(), "cpu").possible, true);
});

Deno.test("tune: a quantised cache is also taken to keep work off the host", () => {
  // It used to be taken only when it bought CONTEXT. A long-context MoE that
  // reaches its full trained length either way can still be paying for it with
  // gigabytes of experts and cache in system RAM — which cross PCIe on every
  // single token. Halving the cache brings some of that back onto the card at the
  // same context, and that is worth more than the quality q8_0 costs.
  const MB = 1024 * 1024;
  const m = meta({
    name: "Long MoE",
    nLayer: 41,
    nCtxTrain: 65536,
    nHeadKv: 2,
    keyLength: 256,
    valueLength: 256,
    nExpert: 128,
    nExpertUsed: 8,
    layers: layers(41, 900 * MB, 820 * MB),
  });
  const machine = hw({ gpus: [gpu(12, 0.5), gpu(12, 0.5)], backend: "cuda" });

  const chosen = tune(m, machine, defaults(), "hybrid");

  assertEquals(chosen.settings.cacheTypeK, "q8_0");
  assertEquals(chosen.settings.cacheTypeV, "q8_0");
  // What it buys moved when the objective did. The tuner now fixes residency
  // FIRST and grows the context only as far as that allows, so halving the
  // cache no longer rescues bytes from the host — it buys the context those
  // bytes were blocking. Either way it has to buy something, and say which.
  assert(
    chosen.reasons.some((r) =>
      r.includes("back onto the GPU") || r.includes("lifts the context")
    ),
    `q8_0 must state what it bought: ${chosen.reasons.join(" | ")}`,
  );
  const asF16Ctx = tune(m, machine, {
    ...defaults(),
    cacheTypeK: "f16",
    cacheTypeV: "f16",
  }, "hybrid");
  assert(
    chosen.ctx >= asF16Ctx.ctx,
    `q8_0 must not cost context: ${chosen.ctx} vs ${asF16Ctx.ctx}`,
  );
  // The counterfactual: this exact placement with a full-precision cache does
  // NOT fit the cards, which is what q8_0 is paying for.
  const asF16 = plan(m, machine, {
    ...chosen.settings,
    cacheTypeK: "f16",
    cacheTypeV: "f16",
  });
  // `fits`, not `vram.overB`: on two cards a placement can be inside the total
  // and still have no division of the layers that any single card can hold.
  assert(
    !asF16.fits,
    `f16 at the same placement must not fit the cards: over by ${asF16.vram.overB}, per-card ${
      JSON.stringify(asF16.devices.bytesB)
    } of ${JSON.stringify(asF16.devices.budgetsB)}`,
  );
});

Deno.test("tune: the context bands are ordered, clamped, and honest about Max", () => {
  // The kata asks for Min / Opt / Big / Max. Only Max is a fact — `nCtxTrain`,
  // the length the model was actually trained for. Opt and Big are estimates and
  // the UI marks them; what this pins is that they are always ORDERED and never
  // past Max, including for the awkward models where the whole range collapses.
  for (const trained of [128, 512, 2048, 8192, 32768, 131072, 1048576]) {
    const b = ctxBands(meta({ nCtxTrain: trained }));
    const target = optimalCtx(meta({ nCtxTrain: trained }));
    assertEquals(b.max, target, `Max is the trained length for ${trained}`);
    assert(
      b.min <= b.opt && b.opt <= b.big && b.big <= b.max,
      `ordered at ${trained}: ${JSON.stringify(b)}`,
    );
    assert(b.min > 0, `every band is usable at ${trained}`);
    for (const [k, v] of Object.entries(b)) {
      assert(
        Number.isFinite(v) && v > 0 && v <= b.max,
        `${k}=${v} must be finite and within Max at ${trained}`,
      );
    }
  }

  // A tiny model collapses to its own ceiling rather than producing a range that
  // runs past what it can do.
  const tiny = ctxBands(meta({ nCtxTrain: 256 }));
  assertEquals(tiny.max, 256);
  assertEquals(tiny.min, 256, "nothing below Max is offered for a 256 model");

  // And the shape on a normal long-context model is the documented one.
  const long = ctxBands(meta({ nCtxTrain: 131072 }));
  assertEquals(long, { min: 4096, opt: 32768, big: 65536, max: 131072 });

  // A YaRN-stretched model: Max is the ADVERTISED length — the Models page
  // reads nCtxTrain directly, and the same file must not have two maxima
  // (DeepSeek-V4 showed 1,048,576 there and 65,536 on the all-in-one page).
  // The native pre-stretch length is the one quality fact in the header, so
  // it anchors Big, with Opt at half of it; the tuner's own aim stays native.
  const ds = meta({ nCtxTrain: 1_048_576, nCtxOrig: 65_536 });
  assertEquals(ctxBands(ds), {
    min: 4096,
    opt: 32_768,
    big: 65_536,
    max: 1_048_576,
  });
  assertEquals(optimalCtx(ds), 65_536, "the auto-tuner still aims native");

  // Exactly one band is presented as measured.
  assertEquals(
    CTX_BANDS.filter((b) => !b.estimated).map((b) => b.id),
    ["max"],
    "Max is read from the model; the rest are estimates and must say so",
  );
});

Deno.test("adapt: headroom buckets react to real change and ignore jitter", () => {
  const GB = 1024 ** 3;
  const cap = 48 * GB;

  // The whole point: a game taking VRAM crosses a boundary, telemetry noise does
  // not. These machines are workstations where the raw number never holds still,
  // and keying a re-tune on it would rewrite the user's settings every second.
  const quiet = headroomBucket(40 * GB, cap);
  assertEquals(headroomBucket(40 * GB - 200 * 1024 ** 2, cap), quiet, "jitter");
  assertEquals(headroomBucket(39.8 * GB, cap), quiet, "still jitter");
  assert(
    headroomBucket(20 * GB, cap) < quiet,
    "a game taking 20 GB is news",
  );
  // And the other direction — it has to notice memory coming BACK, or the app
  // keeps running at a third of the card because a game was open when it planned.
  assert(
    headroomBucket(46 * GB, cap) > headroomBucket(20 * GB, cap),
    "memory freed is news too",
  );

  // Both pools, in one key, and either moving is enough.
  const base = {
    vramFreeB: 40 * GB,
    vramCapacityB: cap,
    ramFreeB: 120 * GB,
    ramCapacityB: 186 * GB,
  };
  assertEquals(headroomKey(base), headroomKey({ ...base }), "stable");
  assert(
    headroomKey({ ...base, vramFreeB: 8 * GB }) !== headroomKey(base),
    "VRAM alone moves the key",
  );
  assert(
    headroomKey({ ...base, ramFreeB: 10 * GB }) !== headroomKey(base),
    "RAM alone moves the key",
  );

  // Hostile and unread inputs give a stable answer, not NaN — this is a cache
  // key, and a NaN in it would re-tune forever.
  for (
    const [free, capacity] of [[NaN, cap], [1, 0], [-5, cap], [cap * 9, cap]]
  ) {
    const b = headroomBucket(free as number, capacity as number);
    assert(Number.isInteger(b) && b >= 0, `bucket(${free},${capacity}) = ${b}`);
  }
  // Bounded, so a re-tune can only fire a few times across a pool's whole range.
  assert(headroomBucket(cap, cap) <= 1 / HEADROOM_FRACTION, "few buckets");
});

Deno.test("adapt: a running model is told about drift, in both directions", () => {
  const GB = 1024 ** 3;
  // A 10 GB / 4 GB run spawned onto a machine with 11 GB / 5 GB free — so a
  // quiet machine shows ~1 GB free in each pool afterwards.
  const running = {
    startedVramB: 10 * GB,
    startedRamB: 4 * GB,
    vramFreeAtStartB: 11 * GB,
    ramFreeAtStartB: 5 * GB,
  };

  // Nothing happening.
  assertEquals(
    drift({
      ...running,
      vramOverB: 0,
      ramOverB: 0,
      vramFreeB: 1 * GB,
      ramFreeB: 1 * GB,
    })
      .kind,
    "none",
  );
  // Someone else took memory this server depends on — either pool counts.
  const squeezedV = drift({
    ...running,
    vramOverB: 3 * GB,
    ramOverB: 0,
    vramFreeB: 0,
    ramFreeB: 1 * GB,
  });
  assertEquals(squeezedV.kind, "squeezed");
  assertEquals(
    drift({
      ...running,
      vramOverB: 0,
      ramOverB: 2 * GB,
      vramFreeB: 1 * GB,
      ramFreeB: 0,
    })
      .kind,
    "squeezed",
    "RAM pressure counts as much as VRAM",
  );
  // Memory came back, and enough of it to be worth a restart.
  assertEquals(
    drift({
      ...running,
      vramOverB: 0,
      ramOverB: 0,
      vramFreeB: 9 * GB,
      ramFreeB: 0,
    })
      .kind,
    "roomier",
  );
  // But a sliver is not worth interrupting anyone over.
  assertEquals(
    drift({
      ...running,
      vramOverB: 0,
      ramOverB: 0,
      vramFreeB: 0.3 * GB,
      ramFreeB: 0.2 * GB,
    }).kind,
    "none",
    "3% more is not news",
  );
  // Squeezed always wins over roomier — one pool starving is the urgent fact.
  assertEquals(
    drift({
      ...running,
      vramOverB: 2 * GB,
      ramOverB: 0,
      vramFreeB: 0,
      ramFreeB: 80 * GB,
    })
      .kind,
    "squeezed",
  );
  // The false positive that shipped: a machine that simply HAD headroom when
  // the run started. 26 GB free at spawn, 16 GB free after loading 10 GB —
  // nothing moved, nothing "came free", and the note must stay quiet.
  assertEquals(
    drift({
      startedVramB: 10 * GB,
      startedRamB: 4 * GB,
      vramFreeAtStartB: 26 * GB,
      ramFreeAtStartB: 20 * GB,
      vramOverB: 0,
      ramOverB: 0,
      vramFreeB: 16 * GB,
      ramFreeB: 16 * GB,
    }).kind,
    "none",
    "headroom that was always there is not news",
  );
  // No baseline recorded (a restored session): the signal is disabled rather
  // than fired from an invented baseline.
  assertEquals(
    drift({
      ...running,
      vramFreeAtStartB: 0,
      ramFreeAtStartB: 0,
      vramOverB: 0,
      ramOverB: 0,
      vramFreeB: 9 * GB,
      ramFreeB: 9 * GB,
    }).kind,
    "none",
    "no baseline, no roomier",
  );
});

Deno.test("tune: a refusal names the constraint that actually binds", () => {
  // The reported bug, in the message layer. This used to say "does not fit in
  // 47.8 GB of VRAM" — the card's CAPACITY — when the real reason was that only
  // 4.8 GB of it was free. That reads as a claim about the model, sends the user
  // looking for a smaller quantisation, and is not what happened.
  const GB = 1024 ** 3;
  const big = meta({
    name: "35B MoE",
    nLayer: 41,
    nCtxTrain: 262144,
    nHeadKv: 2,
    keyLength: 256,
    valueLength: 256,
    nExpert: 256,
    nExpertUsed: 8,
    layers: layers(41, 880 * 1024 ** 2, 840 * 1024 ** 2),
  });
  // Two 24 GB cards with 21.5 GB each already taken by something else.
  const crowded = hw({
    gpus: [gpu(24, 21.5), gpu(24, 21.5)],
    backend: "cuda",
  });
  const t = tune(big, crowded, defaults(), "vram");
  assertEquals(t.possible, false, "it genuinely does not fit right now");

  // It must name what is available, who has the rest, and stay arithmetically
  // coherent — available + held must not exceed capacity.
  assertStringIncludes(t.blocker, "available of");
  assertStringIncludes(t.blocker, "other processes hold");
  const nums = [...t.blocker.matchAll(/([\d.]+) GB/g)].map((m) => Number(m[1]));
  assert(nums.length >= 4, `expected the numbers: ${t.blocker}`);
  const [, available, capacity, held] = nums;
  assert(
    (available ?? 0) + (held ?? 0) <= (capacity ?? 0) + 0.2,
    `available ${available} + held ${held} must not exceed capacity ${capacity}`,
  );
  // And it must NOT claim the capacity is the constraint.
  assert(
    !t.blocker.includes(`does not fit in ${capacity}`),
    `naming capacity as the limit is the bug: ${t.blocker}`,
  );

  // Give the same machine its memory back and the same model fits — proving the
  // refusal was about occupancy, not about the model.
  const free = hw({ gpus: [gpu(24, 1.2), gpu(24, 1.2)], backend: "cuda" });
  const ok = tune(big, free, defaults(), "vram");
  assertEquals(ok.possible, true, "same model, same cards, memory returned");
  assert(ok.ctx > 0, "and it reaches a real context");
  assert(
    plan(big, free, ok.settings).vram.overB === 0,
    "without overflowing",
  );
  assert(big.nLayer === plan(big, free, ok.settings).layersOnGpu, "all on GPU");
  assert(GB > 0);

  // The CPU refusal must be computed from a CPU plan. It was planned with
  // `ngl: 999` — every weight billed to the VRAM pool — so the message read
  // "Needs 0.0 GB … of system RAM" while refusing for lack of RAM.
  const tinyRam = hw({
    gpus: [],
    mem: {
      totalB: 16 * GB,
      availableB: 12 * GB,
      usedB: 4 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });
  const cpu = tune(big, tinyRam, defaults(), "cpu");
  assertEquals(cpu.possible, false, "a 35B does not fit in 16 GB of RAM");
  const needs = Number(cpu.blocker.match(/Needs ([\d.]+) GB/)?.[1] ?? 0);
  assert(
    needs > 12,
    `the need must reflect the weights actually in RAM: ${cpu.blocker}`,
  );
});

Deno.test("mtp: speculative decoding is taken when the model ships the block", () => {
  // The rare optimisation with nothing to weigh: the full model verifies every
  // drafted token, so a rejected draft is discarded and the output is exactly
  // what it would have been. The weights are in the file and loaded either way,
  // so leaving it off pays for them and gets nothing.
  const mtp = meta({ name: "MTP model", nextnLayers: 1 });
  const plain = meta({ name: "plain model", nextnLayers: 0 });
  const machine = hw({ gpus: [gpu(24, 0.5)], backend: "cuda" });

  const t = tune(mtp, machine, defaults(), "vram");
  assertEquals(str(t.settings, "specType"), "draft-mtp");
  assertStringIncludes(
    t.reasons.join(" "),
    "multi-token-prediction",
    "and it says why",
  );
  assertStringIncludes(
    argv("server", { bin: "x", model: "m.gguf", settings: t.settings }).join(
      " ",
    ),
    "--spec-type draft-mtp",
  );

  // NEVER for a model without the block: llama.cpp asserts on
  // `n_layer_nextn > 0` and refuses to load, so this would be "optimal
  // settings" that do not start.
  const off = tune(plain, machine, defaults(), "vram");
  assertEquals(str(off.settings, "specType"), "");
  assert(
    !argv("server", { bin: "x", model: "m.gguf", settings: off.settings })
      .includes("--spec-type"),
    "the flag must not reach argv for a model that cannot honour it",
  );

  // A cache type left over from an MTP model is not inherited by the next.
  const back = tune(plain, machine, t.settings, "vram");
  assertEquals(str(back.settings, "specType"), "", "reset for the next model");
});

Deno.test("mtp: the draft context is planned for, not discovered at load", () => {
  // llama.cpp reserves "context+compute" for the MTP draft before fitting the
  // target model — the drafting block lives on the target so its weights are
  // already counted, but the second context is real. A plan that ignored it could
  // hand back settings that stop fitting the moment the flag is emitted.
  const m = meta({ name: "MTP", nextnLayers: 1, nCtxTrain: 32768 });
  const machine = hw({ gpus: [gpu(24, 0.5)], backend: "cuda" });
  const base = { ...defaults(), ctxSize: 32768, ngl: 999 };

  const without = plan(m, machine, { ...base, specType: "" });
  const with_ = plan(m, machine, { ...base, specType: "draft-mtp" });
  assert(
    with_.vram.usedB > without.vram.usedB,
    `enabling MTP must cost something: ${with_.vram.usedB} vs ${without.vram.usedB}`,
  );
  // One block's KV over the window, not a second model — so it is small.
  const extra = with_.vram.usedB - without.vram.usedB;
  assert(
    extra < without.vram.usedB * 0.25,
    `the draft context is one block, not a second model: ${extra}`,
  );

  // And a model with no MTP block is unaffected by the flag, because nothing
  // will run: the plan must not invent a cost for a flag that cannot apply.
  const plainM = meta({ name: "plain", nextnLayers: 0, nCtxTrain: 32768 });
  assertEquals(
    plan(plainM, machine, { ...base, specType: "draft-mtp" }).vram.usedB,
    plan(plainM, machine, { ...base, specType: "" }).vram.usedB,
  );

  // A CPU-only MTP run pays for the draft context too — in RAM, where a tight
  // fit is exactly the case that cannot afford an unbilled block of KV.
  const cpuBase = { ...base, ngl: 0 };
  assert(
    plan(m, NO_GPU, { ...cpuBase, specType: "draft-mtp" }).ram.usedB >
      plan(m, NO_GPU, { ...cpuBase, specType: "" }).ram.usedB,
    "the draft KV lands in RAM on a CPU run",
  );
});

// ── speed ────────────────────────────────────────────────────────────────────

Deno.test("speed: a MoE model reads only the experts the router picks", () => {
  const m = moeMeta(); // 8 experts, 2 used: 40 MB dense + 720 MB experts/layer
  const machine = hw({ gpus: [gpu(80)] });
  const s = { ...defaults(), ngl: 999, ctxSize: 4096 };
  const p = plan(m, machine, s);
  const b = bytesPerToken(m, p, s, 0);
  const layerB = 40 * 1024 ** 2 + (720 * 1024 ** 2) * (2 / 8);
  const want = 32 * layerB + m.outputBytes;
  assert(
    Math.abs(b.totalB - want) < 1024,
    `active experts only: ${b.totalB} vs ${want}`,
  );
});

Deno.test("speed: the KV read respects a sliding window", () => {
  // Gemma-3-shaped: window 1024, one global layer per 6. At 128k the uniform
  // rate would overstate the read ~5x — the same error class the fit side
  // fixed, and the tps meter must not resurrect it.
  const m = meta({ swaWindow: 1024, swaPattern: 6, nCtxTrain: 131072 });
  const s = { ...defaults(), ngl: 999, ctxSize: 131072 };
  const machine = hw({ gpus: [gpu(80)] });
  const p = plan(m, machine, s);
  const b = bytesPerToken(m, p, s, 131072);
  const uniform = bytesPerToken(
    meta({ nCtxTrain: 131072 }),
    p,
    s,
    131072,
  );
  const kvWindowed = b.totalB - (32 * 128 * 1024 ** 2 + m.outputBytes);
  const kvUniform = uniform.totalB - (32 * 128 * 1024 ** 2 + m.outputBytes);
  assertEquals(kvWindowed, kvTotal(m, s, 131072));
  assert(
    kvWindowed < kvUniform / 3,
    `windowed layers stop growing: ${kvWindowed} vs uniform ${kvUniform}`,
  );
});

Deno.test("speed: the embedding table is a lookup, not a per-token read", () => {
  const m = meta(); // 300 MB embd, 300 MB output
  const s = { ...defaults(), ngl: 0, ctxSize: 2048 };
  const p = plan(m, NO_GPU, s);
  const b = bytesPerToken(m, p, s, 0);
  // Output head billed, embedding table not.
  assertEquals(b.totalB, 32 * 128 * 1024 ** 2 + m.outputBytes);
});

Deno.test("speed: ends placement agrees with the planner at the boundary", () => {
  const m = meta();
  const machine = hw({ gpus: [gpu(80)] });
  // The output head is the LAST of `nLayer + 1` slots, so it moves to the GPU
  // the moment `-ngl >= 1` — the planner's `slotOnGpu(nLayer, offloadRange)`.
  // The speed estimate must place it the same way or the two halves of the page
  // disagree about the same bytes at every partial offload.
  const at = { ...defaults(), ngl: 32, ctxSize: 2048 }; // ngl == nLayer
  const bAt = bytesPerToken(m, plan(m, machine, at), at, 0);
  assert(bAt.gpuB >= m.outputBytes, `head is on the GPU at ngl==nLayer`);
  const zero = { ...defaults(), ngl: 0, ctxSize: 2048 };
  const bZero = bytesPerToken(m, plan(m, machine, zero), zero, 0);
  assert(
    bZero.ramB >= m.outputBytes,
    "and stays in RAM with the GPU idle",
  );
  // Speed and planner must never disagree about where the head lives, for any
  // `-ngl`. The two halves of the page describe the same placement.
  for (const ngl of [0, 1, 16, 31, 32, 33, 999]) {
    const s = { ...defaults(), ngl, ctxSize: 2048 };
    const p = plan(m, machine, s);
    const layersAttention = p.layersOnGpu * 128 * 1024 ** 2;
    const planHeadOnGpu = (p.vram.buckets.find((b) =>
      b.key === "weights"
    )?.bytes ?? 0) >
      layersAttention;
    const b = bytesPerToken(m, p, s, 0);
    const speedHeadOnGpu = b.gpuB > layersAttention;
    assertEquals(
      speedHeadOnGpu,
      planHeadOnGpu,
      `head placement agrees at ngl=${ngl}`,
    );
    assertEquals(
      slotOnGpu(m.nLayer, offloadRange(m.nLayer, s)),
      planHeadOnGpu,
      `planner's own rule at ngl=${ngl}`,
    );
  }
});

Deno.test("speed: a hostile header cannot poison the estimate", () => {
  const m = meta({
    layers: [
      { i: 0, bytes: Number.NaN, expert: -5 },
      { i: 1, bytes: -1, expert: Number.NaN },
    ],
    nLayer: 2,
    embdBytes: Number.NaN,
    outputBytes: -3,
  });
  const s = { ...defaults(), ngl: 999, ctxSize: 2048 };
  const p = plan(m, hw(), s);
  const b = bytesPerToken(m, p, s, 2048);
  assert(Number.isFinite(b.totalB) && b.totalB >= 0, `clamped: ${b.totalB}`);
  assert(Number.isFinite(estimateTps({ gpuB: b.gpuB, ramB: b.ramB })));
});

Deno.test("speed: calibration only trusts a run that lives in one pool", () => {
  assertEquals(calibrate(50, { gpuB: 99, ramB: 1 }).gpuBps, 100 * 50);
  assertEquals(calibrate(5, { gpuB: 1, ramB: 99 }).ramBps, 100 * 5);
  assertEquals(calibrate(20, { gpuB: 60, ramB: 40 }), {});
  assertEquals(calibrate(0, { gpuB: 100, ramB: 0 }), {});
  assertEquals(calibrate(Number.NaN, { gpuB: 100, ramB: 0 }), {});
});

Deno.test("speed: 'measured' means the dominant pool was calibrated", () => {
  const GBps = 1024 ** 3;
  const gpuRun = { gpuB: 5 * GBps, ramB: 0 };
  const cpuRun = { gpuB: 0, ramB: 5 * GBps };
  // A GPU calibration covers a GPU run, and says nothing about a CPU one.
  assert(speedIsMeasured(gpuRun, 400 * GBps, 0));
  assert(!speedIsMeasured(cpuRun, 400 * GBps, 0));
  assert(speedIsMeasured(cpuRun, 0, 40 * GBps));
  // A hybrid run is dominated by its RAM time even when most BYTES are on the
  // GPU — the honest answer follows the time, not the bytes.
  const hybrid = { gpuB: 8 * GBps, ramB: 2 * GBps };
  assert(!speedIsMeasured(hybrid, 400 * GBps, 0), "RAM time dominates");
  assert(speedIsMeasured(hybrid, 400 * GBps, 40 * GBps));
  assert(!speedIsMeasured({ gpuB: 0, ramB: 0 }, 400 * GBps, 40 * GBps));
});

Deno.test("speed: the bands are anchored on reading speed", () => {
  assertEquals(tpsBand(4.9), "poor");
  assertEquals(tpsBand(5), "ok");
  assertEquals(tpsBand(19.9), "ok");
  assertEquals(tpsBand(20), "great");
  assertEquals(tpsBand(Number.NaN), "poor");
});

// ── which card holds which layer (src/lib/devsplit.ts) ─────────────────────

Deno.test("devsplit: -ngl counts the output head as a slot", () => {
  // llama.cpp offloads the last N of `nLayer + 1` slots. So `-ngl 43` on a
  // 43-layer model is NOT "every layer": it is layers 1..42 plus the output,
  // with layer 0 left on the host. Only `-ngl > nLayer` takes everything.
  const all = offloadRange(43, { ...defaults(), ngl: 999 });
  assertEquals({ start: all.start, count: all.count, slots: all.slots }, {
    start: 0,
    count: 44,
    slots: 44,
  });
  assert(slotOnGpu(0, all) && slotOnGpu(43, all), "layers and the head");

  const exact = offloadRange(43, { ...defaults(), ngl: 43 });
  assertEquals({ start: exact.start, count: exact.count }, {
    start: 1,
    count: 43,
  });
  assert(!slotOnGpu(0, exact), "layer 0 stays on the host at -ngl 43");
  assert(slotOnGpu(43, exact), "and the output head is offloaded anyway");

  const none = offloadRange(43, { ...defaults(), ngl: 0 });
  assertEquals(none.count, 0);
  assert(!slotOnGpu(43, none));
});

Deno.test("devsplit: slots are packed into the cards in order, or not at all", () => {
  // Contiguous and in order is llama.cpp's rule, not a simplification of it:
  // each card gets one run of consecutive layers.
  const cheap = Array.from({ length: 10 }, () => 1);
  assertEquals(packSlots(cheap, [4, 6]), [4, 6]);
  assertEquals(loadPerDevice(cheap, [4, 6]), [4, 6]);

  // The shape `--n-cpu-moe` produces: a long cheap head and a heavy tail.
  const moe = [...Array.from({ length: 8 }, () => 1), 10, 10, 10];
  assertEquals(packSlots(moe, [18, 20]), [9, 2], "card 0 takes the cheap run");
  assertEquals(loadPerDevice(moe, [9, 2]), [18, 20]);

  assertEquals(
    packSlots([30], [20, 20]),
    null,
    "one slot bigger than any card",
  );
  assertEquals(packSlots([1], []), null, "nowhere to put it");
});

Deno.test("plan: the device plan carries a per-card picture, even when packing fails", () => {
  // "I really don't have a good idea what takes what size, why it overflows"
  // — because the map pooled two cards into one VRAM bar. The plan knows the
  // per-card placement; it now says it: weights from the slot packing, KV
  // apportioned by each card's share of the offloaded slots, compute scratch
  // per device — and when NO cut fits, a best-effort fill plus the remainder
  // with nowhere to go, which is the picture that explains the refusal.
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 3400 * MB, 3250 * MB),
  });
  const machine = hw({
    gpus: [gpu(24, 0.5), gpu(24, 0.1)],
    backend: "cuda",
  });

  const fitting = tune(m, machine, defaults(), "hybrid");
  const p = plan(m, machine, fitting.settings);
  assertEquals(p.devices.cards.length, 2, "one entry per card");
  for (const c of p.devices.cards) {
    assert(c.capacityB > 0);
    assert(
      c.otherB + c.weightsB + c.kvB + c.computeB <= c.capacityB + c.overB,
      "a card's parts never silently exceed what it can hold",
    );
  }
  if (p.devices.fits) {
    assertEquals(p.devices.unplacedB, 0);
  }

  // Force an impossible placement: everything on the GPUs, experts included,
  // at a long context. The picture must still exist — cards filled as far as
  // they go, the rest reported as unplaced rather than drawn nowhere.
  const doomed = plan(m, machine, {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 0,
    ctxSize: 131_072,
  });
  if (!doomed.devices.fits) {
    assert(
      doomed.devices.unplacedB > 0,
      `the bytes with nowhere to go are named: ${doomed.devices.unplacedB}`,
    );
    assert(
      doomed.notes.some((n) => n.includes("no card")),
      `and the note says so: ${JSON.stringify(doomed.notes)}`,
    );
  }
});

Deno.test("devsplit: -ts boundaries land between slots, never on one", () => {
  // Emitting the counts themselves puts the boundary exactly on `k / n`, where
  // llama.cpp's `upper_bound` decides on the equality of two floats it computed
  // separately. Midpoints cannot be off by one card.
  assertEquals(tensorSplitValue([37, 7]), "36.5,7.5");
  assertEquals(tensorSplitValue([10, 20, 14]), "9.5,20,14.5");
  assertEquals(tensorSplitValue([44]), "", "one card needs no split");
  assertEquals(tensorSplitValue([44, 0]), "", "nor does one card doing it all");
});

Deno.test("devsplit: the split we emit is the split we drew", () => {
  // `tensorSplitValue` and `countsFromSplit` are the two halves of one claim,
  // and the round trip is the claim: if the string llama.cpp receives does not
  // reproduce the cuts the packer chose, the app is drawing per-card bars for
  // a placement that is not going to happen.
  for (const counts of [[37, 7], [10, 20, 14], [34, 10], [1, 43], [22, 22]]) {
    const total = counts.reduce((a, b) => a + b, 0);
    assertEquals(
      countsFromSplit(tensorSplitValue(counts), total, counts.length),
      counts,
      `round trip for ${JSON.stringify(counts)}`,
    );
  }
  // llama.cpp's rule run forwards, on a split a USER might type: 90/10 of 44
  // slots is 40 and 4, by count and not by bytes.
  assertEquals(countsFromSplit("9,1", 44, 2), [40, 4]);
  assertEquals(countsFromSplit("1,1", 44, 2), [22, 22]);
  // Nothing usable is `null`, so the caller falls back to packing — which is
  // also what llama.cpp does with a split it cannot read.
  assertEquals(countsFromSplit("", 44, 2), null);
  assertEquals(countsFromSplit("0,0", 44, 2), null);
  assertEquals(countsFromSplit("nonsense", 44, 2), null);
  assertEquals(countsFromSplit("-1,2", 44, 2), null);
});

Deno.test("plan: server slots multiply the scratch, and the plan knows it", () => {
  // The bug that made a 1,048,576 context impossible on hardware that runs it.
  // llama.cpp's `-np` default is `-1 = auto`, auto chose FOUR slots, each slot
  // costs another copy of the context-sized indexer tensors, and the app knew
  // about none of it: measured 43,517 MiB of VRAM at four slots against 21,467
  // at one, same model, same placement, same context.
  const MB = 1024 ** 2;
  const m = meta({
    nLayer: 43,
    nCtxTrain: 1_048_576,
    indexerTopK: 512,
    swaWindow: 128,
    swaPattern: 1,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 2240 * MB, 2100 * MB),
  });
  const machine = hw({ gpus: [gpu(24, 0.1), gpu(24, 0.1)], backend: "cuda" });
  const one = plan(m, machine, {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 43,
    ctxSize: 1_048_576,
    parallel: 1,
  });
  const four = plan(m, machine, {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 43,
    ctxSize: 1_048_576,
    parallel: 4,
  });
  const scratchOf = (p: typeof one) =>
    p.vram.buckets.find((b) => b.key === "compute")?.bytes ?? 0;
  assert(
    scratchOf(four) > scratchOf(one) * 3,
    `four slots must cost far more: ${scratchOf(one)} vs ${scratchOf(four)}`,
  );
  // And the placement that measured healthy at one slot is offered at one slot.
  assertEquals(one.devices.fits, true, JSON.stringify(one.devices.bytesB));
});

Deno.test("plan: the sparse-attention scratch has a floor, not just a slope", () => {
  // A 65,536 context is worth ~0.5 GB of context-scaled scratch, and llama.cpp
  // asked for 1,985 MiB on one card and died in graph_reserve. Measured at
  // 4,096 tokens — where the slope is worth ~30 MB and everything left is the
  // flat part — it is about 2 GB PER DEVICE: 2,017 MiB on one card, 3,271 MiB
  // spread over two. The app allowed 384 MB.
  const MB = 1024 ** 2;
  const GB = 1024 ** 3;
  const sparse = meta({
    nLayer: 43,
    nCtxTrain: 1_048_576,
    indexerTopK: 512,
    swaWindow: 128,
    swaPattern: 1,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 2240 * MB, 2100 * MB),
  });
  const machine = hw({ gpus: [gpu(24, 0.1), gpu(24, 0.1)], backend: "cuda" });
  const at4k = plan(sparse, machine, {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 43,
    ctxSize: 4096,
    parallel: 1,
  });
  const perCard = at4k.devices.cards[0]?.computeB ?? 0;
  assert(
    perCard > 1.9 * GB,
    `a short context still costs the floor: ${(perCard / GB).toFixed(2)} GB`,
  );
  // A model with no indexer keeps the small, ordinary compute buffer — this is
  // the sparse-attention working set, not a tax on everybody.
  const dense = plan(meta({ layers: layers(32, 200 * MB) }), machine, {
    ...defaults(),
    ngl: 999,
    ctxSize: 4096,
  });
  assert(
    (dense.devices.cards[0]?.computeB ?? 0) < GB,
    "an ordinary model pays a few hundred MB, as it always did",
  );
  assertEquals(scratchFloor(meta({ indexerTopK: 0 })), 0);
});

Deno.test("plan: a card is budgeted for the whole scratch, not its share", () => {
  // Divided between the cards — measured, one card and two cost the same total
  // — but llama.cpp decides WHERE the division falls and it is not by layer
  // count: a card holding 10 slots of 44 wanted about 60% of it, and a plan
  // that had budgeted it 23% died in `graph_reserve`. So each card is charged
  // all of it. The two views answer different questions and both are honest:
  // the pool says what the machine will use, a card says what it must be able
  // to give.
  const MB = 1024 ** 2;
  const m = meta({
    nLayer: 43,
    nCtxTrain: 1_048_576,
    indexerTopK: 512,
    swaWindow: 128,
    swaPattern: 1,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 2240 * MB, 2100 * MB),
  });
  const machine = hw({ gpus: [gpu(24, 0.1), gpu(24, 0.1)], backend: "cuda" });
  // Experts kept on the GPU, so the layers genuinely span both cards.
  const p = plan(m, machine, {
    ...defaults(),
    ngl: 999,
    nCpuMoe: 20,
    ctxSize: 131_072,
    parallel: 1,
  });
  const poolScratch = p.vram.buckets.find((b) => b.key === "compute")?.bytes ??
    0;
  const working = p.devices.cards.filter((c) => c.weightsB > 0);
  assert(working.length >= 2, "this fixture must span both cards");
  assertEquals(
    working[0]!.computeB,
    working[1]!.computeB,
    "every working card is charged the same worst case",
  );
  assert(
    working[0]!.computeB > poolScratch / 2,
    "and it is the whole scratch, not half of it",
  );
});

Deno.test("plan: per-card bytes never double-count the KV cache", () => {
  // The per-card picture drew the KV twice. `slotCostsB` includes `kvPerLayerB`
  // so the PACKER places the cache with its layers, but `loadPerDevice` over it
  // also fed `card.weightsB` — and the card then added its own `kvB` on top.
  // card0 + card1 exceeded the pool's "VRAM used" for the same plan by exactly
  // the cache, so the two views disagreed about the identical machine.
  //
  // The weights are now re-derived from a KV-free array, so weights and KV
  // reconcile with the pool to the byte; only the compute overhead is
  // deliberately more on the pool (billed per device) than on the cards (which
  // show 0 on an idle card), and that is the safe direction.
  const MB = 1024 ** 2;
  const m = meta({ nLayer: 40, layers: layers(40, 900 * MB) });
  const machine = hw({ gpus: [gpu(24, 0.5), gpu(24, 0.5)], backend: "cuda" });
  const p = plan(m, machine, { ...defaults(), ngl: 999, ctxSize: 16384 });

  const poolKv = p.vram.buckets.find((b) => b.key === "kv")?.bytes ?? 0;
  const cardKv = p.devices.cards.reduce((a, c) => a + c.kvB, 0);
  assert(
    Math.abs(poolKv - cardKv) < 1,
    `pool KV ${poolKv} must equal the cards' ${cardKv} — they were doubled before`,
  );

  const poolWeights =
    (p.vram.buckets.find((b) => b.key === "weights")?.bytes ?? 0) +
    (p.vram.buckets.find((b) => b.key === "experts")?.bytes ?? 0);
  const cardWeights = p.devices.cards.reduce((a, c) => a + c.weightsB, 0);
  assert(
    Math.abs(poolWeights - cardWeights) < 1,
    `pool weights ${poolWeights} must equal the cards' ${cardWeights}`,
  );

  // And the card is not credited with more room than it has: its own free
  // figure never exceeds its capacity minus what it is asked to hold.
  for (const c of p.devices.cards) {
    assert(
      c.weightsB + c.kvB + c.computeB + c.otherB + c.reservedB + c.overB <=
        c.capacityB + 1,
      `${c.name} claims more than it holds`,
    );
  }
});

Deno.test("plan: a hand-typed -ts is the placement, not a suggestion", () => {
  // Until `plan` read it, a user who typed a split saw the PACKER's cuts on
  // screen however far the argv disagreed — on the one page whose promise is
  // that what you see is what runs.
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nExpert: 256,
    nExpertUsed: 6,
    // `bytes` is the whole layer and `expert` the routed part of it, so this
    // is 800 MB a layer of which 700 MB is experts — the MoE shape.
    layers: layers(43, 800 * MB, 700 * MB),
  });
  const machine = hw({ gpus: [gpu(24, 0), gpu(24, 0)], backend: "cuda" });
  const base = { ...defaults(), ngl: 999, ctxSize: 4096 }; // 44 slots, ~35 GB
  const packed = plan(m, machine, base);
  // Everything on card 0 is not what the packer would choose, so it is a clean
  // way to see whether the instruction is being obeyed.
  const pinned = plan(m, machine, { ...base, tensorSplit: "1,0" });
  assertEquals(
    pinned.devices.bytesB[1],
    0,
    `card 1 was told to hold nothing; got ${
      JSON.stringify(pinned.devices.bytesB)
    }`,
  );
  assert(
    pinned.devices.bytesB[0]! > packed.devices.bytesB[0]!,
    "and card 0 holds what card 1 was spared",
  );
  // Including when that is a bad idea: an instruction that overfills a card is
  // reported as overfilling it, not quietly re-packed into one that fits.
  assert(
    (pinned.devices.cards[0]?.overB ?? 0) > 0,
    "the picture shows the card being asked for more than it has",
  );
  // And the split it reports back is the one that produces what it drew. From
  // the counts alone this would be "" — one card doing all the work needs no
  // split of OURS — and "" makes llama.cpp divide by free VRAM instead, which
  // is the opposite of what was asked for.
  assertEquals(pinned.devices.tensorSplit, "1,0");
  // And a pin that does not hold is reported as not holding. `fits` used to
  // mean "the packer found an arrangement", and a pin always produces one —
  // so every hand-typed split fitted by definition, including this one.
  assertEquals(pinned.devices.fits, false);
});

Deno.test("tune: an expert-heavy tail is split by bytes, not by layer count", () => {
  // The failure this pins, from a real run: DeepSeek-V4-Flash on two 24 GB
  // cards. `--n-cpu-moe 34` holds the first 34 layers' experts in RAM, so the
  // last 9 layers are ~3.2 GB each and the first 34 are ~0.15 GB each. Split by
  // COUNT — which is what llama.cpp does, using free VRAM only to choose the
  // proportions — the heavy tail lands on one card:
  //
  //   allocating 34020.32 MiB on device 1: cudaMalloc failed: out of memory
  //
  // The totals were never the problem: 38 GB of plan, 42 GB of free VRAM.
  const MB = 1024 * 1024;
  const m = meta({
    name: "Expert-heavy",
    nLayer: 43,
    nCtxTrain: 32768,
    nHeadKv: 1,
    keyLength: 512,
    valueLength: 512,
    swaWindow: 128,
    swaPattern: 1,
    nExpert: 256,
    nExpertUsed: 6,
    embdBytes: 1000 * MB,
    outputBytes: 1000 * MB,
    layers: layers(43, 3400 * MB, 3250 * MB),
  });
  const machine = hw({
    gpus: [gpu(24, 5.6), gpu(24, 0.1)],
    backend: "cuda",
    mem: {
      totalB: 186 * GB,
      availableB: 167 * GB,
      usedB: 19 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });

  const t = tune(m, machine, defaults(), "hybrid");
  assertEquals(t.possible, true);

  const p = plan(m, machine, t.settings);
  assert(
    p.devices.fits,
    "a division of the layers exists that both cards hold",
  );
  assertEquals(p.vram.overB, 0);
  for (let d = 0; d < p.devices.bytesB.length; d++) {
    assert(
      (p.devices.bytesB[d] ?? 0) <= (p.devices.budgetsB[d] ?? 0),
      `card ${d} is asked for ${p.devices.bytesB[d]} of ${
        p.devices.budgetsB[d]
      }`,
    );
  }
  // And the split is actually emitted, so llama.cpp does not have to guess.
  assert(
    String(t.settings.tensorSplit).includes(","),
    `a two-card placement must pin -ts, got ${
      JSON.stringify(t.settings.tensorSplit)
    }`,
  );
  assertEquals(
    argv("server", { bin: "s", model: "m", settings: t.settings }).includes(
      "-ts",
    ),
    true,
  );

  // The counterfactual: llama.cpp's own default, which divides the SLOT COUNT
  // in proportion to each card's free VRAM, overloads one card at this exact
  // placement. That is the whole reason an explicit split is emitted.
  const off = offloadRange(m.nLayer, t.settings);
  const free = machine.gpus.map((g) => g.vramTotalB - g.vramUsedB);
  const total = free.reduce((a, b) => a + b, 0);
  const firstCard = Math.round(off.count * (free[0] ?? 0) / total);
  const byCount = loadPerDevice(
    // Re-derive the per-slot costs the same way the plan does.
    Array.from({ length: off.count }, (_, i) => {
      const slot = off.start + i;
      if (slot >= m.nLayer) return m.outputBytes;
      const l = m.layers[slot]!;
      const expert = slot < Number(t.settings.nCpuMoe) ? 0 : l.expert;
      return l.bytes - l.expert + expert;
    }),
    [firstCard, off.count - firstCard],
  );
  assert(
    (byCount[1] ?? 0) > (machine.gpus[1]?.vramTotalB ?? 0),
    `the default split must overflow card 1 — that is the bug: ${
      JSON.stringify(byCount)
    }`,
  );
});

Deno.test("tune: --mlock is only promised when the kernel would honour it", () => {
  // The report, from the app's own running state: a CPU placement of a 145 GB
  // model came out with `mlock: true` and the reason "pinning them stops the OS
  // paging the model out mid-generation". Stock RLIMIT_MEMLOCK on that machine
  // is 23.3 GB. llama.cpp warns and runs UNPINNED — so the app was describing
  // something that did not happen, which is the one thing it must never do.
  //
  // The partial-offload branch already refused `--mlock`, but a CPU-only
  // placement has `--n-cpu-moe 0` and fell straight past it.
  const MB = 1024 * 1024;
  const big = meta({
    nLayer: 43,
    nExpert: 256,
    nExpertUsed: 6,
    swaWindow: 128,
    layers: layers(43, 2400 * MB, 2250 * MB), // ~100 GB of model
  });
  const roomy = {
    totalB: 186 * GB,
    availableB: 167 * GB,
    usedB: 19 * GB,
    swapTotalB: 0,
    swapUsedB: 0,
    lockableB: 23.3 * GB, // what this machine actually allows
  };
  const cpu = tune(big, hw({ gpus: [], mem: roomy }), defaults(), "cpu");
  assertEquals(cpu.settings.mlock, false, "100 GB cannot be pinned under 23");
  assert(
    cpu.reasons.some((r) => r.includes("RLIMIT_MEMLOCK")),
    `and it names the limit rather than going quiet: ${
      cpu.reasons.join(" | ")
    }`,
  );

  // A model that DOES fit the limit still gets it — the flag is not banned,
  // it is only promised when it means something.
  const small = meta({ nLayer: 32, layers: layers(32, 200 * MB) });
  const fits = tune(small, hw({ gpus: [], mem: roomy }), defaults(), "cpu");
  assertEquals(fits.settings.mlock, true, "6 GB under a 23 GB limit is fine");

  // And a machine that does not report a limit is never promised anything:
  // "unknown" is a reason to stay quiet, not to assume the best.
  const silent = tune(
    small,
    hw({ gpus: [], mem: { ...roomy, lockableB: undefined } }),
    defaults(),
    "cpu",
  );
  assertEquals(silent.settings.mlock, false);
});

Deno.test("tune: never emits two flags that are the same llama.cpp setting", () => {
  // `--mlock` and `--no-mmap` both assign `params.load_mode` (`common/arg.cpp`),
  // so passing both is not "locked and unmapped" — it is whichever came last,
  // silently. The app would then print a reason claiming --mlock while shipping
  // an argv that cancelled it. Checked across the shapes that reach the branch:
  // a MoE with experts on the host, a dense model, and a machine with no room.
  const MB = 1024 * 1024;
  const roomy = {
    totalB: 186 * GB,
    availableB: 167 * GB,
    usedB: 19 * GB,
    swapTotalB: 0,
    swapUsedB: 0,
  };
  const cases: [string, ModelMeta, Hw][] = [
    ["dense, roomy", meta(), hw({ gpus: [gpu(24, 0.5)], mem: roomy })],
    [
      "MoE on two cards",
      meta({
        nLayer: 43,
        nExpert: 256,
        nExpertUsed: 6,
        swaWindow: 128,
        layers: layers(43, 3400 * MB, 3250 * MB),
      }),
      hw({ gpus: [gpu(24, 5.6), gpu(24, 0.1)], mem: roomy, backend: "cuda" }),
    ],
    [
      "MoE, tight RAM",
      moeMeta(),
      hw({
        gpus: [gpu(8, 0.5)],
        mem: {
          totalB: 32 * GB,
          availableB: 20 * GB,
          usedB: 12 * GB,
          swapTotalB: 0,
          swapUsedB: 0,
        },
      }),
    ],
  ];
  for (const [name, m, machine] of cases) {
    for (const p of ["vram", "hybrid", "cpu"] as const) {
      const { settings } = tune(m, machine, defaults(), p);
      const cmd = argv("server", { bin: "s", model: "m", settings });
      assert(
        !(cmd.includes("--mlock") && cmd.includes("--no-mmap")),
        `${name} / ${p}: both flags emitted — ${cmd.join(" ")}`,
      );
      // Routed experts on the host get llama.cpp's mmap default, and NO flag:
      // --no-mmap re-copied the whole file on every start (160 s where mmap
      // takes 6 warm, and 8.9 tok/s where mmap generates 9.6), and --mlock
      // would ask to pin more than stock memlock limits allow, so its stated
      // effect would not happen. Measured on the 145 GB DeepSeek-V4.
      if (Number(settings.nCpuMoe ?? 0) > 0) {
        assert(
          !cmd.includes("--no-mmap") && !cmd.includes("--mlock"),
          `${name} / ${p}: experts on host must stay memory-mapped — ${
            cmd.join(" ")
          }`,
        );
      }
    }
  }
});

// ── measuring what cannot be predicted (src/lib/fitladder.ts) ──────────────

/** Captured verbatim on 2×24 GB, `--n-cpu-moe 29`, when the desktop had grown
 *  from 2 GB to 5.5 GB of VRAM between the tune and the start. Every rung of the
 *  old ladder failed at exactly this same allocation. */
const WEIGHTS_OOM = [
  "0.02.438.752 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 34679.64 MiB on device 1: cudaMalloc failed: out of memory",
  "0.02.438.758 E alloc_tensor_range: failed to allocate CUDA1 buffer of size 36364237312",
  "0.03.121.689 E llama_model_load: error loading model: unable to allocate CUDA1 buffer",
];

Deno.test("fitladder: the model not fitting is a different fault from the cache not fitting", () => {
  assertEquals(fitFault(WEIGHTS_OOM), "weights");
  assertEquals(
    fitFault([
      "E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 71912704256",
      "E graph_reserve: failed to allocate compute buffers",
    ]),
    "context",
  );
  assertEquals(
    fitFault([
      "/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
      "E CUDA error: out of memory",
    ]),
    "context",
    "the lazy generation-time pool scales with the context",
  );
  assertEquals(fitFault(["E unknown argument: --nope"]), null);
});

Deno.test("fitladder: llama.cpp's own numbers are read back out of the log", () => {
  // The exact byte count wins over the rounded MiB when both are present.
  assertEquals(requestedB(WEIGHTS_OOM), 36364237312);
  assertEquals(
    requestedB([
      "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 34679.64 MiB on device 1: cudaMalloc failed: out of memory",
    ]),
    Math.round(34679.64 * 1024 * 1024),
  );
  assertEquals(requestedB(["nothing to see"]), 0);
});

Deno.test("fitladder: a weights overflow moves experts, never the context", () => {
  // The card had 22 GB to give and llama.cpp asked for 33.9 GiB — 11.9 GB short.
  // At 2.5 GB of experts per layer that is 5 layers, +1 for the scratch that
  // still has to live there.
  const GiB = 1024 ** 3;
  const d = fitDecision({
    lines: WEIGHTS_OOM,
    ctx: 65536,
    tries: 0,
    auto: true,
    nCpuMoe: 29,
    nLayer: 43,
    expertPerLayerB: 2.5 * GiB,
    deviceFreeB: [18 * GiB, 22 * GiB],
  });
  assertEquals(d.kind, "offload", "halving 65,536 would not move one byte");
  if (d.kind !== "offload") return;
  assertEquals(d.nCpuMoe, 35);
  assert(d.note.includes("11.9 GB"), d.note);

  // The shortfall is the point: taken at face value, 33.9 GiB of experts is
  // 14 layers and most of the GPU given away for nothing.
  assert(
    movedLayers(33.9 * GiB, 2.5 * GiB) > 14,
    "sized from the request, this would be the step",
  );
});

Deno.test("fitladder: with nothing left to move, a weights overflow still shrinks", () => {
  const GiB = 1024 ** 3;
  const dense = fitDecision({
    lines: WEIGHTS_OOM,
    ctx: 65536,
    tries: 0,
    auto: true,
    nLayer: 43,
    expertPerLayerB: 0, // dense: `--n-cpu-moe` has nothing to hold back
    deviceFreeB: [18 * GiB, 22 * GiB],
  });
  assertEquals(dense.kind, "retry");
  const exhausted = fitDecision({
    lines: WEIGHTS_OOM,
    ctx: 65536,
    tries: 0,
    auto: true,
    nCpuMoe: 43,
    nLayer: 43, // every layer already on the host
    expertPerLayerB: 2.5 * GiB,
    deviceFreeB: [18 * GiB, 22 * GiB],
  });
  assertEquals(exhausted.kind, "retry");
});

Deno.test("fitladder: the offload rung drops the split that just failed", () => {
  const argv = [
    "llama-server",
    "-m",
    "x.gguf",
    "-ngl",
    "999",
    "--n-cpu-moe",
    "29",
    "-ts",
    "33.5,10.5",
    "-c",
    "65536",
  ];
  assertEquals(nCpuMoeOf(argv), 29);
  const next = withNCpuMoe(argv, 35);
  assertEquals(next.includes("-ts"), false, "it pinned the failed placement");
  assertEquals(next.includes("33.5,10.5"), false);
  assertEquals(next[next.indexOf("--n-cpu-moe") + 1], "35");
  assertEquals(
    next[next.indexOf("-c") + 1],
    "65536",
    "one number changes, and it is not this one",
  );
  // A model that was not given the flag still gets it.
  assertEquals(nCpuMoeOf(["llama-server", "-m", "x.gguf"]), 0);
  assert(withNCpuMoe(["llama-server"], 4).join(" ").endsWith("--n-cpu-moe 4"));
});

Deno.test("fitladder: only an allocation failure is worth retrying smaller", () => {
  assert(isFitFailure([
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18432.00 MiB on device 0: cudaMalloc failed: out of memory",
    "E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache",
  ]));
  assert(isFitFailure([
    "E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 71912704256",
    "E graph_reserve: failed to allocate compute buffers",
  ]));
  // A shorter context does not fix any of these, and four slow retries that end
  // in the same error are worse than the error.
  assert(!isFitFailure(["E error loading model: no such file"]));
  assert(!isFitFailure(["E unknown argument: --nope"]));
  assert(!isFitFailure([]));
});

Deno.test("fitladder: a generation-time CUDA pool OOM is a fit failure", () => {
  // Captured live from DeepSeek-V4 on 2×24 GB: the server passed /health, then
  // the first prompt OOM'd allocating compute scratch. Note the word "buffer"
  // appears nowhere — requiring it made the ladder blind to exactly the
  // failure it exists for, and the run was reported instead of retried.
  assert(isFitFailure([
    "/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
    "2.17.177.475 E CUDA error: out of memory",
    "2.17.177.483 E   current device: 0, in function ggml_cuda_kernel_can_use_pdl at /src/ggml-cuda/common.cuh:1622",
    "2.17.177.484 E   cudaFuncGetAttributes(&attr, kernel)",
  ]));
  // A CUDA error that is NOT an allocation failure still is not one: a smaller
  // context does not fix a rejected device.
  assert(
    !isFitFailure([
      "E CUDA error: no CUDA-capable device is detected",
    ]),
  );
});

Deno.test("fitladder: the ladder reaches the context this model actually needs", () => {
  // Measured on a 2x24 GB machine: DeepSeek-V4-Flash declares 1,048,576, and
  // 64k still failed to allocate its compute buffer. 32,768 started. A ladder
  // that gave up at 64k would have stopped one rung short of the answer.
  let ctx = 1_048_576;
  const rungs: number[] = [];
  for (let tries = 0; tries < MAX_FIT_RETRIES; tries++) {
    const d = fitDecision({
      lines: [
        "E cudaMalloc failed: out of memory",
        "E failed to allocate compute buffers",
      ],
      ctx,
      tries,
      auto: true,
    });
    if (d.kind !== "retry") break;
    ctx = d.ctx;
    rungs.push(ctx);
  }
  assertEquals(rungs, [524_288, 262_144, 131_072, 65_536, 32_768, 16_384]);
  assert(rungs.includes(32_768), "the measured working context is reachable");
});

Deno.test("fitladder: a context the user chose is never silently halved", () => {
  const lines = [
    "E cudaMalloc failed: out of memory",
    "E failed to allocate buffer for kv cache",
  ];
  assertEquals(
    fitDecision({ lines, ctx: 131_072, tries: 0, auto: false }).kind,
    "none",
    "auto off — the number is an instruction, not a suggestion",
  );
  assertEquals(
    fitDecision({ lines, ctx: 131_072, tries: MAX_FIT_RETRIES, auto: true })
      .kind,
    "none",
    "and the ladder is bounded",
  );
});

Deno.test("fitladder: the retry is the same command, one number smaller", () => {
  // Re-composing from settings would pick up anything edited since Start; this
  // rewrites what actually ran, so the preview stays honest.
  const cmd = [
    "/bin/llama-server",
    "-m",
    "/m.gguf",
    "-ngl",
    "999",
    "--n-cpu-moe",
    "33",
    "-ts",
    "36.5,7.5",
    "-c",
    "1048576",
    "-fa",
    "on",
    "--port",
    "8080",
  ];
  assertEquals(ctxOf(cmd), 1_048_576);
  const next = withCtx(cmd, 32_768);
  assertEquals(ctxOf(next), 32_768);
  assertEquals(next.length, cmd.length, "nothing else moved");
  assertEquals(
    next.filter((_, i) => cmd[i] !== next[i]),
    ["32768"],
    "exactly one token differs",
  );
  // No -c at all means llama.cpp would use the model's full trained length —
  // precisely the value that just failed — so one is added.
  assertEquals(ctxOf(withCtx(["s", "-m", "x"], 4096)), 4096);
});

Deno.test("fitladder: a measured context is a ceiling, not a target", () => {
  assertEquals(
    openingCtx(32_768, 1_048_576),
    32_768,
    "measured wins over hoped",
  );
  assertEquals(
    openingCtx(32_768, 8_192),
    8_192,
    "but never raises a smaller aim",
  );
  assertEquals(openingCtx(0, 1_048_576), 1_048_576, "never measured: aim high");
});

Deno.test("loadprogress: the bar is measured, labelled, and never overshoots", () => {
  const GB = 1024 ** 3;
  // Mid-load: 20 GB has landed on the cards, 30 GB is resident on the host,
  // against a 100 GB plan — half way, and the log says which phase.
  const p = loadProgress({
    lines: [
      "llama_model_loader: loaded meta data",
      "load_tensors: loading model tensors, this can take a while...",
    ],
    startFreeVramB: 44 * GB,
    freeVramB: 24 * GB,
    rssB: 30 * GB,
    plannedB: 100 * GB,
  });
  assertEquals(p.loadedB, 50 * GB);
  assertEquals(p.fraction, 0.5);
  assertEquals(p.phase, "loading weights");

  // More measured than planned — the plan is an estimate — must clamp, not
  // render a 130% bar.
  const over = loadProgress({
    lines: [],
    startFreeVramB: 44 * GB,
    freeVramB: 4 * GB,
    rssB: 90 * GB,
    plannedB: 100 * GB,
  });
  assertEquals(over.fraction, 1);

  // No plan to compare against: no fraction, never NaN.
  const unknown = loadProgress({
    lines: [],
    startFreeVramB: 0,
    freeVramB: 0,
    rssB: 5 * GB,
    plannedB: 0,
  });
  assertEquals(unknown.fraction, null);
  assertEquals(unknown.loadedB, 5 * GB);
});

Deno.test("loadprogress: the newest log line names the phase", () => {
  assertEquals(loadPhase([]), "starting");
  assertEquals(
    loadPhase([
      "load_tensors: loading model tensors",
      "llama_kv_cache: size 1024.00 MiB",
    ]),
    "allocating the KV cache",
    "the LATER line wins — the load has moved on",
  );
  assertEquals(
    loadPhase(["main: warming up the model with an empty run"]),
    "warming up",
  );
  assertEquals(elapsedLabel(95_000), "1:35");
  assertEquals(elapsedLabel(4_000), "0:04");
});

Deno.test("tune: aimFull hunts to the advertised maximum, not the native aim", () => {
  // "Max on VRAM / Max on Hybrid": the user has said context is the priority,
  // so the search ceiling is the model's advertised edge — past the
  // native-first automatic aim, past the measured ceiling. Memory still has
  // the only vote on where it stops.
  const m = meta({ nCtxTrain: 1_048_576, nCtxOrig: 65_536 });
  const machine = hw({ gpus: [gpu(80, 0)] });
  const auto = tune(m, machine, defaults(), "vram");
  const full = tune(m, machine, defaults(), "vram", undefined, undefined, true);
  assert(
    Number(auto.settings.ctxSize) <= 65_536,
    `auto aims native: ${auto.settings.ctxSize}`,
  );
  assert(
    Number(full.settings.ctxSize) > Number(auto.settings.ctxSize),
    `full aim goes past it when memory allows: ${full.settings.ctxSize}`,
  );
  assert(Number(full.settings.ctxSize) <= 1_048_576, "never past advertised");

  // And the measured ceiling does not cap the hunt — it caps the AUTO path.
  const measured = tune(m, machine, defaults(), "vram", undefined, 32_768);
  assert(Number(measured.settings.ctxSize) <= 32_768);
  const fullPastMeasured = tune(
    m,
    machine,
    defaults(),
    "vram",
    undefined,
    32_768,
    true,
  );
  assert(
    Number(fullPastMeasured.settings.ctxSize) >
      Number(measured.settings.ctxSize),
    "aimFull is the explicit way past the measured ceiling",
  );
});

Deno.test("tune: aimFull drops the residency anchor — context outranks speed", () => {
  // The report: a pinned 262,144 ran fine on hybrid while Max·Hybrid offered
  // 17,920. Same machine, two answers — because the automatic search refuses
  // to push weights back to the host to buy context, and the pin skipped that
  // rule while the button did not. Pressing "Max on Hybrid" IS choosing that
  // trade, so the full-aim hunt only requires placeability, like the pin.
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nCtxTrain: 262_144,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 100 * MB, 700 * MB),
  });
  // A card small enough that the anchor BINDS: the attention weights nearly
  // fill it, so any real context must either stop early (the automatic rule)
  // or push weights back to the host (the trade this button exists to make).
  const machine = hw({
    gpus: [gpu(6, 0.5)],
    backend: "cuda",
  });

  const auto = tune(m, machine, defaults(), "hybrid");
  const full = tune(
    m,
    machine,
    defaults(),
    "hybrid",
    undefined,
    undefined,
    true,
  );
  assert(auto.possible && full.possible);
  assert(
    full.ctx > auto.ctx,
    `the full hunt goes past the residency-bound ${auto.ctx}: got ${full.ctx}`,
  );
  assert(
    full.reasons.some((r) => r.includes("Context first")),
    "and says which trade was made",
  );
});

Deno.test("tune: the residency anchor has slack, because a step function trades badly", () => {
  // Measured (DeepSeek-V4-Flash IQ3_XXS, 2x24 GB): one layer of routed experts
  // on the host costs ~2% of the generation rate — 15.7 tok/s at 28 layers
  // held back, 14.0 at 30, 13.2 at 32. A STRICT anchor spent that 2% defending
  // a context three times shorter, which nobody would choose and the app was
  // choosing silently. The slack is bounded so it cannot become the runaway it
  // replaced: `aimFull` is still the only way to buy context without limit.
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nCtxTrain: 262_144,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 100 * MB, 700 * MB),
  });
  const machine = hw({ gpus: [gpu(24, 0.5), gpu(24, 0)], backend: "cuda" });
  const auto = tune(m, machine, defaults(), "hybrid");
  const full = tune(
    m,
    machine,
    defaults(),
    "hybrid",
    undefined,
    undefined,
    true,
  );
  assert(auto.possible, "a 43-layer MoE on 48 GB of cards is a hybrid run");
  // The slack buys context, and it is still an anchor: the unbounded hunt must
  // reach further than it, or the bound is not doing its job.
  assert(
    auto.ctx > 0 && full.ctx >= auto.ctx,
    `slack ${auto.ctx} must not exceed the unbounded ${full.ctx}`,
  );
  // And the trade stays bounded: whatever context the anchor allows, the
  // weights it leaves on the GPU are within a few percent of the floor's.
  const hostAt = (ctx: number) => {
    const t = tune(m, machine, { ...defaults(), ctxSize: ctx }, "hybrid");
    return plan(m, machine, t.settings).ram.buckets
      .filter((b) => b.key === "weights").reduce((a, b) => a + b.bytes, 0);
  };
  const floorHost = hostAt(2048);
  assert(
    hostAt(auto.ctx) <= floorHost * 1.06,
    "the anchor is 5% of host weights, not a licence to move the model",
  );
});

// ── the memory the user keeps for themselves ───────────────────────────────
//
// A reserve is not a safety margin. The margins in `tune.ts` and `devsplit.ts`
// exist so the allocator does not fail and the user never sees them; this one is
// the user saying "that card also draws my desktop". So the test of it is not
// "did a warning appear" — it is that the plan is SMALLER, everywhere, by
// exactly what was asked for.

Deno.test("reserve: per-GPU is charged to every card, connected only to the one with a screen", () => {
  const r = (perGpuB: number, connectedB: number) => ({
    perGpuB,
    connectedB,
    ramB: 0,
  });
  // The case the feature exists for: a display card and a compute card. The
  // 8 GB defends the desktop and costs the headless card nothing — the old
  // proportional split took 6 GB off a card nobody was drawing on.
  const mixed = [gpu(24, 0.5, true), gpu(24, 0.5, false)];
  assertEquals(vramReserveShares(mixed, r(0, 8 * GB)), [8 * GB, 0]);
  // Per-GPU is the other claim, and it IS charged to everything.
  assertEquals(vramReserveShares(mixed, r(2 * GB, 0)), [2 * GB, 2 * GB]);
  // Both set: they add on the card that qualifies for both.
  assertEquals(vramReserveShares(mixed, r(2 * GB, 8 * GB)), [10 * GB, 2 * GB]);
  // A machine that answers "no displays anywhere" is taken at its word.
  assertEquals(
    vramReserveShares([gpu(24, 0.5, false)], r(0, 8 * GB)),
    [0],
    "a headless machine has no desktop to defend",
  );
  // A machine that cannot be asked is assumed to draw its screen on card 0 —
  // far more often true than not, and the other way round is a driver reset.
  assertEquals(vramReserveShares([gpu(24), gpu(8)], r(0, 4 * GB)), [4 * GB, 0]);
  assert(displayUnknown([gpu(24)]), "and the UI is told it is an assumption");
  assert(!displayUnknown([gpu(24, 0.5, false)]));
  // Never more than the card holds, whatever was typed.
  assertEquals(vramReserveShares([gpu(4, 0, true)], r(0, 99 * GB)), [4 * GB]);
  assertEquals(vramReserveShares([], r(4 * GB, 4 * GB)), []);
  assertEquals(vramReserveShares([gpu(24)], r(0, 0)), [0]);
});

/** A number typed into a box reaches this, so it has to survive anything a
 *  number box can produce. A NaN reserve would poison every fit test it
 *  touched, silently, because NaN comparisons are all false. */
Deno.test("reserve: a hostile or impossible value is clamped, never propagated", () => {
  assertEquals(reserveBytes(Number.NaN), 0);
  assertEquals(reserveBytes(-4), 0);
  assertEquals(reserveBytes(1e9), MAX_RESERVE_GB * GB, "capped, not infinite");
  assertEquals(reserveGb(4 * GB), 4, "and reads back as the number typed");
  // Clamped to the machine it applies to: 64 GB reserved on a 24 GB card means
  // the card is entirely spoken for — which is an honest (if useless) answer —
  // and never a negative capacity.
  const r = reserveOf(
    hw({
      gpus: [gpu(24)],
      reserve: { perGpuB: 64 * GB, connectedB: 64 * GB, ramB: 999 * GB },
    }),
  );
  assertEquals(r.perGpuB, 24 * GB);
  assertEquals(r.connectedB, 24 * GB);
  assertEquals(r.ramB, 64 * GB);
});

Deno.test("plan: a reserve is spent memory — free shrinks by it, and it is labelled apart", () => {
  const m = meta();
  const s = { ...defaults(), ngl: 999 };
  const open = plan(m, hw({ gpus: [gpu(24)] }), s);
  const held = plan(
    m,
    hw({
      gpus: [gpu(24)],
      reserve: { perGpuB: 0, connectedB: 4 * GB, ramB: 16 * GB },
    }),
    s,
  );
  assertEquals(held.vram.reservedB, 4 * GB);
  assertEquals(held.ram.reservedB, 16 * GB);
  assertEquals(
    held.vram.otherB,
    open.vram.otherB,
    "the reserve is NOT filed as another process — it is the one the user can take back",
  );
  assertEquals(
    open.vram.freeB - held.vram.freeB,
    4 * GB,
    "and it comes out of what is free, byte for byte",
  );
  assertEquals(open.ram.freeB - held.ram.freeB, 16 * GB);
  assertEquals(
    held.vram.capacityB,
    open.vram.capacityB,
    "the card is still the size it is — a reserve is not a smaller GPU",
  );
  // Per card, so the packing budgets shrink with it and the picture agrees.
  assertEquals(held.devices.cards[0]?.reservedB, 4 * GB);
});

/** The failure this feature exists to prevent, and the one it must never cause
 *  silently: a plan that no longer fits BECAUSE of the reserve says so, and says
 *  which control gives the memory back. */
Deno.test("plan: when the reserve is what does not fit, the note names it", () => {
  const m = meta({ nLayer: 32, layers: layers(32, 600 * 1024 * 1024) });
  const machine = { gpus: [gpu(24, 0.5)] };
  const s = { ...defaults(), ngl: 999, ctxSize: 8192 };
  const open = plan(m, hw(machine), s);
  assertEquals(open.vram.overB, 0, "it fits with nothing held back");
  const held = plan(
    m,
    hw({
      ...machine,
      reserve: { perGpuB: 0, connectedB: 6 * GB, ramB: 0 },
    }),
    s,
  );
  assert(held.vram.overB > 0, "and not once 6 GB is held back");
  assert(
    held.notes.some((n) => n.includes("reserving")),
    `the way out is on screen: ${held.notes.join(" | ")}`,
  );
});

Deno.test("tune: the reserve binds the tuner, and is given as a reason", () => {
  const m = meta({ nCtxTrain: 131_072 });
  const machine = { gpus: [gpu(24)], backend: "cuda" as const };
  const open = tune(m, hw(machine), defaults(), "vram");
  const held = tune(
    m,
    hw({
      ...machine,
      reserve: { perGpuB: 0, connectedB: 14 * GB, ramB: 16 * GB },
    }),
    defaults(),
    "vram",
  );
  assert(open.possible && held.possible);
  assert(
    held.ctx < open.ctx,
    `holding 14 GB back has to cost context: ${open.ctx} → ${held.ctx}`,
  );
  assert(
    held.reasons.some((r) => r.includes("reserved for your own work")),
    "and the user is told why the ceiling is where it is",
  );
  assert(
    !open.reasons.some((r) => r.includes("reserved for your own work")),
    "while a machine with no reserve says nothing about one",
  );
});

/** A reserve big enough to block the placement must explain itself in the
 *  blocker too — "does not fit" with no mention of the 20 GB the user is holding
 *  is a refusal nobody can act on. */
Deno.test("tune: a blocking reserve is named in the blocker, not just felt", () => {
  const m = meta();
  const held = tune(
    m,
    hw({
      gpus: [gpu(24)],
      reserve: { perGpuB: 0, connectedB: 23 * GB, ramB: 0 },
    }),
    defaults(),
    "vram",
  );
  assertEquals(held.possible, false);
  assertStringIncludes(held.blocker, "you reserve");
});

Deno.test("devsplit: the user's reserve narrows each card's budget on top of the fixed one", () => {
  const gpus = [gpu(24, 0), gpu(24, 0)];
  const open = deviceBudgets(gpus);
  const held = deviceBudgets(gpus, 0, [3 * GB, 1 * GB]);
  assertEquals((open[0] ?? 0) - (held[0] ?? 0), 3 * GB);
  assertEquals((open[1] ?? 0) - (held[1] ?? 0), 1 * GB);
  assert(
    deviceBudgets([gpu(4, 0)], 0, [99 * GB])[0] === 0,
    "and a card wholly reserved has a budget of zero, never a negative one",
  );
});

/**
 * A placement that has already happened is not a prediction to re-check.
 *
 * The fitter's per-card budgets hold back a safety reserve (5% of each card)
 * and, in the live path, our own footprint is re-derived by proportion — so
 * re-packing a model that is LOADED came up about a gigabyte short and the
 * machine panel announced "1010 MB of layers have nowhere to go — no card has
 * room for them, however the cut is made" about a server that was answering
 * prompts, with `vram.overB` reading 0 on the same screen (measured, two 25.6 GB
 * cards, DeepSeek-V4-Flash at `--n-cpu-moe 33`). The measurement decides for a
 * live run; the packer decides for a proposal.
 */
Deno.test("plan: a running placement is described, not re-litigated", () => {
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nCtxTrain: 65_536,
    nExpert: 256,
    nExpertUsed: 6,
    layers: layers(43, 3400 * MB, 3260 * MB),
  });
  // The live shape: our own bytes have already been attributed to us
  // (`plan.ts:withoutOurUsage`), so the cards read mostly free, the machine has
  // the VRAM in aggregate — and the per-card budgets, after the safety reserve
  // and this device's scratch, still cannot take the last of the heavy layers.
  const machine = hw({ gpus: [gpu(24, 2.5), gpu(24, 2.5)], backend: "cuda" });
  const s = { ...defaults(), ngl: 999, nCpuMoe: 33, ctxSize: 2560 };

  const proposed = plan(m, machine, s);
  assertEquals(
    proposed.vram.overB,
    0,
    "the machine has the VRAM — this is the by-card failure, not the total",
  );
  assertEquals(
    proposed.devices.fits,
    false,
    "as a PROPOSAL this cannot be cut across the cards, and saying so is the point",
  );
  assert(proposed.devices.unplacedB > 0);
  assert(
    proposed.notes.some((n) => n.includes("no card that can hold them")),
    "and a proposal must say which layers have nowhere to go",
  );

  const running = plan(m, machine, s, "running");
  assertEquals(
    running.devices.fits,
    true,
    "the same arrangement, already loaded, is a fact — llama.cpp placed it",
  );
  assertEquals(running.devices.unplacedB, 0);
  assert(
    !running.notes.some((n) => n.includes("no card that can hold them")),
    `nothing may claim a loaded model is unplaceable: ${
      running.notes.join(" | ")
    }`,
  );
  assert(
    !running.notes.some((n) => n.includes("fits across the cards but not on")),
    "nor its sibling note",
  );
  // Every other number is the same — this is one prediction being dropped, not
  // a second accounting.
  assertEquals(running.vram.usedB, proposed.vram.usedB);
  assertEquals(running.ram.usedB, proposed.ram.usedB);
  assertEquals(running.kvTotalB, proposed.kvTotalB);
});

/**
 * The per-card picture of a LIVE run is a measurement, not the packer's guess.
 *
 * The report, from the machine this app is developed on: two cards holding
 * 11.6 and 12.2 GB by nvidia-smi, and the panel drawing every byte of the model
 * on GPU 0 and nothing at all on GPU 1. `packSlots` is first-fit, and with
 * `--n-cpu-moe 42` every offloaded slot but one is attention-only — the whole
 * 9.2 GB fits inside card 0, so it answered `[44, 0]`. Nothing was wrong with
 * that as a PROPOSAL; it was never issued as one. No `-ts` is emitted when the
 * packer needs none, so llama.cpp divided the layers by its own free-VRAM rule
 * and used both cards. The panel was describing a placement that did not happen
 * — and filing our own 12 GB on card 1 as somebody else's memory.
 */
Deno.test("plan: a live run's per-card bytes come from the measurement", () => {
  const MB = 1024 * 1024;
  const m = meta({
    nLayer: 43,
    nExpert: 256,
    nExpertUsed: 8,
    layers: layers(43, 40 * MB + 2500 * MB, 2500 * MB),
  });
  const s = { ...defaults(), ngl: 999, nCpuMoe: 42, ctxSize: 8192 };
  // Both cards hold 12 GB of the 24 they have. llama.cpp put half the layers on
  // each; the packer, left to itself, would have put all of them on card 0.
  const machine = hw({ gpus: [gpu(24, 12), gpu(24, 12)], backend: "cuda" });
  // Free per card when this run was spawned: 23.5 GB each, so we took 11.5.
  const atStart = [23.5 * 1024 ** 3, 23.5 * 1024 ** 3];

  const guessed = plan(m, machine, s, "running");
  assertEquals(
    guessed.devices.bytesB[1],
    0,
    "without the baseline this is the packer's first fit — card 1 gets nothing",
  );

  const measured = plan(m, machine, s, "running", atStart);
  assert(
    (measured.devices.bytesB[1] ?? 0) > 0,
    `card 1 holds bytes and the plan must say so: ${
      JSON.stringify(measured.devices.bytesB)
    }`,
  );
  const c1 = measured.devices.cards[1]!;
  assert(
    c1.weightsB + c1.kvB + c1.computeB > 0,
    "and they are drawn as ours, in the map's own bands",
  );
  // The whole point: our bytes stop being counted as another process's.
  assert(
    c1.otherB < guessed.devices.cards[1]!.otherB,
    `our own memory must leave "in use elsewhere": ${c1.otherB} vs ${
      guessed.devices.cards[1]!.otherB
    }`,
  );
  // Each card's total is the measurement, to the byte — the bands inside it are
  // apportioned, the card total is not.
  assertEquals(
    measured.devices.bytesB.map(Math.round),
    [11.5 * 1024 ** 3, 11.5 * 1024 ** 3],
    "free-at-spawn minus free-now, per card",
  );
});

/** The baseline is only trusted when it can be lined up with the cards. A list
 *  that names a different number of them is an index mismatch — this machine
 *  has an AMD iGPU that a CUDA build filters out — and attributing card 2's
 *  memory to card 1 would be worse than the guess it replaces. */
Deno.test("plan: a per-card baseline of the wrong length is ignored", () => {
  const MB = 1024 * 1024;
  const m = meta({ nLayer: 32, layers: layers(32, 300 * MB) });
  const s = { ...defaults(), ngl: 999, ctxSize: 4096 };
  const machine = hw({ gpus: [gpu(24, 8), gpu(24, 8)], backend: "cuda" });
  const short = plan(m, machine, s, "running", [23 * 1024 ** 3]);
  const none = plan(m, machine, s, "running");
  assertEquals(short.devices.bytesB, none.devices.bytesB);
  assertEquals(short.vram.usedB, none.vram.usedB);
  // And a baseline that recorded nothing (an adopted server) is not a machine
  // with an empty model on it.
  const zeroes = plan(m, machine, s, "running", [0, 0]);
  assertEquals(zeroes.devices.bytesB, none.devices.bytesB);
});

/** And real pressure on a live run still lands: when the MEASUREMENT says the
 *  machine is over, "running" changes nothing — that is the case `drift` reads
 *  to tell the user they are being squeezed. */
Deno.test("plan: a running model that is genuinely over capacity still says so", () => {
  const MB = 1024 * 1024;
  const m = meta({ nLayer: 32, layers: layers(32, 900 * MB) });
  // Someone else took almost everything after the model loaded.
  const machine = hw({ gpus: [gpu(24, 23.5)], backend: "cuda" });
  const s = { ...defaults(), ngl: 999, ctxSize: 8192 };
  const running = plan(m, machine, s, "running");
  assert(running.vram.overB > 0, "the measurement is the authority");
  assertEquals(running.fits, false);
  assert(running.notes.some((n) => n.includes("Over VRAM")));
});

// ── what a reply is made of ────────────────────────────────────────────────
//
// A local model answers with markdown, and the two things it answers with most
// are code and file contents. Every rule below is one the component must not be
// allowed to get wrong on its own, which is why they are here.

Deno.test("richtext: a fenced block is a block, and it knows what it holds", () => {
  const blocks = replyBlocks(
    "Here is the fix:\n\n```ts\nconst a = 1;\n```\n\nThat's all.",
  );
  assertEquals(blocks.map((b) => b.kind), ["text", "code", "text"]);
  const code = blocks[1];
  assert(code?.kind === "code");
  assertEquals(code.text, "const a = 1;", "no fence, no header, just the code");
  assertEquals(code.lang, "ts");
  assertEquals(code.file, "");
  assertEquals(code.open, false);
});

/** The four spellings a model actually emits. There is no standard for naming
 *  the file a block belongs to, so refusing all but one would mean showing
 *  "typescript" over a block whose own first line said `src/lib/plan.ts`. */
Deno.test("richtext: every common way of naming the file is read", () => {
  assertEquals(parseInfo("python"), { lang: "python", file: "" });
  assertEquals(parseInfo("src/lib/plan.ts"), {
    lang: "typescript",
    file: "src/lib/plan.ts",
  });
  assertEquals(parseInfo("ts:src/lib/plan.ts"), {
    lang: "ts",
    file: "src/lib/plan.ts",
  });
  assertEquals(parseInfo('ts title="src/plan.ts"'), {
    lang: "ts",
    file: "src/plan.ts",
  });
  assertEquals(parseInfo("ts src/lib/plan.ts"), {
    lang: "ts",
    file: "src/lib/plan.ts",
  });
  assertEquals(parseInfo("file=app.js"), {
    lang: "javascript",
    file: "app.js",
  });
  // No extension and no slash: a language, not a file — otherwise every
  // ```bash block would claim to be a file called "bash".
  assertEquals(parseInfo("bash"), { lang: "bash", file: "" });
  // Except the two that genuinely have no extension.
  assertEquals(parseInfo("Dockerfile").file, "Dockerfile");
  assertEquals(parseInfo(""), { lang: "", file: "" });
});

/** The case that decides whether this is usable while a model is answering: the
 *  closing fence has not arrived yet. Rendering the half-written file as raw
 *  text until it does means the block appears, reflows and re-indents at the
 *  moment the model finishes — exactly when the user is reading it. */
Deno.test("richtext: a block still being written is a block, and says so", () => {
  const blocks = replyBlocks("Sure:\n\n```py\nimport os\nprint(1)");
  assertEquals(blocks.length, 2);
  const code = blocks[1];
  assert(code?.kind === "code");
  assertEquals(code.open, true);
  assertEquals(code.text, "import os\nprint(1)");
});

/** A fence inside a fence. The outer block uses a longer marker precisely so
 *  the inner one does not end it — get this wrong and a reply about markdown,
 *  or any shell snippet with backticks, is cut in half. */
Deno.test("richtext: a longer fence survives the fences inside it", () => {
  const blocks = replyBlocks(
    "````md\nSome docs:\n\n```ts\nx\n```\n\nend\n````",
  );
  assertEquals(blocks.length, 1);
  const code = blocks[0];
  assert(code?.kind === "code");
  assertEquals(code.lang, "md");
  assertStringIncludes(code.text, "```ts");
  assertEquals(code.open, false);
});

Deno.test("richtext: an indented fence keeps the code's own shape", () => {
  const blocks = replyBlocks("  ```ts\n  if (a) {\n      b();\n  }\n  ```");
  const code = blocks[0];
  assert(code?.kind === "code");
  // The fence's own two spaces come off; the four that make the nesting stay.
  assertEquals(code.text, "if (a) {\n    b();\n}");
});

Deno.test("richtext: inline code spans are spans, and a lone backtick is a backtick", () => {
  const parts = inlineChunks("Run `deno task dev` first");
  assertEquals(parts.map((p) => p.code), [false, true, false]);
  assertEquals(parts[1]?.text, "deno task dev");
  // Unclosed: one plain run, never a span that swallows the rest of the reply.
  assertEquals(inlineChunks("a ` b").map((p) => p.code), [false]);
  // Never across a line break.
  assertEquals(inlineChunks("a `b\nc` d").map((p) => p.code), [false]);
});

/** What the copy-chat button puts on the clipboard. Markdown, because that is
 *  what the reply already is — and nothing on screen is silently dropped. */
Deno.test("richtext: the transcript is the conversation, fences and all", () => {
  const text = transcript({
    system: "Be brief.",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "```ts\nconst a = 1;\n```",
        thinking: "they said hi",
        tps: 8.94,
      },
    ],
    partial: "still writ",
  });
  assertStringIncludes(text, "### system\n\nBe brief.");
  assertStringIncludes(text, "### user\n\nhi");
  assertStringIncludes(text, "### assistant · 8.9 tok/s");
  assertStringIncludes(text, "```ts\nconst a = 1;\n```");
  assertStringIncludes(text, "> they said hi");
  assertStringIncludes(
    text,
    "still writ",
    "the reply still arriving is on screen, so it is in the copy",
  );
  // An empty conversation copies nothing rather than a pile of headings.
  assertEquals(transcript({ messages: [] }).trim(), "");
});

/** Four dials, one scale. The quarters are the whole rule: an eye scanning CPU,
 *  GPU, RAM and VRAM should not have to learn four of them. */
Deno.test("thermal: a load reading is coloured by quarter", () => {
  assertEquals(loadTone(0), "busy");
  assertEquals(loadTone(24.9), "busy");
  assertEquals(loadTone(25), "ok");
  assertEquals(loadTone(49.9), "ok");
  assertEquals(loadTone(50), "warn");
  assertEquals(loadTone(74.9), "warn");
  assertEquals(loadTone(75), "bad");
  assertEquals(loadTone(100), "bad");
  // A machine that has not been read yet is idle, not on fire: every figure
  // that reaches this comes out of a division, and 0/0 is how a boot-time
  // reading arrives.
  assertEquals(loadTone(Number.NaN), "busy");
  assertEquals(loadTone(-1), "busy");
});

// ── available on LAN ───────────────────────────────────────────────────────
//
// One llama.cpp flag, `--host`, behind one switch. The rules worth pinning are
// the two that make it a decision rather than trivia: what counts as exposed,
// and which address another machine should actually dial.

Deno.test("lan: exposed means reachable from another machine, and 127.x never is", () => {
  const at = (host: string) => ({ ...defaults(), host });
  assertEquals(isLanExposed(at("127.0.0.1")), false, "llama.cpp's default");
  assertEquals(isLanExposed(at("localhost")), false);
  assertEquals(isLanExposed(at("::1")), false);
  assertEquals(isLanExposed(at("0.0.0.0")), true);
  assertEquals(isLanExposed(at("::")), true);
  // A specific interface is reachable too — the switch is off, but the machine
  // IS exposed, and a UI that showed "off" there would be lying.
  assertEquals(isLanExposed(at("192.168.1.10")), true);
  assertEquals(isLanExposed(defaults()), false, "off by default");
  assertEquals(lanHost(true), "0.0.0.0");
  assertEquals(lanHost(false), "127.0.0.1");
});

/** `0.0.0.0` is what llama-server BINDS; it is not what anyone dials. A
 *  workstation reports half a dozen addresses — docker bridges, VPN tunnels,
 *  loopback — and exactly one of them is worth printing. */
Deno.test("lan: the address offered is one another machine can actually use", () => {
  assertEquals(
    pickLanIp(["127.0.0.1", "172.17.0.1", "192.168.1.24"]),
    "172.17.0.1",
    "both are private; the OS order decides between them",
  );
  assertEquals(pickLanIp(["127.0.0.1", "192.168.1.24"]), "192.168.1.24");
  // Link-local only exists when DHCP failed, so it goes last rather than first.
  assertEquals(
    pickLanIp(["169.254.7.1", "10.0.0.5"]),
    "10.0.0.5",
  );
  assertEquals(
    pickLanIp(["127.0.0.1"]),
    "",
    "loopback is what we are escaping",
  );
  assertEquals(pickLanIp(["8.8.8.8"]), "", "a public address is not this");
  assertEquals(pickLanIp([]), "");
  assertEquals(lanUrl("192.168.1.24", 18080), "http://192.168.1.24:18080");
  // Better nothing than `http://0.0.0.0:8080`, which reaches nothing.
  assertEquals(lanUrl("", 8080), "");
  assertEquals(lanUrl("192.168.1.24", 0), "");
});

/** llama-server has no password unless an API key is set — which is the whole
 *  reason this switch says something instead of just flipping. */
Deno.test("lan: the warning changes with the API key, and always names the risk", () => {
  const open = lanWarning(defaults());
  assertStringIncludes(open, "Anyone on your network");
  assertStringIncludes(open, "API key");
  const keyed = lanWarning({ ...defaults(), apiKey: "s3cret" });
  assertStringIncludes(keyed, "who has the API key");
});

// ── letting the desktop go first ───────────────────────────────────────────

Deno.test("priority: the two queues that make a machine feel slow, both lowered", () => {
  const steps = prioritySteps(4242);
  assertEquals(steps.map((s) => s.cmd), ["renice", "ionice"]);
  assertEquals(steps[0]?.args, ["-n", "19", "-p", "4242"], "the politest nice");
  // CPU is not the half that makes a load hurt: reading 145 GB of weights off
  // an NVMe stutters a desktop whatever the nice value is.
  assertEquals(steps[1]?.args, ["-c", "3", "-p", "4242"], "idle I/O class");
  // The idle class is refused on some kernels and in some containers; the
  // fallback is the lowest best-effort band, which never needs privileges.
  assertEquals(ioFallback(4242).args, ["-c", "2", "-n", "7", "-p", "4242"]);
  assertEquals(prioritySteps(0), [], "no pid, nothing to renice");
  assertEquals(prioritySteps(-1), []);
});

/** A run left at normal priority while the switch said otherwise is the kind of
 *  silent disagreement this app refuses everywhere else — so the log always
 *  says what actually took effect. */
Deno.test("priority: the log line says what happened, including when nothing did", () => {
  assertStringIncludes(
    priorityNote(["nice 19", "idle I/O"], []),
    "nice 19 + idle I/O",
  );
  assertStringIncludes(priorityNote(["nice 19", "idle I/O"], []), "desktop");
  const partial = priorityNote(["nice 19"], ["ionice: not found"]);
  assertStringIncludes(partial, "nice 19");
  assertStringIncludes(partial, "ionice: not found");
  const none = priorityNote([], ["renice: not found"]);
  assertStringIncludes(none, "could not lower the priority");
  assertStringIncludes(none, "sluggish", "and what that means for the user");
});

/** `/proc/<pid>/stat` is how the test below checks the REAL process rather than
 *  that a command was issued — and `comm` can contain spaces and parentheses,
 *  which is why the parse starts after the last `)`. */
Deno.test("priority: the nice value is read out of a hostile /proc line", () => {
  const stat =
    "1234 (llama server (x)) S 1 1234 1234 0 -1 4194304 100 0 0 0 5 2 0 0 20 19 4 0 999 0 0";
  assertEquals(niceFromProcStat(stat), 19);
  assertEquals(
    niceFromProcStat("42 (sleep) S 1 42 42 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 7"),
    0,
  );
  assertEquals(niceFromProcStat("nonsense"), null);
});

// ── the message queue ──────────────────────────────────────────────────────

Deno.test("queue: only a reply that ended on its own moves the queue on", () => {
  // The whole feature turns on this. A cancelled reply means the user said
  // "not like that", and an error means the next send fails identically — six
  // queued messages against a dead server is six identical errors.
  assert(advances("done"));
  assert(!advances("cancelled"), "Stop must not fire the next message");
  assert(!advances("error"), "a dead server must not eat the queue");
});

Deno.test("queue: adding refuses blanks and the cap, visibly", () => {
  assertEquals(queueAdd([], "  hi  "), ["hi"], "and it is trimmed");
  assertEquals(queueAdd(["a"], "   "), ["a"], "whitespace is not a message");
  const full = Array.from({ length: QUEUE_MAX }, (_, i) => `m${i}`);
  assertEquals(
    queueAdd(full, "one more").length,
    QUEUE_MAX,
    "the cap holds, and the caller can SEE it held (length unchanged)",
  );
});

Deno.test("queue: removing one leaves the order of the rest", () => {
  assertEquals(queueRemove(["a", "b", "c"], 1), ["a", "c"]);
  assertEquals(queueRemove(["a"], 5), ["a"], "out of range changes nothing");
  assertEquals(queueRemove(["a"], -1), ["a"]);
});

Deno.test("queue: what Enter means, in every state it can be pressed", () => {
  assertEquals(submitKind("hi", 0, false), "send", "idle: it IS the request");
  assertEquals(submitKind("hi", 0, true), "queue", "busy: hold it");
  assertEquals(submitKind("hi", 3, false), "send", "drains the queue first");
  // The gesture left after a Stop: the queue is holding and nothing is coming
  // to collect it, so an empty box still has something to do.
  assertEquals(submitKind("", 2, false), "send");
  assertEquals(submitKind("", 2, true), null, "the drain will do it");
  assertEquals(submitKind("", 0, false), null, "nothing to do");
  assertEquals(
    submitKind("hi", QUEUE_MAX, true),
    "full",
    "named, so the message is never silently dropped",
  );
});

Deno.test("queue: the note separates 'waiting on the model' from 'waiting on you'", () => {
  // Two states that look identical on screen, and only one of them empties
  // itself. Describing the second as the first is a user watching a spinner
  // that will never stop.
  assertEquals(queueNote(0, false), "", "nothing waiting, nothing said");
  assertStringIncludes(queueNote(1, true), "1 message");
  assertStringIncludes(queueNote(2, true), "2 messages");
  assertStringIncludes(queueNote(2, true), "this reply finishes");
  assertStringIncludes(queueNote(2, false), "press Send");
  assertStringIncludes(queueNote(QUEUE_MAX, true), "full");
});

Deno.test("queue: a chip label elides on a word and keeps the whole text elsewhere", () => {
  assertEquals(queueLabel("short one"), "short one");
  assertEquals(
    queueLabel("now  write\nthe  tests"),
    "now write the tests",
    "newlines and runs collapse — a chip is one line",
  );
  const source = "refactor the parser and then explain every choice";
  const long = queueLabel(source);
  assert(long.endsWith("…"));
  assert(long.length <= 43);
  // The cut lands ON a word boundary: what is shown is a prefix of the source,
  // and the character that follows it there is a space — never a half word,
  // which reads as corruption rather than as elision.
  const shown = long.slice(0, -1);
  assert(source.startsWith(shown), shown);
  assertEquals(source[shown.length], " ", `cut mid-word: ${shown}`);
});

Deno.test("queue: the box grows with the lines typed and stops at a cap", () => {
  assertEquals(draftRows(""), 1, "an empty box is one row");
  assertEquals(draftRows("one line"), 1);
  assertEquals(draftRows("a\nb\nc"), 3, "a pasted file grows the box");
  assertEquals(draftRows("x\n".repeat(40)), 6, "past the cap it scrolls");
  assertEquals(draftRows("x\n".repeat(40), 3), 3, "the cap is the caller's");
});

// ── quantisation and speed ─────────────────────────────────────────────────

Deno.test("quant: this file's bits per weight is measured, not looked up", () => {
  // The label is a MIX — a Q4_K_M is mostly Q4_K with some Q6_K — so two files
  // wearing it are not the same size, and the whole comparison is a ratio
  // against this number. `rust/src/gguf.rs` counts the weights off the tensor
  // table so it can be exact.
  const m = meta({ params: 8_000_000_000, layers: layers(32, 128 * MB) });
  const got = fileBpw(m);
  assertEquals(got.exact, true, "a counted file reports an exact figure");
  assertAlmostEquals(got.bpw, (m.tensorBytes * 8) / 8_000_000_000, 1e-9);

  // A model read by a build older than that field reports 0 weights, and the
  // label is then the only thing there is. It says so rather than pretending.
  const old = { ...m, params: 0 };
  assertEquals(fileBpw(old), { bpw: 4.85, exact: false });

  // A nonsense header must not scale the table. 100 bits per weight is not a
  // quantisation, it is a truncated read.
  const broken = { ...m, params: 1 };
  assertEquals(fileBpw(broken).exact, false, "an absurd figure is refused");
});

Deno.test("quant: rescaling moves the weights and nothing else", () => {
  const m = meta({ params: 8_000_000_000, layers: layers(32, 128 * MB) });
  const from = fileBpw(m).bpw;
  const half = rescale(m, from / 2);
  assertAlmostEquals(half.tensorBytes / m.tensorBytes, 0.5, 1e-6);
  assertAlmostEquals(
    (half.layers[0]?.bytes ?? 0) / (m.layers[0]?.bytes ?? 1),
    0.5,
    1e-6,
  );
  assertAlmostEquals(half.embdBytes / m.embdBytes, 0.5, 1e-6);
  // The geometry is the model, not the file: the KV cache, the context and the
  // layer count are identical at every quantisation, and scaling them would
  // make the comparison a comparison of two different models.
  assertEquals(half.nLayer, m.nLayer);
  assertEquals(half.nCtxTrain, m.nCtxTrain);
  assertEquals(half.nHeadKv, m.nHeadKv);
  assertEquals(half.keyLength, m.keyLength);
});

Deno.test("quant: a smaller file that stops spilling into RAM is worth more than its size", () => {
  // The point of running the real tuner per candidate rather than a ratio. A
  // model that does not fit reads most of its weights over the host bus at a
  // tenth of the bandwidth; the quant that fits does not, and the step is
  // worth far more than the bytes it saved.
  const big = meta({
    name: "Spiller",
    nLayer: 32,
    quant: "Q8_0",
    layers: layers(32, 900 * MB),
    embdBytes: 200 * MB,
    outputBytes: 200 * MB,
  });
  const machine = hw({ gpus: [gpu(16)] });
  const rows = quantOptions(big, machine, defaults());
  const mine = rows.find((r) => r.current);
  assertExists(mine, "the file you have is always a row");
  assertEquals(mine.quant, "Q8_0");

  const q4 = rows.find((r) => r.quant === "Q4_K_M");
  assertExists(q4, "and a Q4 is offered against it");
  assert(q4.bpw < mine.bpw, "which is smaller");
  assert(
    q4.tps > mine.tps,
    `and faster: ${mine.tps.toFixed(1)} → ${q4.tps.toFixed(1)} tok/s`,
  );
  // The gain is more than the byte ratio precisely because the placement moved.
  const byteRatio = mine.bpw / q4.bpw;
  assert(
    q4.speedup > byteRatio,
    `placement, not just size: ${q4.speedup.toFixed(2)}× against a ${
      byteRatio.toFixed(2)
    }× byte ratio`,
  );
  assert(q4.layersInRam < mine.layersInRam, "fewer layers left on the host");
});

Deno.test("quant: nothing bigger than the file you already have is offered", () => {
  // "Download a larger file and go slower" is not advice. A Q4 model offers
  // only what is below it.
  const m = meta({ quant: "Q4_K_M", layers: layers(32, 100 * MB) });
  const rows = quantOptions(m, hw({ gpus: [gpu(24)] }), defaults());
  const mine = rows.find((r) => r.current);
  assertExists(mine);
  for (const r of rows) {
    assert(
      r.current || r.bpw < mine.bpw,
      `${r.quant} at ${r.bpw} is not smaller than ${mine.bpw}`,
    );
  }
  // Sorted big to small, so the table reads in the direction the decision runs.
  const bits = rows.map((r) => r.bpw);
  assertEquals(bits.slice().sort((a, b) => b - a), bits);
});

Deno.test("quant: the advice line stays quiet unless the gain is worth acting on", () => {
  // A line that appears for every model is a line nobody reads.
  const tiny = meta({ quant: "Q4_K_M", layers: layers(32, 60 * MB) });
  const roomy = hw({ gpus: [gpu(48)] });
  const rows = quantOptions(tiny, roomy, defaults());
  const advice = quantAdvice(rows);
  // Everything fits either way here, so the only gain on offer is the byte
  // ratio between Q4 and Q3 — under the threshold, so nothing is said.
  assertEquals(advice, "", `expected silence, got: ${advice}`);

  // And it speaks when the difference is real, naming the price in the same
  // sentence as the gain — advice that names only the gain is half given.
  const spiller = meta({
    quant: "Q8_0",
    layers: layers(32, 900 * MB),
    embdBytes: 200 * MB,
    outputBytes: 200 * MB,
  });
  const loud = quantAdvice(
    quantOptions(spiller, hw({ gpus: [gpu(16)] }), defaults()),
  );
  assert(loud.length > 0, "a real gain is named");
  assertStringIncludes(loud, "faster");
  assert(
    /quality|loss/i.test(loud),
    `the price is in the same sentence: ${loud}`,
  );
});

Deno.test("quant: the quality note follows the curve every perplexity table agrees on", () => {
  // Flat from 8 bits to about 5, a bend through 4, a cliff below 3.
  assertStringIncludes(qualityNote(8.5), "No practical");
  assertStringIncludes(qualityNote(4.85), "small");
  assert(/large/i.test(qualityNote(2.6)), "below 3 bits says so plainly");
});

// ── the bench ──────────────────────────────────────────────────────────────

Deno.test("bench: a reply without timings is not a slow machine", () => {
  // A build too old to report `timings` has measured NOTHING. Writing 0 tok/s
  // would put "this machine is broken" on screen over a build's age.
  const at = {
    kind: "prose" as const,
    latencyMs: 100,
    ctx: 4096,
    modelPath: "/m.gguf",
    at: 1,
  };
  assertEquals(parseBench({}, at), null, "no timings block");
  assertEquals(parseBench({ timings: {} }, at), null, "an empty one");
  assertEquals(
    parseBench({ timings: { predicted_n: 0, predicted_per_second: 0 } }, at),
    null,
    "a run that generated nothing",
  );
  assertEquals(parseBench(null, at), null);
});

Deno.test("bench: llama.cpp's own numbers come through, and ours are measured here", () => {
  const r = parseBench({
    timings: {
      prompt_n: 42,
      prompt_per_second: 364.5,
      predicted_n: 128,
      predicted_per_second: 13.1,
    },
  }, {
    kind: "prose" as const,
    latencyMs: 1234.6,
    ctx: 262144,
    modelPath: "/m.gguf",
    at: 99,
  });
  assertExists(r);
  assertEquals(r.promptTokens, 42);
  assertEquals(r.promptTps, 364.5);
  assertEquals(r.genTokens, 128);
  assertEquals(r.genTps, 13.1);
  // Rounded, because a sub-millisecond figure is precision the measurement
  // does not have.
  assertEquals(r.latencyMs, 1235);
  assertEquals(r.ctx, 262144);
  assertEquals(r.modelPath, "/m.gguf");
});

Deno.test("bench: a measurement belongs to one model at one context", () => {
  const b = parseBench(
    { timings: { predicted_n: 128, predicted_per_second: 13 } },
    {
      kind: "prose" as const,
      latencyMs: 1,
      ctx: 4096,
      modelPath: "/a.gguf",
      at: 5,
    },
  );
  assertExists(b);
  assertEquals(benchApplies(b, "/a.gguf", 4096), true);
  assertEquals(benchApplies(b, "/b.gguf", 4096), false, "another model");
  // The KV cache is read every token, so the same model at 4k and 256k are
  // different measurements. Showing one as the other would be the app's own
  // estimate dressed as a fact.
  assertEquals(benchApplies(b, "/a.gguf", 262144), false, "another context");
  assertEquals(benchApplies(EMPTY_BENCH, "/a.gguf", 4096), false, "never run");
});

Deno.test("bench: the request holds every variable still", () => {
  const body = benchRequest();
  // A cached prompt reports a prefill rate that is really a cache hit.
  assertEquals(body.cache_prompt, false);
  assertEquals(body.temperature, 0, "repeatable");
  assertEquals(body.n_predict, BENCH_TOKENS);
  // A reasoning model asked for 128 tokens can spend all of them thinking,
  // which measures the right rate on the wrong work.
  assertEquals(body.reasoning_budget, 0);
  assertEquals(
    benchRequest("prose", 8).n_predict,
    16,
    "too short to mean anything",
  );
});

Deno.test("bench: the bandwidth note speaks only when host RAM is in the path", () => {
  const GBs = 1024 ** 3;
  // A VRAM-only run reads no host RAM per token: the figure is true and
  // irrelevant, and advice about it would be advice about nothing.
  assertEquals(bandwidthNote(20 * GBs, false), null);
  assertEquals(bandwidthNote(0, true), null, "nothing measured yet");

  const slow = bandwidthNote(18 * GBs, true);
  assertExists(slow);
  assertEquals(slow.tone, "caution");
  // It names something the user can go and look at, rather than asserting what
  // they will find — the DIMMs cannot be read without root.
  assertStringIncludes(slow.message, "EXPO");

  const fine = bandwidthNote(70 * GBs, true);
  assertExists(fine);
  assertEquals(fine.tone, "info");
  assert(!/EXPO/.test(fine.message), "healthy memory gets no BIOS lecture");
});

Deno.test("stability: a draft model's VRAM is named, because the plan cannot see it", () => {
  // `plan.ts` sizes the model at `-m`; nothing reads `-md`. Unsaid, the bars
  // would show room the run is about to spend.
  const withDraft = { ...defaults(), draftModel: "/models/draft-0.5b.gguf" };
  const st = stability(meta(), hw(), withDraft);
  assert(
    st.warnings.some((w) => w.key === "draftModel"),
    "the second set of weights is named",
  );
  assertEquals(
    stability(meta(), hw(), defaults()).warnings.some((w) =>
      w.key === "draftModel"
    ),
    false,
    "and nothing is said when there is no draft model",
  );
});

Deno.test("tune: a model with no MTP block is told what speculative decoding it CAN use", () => {
  // The tuner does not switch on what it has not measured — an n-gram draft
  // that is rejected is work thrown away, and nothing here knows what the user
  // is about to ask for. So it names the option and points at the measurement.
  const dense = tune(
    meta({ nextnLayers: 0 }),
    hw({ gpus: [gpu(24)] }),
    defaults(),
  );
  assertEquals(dense.settings.specType, "", "nothing is turned on");
  const said = dense.reasons.join(" ");
  assertStringIncludes(said, "n-gram");
  assertStringIncludes(said, "Speed panel");

  // And a model that ships the block still gets it, unchanged.
  const mtp = tune(
    meta({ nextnLayers: 1 }),
    hw({ gpus: [gpu(24)] }),
    defaults(),
  );
  assertEquals(mtp.settings.specType, "draft-mtp");
});

Deno.test("quant: a file that does not fit gets a different sentence, not a 0× one", () => {
  // The ratio has no denominator when the current file cannot run: `mine.tps`
  // is 0, and the first version of this printed "about 0.0× faster", which
  // reads as a bug because it is one. A refusal is not a slow run.
  const huge = meta({
    quant: "Q8_0",
    nLayer: 60,
    nExpert: 128,
    nExpertUsed: 8,
    layers: layers(60, 2400 * MB, 2300 * MB),
    embdBytes: 800 * MB,
    outputBytes: 800 * MB,
  });
  const rows = quantOptions(huge, hw({ gpus: [gpu(24), gpu(24)] }), defaults());
  const mine = rows.find((r) => r.current);
  assertExists(mine);
  assertEquals(mine.possible, false, "the fixture is the case under test");
  const advice = quantAdvice(rows);
  assert(!/0\.0×/.test(advice), `no zero ratio: ${advice}`);
  assertStringIncludes(advice, "does not fit");
  assertStringIncludes(advice, "would");
});

Deno.test("quant: a slow rate keeps its decimal", () => {
  // "1 → 4 tokens/s" for a fourfold difference reads as a rounding artefact.
  const slow = meta({
    quant: "Q8_0",
    nLayer: 80,
    nCtxTrain: 32768,
    layers: layers(80, 900 * MB),
    embdBytes: 500 * MB,
    outputBytes: 500 * MB,
  });
  const advice = quantAdvice(
    quantOptions(slow, hw({ gpus: [gpu(24)] }), defaults()),
  );
  assert(advice.length > 0, "this one is worth advice");
  assert(
    /\d\.\d tokens\/s|\d\.\d →/.test(advice),
    `a sub-ten rate shows its decimal: ${advice}`,
  );
});

Deno.test("quant: the advice names the quality knee, not the smallest file", () => {
  // The fastest row is always the smallest file, so recommending it means
  // recommending the most damaged version of the model every single time. On a
  // 70B that read "Q3_K_M, 4.3× faster" where Q4_K_M was already 2.3× at a cost
  // most people would not notice. Advice that always points at the bottom of
  // the table is a slider, not advice.
  const big = meta({
    quant: "Q8_0",
    nLayer: 80,
    nCtxTrain: 32768,
    layers: layers(80, 900 * MB),
    embdBytes: 500 * MB,
    outputBytes: 500 * MB,
  });
  const rows = quantOptions(big, hw({ gpus: [gpu(24)] }), defaults());
  const advice = quantAdvice(rows);
  assertStringIncludes(advice, "Q4_K_M", `got: ${advice}`);
  assert(!advice.includes("Q3_K_M"), "the cliff is not the recommendation");
  // The table still carries every option, so someone who wants the last 2×
  // can read down to it and see the price on the row.
  assert(
    rows.some((r) => r.quant === "Q3_K_M"),
    "and it is still offered",
  );
});

// ── source refs: master, tags, and pull requests ───────────────────────────

Deno.test("srcref: a pull request round-trips through the ref string", () => {
  // The ref is stored in `cfg`/`builds` state and appears in a build id, so it
  // has to survive being written down and read back.
  for (
    const r of [
      { kind: "master" } as const,
      { kind: "tag", tag: "b7421" } as const,
      { kind: "pr", pr: 27754, mode: "merge" } as const,
      { kind: "pr", pr: 27754, mode: "head" } as const,
    ]
  ) {
    assertEquals(parseRef(formatRef(r)), r);
  }
  // Anything unrecognised is a tag — every ref was one before pull requests.
  assertEquals(parseRef("b9999"), { kind: "tag", tag: "b9999" });
  assertEquals(parseRef(""), { kind: "master" });
});

Deno.test("srcref: the number is taken from whatever the user pasted", () => {
  // People arrive from a browser, so the URL is the commonest input.
  assertEquals(
    parsePrInput("https://github.com/ggml-org/llama.cpp/pull/27754"),
    27754,
  );
  assertEquals(
    parsePrInput("https://github.com/ggml-org/llama.cpp/pull/27754/files"),
    27754,
  );
  assertEquals(parsePrInput("#27754"), 27754);
  assertEquals(parsePrInput(" 27754 "), 27754);
  // And a paste error is refused rather than turned into a 404 nobody can
  // explain: a box that silently builds the wrong pull request is worse than
  // one that says it did not understand.
  assertEquals(parsePrInput(""), null);
  assertEquals(parsePrInput("master"), null);
  assertEquals(parsePrInput("v1.2.3"), null);
  assertEquals(parsePrInput("0"), null);
  assertEquals(parsePrInput("99999999999"), null);
});

Deno.test("srcref: only a tag may be cached, because only a tag cannot change", () => {
  // THE bug behind "I don't have the latest llama.cpp updates". The source
  // cache was keyed on the ref name and reused whenever CMakeLists.txt
  // existed, so the first `master` build pinned that machine to that day's
  // master for ever — measured on the developer's own cache, which held a
  // master five weeks old while the log said "Reusing cached source".
  assertEquals(refMoves(parseRef("master")), true);
  assertEquals(refMoves(parseRef("pr/27754")), true, "master moves under it");
  assertEquals(refMoves(parseRef("pr/27754@head")), true, "so can a branch");
  assertEquals(refMoves(parseRef("b7421")), false, "a tag is immutable");
});

Deno.test("srcref: a ref becomes exactly one path segment", () => {
  // Both the source cache and the builds registry turn a ref into a directory,
  // and `pr/27754` carries a slash. Same class as the builds-root containment
  // check, closed at the source rather than at every consumer.
  const dirs = ["master", "b7421", "pr/27754", "pr/27754@head"].map((r) =>
    refDirName(parseRef(r))
  );
  for (const d of dirs) {
    assert(!d.includes("/"), `${d} must be one segment`);
    assert(!d.includes("\\"), `${d} must not carry a Windows separator`);
    assert(d !== "." && d !== "..", `${d} must not be a traversal`);
    assertEquals(d, d.replace(/[^A-Za-z0-9._-]/g, "_"), `${d} is safe`);
  }
  // And the two modes of one pull request are different builds.
  assertNotEquals(
    refDirName(parseRef("pr/1")),
    refDirName(parseRef("pr/1@head")),
  );
  // A hostile tag cannot escape either — tags come from GitHub, but they land
  // in a path, so they are filtered rather than trusted.
  assert(!refDirName({ kind: "tag", tag: "../../etc" }).includes("/"));
});

Deno.test("srcref: each ref names the right tarball", () => {
  // Verified live against ggml-org/llama.cpp on 2026-09-03: the `head` ref
  // answered 200 for both PRs tried, and `merge` answered 200 for one and 404
  // for the one whose branch had drifted behind master.
  assertEquals(
    tarballUrl(parseRef("master")),
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master",
  );
  assertEquals(
    tarballUrl(parseRef("b7421")),
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b7421",
  );
  assertEquals(
    tarballUrl(parseRef("pr/27754")),
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/pull/27754/merge",
  );
  assertEquals(
    tarballUrl(parseRef("pr/27754@head")),
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/pull/27754/head",
  );
});

Deno.test("srcref: a withdrawn merge ref is a verdict, not a missing file", () => {
  // GitHub publishes the merged form of a pull request only while it still
  // applies to master. The 404 therefore MEANS something, and the user needs
  // that sentence before they wait out a twenty-minute compile.
  const merge = refNotFound(parseRef("pr/27742"));
  assertStringIncludes(merge.reason, "does not currently merge");
  assert(merge.steps.length >= 2, "and it names what can be done instead");
  assert(
    merge.steps.some((s) => /on its own/.test(s)),
    "the branch alone is the escape hatch",
  );
  // A missing PR is a different sentence from a PR that stopped merging.
  const head = refNotFound(parseRef("pr/999999@head"));
  assertStringIncludes(head.reason, "not found");
  assert(!/merge/.test(head.reason));
});

Deno.test("srcref: a pull-request build says it is dated, because it is", () => {
  // "master + PR #27754" means something different tomorrow, and a name that
  // cannot carry that has to be helped.
  const at = Date.parse("2026-09-03T10:00:00Z");
  const p = refProvenance(parseRef("pr/27754"), at);
  assertStringIncludes(p, "2026-09-03");
  assertStringIncludes(p, "27754");
  assertStringIncludes(refProvenance(parseRef("b7421"), at), "b7421");
  assertStringIncludes(refProvenance(parseRef("master"), at), "2026-09-03");
});

Deno.test("backend: the release route refuses a pull request before the button lights", () => {
  // Nobody publishes prebuilt binaries for unmerged code, so the answer is
  // knowable up front — which is this function's whole job.
  const ctx = {
    platform: "linux",
    arch: "x86_64",
    found: new Set<string>(["cmake", "compiler"]),
    availableBackends: ["cpu", "cuda"] as const,
    assetCount: 12,
  };
  const r = targetReadiness("release", "cuda", { ...ctx, pr: 27754 });
  assertEquals(r.ok, false);
  assertEquals(r.pending, false, "this is an answer, not a wait");
  assertStringIncludes(r.diagnosis?.reason ?? "", "27754");
  assertStringIncludes(r.diagnosis?.reason ?? "", "no prebuilt");
  // And it points at the route that CAN do it.
  assert(
    r.diagnosis?.steps.some((s) =>
      typeof s !== "string" && s.action?.kind === "switch-origin"
    ),
    "with a button that switches route",
  );
  // Without a pull request the release route is unaffected.
  assertEquals(targetReadiness("release", "cuda", ctx).ok, true);
});

Deno.test("srcref: several pull requests become a stack, and order is kept", () => {
  // Order is load-bearing: merging A then B is not merging B then A when they
  // touch the same lines, so it is part of the ref and part of the directory.
  const r = parseRef("master+pr/27773+pr/28136+pr/27269");
  assertEquals(r, { kind: "stack", prs: [27773, 28136, 27269] });
  assertEquals(formatRef(r), "master+pr/27773+pr/28136+pr/27269");
  assertEquals(refDirName(r), "stack-27773-28136-27269");
  assertNotEquals(
    refDirName(parseRef("master+pr/2+pr/1")),
    refDirName(parseRef("master+pr/1+pr/2")),
    "a different order is a different build",
  );
  // One is not a stack, and none is master: two spellings of one thing is how
  // a cache ends up holding the same tree twice.
  assertEquals(parseRef("master+pr/27773"), {
    kind: "pr",
    pr: 27773,
    mode: "merge",
  });
  assertEquals(parseRef("master+"), { kind: "master" });
});

Deno.test("srcref: only a stack needs git", () => {
  // GitHub merges ONE pull request into master for us and serves a tarball of
  // it, so the common case installs nothing. It will not merge two into each
  // other, and nothing but a real merge can.
  assertEquals(refNeedsGit(parseRef("master")), false);
  assertEquals(refNeedsGit(parseRef("b7421")), false);
  assertEquals(refNeedsGit(parseRef("pr/27773")), false);
  assertEquals(refNeedsGit(parseRef("master+pr/1+pr/2")), true);
});

Deno.test("srcref: a list of pull requests is read the way people write one", () => {
  assertEquals(parsePrList("27773, 28136, 27269"), [27773, 28136, 27269]);
  assertEquals(parsePrList("27773 28136"), [27773, 28136]);
  assertEquals(parsePrList("#27773; #28136"), [27773, 28136]);
  assertEquals(
    parsePrList(
      "https://github.com/ggml-org/llama.cpp/pull/27773 28136",
    ),
    [27773, 28136],
  );
  // Duplicates are dropped rather than merged twice — git refuses the second
  // one anyway, with a message about nothing to do.
  assertEquals(parsePrList("27773, 27773"), [27773]);
  assertEquals(parsePrList("master"), []);
  // And the ref that a list becomes.
  assertEquals(refForPrs([]), { kind: "master" });
  assertEquals(refForPrs([5]), { kind: "pr", pr: 5, mode: "merge" });
  assertEquals(refForPrs([5, 6]), { kind: "stack", prs: [5, 6] });
  assertEquals(refPrs(parseRef("master+pr/5+pr/6")), [5, 6]);
  assertEquals(refPrs(parseRef("master")), []);
});

Deno.test("srcref: a merged pull request is the failure that looks like success", () => {
  // GitHub keeps `refs/pull/N/merge` after a PR lands, still pointing at the
  // merge computed back then. So the day it merges, that ref quietly starts
  // meaning "an older master, plus a change master already has" — the build
  // succeeds, the name looks right, and the binary is behind plain master.
  const merged = prStateNote(27742, "merged");
  assertExists(merged);
  assertEquals(merged.useMaster, true);
  assertStringIncludes(merged.message, "OLDER");
  assertStringIncludes(merged.message, "Build master instead");

  const closed = prStateNote(1, "closed");
  assertExists(closed);
  assertStringIncludes(closed.message, "closed without being merged");

  // An open one is the normal case and says nothing.
  assertEquals(prStateNote(1, "open"), null);
  assertEquals(prStateNote(1, "unknown"), null, "a hiccup must not nag");
});

// ── CUDA that is too old for the GPU ───────────────────────────────────────

Deno.test("cuda: a toolkit older than the card is an offer, not just a warning", () => {
  // Measured on the machine this was written for: CUDA 12.0 against Blackwell
  // (sm_120) can emit PTX only, the driver re-compiles it at every load, and
  // the same model then generates at 19 tok/s instead of 42.
  const up = cudaUpgradeFor("release 12.0, V12.0.140", [12.0, 12.0]);
  assertExists(up);
  assertEquals(up.have, 12.0);
  assertEquals(up.newestCap, 12.0);
  // The OLDEST release that covers the card, not the newest that exists: a
  // bigger jump is a bigger download for no gain.
  assertEquals(up.need, 12.8);
  assertEquals(cudaUpgradeFor("release 13.3", [12.0]), null, "already covered");
  assertEquals(cudaUpgradeFor("release 12.0", [8.9]), null, "an older card");
  assertEquals(cudaUpgradeFor("release 12.0", []), null, "no NVIDIA card");
  assertEquals(cudaUpgradeFor("nonsense", [12.0]), null, "no readable version");
});

Deno.test("cuda: the driver is the ceiling, and it is read before anything is offered", () => {
  // A toolkit newer than the driver produces binaries the machine cannot load.
  // Installing one would turn a slow build into a broken one, so when the
  // driver is the limit the app says nothing at all.
  assertEquals(
    driverCudaVersion("| NVIDIA-SMI 610.43.02   CUDA UMD Version: 13.3 |"),
    13.3,
  );
  assertEquals(
    driverCudaVersion("| NVIDIA-SMI 550.90   CUDA Version: 12.4 |"),
    12.4,
  );
  assertEquals(driverCudaVersion("no version here"), 0);

  const caps = [12.0];
  // This machine, exactly: driver runs 13.3, toolkit is 12.0.
  assertExists(cudaOffer("release 12.0", caps, 13.3));
  // A driver that cannot run what the card needs: installing would achieve
  // nothing, so nothing is offered.
  assertEquals(cudaOffer("release 12.0", caps, 12.4), null);
  assertEquals(
    cudaOffer("release 12.0", caps, 0),
    null,
    "unknown ≠ optimistic",
  );
});

Deno.test("cudaredist: only the pieces a llama.cpp build reaches for", () => {
  const keys = CUDA_COMPONENTS.map((c) => c.key);
  for (const must of ["cuda_nvcc", "cuda_cudart", "libcublas", "cccl"]) {
    assert(keys.includes(must), `${must} is required to build ggml-cuda`);
  }
  // Nothing that installs a driver or a kernel module — the entire safety
  // claim of this fix rests on this list.
  for (const never of ["nvidia_driver", "nvidia_fs", "fabricmanager", "imex"]) {
    assertEquals(keys.includes(never), false, `${never} must never be fetched`);
  }
  for (const c of CUDA_COMPONENTS) {
    assert(c.why.length > 0, `${c.key} must say why it is needed`);
  }
});

Deno.test("cudaredist: a manifest becomes checksummed downloads, and a rename is survivable", () => {
  // Shape taken from NVIDIA's real redistrib_13.3.1.json.
  const manifest = {
    cuda_nvcc: {
      "linux-x86_64": {
        relative_path:
          "cuda_nvcc/linux-x86_64/cuda_nvcc-13.3.73-archive.tar.xz",
        sha256: "a".repeat(64),
        size: "31600000",
      },
    },
    libcublas: {
      "linux-x86_64": {
        relative_path:
          "libcublas/linux-x86_64/libcublas-13.6.0.2-archive.tar.xz",
        sha256: "b".repeat(64),
        size: "818000000",
      },
    },
  };
  const { files, totalB, missing } = cudaFiles(manifest);
  assertEquals(files.length, 2);
  assertEquals(totalB, 849_600_000);
  assertEquals(files[0]?.sha256, "a".repeat(64));
  assertStringIncludes(files[0]?.url ?? "", "developer.download.nvidia.com");
  // Everything else is reported absent rather than throwing: NVIDIA renames
  // components between releases, and refusing to install over a name change in
  // a release nobody asked about is worse than a missing optional header.
  assert(missing.includes("cccl"));
  assertEquals(cudaFilesUsable(files), true);
  // Without a compiler there is nothing to install, and that IS fatal.
  assertEquals(
    cudaFilesUsable(files.filter((f) => f.key !== "cuda_nvcc")),
    false,
  );
  assertEquals(cudaFiles({}).files.length, 0);
  assertEquals(cudaFiles(null).files.length, 0);
});

Deno.test("cudaredist: disk is checked with room for the unpacking, not just the download", () => {
  // Measured: 929 MB of archives became 2.0 GB installed, and both exist at
  // once. On a disk at 99% full, refusing politely beats running it to zero.
  const oneGb = 1024 ** 3;
  assert(cudaDiskNeededB(oneGb) >= 3 * oneGb, "the unpacked size counts too");
  assert(cudaDiskNeededB(0) > 0, "there is always some slack");
});

Deno.test("cudaredist: the fix says what it will NOT do, before it does anything", () => {
  const said = cudaFixSummary("13.3.1", 1024 ** 3, "/home/u/.llama-master/x")
    .join(" ");
  assertStringIncludes(said, "No administrator rights");
  assert(/[Nn]o driver/.test(said), "the driver is left alone, and it says so");
  assertStringIncludes(said, "SHA-256");
  assertStringIncludes(said, "delete that folder");
});

Deno.test("fix: the CUDA fix is a download, never a package or a script", () => {
  // `apt install cuda` pulls a new DRIVER, and swapping a driver under a
  // running desktop is the one failure a Fix button must never be able to
  // cause. `download` is the app doing it itself, with no elevation at all.
  const plan = fixPlan("cuda-arch", "linux", "apt");
  assertEquals(plan.kind, "download");
  if (plan.kind !== "download") throw new Error("unreachable");
  assertStringIncludes(plan.label, "no root");
  assertStringIncludes(plan.label, "no driver change");
});

// ── The bench pair, and what it settles ────────────────────────────────────
//
// One rate could not answer "should speculative decoding be on", and the tuner
// was pointing at this panel to answer exactly that. The pair can, because with
// no drafter the two prompts agree — so the gap that appears with one attached
// is the drafter's, measured here rather than quoted from someone else's rig.

const benched = (kind: "prose" | "code", genTps: number): BenchResult => ({
  ...EMPTY_BENCH,
  kind,
  genTps,
  genTokens: 128,
  promptTokens: 40,
  promptTps: 300,
  at: 1,
  ctx: 4096,
  modelPath: "/m.gguf",
});

Deno.test("bench: the two prompts differ in the one property that matters", () => {
  assertEquals(benchRequest("prose").prompt, BENCH_PROMPTS.prose);
  assertEquals(benchRequest("code").prompt, BENCH_PROMPTS.code);
  assert(
    BENCH_PROMPTS.prose !== BENCH_PROMPTS.code,
    "two names for one prompt would measure the same thing twice",
  );
  // The code prompt has to actually produce repetitive output, or it measures
  // prose with a different vocabulary and the whole comparison is noise.
  assert(
    /typescript|function/i.test(BENCH_PROMPTS.code),
    "the code prompt must ask for code",
  );
  // Both must refuse thinking: a reasoning model that spends the budget
  // thinking measures the right rate on the wrong work.
  for (const k of ["prose", "code"] as const) {
    assert(
      /do not think/i.test(BENCH_PROMPTS[k]),
      `${k} prompt must suppress thinking`,
    );
    assertEquals(benchRequest(k).temperature, 0);
    assertEquals(benchRequest(k).cache_prompt, false);
  }
  // The default is prose, because prose is what calibrates bandwidth.
  assertEquals(benchRequest().prompt, BENCH_PROMPTS.prose);
});

Deno.test("bench: a result carries the prompt it came from", () => {
  const r = parseBench({
    timings: { predicted_n: 64, predicted_per_second: 9 },
  }, {
    kind: "code",
    latencyMs: 10,
    ctx: 4096,
    modelPath: "/m.gguf",
    at: 7,
  });
  assertEquals(r?.kind, "code");
});

Deno.test("specVerdict: half a comparison is not published", () => {
  assertEquals(specVerdict(null, null, ""), null);
  assertEquals(
    specVerdict(benched("prose", 20), null, ""),
    null,
    "prose alone",
  );
  assertEquals(specVerdict(null, benched("code", 60), ""), null, "code alone");
  // A code rate with no baseline is the "2.9x faster!" number with nothing
  // under it. That is the reading this exists to refuse.
});

Deno.test("specVerdict: drafting on and paying is reported as a multiple", () => {
  const v = specVerdict(benched("prose", 20), benched("code", 58), "draft-mtp");
  assertEquals(v?.tone, "ok");
  assert(v!.ratio > 2.8 && v!.ratio <= 2.9, `ratio ${v!.ratio}`);
  assert(v!.headline.includes("2.90x"), v!.headline);
  assert(/lossless/i.test(v!.advice), "the reason it is free must be said");
});

Deno.test("specVerdict: drafting on and NOT paying names the cost", () => {
  // Rejected drafts are work thrown away, so "on and slower" is a real and
  // actionable state — and the one a user would never guess at.
  const v = specVerdict(
    benched("prose", 20),
    benched("code", 15),
    "ngram-cache",
  );
  assertEquals(v?.tone, "caution");
  assert(/SLOWER/.test(v!.headline), v!.headline);
  assert(/switch/i.test(v!.advice), "must name the way out");
});

Deno.test("specVerdict: MTP that is not paying is still kept", () => {
  // The model's own block is already loaded — its weights are in the file
  // whether it drafts or not — so "off" buys nothing back. An n-gram kind is a
  // different answer, because it is a choice with alternatives.
  const mtp = specVerdict(
    benched("prose", 20),
    benched("code", 21),
    "draft-mtp",
  );
  assertEquals(mtp?.tone, "info");
  assert(/Leave it/i.test(mtp!.advice), mtp!.advice);
  const ngram = specVerdict(
    benched("prose", 20),
    benched("code", 21),
    "ngram-mod",
  );
  assert(/worth trying|switch it off/i.test(ngram!.advice), ngram!.advice);
});

Deno.test("specVerdict: drafting off, agreement IS the finding", () => {
  const v = specVerdict(benched("prose", 20), benched("code", 20.4), "");
  assertEquals(v?.tone, "info");
  assert(/baseline/i.test(v!.headline), v!.headline);
  assert(/k4v/.test(v!.advice), "must name the setting to try next");
});

Deno.test("specVerdict: drafting off with a gap blames the tokenizer, not the machine", () => {
  // With no drafter involved the model reads the same weights either way, so a
  // gap cannot be speed. Saying "code is 30% faster" here would invent a
  // hardware finding out of a tokenization detail.
  const v = specVerdict(benched("prose", 20), benched("code", 26), "");
  assert(/tokenizer/i.test(v!.headline), v!.headline);
});

// ── The drafter that ships beside the model ───────────────────────────────

Deno.test("mtp: the marker is a token, never a substring", () => {
  assert(isMtpName("gemma-4-E4B-it-mtp-Q4_0.gguf"));
  assert(isMtpName("model.mtp.gguf"));
  assert(isMtpName("mtp-head-Q8_0.gguf"));
  assert(isMtpName("something_mtp.gguf"));
  // A model whose name merely contains the letters is not a drafter, and
  // pairing one would hand llama.cpp a second full model whose drafts are
  // never accepted — a silent slowdown, the worst failure available here.
  assertEquals(isMtpName("Mtptune-7B-Q4_K_M.gguf"), false);
  assertEquals(isMtpName("prompt-Q4_0.gguf"), false);
});

Deno.test("mtp: quantisation labels do not have to agree", () => {
  // A drafter is routinely published at a different quant from the model it
  // drafts for; requiring a match would reject the exact pairing sought.
  assertEquals(
    pairKey("gemma-4-E4B-it-Q8_0.gguf"),
    pairKey("gemma-4-E4B-it-mtp-Q4_0.gguf"),
  );
  assert(
    pairKey("gemma-4-E4B-it-Q8_0.gguf") !== pairKey("gemma-4-E2B-it-Q8_0.gguf"),
    "different models must not pair",
  );
});

Deno.test("mtp: the sibling is found, and only the right one", () => {
  const dir = "/models/gemma";
  const f = (file: string, sizeB = 300 * 1024 ** 2, d = dir) => ({
    path: `${d}/${file}`,
    file,
    sizeB,
    dir: d,
  });
  const main = f("gemma-4-E4B-it-Q4_0.gguf", 4 * 1024 ** 3);
  const draft = f("gemma-4-E4B-it-mtp-Q4_0.gguf");
  const other = f("Qwen3.5-9B-mtp-Q4_0.gguf");
  const elsewhere = f(
    "gemma-4-E4B-it-mtp-Q4_0.gguf",
    300 * 1024 ** 2,
    "/other",
  );
  const list = [main, draft, other, elsewhere];

  assertEquals(findMtpSibling(main.path, list)?.path, draft.path);
  // A drafter does not draft for itself.
  assertEquals(findMtpSibling(draft.path, list), null);
  // A drafter for a different family is not offered: the vocabularies differ,
  // so every draft would be rejected.
  assertEquals(
    findMtpSibling(main.path, [main, other]),
    null,
    "wrong family must not pair",
  );
  // Nor one filed somewhere else — that is a guess about somebody's folders.
  assertEquals(findMtpSibling(main.path, [main, elsewhere]), null);
  assertEquals(findMtpSibling("/nope.gguf", list), null);
});

Deno.test("tune: a separate MTP head is attached, and a huge one is not", () => {
  // The header is honest and incomplete: Gemma 4 keeps its heads in another
  // file, so `nextnLayers` is 0 and the old tuner printed "this model ships no
  // multi-token-prediction block" over a 2-3x speed-up in the same directory.
  const m = meta({ name: "Gemma", nextnLayers: 0, nCtxTrain: 8192 });
  const machine = hw({ gpus: [gpu(24, 0)] });
  const sib = {
    path: "/models/g-mtp-Q4_0.gguf",
    file: "g-mtp-Q4_0.gguf",
    sizeB: 300 * 1024 ** 2,
  };

  const off = tune(m, machine, defaults(), "vram");
  assertEquals(off.settings.specType, "", "no sibling, no drafting");

  const on = tune(m, machine, defaults(), "vram", undefined, undefined, false, {
    mtpSibling: sib,
  });
  assertEquals(on.settings.specType, "draft-mtp");
  assertEquals(on.settings.draftModel, sib.path);
  assert(
    on.reasons.some((r) => r.includes("separate file")),
    "the reason must say the head was not in the model file",
  );

  // A draft model's weights are VRAM `plan.ts` does not bill, so an
  // implausibly large "drafter" is named rather than spent.
  const big = tune(
    m,
    machine,
    defaults(),
    "vram",
    undefined,
    undefined,
    false,
    {
      mtpSibling: { ...sib, sizeB: 9 * 1024 ** 3 },
    },
  );
  assertEquals(big.settings.specType, "", "too big to attach unbudgeted");
  assertEquals(big.settings.draftModel, "");
  assert(
    big.reasons.some((r) => r.includes("larger than a drafting head")),
    "and it must say why it declined",
  );
});

Deno.test("stability: GPU sampling beside a drafter is NOT a conflict", () => {
  // It was, and this app said so for one round of work, reading a source cache
  // six weeks stale: `backend_sampling &= !(slot.can_speculate())` was removed
  // from llama.cpp between 2026-07-27 and 2026-09-07. The two stack now, and a
  // warning here would talk the user out of the better configuration. The
  // regression guarded is the WARNING coming back, not the flag.
  const m = meta({ name: "M", nCtxTrain: 4096 });
  const machine = hw({ gpus: [gpu(24, 0)] });
  const both = stability(m, machine, {
    ...defaults(),
    backendSampling: true,
    specType: "draft-mtp",
  });
  assertEquals(
    both.warnings.some((x) => x.key === "backendSampling"),
    false,
    "drafting and GPU sampling are compatible upstream",
  );

  // What IS still dropped, and still worth naming: a reasoning budget builds a
  // sampler llama.cpp cannot run on the device, and it says so only in the log.
  const thinking = stability(m, machine, {
    ...defaults(),
    backendSampling: true,
    reasoning: "on",
  });
  const w = thinking.warnings.find((x) => x.key === "backendSampling");
  assert(w, "the reasoning-budget drop must still be reported");
  assert(/reasoning/i.test(w!.message), w!.message);
});

// ── What THIS build can be told to do ─────────────────────────────────────

Deno.test("caps: the help text is parsed, and its prose is not", () => {
  // Real llama.cpp layout: aliases separated by comma-and-spaces, a value
  // placeholder, then the description at a fixed column, then indented
  // continuation lines.
  const help = [
    "----- common params -----",
    "",
    "-h,    --help, --usage                  print usage and exit",
    "-t,    --threads N                      number of CPU threads (default: -1)",
    "                                        (env: LLAMA_ARG_THREADS)",
    "-tb,   --threads-batch N                number of threads, same as --threads",
    "--cpu-strict <0|1>                      use strict CPU placement (default: 0)",
    "-Cr,   --cpu-range lo-hi                range of CPUs for affinity",
    "-bs,   --backend-sampling               enable backend sampling (experimental)",
  ].join("\n");
  const f = parseHelpFlags(help);

  // Every alias of every option, short and long.
  for (
    const k of [
      "-h",
      "--help",
      "--usage",
      "-t",
      "--threads",
      "-bs",
      "--backend-sampling",
      "--cpu-strict",
      "-Cr",
      "--cpu-range",
    ]
  ) {
    assert(f.has(k), `missing ${k}`);
  }
  // Value placeholders are not flags, however they are spelled.
  for (const junk of ["N", "<0|1>", "lo-hi", "FNAME", "-1"]) {
    assertEquals(f.has(junk), false, `${junk} is not a flag`);
  }
  // A description that MENTIONS a flag does not declare it. `--threads` is
  // real here only because line 2 declares it; the point is that the parse cuts
  // the description off before looking, so a build could not be credited with a
  // flag that appears only in someone else's prose.
  assertEquals(f.has("params"), false);
  assertEquals(f.has("--usage") && f.has("--cpu-range"), true);
  // Headings and blank lines are not options.
  assertEquals(f.has("-----"), false);
});

Deno.test("caps: an unprobed build supports nothing", () => {
  // The safe reading of "no answer" is "switch nothing extra on". Reading it as
  // "assume modern" is how a tuner writes a command the binary cannot parse.
  assertEquals(supportsFlag(null, "-bs"), false);
  assertEquals(supportsFlag(undefined, "-bs"), false);
  assertEquals(supportsFlag([], "-bs"), false);
  assertEquals(supportsFlag(["-bs"], "-bs"), true);
  assertEquals(supportsFlag(["-bs"], "--fit"), false);
});

Deno.test("tune: GPU sampling is taken only when the build declares it", () => {
  const m = meta({ name: "M", nCtxTrain: 4096 });
  const machine = hw({ gpus: [gpu(24, 0)] });
  const T = (o: Record<string, unknown>, placement = "vram") =>
    tune(
      m,
      machine,
      defaults(),
      placement as "vram",
      undefined,
      undefined,
      false,
      o,
    );

  // Probed and present: taken. It costs no memory, it composes with drafting,
  // and it removes a per-token stall.
  const modern = T({ caps: ["-bs", "-ngl", "-c"] });
  assertEquals(modern.settings.backendSampling, true);
  assert(modern.reasons.some((r) => r.includes("Sampling runs on the GPU")));

  // Probed and absent: NOT taken — this is the one setting an older binary
  // refuses outright — and the build's age is named rather than left as an
  // unexplained difference in speed.
  const old = T({ caps: ["-ngl", "-c"] });
  assertEquals(old.settings.backendSampling, false);
  assert(
    old.reasons.some((r) => r.includes("does not know `-bs`")),
    "an absent flag must be explained, not silent",
  );

  // Never probed: not taken, and not complained about either — the app has no
  // finding to report about a question it did not ask.
  const unknown = T({});
  assertEquals(unknown.settings.backendSampling, false);
  assertEquals(
    unknown.reasons.some((r) => r.includes("-bs")),
    false,
    "silence about an unprobed build",
  );

  // CPU-only: there is no device to keep the logits on, so the flag would be a
  // claim about nothing.
  assertEquals(T({ caps: ["-bs"] }, "cpu").settings.backendSampling, false);
});
