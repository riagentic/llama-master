// test/lib.test.ts — the pure core.
//
// Everything the app decides (what the command line is, what fits in VRAM, what
// "optimal" means) is a pure function, and this is where those decisions are
// pinned. No cells, no DOM, no I/O: if one of these fails, the app is wrong, not
// flaky.

import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";

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
  effectiveCtx,
  kvPerToken,
  kvTotal,
  NO_MODEL,
  plan,
  swaSplit,
  withoutOurUsage,
} from "../src/lib/plan.ts";
import {
  bestPlacement,
  CTX_PRESETS,
  ctxLabel,
  MIN_CTX,
  optimalCtx,
  tune,
  tuneAll,
} from "../src/lib/tune.ts";
import { stability } from "../src/lib/stability.ts";
import { autoJobs } from "../src/cell/builds.ts";
import { buildNumberFlags } from "../src/cell/builds.server.ts";
import { bytes, duration, pct, shortPath, tps } from "../src/lib/format.ts";
import { coresUtilPct, pushHistory, utilPct } from "../src/lib/procstat.ts";
import { appendLog, isError, progressOf } from "../src/lib/buildlog.ts";
import { deltaText, parseSse, timingsTps } from "../src/lib/sse.ts";
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
  targetReadiness,
  usableGpus,
} from "../src/lib/backend.ts";
import { diagnoseFailure, diagnoseNoAsset } from "../src/lib/diagnose.ts";
import { isNearBottom, stickToBottom } from "../src/lib/scroll.ts";
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
  cudaPlan,
  cudaVersionForCap,
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
import { gpu, hw, layers, meta, moeMeta, NO_GPU } from "./fixtures.ts";

const GB = 1024 ** 3;

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

Deno.test("params: typed readers fall back to catalog defaults", () => {
  assertEquals(num({}, "ctxSize"), 4096);
  assertEquals(str({}, "host"), "127.0.0.1");
  assertEquals(bool({}, "mlock"), false);
  assertEquals(num({ ctxSize: 32768 }, "ctxSize"), 32768);
});

// ── command building ───────────────────────────────────────────────────────

Deno.test("command: defaults produce only the binary and the model", () => {
  const cmd = argv("server", {
    bin: "/b/llama-server",
    model: "/m/x.gguf",
    settings: defaults(),
  });
  assertEquals(cmd, ["/b/llama-server", "-m", "/m/x.gguf"]);
});

Deno.test("command: only changed values appear, in catalog order", () => {
  const cmd = argv("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), ngl: 99, ctxSize: 16384, flashAttn: "on" },
  });
  assertEquals(cmd, [
    "llama-server",
    "-m",
    "/m/x.gguf",
    "-ngl",
    "99",
    "-c",
    "16384",
    "-fa",
    "on",
  ]);
});

Deno.test("command: a default-on boolean emits its negative flag when off", () => {
  const on = argv("server", {
    bin: "s",
    model: "",
    settings: { ...defaults(), contBatching: true },
  });
  assertEquals(on, ["s"], "on is the default — nothing emitted");
  const off = argv("server", {
    bin: "s",
    model: "",
    settings: { ...defaults(), contBatching: false },
  });
  assertEquals(off, ["s", "--no-cont-batching"]);
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
  assertEquals(cmd, ["s"]);
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
  assertEquals(line, "s -m '/my models/a.gguf'");
});

Deno.test("command: the block form keeps each flag with its value", () => {
  const lines = commandBlock("server", {
    bin: "llama-server",
    model: "/m/x.gguf",
    settings: { ...defaults(), ngl: 99, mlock: true },
  });
  assertEquals(lines, [
    "llama-server",
    "  -m /m/x.gguf",
    "  -ngl 99",
    "  --mlock",
  ]);
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

Deno.test("plan: full offload puts weights, head and embeddings in VRAM", () => {
  const m = meta();
  const p = plan(m, hw({ gpus: [gpu(48)] }), { ...defaults(), ngl: 999 });
  assertEquals(p.layersOnGpu, m.nLayer);
  const w = p.vram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  assertEquals(w, m.tensorBytes);
  assertEquals(
    p.ram.buckets.find((b) => b.key === "weights")?.bytes,
    undefined,
  );
});

Deno.test("plan: partial offload splits weights and KV proportionally", () => {
  const m = meta();
  const p = plan(m, hw(), { ...defaults(), ngl: 16 });
  const vw = p.vram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  const rw = p.ram.buckets.find((b) => b.key === "weights")?.bytes ?? 0;
  assertEquals(vw + rw, m.tensorBytes, "no byte is lost or double-counted");
  const vkv = p.vram.buckets.find((b) => b.key === "kv")?.bytes ?? 0;
  assertEquals(
    Math.round(vkv),
    Math.round(p.kvTotalB / 2),
    "half the layers → half the KV",
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
    attentionOnly + m.embdBytes + m.outputBytes,
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
  assertEquals(settings.threads, 1, "never 0 threads");
  assertEquals(settings.threadsBatch, 1, "prefill too");
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
  const st = stability(meta(), machine, tuned);
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

Deno.test("stability: claiming every core warns about the desktop", () => {
  const st = stability(meta(), hw(), { ...defaults(), threads: 16, ngl: 0 });
  assert(st.warnings.some((w) => w.key === "threads"));
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

// ── build readiness ────────────────────────────────────────────────────────

const FULL = new Set(["cmake", "compiler", "cuda", "vulkan", "spirv", "hip"]);
const BARE = new Set(["cmake", "compiler"]);

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
  assertEquals(argv("server", base), ["llama-server", "-m", "/m.gguf"]);

  // The escape hatch: a flag the catalog does not carry still gets through,
  // as ordinary argv tokens rather than one quoted blob.
  const withExtra = {
    ...base,
    settings: { extraArgs: "--lora /a.gguf  --cache-reuse 256" },
  };
  assertEquals(argv("server", withExtra), [
    "llama-server",
    "-m",
    "/m.gguf",
    "--lora",
    "/a.gguf",
    "--cache-reuse",
    "256",
  ]);
  // And it is visible in the preview, so what you see is still what runs.
  assertStringIncludes(commandLine("server", withExtra), "--cache-reuse 256");
  // Empty stays empty — no stray token.
  assertEquals(argv("server", { ...base, settings: { extraArgs: "   " } }), [
    "llama-server",
    "-m",
    "/m.gguf",
  ]);
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
  assertEquals(restricted.slice(-2), ["-dev", "CUDA0"]);
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

Deno.test("stability: claiming every core for prompt processing warns too", () => {
  // `-tb` had no check at all, at any value — the one thread setting that can
  // freeze a desktop for the length of a long prompt.
  const machine = hw(); // 16 physical / 32 logical
  const over = stability(meta(), machine, { ...defaults(), threadsBatch: 64 });
  assert(
    over.warnings.some((w) =>
      w.key === "threadsBatch" && w.severity === "risk"
    ),
    "more batch threads than logical CPUs is a risk",
  );
  const all = stability(meta(), machine, { ...defaults(), threadsBatch: 16 });
  assert(
    all.warnings.some((w) => w.key === "threadsBatch"),
    "claiming every physical core is at least a caution",
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
  const tiny =
    tune(m, hw({ gpus: [gpu(1, 0)] }), defaults(), "hybrid").settings;
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

  // A window with no pattern means every layer is local (llama.cpp's default).
  assertEquals(
    swaSplit(meta({ ...full, swaWindow: 512, swaPattern: 1 })),
    { full: 0, windowed: 48 },
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
  // Powers of two, because that is how models and benchmarks describe context —
  // and because the KV cache doubles with each rung, so the ladder is also the
  // cost ladder.
  assertEquals(CTX_PRESETS.map(ctxLabel), [
    "16k",
    "32k",
    "64k",
    "128k",
    "256k",
    "512k",
    "1M",
  ]);
  // Ascending, and every rung a real power of two.
  for (let i = 1; i < CTX_PRESETS.length; i++) {
    assertEquals(CTX_PRESETS[i], CTX_PRESETS[i - 1]! * 2);
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
