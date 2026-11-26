// tests/basalt.test.ts — the Blackwell engine: which files it stands in for,
// the argv it is handed, and what the app says about it.

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { diagnoseServerExit } from "../src/lib/serverlog.ts";
import { enginePlan } from "../src/lib/plan.ts";
import { gpu, hw } from "./fixtures.ts";
import {
  BASALT_REF,
  basaltArgs,
  basaltCap,
  basaltCard,
  basaltContext,
  basaltMaxContext,
  basaltMismatch,
  basaltModel,
  basaltOffer,
  basaltParts,
  basaltSplitMaxContext,
  INKLING_PACK,
  isBasaltRef,
} from "../src/lib/basalt.ts";
import { packEngine, packInfo, packMismatch } from "../src/lib/packs.ts";
import {
  isStrataRef,
  strataModel,
  strataNeeds,
  strataRunOf,
  strataSettings,
  strataSetupRows,
} from "../src/lib/strata.ts";
import { loadPhase } from "../src/lib/loadprogress.ts";
import { parseRef, refMoves } from "../src/lib/srcref.ts";
import { defaults } from "../src/lib/params.ts";

const LM =
  "/m/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf";

Deno.test("basalt: its ref is a fork that follows its default branch", () => {
  assertEquals(BASALT_REF, "fork:jesdga95/basalt");
  assert(isBasaltRef(BASALT_REF));
  assert(!isStrataRef(BASALT_REF));
  assert(!isBasaltRef("fork:Niko1221/Strata"));
  assert(refMoves(parseRef(BASALT_REF)));
});

Deno.test("basalt: official Inkling quantisations only — a fine-tune is other weights", () => {
  for (
    const f of [
      LM,
      "/m/Inkling-2-Mini-UD-Q4_K_XL.gguf",
      "/m/Inkling-2-Mini-IQ4_NL.gguf",
      "/m/Inkling-2-Mini-BF16-00001-of-00002.gguf",
    ]
  ) assert(basaltModel(f), f);
  for (
    const f of [
      "/m/Inkling-2-Mini-Uncensored-Q4_K_M.gguf",
      "/m/Inkling-2-Mini-Heretic-GGUF/Inkling-2-Mini-Heretic-Q4_K_M.gguf",
      "/m/mmproj-Inkling-2-Mini-BF16.gguf",
      "/m/Qwen3.8-27B-Q4_K_M.gguf",
    ]
  ) assertEquals(basaltModel(f), null, f);
  const m = basaltModel(LM)!;
  assertEquals([m.cards, m.packGB, m.ramGB, m.title], [
    "one",
    19.6,
    1.1,
    "Inkling 2 Mini",
  ]);
});

Deno.test("basalt: the argv names the pack, the card, an unrounded cache and the reserve", () => {
  const a = basaltArgs({
    buildDir: "/b/source-fork-jesdga95_basalt-cuda",
    modelPath: LM,
    model: basaltModel(LM)!,
    ctx: 100_000,
    reserveB: 4 * 2 ** 30,
    ramReserveB: 0,
    host: "127.0.0.1",
    port: 18080,
    gpus: [1],
    kv: "fp16",
  });
  const at = (f: string) => a[a.indexOf(f) + 1];
  assertEquals(a.slice(0, 2), [
    "serve",
    `/b/source-fork-jesdga95_basalt-cuda/packs/${INKLING_PACK.file}`,
  ]);
  assertEquals(at("--gpu"), "1");
  assertEquals(at("--kv"), "fp16", "fp8 rounds the cache");
  assertEquals(at("--max-context"), "65536", "rounded DOWN to a step");
  assertEquals(at("--vram-reserve-mib"), String(700 + 4096));
  assertEquals(at("--ram-reserve-mib"), "2048", "Basalt's own floor");
  // Nothing lossy, ever (Basalt's docs/precision/quality.md).
  for (const f of ["--route-prior", "--route-drop", "--kv-resident"]) {
    assertEquals(a.includes(f), false, f);
  }
  // The live view reads its run back from this same argv.
  assertEquals(strataRunOf(a), {
    ctx: 65536,
    reserveMib: 4796,
    gpus: [1],
    kv: "fp16",
  });
  assertEquals(a.includes("--gpu-mode"), false, "one card is not a split");
});

// Measured 2026-11-25 on a 24 GB card, 24,450 MiB free as the app counts it:
// 1,048,576 was refused 19,084 MiB short, 524,288 652 short, 262,144 ran.
const FREE_24G = 24_450 * 2 ** 20;

Deno.test("basalt: the context is capped at what the card can hold", () => {
  assertEquals(basaltMaxContext(FREE_24G, 700), 393_216);
  // The user's reserve comes out of the same card.
  assertEquals(basaltMaxContext(FREE_24G, 700 + 12_288), 131_072);
  // A 48 GB card holds the whole 1M cache.
  assertEquals(basaltMaxContext(48_900 * 2 ** 20, 700), 1_048_576);
  // Nothing fits: the smallest step, and Basalt says why.
  assertEquals(basaltMaxContext(0, 700), 8192);
  const run = {
    buildDir: "/b",
    modelPath: LM,
    model: basaltModel(LM)!,
    ctx: 1_048_576,
    reserveB: 0,
    ramReserveB: 0,
    host: "127.0.0.1",
    port: 18080,
    gpus: [1],
    kv: "fp16" as const,
  };
  const ctxOf = (a: string[]) => a[a.indexOf("--max-context") + 1];
  // The reported failure: a 1M context from another model, on a 24 GB card.
  assertEquals(ctxOf(basaltArgs({ ...run, freeB: [FREE_24G] })), "393216");
  // Unknown free memory: no cap (Basalt itself refuses what does not fit).
  assertEquals(ctxOf(basaltArgs(run)), "1048576");
});

Deno.test('basalt: its own refusal is the diagnosis, not "printed nothing"', () => {
  const d = diagnoseServerExit(1, [
    "[server] planning weight allocation",
    "[server] error: inkling2: spill plan: even with every expert in RAM the card is 19084 MiB short (--max-context 1048576)",
    "[server] stopped",
  ]);
  assertStringIncludes(d.reason, "1,048,576-token context");
  assertStringIncludes(d.reason, "19,084 MiB short");
  assertEquals(d.steps.some((s) => s.action?.kind === "open-tab"), false);
});

Deno.test("basalt: contexts are steps, and settings record the one that ran", () => {
  assertEquals(basaltContext(1), 8192);
  assertEquals(basaltContext(262_144), 262_144);
  assertEquals(basaltContext(2_000_000), 1_048_576);
  // The run records what its argv carries: rounded to a step, and capped to
  // the card (the reported 1M on a 24 GB card ran — and benched — at 262K).
  const argv = (ctx: number, freeB?: number[]) => [
    "/b/bin/basalt",
    ...basaltArgs({
      buildDir: "/b",
      modelPath: LM,
      model: basaltModel(LM)!,
      ctx,
      reserveB: 0,
      ramReserveB: 0,
      host: "127.0.0.1",
      port: 18080,
      gpus: [1],
      kv: "fp16",
      freeB,
    }),
  ];
  const ran = (ctx: number, freeB?: number[]) =>
    strataSettings({ ...defaults(), ctxSize: ctx }, argv(ctx, freeB)).ctxSize;
  assertEquals(ran(200_000), 131_072);
  assertEquals(ran(1_048_576, [24_450 * 2 ** 20]), 393_216);
});

Deno.test("basalt: memory is the card's free VRAM and a fixed host share", () => {
  const m = basaltModel(LM)!;
  const n = strataNeeds({
    fileB: 18.66e9,
    freeVramB: 24e9,
    cards: 1,
    reserveMib: 700,
    fixedRamB: m.ramGB! * 1e9,
  });
  assertEquals(n.ramB, 1.1e9, "not a share of the GGUF it stands in for");
  const rows = strataSetupRows({
    model: m,
    ctx: 65536,
    reserveMib: 700,
    cards: 1,
    ramB: n.ramB,
    name: "Basalt",
  });
  assert(rows[0]!.value.startsWith("Basalt · Inkling 2 Mini"));
  assert(!rows.some((r) => /Strata/.test(r.value + r.tip)), "named as Basalt");
});

Deno.test("basalt: offered only on Blackwell, never to itself, and blocks other models", () => {
  const llama = [{ id: "source-master-cuda", ref: "master" }];
  const blackwell = [12];
  assert(basaltCard(blackwell));
  assert(!basaltCard([8.9, 9]), "Ada and Hopper cannot load sm_120 kernels");
  const offer = basaltOffer(LM, "master", llama, blackwell)!;
  assertEquals(offer.steps[0]!.action, { kind: "use-ref", ref: BASALT_REF });
  assert(/19\.6 GB/.test(offer.steps[0]!.text), "the download is stated");
  assertEquals(basaltOffer(LM, "master", llama, [8.9]), null);
  assertEquals(basaltOffer(LM, BASALT_REF, llama, blackwell), null);
  assertEquals(
    basaltOffer("/m/Qwen3.8-27B-Q4_K_M.gguf", "master", llama, blackwell),
    null,
  );
  const have = [...llama, {
    id: "source-fork-jesdga95_basalt-cuda",
    ref: BASALT_REF,
  }];
  assertEquals(
    basaltOffer(LM, "master", have, blackwell)!.steps[0]!.action,
    { kind: "use-build", id: "source-fork-jesdga95_basalt-cuda" },
  );
  // A Basalt build asked for another model: blocked, with the way back.
  const block = basaltMismatch("/m/Qwen3.8-27B-Q4_K_M.gguf", BASALT_REF, have)!;
  assertEquals(block.steps[0]!.action, {
    kind: "use-build",
    id: "source-master-cuda",
  });
  assertEquals(basaltMismatch(LM, BASALT_REF, have), null);
  assertEquals(
    basaltMismatch("/m/Qwen3.8-27B-Q4_K_M.gguf", "master", have),
    null,
  );
});

Deno.test("basalt: the load bar reads its log, and its early `ready` is not a phase", () => {
  const log = [
    "[server] Inkling 2 Mini  kv=fp16 ctx=65536 spec=8 temp=1.0",
    "[server] ready    http://127.0.0.1:18080/v1  openai+anthropic",
  ];
  assertEquals(loadPhase(log), "starting");
  log.push("[server] planning weight allocation");
  assertEquals(loadPhase(log), "loading weights");
  log.push("[server] placed   cuda1 16.0GiB 3.2s");
  assertEquals(loadPhase(log), "loading weights");
  log.push("[server] loaded   17.0GiB 3.6s  cuda1=16.0GiB ram=1.0GiB");
  assertEquals(loadPhase(log), "almost ready");
});

Deno.test("packs: a pack is the model itself, run from where it is", () => {
  const pack = `/home/u/models/${INKLING_PACK.file}`;
  const m = basaltModel(pack)!;
  assertEquals([m.model, m.title], ["NVFP4-MX8", "Inkling 2 Mini"]);
  const run = {
    buildDir: "/b",
    model: m,
    ctx: 32768,
    reserveB: 0,
    ramReserveB: 0,
    host: "127.0.0.1",
    port: 8080,
    gpus: [0],
    kv: "fp16" as const,
  };
  assertEquals(basaltArgs({ ...run, modelPath: pack })[1], pack);
  // A GGUF only names the model: the pack runs from the build's `packs/`.
  assertEquals(
    basaltArgs({ ...run, modelPath: LM })[1],
    `/b/packs/${INKLING_PACK.file}`,
  );
  assertEquals(basaltModel("/m/Other-Model.basalt"), null);
  const sq = strataModel(
    "/c/strata/models/inkling-2-mini/Inkling-2-Mini-SQ-20.strata",
  )!;
  assertEquals([sq.model, sq.packGB, sq.cards], ["SQ-20", 19.35, "one"]);
  assertEquals(packEngine("/m/x.BASALT"), "basalt");
  assertEquals(packEngine(LM), null);
  assertEquals(packInfo(pack), {
    engine: "Basalt",
    title: "Inkling 2 Mini",
    quant: "NVFP4-MX8",
    maxContext: 1_048_576,
  });
  assertEquals(packInfo("/m/Other-Model.basalt"), null);
});

Deno.test("packs: only the pack's own engine reads it — any other build is blocked", () => {
  const pack = `/m/${INKLING_PACK.file}`;
  const llama = [{ id: "source-master-cuda", ref: "master" }];
  const have = [...llama, {
    id: "source-fork-jesdga95_basalt-cuda",
    ref: BASALT_REF,
  }];
  // A GGUF is never this rule's business.
  assertEquals(packMismatch(LM, "master", have, [12]), null);
  // Its own engine judges it by its own rules.
  assertEquals(packMismatch(pack, BASALT_REF, have, [12]), null);
  assertEquals(
    packMismatch(pack, "master", have, [12])!.steps[0]!.action,
    { kind: "use-build", id: "source-fork-jesdga95_basalt-cuda" },
  );
  assertEquals(
    packMismatch(pack, "fork:Niko1221/Strata", llama, [12])!.steps[0]!.action,
    { kind: "use-ref", ref: BASALT_REF },
  );
  // No Blackwell card: nothing to build, and it says why.
  const old = packMismatch(pack, "master", have, [8.9])!;
  assert(/Blackwell/.test(old.reason));
  assertEquals(old.steps.length, 1);
  // A Basalt pack of a model this app does not run with Basalt.
  assert(
    /does not run/.test(
      packMismatch("/m/X.basalt", "master", have, [12])!.reason,
    ),
  );
  const sq = "/c/Inkling-2-Mini-SQ-20.strata";
  assertEquals(packMismatch(sq, "fork:Niko1221/Strata", have, [12]), null);
  assertEquals(
    packMismatch(sq, BASALT_REF, have, [12])!.steps[0]!.action,
    { kind: "use-ref", ref: "fork:Niko1221/Strata" },
  );
});

// Trimmed from a live `/metrics` (engine 0.5.7, Inkling 2 Mini at 262K on
// nvidia-smi card 1): the three parts sum to the card's `mem_used`.
const METRICS = {
  engine: { model: "Inkling 2 Mini", max_context: 262144 },
  hardware: {
    gpus: [{
      index: 1,
      mem_used: 22_287_941_632,
      mem_total: 25_655_508_992,
      vram_parts: {
        experts: 9_324_986_368,
        kv: 8_759_803_904,
        other: 4_203_151_360,
      },
    }],
    ram_parts: { experts: 6_344_933_376, page_cache: 152e9, other: 37e9 },
  },
};

Deno.test("basalt: /metrics is read per card, and anything else is nothing", () => {
  assertEquals(basaltParts(METRICS), {
    cards: [{
      index: 1,
      expertsB: 9_324_986_368,
      kvB: 8_759_803_904,
      otherB: 4_203_151_360,
    }],
    ramExpertsB: 6_344_933_376,
  });
  for (
    const junk of [null, "# HELP llamacpp", {}, { hardware: { gpus: [{}] } }]
  ) {
    assertEquals(basaltParts(junk), null);
  }
  // A hostile figure is clamped, never NaN.
  const bad = basaltParts({
    hardware: { gpus: [{ index: 0, vram_parts: { kv: -5, experts: "x" } }] },
  });
  assertEquals(bad?.cards[0]?.kvB, 0);
  assertEquals(bad?.cards[0]?.expertsB, 0);
});

Deno.test("basalt: the live map is the engine's own figures, on its own card", () => {
  const GB = 1e9;
  const parts = basaltParts(METRICS)!;
  // Card 0 draws the desktop; card 1 holds the run (nvidia-smi read lower
  // than the engine's own total, which counts its CUDA context).
  const machine = hw({
    gpus: [gpu(24, 3.5), gpu(24, 20.3)],
    mem: {
      totalB: 186 * GB,
      availableB: 150 * GB,
      usedB: 36 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
  });
  const p = enginePlan(machine, parts, 262_144, 10 * GB);
  const [c0, c1] = p.devices.cards;
  assertEquals([c0!.weightsB, c0!.expertsB, c0!.kvB], [0, 0, 0]);
  assertEquals(
    c0!.otherB,
    machine.gpus[0]!.vramUsedB,
    "the desktop's, not ours",
  );
  assertEquals(c1!.expertsB, 9_324_986_368);
  assertEquals(c1!.kvB, 8_759_803_904);
  assertEquals(c1!.weightsB, 4_203_151_360);
  assertEquals(c1!.otherB, 0, "never negative");
  assertEquals(p.vram.usedB, 22_287_941_632);
  assertEquals(p.vram.otherB, machine.gpus[0]!.vramUsedB);
  assertEquals(
    p.vram.buckets.map((b) => b.key),
    ["weights", "experts", "kv"],
  );
  // RAM: the experts the engine moved there, and the rest of its resident set.
  assertEquals(
    p.ram.buckets.find((b) => b.key === "experts")?.bytes,
    6_344_933_376,
  );
  assertEquals(p.ram.usedB, 10 * GB);
  assertEquals(p.ram.otherB, 26 * GB, "in use minus ours");
  assertEquals(p.ctx, 262_144);
  assert(p.fits);
});

// Measured 2026-11-25: one 24 GB card (24,450 MiB free) and the display card
// beside it (21,313 MiB free), 700 MiB kept free on each.
const FREE_DISPLAY = 21_313 * 2 ** 20;

Deno.test("basalt: a smaller cache buys context on one card, never 1M on 24 GB", () => {
  // At 1M every format was refused: fp16 19,084 MiB short, int8 4,677, fp8 2,619.
  assertEquals(basaltMaxContext(FREE_24G, 700, "fp16"), 393_216);
  assertEquals(basaltMaxContext(FREE_24G, 700, "int8"), 786_432);
  assertEquals(basaltMaxContext(FREE_24G, 700, "fp8"), 786_432);
});

Deno.test("basalt: a layer split reaches 1M with fp8, and not with int8", () => {
  const both = [FREE_24G, FREE_DISPLAY];
  // fp8 1M ran (2.9 + 1.4 GB left); int8 1M was refused 1 GB short.
  assertEquals(basaltSplitMaxContext(both, 700, "fp8"), 1_048_576);
  assertEquals(basaltSplitMaxContext(both, 700, "int8"), 786_432);
  assertEquals(basaltSplitMaxContext(both, 700, "fp16"), 524_288);
  // The user's 4 GB charged to BOTH cards is what refused 524K at fp16.
  assertEquals(basaltSplitMaxContext(both, 4796, "fp16"), 393_216);
  // One entry is one card; two are a split.
  assertEquals(basaltCap([FREE_24G], 700, "fp8"), 786_432);
  assertEquals(basaltCap(both, 700, "fp8"), 1_048_576);
});

Deno.test("basalt: the split argv is the one measured, and reads back", () => {
  const a = basaltArgs({
    buildDir: "/b",
    modelPath: LM,
    model: basaltModel(LM)!,
    ctx: 1_048_576,
    reserveB: 0,
    ramReserveB: 0,
    host: "127.0.0.1",
    port: 18080,
    gpus: [1, 0],
    freeB: [FREE_24G, FREE_DISPLAY],
    kv: "fp8",
  });
  const at = (f: string) => a[a.indexOf(f) + 1];
  assertEquals(at("--gpu"), "1,0", "the roomier card leads");
  assertEquals(at("--gpu-mode"), "split");
  assertEquals(
    at("--gpu-share"),
    "1,1",
    "Basalt's own share cut at 15 and failed",
  );
  assertEquals(at("--kv"), "fp8");
  assertEquals(at("--max-context"), "1048576");
  assertEquals(at("--vram-reserve-mib"), "700");
  const run = strataRunOf(["/b/bin/basalt", ...a])!;
  assertEquals([run.gpus, run.kv, run.ctx], [[1, 0], "fp8", 1_048_576]);
  // Setup says what is unusual about it.
  const rows = strataSetupRows({
    model: basaltModel(LM)!,
    ctx: run.ctx,
    reserveMib: run.reserveMib,
    cards: run.gpus.length,
    ramB: 1e9,
    name: "Basalt",
    kv: run.kv,
  });
  const short = (l: string) => rows.find((r) => r.label === l)?.short;
  assertEquals(short("Runs on"), "2 GPUs split");
  assertEquals(short("Cache"), "cache fp8");
});

Deno.test("basalt: a split refusal names the context and the fix, either way", () => {
  for (
    const line of [
      "[server] error: inkling2: the layer split at layer 13: card 0 needs 16131 MiB beside its weights for --max-context 1048576, 15133 MiB are free (a smaller --max-context, --kv fp8, or one card)",
      "[server] error: gpu-mode split: GPU 1 (RTX PRO 4000 Blackwell) takes layers 11-23 at share 0.60 and needs 26.0 GiB, 20.2 GiB free; lower its gpu-share, max-context or parallel",
    ]
  ) {
    const d = diagnoseServerExit(1, [line, "[server] stopped"]);
    assertStringIncludes(d.reason, "Split over the cards");
    assert(d.steps.some((s) => s.text.includes("fp8")), line);
  }
});
