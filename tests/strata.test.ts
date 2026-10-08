// tests/strata.test.ts — the second engine: which files it takes, what it is
// handed, and what the app says about it.
//
// Pure, like the rest of `src/lib`: Strata's own setup refuses every file it
// has no kernels for, so an offer that matched loosely would build an engine
// that then says no.

import { assert, assertEquals } from "@std/assert";
import {
  isStrataRef,
  STRATA_REF,
  strataArgs,
  strataContext,
  strataMismatch,
  strataModel,
  strataNeeds,
  strataOffer,
  strataRamShort,
  strataReserveMib,
  strataRunOf,
  strataSettings,
  strataSetupRows,
} from "../src/lib/strata.ts";
import { loadPhase } from "../src/lib/loadprogress.ts";
import { argvBlock } from "../src/lib/command.ts";
import { parseRef, refMoves, tarballUrl } from "../src/lib/srcref.ts";
import { defaults } from "../src/lib/params.ts";

const Q4 =
  "/m/unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
const LLAMA = { id: "source-master-cuda", ref: "master" };
const STRATA = { id: "source-fork-x-cuda", ref: STRATA_REF };

Deno.test("strata: the ref is an ordinary fork ref, fetched fresh each time", () => {
  const r = parseRef(STRATA_REF);
  assertEquals(r.kind, "fork");
  assert(refMoves(r));
  assert(tarballUrl(r).includes("Niko1221/Strata"));
  assert(isStrataRef(STRATA_REF));
  assert(isStrataRef("fork:niko1221/strata@main"));
  assert(!isStrataRef("master"));
  assert(!isStrataRef("fork:PrismML-Eng/llama.cpp"));
});

Deno.test("strata: only the files its setup knows are its models", () => {
  assertEquals(strataModel(Q4), {
    family: "unsloth",
    model: "UD-Q4_K_XL",
    tag: "unsloth-ud-q4_k_xl",
  });
  assertEquals(
    strataModel("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"),
    { family: "qwen", model: "Q2_0", tag: "q2_0" },
  );
  assertEquals(
    strataModel("Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")?.family,
    "coder",
  );
  assertEquals(
    strataModel("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf")
      ?.tag,
    "swift-iq2_xs",
  );
  // Not a first shard, another Unsloth size, a K-quant, another model.
  for (
    const no of [
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
      "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00003.gguf",
      "Qwen3.8-Flash-Next-Q4_K_M.gguf",
      "Qwen3.8-27B-Q8_0.gguf",
      "",
    ]
  ) assertEquals(strataModel(no), null, no);
});

Deno.test("strata: the context is rounded down to a size it was tuned at", () => {
  assertEquals(strataContext(119_040), 65_536);
  assertEquals(strataContext(262_144), 262_144);
  assertEquals(strataContext(1_048_576), 262_144);
  assertEquals(strataContext(4096), 8192);
  assertEquals(strataContext(0), 8192);
  assertEquals(strataContext(NaN), 8192);
  assertEquals(
    strataSettings({ ...defaults(), ctxSize: 119_040 }).ctxSize,
    65_536,
  );
});

Deno.test("strata: the reserve is the display card's, never under its own floor", () => {
  assertEquals(strataReserveMib(0, 4 * 2 ** 30), 4096);
  assertEquals(strataReserveMib(2 ** 30, 8 * 2 ** 30), 9216);
  assertEquals(strataReserveMib(0, 0), 700);
  assertEquals(strataReserveMib(NaN, 0), 700);
});

Deno.test("strata: the argv hands over the files on disk and downloads no model", () => {
  const a = strataArgs({
    modelPath: Q4,
    model: strataModel(Q4)!,
    ctx: 119_040,
    reserveMib: 4096,
    host: "0.0.0.0",
    port: 18080,
  });
  assertEquals(a[0], "unsloth-ud-q4_k_xl");
  const at = (flag: string) => a[a.indexOf(flag) + 1];
  assertEquals(at("--gguf-dir"), "/m/unsloth/Qwen3.8-Flash-Next-GGUF");
  assertEquals(at("--family"), "unsloth");
  assertEquals(at("--model"), "UD-Q4_K_XL");
  assertEquals(at("--context"), "65536");
  assertEquals(at("--vram-reserve-mib"), "4096");
  assertEquals(at("--host"), "0.0.0.0");
  assertEquals(at("--port"), "18080");
  // Drawn like any other command: the tag rides on its own line.
  assertEquals(argvBlock(["/b/llama-master-strata", ...a]).slice(0, 3), [
    "/b/llama-master-strata",
    "  unsloth-ud-q4_k_xl",
    "  --family unsloth",
  ]);
});

Deno.test("strata: a Strata build blocks a model it cannot run, and names the way out", () => {
  assertEquals(strataMismatch(Q4, STRATA_REF, [STRATA]), null);
  assertEquals(
    strataMismatch("Qwen3.8-27B-Q8_0.gguf", "master", [LLAMA]),
    null,
  );
  const d = strataMismatch("Qwen3.8-27B-Q8_0.gguf", STRATA_REF, [
    STRATA,
    LLAMA,
  ])!;
  assertEquals(d.steps[0]!.action, { kind: "use-build", id: LLAMA.id });
  const alone = strataMismatch("Qwen3.8-27B-Q8_0.gguf", STRATA_REF, [STRATA])!;
  assertEquals(alone.steps[0]!.action, { kind: "open-tab", tab: "build" });
});

Deno.test("strata: the offer is made once, to the right model, on the right machine", () => {
  const build = strataOffer(Q4, "master", [LLAMA], true)!;
  assertEquals(build.steps[0]!.action, { kind: "use-ref", ref: STRATA_REF });
  // The one download is named BEFORE the button is pressed.
  assert(build.steps[0]!.text.includes("6.5 GB"));
  const have = strataOffer(Q4, "master", [LLAMA, STRATA], true)!;
  assertEquals(have.steps[0]!.action, { kind: "use-build", id: STRATA.id });
  assertEquals(strataOffer(Q4, STRATA_REF, [STRATA], true), null);
  assertEquals(strataOffer(Q4, "master", [LLAMA], false), null);
  assertEquals(
    strataOffer("Qwen3.8-27B-Q8_0.gguf", "master", [LLAMA], true),
    null,
  );
});

Deno.test("strata: a run is described from its OWN argv, not from llama.cpp's settings", () => {
  const model = strataModel(Q4)!;
  const args = strataArgs({
    modelPath: Q4,
    model,
    ctx: 100_000,
    reserveMib: 4096,
    host: "127.0.0.1",
    port: 18080,
  });
  // What ran: the context rounded DOWN, the reserve as given.
  assertEquals(strataRunOf(["/b/llama-master-strata", ...args]), {
    ctx: 65536,
    reserveMib: 4096,
  });
  // A llama-server argv is not a Strata run.
  assertEquals(strataRunOf(["/b/llama-server", "-c", "4096"]), null);

  const rows = strataSetupRows({
    model,
    ctx: 65536,
    reserveMib: 4096,
    cards: 2,
    ramB: 83e9,
  });
  const text = rows.map((r) => `${r.label} ${r.value} ${r.short}`).join(" ");
  assert(text.includes("65,536"));
  assert(text.includes("2 GPUs"));
  // None of llama.cpp's vocabulary: those flags do not exist here.
  assert(!/MoE|q8_0|-ngl|\bnp\b/.test(text), text);
});

Deno.test("strata: its footprint is the free VRAM less the reserve, and a share of the file", () => {
  // The measured run: 111.3 GB file, 42.7 GB free on two cards, 4 GiB each
  // kept back. It took 35.6 GB of VRAM and 82.7 GB of RAM.
  const n = strataNeeds({
    fileB: 111_334_654_784,
    freeVramB: 42.7e9,
    cards: 2,
    reserveMib: 4096,
  });
  assert(Math.abs(n.vramB - 35.6e9) < 2e9, String(n.vramB));
  assert(Math.abs(n.ramB - 82.7e9) < 3e9, String(n.ramB));
  // A reserve larger than what is free is zero, never negative; junk is zero.
  assertEquals(
    strataNeeds({ fileB: NaN, freeVramB: 1e9, cards: 2, reserveMib: 4096 }),
    { vramB: 0, ramB: 0 },
  );
});

Deno.test("strata: too little RAM blocks Start, and an unknown reading does not", () => {
  const msg = strataRamShort(83e9, 59e9);
  // Both figures, in the app's own units (GiB, as every other panel).
  assert(msg.includes("77.3 GB") && msg.includes("54.9 GB"), msg);
  assert(msg.includes("llama.cpp"), "the way out is named");
  assertEquals(strataRamShort(83e9, 141e9), "");
  // No reading (0) is not a reason to refuse.
  assertEquals(strataRamShort(83e9, 0), "");
});

Deno.test("strata: the load names its phase from Strata's own log", () => {
  const log = [
    "=== Step 7: writing the start script ===",
    "[strata] starting the engine: reading the model's weights ...",
  ];
  assertEquals(loadPhase(log.slice(0, 1)), "checking the Strata setup");
  assertEquals(loadPhase(log), "reading the weights");
  log.push(
    "[strata] loading the experts into RAM (tens of GB) and locking part of them for the GPU.",
    "[strata] still starting (24 s) - please wait ...",
  );
  // "still starting" is not a phase: the one before it stands.
  assertEquals(loadPhase(log), "copying experts into RAM");
  log.push(
    "[strata] filling the GPU's expert cache (3569 experts, 10.47 GiB of VRAM) ...",
  );
  assertEquals(loadPhase(log), "filling the GPUs with experts");
  log.push("[strata] almost ready ...");
  assertEquals(loadPhase(log), "almost ready");
});
