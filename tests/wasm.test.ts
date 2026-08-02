// test/wasm.test.ts — the Rust core through the real WASM boundary, and the
// host layer that feeds it.
//
// `cargo test` already covers the parsers as Rust functions. What this file
// proves is the part cargo cannot: that the memory protocol between Deno and
// WASM is correct, that a real GGUF file on disk comes back as the right
// numbers, and that the progressive header read finds its own way to a header
// bigger than the first chunk.

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { cpuJson, gguf, gpuJson, memJson } from "../src/cell/wasm.server.ts";
import {
  defaultDirs,
  readMeta,
  readModel,
  scan,
} from "../src/cell/models.server.ts";
import { snapshot } from "../src/cell/hw.server.ts";
import { join } from "@std/path";
import { moeGguf, q4kBytes, shardName, splitMoeGguf } from "./gguf-fixture.ts";

Deno.test("wasm: a GGUF header parses into exact per-layer byte accounting", async () => {
  const r = await gguf(moeGguf());
  assert(r.ok, `expected a parse, got ${JSON.stringify(r)}`);
  const m = r.json as Record<string, number | string | unknown[]>;
  assertEquals(m.arch, "qwen3moe");
  assertEquals(m.name, "Fixture MoE");
  assertEquals(m.quant, "Q4_K_M");
  assertEquals(m.nLayer, 4);
  assertEquals(m.nCtxTrain, 32768);
  assertEquals(m.nHeadKv, 4);
  assertEquals(m.keyLength, 128, "derived from n_embd / n_head");

  const layers = m.layers as { i: number; bytes: number; expert: number }[];
  assertEquals(layers.length, 4);
  const attn = q4kBytes(2048 * 2048);
  const experts = q4kBytes(2048 * 2048 * 8);
  assertEquals(layers[0]?.bytes, attn + experts);
  assertEquals(layers[0]?.expert, experts, "routed experts are separated out");
  assertEquals(m.embdBytes, q4kBytes(2048 * 4096));
  assertEquals(m.outputBytes, 2048 * 4, "F32 norm");
  assertEquals(m.unknownTypes, 0);
});

Deno.test("wasm: a truncated header asks for more bytes instead of failing", async () => {
  const full = moeGguf();
  const r = await gguf(full.subarray(0, 64));
  assert(!r.ok);
  assert(r.truncated !== null && r.truncated > 64, `need = ${r.truncated}`);
});

Deno.test("wasm: a file that is not GGUF is rejected, not guessed at", async () => {
  const r = await gguf(
    new TextEncoder().encode("this is a text file, honestly"),
  );
  assert(!r.ok);
  assertEquals(r.truncated, null);
  assertStringIncludes(r.error, "GGUF");
});

Deno.test("wasm: repeated calls do not leak or corrupt WASM memory", async () => {
  // 200 round trips through alloc/free — a leak or a double free shows up as a
  // wrong answer or a trap, not as a slow test.
  for (let i = 0; i < 200; i++) {
    const r = await gguf(moeGguf(4));
    assert(r.ok, `iteration ${i} failed`);
  }
  const last = await gguf(moeGguf(4));
  assert(last.ok && (last.json.nLayer as number) === 4);
});

Deno.test("wasm: telemetry parsers round-trip through the same boundary", async () => {
  const cpu = await cpuJson(
    "processor\t: 0\nmodel name\t: Fixture CPU\ncpu MHz\t: 3000.0\nphysical id\t: 0\ncore id\t: 0\n",
    "cpu  1 2 3 4\ncpu0 1 2 3 4\n",
    "Tctl\t55000\n",
  );
  assertEquals(cpu.model, "Fixture CPU");
  assertEquals(cpu.cores, 1);
  assertEquals(cpu.tempC, 55);

  const mem = await memJson("MemTotal: 1024 kB\nMemAvailable: 512 kB\n");
  assertEquals(mem.totalB, 1024 * 1024);
  assertEquals(mem.usedB, 512 * 1024);

  const gpus = await gpuJson("Fixture GPU, 50, 10, 1024, 256, 30, 300\n", "");
  assertEquals(gpus.length, 1);
  assertEquals(gpus[0]?.name, "Fixture GPU");
  assertEquals(gpus[0]?.vramTotalB, 1024 * 1024 * 1024);
});

// ── the host layer, against real files ─────────────────────────────────────

Deno.test("models: a real file on disk is scanned, shards collapse to part 1", async () => {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-test-" });
  try {
    const body = moeGguf();
    await Deno.writeFile(join(dir, "fixture-Q4_K_M.gguf"), body);
    // A split model: only part 1 should appear, with the combined size.
    await Deno.writeFile(join(dir, "big-00001-of-00002.gguf"), body);
    await Deno.writeFile(join(dir, "big-00002-of-00002.gguf"), body);
    // A file that is not a GGUF at all must be reported, not hidden.
    await Deno.writeFile(
      join(dir, "broken.gguf"),
      new TextEncoder().encode("nope"),
    );
    // Nested directories are searched too.
    await Deno.mkdir(join(dir, "sub"));
    await Deno.writeFile(join(dir, "sub", "nested.gguf"), body);

    const seen: number[] = [];
    const found = await scan([dir], (done) => seen.push(done));

    assertEquals(found.map((m) => m.file).sort(), [
      "big-00001-of-00002.gguf",
      "broken.gguf",
      "fixture-Q4_K_M.gguf",
      "nested.gguf",
    ]);
    assert(seen.length > 0, "scan reports progress for the bar");

    const shard = found.find((m) => m.file.startsWith("big"))!;
    assertEquals(shard.sizeB, body.length * 2, "shard sizes are summed");

    const ok = found.find((m) => m.file === "fixture-Q4_K_M.gguf")!;
    assertEquals(ok.meta?.arch, "qwen3moe");
    assertEquals(ok.metaError, null);

    const bad = found.find((m) => m.file === "broken.gguf")!;
    assertEquals(bad.meta, null);
    assert(bad.metaError, "an unreadable header is surfaced, not swallowed");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("models: a header larger than the first read is retried, not truncated", async () => {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-test-" });
  try {
    // ~40k vocab entries of 24 bytes ≈ 1.3 MB of header — past the 1 MB first
    // read, so this only passes if the retry loop works.
    const body = moeGguf(40_000);
    assert(body.length > 1024 * 1024, `fixture is only ${body.length} bytes`);
    const path = join(dir, "big-header.gguf");
    await Deno.writeFile(path, body);
    const { meta, error } = await readMeta(path);
    assertEquals(error, null);
    assertEquals(meta?.nLayer, 4);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

/**
 * The bug this exists to stop: a split model read as its first part.
 *
 * Part 1 of DeepSeek-V4-Flash parses perfectly and describes 38 tensors and 37
 * GB — of the 1,328 tensors and 145 GB on disk. Nothing downstream can tell:
 * the planner sized it at a quarter, "VRAM only" looked possible on 48 GB of
 * cards, the tuner proposed it, and llama-server was killed loading the rest.
 */
Deno.test("models: a split model is read as one model, not as its first part", async () => {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-test-" });
  try {
    const parts = splitMoeGguf(3);
    const stem = join(dir, "split-moe");
    let sizeB = 0;
    for (let i = 0; i < parts.length; i++) {
      await Deno.writeFile(shardName(stem, i + 1, parts.length), parts[i]!);
      sizeB += parts[i]!.length;
    }
    const first = shardName(stem, 1, parts.length);

    // What the old code saw, and why it looked like a whole model.
    const one = await readMeta(first);
    assertEquals(one.meta?.splitCount, 3, "the part knows it is a part");
    assertEquals(one.meta?.nTensors, 6, "…and holds 6 of the 14 tensors");

    const { meta, error } = await readModel(first, sizeB);
    assertEquals(error, null);
    assertEquals(meta?.nTensors, 14, "every tensor across every part");
    assertEquals(meta?.splitTensors, 14);
    assertEquals(meta?.nLayer, 6);
    assertEquals(
      meta?.layers.filter((l) => l.bytes > 0).length,
      6,
      "parts 2 and 3 declare no block_count, and their layers are still layers",
    );
    const attn = q4kBytes(2048 * 2048);
    const experts = q4kBytes(2048 * 2048 * 8);
    for (const l of meta!.layers) {
      assertEquals(l.bytes, attn + experts, `layer ${l.i}`);
      assertEquals(l.expert, experts, `layer ${l.i} experts`);
    }
    assertEquals(
      meta?.outputBytes,
      2048 * 4,
      "output_norm only — not 8/14ths of the model filed as the output head",
    );
    assert(
      meta!.tensorBytes > one.meta!.tensorBytes * 2.5,
      "the merge is most of the model, not a third of it",
    );
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

/** A partial set is not a smaller model. llama.cpp cannot load it either, and a
 *  plan drawn over the readable parts is a plan for something that does not
 *  exist — so it has to fail loud rather than quietly shrink. */
Deno.test("models: a split model missing a part is an error, not a smaller model", async () => {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-test-" });
  try {
    const parts = splitMoeGguf(3);
    const stem = join(dir, "split-moe");
    let sizeB = 0;
    for (let i = 0; i < parts.length; i++) {
      sizeB += parts[i]!.length;
      if (i === 1) continue; // part 2 never finished downloading
      await Deno.writeFile(shardName(stem, i + 1, parts.length), parts[i]!);
    }
    const { meta, error } = await readModel(
      shardName(stem, 1, parts.length),
      sizeB,
    );
    assertEquals(meta, null, "no meta means no plan, which is the point");
    assertStringIncludes(error ?? "", "incomplete");
    assertStringIncludes(error ?? "", "2 of 3 parts");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("hw: this machine reports a coherent snapshot", async () => {
  const s = await snapshot();
  // Every field is best-effort, but the shape and the invariants are not.
  if (s.cpu) {
    assert(s.cpu.threads > 0, "a running process implies at least one CPU");
    assert(s.cpu.cores > 0 && s.cpu.cores <= s.cpu.threads);
    assert(s.cpu.tempC >= 0 && s.cpu.tempC < 150);
  }
  if (s.mem) {
    assert(s.mem.totalB > 0);
    assert(s.mem.availableB <= s.mem.totalB);
    assertEquals(s.mem.usedB, s.mem.totalB - s.mem.availableB);
  }
  for (const g of s.gpus) {
    assert(g.name.length > 0);
    assert(g.vramUsedB <= g.vramTotalB, `${g.name}: used > total`);
    assert(g.utilPct >= 0 && g.utilPct <= 100);
  }
});

Deno.test("models: an ollama store is scanned through its manifests", async () => {
  // ollama keeps no .gguf files — weights are `blobs/sha256-<hex>`, reachable
  // only via the manifests. A plain extension walk finds nothing here, which
  // is the whole reason the resolver exists.
  const root = await Deno.makeTempDir({ prefix: "llama-master-ollama-" });
  try {
    const body = moeGguf();
    const digest = "sha256:" + "ab".repeat(32);
    const blobs = join(root, "blobs");
    const manifests = join(
      root,
      "manifests",
      "registry.ollama.ai",
      "library",
      "qwen3",
    );
    await Deno.mkdir(blobs, { recursive: true });
    await Deno.mkdir(manifests, { recursive: true });
    await Deno.writeFile(join(blobs, digest.replace(":", "-")), body);
    await Deno.writeTextFile(
      join(manifests, "4b"),
      JSON.stringify({
        schemaVersion: 2,
        layers: [
          {
            mediaType: "application/vnd.ollama.image.model",
            digest,
            size: body.length,
          },
          {
            mediaType: "application/vnd.ollama.image.template",
            digest: "sha256:cc",
          },
        ],
      }),
    );
    // A cloud entry alongside it: registered locally, no weights on disk.
    const cloudDir = join(
      root,
      "manifests",
      "registry.ollama.ai",
      "library",
      "glm",
    );
    await Deno.mkdir(cloudDir, { recursive: true });
    await Deno.writeTextFile(
      join(cloudDir, "cloud"),
      JSON.stringify({ schemaVersion: 2, layers: null }),
    );

    const found = await scan([root], () => {});
    assertEquals(found.length, 1, "the cloud model must not be listed");
    const m = found[0]!;
    assertEquals(m.file, "qwen3:4b", "named as the user would type it");
    assertEquals(m.source, "ollama");
    assertEquals(m.sizeB, body.length);
    assertEquals(m.meta?.arch, "qwen3moe", "the blob is parsed as a real GGUF");
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("models: LM Studio files are found and projectors are not listed", async () => {
  const root = await Deno.makeTempDir({ prefix: "llama-master-lms-" });
  try {
    const dir = join(root, ".lmstudio", "models", "unsloth", "Qwen3-GGUF");
    await Deno.mkdir(dir, { recursive: true });
    await Deno.writeFile(join(dir, "Qwen3-Q8_0.gguf"), moeGguf());
    // A multimodal projector ships beside the model and is not loadable alone.
    await Deno.writeFile(join(dir, "mmproj-F32.gguf"), moeGguf());

    const found = await scan([join(root, ".lmstudio", "models")], () => {});
    assertEquals(found.map((m) => m.file), ["Qwen3-Q8_0.gguf"]);
    assertEquals(found[0]?.source, "lmstudio");
  } finally {
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("models: the default search paths cover the other tools' stores", () => {
  const dirs = defaultDirs();
  const has = (frag: string) => dirs.some((d) => d.includes(frag));
  assert(has(".lmstudio/models"), `LM Studio missing from ${dirs.join(", ")}`);
  assert(has(".ollama/models"), "ollama (user install) missing");
  assert(
    dirs.includes("/usr/share/ollama/.ollama/models"),
    "ollama (system service) missing",
  );
  assert(has(".cache/huggingface"), "the huggingface cache is still covered");
});

Deno.test("wasm: the sliding-window and MLA keys survive the round trip", async () => {
  // The planner's KV arithmetic now depends on these three, so a reader that
  // dropped them would silently restore the old several-fold overestimate.
  const r = await gguf(moeGguf());
  assert(r.ok);
  const m = r.json as Record<string, number>;
  // The fixture declares none of them, and "absent" has to mean full attention
  // on every layer — not a zero that means something else downstream.
  assertEquals(m.swaWindow, 0, "absent means full attention");
  assertEquals(m.kvLoraRank, 0, "absent means not MLA");
  assertEquals(m.swaPattern, 1, "and the pattern defaults to 1, never 0");
});
