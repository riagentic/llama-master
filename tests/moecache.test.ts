// tests/moecache.test.ts — llama.cpp 1.0's expert cache (`--moe-cache-mib`).
//
// Measured on Qwen3.8-Flash-Next with every expert in RAM on one 24 GB card:
// 5 GB of cache with `--moe-cache-cpu-share 1` took generation 15.6 → 17.6
// tok/s, and the default share of 0 made it a LOSS (13.6). So the tuner sets
// the two together, only for a build that declares both, and the plan bills it.

import { assert, assertEquals } from "@std/assert";
import { tune } from "../src/lib/tune.ts";
import { plan } from "../src/lib/plan.ts";
import { defaults } from "../src/lib/params.ts";
import type { Settings } from "../src/lib/types.ts";
import { gpu, hw, layers, meta, moeMeta } from "./fixtures.ts";

const CAPS = ["--moe-cache-mib", "--moe-cache-cpu-share", "-ngl", "-c"];
const MB = 1024 ** 2;
// Flash-Next-shaped: ~2.9 GB of experts per layer, so whole layers on the card
// leave a remainder too small for another layer and big enough for a cache.
const model = () =>
  meta({
    name: "Big-expert MoE",
    nLayer: 16,
    nExpert: 64,
    nExpertUsed: 4,
    layers: layers(16, 3000 * MB, 2900 * MB),
  });
const machine = hw({ gpus: [gpu(24)] });
const T = (caps?: string[], base: Settings = defaults()) =>
  tune(model(), machine, base, "hybrid", 4096, undefined, false, { caps });

Deno.test("moe cache: leftover VRAM caches host experts, misses on the CPU", () => {
  const r = T(CAPS);
  assert(Number(r.settings.nCpuMoe) > 0, "the fixture holds experts in RAM");
  assert(Number(r.settings.moeCacheMib) >= 1024, `${r.settings.moeCacheMib}`);
  assertEquals(Number(r.settings.moeCacheMib) % 1024, 0);
  assertEquals(r.settings.moeCacheCpuShare, 1);
  // It still fits: the plan bills what the tuner spent.
  const p = plan(model(), machine, r.settings);
  assertEquals(p.vram.overB, 0);
  assert(p.devices.fits);
  const without = plan(model(), machine, { ...r.settings, moeCacheMib: 0 });
  assertEquals(
    p.vram.usedB - without.vram.usedB,
    Number(r.settings.moeCacheMib) * 1024 ** 2,
  );
});

Deno.test("moe cache: never for a build that has not declared both flags", () => {
  for (const caps of [undefined, ["--moe-cache-mib"]]) {
    const r = T(caps);
    assertEquals([r.settings.moeCacheMib, r.settings.moeCacheCpuShare], [0, 0]);
  }
});

Deno.test("moe cache: one model's cache does not ride into the next", () => {
  const big = hw({ gpus: [gpu(96)] });
  // Everything fits in VRAM: no expert in RAM, so no cache — even when the
  // settings came from a run that had one.
  const r = tune(
    moeMeta(),
    big,
    { ...defaults(), moeCacheMib: 8192, moeCacheCpuShare: 1 },
    "vram",
    4096,
    undefined,
    false,
    { caps: CAPS },
  );
  assertEquals(r.settings.nCpuMoe, 0);
  assertEquals([r.settings.moeCacheMib, r.settings.moeCacheCpuShare], [0, 0]);
});
