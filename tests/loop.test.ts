// tests/loop.test.ts — a looped core (Inkling-2, `inkling2_loop`).
//
// llama.cpp expands the 24 layers to 48 cache slots at load
// (`src/models/inkling2-loop.cpp`): each pass of layers 6..17 keeps its own KV,
// `-ngl`/`-ts` count slots, and the shared weights stay with the PHYSICAL slots.

import { assert, assertEquals } from "@std/assert";
import {
  cacheSlots,
  kvByLayer,
  layerPasses,
  loopGraphB,
  physLayer,
  plan,
} from "../src/lib/plan.ts";
import { bytesPerToken } from "../src/lib/speed.ts";
import { defaults } from "../src/lib/params.ts";
import { gpu, hw, layers, meta } from "./fixtures.ts";

const MB = 1024 ** 2;

/** Inkling-2-Mini's header shape, as the reader returns it. */
const inkling = (loop = true) =>
  meta({
    arch: "inkling2_loop",
    nLayer: 24,
    nEmbd: 2304,
    nHeadKv: 8,
    nExpert: 128,
    nExpertUsed: 8,
    swaWindow: 1024,
    swaLayers: Array.from({ length: 24 }, (_, i) => (i % 6 === 5 ? 0 : 1)),
    layers: layers(24, 760 * MB, 743 * MB),
    ...(loop ? { loopStart: 6, loopEnd: 18, loopCount: 3 } : {}),
  });

Deno.test("loop: 24 layers become 48 cache slots, mapped like layer_phys", () => {
  const m = inkling();
  assertEquals(cacheSlots(m), 48);
  assertEquals(cacheSlots(inkling(false)), 24);
  assertEquals([23, 24, 35, 36, 47].map((i) => physLayer(m, i)), [
    23,
    6,
    17,
    6,
    17,
  ]);
  assertEquals([5, 6, 17, 18].map((i) => layerPasses(m, i)), [1, 3, 3, 1]);
});

Deno.test("loop: every pass carries its own KV cache", () => {
  const s = { ...defaults(), ctxSize: 32768 };
  const flat = kvByLayer(inkling(false), s, 32768);
  const looped = kvByLayer(inkling(), s, 32768);
  assertEquals(looped.length, 48);
  const core = flat.slice(6, 18).reduce((a, b) => a + b, 0);
  const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
  assertEquals(sum(looped), sum(flat) + 2 * core);
});

Deno.test("loop: weights stay on the physical slots, whatever card holds the copies", () => {
  const m = inkling();
  const two = hw({ gpus: [gpu(24), gpu(24)] });
  // 24 of 49 offloaded slots on card 0: every physical layer, nothing else.
  const p = plan(m, two, { ...defaults(), ngl: 999, tensorSplit: "24,25" });
  assertEquals(p.devices.cards[1]?.weightsB, m.outputBytes);
  assert((p.devices.cards[1]?.kvB ?? 0) > 0, "card 1 holds the loop caches");
  // `-ngl 25` offloads the head and the 24 loop slots — none of the weights.
  const q = plan(m, two, { ...defaults(), ngl: 25 });
  assertEquals(q.layersOnGpu, 0);
});

Deno.test("loop: the core's weights are read once per pass", () => {
  const s = { ...defaults(), ngl: 999 };
  const one = hw({ gpus: [gpu(48)] });
  const read = (m: ReturnType<typeof inkling>) =>
    bytesPerToken(m, plan(m, one, s), s, 0).gpuB;
  const perLayer = 17 * MB + 743 * MB * (8 / 128);
  const extra = read(inkling()) - read(inkling(false));
  assert(Math.abs(extra - 2 * 12 * perLayer) < 1, `extra ${extra}`);
});

Deno.test("loop: the GPU graph is the measured one, not the fitted mask", () => {
  const MiB = 1024 ** 2;
  const m = inkling();
  // Inkling-2-Mini Q4_K_M, llama.cpp 1.0-dev, one card: CUDA0 compute buffer.
  const at = (ctx: number, ub: number, moe: number) =>
    (loopGraphB(m, ctx, ub, moe) - 768 * MiB) / MiB;
  const close = (got: number, want: number) =>
    assert(got >= want && got <= want * 1.02, `${got} vs ${want}`);
  close(at(32768, 512, 6), 3545);
  close(at(32768, 512, 18), 10903);
  close(at(32768, 512, 24), 10903); // the tail adds nothing
  close(at(8192, 512, 24), 8527);
  assertEquals(loopGraphB(inkling(false), 32768, 512, 24), 0);
});

Deno.test("loop: the hybrid search counts slots, not layers", async () => {
  const { tune } = await import("../src/lib/tune.ts");
  const { offloadRange } = await import("../src/lib/devsplit.ts");
  const m = inkling();
  // 11 GB: the full -ngl does not fit, so the -ub 256 retry must run. Judged
  // against the 24 LAYERS it was skipped from -ngl 25 up, and the card was
  // handed one layer where -ub 256 buys most of them.
  const t = tune(m, hw({ gpus: [gpu(11)] }), defaults(), "hybrid");
  assert(t.possible);
  const onGpu = m.nLayer - offloadRange(cacheSlots(m), t.settings).start;
  assert(onGpu > 1, `${onGpu} layer(s) on the card at -ngl ${t.settings.ngl}`);
  assertEquals(t.settings.ubatchSize, 256);
});
