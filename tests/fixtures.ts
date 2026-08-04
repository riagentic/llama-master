// test/fixtures.ts — the machines and models the planner tests reason about.
//
// Sizes are realistic on purpose: a planner that is right on a toy 2-layer
// model and wrong on a 70B is not a planner. These numbers come from real GGUF
// headers (llama-3 8B Q4_K_M, a 32B dense, and a Mixtral-shaped MoE).

import type { Gpu, Hw, LayerBytes, ModelMeta } from "../src/lib/types.ts";

const GB = 1024 ** 3;
const MB = 1024 ** 2;

export function layers(
  n: number,
  bytesEach: number,
  expertEach = 0,
): LayerBytes[] {
  return Array.from({ length: n }, (_, i) => ({
    i,
    bytes: bytesEach,
    expert: expertEach,
  }));
}

export function meta(over: Partial<ModelMeta> = {}): ModelMeta {
  const base: ModelMeta = {
    version: 3,
    arch: "llama",
    name: "Test 8B",
    quant: "Q4_K_M",
    nLayer: 32,
    nCtxTrain: 8192,
    nEmbd: 4096,
    nHead: 32,
    nHeadKv: 8,
    keyLength: 128,
    swaWindow: 0,
    swaPattern: 1,
    kvLoraRank: 0,
    nextnLayers: 0,
    valueLength: 128,
    nExpert: 0,
    nExpertUsed: 0,
    ropeFreqBase: 500000,
    nTensors: 291,
    tensorBytes: 0,
    embdBytes: 300 * MB,
    outputBytes: 300 * MB,
    unknownTypes: 0,
    nCtxOrig: 0,
    indexerTopK: 0,
    splitNo: 0,
    splitCount: 0,
    splitTensors: 0,
    layers: layers(32, 128 * MB),
  };
  const m = { ...base, ...over };
  m.tensorBytes = m.layers.reduce((a, l) => a + l.bytes, 0) + m.embdBytes +
    m.outputBytes;
  return m;
}

/** Mixtral-shaped: small attention, enormous routed experts. */
export function moeMeta(): ModelMeta {
  return meta({
    name: "Test MoE",
    nLayer: 32,
    nExpert: 8,
    nExpertUsed: 2,
    // 40 MB of attention + 720 MB of experts per layer.
    layers: layers(32, 760 * MB, 720 * MB),
  });
}

/** A card. `display` is left undefined on purpose — that is what a machine
 *  which does not report display attachment looks like, and it is the reading
 *  the connected-GPU reserve has to cope with (`src/lib/reserve.ts`). Pass it
 *  explicitly to test a machine that does answer. */
export function gpu(vramGb: number, usedGb = 0.5, display?: boolean): Gpu {
  return {
    display,
    vendor: "nvidia",
    name: `Test GPU ${vramGb}G`,
    tempC: 42,
    utilPct: 3,
    vramTotalB: vramGb * GB,
    vramUsedB: usedGb * GB,
    powerW: 30,
    powerLimitW: 450,
    // Ada by default: old enough that any modern CUDA can target it natively.
    computeCap: 8.9,
  };
}

export function hw(over: Partial<Hw> = {}): Hw {
  return {
    cpu: {
      model: "Test CPU",
      cores: 16,
      threads: 32,
      mhz: 3800,
      tempC: 48,
      utilPct: 5,
      stat: "cpu  1 2 3 4",
      coreStats: [],
      coresUtil: [],
    },
    mem: {
      totalB: 64 * GB,
      availableB: 56 * GB,
      usedB: 8 * GB,
      swapTotalB: 8 * GB,
      swapUsedB: 0,
    },
    gpus: [gpu(24)],
    os: "linux",
    arch: "x86_64",
    // The fixture's GPUs are NVIDIA, so CUDA is the coherent build for them —
    // and the tuner needs to know, because which flags are even loadable
    // (quantised KV, forced flash attention) depends on the backend.
    backend: "cuda",
    ...over,
  };
}

export const NO_GPU: Hw = hw({ gpus: [] });
