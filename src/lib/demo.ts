// src/lib/demo.ts — a machine and a model library that do not exist.
//
// Set `LLAMA_MASTER_DEMO=1` and the app reports this hardware and these models
// instead of reading the real ones. It exists for three honest reasons:
//
//   * screenshots for the README that show nobody's actual machine, paths or
//     model collection,
//   * letting someone see what the app does before they own a GPU or have
//     downloaded a single GGUF,
//   * reproducing a layout bug without attaching your own hardware to the issue.
//
// The numbers are plausible but invented, and the model names are fictional on
// purpose — no real repository is implied. Everything downstream (the planner,
// the tuner, the placement search) runs for real against these figures, so a
// demo screenshot shows the app genuinely working, not a mock-up.
//
// Pure data: no I/O, so it is safe to import anywhere.

import type { Build, Cpu, Gpu, Mem, ModelMeta } from "./types.ts";

const GB = 1024 ** 3;

export const DEMO_ENV = "LLAMA_MASTER_DEMO";

export function demoCpu(): Cpu {
  return {
    model: "Example CPU 12-Core Processor",
    cores: 12,
    threads: 24,
    mhz: 4200,
    tempC: 52,
    utilPct: 17,
    stat: "",
    coreStats: [],
    coresUtil: [31, 12, 8, 22, 41, 6, 14, 9, 27, 11, 5, 19],
  };
}

export function demoMem(): Mem {
  return {
    totalB: 64 * GB,
    availableB: 48 * GB,
    usedB: 16 * GB,
    swapTotalB: 8 * GB,
    swapUsedB: 0,
  };
}

export function demoGpus(): Gpu[] {
  return [
    {
      vendor: "nvidia",
      name: "Example GPU 24GB",
      computeCap: 8.9,
      tempC: 46,
      utilPct: 8,
      vramTotalB: 24 * GB,
      vramUsedB: 1.4 * GB,
      powerW: 62,
      powerLimitW: 320,
    },
  ];
}

/** A GGUF header for a model that does not exist, sized like one that does. */
function meta(over: Partial<ModelMeta>): ModelMeta {
  return {
    version: 3,
    arch: "llama",
    name: "Demo",
    quant: "Q4_K_M",
    nLayer: 32,
    nCtxTrain: 32768,
    nEmbd: 4096,
    nHead: 32,
    nHeadKv: 8,
    keyLength: 128,
    valueLength: 128,
    swaWindow: 0,
    swaPattern: 1,
    kvLoraRank: 0,
    nExpert: 0,
    nExpertUsed: 0,
    ropeFreqBase: 500000,
    nTensors: 291,
    tensorBytes: 0,
    embdBytes: 0,
    outputBytes: 0,
    unknownTypes: 0,
    layers: [],
    ...over,
  };
}

function layers(n: number, each: number, expert = 0) {
  return Array.from({ length: n }, (_, i) => ({ i, bytes: each, expert }));
}

/** Fictional models, in the shape the scanner would return them. */
export function demoModels(): {
  path: string;
  file: string;
  sizeB: number;
  source: "file";
  meta: ModelMeta;
}[] {
  const mk = (
    file: string,
    sizeB: number,
    m: Partial<ModelMeta>,
  ) => ({
    path: `/models/${file}`,
    file,
    sizeB,
    source: "file" as const,
    meta: meta({ ...m, name: file.replace(/-[^-]*\.gguf$/, "") }),
  });

  return [
    mk("example-8b-instruct-Q4_K_M.gguf", 4.9 * GB, {
      nLayer: 32,
      nCtxTrain: 131072,
      layers: layers(32, 145 * 1024 * 1024),
    }),
    mk("example-moe-8x3b-Q4_K_M.gguf", 26 * GB, {
      arch: "llama-moe",
      nLayer: 32,
      nCtxTrain: 32768,
      nExpert: 8,
      nExpertUsed: 2,
      layers: layers(32, 800 * 1024 * 1024, 760 * 1024 * 1024),
    }),
    mk("example-32b-chat-Q5_K_M.gguf", 23 * GB, {
      nLayer: 64,
      nCtxTrain: 32768,
      nEmbd: 5120,
      quant: "Q5_K_M",
      layers: layers(64, 360 * 1024 * 1024),
    }),
  ];
}

/** A llama.cpp build that was never installed, so the demo has something to
 *  run with — the backend matters, because it decides which flags are even
 *  loadable and therefore what the tuner is allowed to propose. */
export function demoBuilds(): Build[] {
  return [{
    id: "release-b0000-cuda",
    ref: "b0000",
    origin: "release",
    backend: "cuda",
    dir: "/builds/release-b0000-cuda",
    serverBin: "/builds/release-b0000-cuda/llama-server",
    cliBin: "/builds/release-b0000-cuda/llama-cli",
    createdAt: 0,
    sizeB: 512 * 1024 * 1024,
  }];
}
