// src/lib/shards.ts — a split GGUF is one model, and must be read as one.
//
// Anything past about 40 GB is shipped in parts (`-00001-of-00004.gguf`), and
// llama.cpp loads the set by being pointed at part 1. But the tensor table is
// NOT duplicated: each part carries only its own slice of it. Part 1 of
// DeepSeek-V4-Flash parses perfectly and describes 38 tensors and 37 GB — of the
// 1,328 tensors and 145 GB that are on disk.
//
// Reading part 1 alone therefore produces a model-shaped object that is a
// quarter of a model, and every number downstream inherits the error silently:
// the planner sizes 37 GB of weights, "VRAM only" looks possible on a 48 GB
// machine, the tuner proposes it, and llama-server dies loading the other 108
// GB. That is exactly what happened.
//
// So: read every part, sum the tensor accounting, keep the hyper-parameters
// (which every part repeats identically), and — because a partial answer here is
// worse than no answer — refuse to hand back a merge that did not see all of it.

import type { ModelMeta } from "./types.ts";

/** `…-00002-of-00004.gguf`. Fixed five digits, by the GGUF split convention. */
const SHARD = /-(\d{5})-of-(\d{5})\.gguf$/i;

export type ShardInfo = {
  /** This file is one part of a split set. */
  isShard: boolean;
  /** It is the part to point `-m` at (part 1, or a whole single file). */
  isFirst: boolean;
  /** 1-based index of this part; 1 for a single file. */
  index: number;
  /** Parts in the set; 1 for a single file. */
  count: number;
};

/** What a file name says about its place in a split set. */
export function shardInfo(file: string): ShardInfo {
  const m = SHARD.exec(file);
  if (!m) return { isShard: false, isFirst: true, index: 1, count: 1 };
  const index = Number(m[1]);
  const count = Number(m[2]);
  return { isShard: true, isFirst: index === 1, index, count };
}

/** The identity a set of parts shares, so sizes can be summed onto part 1. */
export function shardKey(path: string): string {
  return path.replace(SHARD, "");
}

/**
 * Every part of the set that `first` begins, in order — including `first`.
 *
 * Derived from the name rather than from a directory listing on purpose: the
 * name states the expected count, so a missing part is a gap we can NAME instead
 * of a part that quietly never appears.
 *
 * `count` overrides the name when the header disagrees (the header is the
 * authority; a renamed file is not). A file with no shard suffix and a header
 * claiming parts has no derivable siblings, and gets an empty list — the caller
 * turns that into an honest error rather than a guess.
 */
export function shardPaths(first: string, count?: number): string[] {
  const m = SHARD.exec(first);
  if (!m) return [];
  const total = count && count > 0 ? count : Number(m[2]);
  if (!Number.isFinite(total) || total < 1 || total > 999_99) return [];
  const pad = (n: number) => String(n).padStart(5, "0");
  return Array.from(
    { length: total },
    (_, i) => first.replace(SHARD, `-${pad(i + 1)}-of-${pad(total)}.gguf`),
  );
}

/** Does this header say it is only part of the story? */
export function isSplit(meta: ModelMeta): boolean {
  return (meta.splitCount ?? 0) > 1;
}

/**
 * One model out of its parts.
 *
 * Hyper-parameters come from part 1 — every part repeats them, and part 1 is the
 * one llama.cpp reads. Only the tensor accounting is summed, because only the
 * tensor table is divided: bytes per layer, routed experts, the embedding table
 * and the output head, and the tensor count itself.
 *
 * Layer arrays are summed index-wise rather than concatenated: `block_count` is
 * the same in every part, so each part contributes bytes to the layers it
 * happens to hold and zero to the rest.
 */
export function mergeShards(parts: ModelMeta[]): ModelMeta {
  const first = parts[0];
  if (!first) throw new Error("mergeShards: no parts");
  if (parts.length === 1) return first;

  const layers = first.layers.map((l) => ({ ...l }));
  const add = (m: ModelMeta) => {
    for (const l of m.layers) {
      const at = layers[l.i];
      if (at) {
        at.bytes += l.bytes;
        at.expert += l.expert;
      } else {
        layers[l.i] = { ...l };
      }
    }
  };
  for (const m of parts.slice(1)) add(m);

  const sum = (pick: (m: ModelMeta) => number) =>
    parts.reduce((a, m) => a + (pick(m) || 0), 0);

  return {
    ...first,
    // `block_count` is the model's own statement and only part 1 makes it, so
    // it wins: it drives the KV geometry, and a tensor table that happens to
    // run one index further must not silently re-shape the cache. The table
    // length is the fallback for the case where nobody said.
    nLayer: first.nLayer || layers.length,
    nTensors: sum((m) => m.nTensors),
    tensorBytes: sum((m) => m.tensorBytes),
    embdBytes: sum((m) => m.embdBytes),
    outputBytes: sum((m) => m.outputBytes),
    unknownTypes: sum((m) => m.unknownTypes),
    layers,
  };
}

/**
 * Why this merge is not the whole model — or `null` when it is.
 *
 * Two independent checks, because they catch different failures. The tensor
 * count is the model's own statement of how many tensors exist and is exact.
 * The file size catches the case the header cannot: parts that are present but
 * truncated, or a set whose `split.tensors.count` is simply absent.
 *
 * `sizeB` is the summed on-disk size of every part. Tensor bytes are always a
 * little under it (headers, alignment padding), so the test is generous — it is
 * looking for a QUARTER of a model, not for rounding.
 */
export function shardGap(
  meta: ModelMeta,
  sizeB: number,
  read: number,
  expected: number,
): string | null {
  if (read < expected) {
    return `only ${read} of ${expected} parts could be read`;
  }
  const want = meta.splitTensors ?? 0;
  if (want > 0 && meta.nTensors < want) {
    return `${meta.nTensors} of ${want} tensors were found across ${expected} parts`;
  }
  const accounted = meta.tensorBytes;
  if (sizeB > 0 && accounted > 0 && accounted < sizeB * 0.8) {
    return `the parts account for ${gb(accounted)} of ${
      gb(sizeB)
    } on disk — some are truncated`;
  }
  return null;
}

function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}
