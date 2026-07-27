// test/gguf-fixture.ts — build a real GGUF file, byte by byte.
//
// Shared by the WASM tests and the UI tests. Writing the format by hand (rather
// than checking in a binary) means the fixture IS the spec: if the parser and
// this builder ever disagree about the layout, one of them is wrong and a test
// says so.

export class GgufBuilder {
  private parts: Uint8Array[] = [];
  private enc = new TextEncoder();

  constructor() {
    this.raw(this.enc.encode("GGUF"));
    this.u32(3);
  }
  private raw(b: Uint8Array): this {
    this.parts.push(b);
    return this;
  }
  u32(v: number): this {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v, true);
    return this.raw(b);
  }
  u64(v: number): this {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, BigInt(v), true);
    return this.raw(b);
  }
  s(v: string): this {
    const bytes = this.enc.encode(v);
    this.u64(bytes.length);
    return this.raw(bytes);
  }
  kvStr(k: string, v: string): this {
    return this.s(k).u32(8).s(v);
  }
  kvU32(k: string, v: number): this {
    return this.s(k).u32(4).u32(v);
  }
  /** A string array — the shape that makes real headers megabytes long. */
  kvStrArray(k: string, items: string[]): this {
    this.s(k).u32(9).u32(8).u64(items.length);
    for (const i of items) this.s(i);
    return this;
  }
  tensor(name: string, dims: number[], type: number): this {
    this.s(name).u32(dims.length);
    for (const d of dims) this.u64(d);
    return this.u32(type).u64(0);
  }
  bytes(): Uint8Array {
    const total = this.parts.reduce((a, p) => a + p.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of this.parts) {
      out.set(p, off);
      off += p.length;
    }
    return out;
  }
}

export const Q4_K = 12;
export const F32 = 0;

/** Bytes a Q4_K tensor of `elems` elements occupies (144 bytes per 256). */
export const q4kBytes = (elems: number) => (elems / 256) * 144;

/**
 * A structurally complete mixture-of-experts model: four layers, each with a
 * small attention tensor and a large routed-expert tensor, plus an embedding
 * table and an output norm. Small enough to write in a test, shaped exactly
 * like the models the planner has to be right about.
 *
 * `vocab` inflates the tokenizer array, which is how a real header grows past
 * the scanner's first read.
 */
export function moeGguf(vocab = 8): Uint8Array {
  const g = new GgufBuilder();
  const nLayer = 4;
  g.u64(nLayer * 2 + 2);
  g.u64(9);
  g.kvStr("general.architecture", "qwen3moe");
  g.kvStr("general.name", "Fixture MoE");
  g.kvU32("general.file_type", 15);
  g.kvU32("qwen3moe.block_count", nLayer);
  g.kvU32("qwen3moe.context_length", 32768);
  g.kvU32("qwen3moe.embedding_length", 2048);
  g.kvU32("qwen3moe.attention.head_count", 16);
  g.kvU32("qwen3moe.attention.head_count_kv", 4);
  g.kvStrArray(
    "tokenizer.ggml.tokens",
    Array.from({ length: vocab }, (_, i) => `tok${i}`.padEnd(24, "x")),
  );
  for (let i = 0; i < nLayer; i++) {
    g.tensor(`blk.${i}.attn_q.weight`, [2048, 2048], Q4_K);
    g.tensor(`blk.${i}.ffn_gate_exps.weight`, [2048, 2048, 8], Q4_K);
  }
  g.tensor("token_embd.weight", [2048, 4096], Q4_K);
  g.tensor("output_norm.weight", [2048], F32);
  return g.bytes();
}
