// src/cell/wasm.server.ts — loader for the Rust core (rust/src/*.rs → WASM).
//
// SERVER ONLY, loaded lazily and cached. The module is instantiated on first
// call; if `src/llama-sys.wasm` is missing (someone edited the Rust and forgot
// `deno task wasm`) every call throws with that exact instruction rather than
// returning plausible-looking zeroes.

type Exports = {
  memory: WebAssembly.Memory;
  alloc: (len: number) => number;
  dealloc: (ptr: number, len: number) => void;
  str_len: (ptr: number) => number;
  free_str: (ptr: number) => void;
  gguf_parse: (ptr: number, len: number) => number;
  sys_cpu: (
    ci: number,
    cil: number,
    st: number,
    stl: number,
    hw: number,
    hwl: number,
  ) => number;
  sys_mem: (ptr: number, len: number) => number;
  sys_gpu: (nv: number, nvl: number, sf: number, sfl: number) => number;
};

let cached: Exports | null = null;
let loading: Promise<Exports> | null = null;

async function load(): Promise<Exports> {
  if (cached) return cached;
  loading ??= (async () => {
    const url = new URL("../llama-sys.wasm", import.meta.url);
    let bytes: Uint8Array<ArrayBuffer>;
    try {
      bytes = await Deno.readFile(url);
    } catch (e) {
      throw new Error(
        `llama-sys.wasm is missing (${url.pathname}) — run \`deno task wasm\` to build it. Cause: ${e}`,
      );
    }
    // compile+instantiate rather than the one-shot overload: the two-step
    // form is unambiguous about returning an Instance.
    const module = await WebAssembly.compile(bytes);
    const instance = await WebAssembly.instantiate(module, {});
    cached = instance.exports as unknown as Exports;
    return cached;
  })();
  return await loading;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

/** Copy a JS value into WASM memory; returns the pointer and the length the
 *  caller must hand back to `dealloc`. */
function put(w: Exports, data: Uint8Array): [number, number] {
  if (data.length === 0) return [0, 0];
  const ptr = w.alloc(data.length);
  new Uint8Array(w.memory.buffer, ptr, data.length).set(data);
  return [ptr, data.length];
}

/** Read a length-prefixed string back out and release it. */
function take(w: Exports, ptr: number): string {
  const len = w.str_len(ptr);
  const view = new Uint8Array(w.memory.buffer, ptr + 4, len);
  const s = dec.decode(view.slice());
  w.free_str(ptr);
  return s;
}

/** Call a WASM export with N string inputs, always freeing what we allocated. */
async function callWasm(
  fn: (w: Exports, args: number[]) => number,
  inputs: (string | Uint8Array)[],
): Promise<string> {
  const w = await load();
  const owned: [number, number][] = [];
  try {
    const args: number[] = [];
    for (const i of inputs) {
      const bytes = typeof i === "string" ? enc.encode(i) : i;
      const pair = put(w, bytes);
      owned.push(pair);
      args.push(pair[0], pair[1]);
    }
    return take(w, fn(w, args));
  } finally {
    for (const [ptr, len] of owned) w.dealloc(ptr, len);
  }
}

export type GgufResult =
  | { ok: true; json: Record<string, unknown> }
  | { ok: false; truncated: number | null; error: string };

/** Parse a GGUF header prefix. `truncated` is the byte count to re-read with. */
export async function gguf(header: Uint8Array): Promise<GgufResult> {
  const raw = await callWasm(
    (w, [p, l]) => w.gguf_parse(p ?? 0, l ?? 0),
    [header],
  );
  const parsed = JSON.parse(raw) as Record<string, unknown>;
  if (parsed.ok === true) return { ok: true, json: parsed };
  const error = String(parsed.error ?? "unreadable");
  return {
    ok: false,
    truncated: error === "truncated" ? Number(parsed.need ?? 0) : null,
    error,
  };
}

export async function cpuJson(
  cpuinfo: string,
  stat: string,
  hwmon: string,
): Promise<Record<string, unknown>> {
  return JSON.parse(
    await callWasm(
      (w, a) =>
        w.sys_cpu(
          a[0] ?? 0,
          a[1] ?? 0,
          a[2] ?? 0,
          a[3] ?? 0,
          a[4] ?? 0,
          a[5] ?? 0,
        ),
      [cpuinfo, stat, hwmon],
    ),
  );
}

export async function memJson(
  meminfo: string,
): Promise<Record<string, unknown>> {
  return JSON.parse(
    await callWasm((w, a) => w.sys_mem(a[0] ?? 0, a[1] ?? 0), [meminfo]),
  );
}

export async function gpuJson(
  nvidiaCsv: string,
  sysfs: string,
): Promise<Record<string, unknown>[]> {
  return JSON.parse(
    await callWasm(
      (w, a) => w.sys_gpu(a[0] ?? 0, a[1] ?? 0, a[2] ?? 0, a[3] ?? 0),
      [nvidiaCsv, sysfs],
    ),
  );
}
