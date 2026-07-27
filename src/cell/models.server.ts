// src/cell/models.server.ts — find GGUF files and read their headers.
// SERVER ONLY.
//
// Header reads are progressive: ask the Rust parser for 1 MB, and if it says
// "truncated, need N" read N and ask again. A 4-bit 70B has a ~2 MB header
// (the tokenizer vocab dominates) while a small model has ~200 KB — this reads
// what each file actually needs instead of a fixed worst case, which is the
// difference between a scan taking a second and taking a minute.

import { basename, dirname, join } from "@std/path";
import type { Model, ModelMeta, ModelSource } from "../lib/types.ts";
import {
  isOllamaStore,
  manifestSkipReason,
  nameFromManifestPath,
  resolveManifest,
} from "../lib/ollama.ts";
import { exists, PLATFORM } from "./host.server.ts";
import { gguf } from "./wasm.server.ts";
import { DEMO_ENV, demoModels } from "../lib/demo.ts";

/** Directories worth looking in before the user has configured anything.
 *
 *  The last three are other tools' stores. People who already run LM Studio or
 *  ollama have tens of gigabytes of models on disk; asking them to re-download
 *  or to hunt for a path is the wrong first impression. */
export function defaultDirs(): string[] {
  const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? "";
  // ollama's system-service store. Linux packages install it here; on macOS
  // and Windows ollama runs as the user, so only the per-user path below
  // applies and this one simply will not exist.
  const dirs = PLATFORM === "linux" ? ["/usr/share/ollama/.ollama/models"] : [];
  if (!home) return dirs;
  const platformDirs = PLATFORM === "darwin"
    // Where a Mac user actually keeps large files, and LM Studio's own default.
    ? [
      join(home, "Documents", "models"),
      join(home, "Library", "Caches", "llama.cpp"),
    ]
    : PLATFORM === "windows"
    ? [
      join(home, "Documents", "models"),
      join(home, "AppData", "Local", "llama.cpp"),
    ]
    : [];
  return [
    join(home, "models"),
    join(home, "gguf"),
    // llama.cpp's own `--hf-repo` download cache.
    join(home, ".cache", "llama.cpp"),
    // huggingface-cli / transformers cache.
    join(home, ".cache", "huggingface", "hub"),
    join(home, ".local", "share", "models"),
    // LM Studio keeps plain .gguf files under publisher/repo/.
    join(home, ".lmstudio", "models"),
    // ollama installed as the current user.
    join(home, ".ollama", "models"),
    ...platformDirs,
    ...dirs,
  ];
}

/** Which tool a path belongs to, for the "source" column. */
function sourceOf(path: string): ModelSource {
  return path.includes("/.lmstudio/") ? "lmstudio" : "file";
}

/** `mmproj-*.gguf` is a multimodal projector — a companion file, not a model.
 *  Listing them puts unloadable entries in the library. */
function isProjector(file: string): boolean {
  return /^mmproj[-.]/i.test(file);
}

const SHARD = /-(\d{5})-of-(\d{5})\.gguf$/i;

/** First shard of a split model, or the file itself. Multi-part GGUFs are
 *  loaded by pointing `-m` at part 1; listing all five parts as five models
 *  would be five wrong answers. */
function shardInfo(file: string): { isShard: boolean; isFirst: boolean } {
  const m = SHARD.exec(file);
  if (!m) return { isShard: false, isFirst: true };
  return { isShard: true, isFirst: m[1] === "00001" };
}

async function* walk(
  dir: string,
  depth: number,
): AsyncGenerator<{ path: string; size: number; mtime: number }> {
  if (depth < 0) return;
  let entries: AsyncIterable<Deno.DirEntry>;
  try {
    entries = Deno.readDir(dir);
  } catch {
    return; // unreadable or missing — not an error, just nothing here
  }
  for await (const e of entries) {
    if (e.name.startsWith(".")) continue;
    const p = join(dir, e.name);
    if (e.isDirectory) {
      yield* walk(p, depth - 1);
    } else if (e.isFile && e.name.toLowerCase().endsWith(".gguf")) {
      try {
        const st = await Deno.stat(p);
        yield { path: p, size: st.size, mtime: st.mtime?.getTime() ?? 0 };
      } catch {
        // Vanished between readDir and stat.
      }
    }
  }
}

/** Resolve every locally-present model in an ollama store.
 *
 *  Cloud models (`"layers": null`) are registered locally but have no weights
 *  on disk; they are skipped, because a model that cannot be loaded has no
 *  business in a list of models you can run. */
async function walkOllama(
  root: string,
): Promise<
  { path: string; name: string; size: number; mtime: number; error?: string }[]
> {
  const out: {
    path: string;
    name: string;
    size: number;
    mtime: number;
    error?: string;
  }[] = [];
  const manifests = join(root, "manifests");
  const blobs = join(root, "blobs");

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth < 0) return;
    let entries: AsyncIterable<Deno.DirEntry>;
    try {
      entries = Deno.readDir(dir);
    } catch {
      return; // unreadable (another user's store) — not an error
    }
    for await (const e of entries) {
      const p = join(dir, e.name);
      if (e.isDirectory) {
        await walk(p, depth - 1);
        continue;
      }
      if (!e.isFile) continue;
      let json: string;
      try {
        json = await Deno.readTextFile(p);
      } catch {
        continue;
      }
      const m = resolveManifest(p, json);
      if (!m) {
        // A cloud-only entry has nothing on disk to run and is skipped in
        // silence. A manifest this app cannot parse is a different thing: a
        // model the user HAS, which would simply not appear, for a reason the
        // app knew and kept to itself. Listed with the reason instead — that is
        // what `metaError` is for.
        if (manifestSkipReason(p, json) === "unreadable") {
          out.push({
            path: p,
            name: nameFromManifestPath(p) || p,
            size: 0,
            mtime: 0,
            error: "ollama manifest could not be read",
          });
        }
        continue;
      }
      const blobPath = join(blobs, m.blob);
      try {
        const st = await Deno.stat(blobPath);
        out.push({
          path: blobPath,
          name: m.name,
          size: m.sizeB || st.size,
          mtime: st.mtime?.getTime() ?? 0,
        });
      } catch {
        // Manifest references a blob that is not here — a partial pull.
      }
    }
  }

  await walk(manifests, 6);
  return out;
}

const FIRST_READ = 1024 * 1024;
const MAX_READ = 64 * 1024 * 1024;

/** Read as much of the header as the parser asks for, and no more. */
export async function readMeta(
  path: string,
): Promise<{ meta: ModelMeta | null; error: string | null }> {
  let want = FIRST_READ;
  for (let attempt = 0; attempt < 4; attempt++) {
    let head: Uint8Array;
    try {
      const f = await Deno.open(path, { read: true });
      try {
        head = new Uint8Array(want);
        const n = await f.read(head);
        head = head.subarray(0, n ?? 0);
      } finally {
        f.close();
      }
    } catch (e) {
      return { meta: null, error: `cannot read: ${e}` };
    }

    const r = await gguf(head);
    if (r.ok) return { meta: r.json as unknown as ModelMeta, error: null };
    if (r.truncated === null) return { meta: null, error: r.error };
    // The parser reports the offset it needed; add slack so the next tensor
    // entry is covered too rather than round-tripping per entry.
    want = Math.min(MAX_READ, Math.max(r.truncated * 2, want * 4));
    if (want >= head.length && head.length < want) continue;
  }
  return {
    meta: null,
    error: "header larger than 64 MB — refusing to read on",
  };
}

export type ScanProgress = (
  done: number,
  total: number,
  current: string,
) => void;

/** Find every GGUF under `dirs` and parse each header. */
export async function scan(
  dirs: string[],
  onProgress: ScanProgress,
  depth = 4,
): Promise<Model[]> {
  // Demo mode: a fictional library, so no screenshot or bug report carries the
  // author's model collection or their paths (src/lib/demo.ts).
  if (Deno.env.get(DEMO_ENV) === "1") {
    const list: Model[] = demoModels().map((m) => ({
      ...m,
      dir: "/models",
      mtime: 0,
      metaError: null,
    }));
    onProgress(list.length, list.length, "");
    return list;
  }
  const found: { path: string; size: number; mtime: number }[] = [];
  const seen = new Set<string>();
  // ollama models are keyed by digest, so they carry a name the file system
  // does not know. Collected separately and merged in below.
  const named = new Map<string, string>();
  // Manifests found but unreadable: listed with the reason rather than dropped.
  const brokenManifests = new Map<string, string>();

  for (const dir of dirs) {
    if (!(await exists(dir))) continue;

    // An ollama store has no `.gguf` files at all — its weights are
    // `blobs/sha256-<hex>`, reachable only through the manifests.
    let entries: string[] = [];
    try {
      for await (const e of Deno.readDir(dir)) entries.push(e.name);
    } catch {
      entries = [];
    }
    if (isOllamaStore(entries)) {
      for (const m of await walkOllama(dir)) {
        if (seen.has(m.path)) continue;
        seen.add(m.path);
        named.set(m.path, m.name);
        if (m.error) brokenManifests.set(m.path, m.error);
        found.push({ path: m.path, size: m.size, mtime: m.mtime });
      }
      continue;
    }

    for await (const f of walk(dir, depth)) {
      if (seen.has(f.path)) continue; // overlapping roots
      seen.add(f.path);
      found.push(f);
    }
  }

  // Collapse shard sets onto their first part, summing the size.
  const shardTotals = new Map<string, number>();
  for (const f of found) {
    const info = shardInfo(basename(f.path));
    if (!info.isShard) continue;
    const key = f.path.replace(SHARD, "");
    shardTotals.set(key, (shardTotals.get(key) ?? 0) + f.size);
  }
  const primary = found
    .filter((f) => shardInfo(basename(f.path)).isFirst)
    .filter((f) => !isProjector(basename(f.path)));

  const models: Model[] = [];
  for (let i = 0; i < primary.length; i++) {
    const f = primary[i];
    if (!f) continue;
    onProgress(i, primary.length, basename(f.path));
    // A manifest we could not parse has no blob to read a header from, so do
    // not try — report the manifest problem itself.
    const broken = brokenManifests.get(f.path);
    const { meta, error } = broken
      ? { meta: null, error: broken }
      : await readMeta(f.path);
    const key = f.path.replace(SHARD, "");
    const ollamaName = named.get(f.path);
    models.push({
      path: f.path,
      file: ollamaName ?? basename(f.path),
      source: ollamaName ? "ollama" : sourceOf(f.path),
      dir: dirname(f.path),
      sizeB: shardTotals.get(key) ?? f.size,
      mtime: f.mtime,
      meta,
      metaError: error,
    });
  }
  onProgress(primary.length, primary.length, "");
  return models.sort((a, b) => a.file.localeCompare(b.file));
}
