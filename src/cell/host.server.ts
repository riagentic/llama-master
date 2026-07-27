// src/cell/host.server.ts — the host boundary: paths, process spawning, HTTP
// downloads, archive extraction.
//
// SERVER ONLY. Never import this from a cell module's top level or from any
// TSX — cells reach it with `await import("./host.server.ts")` inside an async
// method, which aio's bundler marks external, so `Deno.*` never reaches the
// browser (dep/aio/docs/build/imports.md).
//
// Every function here is loud on failure: a non-zero exit, a 404, a missing
// binary all raise with the command or URL in the message. Silence is the one
// behaviour this layer is not allowed to have.

import { appDirs } from "aio/server";
import { dirname, join } from "@std/path";
import { safeEntries, stripRoot, untargz, unzip } from "../lib/archive.ts";
import { isRateLimited, rateLimitMessage } from "../lib/github.ts";

export const APP_ID = "llama-master";
export const PLATFORM = Deno.build.os;
export const ARCH = Deno.build.arch;

/** Every directory llama.master writes to, derived from aio's one app home so
 *  a backup of `~/.llama-master` is genuinely everything. */
export type Paths = {
  home: string;
  /** Durable: finished builds the user expects to survive. */
  builds: string;
  /** Disposable: tarballs, extracted sources, cmake build trees. */
  cache: string;
  downloads: string;
  sources: string;
  toolchain: string;
  logs: string;
};

/**
 * Where the app home lives, when it must not be `~/.llama-master`.
 *
 * Two callers: a user relocating the app's data (builds are gigabytes), and the
 * tests — which otherwise write fixture builds into the real install, because
 * `bootCells` does not redirect the app home the way `testServer` does and aio
 * exports no `registerAppDirs` (reported in dep/aio/feedback/llama-master.md).
 */
export const HOME_ENV = "LLAMA_MASTER_HOME";

export function paths(): Paths {
  const d = appDirs(APP_ID, Deno.env.get(HOME_ENV) || undefined);
  // `<home>/cache` is ours to name: aio's AppDirs stopped exposing a `cache`
  // field in alpha38. It stays inside the one app directory (so "delete the
  // app" is still one `rm -rf`) and outside `data/` (so a backup does not drag
  // in a 20 GB source tree that a rebuild would recreate).
  const cache = join(d.home, "cache");
  return {
    home: d.home,
    builds: join(d.files, "builds"),
    cache,
    downloads: join(cache, "downloads"),
    sources: join(cache, "sources"),
    toolchain: join(cache, "toolchain"),
    logs: d.logs,
  };
}

export async function ensureDir(dir: string): Promise<void> {
  await Deno.mkdir(dir, { recursive: true });
}

export async function exists(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

/** Total size of a directory tree, in bytes. Used for "reclaim space". */
export async function dirSize(dir: string): Promise<number> {
  let total = 0;
  try {
    for await (const e of Deno.readDir(dir)) {
      const p = join(dir, e.name);
      if (e.isDirectory) total += await dirSize(p);
      else if (e.isFile) total += (await Deno.stat(p)).size;
    }
  } catch {
    // A directory that vanished mid-walk contributes nothing; the caller is
    // showing a size, not enforcing a quota.
  }
  return total;
}

// ── processes ──────────────────────────────────────────────────────────────

export type Exec = { code: number; stdout: string; stderr: string };

/** Run a command to completion and capture its output. */
export async function exec(
  bin: string,
  args: string[] = [],
  opts: { cwd?: string; env?: Record<string, string> } = {},
): Promise<Exec> {
  try {
    const out = await new Deno.Command(bin, {
      args,
      cwd: opts.cwd,
      env: opts.env,
      stdout: "piped",
      stderr: "piped",
    }).output();
    const d = new TextDecoder();
    return {
      code: out.code,
      stdout: d.decode(out.stdout),
      stderr: d.decode(out.stderr),
    };
  } catch (e) {
    // NotFound is the common case (binary absent) — report it as an exit
    // status rather than throwing, so callers can probe without try/catch.
    return { code: 127, stdout: "", stderr: String(e) };
  }
}

/** Resolve a binary on PATH without depending on `which`/`where` existing. */
export async function which(bin: string): Promise<string | null> {
  const path = Deno.env.get("PATH") ?? "";
  const sep = PLATFORM === "windows" ? ";" : ":";
  const exts = PLATFORM === "windows" ? [".exe", ".cmd", ".bat", ""] : [""];
  for (const dir of path.split(sep).filter(Boolean)) {
    for (const ext of exts) {
      const p = join(dir, bin + ext);
      if (await exists(p)) return p;
    }
  }
  return null;
}

export type LineSink = (line: string, stream: "out" | "err") => void;

/** Run a command, streaming each output line to `onLine` as it appears.
 *
 *  Long builds are the reason this exists: a cmake run that prints nothing for
 *  four minutes is indistinguishable from a hang, so the UI needs the lines
 *  while they happen, not at the end. */
export async function execStream(
  bin: string,
  args: string[],
  opts: { cwd?: string; env?: Record<string, string>; signal?: AbortSignal },
  onLine: LineSink,
): Promise<number> {
  const child = new Deno.Command(bin, {
    args,
    cwd: opts.cwd,
    env: opts.env,
    stdout: "piped",
    stderr: "piped",
  }).spawn();

  const abort = () => {
    try {
      child.kill("SIGTERM");
    } catch {
      // Already gone — the exit status below is what the caller acts on.
    }
  };
  opts.signal?.addEventListener("abort", abort, { once: true });

  const pump = async (
    stream: ReadableStream<Uint8Array>,
    tag: "out" | "err",
  ) => {
    const reader = stream.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const l of lines) onLine(l, tag);
    }
    if (buf) onLine(buf, tag);
  };

  const [status] = await Promise.all([
    child.status,
    pump(child.stdout, "out"),
    pump(child.stderr, "err"),
  ]);
  opts.signal?.removeEventListener("abort", abort);
  return status.code;
}

// ── network ────────────────────────────────────────────────────────────────

const UA = "llama-master/1.0 (+https://github.com/ggml-org/llama.cpp)";

/** A token raises GitHub's limit from 60/hour to 5000. Optional by design —
 *  the app works without one (see lib/github.ts), this just makes it roomier. */
function githubAuth(url: string): Record<string, string> {
  if (!url.includes("github.com")) return {};
  const token = Deno.env.get("GITHUB_TOKEN") ?? Deno.env.get("GH_TOKEN") ?? "";
  return token ? { authorization: `Bearer ${token}` } : {};
}

/** Thrown when GitHub refused on quota — callers can fall back rather than
 *  showing the user a bare 403. */
export class RateLimited extends Error {
  constructor(message: string, readonly resetAt: number) {
    super(message);
    this.name = "RateLimited";
  }
}

export async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: {
      "user-agent": UA,
      accept: "application/json",
      ...githubAuth(url),
    },
  });
  if (!res.ok) {
    const headers = res.headers;
    await res.body?.cancel();
    if (isRateLimited(res.status, headers)) {
      const { rateLimitReset } = await import("../lib/github.ts");
      throw new RateLimited(rateLimitMessage(headers), rateLimitReset(headers));
    }
    throw new Error(`GET ${url} → ${res.status} ${res.statusText}`);
  }
  return await res.json() as T;
}

/** Fetch a plain github.com page (not the API — no rate limit). */
export async function fetchText(url: string): Promise<string> {
  const res = await fetch(url, { headers: { "user-agent": UA } });
  if (!res.ok) throw new Error(`GET ${url} → ${res.status} ${res.statusText}`);
  return await res.text();
}

/** Follow a redirect without downloading the body — used to read the tag the
 *  `/releases/latest` URL points at. */
export async function resolveRedirect(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: { "user-agent": UA },
    redirect: "manual",
  });
  await res.body?.cancel();
  return res.headers.get("location") ?? res.url ?? url;
}

/** Download to memory with progress. Returns the bytes; the caller decides
 *  whether they land in a file or straight into an extractor. */
export async function download(
  url: string,
  onProgress: (received: number, total: number | null) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const res = await fetch(url, {
    headers: { "user-agent": UA, ...githubAuth(url) },
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`GET ${url} → ${res.status} ${res.statusText}`);
  }
  const len = res.headers.get("content-length");
  const total = len ? parseInt(len, 10) : null;
  const chunks: Uint8Array[] = [];
  let received = 0;
  let lastTick = 0;
  for await (const chunk of res.body) {
    chunks.push(chunk);
    received += chunk.length;
    // Report at most ~20×/s: each call is a state dispatch and a re-render.
    const now = performance.now();
    if (now - lastTick > 50) {
      lastTick = now;
      onProgress(received, total);
    }
  }
  onProgress(received, total);
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

// ── archives ───────────────────────────────────────────────────────────────

/** Extract a `.tar.gz` or `.zip` into `dest`, dropping the wrapper directory
 *  every release archive has. Returns the number of files written. */
export async function extract(
  archive: Uint8Array,
  dest: string,
  kind: "tar.gz" | "zip",
): Promise<number> {
  const raw = kind === "zip" ? await unzip(archive) : await untargz(archive);
  const entries = safeEntries(stripRoot(raw));
  if (entries.length === 0) {
    throw new Error("archive contained no extractable files");
  }
  // Files first: a symlink written before its target would dangle on any
  // filesystem that validates the link, and would copy nothing on Windows.
  const links = entries.filter((e) => e.link !== undefined);
  for (const e of entries) {
    if (e.link !== undefined) continue;
    const target = join(dest, e.name);
    await ensureDir(dirname(target));
    await Deno.writeFile(target, e.bytes);
    if (PLATFORM !== "windows" && e.mode) {
      await Deno.chmod(target, e.mode);
    }
  }
  for (const e of links) {
    const target = join(dest, e.name);
    await ensureDir(dirname(target));
    await Deno.remove(target).catch(() => {});
    try {
      await Deno.symlink(e.link as string, target);
    } catch {
      // Windows without developer mode cannot create symlinks. Copy the target
      // instead: the point is that `libllama.so.0` resolves, not how.
      const source = join(dirname(target), e.link as string);
      await Deno.copyFile(source, target).catch(() => {
        throw new Error(
          `cannot create ${e.name} → ${e.link}: neither a symlink nor a copy worked`,
        );
      });
    }
  }
  return entries.length;
}

/** Mark a file executable (extraction from a zip often loses the bit). */
export async function makeExecutable(path: string): Promise<void> {
  if (PLATFORM === "windows") return;
  try {
    await Deno.chmod(path, 0o755);
  } catch (e) {
    throw new Error(`cannot make ${path} executable: ${e}`);
  }
}
