// src/cell/srv.server.ts — own the llama-server child process. SERVER ONLY.
//
// The process handle lives here, in one module-scope slot, because a
// `Deno.ChildProcess` is not serializable state and must not be put in a cell.
// What the cell holds is the observable shadow of it — pid, status, exit code,
// log tail — refreshed by a poll (src/app.ts). The runtime is told about the
// process through an `own` effect in srv.ts, so shutdown kills it even if the
// window is closed mid-generation (dep/aio/docs/state/methods.md#ownset).
//
// One server at a time, by design: two llama-servers on one GPU is a
// configuration mistake, not a feature.

import type { Exec } from "./host.server.ts";
import { paths, PLATFORM } from "./host.server.ts";

/** Trailing separator so `/builds-evil` cannot pass a `/builds` prefix test. */
const BIN_NAME = PLATFORM === "windows" ? "llama-server.exe" : "llama-server";

function buildsRoot(): string {
  const b = paths().builds;
  return b.endsWith("/") ? b : `${b}/`;
}

type Slot = {
  child: Deno.ChildProcess;
  pid: number;
  startedAt: number;
  argv: string[];
};

let slot: Slot | null = null;
let exitCode: number | null = null;
let exitedAt = 0;
/** Bounded ring of output lines, drained by the poll. */
let buffer: string[] = [];
let seq = 0;

const KEEP_LINES = 500;

function push(line: string): void {
  buffer.push(line);
  seq++;
  if (buffer.length > KEEP_LINES) buffer = buffer.slice(-KEEP_LINES);
}

async function pump(stream: ReadableStream<Uint8Array>): Promise<void> {
  const reader = stream.getReader();
  const dec = new TextDecoder();
  let rest = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    rest += dec.decode(value, { stream: true });
    const lines = rest.split("\n");
    rest = lines.pop() ?? "";
    for (const l of lines) push(l);
  }
  if (rest) push(rest);
}

/** A llama-server this app did not start, but that came from its builds
 *  directory — an orphan from a previous run of llama.master.
 *
 *  These are the reason a Start can fail on a machine that looks idle: the old
 *  process still holds its VRAM, and the new one dies with `cudaMalloc failed:
 *  out of memory`. The app knows the pid, so it can offer to stop it. */
export type Orphan = { pid: number; argv: string };

export async function findOrphans(): Promise<Orphan[]> {
  if (PLATFORM === "windows") return [];
  const out: Orphan[] = [];
  const root = buildsRoot();
  const mine = slot?.pid ?? 0;
  try {
    for await (const e of Deno.readDir("/proc")) {
      const pid = Number(e.name);
      if (!pid || pid === mine || pid === Deno.pid) continue;
      let cmd: string;
      try {
        cmd = await Deno.readTextFile(`/proc/${pid}/cmdline`);
      } catch {
        continue; // exited between readDir and read, or not ours to see
      }
      const argv = cmd.split("\0").filter(Boolean);
      if (argv.length === 0) continue;

      // Scope: ONLY processes running a binary from this app's builds
      // directory. A llama-server the user launched by hand from elsewhere is
      // theirs and must never be killed by us.
      //
      // `/proc/<pid>/exe` is the definitive answer for a normal binary; argv is
      // the fallback for anything launched through an interpreter or a wrapper
      // (`taskset … llama-server`), where argv[0] is not the binary.
      let exe = "";
      try {
        exe = await Deno.readLink(`/proc/${pid}/exe`);
      } catch {
        // Not permitted or already gone — argv still decides.
      }
      const isOurs = (path: string) =>
        path.startsWith(root) && path.endsWith(BIN_NAME);
      if (isOurs(exe) || argv.some(isOurs)) {
        out.push({ pid, argv: argv.join(" ") });
      }
    }
  } catch {
    // No /proc — nothing to find.
  }
  return out;
}

/** Stop an orphan by pid, after re-checking it is one of ours. */
export async function stopOrphan(pid: number): Promise<void> {
  const orphans = await findOrphans();
  if (!orphans.some((o) => o.pid === pid)) {
    throw new Error(`pid ${pid} is not a llama-server started from this app`);
  }
  try {
    Deno.kill(pid, "SIGTERM");
  } catch (e) {
    throw new Error(`could not stop pid ${pid}: ${e}`);
  }
  // Give it a moment to release its VRAM, then insist.
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      Deno.kill(pid, "SIGCONT" as Deno.Signal); // probe: throws when gone
    } catch {
      return;
    }
  }
  try {
    Deno.kill(pid, "SIGKILL");
  } catch {
    // Already gone.
  }
}

export type Status = {
  running: boolean;
  pid: number;
  startedAt: number;
  exitCode: number | null;
  exitedAt: number;
  argv: string[];
  /** Monotonic line counter, so the cell can tell "nothing new" cheaply. */
  seq: number;
  lines: string[];
};

/**
 * What the running process actually holds in RAM, from /proc.
 *
 * Kept out of `status()` and asynchronous on purpose: `status()` is a sync
 * snapshot of in-process state, and a blocking file read inside it would put
 * every client's next action behind a disk touch on the 1 s poll.
 */
export async function rss(): Promise<number> {
  const pid = slot?.pid ?? 0;
  if (!pid || PLATFORM === "windows") return 0;
  try {
    // statm: size resident shared text lib data dt — all in pages.
    const txt = await Deno.readTextFile(`/proc/${pid}/statm`);
    const resident = Number(txt.split(/\s+/)[1] ?? 0);
    return resident > 0 ? resident * 4096 : 0;
  } catch {
    return 0; // exited between the check and the read
  }
}

export function status(): Status {
  return {
    running: slot !== null,
    pid: slot?.pid ?? 0,
    startedAt: slot?.startedAt ?? 0,
    exitCode,
    exitedAt,
    argv: slot?.argv ?? [],
    seq,
    lines: buffer.slice(),
  };
}

/** Spawn llama-server. Throws if one is already running or the binary will not
 *  start — never returns a half-started state. */
export function start(argv: string[]): { pid: number } {
  if (slot) {
    throw new Error(`llama-server is already running (pid ${slot.pid})`);
  }
  const [bin, ...args] = argv;
  if (!bin) throw new Error("empty command");
  // The command comes from the UI (so the preview and the spawn cannot drift),
  // which means the binary must be one this app installed — not an arbitrary
  // path a stray action could smuggle in.
  if (!bin.startsWith(buildsRoot())) {
    throw new Error(
      `refusing to run ${bin}: only binaries under ${buildsRoot()} may be started`,
    );
  }

  exitCode = null;
  exitedAt = 0;
  buffer = [];
  push(`$ ${argv.join(" ")}`);

  let child: Deno.ChildProcess;
  try {
    child = new Deno.Command(bin, {
      args,
      stdout: "piped",
      stderr: "piped",
      stdin: "null",
    }).spawn();
  } catch (e) {
    throw new Error(`cannot start ${bin}: ${e}`);
  }

  const s: Slot = { child, pid: child.pid, startedAt: Date.now(), argv };
  slot = s;

  // The pumps must FINISH before the exit becomes visible. `child.status`
  // resolves as soon as the process is reaped, which can beat the last of its
  // stderr through the pipe — and the cell diagnoses the exit from exactly
  // those lines. Reporting "not running" too early therefore replaced the real
  // reason ("cudaMalloc failed: out of memory") with the generic fallback, on a
  // server that failed fast. Draining first costs nothing: the streams are
  // already closed by the time we get here.
  const drained = Promise.all([pump(child.stdout), pump(child.stderr)]);
  void child.status.then(async (st) => {
    await drained;
    exitCode = st.code;
    exitedAt = Date.now();
    push(`[llama.master] llama-server exited with code ${st.code}`);
    if (slot === s) slot = null;
  });

  return { pid: s.pid };
}

/**
 * Stop the server, but only if it is still the one identified by `pid`.
 *
 * This exists because of a bug worth remembering: the cell hands the runtime an
 * `own` effect keyed "srv:process" whose close calls stop(). Starting a second
 * time replaces that effect, and replacing DISPOSES the old one — so the first
 * start's teardown ran against the second start's process and SIGTERMed it a
 * moment after it came up. Every start after the first died with code 143.
 * Naming the process the effect owns makes the teardown a no-op once that
 * process is gone.
 */
export function stopOwned(pid: number, graceMs = 5000): Promise<void> {
  if (slot?.pid !== pid) return Promise.resolve();
  return stop(graceMs);
}

/** SIGTERM, then SIGKILL if it is still there. Resolves once it is gone. */
export async function stop(graceMs = 5000): Promise<void> {
  const s = slot;
  if (!s) return;
  try {
    s.child.kill("SIGTERM");
  } catch {
    // Already dead; the status promise below settles immediately.
  }
  const timeout = new Promise<"timeout">((r) =>
    setTimeout(() => r("timeout"), graceMs)
  );
  const done = s.child.status.then(() => "exited" as const);
  if (await Promise.race([done, timeout]) === "timeout") {
    push(`[llama.master] no exit after ${graceMs}ms — sending SIGKILL`);
    try {
      s.child.kill("SIGKILL");
    } catch {
      // Nothing left to kill.
    }
    await s.child.status;
  }
  slot = null;
}

/** Ask the running server whether it is ready to serve. */
export async function health(
  baseUrl: string,
  timeoutMs = 1500,
): Promise<{ ok: boolean; detail: string }> {
  try {
    const res = await fetch(`${baseUrl}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (res.ok) {
      await res.body?.cancel();
      return { ok: true, detail: "ready" };
    }
    // 503 while the model loads is the expected pre-ready state.
    const body = await res.text();
    return { ok: false, detail: `${res.status}: ${body.slice(0, 120)}` };
  } catch (e) {
    return { ok: false, detail: String(e) };
  }
}

/** `/props` — what the server says it actually loaded. Worth showing, because
 *  it is the ground truth against which the settings panel is only a request. */
export async function props(
  baseUrl: string,
  timeoutMs = 2000,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${baseUrl}/props`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) {
      await res.body?.cancel();
      return null;
    }
    return await res.json();
  } catch {
    return null;
  }
}

/** `llama-server --version`, for the build panel. */
export async function version(bin: string): Promise<Exec> {
  const { exec } = await import("./host.server.ts");
  return await exec(bin, ["--version"]);
}
