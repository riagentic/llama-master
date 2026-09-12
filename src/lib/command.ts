// src/lib/command.ts — settings → the exact llama.cpp command line.
//
// Pure and total: same inputs, same argv, no I/O, no clock. The UI shows what
// this returns and the server cell spawns what this returns — there is no
// second code path that could drift from the preview the user read.
//
// Only non-default values are emitted. A command line that shows twelve flags
// means the user changed twelve things; everything else is llama.cpp's own
// default, which is the honest way to present it.

import { PARAMS } from "./params.ts";
import type { Param, Settings } from "./types.ts";
import type { EnvVar } from "./envvars.ts";
import { envPrefix } from "./envvars.ts";

export type Target = "server" | "cli";

function applies(p: Param, target: Target): boolean {
  return p.scope === "both" || p.scope === target;
}

/**
 * The flag/value pair for one parameter, or [] when llama.cpp would do the same
 * thing without it.
 *
 * Omission is judged against `llamaDef` when the catalog carries one, and only
 * otherwise against `def`. They are different questions: `def` is where the app
 * starts, `llamaDef` is what happens if the flag is absent. Conflating them
 * shipped two silent lies when upstream changed its defaults — "CPU only" that
 * offloaded to the GPU, and a 4,096-token plan that ran at the model's full
 * 1,048,576 (`types.ts:Param.llamaDef`).
 */
function emit(p: Param, value: unknown): string[] {
  const omitAt = p.llamaDef ?? p.def;
  if (p.kind === "bool") {
    const on = value === true;
    if (on === omitAt) return [];
    return on ? [p.flag] : p.offFlag ? [p.offFlag] : [];
  }
  if (value === omitAt) return [];
  const s = String(value);
  // An empty text/enum means "not set" — never emit a bare flag with no value.
  if (s === "") return [];
  // A catalog entry with no flag of its own IS its value: the extra-arguments
  // escape hatch, split on whitespace so it reads as ordinary argv.
  if (p.flag === "") return s.trim().split(/\s+/).filter(Boolean);
  return [p.flag, s];
}

/**
 * What a build declares it accepts (`src/lib/caps.ts`), or nothing.
 *
 * `null`, `undefined` and `[]` all mean NOT PROBED, and the answer to that is
 * "emit everything" — the opposite of `caps.ts:supportsFlag`, deliberately.
 * There the question is "may the tuner switch this extra thing ON?", and the
 * safe reading of silence is no. Here the question is "should the app DELETE a
 * setting the user can see?", and deleting on a guess would quietly drop flags
 * for every build the app never managed to probe. Silence changes nothing; only
 * a build that has answered, and answered that it does not know the flag, can
 * take a flag off the command line.
 */
export type BuildFlags = readonly string[] | null | undefined;

/** Has this build ANSWERED that it does not know `flag`? */
function absent(caps: BuildFlags, flag: string): boolean {
  return !!caps && caps.length > 0 && !caps.includes(flag);
}

/**
 * Has this build answered that `flag` does not take this VALUE?
 *
 * The flag existing is only half the question. `--lazy-mode` is in master and
 * its `on-direct` value is not — that arrived in PR #28136 — and master answers
 * `--lazy-mode on-direct` with `error while handling argument: invalid value`
 * and exits, which from the user's side is the same blank failure an unknown
 * flag gives. `caps.ts` records the listed values as `flag=value` entries, so
 * this is a lookup.
 *
 * A flag whose help lists NO values (`--host`, `-ts`, every number) has no
 * entries and is never judged here: silence means "the help does not say", not
 * "nothing is allowed".
 */
function valueAbsent(caps: BuildFlags, flag: string, value: string): boolean {
  if (!caps || caps.length === 0) return false;
  const prefix = flag + "=";
  if (!caps.some((c) => c.startsWith(prefix))) return false;
  return !caps.includes(prefix + value);
}

/** One parameter's tokens, with the build's own vocabulary respected.
 *
 *  Three outcomes, in order:
 *  - the build knows the flag (or has not been probed) — emit it;
 *  - it does not, but it knows this setting's OLDER spelling — emit that, so a
 *    release from before the rename still gets what the user asked for;
 *  - it knows neither — emit nothing, and `droppedFlags` says so. A flag the
 *    binary has never heard of is not a setting that is ignored, it is
 *    `unknown argument` and an exit before the model path is read. */
function emitFor(p: Param, value: unknown, caps: BuildFlags): string[] {
  const tokens = emit(p, value);
  if (tokens.length === 0) return tokens;
  const used = tokens[0] as string;
  // `p.flag === ""` is the extra-arguments escape hatch: the user typed raw
  // argv and owns it. Nothing here can tell a flag from a value in it.
  if (p.flag === "") return tokens;
  const known = !absent(caps, used) &&
    (tokens.length < 2 || !valueAbsent(caps, used, tokens[1] as string));
  if (known) return tokens;
  const old = p.legacy?.[String(value)];
  if (old && old.every((t) => !absent(caps, t))) return old;
  return [];
}

/** Settings this build cannot be told about, with the flag each one wanted.
 *
 *  Two reasons, reported the same way because the consequence is the same: the
 *  flag is not in this build at all, or it is and this VALUE is not.
 *
 *  For the UI: a setting silently dropped is a setting the user believes in
 *  that does not exist, which is the same failure the environment-variable box
 *  refuses to commit (`src/lib/envvars.ts`). Empty for an unprobed build, and
 *  empty for a setting sitting at llama.cpp's own default — there is nothing to
 *  drop when nothing was going to be emitted. */
export function droppedFlags(
  target: Target,
  opts: { settings: Settings; caps: BuildFlags },
): { key: string; label: string; flag: string }[] {
  const out: { key: string; label: string; flag: string }[] = [];
  for (const p of PARAMS) {
    if (!applies(p, target)) continue;
    const value = opts.settings[p.key] ?? p.def;
    const wanted = emit(p, value);
    if (wanted.length === 0 || p.flag === "") continue;
    if (emitFor(p, value, opts.caps).length === 0) {
      out.push({ key: p.key, label: p.label, flag: wanted[0] as string });
    }
  }
  return out;
}

/** Build argv for `llama-server` / `llama-cli`.
 *
 *  `bin` is the absolute binary path and `model` the absolute GGUF path; both
 *  are passed through untouched so the preview and the spawn agree exactly. */
export function argv(
  target: Target,
  opts: {
    bin: string;
    model: string;
    settings: Settings;
    /** The flags this build declares (`builds.caps`). Omitted = not probed,
     *  which emits everything — see `BuildFlags`. */
    caps?: BuildFlags;
  },
): string[] {
  const out: string[] = [opts.bin];
  if (opts.model) out.push("-m", opts.model);
  for (const p of PARAMS) {
    if (!applies(p, target)) continue;
    out.push(...emitFor(p, opts.settings[p.key] ?? p.def, opts.caps));
  }
  return out;
}

/** Shell-safe without quoting — the set `quote` leaves untouched. */
const SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;

/** POSIX-quote a single argv token for display and for copy-paste. */
export function quote(token: string): string {
  if (token === "") return "''";
  if (SAFE.test(token)) return token;
  return `'${token.replaceAll("'", `'\\''`)}'`;
}

/**
 * One token, quoted for display — with the user's home compacted to `$HOME`.
 *
 * Presentation only: the argv that is SPAWNED keeps the absolute path. The
 * copy-pasted line still runs identically because an unquoted `$HOME` expands
 * back to the same absolute path in any POSIX shell — which is also why the
 * compaction only applies when the remainder needs no quoting: quotes would
 * silence the `$HOME` and paste a path that does not exist. A path that needs
 * quoting keeps its absolute spelling instead.
 */
function displayToken(token: string, home: string): string {
  if (home.length > 1 && (token === home || token.startsWith(home + "/"))) {
    const rest = token.slice(home.length);
    if (rest === "" || SAFE.test(rest)) return "$HOME" + rest;
  }
  return quote(token);
}

/** The copy-pasteable one-liner shown read-only in the UI. The environment
 * prefix, when there is one, leads the way a shell would read it — the argv
 * below is unchanged, so the preview and the spawn cannot drift. */
export function commandLine(
  target: Target,
  opts: {
    bin: string;
    model: string;
    settings: Settings;
    home?: string;
    env?: readonly EnvVar[];
    caps?: BuildFlags;
  },
): string {
  const prefix = opts.env?.length ? envPrefix(opts.env) + " " : "";
  return prefix +
    argv(target, opts).map((t) => displayToken(t, opts.home ?? "")).join(" ");
}

/** Is an argv token a flag rather than a value? A value, even a numeric one,
 *  must never be mistaken for a flag. Negative numbers are the trap: `-1` and
 *  `-0.5` start with `-` exactly like `--repeat-last-n` and `-c` do. Distinguish
 *  by shape — a token is a VALUE when it is a plain negative number
 *  (`-` followed immediately by a digit), which every real llama.cpp flag never
 *  is (flags are `-m`, `-c`, `--repeat-last-n`, never `-5`). */
function isFlag(token: string): boolean {
  return token.startsWith("-") && !/^-\d/.test(token);
}

/** The same command, wrapped for reading: one flag per line with a continuation
 * marker. Long llama.cpp invocations are unreadable on one line.
 *
 * The environment prefix is one line of its own ABOVE the binary, because it
 * is not part of the argv and a `\`-continuation line that began with
 * `NAME=value` would not be one the shell reads as an assignment for the
 * command at the end of it. Copy is unaffected — the copy button takes
 * `commandLine`. */
export function commandBlock(
  target: Target,
  opts: {
    bin: string;
    model: string;
    settings: Settings;
    home?: string;
    env?: readonly EnvVar[];
    caps?: BuildFlags;
  },
): string[] {
  const parts = argv(target, opts).map((t) => displayToken(t, opts.home ?? ""));
  const lines: string[] = [];
  let cur = parts.shift() ?? "";
  while (parts.length) {
    const flag = parts.shift() as string;
    // A token that is not a flag belongs to the flag before it — including a
    // negative VALUE like `-1`, which `isFlag` keeps attached so a line never
    // reads `--repeat-last-n` with its value orphaned on the next line.
    const value = parts[0] && !isFlag(parts[0]) ? parts.shift() : null;
    lines.push(cur);
    cur = value ? `  ${flag} ${value}` : `  ${flag}`;
  }
  lines.push(cur);
  // The env prefix leads, as its own line — never appended after the command,
  // because a shell reads `NAME=value` only in FRONT of the program.
  if (opts.env?.length) lines.unshift(envPrefix(opts.env));
  return lines;
}

/** The base URL a client should use for the configured server settings. */
export function serverUrl(settings: Settings): string {
  const host = String(settings.host ?? "127.0.0.1");
  const port = Number(settings.port ?? 8080);
  // 0.0.0.0 is a bind address, not a destination — clients must dial loopback.
  const dial = host === "0.0.0.0" || host === "::" ? "127.0.0.1" : host;
  return `http://${dial}:${port}`;
}
