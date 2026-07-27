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

export type Target = "server" | "cli";

function applies(p: Param, target: Target): boolean {
  return p.scope === "both" || p.scope === target;
}

/** The flag/value pair for one parameter, or [] when it is at its default. */
function emit(p: Param, value: unknown): string[] {
  if (p.kind === "bool") {
    const on = value === true;
    if (on === p.def) return [];
    return on ? [p.flag] : p.offFlag ? [p.offFlag] : [];
  }
  if (value === p.def) return [];
  const s = String(value);
  // An empty text/enum means "not set" — never emit a bare flag with no value.
  if (s === "") return [];
  // A catalog entry with no flag of its own IS its value: the extra-arguments
  // escape hatch, split on whitespace so it reads as ordinary argv.
  if (p.flag === "") return s.trim().split(/\s+/).filter(Boolean);
  return [p.flag, s];
}

/** Build argv for `llama-server` / `llama-cli`.
 *
 *  `bin` is the absolute binary path and `model` the absolute GGUF path; both
 *  are passed through untouched so the preview and the spawn agree exactly. */
export function argv(
  target: Target,
  opts: { bin: string; model: string; settings: Settings },
): string[] {
  const out: string[] = [opts.bin];
  if (opts.model) out.push("-m", opts.model);
  for (const p of PARAMS) {
    if (!applies(p, target)) continue;
    out.push(...emit(p, opts.settings[p.key] ?? p.def));
  }
  return out;
}

/** POSIX-quote a single argv token for display and for copy-paste. */
export function quote(token: string): string {
  if (token === "") return "''";
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(token)) return token;
  return `'${token.replaceAll("'", `'\\''`)}'`;
}

/** The copy-pasteable one-liner shown read-only in the UI. */
export function commandLine(
  target: Target,
  opts: { bin: string; model: string; settings: Settings },
): string {
  return argv(target, opts).map(quote).join(" ");
}

/** The same command, wrapped for reading: one flag per line with a continuation
 *  marker. Long llama.cpp invocations are unreadable on one line. */
export function commandBlock(
  target: Target,
  opts: { bin: string; model: string; settings: Settings },
): string[] {
  const parts = argv(target, opts).map(quote);
  const lines: string[] = [];
  let cur = parts.shift() ?? "";
  while (parts.length) {
    const flag = parts.shift() as string;
    // A token that is not a flag belongs to the flag before it.
    const value = parts[0] && !parts[0].startsWith("-") ? parts.shift() : null;
    lines.push(cur);
    cur = value ? `  ${flag} ${value}` : `  ${flag}`;
  }
  lines.push(cur);
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
