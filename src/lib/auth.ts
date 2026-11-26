// src/lib/auth.ts — the bearer token the running llama-server demands.
//
// `apiKey` reaches llama-server as `--api-key <key>` (src/lib/params.ts), and
// with one set llama-server answers 401 to everything but its public
// endpoints. Every request this app makes to its own server therefore has to
// carry the same key — read off the argv the server was actually started
// with, so it is the key in force, not whatever the settings panel says now.
// Pure: no I/O, so the rule is testable without a server.

import type { EnvVar } from "./envvars.ts";

/** The key a llama-server started with `argv` (and `env`) requires, or "". */
export function apiKeyOf(
  argv: readonly string[],
  env: readonly EnvVar[] = [],
): string {
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] ?? "";
    if (a === "--api-key") return argv[i + 1] ?? "";
    if (a.startsWith("--api-key=")) return a.slice("--api-key=".length);
  }
  // llama.cpp's own spelling of the same setting as a variable.
  const v = env.find((e) => e.name === "LLAMA_API_KEY");
  return v?.value ?? "";
}

/** The header that carries it, or none. Keys are comma-separated lists to
 *  llama-server; the first one is as good as any. */
export function authHeaders(key: string): Record<string, string> {
  const first = key.split(",")[0]?.trim() ?? "";
  return first ? { authorization: `Bearer ${first}` } : {};
}
