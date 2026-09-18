// src/lib/envvars.ts — the user's own environment variables for llama-server.
//
// Not argv. `GGML_CUDA_DISABLE_GRAPHS=1 llama-server …` is a shell-level
// assignment: it sets an environment variable the process reads at startup
// (GGML's CUDA graph capture, LLAMA_CURL, GGML_SCHED_MAX_SPLIT_INPUTS, …) and
// llama.cpp parses no flag for it. So the input the user types is a run of
// `NAME=value` tokens, the preview shows them as a prefix on the command the
// way a shell would, and the SPAWN carries them as `env` — the argv itself is
// untouched, and everything downstream that indexes it (`-m`, `-c`, the fit
// ladder's `-ts` surgery) cannot be moved by one byte.
//
// Pure: text in, tokens and problems out. The spawn path is in
// srv.server.ts. The tokenizer is shell-shaped — whitespace separates, except
// inside quotes — because the value is the one place a pasted line carries
// spaces, and a variable that arrives mangled is a run that differs from the
// command on screen in exactly the way this app exists to refuse.

import { quote } from "./command.ts";

/** One `NAME=value` assignment, parsed. */
export type EnvVar = { name: string; value: string };

/**
 * Tokenize the way a shell would: whitespace separates, except inside quotes.
 *
 * A pasted `LLAMA_CURL='-x http://host:1'` is ONE assignment with a space in
 * its value — splitting on whitespace alone made it two broken tokens, and a
 * variable that arrives mangled is a run that differs from the command on
 * screen in exactly the way this app refuses everywhere else. Quotes are
 * stripped by the scan (they are shell syntax, not value bytes), so the spawn
 * gets exactly what a shell would have passed.
 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  let cur = "";
  let quote: '"' | "'" | null = null;
  let started = false;
  const end = () => {
    if (started) tokens.push(cur);
    cur = "";
    started = false;
  };
  for (const ch of text) {
    if (quote) {
      if (ch === quote) quote = null;
      else cur += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      started = true; // `a"b c"` is one token, not `ab` + `c`
      continue;
    }
    if (/\s/.test(ch)) {
      end();
      continue;
    }
    cur += ch;
    started = true;
  }
  end();
  return tokens;
}

/**
 * Parse the input into assignments.
 *
 * `NAME=value` tokens. Everything else — a bare word, a `NAME=` with no
 * value, a value with a `$` — is refused and NAMED, rather than dropped: a
 * token that disappears silently is a setting the user believes in that does
 * not exist.
 *
 * `NAME=` is refused on purpose. In a shell it sets an empty variable, which
 * is occasionally meaningful for GGML (a variable that is SET-but-empty can be
 * read as "off"); through this app it is far more likely a half-typed line or
 * a pasted `NAME=$SOMETHING` whose expansion this input cannot perform — and
 * an empty variable passed through is a value the running build may treat very
 * differently from an unset one. Refusing it makes the user's next step
 * visible instead of mysterious.
 */
export function parseEnvVars(text: string): {
  vars: EnvVar[];
  bad: string[];
} {
  const vars: EnvVar[] = [];
  const bad: string[] = [];
  for (const token of tokenize(text)) {
    const eq = token.indexOf("=");
    if (eq <= 0) {
      bad.push(token);
      continue;
    }
    const name = token.slice(0, eq);
    const value = token.slice(eq + 1);
    // The shell-legal name charset: an identifier the shell would accept as an
    // assignment target. Anything else is a typo, not a variable.
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      bad.push(token);
      continue;
    }
    if (value === "" || value.includes("$")) {
      bad.push(token);
      continue;
    }
    vars.push({ name, value });
  }
  return { vars, bad };
}

/**
 * The same input as a `Record` for `Deno.Command`'s `env`.
 *
 * Last one wins, exactly as a shell would: the two entries are one input
 * line, and a duplicate is the user editing in place far more often than two
 * variables competing.
 */
/**
 * The environment a server is spawned with: the app's own, minus any
 * `LLAMA_ARG_*` the user did not put in the Command panel, plus theirs.
 *
 * llama.cpp reads every flag from `LLAMA_ARG_<NAME>` when the flag is absent
 * — and this app omits a flag whenever the setting is at llama.cpp's default.
 * So a `LLAMA_ARG_CTX_SIZE` or `LLAMA_ARG_N_GPU_LAYERS` left in a shell
 * profile or a service unit would silently rewrite a run whose command line
 * says otherwise. Pure: the parent environment is handed in.
 */
export function spawnEnv(
  parent: Readonly<Record<string, string>>,
  vars: readonly EnvVar[],
): { env: Record<string, string>; dropped: string[] } {
  const mine = new Set(vars.map((v) => v.name));
  const env: Record<string, string> = {};
  const dropped: string[] = [];
  for (const [k, v] of Object.entries(parent)) {
    if (k.startsWith("LLAMA_ARG_") && !mine.has(k)) dropped.push(k);
    else env[k] = v;
  }
  return { env: { ...env, ...envRecord(vars) }, dropped: dropped.sort() };
}

export function envRecord(vars: readonly EnvVar[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const v of vars) out[v.name] = v.value;
  return out;
}

/**
 * The prefix a shell command would carry, quoted for display and copy.
 *
 * `GGML_CUDA_DISABLE_GRAPHS=1` for one variable, the two joined by a space for
 * two. What a copy-pasted line needs to reproduce the run: assignments in
 * front, command after — which is how `commandLine`/`commandBlock` render it.
 * Quoted as one token per assignment, because a value containing a space is a
 * single shell word only when quoted.
 */
export function envPrefix(vars: readonly EnvVar[]): string {
  return vars.map((v) => `${v.name}=${quote(v.value)}`).join(" ");
}
