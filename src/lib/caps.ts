// src/lib/caps.ts — what THIS build of llama.cpp can actually be told to do.
//
// The app's promise is that the command it shows is the command that runs, and
// a settings panel is a set of claims about a binary. Both quietly assume every
// build understands every flag in the catalog, and that is false in two
// directions at once: a release from six months ago has never heard of `-bs` or
// `--fit`, and a source build carrying a pull request understands `--lazy-mode
// on-direct`, which nothing else does.
//
// Until now the app guessed, and the guess was "off by default, so it cannot
// hurt". That works right up to the moment the tuner wants to switch something
// ON — which is the whole point of a tuner. `-bs` is the case that forced this:
// it is a real speed lever, it is safe to take, and setting it unconditionally
// would make every older build fail to start with `unknown argument: -bs`. A
// failed start is a bad way to learn a build's age.
//
// So ask the binary. `llama-server --help` prints every flag it accepts, it
// costs about ten milliseconds, and the answer is a FACT about that build
// rather than an inference from its version string — which is what the app
// would otherwise be reduced to, and which is wrong for exactly the interesting
// builds (a PR stack has no version number that means anything).
//
// Pure: help text in, a set of flags out.

/**
 * Every flag this help text declares.
 *
 * The parse is deliberately structural rather than clever. llama.cpp lays out
 * one option per line — aliases first, then a value placeholder, then the
 * description at a fixed column — and continuation lines are indented:
 *
 *     -t,    --threads N                      number of CPU threads to use
 *                                             (env: LLAMA_ARG_THREADS)
 *     --cpu-strict <0|1>                      use strict CPU placement
 *
 * So a line that starts with `-` opens an option, and the aliases are whatever
 * precedes the description. Finding where the description starts by COLUMN
 * would be a guess about a layout that upstream is free to change; finding it
 * by "two or more spaces" alone would cut `-t,    --threads` in half. The rule
 * that survives both is: split at the first run of two-plus spaces that is
 * NOT followed by another flag. Alias gaps always are; the description never
 * is, because descriptions begin with a word.
 *
 * Descriptions mention other flags all the time ("same as --threads"), which is
 * exactly why the description is cut off before anything is extracted.
 */
export function parseHelpFlags(help: string): Set<string> {
  const out = new Set<string>();
  /** The option the indented lines below currently belong to. */
  let openFlags: string[] = [];
  for (const line of help.split("\n")) {
    if (!line.startsWith("-")) {
      for (const v of valuesFromContinuation(line)) {
        for (const f of openFlags) out.add(`${f}=${v}`);
      }
      continue;
    }
    const head = splitHead(line);
    const flags: string[] = [];
    for (const tok of head.split(/[,\s]+/)) {
      // `-h`, `--help`, `--cache-reuse`. Rejects value placeholders (`N`,
      // `<0|1>`, `lo-hi`) and anything that is not shaped like a flag.
      if (/^--?[A-Za-z][A-Za-z0-9-]*$/.test(tok)) flags.push(tok);
    }
    for (const f of flags) out.add(f);
    openFlags = flags;
    for (const v of valuesFromHead(head, flags)) {
      for (const f of flags) out.add(`${f}=${v}`);
    }
  }
  return out;
}

/**
 * The values an option ACCEPTS, as `--flag=value` entries beside the flag.
 *
 * The flag existing is not the whole question, and the half that was missing is
 * the half that bites: master knows `--lazy-mode` and answers `--lazy-mode
 * on-direct` with `error while handling argument "--lazy-mode": invalid value`,
 * because that value arrived in a pull request (#28136) that master does not
 * carry. Same shape as an unknown flag from the user's side — the server exits
 * before it reads the model path — and the app has to tell them apart to say
 * anything useful, since dropping the whole setting and falling back to
 * llama.cpp's default is right in both cases.
 *
 * llama.cpp writes the list four ways and all four are read, because which one
 * an option uses is not something this app gets to choose:
 *
 *   -ctk,  --cache-type-k TYPE       KV cache data type for K
 *                                    allowed values: f32, f16, bf16, q8_0, …
 *   -lm,   --load-mode MODE          model loading mode (default: auto)
 *                                    - auto: mmap, unless a device does not …
 *   -sm,   --split-mode {none,layer,row,tensor}
 *   --spec-type none,draft-simple,draft-eagle3,draft-mtp,…
 *
 * An option whose values are NOT listed contributes no `=` entries at all, and
 * the caller must then emit whatever it was going to — silence is "this help
 * does not say", never "no values are allowed".
 */
const VALUE = /^[A-Za-z0-9][A-Za-z0-9_.+-]*$/;

/** A comma/pipe list in the option's own placeholder. */
function valuesFromHead(head: string, flags: readonly string[]): string[] {
  // Everything after the last recognised alias is the placeholder.
  const last = flags[flags.length - 1];
  const at = last ? head.lastIndexOf(last) + last.length : 0;
  const ph = head.slice(at).trim().replace(/^[{[(]|[}\])]$/g, "");
  // `N0,N1,N2,...` and `MiB0,MiB1,...` are a SHAPE, not a list of choices, and
  // they say so with the ellipsis. Without this check `-ts` would "allow" the
  // literal value `N0`.
  if (!ph || ph.includes("...")) return [];
  if (!/[,|]/.test(ph)) return [];
  const parts = ph.split(/[,|]/).map((x) => x.trim());
  return parts.every((x) => VALUE.test(x)) ? parts : [];
}

/** `allowed values: …` and `- value: description` under an option. */
function valuesFromContinuation(line: string): string[] {
  const t = line.trim();
  const allowed = t.match(/^allowed values:\s*(.+)$/i);
  if (allowed) {
    const parts = (allowed[1] as string).split(",").map((x) => x.trim());
    return parts.every((x) => VALUE.test(x)) ? parts : [];
  }
  // `- layer (default): split layers …` — the parenthetical is llama.cpp's, not
  // part of the value.
  const bullet = t.match(/^-\s+([A-Za-z0-9][A-Za-z0-9_.+-]*)(\s+\([^)]*\))?:/);
  return bullet ? [bullet[1] as string] : [];
}

/** The part of an option line before its description. */
function splitHead(line: string): string {
  const re = /\s{2,}/g;
  for (let m = re.exec(line); m; m = re.exec(line)) {
    const next = line[m.index + m[0].length];
    // A gap followed by another flag is an alias separator, not the margin.
    if (next !== "-") return line.slice(0, m.index);
  }
  return line;
}

/**
 * Does this build accept `flag`?
 *
 * `null` capabilities mean "not probed", and the answer is then `false` for
 * every flag — deliberately. An unprobed build is one the app knows nothing
 * about, and the safe reading of nothing is "do not switch anything extra on".
 * The alternative reading, "assume modern", is how a tuner produces a command
 * that cannot start.
 */
export function supportsFlag(
  flags: readonly string[] | null | undefined,
  flag: string,
): boolean {
  return flags ? flags.includes(flag) : false;
}

/**
 * How long a probe may take.
 *
 * `--help` prints and exits without touching a model or a device, so a second
 * is generous. It is bounded at all because this runs against a binary the app
 * did not necessarily build: a wedged or mismatched executable must not hold up
 * a scan (the same reasoning as `hw.server.ts:nvidiaSmi`).
 */
export const CAPS_TIMEOUT_MS = 5_000;
