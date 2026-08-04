// client/sync-shared.ts — copy the pure libraries the client shares with
// llama.master into a directory the client's own dev server can serve.
//
// Why a copy and not an import: aio serves the browser bundle out of the app's
// baseDir and REFUSES to follow a symlink that escapes it (server-static.ts —
// "Symlinks inside baseDir must not escape it either"). That is a directory
// traversal guard and it is right; the client just has to live inside its own
// root. Two apps, two roots, one source of truth — so the copy is mechanical,
// marked as generated, and `tests/shared.test.ts` fails the moment it drifts.
//
// The same shape as this repo's other generated artifact, `src/llama-sys.wasm`:
// committed so nothing has to be built to run, with a guard test that refuses
// to let it fall behind its source.
//
//   deno task sync   (after changing anything in ../src/lib that is listed here)

import { dirname, fromFileUrl, join } from "@std/path";

/** What the client actually uses, and nothing more. Every entry is imported by
 *  code that reaches the BROWSER, which is the only reason a copy is needed. */
export const SHARED = [
  "format.ts", // bytes / tok/s / durations, in the app's own words
  "richtext.ts", // a reply is text and fenced blocks
  "scroll.ts", // keep the newest line in view
  "sse.ts", // what a token event contains, and how often to publish
  "types.ts", // ChatMessage, and what richtext reads from it
  "disk.ts", // types.ts re-exports it
] as const;

const HERE = dirname(fromFileUrl(import.meta.url));
export const SRC = join(HERE, "..", "src", "lib");
export const DEST = join(HERE, "src", "shared");

const BANNER = (name: string) =>
  `// GENERATED — do not edit. Copied from ../../../src/lib/${name} by
// \`deno task sync\` (client/sync-shared.ts), because aio serves the browser
// bundle only from inside the app's own root. Edit the original.
`;

/** The text as it must appear in `src/shared`. Exported so the guard test can
 *  compare without re-implementing the rule. */
export async function rendered(name: string): Promise<string> {
  return BANNER(name) + await Deno.readTextFile(join(SRC, name));
}

if (import.meta.main) {
  await Deno.mkdir(DEST, { recursive: true });
  for (const name of SHARED) {
    await Deno.writeTextFile(join(DEST, name), await rendered(name));
    console.log(`synced ${name}`);
  }
}
