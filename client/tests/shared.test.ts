// client/tests/shared.test.ts — the copy must not drift from the original.
//
// `src/shared` is a mechanical copy of llama.master's pure libraries, made by
// `deno task sync` because aio serves the browser bundle only from inside the
// app's own root and refuses to follow a symlink out of it. A copy that can
// drift is the reason people distrust copies — so this test is the thing that
// makes it safe: change `src/lib/richtext.ts` and forget the client, and the
// suite says so, naming the command that fixes it.

import { assertEquals } from "@std/assert";
import { join } from "@std/path";
import { DEST, rendered, SHARED } from "../sync-shared.ts";

Deno.test("shared: the client's copies are the originals, byte for byte", async () => {
  for (const name of SHARED) {
    const have = await Deno.readTextFile(join(DEST, name)).catch(() => "");
    assertEquals(
      have,
      await rendered(name),
      `client/src/shared/${name} is out of date — run \`deno task sync\` in client/`,
    );
  }
});

/** The list is the contract: a shared file that grows a new runtime import
 *  would break the browser bundle at boot with a 403, not at build time. */
Deno.test("shared: nothing in the copy imports outside the copy", async () => {
  for (const name of SHARED) {
    const src = await Deno.readTextFile(join(DEST, name));
    const imports = [...src.matchAll(/from\s+"(\.[^"]+)"/g)].map((m) => m[1]!);
    for (const spec of imports) {
      const file = spec.replace(/^\.\//, "");
      assertEquals(
        SHARED.includes(file as typeof SHARED[number]),
        true,
        `${name} imports ${spec}, which is not synced — add it to SHARED in client/sync-shared.ts`,
      );
    }
  }
});
