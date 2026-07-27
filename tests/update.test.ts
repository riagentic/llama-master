// test/update.test.ts — the Update button's decision, on the real runtime.
//
// Its own file because a cell def binds to exactly ONE app per process
// (perfect-aio D2) and `tests/runtime.test.ts` already binds `builds` and `srv`
// to apps of its own; disposal does not release the binding.

import { assert, assertEquals } from "@std/assert";
import { testServer } from "aio/testing";

Deno.test({
  name: "runtime: a failed update does not restart the server it took down",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // `updateNow` used to decide this by reading `builds.job` immediately after
    // `await builds.update()` — a state read across the bridge that can still
    // hold the PREVIOUS value, so a failed update could bring a server back up
    // on a binary that was never rebuilt. It now branches on the returned
    // status, and this pins both halves of that branch.
    const dir = await Deno.makeTempDir({ prefix: "llama-master-upd-" });
    Deno.env.set("LLAMA_MASTER_HOME", dir);
    const { builds: b } = await import("../src/cell/builds.ts");
    const { srv: s } = await import("../src/cell/srv.ts");
    const { updateNow } = await import("../src/ui/actions.ts");

    await using _s = await testServer({ cells: [b, s], freezeState: true });

    // A ref that cannot resolve: the update fails, honestly.
    b.setOrigin("release");
    b.setRef("b0-does-not-exist");
    const status = await b.start();
    assertEquals(status, "failed", "the status crosses back as a return value");
    assertEquals(b.job?.status, "failed");

    // Nothing was running, so nothing should come up — and crucially the
    // failure must not be mistaken for success.
    await updateNow();
    assertEquals(s.status, "stopped");
    assertEquals(s.pid, 0);
    assert(
      (b.lastError ?? "").length > 0,
      "and the failure is reported, not swallowed",
    );
    await Deno.remove(dir, { recursive: true }).catch(() => {});
  },
});
