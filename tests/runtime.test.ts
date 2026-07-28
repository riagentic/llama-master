// test/runtime.test.ts — the cells on the REAL server runtime.
//
// Why this file exists, concretely: `builds.start` had a bug where the progress
// callback built its next value by spreading `s.job` read back out of state.
// On the real dispatch path that hands the store a proxy-derived object and the
// WHOLE action is rejected — the job froze at step 0 with nothing in the log and
// no error anywhere in the UI. `testCell` and `testUI` both went green, because
// the in-process harness does not install the same proxy guard.
//
// So: anything that writes to state from inside an async callback gets a test
// here, on `testServer`, which boots the same runtime `deno task dev` does.

import { assert, assertEquals } from "@std/assert";
import { testMultiClient, testServer } from "aio/testing";
import type { StateOf } from "aio";
import { builds } from "../src/cell/builds.ts";
import { cfg } from "../src/cell/cfg.ts";
import { models } from "../src/cell/models.ts";
import { ui } from "../src/cell/ui.ts";
import { num } from "../src/lib/params.ts";
import type { Settings } from "../src/lib/types.ts";
import { moeGguf } from "./gguf-fixture.ts";
import { join } from "@std/path";

/** Pull one cell's slice out of the server's authoritative state. */
function slice<T>(state: unknown, cellName: string): T {
  const s = (state as Record<string, unknown>)[cellName];
  if (s === undefined) {
    throw new Error(`no "${cellName}" slice in server state`);
  }
  return s as T;
}

Deno.test({
  name: "runtime: a job writes progress into state from its async callback",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // freezeState is what makes an illegal write fail loudly instead of
    // silently — it is the dev/prod default, so a test must run with it on.
    await using srv = await testServer({ cells: [builds], freezeState: true });

    builds.setOrigin("release");
    // A ref that cannot resolve: the callback fires once (step 0) before the
    // lookup fails, which is exactly the write that used to be rejected.
    builds.setRef("b0-does-not-exist");
    await builds.start();

    const s = slice<StateOf<typeof builds>>(srv.state(), "builds");
    assert(s.log.length > 0, "the progress callback must reach state");
    assert(
      s.log[0]?.includes("Looking up"),
      `expected the lookup line, got ${JSON.stringify(s.log)}`,
    );
    assertEquals(s.job?.status, "failed");
    assert((s.job?.error ?? "").length > 0, "the failure is reported");
    assert(s.lastError.length > 0);
  },
});

Deno.test({
  name: "runtime: a model scan writes its progress and results from a callback",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const dir = await Deno.makeTempDir({ prefix: "llama-master-rt-" });
    try {
      await Deno.writeFile(join(dir, "fixture.gguf"), moeGguf());
      await using srv = await testServer({
        cells: [models],
        freezeState: true,
      });

      await models.addDir(dir);
      await models.scan();

      const s = slice<StateOf<typeof models>>(srv.state(), "models");
      assertEquals(s.items.length, 1);
      assertEquals(s.items[0]?.meta?.arch, "qwen3moe");
      assertEquals(s.selected, s.items[0]?.path);
      assertEquals(s.progress, null, "progress is cleared when the scan ends");
      assertEquals(s.scanning, false);
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  },
});

Deno.test({
  name: "runtime: settings survive a round trip through the real store",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    await using srv = await testServer({ cells: [cfg], freezeState: true });
    await cfg.set("ngl", "40");
    await cfg.apply({ ctxSize: 16384 }, ["tuned"]);

    const s = slice<StateOf<typeof cfg>>(srv.state(), "cfg");
    assertEquals(s.settings.ngl, 40);
    assertEquals(s.settings.ctxSize, 16384);
    assertEquals(s.reasons, ["tuned"]);
    assert(s.touched.includes("ngl") && s.touched.includes("ctxSize"));
  },
});

Deno.test({
  name: "runtime: a restart does not kill the server it just started",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // The bug this pins, and why it is HERE and not in server.test.ts: `start`
    // hands the runtime an `own` effect keyed "srv:process". Starting again
    // REPLACES that effect, and replacing disposes the old one — whose close
    // called an unqualified stop(), so the first start's teardown SIGTERMed the
    // process the second start had just spawned. Every restart after the first
    // died with code 143 and the UI said "start doesn't work".
    //
    // `bootCells` does not run the runtime's own-effect disposal, so it stays
    // green with the bug in place. Only the real server reproduces it.
    const dir = await Deno.makeTempDir({ prefix: "llama-master-restart-" });
    Deno.env.set("LLAMA_MASTER_HOME", dir);
    const { paths } = await import("../src/cell/host.server.ts");
    const { srv: srvCell } = await import("../src/cell/srv.ts");

    const binDir = join(paths().builds, "test-build");
    await Deno.mkdir(binDir, { recursive: true });
    const bin = join(binDir, "llama-server");
    await Deno.copyFile(
      new URL("./stub-llama-server.ts", import.meta.url),
      bin,
    );
    await Deno.chmod(bin, 0o755);

    const port = () => {
      const l = Deno.listen({ port: 0 });
      const p = (l.addr as Deno.NetAddr).port;
      l.close();
      return p;
    };

    await using _s = await testServer({
      cells: [srvCell],
      freezeState: true,
    });

    const up = async () => {
      const p = port();
      await srvCell.start(
        [bin, "-m", "/models/fixture.gguf", "--port", String(p)],
        `http://127.0.0.1:${p}`,
      );
      for (let i = 0; i < 200 && srvCell.status !== "ready"; i++) {
        await new Promise((r) => setTimeout(r, 50));
        await srvCell.poll();
      }
      assertEquals(srvCell.status, "ready");
      return srvCell.pid;
    };

    const first = await up();
    // CRASH it rather than stopping it: `stop` disposes the effect cleanly, so
    // only the crash path leaves the first start's effect live to be replaced
    // — which is precisely the situation the user was in.
    Deno.kill(first, "SIGKILL");
    for (let i = 0; i < 200 && srvCell.status !== "crashed"; i++) {
      await new Promise((r) => setTimeout(r, 50));
      await srvCell.poll();
    }
    assertEquals(srvCell.status, "crashed");

    const second = await up();
    assert(second !== first, "a genuinely new process");

    // Outlive the disposal of the first start's effect.
    await new Promise((r) => setTimeout(r, 1500));
    await srvCell.poll();
    assertEquals(
      srvCell.status,
      "ready",
      `the restarted server must still be up; log:\n${srvCell.log.join("\n")}`,
    );

    await srvCell.stop();
    await Deno.remove(dir, { recursive: true }).catch(() => {});
  },
});

Deno.test("two surfaces, one state — the claim this app is built on", async () => {
  // llama.master ships an Electron window AND a browser client, and the reason it
  // could be written without a line of transport code is aio's promise that both
  // read one state. This app leaned on that promise harder than on anything else
  // and had never once tested it, because until `testMultiClient` there was
  // nothing to test it with (reported as llama-master #16).
  //
  // Real server, real sockets, real broadcast — a harness that faked the
  // transport would report success for the exact thing it exists to check.
  await using m = await testMultiClient({ cells: [cfg, ui] }, 2);

  // A setting changed on one surface reaches the other. This is the everyday
  // case: the user edits context in the window while a browser tab is open.
  await m.clients[0]!.dispatch(cfg.set.action("ctxSize", "16384"));
  await m.converged();
  assertEquals(
    num(m.clients[1]!.state<{ settings: Settings }>("cfg").settings, "ctxSize"),
    16384,
    "the second surface sees a setting changed on the first",
  );

  // Navigation is shared too, and deliberately so: `ui` is a shared cell because
  // a second window showing the same panel is the expected behaviour here.
  await m.clients[1]!.dispatch(ui.go.action("storage"));
  await m.converged();
  assertEquals(
    m.clients[0]!.state<{ tab: string }>("ui").tab,
    "storage",
    "navigation on one surface moves the other",
  );

  // The case that cannot be reasoned about from the outside: both surfaces
  // dispatch the same action in the same tick. `cfg.touched` records which
  // parameters the user has changed, and a lost update would leave it wrong.
  await m.dispatchAll(cfg.set.action("temp", "0.5"));
  await m.converged();
  const server = m.serverState<{ settings: Settings; touched: string[] }>(
    "cfg",
  );
  assertEquals(num(server.settings, "temp"), 0.5);
  assertEquals(
    server.touched.filter((k) => k === "temp").length,
    1,
    "a parameter changed by both surfaces at once is recorded once, not twice",
  );

  // And every surface agrees with the server, which is the whole promise.
  for (const c of m.clients) {
    assertEquals(
      c.state<{ touched: string[] }>("cfg").touched.slice().sort(),
      server.touched.slice().sort(),
      `client ${c.index} agrees with the server`,
    );
  }
});
