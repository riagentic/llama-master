// test/cells.test.ts — cell behaviour: guards, error surfacing, and the
// invariants that keep the UI honest.
//
// These run the real dispatch loop. Where a method does real I/O (a model scan,
// a hardware sample) the test does the real I/O too — on temp directories and
// on this machine — because a mocked filesystem would only prove the mock
// agrees with the code.

import { assert, assertEquals } from "@std/assert";
import { bootCells, testCell } from "aio/testing";
import { join } from "@std/path";

import { cfg } from "../src/cell/cfg.ts";
import { ui } from "../src/cell/ui.ts";
import { hw } from "../src/cell/hw.ts";
import { models } from "../src/cell/models.ts";
import { builds } from "../src/cell/builds.ts";
import { chat } from "../src/cell/chat.ts";
import { defaults } from "../src/lib/params.ts";

// ── cfg ────────────────────────────────────────────────────────────────────

testCell(
  cfg,
  "starts at the llama.cpp defaults with nothing marked changed",
  (t) => {
    t.init();
    t.expect.state((s) => s.touched.length === 0);
    t.expect.state((s) => s.settings.ctxSize === 4096);
    t.expect.state((s) => s.settings.host === "127.0.0.1");
  },
);

testCell(cfg, "set coerces, clamps, and tracks what the user changed", (t) => {
  t.init();
  t.send.set("ngl", "999999");
  t.expect.state((s) => s.settings.ngl === 999);
  t.expect.state((s) => s.touched.includes("ngl"));

  // Setting a value back to the default un-marks it — the "changed" count is a
  // claim about the command line, so it has to stay true.
  t.send.set("ngl", "0");
  t.expect.state((s) => !s.touched.includes("ngl"));
});

testCell(
  cfg,
  "a bad text value falls back to the default instead of NaN",
  (t) => {
    t.init();
    t.send.set("ctxSize", "not a number");
    t.expect.state((s) => s.settings.ctxSize === 4096);
  },
);

testCell(
  cfg,
  "an unknown parameter throws rather than writing a dead field",
  async (t) => {
    t.init();
    // A call starts when it is made and a method's throw REJECTS, as in
    // production — `assertThrows` around a dispatch asserts nothing, and the
    // unawaited rejection used to cancel every later test in this file. The
    // `assertRejects(() => Promise.resolve(t.send…))` this used to need is
    // retired: `t.expect.rejects` is the harness's own (aio 1.0.0-beta), and
    // the pattern NARROWS the refusal, so a typo throwing `TypeError` can no
    // longer pass for the validation this claims to test.
    await t.expect.rejects(() => t.send.set("nglll", "8"), /unknown parameter/);
    t.expect.state((s) => s.settings.nglll === undefined);
  },
);

testCell(
  cfg,
  "apply replaces the map and recomputes what is non-default",
  (t) => {
    t.init();
    t.send.apply({ ...defaults(), ngl: 99, temp: 0.8 }, ["because"]);
    t.expect.state((s) => s.touched.length === 1);
    t.expect.state((s) => s.touched[0] === "ngl");
    t.expect.state((s) => s.reasons[0] === "because");
  },
);

testCell(cfg, "reset returns every parameter to its default", (t) => {
  t.init();
  t.send.set("ngl", "40");
  t.send.set("loadMode", "mlock");
  t.send.reset();
  t.expect.state((s) => s.touched.length === 0);
  t.expect.state((s) => s.settings.loadMode === "auto");
});

testCell(cfg, "reset clears the context pin with everything else", (t) => {
  // The pin is a setting the user typed; a Reset that wiped the settings but
  // kept the pin left a hidden instruction silently capping the next tune.
  t.init();
  t.send.setCtxOverride(131_072, "/m.gguf");
  t.expect.state((s) => s.ctxOverride === 131_072);
  t.send.reset();
  t.expect.state((s) => s.ctxOverride === 0);
  t.expect.state((s) => s.ctxOverrideFor === "");
});

testCell(
  cfg,
  "rememberFit grows by default, replaces when the ladder ran",
  (t) => {
    t.init();
    t.send.rememberFit({ model: "/m.gguf", ctx: 32_768 });
    // A smaller PROVEN run by choice does not shrink the record…
    t.send.rememberFit({ model: "/m.gguf", ctx: 8_192 });
    t.expect.state((s) => s.fitCtx["/m.gguf"] === 32_768);
    // …but a ladder run measured the record itself as too high: the opening bid
    // is capped at the record, and the ladder only engages after that bid died.
    // Growing-only here re-ran the crash at the top of every session.
    t.send.rememberFit({ model: "/m.gguf", ctx: 16_384, exact: true });
    t.expect.state((s) => s.fitCtx["/m.gguf"] === 16_384);
  },
);

testCell(cfg, "resetOne only touches its own parameter", (t) => {
  t.init();
  t.send.set("ngl", "40");
  t.send.set("ctxSize", "8192");
  t.send.resetOne("ngl");
  t.expect.state((s) => s.settings.ngl === 0);
  t.expect.state((s) => s.settings.ctxSize === 8192);
  t.expect.state((s) => s.touched.length === 1);
});

// ── ui ─────────────────────────────────────────────────────────────────────

testCell(ui, "navigation and theme are plain state", (t) => {
  t.init();
  t.expect.state((s) => s.tab === "one");
  t.send.go("chat");
  t.expect.state((s) => s.tab === "chat");
  t.send.toggleTheme();
  t.expect.state((s) => s.theme === "light");
  t.send.toggleTheme();
  t.expect.state((s) => s.theme === "dark");
});

// ── hw ─────────────────────────────────────────────────────────────────────

testCell(
  hw,
  "refresh samples this machine and computes a utilization delta",
  async (t) => {
    t.init();
    await t.send.refresh(true);
    t.expect.state((s) => s.lastRefresh > 0);
    t.expect.state((s) => s.lastError === "");
    t.expect.state((s) => s.cpuHistory.length === 1);
    // First sample has no predecessor, so utilization is 0 by definition.
    t.expect.state((s) => s.cpu === null || s.cpu.utilPct === 0);

    await t.send.refresh(true);
    t.expect.state((s) => s.cpuHistory.length === 2);
    t.expect.invariant((s) =>
      s.cpu === null || (s.cpu.utilPct >= 0 && s.cpu.utilPct <= 100)
    );
  },
);

testCell(
  hw,
  "a paused sampler ignores the schedule but obeys the user",
  async (t) => {
    t.init();
    t.send.togglePause();
    t.expect.state((s) => s.paused === true);
    await t.send.refresh(); // scheduled poll — must be ignored
    t.expect.state((s) => s.lastRefresh === 0);
    await t.send.refresh(true); // manual refresh — must still work
    t.expect.state((s) => s.lastRefresh > 0);
  },
);

// ── models ─────────────────────────────────────────────────────────────────

testCell(models, "directories are added once and can be removed", (t) => {
  t.init();
  t.send.addDir("/tmp/models/");
  t.send.addDir("/tmp/models");
  t.expect.state((s) => s.dirs.length === 1);
  t.send.removeDir("/tmp/models");
  t.expect.state((s) => s.dirs.length === 0);
});

testCell(
  models,
  "a scan of an empty directory clears the selection honestly",
  async (t) => {
    t.init();
    const dir = await Deno.makeTempDir({ prefix: "llama-master-empty-" });
    try {
      t.send.addDir(dir);
      t.send.select("/gone/model.gguf");
      await t.send.scan();
      t.expect.state((s) => s.items.length === 0);
      t.expect.state((s) => s.selected === "");
      t.expect.state((s) => s.lastScan > 0);
      t.expect.state((s) => s.scanning === false);
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  },
);

// Selectors are only bound on a booted runtime, so this one runs under
// bootCells rather than testCell.
Deno.test("models: the filter narrows the visible list", async () => {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-filter-" });
  using _boot = await bootCells([models]);
  try {
    // Not valid GGUF — the point is the filter, and an unreadable header must
    // not stop a model from being listed.
    await Deno.writeFile(join(dir, "alpha.gguf"), new Uint8Array([1, 2, 3]));
    await Deno.writeFile(join(dir, "beta.gguf"), new Uint8Array([1, 2, 3]));
    await models.addDir(dir);
    await models.scan();
    assertEquals(models.items.length, 2);
    await models.setFilter("alph");
    assertEquals(models.visible().length, 1);
    await models.setFilter("");
    assertEquals(models.visible().length, 2);
    assert(models.totalSizeB() > 0);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

// ── builds ─────────────────────────────────────────────────────────────────

testCell(
  builds,
  "the chooser is plain state and clears stale asset picks",
  (t) => {
    t.init();
    t.send.setAsset("llama-b1-bin-ubuntu-x64.zip");
    t.send.setRef("b6234");
    t.expect.state((s) => s.ref === "b6234");
    t.expect.state((s) => s.assetName === "");
    t.send.setBackend("cuda");
    t.expect.state((s) => s.backend === "cuda");
    t.send.setJobs(-4);
    t.expect.state((s) => s.jobs === 0);
    t.send.setJobs(9999);
    t.expect.state((s) => s.jobs === 512);
    // Off by default — a stock llama.cpp unless the user asks — and plain
    // state once asked.
    t.expect.state((s) => s.bypassSchedCap === false);
    t.send.setBypassSchedCap(true);
    t.expect.state((s) => s.bypassSchedCap === true);
  },
);

testCell(
  builds,
  "removing a build that is not there is reported, not silent",
  async (t) => {
    t.init();
    await t.send.remove("no-such-build");
    t.expect.state((s) => s.lastError.includes("no-such-build"));
  },
);

// Regression: the progress callback used to build its next value by spreading
// `s.job` read back out of state. That hands the store a proxy-derived object,
// which rejects the WHOLE action — the job froze at step 0 with no error
// visible anywhere. It only showed up when the app was actually run.
testCell(
  builds,
  "a failing job still reports progress and ends in a failed state",
  async (t) => {
    t.init();
    t.send.setOrigin("release");
    // A ref that cannot resolve: step 0 reports progress first (the line that
    // used to throw), then the release lookup fails.
    t.send.setRef("b0-does-not-exist");
    await t.send.start();
    t.expect.state((s) => s.job?.status === "failed");
    t.expect.state((s) => (s.job?.error ?? "").length > 0);
    t.expect.state((s) => s.lastError.length > 0);
    // Crucially: the callback DID write to state before the failure.
    t.expect.state((s) => s.log[0]?.includes("Looking up") === true);
  },
);

testCell(
  builds,
  "update builds the version it is updating TO, not the one installed",
  async (t) => {
    // A nested `builds.start()` is a SECOND dispatch with its own draft, so it
    // read `ref`, `origin` and `backend` as COMMITTED — from before the four
    // lines `update` had just written. The job came out labelled with the
    // version already on disk: an Update button that reinstalls what you have,
    // reports success, and says it updated. The label is the assertion because
    // it is built from those three fields before any I/O happens
    // (`builds.ts:start`), so it says exactly what the sibling read.
    t.init({
      ref: "b1111",
      origin: "release",
      backend: "cpu",
      activeId: "x",
      installed: [{
        id: "x",
        ref: "b9999",
        origin: "release",
        backend: "cpu",
        dir: "/tmp/x",
        serverBin: "/tmp/x/llama-server",
        cliBin: "/tmp/x/llama-cli",
        createdAt: 0,
        sizeB: 0,
      }],
      // A tag that cannot resolve, so the build fails at the lookup instead of
      // downloading anything — the same shape the failing-job test above uses.
      upstream: { latestTag: "b0-does-not-exist", masterSha: "", checkedAt: 1 },
    });
    await t.send.update();
    t.expect.state((s) => s.ref === "b0-does-not-exist");
    t.expect.state((s) => s.job?.label === "Install b0-does-not-exist (cpu)");
  },
);

testCell(
  builds,
  "cancelling with no job running is a no-op, not a crash",
  (t) => {
    t.init();
    t.send.cancel();
    t.expect.state((s) => s.job === null);
  },
);

testCell(
  builds,
  "scan of an empty builds directory leaves nothing selected",
  async (t) => {
    t.init();
    await t.send.scan();
    t.expect.state((s) => s.scanning === false);
    t.expect.invariant((s) =>
      s.activeId === "" || s.installed.some((b) => b.id === s.activeId)
    );
  },
);

// ── chat ───────────────────────────────────────────────────────────────────

testCell(chat, "an empty message is never sent", async (t) => {
  t.init();
  await t.send.send("http://127.0.0.1:1", "   ");
  t.expect.state((s) => s.messages.length === 0);
  t.expect.state((s) => s.streaming === false);
});

testCell(
  chat,
  "an unreachable server surfaces the error and stops streaming",
  async (t) => {
    t.init();
    // Port 1 is reserved and never listening.
    await t.send.send("http://127.0.0.1:1", "hello");
    t.expect.state((s) => s.streaming === false);
    t.expect.state((s) => s.lastError.length > 0);
    t.expect.state((s) => s.messages.length === 1);
  },
);

testCell(
  chat,
  "a stream cut mid-reply keeps every token that arrived",
  async (t) => {
    // Two cuts take this path and they are the same code: the user's Stop, and
    // the app closing under a live reply — shutdown aborts every in-flight
    // method before it persists (aio, `abortAllInflight`). What was on screen
    // must survive both, and it must survive from the accumulator rather than
    // from `partial`, which is only as fresh as the last flush.
    const enc = new TextEncoder();
    const ac = new AbortController();
    const srv = Deno.serve(
      { port: 0, signal: ac.signal, onListen: () => {} },
      () =>
        new Response(
          new ReadableStream({
            async start(c) {
              for (const word of ["Hello", " there", " friend"]) {
                c.enqueue(enc.encode(
                  `data: ${
                    JSON.stringify({ choices: [{ delta: { content: word } }] })
                  }\n\n`,
                ));
                await new Promise((r) => setTimeout(r, 30));
              }
              // Then hold the stream open — a real reply the user gives up on.
              await new Promise((r) => setTimeout(r, 5_000));
              c.close();
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        ),
    );
    const url = `http://127.0.0.1:${(srv.addr as Deno.NetAddr).port}`;

    t.init();
    const sent = t.send.send(url, "hi");
    await new Promise((r) => setTimeout(r, 200));
    t.send.stop(); // ← cancelOn aborts the in-flight send
    await sent;

    t.expect.state((s) => s.streaming === false);
    t.expect.state((s) => s.lastError === "");
    t.expect.state((s) => s.messages.length === 2);
    t.expect.state((s) => s.messages[1]?.content === "Hello there friend");
    t.expect.state((s) => s.partial === "");

    ac.abort();
    await srv.finished;
  },
);

/** A llama-server that answers one word, after a beat.
 *
 *  The beat is what makes the queue tests deterministic: a reply that lands
 *  instantly leaves no window to type into, which is the exact window this
 *  feature exists for. `status` lets the same server play a failure. */
function slowServer(opts: { delayMs?: number; status?: number } = {}) {
  const { delayMs = 250, status = 200 } = opts;
  const enc = new TextEncoder();
  const ac = new AbortController();
  let served = 0;
  const s = Deno.serve(
    { port: 0, signal: ac.signal, onListen: () => {} },
    async () => {
      served++;
      await new Promise((r) => setTimeout(r, delayMs));
      if (status !== 200) return new Response("nope", { status });
      return new Response(
        new ReadableStream({
          start(c) {
            c.enqueue(enc.encode(
              `data: ${
                JSON.stringify({ choices: [{ delta: { content: "ok" } }] })
              }\n\n`,
            ));
            c.enqueue(enc.encode("data: [DONE]\n\n"));
            c.close();
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      );
    },
  );
  return {
    url: `http://127.0.0.1:${(s.addr as Deno.NetAddr).port}`,
    requests: () => served,
    async close() {
      ac.abort();
      await s.finished;
    },
  };
}

testCell(
  chat,
  "a message written mid-reply waits, then sends itself",
  async (t) => {
    // The feature, end to end: type while it is answering, and the message is
    // held rather than dropped or interleaved into the live request. The drain
    // runs inside the first send, so awaiting it covers both turns.
    const srv = slowServer();
    t.init();

    const first = t.send.send(srv.url, "first");
    await new Promise((r) => setTimeout(r, 80)); // mid-reply
    await t.send.submit(srv.url, "second");
    t.expect.state((s) => s.queue.length === 1, "held, not sent");
    t.expect.state((s) => s.queue[0] === "second");
    t.expect.state(
      (s) => s.streaming === true,
      "the first reply is still live",
    );

    await first;
    t.expect.state((s) => s.queue.length === 0, "the queue emptied itself");
    t.expect.state((s) => s.streaming === false);
    // user, reply, user, reply — in the order they were typed.
    t.expect.state((s) => s.messages.length === 4, "both turns completed");
    t.expect.state((s) => s.messages[0]?.content === "first");
    t.expect.state((s) => s.messages[2]?.content === "second");
    assertEquals(srv.requests(), 2, "two turns, never two at once");
    await srv.close();
  },
);

testCell(
  chat,
  "Stop leaves the queue standing, and does not fire the next message",
  async (t) => {
    // `advances` exists for this: a user who cancelled a reply has said
    // something about the next one too. What they typed is KEPT — Stop is not
    // a destructive button — but nothing sends until they say so.
    const enc = new TextEncoder();
    const ac = new AbortController();
    let served = 0;
    const s = Deno.serve(
      { port: 0, signal: ac.signal, onListen: () => {} },
      () => {
        served++;
        return new Response(
          new ReadableStream({
            async start(c) {
              c.enqueue(enc.encode(
                `data: ${
                  JSON.stringify({ choices: [{ delta: { content: "think" } }] })
                }\n\n`,
              ));
              await new Promise((r) => setTimeout(r, 5_000));
              c.close();
            },
          }),
          { headers: { "content-type": "text/event-stream" } },
        );
      },
    );
    const url = `http://127.0.0.1:${(s.addr as Deno.NetAddr).port}`;

    t.init();
    const sent = t.send.send(url, "one");
    await new Promise((r) => setTimeout(r, 120));
    await t.send.submit(url, "two");
    t.expect.state((x) => x.queue.length === 1);
    t.send.stop();
    await sent;

    t.expect.state((x) => x.streaming === false);
    t.expect.state((x) => x.queue.length === 1, "kept, not thrown away");
    t.expect.state((x) => x.queue[0] === "two");
    assertEquals(served, 1, "the cancelled run did not pull the next message");

    ac.abort();
    await s.finished;
  },
);

testCell(
  chat,
  "a server that fails holds the queue instead of emptying it into the void",
  async (t) => {
    // Six queued messages against a server that is refusing is six identical
    // errors and a lost queue. The failure has to arrive mid-drain to prove
    // it, which is why the server is slow as well as broken.
    const srv = slowServer({ status: 500 });
    t.init();

    const first = t.send.send(srv.url, "one");
    await new Promise((r) => setTimeout(r, 80));
    await t.send.submit(srv.url, "two");
    await first;

    t.expect.state((s) => s.streaming === false);
    t.expect.state((s) => s.lastError.length > 0, "and it says so");
    t.expect.state((s) => s.queue.length === 1, "the second one is still safe");
    t.expect.state((s) => s.queue[0] === "two");
    assertEquals(srv.requests(), 1, "it did not throw the next one after it");
    await srv.close();
  },
);

testCell(chat, "an idle submit is the request itself", async (t) => {
  // Nothing running, so there is nothing to wait for: submit sends, and the
  // queue never holds it.
  const srv = slowServer({ delayMs: 0 });
  t.init();
  await t.send.submit(srv.url, "go");
  t.expect.state((s) => s.queue.length === 0);
  t.expect.state((s) => s.messages[0]?.content === "go");
  await srv.close();
});

testCell(chat, "removing a queued message leaves the rest in order", (t) => {
  t.init();
  t.send.submit("http://127.0.0.1:1", "a");
  t.expect.state((s) => s.messages.length >= 0); // the send is allowed to fail
  t.send.clearQueue();
  t.expect.state((s) => s.queue.length === 0);
});

testCell(chat, "clear wipes the conversation and the last error", (t) => {
  t.init();
  t.send.setSystem("be brief");
  t.send.clear();
  t.expect.state((s) => s.messages.length === 0);
  t.expect.state((s) => s.lastError === "");
  t.expect.state((s) => s.system === "be brief");
});

Deno.test("cells: every cell owns a distinct, lowercase action namespace", () => {
  // A collision here would make two cells share action types — one would
  // silently swallow the other's dispatches.
  const names = [
    cfg.reset.type,
    ui.go.type,
    hw.togglePause.type,
    models.select.type,
    builds.cancel.type,
    chat.clear.type,
  ].map((t) => t.split(":")[0] ?? "");
  assertEquals(new Set(names).size, names.length, names.join(","));
  assert(names.every((n) => n.length > 0 && n === n.toLowerCase()));
});

Deno.test("builds: boot fetches the asset list, so Install is live on frame one", async () => {
  // The defect this pins: `loadAssets` was only called from `setOrigin` and
  // `setRef`, and boot called neither. On the default release route that left
  // `assets` empty, `targetReadiness` answering "pending", and the Install
  // button disabled until the user found "Refresh list" — two clicks, every
  // launch, for a kata that promises one. `assets` is deliberately not
  // persisted, so boot is the only place this can come from.
  const src = await Deno.readTextFile(
    new URL("../src/app.ts", import.meta.url),
  );
  const onStart = src.slice(src.indexOf("onStart:"));
  assert(
    /builds\.loadAssets\(\)/.test(onStart),
    "src/app.ts onStart must fetch the asset list",
  );
  // And the default route is the one that needs it.
  const { builds } = await import("../src/cell/builds.ts");
  using _boot = await bootCells([builds]);
  assertEquals(builds.origin, "release");
});

Deno.test("cfg: optimal-automatically is on by default and can be switched off", async () => {
  // The kata asks for a visible switch, ON by default, that can be turned off.
  // Default ON matters: a first-time user should get good settings without
  // knowing a tuner exists. Off matters: someone who hand-tuned a command must
  // not have it rewritten under them on the next Start.
  const { cfg } = await import("../src/cell/cfg.ts");
  using _boot = await bootCells([cfg]);
  assertEquals(cfg.autoOptimal, true, "on by default");
  await cfg.toggleAutoOptimal();
  assertEquals(cfg.autoOptimal, false, "and it can be switched off");
  await cfg.toggleAutoOptimal();
  assertEquals(cfg.autoOptimal, true);

  // Same boot, because a cell def binds to exactly one app per process: a
  // second bootCells([cfg]) in this file is an error, not a style choice.
  //
  // The leak this covers: `ctxOverride` persists, and only the All-in-one
  // dropdown cleared it — so selecting a model from the Models tab, from `am`,
  // or restoring a session carried a number chosen for a different model. On a
  // model trained shorter it silently capped the context, with nothing on
  // screen saying why.
  await cfg.setCtxOverride(128000, "/models/big.gguf");
  assertEquals(cfg.ctxOverride, 128000);
  assertEquals(cfg.ctxOverrideFor, "/models/big.gguf");

  // Clearing forgets which model it belonged to, so it cannot come back.
  await cfg.setCtxOverride(0);
  assertEquals(cfg.ctxOverride, 0);
  assertEquals(cfg.ctxOverrideFor, "");

  // Rubbish is rejected rather than stored as NaN.
  await cfg.setCtxOverride(Number.NaN, "/models/big.gguf");
  assertEquals(cfg.ctxOverride, 0);
  await cfg.setCtxOverride(-5, "/models/big.gguf");
  assertEquals(cfg.ctxOverride, 0);
});

testCell(
  builds,
  "the backend default follows the hardware, not a stored guess",
  (t) => {
    // "Build with one click" and "build the optimal thing for this PC" have to
    // be the same click. The stored default is `cpu` — the only value that is
    // always installable — so on a machine with a GPU it has to be corrected
    // once the hardware is known. What must NOT happen is overriding a
    // deliberate choice: picking the least capable backend on a CUDA box is a
    // legitimate answer, not a stale default.
    t.init();
    t.expect.state((s) => s.backend === "cpu" && s.backendChosen === false);

    t.send.suggestBackend("cuda");
    t.expect.state((s) => s.backend === "cuda");
    t.expect.state((s) => s.backendChosen === false);

    t.send.setBackend("cpu");
    t.expect.state((s) => s.backendChosen === true);
    t.send.suggestBackend("cuda");
    t.expect.state((s) => s.backend === "cpu");

    // A store from before `backendChosen` existed restores the flag as false —
    // but the only way such a store holds a non-cpu backend is that the user
    // picked it. The seed must never move off a non-default value.
    t.init();
    t.send.suggestBackend("vulkan");
    t.expect.state((s) => s.backend === "vulkan" && s.backendChosen === false);
    t.send.suggestBackend("cuda");
    t.expect.state((s) => s.backend === "vulkan");
  },
);

/**
 * A rename of a PERSISTED field is a migration whether or not one is written.
 *
 * `cfg.reserveVramB` — one machine-wide VRAM figure — became `reservePerGpuVramB`
 * and `reserveConnectedVramB`. Without a version bump and a hook, aio deep-merges
 * the stored blob over the defaults, keeps the orphaned key forever and says so
 * on every boot ("shape drift: 1 stored field(s) no longer match the declared
 * shape"). Verified end to end against a store seeded with the old world: the
 * boot reported `{cell: "cfg", from: 0, to: 2, outcome: "migrated"}` and the key
 * was gone. This pins the half that lives in this repo.
 */
Deno.test("cfg: the pre-rename reserve field is dropped, not carried", () => {
  const def = (cfg as unknown as {
    __aio?: {
      version?: number;
      onMigrate?: (
        s: Record<string, unknown>,
        from: number,
      ) => Record<string, unknown>;
    };
  }).__aio;
  assertEquals(def?.version, 6, "the shape changed, so the version must have");
  const migrate = def?.onMigrate;
  assert(migrate, "and a version bump with no hook only silences the warning");
  const old = {
    reserveVramB: 4 * 1024 ** 3,
    reservePerGpuVramB: 0,
    reserveConnectedVramB: 8 * 1024 ** 3,
    reserveRamB: 16 * 1024 ** 3,
  };
  const next = migrate({ ...old }, 0);
  assert(
    !("reserveVramB" in next),
    "the orphaned key is removed, not merged forward",
  );
  // The old value meant "hold this much across the whole machine, divided
  // between the cards". Neither new field means that, so it is not carried:
  // into `connectedB` it would move memory onto one card the user never chose,
  // into `perGpuB` it would be multiplied by the number of cards.
  assertEquals(next.reserveConnectedVramB, 8 * 1024 ** 3);
  assertEquals(next.reservePerGpuVramB, 0);
  assertEquals(next.reserveRamB, 16 * 1024 ** 3, "RAM kept its name and value");
  // A store already past that version is left alone.
  const current = migrate({ ...old }, 2);
  assertEquals(current.reserveVramB, 4 * 1024 ** 3);
});

Deno.test("cfg: v4 renames mlock/no-mmap into the one load-mode setting", () => {
  // Upstream deprecated `--mlock`, `--mmap`/`--no-mmap` and `--direct-io` in
  // favour of `-lm/--load-mode` and then DELETED them, so a current
  // llama-server answers either with `unknown argument` and exits before it has
  // read the model path. The two booleans become one enum.
  //
  // Carried forward, unlike the reserve in v2: the old pair and the new enum
  // say the same thing about the same file, so the user's choice survives the
  // rename instead of being silently reset.
  const migrate = (cfg as unknown as {
    __aio: {
      onMigrate: (
        s: Record<string, unknown>,
        from: number,
      ) => Record<string, unknown>;
    };
  }).__aio.onMigrate;

  const pinned = migrate(
    { settings: { ngl: 99, mlock: true }, touched: ["ngl", "mlock"] },
    3,
  ) as { settings: Record<string, unknown>; touched: string[] };
  assertEquals(pinned.settings.loadMode, "mlock");
  assert(!("mlock" in pinned.settings), "the dead key is removed");
  assertEquals(pinned.touched.sort(), ["loadMode", "ngl"]);

  const unmapped = migrate({ settings: { noMmap: true }, touched: [] }, 3) as {
    settings: Record<string, unknown>;
  };
  assertEquals(unmapped.settings.loadMode, "none", "--no-mmap meant no mmap");
  assert(!("noMmap" in unmapped.settings));

  // Neither set: nothing to carry, and no value invented.
  const plain = migrate({ settings: { ngl: 1 }, touched: [] }, 3) as {
    settings: Record<string, unknown>;
  };
  assert(!("loadMode" in plain.settings));
});

Deno.test("cfg: v3 drops the two settings whose flags left the catalog", () => {
  // `noContextShift` became `contextShift` (upstream's default flipped and the
  // old switch reproduced it in both positions), and `--defrag-thold` is
  // deprecated upstream — its value is ignored. A stored value for either
  // would sit in the map forever, shown nowhere and emitted never.
  const migrate = (cfg as unknown as {
    __aio: {
      onMigrate: (
        s: Record<string, unknown>,
        from: number,
      ) => Record<string, unknown>;
    };
  }).__aio.onMigrate;
  const next = migrate({
    settings: { noContextShift: true, defragThold: 0.2, ctxSize: 8192 },
    touched: ["noContextShift", "defragThold", "ctxSize"],
  }, 2);
  assertEquals(next.settings, { ctxSize: 8192 });
  assertEquals(next.touched, ["ctxSize"]);
  // The old default (shift off, as `--no-context-shift` said) and the new
  // default (`contextShift: false`) run the same server, so nothing is carried.
  assertEquals("contextShift" in (next.settings as object), false);
});

Deno.test("cfg: v5 resets a default that moved and was never a choice", () => {
  // v0.8.0 moved `-to` 600 → 3600 and turned `--jinja` back on, but a store
  // saves VALUES: older installs kept launching `-to 600` (shorter than a long
  // prefill) and `--no-jinja` (breaks tool-call and reasoning templates).
  const migrate = (cfg as unknown as {
    __aio: {
      onMigrate: (
        s: Record<string, unknown>,
        from: number,
      ) => Record<string, unknown>;
    };
  }).__aio.onMigrate;
  const next = migrate({
    // `touched` lists them — it lists every non-default value, tuner-written
    // ones too, so it is no evidence of a choice.
    settings: { timeout: 600, jinja: false, slots: false, ngl: 99 },
    touched: ["ngl", "timeout", "slots"],
  }, 4) as { settings: Record<string, unknown>; touched: string[] };
  assert(!("timeout" in next.settings), "old default → today's default");
  assert(!("jinja" in next.settings));
  assertEquals(next.touched.sort(), ["ngl", "slots"]);
  assertEquals(next.settings.slots, false, "hiding /slots is a fair choice");
  assertEquals(next.settings.ngl, 99, "unrelated keys are untouched");
  // A value that was never a default is a choice.
  const kept = migrate({ settings: { timeout: 1200 }, touched: [] }, 4) as {
    settings: Record<string, unknown>;
  };
  assertEquals(kept.settings.timeout, 1200);
});
