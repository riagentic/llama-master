// test/cells.test.ts — cell behaviour: guards, error surfacing, and the
// invariants that keep the UI honest.
//
// These run the real dispatch loop. Where a method does real I/O (a model scan,
// a hardware sample) the test does the real I/O too — on temp directories and
// on this machine — because a mocked filesystem would only prove the mock
// agrees with the code.

import { assert, assertEquals, assertThrows } from "@std/assert";
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
  (t) => {
    t.init();
    assertThrows(() => t.send.set("nglll", "8"), Error, "unknown parameter");
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
  t.send.set("mlock", true);
  t.send.reset();
  t.expect.state((s) => s.touched.length === 0);
  t.expect.state((s) => s.settings.mlock === false);
});

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
  t.send.setInput("   ");
  await t.send.send("http://127.0.0.1:1");
  t.expect.state((s) => s.messages.length === 0);
  t.expect.state((s) => s.streaming === false);
});

testCell(
  chat,
  "an unreachable server surfaces the error and stops streaming",
  async (t) => {
    t.init();
    t.send.setInput("hello");
    // Port 1 is reserved and never listening.
    await t.send.send("http://127.0.0.1:1");
    t.expect.state((s) => s.streaming === false);
    t.expect.state((s) => s.lastError.length > 0);
    t.expect.state((s) => s.messages.length === 1);
  },
);

testCell(chat, "clear wipes the conversation and the last error", (t) => {
  t.init();
  t.send.setInput("x");
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
  },
);
