// test/ui.test.ts — the app as a user meets it.
//
// Driven through aio's semantic surface: no selectors, no sleeps. Actions queue
// and are awaited only where something is observed. Names are LABEL+ROLE from
// the TSX, or the verbatim `t` prop where the visible copy might change.
//
// These cover the journeys the kata promises: see the machine, get llama.cpp,
// find models, tune with a live memory picture, run the server, chat. Where a
// step needs real data (a GGUF header) the test writes a real file rather than
// faking cell state — a UI test that asserts on injected state proves nothing
// about the code that produces it.

import {
  assert,
  assertEquals,
  assertExists,
  assertStringIncludes,
} from "@std/assert";
import { testUI } from "aio/testing";
import { join } from "@std/path";

// One throwaway app home for the whole file. It has to be set before anything
// calls `paths()`, and it must be an env var rather than an argument because the
// static imports below are hoisted above every statement in this file — which is
// why the fixture builds these tests install land in a temp dir and not in the
// user's real ~/.llama-master.
const HOME = await Deno.makeTempDir({ prefix: "llama-master-ui-" });
Deno.env.set("LLAMA_MASTER_HOME", HOME);

import App from "../src/App.tsx";
import { OnePage } from "../src/ui/OnePage.tsx";
import { TunePanel } from "../src/ui/TunePanel.tsx";
import { About } from "../src/ui/About.tsx";
import { ServerPanel } from "../src/ui/ServerPanel.tsx";
import {
  CTX_BANDS,
  CTX_PRESETS,
  ctxBands,
  ctxLabel,
  optimalCtx,
} from "../src/lib/tune.ts";
import { tuneAll } from "../src/lib/tune.ts";
import { builds } from "../src/cell/builds.ts";
import { cfg } from "../src/cell/cfg.ts";
import { models } from "../src/cell/models.ts";
import { prereq } from "../src/cell/prereq.ts";
import { srv } from "../src/cell/srv.ts";
import { ui } from "../src/cell/ui.ts";
import { moeGguf } from "./gguf-fixture.ts";
import { meta } from "./fixtures.ts";
import { hw } from "../src/cell/hw.ts";
import { devices, isEnabled } from "../src/lib/gpu.ts";
import { plan as computePlan, withoutOurUsage } from "../src/lib/plan.ts";
import {
  activeBuild,
  currentStatePlan,
  headroomNow,
  hwSnapshot,
  paramBlocker,
  planningHw,
  projectedStatePlan,
} from "../src/ui/derive.ts";
import { chat } from "../src/cell/chat.ts";
import {
  betterPlacement,
  placements,
  startBlocker,
  startServer,
  stopServer,
} from "../src/ui/actions.ts";
import type { Build } from "../src/lib/types.ts";

/** A temp directory holding one real GGUF, cleaned up by the caller. */
async function withModel(fn: (dir: string) => Promise<void>): Promise<void> {
  const dir = await Deno.makeTempDir({ prefix: "llama-master-ui-" });
  try {
    await Deno.writeFile(join(dir, "fixture-8x2B-Q4_K_M.gguf"), moeGguf());
    await fn(dir);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
}

/** A quiet workstation with two idle 24 GB cards and plenty of RAM.
 *
 *  Seeded into the `hw` cell so a test asserts the branch it is about rather than
 *  whatever the developer's GPU happens to be doing that second — the machine is
 *  an input to almost every decision this app makes, and leaving it live made the
 *  interesting cases untestable. */
function roomyMachine() {
  const GB = 1024 ** 3;
  const card = {
    vendor: "nvidia",
    name: "Test RTX 24G",
    tempC: 45,
    utilPct: 3,
    vramTotalB: 24 * GB,
    vramUsedB: 1 * GB,
    powerW: 40,
    powerLimitW: 300,
    computeCap: 8.9,
  };
  return {
    cpu: {
      model: "Test CPU",
      cores: 16,
      threads: 32,
      mhz: 4200,
      tempC: 50,
      utilPct: 5,
      coresUtil: [],
      stat: "",
      coreStats: [],
    },
    mem: {
      totalB: 128 * GB,
      availableB: 110 * GB,
      usedB: 18 * GB,
      swapTotalB: 0,
      swapUsedB: 0,
    },
    // Card 0 draws the desktop, card 1 is headless — the machine the
    // connected-GPU reserve is for, and the one that shows the difference
    // between "on every GPU" and "on the one with a screen".
    gpus: [{ ...card, display: true }, { ...card, display: false }],
    // A machine with a LAN address and the usual noise around it: the
    // "Available on LAN" line has to pick the one another machine can dial.
    lanIps: ["127.0.0.1", "192.168.1.24"],
    os: "linux",
    arch: "x86_64",
    lastRefresh: 1,
  };
}

// ── shell ──────────────────────────────────────────────────────────────────

testUI(
  App,
  "boots with the brand and every tab",
  async (ui_) => {
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "llama");
    for (
      const label of ["Machine", "Build", "Models", "Tune", "Server", "Chat"]
    ) {
      assertStringIncludes(html, label);
    }
  },
);

testUI(App, "the rail navigates between every panel", async (ui_) => {
  await ui_.settle();
  const expectations: [string, string][] = [
    ["tab-one", "Run a model"],
    ["tab-build", "Get llama.cpp"],
    ["tab-models", "Detected models"],
    ["tab-settings", "Memory plan"],
    ["tab-server", "llama-server"],
    ["tab-chat", "Test chat"],
    ["tab-dashboard", "Prerequisites"],
    ["tab-about", "Llama.cpp Master"],
    ["tab-one", "All-in-one"],
  ];
  for (const [handle, marker] of expectations) {
    ui_.App[handle].click();
    await ui_.settle();
    assertStringIncludes(ui_.html(), marker, `${handle} should show ${marker}`);
  }
});

testUI(App, "the theme toggle flips the whole shell", async (ui_) => {
  await ui_.settle();
  assertEquals(ui.theme, "dark");
  ui_.App.theme.click();
  await ui_.expectCell(ui, (s) => s.theme === "light");
  assertStringIncludes(ui_.html(), 'data-theme="light"');
});

// ── the command preview ────────────────────────────────────────────────────

testUI(
  OnePage as never,
  "the command section shows exactly what a setting change produces",
  async (ui_) => {
    // Mounted directly rather than through App: `ui.tab` is persisted and
    // rehydrates asynchronously, so "the default tab happens to be the one with
    // the command on it" is a race, not a fact. The command lives on the pages
    // where the settings are edited now (all-in-one, Tune, Server) instead of
    // in a strip pinned under every tab.
    await ui_.settle();
    if (!ui.showCommand) ui.toggleCommand();
    await ui_.settle();
    assert(!ui_.html().includes("-ngl"), "a default config emits no -ngl");

    await cfg.set("ngl", "99");
    await cfg.set("ctxSize", "16384");
    await ui_.settle();

    const html = ui_.html();
    assertStringIncludes(html, "llama-server", "the command that Start issues");
    assert(
      !html.includes("llama-cli"),
      "the cli equivalent is a reference, and it is on the pages with room for one",
    );
    assertStringIncludes(html, "-ngl 99");
    assertStringIncludes(html, "-c 16384");
    // Server-only flags must not appear in the cli line, and vice versa.
    await cfg.set("port", "9099");
    await ui_.settle();
    assertStringIncludes(ui_.html(), "--port 9099");

    // And it folds away — on a column that is a screenful of flags, and the
    // preference is persisted (`ui.showCommand`), so it must actually bind.
    ui_.find("OnePage")["one-cmd-toggle"].click();
    await ui_.expectCell(ui, (s) => s.showCommand === false);
    await ui_.settle();
    assert(
      !ui_.html().includes("--port 9099"),
      "hidden means hidden, not merely collapsed to a scroll",
    );
    ui_.find("OnePage")["one-cmd-toggle"].click();
    await ui_.expectCell(ui, (s) => s.showCommand === true);
  },
);

// ── tune ───────────────────────────────────────────────────────────────────

testUI(
  App,
  "every parameter is rendered with its flag and an explanation",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-settings"].click();
    await ui_.settle();
    const html = ui_.html();
    // A sample across all five groups — label, flag, and tooltip text.
    for (
      const marker of [
        "GPU layers",
        "-ngl",
        "Context size",
        "Flash attention",
        "Temperature",
        "Port",
      ]
    ) {
      assertStringIncludes(html, marker);
    }
    assertStringIncludes(html, "Transformer layers to run on the GPU");
    // Advanced flags stay hidden until asked for.
    assert(!html.includes("Tensor override"), "advanced flags start hidden");
  },
);

testUI(
  App,
  "the advanced toggle reveals the rarely-needed flags",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-settings"].click();
    await ui_.settle();
    await cfg.toggleAdvanced();
    await ui_.settle();
    assertStringIncludes(ui_.html(), "Tensor override");
    assertStringIncludes(ui_.html(), "-ot");
  },
);

testUI(
  App,
  "optimal settings is blocked until a model is readable",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-settings"].click();
    await ui_.settle();
    assertEquals(ui_.App.optimal.disabled, true);
    assertStringIncludes(ui_.html(), "Nothing to plan");
  },
);

// ── the whole journey ──────────────────────────────────────────────────────

testUI(
  App,
  "detect a model, tune it, and watch the memory plan appear",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    // A CUDA build has to be selected before the tuner will force flash
    // attention on: which flags are even loadable depends on the backend, and
    // with none chosen the honest answer is `auto`.
    await installStubBuild("stub-cuda", { backend: "cuda" });
    await builds.scan();
    await builds.setActive("stub-cuda");
    await withModel(async (dir) => {
      ui_.App["tab-models"].click();
      await ui_.settle();
      assertStringIncludes(ui_.html(), "Nothing scanned yet");

      // Point the scan at the fixture directory only — the real default
      // locations would make this test depend on the developer's disk.
      await models.addDir(dir);
      await models.scan();
      await ui_.settle();

      assertEquals(models.items.length, 1);
      const found = models.items[0]!;
      assertEquals(found.meta?.arch, "qwen3moe");
      assertEquals(models.selected, found.path, "the first model is selected");

      const modelsHtml = ui_.html();
      assertStringIncludes(modelsHtml, "fixture-8x2B-Q4_K_M.gguf");
      assertStringIncludes(modelsHtml, "Q4_K_M");
      assertStringIncludes(modelsHtml, "qwen3moe");

      // The plan is live on the Tune tab.
      ui_.App["tab-settings"].click();
      await ui_.settle();
      const tune = ui_.html();
      assertStringIncludes(tune, "VRAM");
      assertStringIncludes(tune, "RAM");
      assertStringIncludes(tune, "KV cache");
      assert(!tune.includes("Nothing to plan"));

      // "Optimal settings" is now available, and it explains itself.
      assertEquals(ui_.App.optimal.disabled, false);
      ui_.App.optimal.click();
      await ui_.expectCell(cfg, (s) => s.reasons.length > 0);
      assertStringIncludes(ui_.html(), "Flash attention on");
      assertEquals(cfg.settings.flashAttn, "on");

      // …and the change is visible in the command the app would run.
      assertStringIncludes(ui_.html(), "-fa on");
    });
    await removeStubBuild("stub-cuda");
  },
);

testUI(
  App,
  "the memory map draws both pools with a legend and hover detail",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      ui_.App["tab-settings"].click();
      await ui_.settle();
      const html = ui_.html();
      // The kata's interactive memory map: every byte on the machine, both
      // pools coloured apart, a legend, and the configuration laid over it.
      assertStringIncludes(html, "Memory map");
      assertStringIncludes(html, "region-vram");
      assertStringIncludes(html, "region-ram");
      assertStringIncludes(html, "map-fill");
      assertStringIncludes(html, "total on this machine");
      for (const area of ["Weights", "KV cache", "Compute (est.)"]) {
        assertStringIncludes(html, area);
      }
      assertStringIncludes(html, "Hover a band");
    });
  },
);

testUI(
  App,
  "the three placements are offered, the fastest first",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-settings"].click();
    await ui_.settle();
    const html = ui_.html();
    // One set of optimal settings; the choice is WHERE the model runs.
    for (const label of ["VRAM only", "Hybrid", "CPU only"]) {
      assertStringIncludes(html, label);
    }
    assert(
      !html.includes("Power saver") && !html.includes("Balanced"),
      "the old quality modes are gone",
    );
    assertEquals(cfg.placement, "vram", "the fastest placement is the default");
  },
);

testUI(
  App,
  "a hand-edited unsafe setting warns before anything is started",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      ui_.App["tab-settings"].click();
      await ui_.settle();

      // Ask for a context far past what this fixture was trained for, and for
      // more threads than the machine has logical CPUs.
      await cfg.set("ctxSize", "999999");
      await cfg.set("threads", "512");
      await ui_.settle();

      const html = ui_.html();
      assertStringIncludes(html, "will probably fail");
      assertStringIncludes(html, "threads");
      // The controls still work — a warning, not a cage.
      assertEquals(cfg.settings.threads, 512);
    });
  },
);

// ── server ─────────────────────────────────────────────────────────────────

testUI(
  App,
  "the server panel refuses to start and says exactly why",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-server"].click();
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "No llama.cpp build installed");
    assertEquals(ui_.App["start-server"].disabled, true);
    assertEquals(srv.status, "stopped");
    assertStringIncludes(html, "No output yet");
  },
);

testUI(
  App,
  "the chat tab tells the user the server is not up yet",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-chat"].click();
    await ui_.settle();
    assertStringIncludes(ui_.html(), "Start the server first");
  },
);

// ── build ──────────────────────────────────────────────────────────────────

testUI(
  App,
  "the build tab offers both routes and names the missing tool",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-build"].click();
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "Prebuilt release");
    assertStringIncludes(html, "Build from source");
    for (const backend of ["CPU", "CUDA", "Vulkan", "ROCm", "Metal"]) {
      assertStringIncludes(html, backend);
    }
    assertStringIncludes(html, "No llama.cpp yet");
    assertEquals(builds.installed.length, 0);
  },
);

testUI(
  App,
  "choosing the source route warns when the toolchain is incomplete",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-build"].click();
    await ui_.settle();
    await prereq.scan();
    await builds.setOrigin("source");
    await ui_.settle();

    // Hermetic: this runs against a throwaway app home, so a CMake the app
    // downloaded into the developer's real home does not decide the outcome.
    const found = (id: string) =>
      prereq.items.find((i) => i.id === id)?.found === true;
    const html = ui_.html();
    if (found("cmake") && found("compiler")) {
      assertEquals(ui_.App["get-llama"].disabled, false);
    } else {
      // The promise is "refuse up front and NAME the tool", not "refuse".
      assertEquals(ui_.App["get-llama"].disabled, true);
      const named = ["CMake", "cmake", "compiler", "C++"].some((w) =>
        html.includes(w)
      );
      assert(named, "the refusal must name the missing tool");
      // And it must offer the route that needs no toolchain at all.
      assertStringIncludes(html, "Prebuilt release");
    }
  },
);

// ── dashboard ──────────────────────────────────────────────────────────────

testUI(
  App,
  "the machine tab reports this machine, and Prerequisites the software",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await prereq.scan();
    // Machine is the hardware summary; the named tools live on their own page,
    // which is what the kata asks for. Both are checked, on the page that owns
    // them.
    ui_.App["tab-dashboard"].click();
    await ui_.expectCell(ui, (s) => s.tab === "dashboard");
    for (const marker of ["CPU", "GPU", "Memory", "Storage", "Software"]) {
      assertStringIncludes(ui_.html(), marker, `Machine should show ${marker}`);
    }

    ui_.App["tab-prereq"].click();
    await ui_.expectCell(ui, (s) => s.tab === "prereq");
    const html = ui_.html();
    for (
      const marker of [
        "Prerequisites",
        "Deno",
        "CMake",
        "C++ compiler",
      ]
    ) {
      assertStringIncludes(html, marker);
    }
    // Deno is by definition present — the app is running on it.
    assertEquals(prereq.byId("deno")?.found, true);
    // The compiler is the one thing that cannot be downloaded, and the panel
    // must say so rather than offering a button that cannot work.
    assertStringIncludes(html, "need a decision the app should not make");
  },
);

// ── the all-in-one page ────────────────────────────────────────────────────

testUI(App, "the app opens on the all-in-one page", async (ui_) => {
  await ui_.settle();
  assertEquals(ui.tab, "one");
  const html = ui_.html();
  // Everything the kata asks to be on one page.
  for (
    const marker of [
      "CPU",
      "GPU",
      "VRAM",
      "RAM",
      "Run a model",
      "Detect",
      "Optimal",
      "Start server",
    ]
  ) {
    assertStringIncludes(html, marker, `one page should show ${marker}`);
  }
  // The chat, with a place for tokens/second to appear.
  assertStringIncludes(html, "Start the server to chat");
});

testUI(
  App,
  "the all-in-one page picks a model and shows how it fits",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await withModel(async (dir) => {
      ui_.App["tab-one"].click();
      await models.addDir(dir);
      await models.scan();
      await ui_.settle();
      const html = ui_.html();
      assertStringIncludes(html, "fixture-8x2B-Q4_K_M.gguf");
      assertStringIncludes(html, "region-vram");
      assertStringIncludes(html, "region-ram");
      // Start is still blocked (no build installed) and says so on the page.
      assertEquals(ui_.App["one-start"].disabled, true);
      assertStringIncludes(html, "No llama.cpp build installed");
    });
  },
);

// ── readability ────────────────────────────────────────────────────────────

testUI(
  App,
  "text size is a control, and it starts comfortably",
  async (ui_) => {
    await ui_.settle();
    assertEquals(ui.fontPx, 14, "a comfortable default, not a dense one");
    assertStringIncludes(ui_.html(), "--fs: 14px");

    ui_.App["zoom-in"].click();
    ui_.App["zoom-in"].click();
    await ui_.expectCell(ui, (s) => s.fontPx === 16);
    assertStringIncludes(ui_.html(), "--fs: 16px");

    // Clamped at both ends so the layout cannot be broken from the toolbar.
    for (let i = 0; i < 20; i++) ui_.App["zoom-out"].click();
    await ui_.expectCell(ui, (s) => s.fontPx === 12);
    for (let i = 0; i < 20; i++) ui_.App["zoom-in"].click();
    await ui_.expectCell(ui, (s) => s.fontPx === 20);
  },
);

// ── update ─────────────────────────────────────────────────────────────────

testUI(
  App,
  "no Update button appears until upstream is known to be ahead",
  async (ui_) => {
    await ui_.settle();
    assert(
      !ui_.html().includes("Update →"),
      "nothing installed, nothing to update",
    );
    assertEquals(builds.updateInfo().available, false);
  },
);

// ── prerequisites: Fix and Fix all ─────────────────────────────────────────

testUI(
  App,
  "every unmet prerequisite offers a Fix, and there is a Fix all",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await prereq.scan();
    // Prerequisites is its own page now — a task with an action on most rows,
    // which is why the kata asks for it as a page rather than a dashboard card.
    ui_.App["tab-prereq"].click();
    await ui_.expectCell(ui, (s) => s.tab === "prereq");

    const missing = prereq.items.filter((i) => !i.found);
    const html = ui_.html();

    for (const p of missing) {
      const plan = prereq.plans[p.id];
      assert(plan, `no fix plan computed for ${p.id}`);
      if (plan.kind === "manual") {
        // Honest about what it will not do, rather than a button that lies.
        assertStringIncludes(html, "manual");
      } else {
        assertExists(
          ui_.App[`fix-${p.id}`],
          `${p.id} is missing and fixable but has no Fix button`,
        );
      }
    }

    const fixable = missing.filter((p) =>
      prereq.plans[p.id]?.kind !== "manual"
    );
    if (fixable.length > 0) {
      assertExists(ui_.App["fix-all"], "Fix all must be offered");
      assertStringIncludes(html, `Fix all (${fixable.length})`);
    }
  },
);

testUI(
  App,
  "a Fix button shows the exact command before it runs",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-dashboard"].click();
    await ui_.settle();
    await prereq.scan();
    await ui_.settle();

    // The invariant, on any machine: every MISSING prerequisite explains
    // itself on screen — a package plan shows the exact command it will run,
    // and a manual one shows why the app will not do it for you. Nothing
    // privileged is ever hidden behind a bare "Fix".
    const html = ui_.html();
    const missing = prereq.items.filter((i) => !i.found);
    for (const p of missing) {
      const plan = prereq.plans[p.id];
      assert(plan, `no fix plan computed for ${p.id}`);
      if (plan.kind === "package") {
        assertStringIncludes(html, plan.command.join(" "));
      } else if (plan.kind === "manual") {
        assertStringIncludes(html, plan.reason.slice(0, 40));
      }
    }
    assertEquals(
      Object.keys(prereq.plans).length,
      prereq.items.length,
      "every prerequisite gets a plan, present or not",
    );
  },
);

// ── build readiness ────────────────────────────────────────────────────────

testUI(
  App,
  "the Build tab says whether the SELECTED target will work",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-build"].click();
    await ui_.settle();
    await prereq.scan();
    await builds.setOrigin("release");
    await builds.setBackend("cuda");
    await builds.loadAssets();
    await ui_.settle();

    const html = ui_.html();
    if (builds.assets.length > 0) {
      // The user's report: prerequisites all green, then the build failed with a
      // list of filenames. It must be refused HERE, with a reason and a button.
      assertStringIncludes(html, "Windows only");
      assert(!html.includes(".tar.gz,"), "no filename dumps in the banner");
      assertEquals(ui_.App["get-llama"].disabled, true);
      assertExists(
        ui_.App["not-ready"],
        "the guidance block must be on screen",
      );
      // And the way out is a button, not a sentence.
      assertStringIncludes(html, "Build from source");
    }
  },
);

testUI(App, "a workable selection says so, plainly", async (ui_) => {
  ui_.App["tab-build"].click();
  await ui_.settle();
  await prereq.scan();
  await builds.setOrigin("release");
  await builds.setBackend("cpu");
  await builds.loadAssets();
  await ui_.settle();
  if (builds.assets.length > 0) {
    assertStringIncludes(ui_.html(), "Ready");
    assertEquals(ui_.App["get-llama"].disabled, false);
  }
});

testUI(
  App,
  "a source build is refused when the backend's toolchain is absent",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-build"].click();
    await ui_.settle();
    await prereq.scan();
    await builds.setOrigin("source");
    await builds.setBackend("cuda");
    await ui_.settle();

    // A CUDA source build needs the base toolchain AND nvcc. The point of the
    // test is that a missing one is caught HERE, not four minutes into cmake
    // configure — and that the message names which one.
    const found = (id: string) =>
      prereq.items.find((i) => i.id === id)?.found === true;
    const complete = found("cmake") && found("compiler") && found("cuda");
    if (complete) {
      assertEquals(ui_.App["get-llama"].disabled, false);
    } else {
      assertEquals(ui_.App["get-llama"].disabled, true);
      const html = ui_.html();
      const named = ["nvcc", "CUDA", "CMake", "cmake", "compiler"].some((w) =>
        html.includes(w)
      );
      assert(named, `the refusal must name the missing tool; got: ${html}`);
    }
  },
);

testUI(
  App,
  "an installed build is selected by clicking its row, not just the radio",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    ui_.App["tab-build"].click();
    await ui_.settle();
    await builds.scan();
    await ui_.settle();

    if (builds.installed.length < 2) return; // nothing to switch between here
    const [a, b] = builds.installed;
    await builds.setActive(a!.id);
    await ui_.settle();

    // The whole row is the target — a 6 px radio should not be the only way in.
    ui_.App[`build-row-${b!.id}`].click();
    await ui_.expectCell(builds, (s2) => s2.activeId === b!.id);
    assertStringIncludes(
      ui_.html(),
      "row-pick",
      "rows are styled as clickable",
    );
  },
);

testUI(
  OnePage as never,
  "the all-in-one page picks the build as well as the model",
  async (ui_) => {
    // Mount the panel, not `App` plus a rail click: `ui.tab` is persisted and its
    // rehydration on mount is async, so a click issued before it lands is
    // silently reverted to whatever the previous test left in the store. This
    // shipped as a ~40% flake and the rule is in CLAUDE.md — it failed here by
    // rendering the Build tab while asserting on the all-in-one page.
    await ui_.settle();
    await builds.scan();
    await ui_.settle();

    const html = ui_.html();
    assertStringIncludes(
      html,
      "one-build",
      "a build picker sits beside the model",
    );

    if (builds.installed.length === 0) {
      // Nothing installed: say so and offer the way out, rather than an empty box.
      assertStringIncludes(html, "No llama.cpp yet");
      assertExists(ui_.find("OnePage")["one-getllama"]);
      return;
    }
    // Every installed build is offered, named the way the Build tab names it.
    for (const b of builds.installed) {
      assertStringIncludes(html, `${b.ref} · ${b.backend}`);
    }
    if (builds.installed.length >= 2) {
      const other = builds.installed.find((b) => b.id !== builds.activeId)!;
      await builds.setActive(other.id);
      await ui_.settle();
      assertEquals(builds.activeId, other.id);
    }
  },
);

testUI(
  // The panel, not App-plus-navigation: `ui.tab` is persisted and rehydrates
  // asynchronously, so a rail click issued before that lands is silently
  // reverted (~40% flake). The rail has its own test.
  About as never,
  "the About page credits the project and its upstream",
  async (ui_) => {
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "Llama.cpp Master");
    assertStringIncludes(html, "User friendly application to master Llama.cpp");
    assertStringIncludes(html, "riagentic");
    assertStringIncludes(html, "MIT");
    assertStringIncludes(html, "github.com/riagentic/llama-master");
    // llama.cpp does the actual work; saying so is the honest thing.
    assertStringIncludes(html, "ggml-org/llama.cpp");
    // And the environment block a bug report would ask for.
    assertStringIncludes(html, "This machine");
  },
);

/** Put the stub where a real build would be — `srv.start` refuses any binary
 *  outside `paths().builds`, and `findOrphans` only looks there. */
/**
 * Install a runnable llama-server where a real build would be.
 *
 * The stub speaks the endpoints the app uses, so a test that "starts the
 * server" starts a real process on a real socket. `srv.start` refuses any
 * binary outside `paths().builds` and `findOrphans` only looks there, so the
 * location is not incidental. The metadata file is what makes the directory a
 * build: `listBuilds` reads the tree, not an index.
 */
async function installStubBuild(
  id = "stub",
  meta: Partial<Build> = {},
): Promise<string> {
  const { paths } = await import("../src/cell/host.server.ts");
  const dir = join(paths().builds, id);
  await Deno.mkdir(dir, { recursive: true });
  const bin = join(dir, "llama-server");
  const cli = join(dir, "llama-cli");
  await Deno.copyFile(new URL("./stub-llama-server.ts", import.meta.url), bin);
  await Deno.chmod(bin, 0o755);
  await Deno.copyFile(bin, cli);
  await Deno.chmod(cli, 0o755);
  await Deno.writeTextFile(
    join(dir, "llama-master.json"),
    JSON.stringify({
      id,
      ref: "master",
      origin: "source",
      backend: "cpu",
      dir,
      serverBin: bin,
      cliBin: cli,
      createdAt: 1,
      sizeB: 0,
      ...meta,
    }),
  );
  return bin;
}

async function removeStubBuild(id = "stub"): Promise<void> {
  const { paths } = await import("../src/cell/host.server.ts");
  await Deno.remove(join(paths().builds, id), { recursive: true }).catch(
    () => {},
  );
}

async function withStubBuild(
  fn: (bin: string) => Promise<void>,
): Promise<void> {
  const bin = await installStubBuild();
  // Register it, don't just write the files. Without this the panel under test
  // truthfully reports "No llama.cpp build installed" while the test drives a
  // server from that very build — an incoherent premise, and one that made the
  // rendered assertions depend on whether some earlier test had happened to scan.
  await builds.scan();
  await builds.setActive("stub");
  try {
    await fn(bin);
  } finally {
    // Remove the files but do NOT rescan: `builds.installed` is shared, and
    // emptying it here left whichever test ran next reporting "No llama.cpp
    // build installed" depending purely on order.
    await removeStubBuild();
  }
}

/** Drive the 1 s poll by hand until the server reports ready. `waitFor` takes
 *  a synchronous predicate, and liveness only advances when `poll` runs. */
async function untilReady(ui_: { settle: () => Promise<void> }): Promise<void> {
  for (let i = 0; i < 200; i++) {
    await srv.poll();
    // Both, not just the status: `poll` can see the process alive and /health
    // not answering yet, which leaves `ready` with `healthy` still false for a
    // tick. Waiting on the status alone made the health assertion below a coin
    // toss (~1 run in 5).
    if (srv.status === "ready" && srv.healthy) {
      // Re-check AFTER settling rather than assuming: a state patch computed
      // before the flush can land during it, so the condition that was true
      // when we tested it is not necessarily true when the caller reads it.
      // Asserting on a condition the wait no longer guarantees is how this
      // became a 1-in-5 flake twice over.
      await ui_.settle();
      if (srv.status === "ready" && srv.healthy) return;
    }
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(
    `server never became ready and healthy; log:\n${srv.log.join("\n")}`,
  );
}

/** A free port. Never a constant: two tests hardcoding one flake. */
/** Ports already handed out in THIS process, so two tests never share one. */
const handedOut = new Set<number>();

/**
 * A port nothing is listening on.
 *
 * Asking the OS for port 0 and closing gives a port that was free a moment ago —
 * check-then-close-then-use, so the same ephemeral port can come back twice while
 * the first holder is still binding it. Eight tests do this in one process, and a
 * collision shows up as a server that "did not start" for no visible reason.
 * Remembering what we handed out removes the intra-process half of that race,
 * which is the half we can actually control.
 */
function freePort(): number {
  for (let i = 0; i < 64; i++) {
    const l = Deno.listen({ port: 0 });
    const { port } = l.addr as Deno.NetAddr;
    l.close();
    if (!handedOut.has(port)) {
      handedOut.add(port);
      return port;
    }
  }
  throw new Error("could not find an unused port after 64 tries");
}

/** Both places a server failure can be shown. A diagnosis that says "the log
 *  below" has to be complete on whichever one the user is looking at. */
const SERVER_SURFACES: [string, () => unknown][] = [
  ["ServerPanel", ServerPanel],
  ["OnePage", OnePage],
];

for (const [name, Surface] of SERVER_SURFACES) {
  testUI(
    Surface as never,
    `${name}: a crash is explained, with the log the guidance promises`,
    async (ui_) => {
      // The real report: the server dies because something else holds the
      // VRAM. The app must say so, and show the output it points at.
      await withStubBuild(async (bin) => {
        // Known slate first. `srv.start` returns immediately when the status is
        // already "starting" or "ready" — one model runs at a time — so a
        // server left up by an earlier test means nothing is spawned here, no
        // crash happens, and this fails five seconds later on a missing
        // diagnosis. The cells are singletons across every test in the process.
        await srv.stop();
        await srv.poll();
        await srv.start([bin, "--oom"], "http://127.0.0.1:1");
        // Wait for what is actually asserted, not a proxy for it: the status
        // turns "crashed" the moment the process is gone, while the diagnosis
        // is built from its captured output — so gating on the status alone
        // left the rendering assertions racing the log.
        for (let i = 0; i < 200; i++) {
          await srv.poll();
          if (srv.status === "crashed" && srv.diagnosis) break;
          await new Promise((r) => setTimeout(r, 25));
        }
        assertEquals(srv.status, "crashed");
        assertExists(
          srv.diagnosis,
          `no diagnosis; log:\n${srv.log.join("\n")}`,
        );
        // Wait for the RENDER, settling each time. `waitFor` polls the surface
        // but does not flush pending renders, and this panel could sit on a DOM
        // that had not caught up with the crash — `status=crashed` with the
        // diagnosis built, and "stopped" still on screen. Settling in the loop is
        // what actually advances it.
        for (let i = 0; i < 100; i++) {
          await ui_.settle();
          if (ui_.html().includes("GPU ran out of memory")) break;
          await new Promise((r) => setTimeout(r, 20));
        }
        assertStringIncludes(
          ui_.html(),
          "GPU ran out of memory",
          `the reason is rendered; status=${srv.status}`,
        );
        const html = ui_.html();
        assertStringIncludes(html, "still running", "the next step");
        assertStringIncludes(html, "cudaMalloc failed", "the log itself");
        await srv.clearLog();
      });
    },
  );

  testUI(
    Surface as never,
    `${name}: a stray server is named and offered up for freeing`,
    async (ui_) => {
      await withStubBuild(async (bin) => {
        const stray = new Deno.Command(bin, {
          args: ["--port", "0", "--fail-after", "60000"],
          stdout: "null",
          stderr: "null",
        }).spawn();
        try {
          // Poll rather than sleep: the stray has to be far enough along that
          // /proc reports its argv, and a fixed wait is a coin toss — this
          // flaked roughly one run in five at 400 ms.
          for (let i = 0; i < 100; i++) {
            await srv.scanOrphans();
            if (srv.orphans.some((o) => o.pid === stray.pid)) break;
            await new Promise((r) => setTimeout(r, 50));
          }
          assert(
            srv.orphans.some((o) => o.pid === stray.pid),
            `the stray must be seen: ${JSON.stringify(srv.orphans)}`,
          );
          await ui_.settle();

          const html = ui_.html();
          assertStringIncludes(html, "earlier session", "the banner");
          assertStringIncludes(html, String(stray.pid), "which process");
          assertStringIncludes(html, "Free memory", "the way out");

          // And the button's action does what the banner says it will.
          await srv.freeMemory();
          assertEquals(srv.orphans.length, 0);
          assert((await stray.status).code !== null, "the stray is gone");
        } finally {
          try {
            stray.kill("SIGKILL");
          } catch { /* already gone */ }
          try {
            await stray.status;
          } catch { /* already reaped */ }
        }
      });
    },
  );
}

// ── the journeys the spec promises, end to end ─────────────────────────────
//
// These four were the coverage holes: the kata asks for one-click start, a
// working test chat with tok/s, an Update button that rebuilds and restarts,
// and a progress bar on every long job — and each was only tested in its
// refusal state. They run against the stub binary on a real socket.

testUI(
  App,
  "the server starts, reports ready, and stops again",
  async (ui_) => {
    await ui_.settle();
    const bin = await installStubBuild();
    try {
      await withModel(async (dir) => {
        // Known slate: `startBlocker()` reports the run lock, so a server left
        // running by any earlier test would fail this one for an unrelated
        // reason. Tests share a process and these cells are singletons.
        await srv.stop();
        await srv.poll();
        await models.addDir(dir);
        await models.scan();
        await builds.scan();
        // Activate the build this test installed. `activeId` is shared state and
        // other tests point it at builds they later delete, so scanning alone
        // leaves `activeBuild()` null and every start blocked — an order-dependent
        // failure that had nothing to do with what was under test.
        await builds.setActive("stub");
        // And select the model this test scanned for. `scan` keeps a valid
        // selection, but only when its results are the ones that land: it is
        // newest-wins by epoch, so a scan racing with the scheduled one (or
        // with another test's) returns having discarded its own findings, and
        // `models.selected` is then whatever the previous test left — often a
        // fixture in a temp directory that no longer exists. `startBlocker()`
        // said "No model selected — scan for models first" immediately after a
        // scan, roughly one full-suite run in six.
        await ui_.waitFor(
          () => models.items.some((x) => x.meta),
          "the fixture model is in the library",
        );
        const model = models.items.find((x) => x.meta);
        assertExists(model);
        models.select(model.path);
        const port = freePort();
        await cfg.set("port", String(port));
        await ui_.settle();

        assertEquals(startBlocker(), "", "nothing should block the start");
        await startServer();
        assert(srv.pid > 0, "a pid is recorded");
        await untilReady(ui_);

        assertEquals(srv.healthy, true);
        // The command that ran is the command the UI showed.
        assertEquals(srv.argv[0], bin);
        assert(
          srv.argv.includes(String(port)),
          `the chosen port is in argv: ${srv.argv.join(" ")}`,
        );

        await stopServer();
        await srv.poll();
        assertEquals(srv.status, "stopped");
        assertEquals(srv.pid, 0);
      });
    } finally {
      await srv.stop();
      await removeStubBuild();
    }
  },
);

testUI(
  App,
  "the test chat sends a message and reports tokens per second",
  async (ui_) => {
    await ui_.settle();
    const bin = await installStubBuild();
    try {
      await srv.stop(); // known slate — see the note in the lifecycle test
      await srv.poll();
      const port = freePort();
      const url = `http://127.0.0.1:${port}`;
      await srv.start([bin, "--port", String(port)], url);
      await untilReady(ui_);

      // Known slate. `chat.send` returns immediately if `input` is empty or a
      // stream is still open (`src/cell/chat.ts`), and `messages` is persisted —
      // so an earlier test's in-flight stream, or a late rehydration landing
      // after `clear()`, makes this send a silent no-op. Waiting on the
      // precondition rather than on `settle()` removes the ordering dependency.
      await chat.stop();
      await chat.clear();
      await chat.setInput("hi");
      await ui_.waitFor(
        () => chat.input === "hi" && !chat.streaming,
        "chat is idle with the message typed",
      );
      await chat.send(url);
      await ui_.settle();

      assertEquals(chat.streaming, false);
      assertEquals(chat.messages.length, 2);
      assertEquals(chat.messages[1]?.content, "Hello from the stub");
      assert(chat.lastTps > 0, "tok/s is reported");
      // And it is on screen, on both surfaces that offer a chat.
      await ui.go("chat");
      await ui_.waitFor(
        () => ui_.html().includes("Hello from the stub"),
        "the reply renders",
      );
      assertStringIncludes(ui_.html(), "tok/s");
    } finally {
      await srv.stop();
      await removeStubBuild();
    }
  },
);

testUI(
  App,
  "a long job reports progress, not just a spinner",
  async (ui_) => {
    await ui_.settle();
    ui_.App["tab-build"].click();
    // A ref that cannot resolve fails, but only after the progress callback has
    // written its first step — which is the thing being asserted: the panel
    // shows named steps and a bar, never an unexplained wait.
    await builds.setRef("b0-does-not-exist");
    await builds.start();
    await ui_.settle();

    const html = ui_.html();
    assertStringIncludes(html, "Find release", "the steps are named");
    // The job's own output is on the same page as the job.
    assertStringIncludes(html, "Looking up", "the log is on the page");
    assert(builds.job !== null, "a job was recorded");
    assertEquals(builds.job?.status, "failed");
    assert(
      (builds.job?.steps.length ?? 0) > 1,
      "a stepper needs more than one step",
    );
    // A failure is explained, never a raw error.
    assert((builds.job?.error ?? "").length > 0, "the failure has a reason");
  },
);

testUI(
  TunePanel as never,
  "TunePanel: any GPU can be switched off, and the plan follows",
  async (ui_) => {
    await ui_.settle();
    // Real detected hardware, plus a build whose backend can address it — the
    // device NAMES come from the backend, so both halves are needed before the
    // control has anything honest to show.
    await hw.refresh(true);
    await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await cfg.resetOne("device");
      await ui_.settle();

      const list = devices(activeBuild()?.backend, hw.gpus);
      if (list.length === 0) {
        // No NVIDIA card here: the control must SAY which case it is, never
        // render an empty box.
        const html = ui_.html();
        assert(
          html.includes("No GPU detected") || html.includes("addresses no GPU"),
          `an empty picker must explain itself; got: ${html}`,
        );
        return;
      }

      // Every addressable device is offered, by its real name AND by the name
      // llama.cpp will be given.
      const html = ui_.html();
      for (const d of list) {
        assertStringIncludes(html, d.id, `${d.id} must be offered`);
        assertStringIncludes(html, d.label, `${d.label} must be named`);
      }
      // Unrestricted by default: nothing on the command line.
      assertEquals(cfg.settings.device ?? "", "");

      if (list.length < 2) return; // one device: nothing to switch off safely

      const before = computePlan(meta(), hwSnapshot(), cfg.settings);

      // Click the first card off, through the control the user sees. Addressed
      // by "<name> (<id>)" because two identical cards would otherwise share a
      // label — which is why the checkbox carries the id.
      ui_.find("DevicePicker")[list[0]!.id].click();
      await ui_.expectCell(cfg, (s) => s.settings.device !== "");

      const value = String(cfg.settings.device);
      assertEquals(isEnabled(value, list[0]!.id), false, "it is off");
      assertEquals(isEnabled(value, list[1]!.id), true, "the others stay on");
      // It reaches the command line, so what you see is what runs.
      assertStringIncludes(ui_.html(), "-dev");

      // And the plan shrank, because the plan honours the choice — this is the
      // half that was missing entirely: the flag existed, the picture ignored it.
      const after = computePlan(meta(), hwSnapshot(), cfg.settings);
      assert(
        after.vram.capacityB < before.vram.capacityB,
        `disabling a GPU must shrink the plan: ${before.vram.capacityB} → ${after.vram.capacityB}`,
      );
    } finally {
      await cfg.resetOne("device");
      await removeStubBuild("stub-cuda");
    }
  },
);

testUI(
  App,
  "with optimal-automatically on, Start re-tunes for the selected model",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    const bin = await installStubBuild();
    try {
      await withModel(async (dir) => {
        await srv.stop(); // known slate — see the note in the lifecycle test
        await srv.poll();
        await models.addDir(dir);
        await models.scan();
        await builds.scan();
        // Activate the build this test installed. `activeId` is shared state and
        // other tests point it at builds they later delete, so scanning alone
        // leaves `activeBuild()` null and every start blocked — an order-dependent
        // failure that had nothing to do with what was under test.
        await builds.setActive("stub");
        const port = freePort();
        await cfg.set("port", String(port));
        // The tuner refuses to plan against hardware it has not measured — that
        // refusal is deliberate (see `fitsRam`), so a test of what the tuner
        // CHOOSES has to give it a machine to choose for.
        await hw.refresh(true);
        // A value nobody would choose deliberately. If Start honours the switch,
        // the tuner replaces it; if not, this is what would be spawned. The
        // assertion is "not 999" rather than a specific number, because the right
        // `-ngl` depends on the machine the suite happens to run on.
        await cfg.set("ngl", "999");
        assertEquals(cfg.autoOptimal, true, "on by default");
        await ui_.settle();

        await startServer();
        assert(srv.pid > 0, "it started");
        // The argv that ran is the tuned argv, and the panel agrees with it.
        assert(
          Number(cfg.settings.ngl) !== 999,
          "the tuner must replace a hand-set value when the switch is on",
        );
        assert(
          !srv.argv.includes("999"),
          `the untuned value must not reach argv: ${srv.argv.join(" ")}`,
        );
        assert(cfg.reasons.length > 0, "and it says what it chose");
        await stopServer();

        // Switched off, the user's own value is honoured verbatim.
        await cfg.toggleAutoOptimal();
        await cfg.set("ngl", "999");
        await startServer();
        assertEquals(
          Number(cfg.settings.ngl),
          999,
          "off means the hand-set value stands",
        );
        assert(
          srv.argv.includes("999"),
          `it must reach argv when auto is off: ${srv.argv.join(" ")}`,
        );
        await stopServer();
        await cfg.toggleAutoOptimal();
      });
    } finally {
      await srv.stop();
      await removeStubBuild();
    }
    assert(bin.length > 0);
  },
);

testUI(
  TunePanel as never,
  "TunePanel: the optimal-automatically switch is visible and on",
  async (ui_) => {
    await ui_.settle();
    assertEquals(cfg.autoOptimal, true);
    const html = ui_.html();
    assertStringIncludes(
      html,
      "Optimal automatically",
      "the switch is labelled",
    );
    // And it is a real control, not a label: clicking it turns the feature off.
    ui_.find("TunePanel")["auto-optimal"].click();
    await ui_.expectCell(cfg, (s) => s.autoOptimal === false);
    ui_.find("TunePanel")["auto-optimal"].click();
    await ui_.expectCell(cfg, (s) => s.autoOptimal === true);
  },
);

testUI(
  OnePage as never,
  "OnePage: the optimal-automatically switch is offered here too",
  async (ui_) => {
    await ui_.settle();
    // Both surfaces that can start a server must expose the switch that decides
    // what gets started.
    assertExists(ui_.find("OnePage")["one-auto-optimal"]);
    ui_.find("OnePage")["one-auto-optimal"].click();
    await ui_.expectCell(cfg, (s) => s.autoOptimal === false);
    ui_.find("OnePage")["one-auto-optimal"].click();
    await ui_.expectCell(cfg, (s) => s.autoOptimal === true);
  },
);

testUI(
  OnePage as never,
  "OnePage: the chat can be cleared without leaving the page",
  async (ui_) => {
    // The all-in-one page is where a conversation actually happens, so the one
    // gesture that ends it must be here too — otherwise the only way to drop a
    // stale context is a trip to the Chat tab. Driven through the real send
    // path against a real stream, so this clears a genuine conversation.
    await ui_.settle();
    const port = freePort();
    const ac = new AbortController();
    const server = Deno.serve(
      { port, signal: ac.signal, onListen: () => {} },
      () =>
        new Response(
          'data: {"choices":[{"delta":{"content":"a stale answer"}}]}\n\n' +
            "data: [DONE]\n\n",
          { headers: { "content-type": "text/event-stream" } },
        ),
    );
    try {
      await chat.stop();
      await chat.clear();
      await chat.setInput("hi");
      await ui_.waitFor(
        () => chat.input === "hi" && !chat.streaming,
        "chat is idle with the message typed",
      );
      await chat.send(`http://127.0.0.1:${port}`);
      await ui_.waitFor(
        () => ui_.html().includes("a stale answer"),
        "the reply is on the page",
      );
      ui_.find("OnePage")["one-chat-clear"].click();
      await ui_.expectCell(chat, (s) => s.messages.length === 0);
      await ui_.settle();
      assert(
        !ui_.html().includes("a stale answer"),
        "and the transcript is gone from the page",
      );
    } finally {
      ac.abort();
      await server.finished;
    }
  },
);

testUI(
  OnePage as never,
  "OnePage: a code block is a block, with its own copy button, and tok/s follows the answer",
  async (ui_) => {
    // Three complaints in one reply, and all three are about the same thing —
    // a reply is not one undifferentiated string. The fences were visible, the
    // only way to take a file was a drag-select that caught them, and the speed
    // of the answer was printed above the answer, where it cannot be known yet.
    await ui_.settle();
    const port = freePort();
    const ac = new AbortController();
    const reply =
      "Here is the fix:\n\n```ts src/lib/plan.ts\nconst a = 1;\n```\n";
    const server = Deno.serve(
      { port, signal: ac.signal, onListen: () => {} },
      () =>
        new Response(
          `data: ${
            JSON.stringify({ choices: [{ delta: { content: reply } }] })
          }\n\n` +
            `data: ${
              JSON.stringify({
                choices: [{ delta: {} }],
                timings: { predicted_per_second: 8.94 },
              })
            }\n\n` +
            "data: [DONE]\n\n",
          { headers: { "content-type": "text/event-stream" } },
        ),
    );
    try {
      await chat.stop();
      await chat.clear();
      await chat.setInput("fix it");
      await ui_.waitFor(
        () => chat.input === "fix it" && !chat.streaming,
        "chat is idle with the message typed",
      );
      await chat.send(`http://127.0.0.1:${port}`);
      await ui_.waitFor(
        () => ui_.html().includes("const a = 1;"),
        "the reply renders",
      );
      const html = ui_.html();
      // The fence is gone, and what it said is the block's header.
      assert(!html.includes("```"), `no raw fences on screen: ${html}`);
      assertStringIncludes(html, "src/lib/plan.ts");
      assertStringIncludes(html, "codeblock");
      // Its own copy button, because the file is the unit people want — not
      // the message, and certainly not the message plus the prose around it.
      assertExists(
        ui_.find("OnePage")["codeblock-copy"],
        "the block carries a copy button",
      );
      // And the whole conversation has one, beside Clear.
      assertExists(ui_.find("OnePage")["one-chat-copy"]);
      // tok/s AFTER the answer: a measurement of the reply belongs under it.
      const answer = html.indexOf("const a = 1;");
      const speed = html.indexOf("tok/s", answer);
      assert(speed > answer, "tok/s must follow the answer, not head it");
      assert(
        html.slice(0, answer).indexOf("tok/s") === -1 ||
          html.slice(0, answer).lastIndexOf("tok/s") <
            html.lastIndexOf("msg-role", answer),
        "nothing prints the speed above the message body",
      );
    } finally {
      ac.abort();
      await server.finished;
      await chat.clear();
    }
  },
);

testUI(
  OnePage as never,
  "OnePage: a model whose header cannot be read says so",
  async (ui_) => {
    await ui_.settle();
    // Driven through the real scanner with a real (corrupt) ollama store, so
    // this covers both halves: the scanner listing an unreadable manifest
    // instead of dropping it, and the page billed as "everything you need"
    // explaining it. Before, the model vanished entirely; once listed, the page
    // showed a bare filename and an empty state reading "select a model" —
    // while a model WAS selected.
    const dir = await Deno.makeTempDir({ prefix: "llama-master-ollama-" });
    try {
      const man = join(
        dir,
        "manifests",
        "registry.ollama.ai",
        "library",
        "brokenmodel",
      );
      await Deno.mkdir(man, { recursive: true });
      await Deno.mkdir(join(dir, "blobs"), { recursive: true });
      await Deno.writeTextFile(join(man, "latest"), "{ this is not json");

      await models.addDir(dir);
      await models.scan();
      await ui_.settle();

      const broken = models.items.find((x) => x.file.includes("brokenmodel"));
      assertExists(broken, "an unreadable manifest must still be listed");
      assert(broken.metaError, "with the reason attached");

      models.select(broken.path);
      await ui_.settle();
      const html = ui_.html();
      assertStringIncludes(html, "header could not be read");
      assert(
        !html.includes("Select a model to see how it fits"),
        "must not claim nothing is selected when something is",
      );
    } finally {
      await Deno.remove(dir, { recursive: true }).catch(() => {});
    }
  },
);

testUI(
  OnePage as never,
  "OnePage: one click each for the standard context sizes, and for the optimum",
  async (ui_) => {
    await ui_.settle();
    const bin = await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await withModel(async (dir) => {
        await models.addDir(dir);
        await models.scan();
        await ui_.settle();

        const m = models.items.find((x) => x.meta);
        assertExists(m, "the fixture model must parse");
        models.select(m.path);
        await cfg.setCtxOverride(0);
        await ui_.settle();

        const trained = optimalCtx(m.meta!);
        const picker = ui_.find("CtxControls");

        // Every rung is offered, whatever the model — a row that changes length
        // per model is harder to use than one that does not.
        for (const n of CTX_PRESETS) {
          assertExists(
            picker[`one-ctx-${ctxLabel(n)}`],
            `${ctxLabel(n)} must be offered`,
          );
        }

        // A preset the model cannot use is disabled rather than a button that
        // silently does nothing (the tuner caps at the trained length).
        const tooBig = CTX_PRESETS.filter((n) => n > trained);
        for (const n of tooBig) {
          assertEquals(
            picker[`one-ctx-${ctxLabel(n)}`].disabled,
            true,
            `${ctxLabel(n)} is past the ${trained} this model was trained for`,
          );
        }

        // A usable one sets the context, for this model.
        const usable = CTX_PRESETS.find((n) => n <= trained);
        if (usable) {
          picker[`one-ctx-${ctxLabel(usable)}`].click();
          await ui_.expectCell(cfg, (s) => s.ctxOverride === usable);
          assertEquals(cfg.ctxOverrideFor, m.path, "pinned to THIS model");
        }

        // Each named band is offered and pins its own value. Only Max is read
        // from the model; the rest are estimates and the button says so with a
        // marker, which is checked below.
        const bands = ctxBands(m.meta!);
        for (const band of CTX_BANDS) {
          const b = picker[`one-ctx-${band.id}`];
          assertExists(b, `${band.label} CTX must be offered`);
          assertEquals(b.disabled, false, `${band.label} is usable here`);
          b.click();
          await ui_.expectCell(cfg, (st) => st.ctxOverride === bands[band.id]);
          assertEquals(cfg.ctxOverrideFor, m.path, "pinned to THIS model");
        }
        // Ordered, and never past what the model was trained for.
        assert(
          bands.min <= bands.opt && bands.opt <= bands.big &&
            bands.big <= bands.max,
          `bands must be ordered: ${JSON.stringify(bands)}`,
        );
        assertEquals(bands.max, trained, "Max is the trained length, exactly");

        // The estimate is marked as one wherever a band appears.
        assertStringIncludes(
          ui_.html(),
          "≈",
          "estimated bands must be marked",
        );
        assertStringIncludes(
          ui_.html(),
          "no quality signal",
          "and the page must say why",
        );

        // The usable range is drawn, not just listed.
        assertExists(picker["ctx-range"], "the range visual is rendered");
        assertExists(picker["ctx-needle"], "with the current value on it");

        // "Max on VRAM" and "Max on Hybrid": one click for the priority most
        // sessions have — this placement, at the biggest context it holds.
        // On a machine where the placement is impossible the button is
        // disabled with the blocker, never a click that silently does nothing.
        for (const pl of ["vram", "hybrid"] as const) {
          const btn = picker[`one-ctx-max-${pl}`];
          assertExists(btn, `Max·${pl} must be offered`);
          if (!btn.disabled) {
            btn.click();
            await ui_.expectCell(cfg, (s) => s.placement === pl);
            await ui_.expectCell(cfg, (s) => s.ctxOverride > 0);
            assertEquals(cfg.ctxOverrideFor, m.path, "pinned to THIS model");
          }
        }

        // And "Auto" is always there — not only once you have overridden
        // something — and hands the choice back to the tuner.
        picker["one-ctx-optimal"].click();
        await ui_.expectCell(cfg, (s) => s.ctxOverride === 0);
      });
    } finally {
      await removeStubBuild("stub-cuda");
      assert(bin.length > 0);
    }
  },
);

testUI(
  OnePage as never,
  "OnePage: both memory states are in one section, and told apart",
  async (ui_) => {
    await ui_.settle();
    const bin = await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await withModel(async (dir) => {
        await models.addDir(dir);
        await models.scan();
        await ui_.settle();

        // Both, simultaneously — they answer different questions, and showing
        // one with a mode switch meant whichever you were not asking about was
        // simply unavailable. One panel holds them, because they are two
        // answers about one machine rather than two subjects.
        const html = ui_.html();
        assertStringIncludes(html, "As it is now");
        assertStringIncludes(html, "After starting");

        // And llama.cpp's own share is distinguishable from everyone else's and
        // from free space, in words as well as colour.
        assertStringIncludes(html, "In use elsewhere");
        assertStringIncludes(html, "KV cache");
        assertStringIncludes(html, "Weights");

        // Each state says which question it answers, once — the section titles
        // above the maps. The current state of an idle machine is not a
        // projection, and calling it one was wrong; saying it twice, in the
        // title and again in a pill under the map, was merely wasteful in the
        // one column that has no room to waste.
        assertEquals(
          (html.match(/As it is now/g) ?? []).length,
          1,
          "said once",
        );
        assertEquals((html.match(/After starting/g) ?? []).length, 1);
        // And one legend for the two maps that share it.
        assertEquals(
          (html.match(/legend-dot seg-reserved/g) ?? []).length,
          1,
          "one key, not one per map",
        );

        // Three columns, one question each. The machine column stacks the two
        // memory states — same width, same scale, so the difference between
        // "as it is" and "as it would be" is readable at a glance — and the
        // decision column follows with the run strip. Current before
        // projected, and the whole machine column before the decision one.
        const iCur = html.indexOf("As it is now");
        const iRun = html.indexOf("Run a model");
        const iProj = html.indexOf("After starting");
        assert(
          iCur < iProj,
          `current state reads before the projection, got ${iCur}/${iProj}`,
        );
        assert(
          iProj < iRun,
          `the machine column precedes the decision column, got ${iProj}/${iRun}`,
        );
      });
    } finally {
      await removeStubBuild("stub-cuda");
      assert(bin.length > 0);
    }
  },
);

Deno.test("guard: every page the kata lists is reachable", async () => {
  // .katana/pages.md enumerates the pages this app has. A page that exists as a
  // component but is not in TABS is unreachable, and a tab with no case in the
  // router silently falls through to All-in-one — both look fine in isolation.
  const { TABS } = await import("../src/cell/ui.ts");
  const labels = TABS.map((t) => t.label);
  for (
    const want of [
      "All-in-one",
      "Machine",
      "CPU",
      "GPU",
      "Memory",
      "Build",
      "Models",
      "Tune",
      "Server",
      "Chat",
      "About",
    ]
  ) {
    assert(labels.includes(want), `no tab labelled "${want}" (have ${labels})`);
  }
  // And every tab is routed, not silently falling through to the default.
  const app = await Deno.readTextFile(
    new URL("../src/App.tsx", import.meta.url),
  );
  for (const t of TABS) {
    if (t.id === "one") continue; // the default case
    assert(
      app.includes(`case "${t.id}":`),
      `tab "${t.id}" has no case in the router`,
    );
  }
});

for (
  const [name, tab, marker] of [
    ["CPU", "cpu", "What llama.cpp is told to use"],
    ["GPU", "gpu", "Which GPUs llama.cpp may use"],
    ["Memory", "memory", "Storage"],
    // Their own pages, because the kata asks for them as pages: prerequisites
    // are a task with an action on most rows, and storage is the pool that
    // fails a build minutes in.
    ["Prerequisites", "prereq", "Prerequisites"],
    ["Storage", "storage", "What llama.master is using"],
  ] as const
) {
  testUI(App, `the ${name} page renders its own detail`, async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    await hw.refresh(true);
    ui_.App[`tab-${tab}`].click();
    await ui_.expectCell(ui, (s) => s.tab === tab);
    assertStringIncludes(ui_.html(), marker);
  });
}

// ── the polish the katas ask for, pinned ───────────────────────────────────

testUI(
  App,
  "the chat says it is waiting before the first token arrives",
  async (ui_) => {
    // The gap between Send and the first token is the whole point: a local model
    // on a cold cache can think for seconds, and with nothing on screen that is
    // indistinguishable from a dead server. So this drives the real code path
    // against a real socket that accepts the request and then says nothing —
    // exactly the state the indicator exists for.
    await ui_.settle();
    const port = freePort();
    const ac = new AbortController();
    const server = Deno.serve(
      { port, signal: ac.signal, onListen: () => {} },
      () =>
        new Response(
          // A well-formed event stream that never emits an event.
          new ReadableStream({ start() {} }),
          { headers: { "content-type": "text/event-stream" } },
        ),
    );
    try {
      await chat.clear();
      await chat.setInput("hi");
      const sending = chat.send(`http://127.0.0.1:${port}`);
      await ui_.waitFor(
        () => chat.streaming && ui_.html().includes("chat-wait"),
        "the waiting indicator is on screen while nothing has arrived",
      );
      assertEquals(chat.partial, "", "and there is genuinely nothing yet");
      await chat.stop();
      await sending;
      await ui_.settle();
      assert(
        !ui_.html().includes("chat-wait"),
        "and it goes away once the wait is over",
      );
    } finally {
      ac.abort();
      await server.finished;
    }
  },
);

for (
  const [name, panel] of [
    ["All-in-one", OnePage],
    ["Server", ServerPanel],
  ] as const
) {
  testUI(
    panel as never,
    `the ${name} page states whether a server is running, unmissably`,
    async (ui_) => {
      // A page with a Start button has to answer "is one running?" before the
      // user reads anything else. It used to answer in the app's smallest type,
      // in a row of chips that looked identical to the model filename.
      await ui_.settle();
      const html = ui_.html();
      assertStringIncludes(html, "status-big", `${name} shows the big status`);
      assertStringIncludes(html, "STOPPED", "and says which state it is in");
    },
  );
}

testUI(App, "the Machine page covers storage too", async (ui_) => {
  // A build needs gigabytes and "No space left on device" is a failure this app
  // already knows how to explain — better before the build than after. The page
  // that summarises the machine has to include the pool that fails it.
  await ui_.settle();
  await hw.refresh(true);
  await hw.refreshDisks();
  ui_.App["tab-dashboard"].click();
  await ui_.expectCell(ui, (s) => s.tab === "dashboard");
  await ui_.waitFor(
    () => ui_.html().includes("Storage"),
    "the Machine page names storage",
  );
});

testUI(
  App,
  "the current memory state counts this app's own bytes once",
  async (ui_) => {
    // `plan` reads "in use" from device-wide telemetry, and our own llama-server
    // is already inside those totals. Itemising our buckets on top of them
    // counted our memory twice and could paint the over-capacity hatch on a
    // machine that comfortably fits. Only a live server reproduces it.
    if (Deno.build.os !== "linux") return; // RSS comes from /proc
    await ui_.settle();
    await installStubBuild();
    try {
      await withModel(async (dir) => {
        await srv.stop();
        await srv.poll();
        await models.addDir(dir);
        await models.scan();
        await builds.scan();
        // Activate the build this test installed. `activeId` is shared state and
        // other tests point it at builds they later delete, so scanning alone
        // leaves `activeBuild()` null and every start blocked — an order-dependent
        // failure that had nothing to do with what was under test.
        await builds.setActive("stub");
        await cfg.set("port", String(freePort()));
        // Real telemetry, or `otherB` is zero on both sides and the assertion
        // below proves nothing.
        await hw.refresh(true);
        await ui_.settle();
        await startServer();
        await untilReady(ui_);
        await srv.poll();

        const m = models.items.find((i) => i.path === models.selected)?.meta;
        assertExists(m, "the selected model parsed");
        assertExists(srv.runSettings, "the run is recorded");
        assert(srv.rssB > 0, `RSS should be measured, got ${srv.rssB}`);

        // What the map would have drawn before the fix, versus what it draws now.
        const raw = computePlan(m, hwSnapshot(), srv.runSettings);
        const shown = currentStatePlan();

        // Our own share is identical either way — it is our accounting, not the
        // driver's — but it must no longer also be inside "everyone else".
        assertEquals(shown.ram.usedB, raw.ram.usedB, "our share is unchanged");
        assert(
          shown.ram.otherB < raw.ram.otherB,
          `our RSS must come out of "in use elsewhere": ${shown.ram.otherB} vs ${raw.ram.otherB}`,
        );
        // Within a byte, not exactly: `withoutOurUsage` subtracts proportionally
        // across pools, so the arithmetic is floating point and an exact compare
        // fails on the eighth decimal (8589934592.000008 vs 8589934592).
        assert(
          Math.abs((raw.ram.otherB - shown.ram.otherB) - srv.rssB) < 1,
          `and exactly our RSS, no more: ${
            raw.ram.otherB - shown.ram.otherB
          } vs ${srv.rssB}`,
        );
      });
    } finally {
      await srv.stop();
      await removeStubBuild();
    }
  },
);

testUI(
  TunePanel as never,
  "the Tune page offers the same context control as the all-in-one page",
  async (ui_) => {
    // The kata asks for the bands on both pages, and "the same" has to mean the
    // same component — two renderings of a band would be two chances to disagree
    // about what it is worth.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await cfg.setCtxOverride(0);
      await ui_.settle();

      const picker = ui_.find("CtxControls");
      const bands = ctxBands(m.meta!);
      for (const band of CTX_BANDS) {
        assertExists(
          picker[`tune-ctx-${band.id}`],
          `${band.label} CTX on Tune`,
        );
      }
      assertExists(picker["tune-ctx-optimal"], "and Auto");
      assertExists(picker["ctx-range"], "and the usable range");

      // It is wired, not decoration.
      picker["tune-ctx-big"].click();
      await ui_.expectCell(cfg, (s) => s.ctxOverride === bands.big);
      assertEquals(cfg.ctxOverrideFor, m.path, "pinned to THIS model");
    });
  },
);

testUI(
  OnePage as never,
  "a stranded CPU-only choice is pointed out, not left to rot",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // `cfg.placement` is persisted, which is right — it is a choice. But the boot
    // race that used to degrade it to `cpu` before the hardware was read left the
    // value STORED, so a machine with idle GPUs kept running on the CPU every
    // session afterwards with nothing on screen to explain it. Stopping the race
    // could not un-store it; this is what tells the user.
    //
    // Seeded, so the branch under test is the one that runs. This assertion used
    // to be derived at runtime and skipped whenever the developer's own GPU was
    // busy — which is exactly when it mattered.
    await ui_.settle();
    const bin = await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await withModel(async (dir) => {
        await models.addDir(dir);
        await models.scan();
        const m = models.items.find((x) => x.meta);
        assertExists(m, "the fixture model must parse");
        models.select(m.path);
        await cfg.setCtxOverride(0);
        await cfg.setPlacement("cpu");
        await ui_.settle();

        assertEquals(
          betterPlacement(),
          "vram",
          "two idle 24 GB cards beat CPU for this model",
        );
        const page = ui_.find("PlacementAdvice");
        assertExists(page["placement-advice"], "the stranded choice is named");
        assertStringIncludes(
          ui_.html(),
          "leaves the GPU out",
          "and says what it costs",
        );

        // And the way out is one click, not a hunt through the Tune tab.
        page["use-better-placement"].click();
        await ui_.expectCell(cfg, (st) => st.placement === "vram");
        // The component name, not just the element handle: a component that
        // renders nothing now counts as absent, which is the question a test is
        // actually asking.
        await ui_.waitFor(
          () => ui_.absent("PlacementAdvice"),
          "the advice goes away once taken",
        );
        assertEquals(
          betterPlacement(),
          null,
          "and there is nothing left to advise",
        );
      });
    } finally {
      await cfg.setPlacement("hybrid");
      await removeStubBuild("stub-cuda");
      assert(bin.length > 0);
    }
  },
);

testUI(
  OnePage as never,
  "the settings follow memory that moves underneath them",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // The machines this runs on are workstations: a game takes 20 GB of VRAM, a
    // compile takes 8 GB of RAM, and each of those FINISHING changes the right
    // answer again. What must not happen is a re-tune on every 1 s poll, which
    // would rewrite settings the user is reading.
    //
    // Seeded and then MOVED mid-test, so this drives the real thing — the machine
    // changing under a mounted page — instead of asserting that a pure function
    // returns different strings for different inputs, which is all it could do
    // while `hw` was live telemetry.
    await ui_.settle();
    const bin = await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await withModel(async (dir) => {
        await models.addDir(dir);
        await models.scan();
        const m = models.items.find((x) => x.meta);
        assertExists(m, "the fixture model must parse");
        models.select(m.path);
        await cfg.setCtxOverride(0);
        await cfg.setPlacement("vram");
        await ui_.settle();

        const roomy = roomyMachine();
        const quiet = headroomNow();
        assertEquals(cfg.placement, "vram", "it fits on two idle cards");

        // Jitter must NOT be news. 200 MB either way is what a workstation does
        // while sitting still, and re-tuning on it would fight the user's typing.
        ui_.seed({
          hw: {
            gpus: roomy.gpus.map((g) => ({
              ...g,
              vramUsedB: g.vramUsedB + 200 * 1024 ** 2,
            })),
          },
        });
        await ui_.settle();
        assertEquals(headroomNow(), quiet, "200 MB of wobble is not a re-tune");

        // Now something else fills the cards. That IS news, and the page must react.
        ui_.seed({
          hw: {
            gpus: roomy.gpus.map((g) => ({
              ...g,
              vramUsedB: 23.9 * 1024 ** 3,
            })),
          },
        });
        await ui_.settle();
        const squeezed = headroomNow();
        assert(squeezed !== quiet, "memory taken is news");
        console.log(
          "DEBUG squeezed vram:",
          JSON.stringify(
            placements()?.vram && {
              possible: placements()!.vram.possible,
              ctx: placements()!.vram.ctx,
              blocker: placements()!.vram.blocker,
            },
          ),
        );
        console.log(
          "DEBUG model weights GB:",
          (m.meta!.tensorBytes / 1024 ** 3).toFixed(3),
          "nLayer",
          m.meta!.nLayer,
        );

        // And the other direction: the game exits, the room comes back, and the
        // app must offer the card again rather than staying small forever.
        ui_.seed({ hw: { gpus: roomy.gpus } });
        await ui_.settle();
        assertEquals(headroomNow(), quiet, "memory returned is news too");
        await ui_.waitFor(
          () => placements()?.vram.possible === true,
          "VRAM only comes back when the memory does",
        );
      });
    } finally {
      await removeStubBuild("stub-cuda");
      assert(bin.length > 0);
    }
  },
);

testUI(
  OnePage as never,
  "a running model is not counted as somebody else's memory",
  async (ui_) => {
    // THE reported bug. The driver reports device-wide VRAM, so while our own
    // llama-server holds 39 GB that figure is inside the telemetry. Every "what
    // would happen if we started this" question — the placement picker, the
    // tuner, the stability check, the Tune and Models memory plans — was asked
    // against that raw number, so the app said "VRAM only: does not fit" about a
    // model that was at that moment running in VRAM only, and refused the
    // placement it was already using. A message that is false is the worst
    // failure this app can have.
    await ui_.settle();
    await hw.refresh(true);
    if (hw.gpus.length === 0 || hw.mem === null) return;
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await cfg.setCtxOverride(0);
      await ui_.settle();

      // What the machine says while nothing of ours runs.
      const idle = planningHw();
      const idleVram = idle.gpus.reduce((a, g) => a + g.vramUsedB, 0);

      // Now pretend a large model of ours is loaded, exactly as `srv` would
      // report it: the plan for the run, and the telemetry already containing it.
      const ourVramB = Math.min(
        idle.gpus.reduce((a, g) => a + g.vramTotalB, 0) * 0.6,
        8 * 1024 ** 3,
      );
      const withOurs = withoutOurUsage(hwSnapshot(), ourVramB, 0);
      const withOursVram = withOurs.gpus.reduce((a, g) => a + g.vramUsedB, 0);

      // The whole fix in one assertion: attributing our bytes to us gives the
      // planner MORE room, never less — it is the same machine minus us.
      assert(
        withOursVram <= idleVram,
        `our own usage must come out of "in use elsewhere": ${withOursVram} vs ${idleVram}`,
      );
      // Within a byte: `withoutOurUsage` subtracts proportionally across the
      // cards, so this is floating-point arithmetic and an exact compare fails
      // on the eighth decimal.
      assert(
        Math.abs((idleVram - withOursVram) - Math.min(ourVramB, idleVram)) < 1,
        `and exactly our share, no more: ${idleVram - withOursVram} vs ${
          Math.min(ourVramB, idleVram)
        }`,
      );

      // And nothing on the page evaluates placements against raw telemetry any
      // more: the two must agree about what fits.
      const shown = placements();
      assertExists(shown, "placements are computed");
      const truth = tuneAll(m.meta!, planningHw(), cfg.settings, undefined);
      for (const id of ["vram", "hybrid", "cpu"] as const) {
        assertEquals(
          shown[id].possible,
          truth[id].possible,
          `${id} must be judged against the planning machine`,
        );
      }
    });
  },
);

testUI(
  App,
  "the Machine page summarises software and hands over",
  async (ui_) => {
    // The kata asks Machine for a summary of hardware AND software, and asks for
    // Prerequisites as its own page. So Machine counts what is present, NAMES what
    // is missing — a summary that silently hid two missing tools would be worse
    // than none — and sends you to the page that can act on it.
    await ui_.settle();
    await hw.refresh(true);
    await prereq.scan();
    ui_.App["tab-dashboard"].click();
    await ui_.expectCell(ui, (s) => s.tab === "dashboard");
    await ui_.waitFor(
      () => ui_.html().includes("Software"),
      "Machine carries the software summary",
    );

    const dash = ui_.find("PrereqSummary");
    dash["go-prereq"].click();
    await ui_.expectCell(ui, (s) => s.tab === "prereq");
    await ui_.waitFor(
      () => ui_.html().includes("Prerequisites"),
      "and the button reaches the page that can fix them",
    );
  },
);

testUI(App, "the Storage page says which of the disk is ours", async (ui_) => {
  // `df` says a disk is full; it does not say that 40 GB of it is three
  // llama.cpp builds you stopped using. That attribution is the reason this is a
  // page rather than a panel on Memory.
  await ui_.settle();
  await hw.refresh(true);
  await hw.refreshDisks();
  await builds.scan();
  ui_.App["tab-storage"].click();
  await ui_.expectCell(ui, (s) => s.tab === "storage");
  await ui_.waitFor(
    () => ui_.html().includes("What llama.master is using"),
    "the footprint panel renders",
  );
  const html = ui_.html();
  assertStringIncludes(html, "Installed builds");
  assertStringIncludes(html, "Models found");
  // And the REAL paths, so it can be cleaned up by hand too — `paths()`
  // honours LLAMA_MASTER_HOME, and printing a hardcoded `~/.llama-master`
  // would name a directory that does not exist on this install.
  const { paths } = await import("../src/cell/host.server.ts");
  assertStringIncludes(html, paths().builds);
  assertStringIncludes(html, paths().cache);
});

testUI(
  TunePanel as never,
  "a flag this model cannot honour is refused, not offered",
  async (ui_) => {
    // `--spec-type draft-mtp` against a model with no MTP block is not a slow
    // server: llama.cpp asserts on `n_layer_nextn > 0` and refuses to load. The
    // catalog knows what llama.cpp accepts; only the model knows what it supports.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      assertEquals(m.meta!.nextnLayers, 0, "the fixture ships no MTP block");
      models.select(m.path);
      await ui_.settle();

      assertStringIncludes(
        paramBlocker("specType"),
        "no multi-token-prediction block",
        "the reason is stated in the user's terms",
      );
      const html = ui_.html();
      assertStringIncludes(
        html,
        "Speculative decoding",
        "the control is present",
      );
      assertStringIncludes(
        html,
        "refuses to load when one is asked for",
        "and says why it cannot be used here",
      );
      // The options read as English, not as llama.cpp's vocabulary.
      assertStringIncludes(html, "MTP (model's own block)");
    });
  },
);

testUI(
  TunePanel as never,
  "a dropdown shows the value that is actually set",
  async (ui_) => {
    // Two instruments, one right: `value` on a <select> is a DOM PROPERTY, never
    // an attribute, so `html()` cannot see it and asserting on markup reports a
    // failure that is not there. The surface is the honest view — the same one
    // `am surface` gives and aio's own regression test uses.
    //
    // `value` on a <select> is a DOM PROPERTY, never an attribute — `html()`
    // cannot see it, so the surface is the only honest instrument here (the same
    // view `am surface` gives, and what aio's own regression test uses).
    await ui_.settle();
    await cfg.set("flashAttn", "on");
    await cfg.set("cacheTypeK", "q8_0");
    await ui_.settle();

    const selects = new Map<string, string>();
    const walk = (n: Record<string, unknown>) => {
      for (const e of (n.elements ?? []) as Record<string, unknown>[]) {
        if (e.tag === "select") {
          selects.set(String(e.name), String(e.value ?? ""));
        }
      }
      for (const c of (n.children ?? []) as Record<string, unknown>[]) walk(c);
    };
    walk(ui_.surface() as unknown as Record<string, unknown>);

    for (
      const [handle, want] of [
        ["FlashAttentionSelect", "on"],
        ["KVCacheTypeKSelect", "q8_0"],
      ] as const
    ) {
      assertEquals(
        selects.get(handle),
        want,
        `${handle} must show the value that is set, not whichever option is first`,
      );
    }
  },
);

// ── the memory the user keeps for themselves ───────────────────────────────

testUI(
  OnePage as never,
  "OnePage: reserved VRAM and RAM are held back from the plan that runs",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // The complaint this exists for: the tuner fills the card, the card is also
    // the one drawing the desktop, and the desktop is what pays. So the test is
    // not that a control exists — it is that a number typed into it comes out
    // the other end as memory NOTHING is allowed to place a model in.
    await ui_.settle();
    const bin = await installStubBuild("stub-cuda", { backend: "cuda" });
    try {
      await builds.scan();
      await builds.setActive("stub-cuda");
      await withModel(async (dir) => {
        await models.addDir(dir);
        await models.scan();
        const m = models.items.find((x) => x.meta);
        assertExists(m, "the fixture model must parse");
        models.select(m.path);
        await cfg.setCtxOverride(0);
        await cfg.setPlacement("vram");
        await cfg.setReserve("gpu", 0);
        await cfg.setReserve("connected", 0);
        await cfg.setReserve("ram", 0);
        await ui_.settle();

        const GiB = 1024 ** 3;
        const open = projectedStatePlan();
        assertExists(open, "a model with a header must project");
        assertEquals(open.vram.reservedB, 0, "nothing held back to begin with");
        const openFreeVramB = open.vram.freeB;
        const openFreeRamB = open.ram.freeB;

        // Typed into the real controls, on the all-in-one page.
        const page = ui_.find("OnePage");
        assertExists(page["one-reserve-gpu"], "per-GPU VRAM is on this page");
        assertExists(
          page["one-reserve-connected"],
          "and the display-card figure beside it",
        );
        assertExists(page["one-reserve-ram"], "and RAM with them");
        await page["one-reserve-connected"].setValue("4");
        await page["one-reserve-ram"].setValue("16");
        await ui_.expectCell(
          cfg,
          (s) =>
            s.reserveConnectedVramB === 4 * GiB &&
            s.reserveRamB === 16 * GiB,
          "the boxes speak GB and the cell holds bytes",
        );
        await ui_.settle();

        const held = projectedStatePlan();
        assertExists(held);
        assertEquals(held.vram.reservedB, 4 * GiB);
        assertEquals(held.ram.reservedB, 16 * GiB);
        assertEquals(
          openFreeVramB - held.vram.freeB,
          4 * GiB,
          "and it is gone from what the plan may spend, byte for byte",
        );
        assertEquals(openFreeRamB - held.ram.freeB, 16 * GiB);
        // Two cards, one screen: the connected figure lands on the card that
        // drives it and costs the headless card nothing. Charging both would
        // refuse 4 GB nobody asked to keep.
        assertEquals(held.devices.cards[0]?.reservedB, 4 * GiB);
        assertEquals(held.devices.cards[1]?.reservedB, 0);
        // The per-GPU figure is the other claim, and it IS charged to both.
        await page["one-reserve-gpu"].setValue("2");
        await ui_.expectCell(cfg, (s) => s.reservePerGpuVramB === 2 * GiB);
        await ui_.settle();
        const both = projectedStatePlan();
        assertExists(both);
        assertEquals(both.devices.cards[0]?.reservedB, 6 * GiB, "2 + 4 here");
        assertEquals(both.devices.cards[1]?.reservedB, 2 * GiB, "2 there");
        assertEquals(both.vram.reservedB, 8 * GiB, "8 GB of machine in total");
        await page["one-reserve-gpu"].setValue("0");
        await ui_.expectCell(cfg, (s) => s.reservePerGpuVramB === 0);
        await ui_.settle();
        // The machine itself has not changed size, and the page still says so.
        assertEquals(held.vram.capacityB, open.vram.capacityB);
        assertStringIncludes(ui_.html(), "reserved");
        // And the page NAMES the card it landed on. "8 GB reserved" on a
        // two-card machine is ambiguous in exactly the way that sends someone
        // hunting the wrong GPU for the memory.
        assertStringIncludes(ui_.html(), "Display card: GPU 0");
        // And the map DRAWS it. It was drawn all along in a grey that read as
        // empty track and named in no legend, so the one band the user put
        // there themselves was the one band they could not identify.
        assertStringIncludes(
          ui_.html(),
          "map-band seg-reserved",
          "reserved memory is a band of the map, not a gap in it",
        );
        assertStringIncludes(
          ui_.html(),
          '<i class="legend-dot seg-reserved"></i>Reserved',
          "and the legend names it",
        );
        assertStringIncludes(
          ui_.html(),
          "<span>4.00 GB reserved</span>",
          "the map foot counts it apart from used and from free",
        );
        // And the summary is ONE sentence: adjacent conditional strings in a
        // fragment left the previous one on screen beside the new one.
        assertEquals(
          (ui_.html().match(/every plan below is made from what is left/g) ??
            [])
            .length,
          1,
          "the summary is written once, not once per edit",
        );

        // And what actually STARTS is planned the same way: the tuner re-runs
        // on a reserve change (it is in the auto-tune key), so the settings on
        // file are the ones the reserve allows.
        await ui_.waitFor(
          () => (placements()?.vram.ctx ?? 0) > 0,
          "the tuner still finds a plan with the reserve held back",
        );
        const tuned = tuneAll(
          m.meta!,
          planningHw(),
          cfg.settings,
        );
        assertEquals(
          computePlan(m.meta!, planningHw(), tuned.vram.settings).vram
            .reservedB,
          4 * 1024 ** 3,
          "the plan behind Start carries the reserve too",
        );
      });
    } finally {
      await removeStubBuild("stub-cuda");
      assert(bin.length > 0);
    }
  },
);

testUI(
  TunePanel as never,
  "Tune: the same reserve controls, above the plan they change",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // Two surfaces, one component — the same rule as the context control. A
    // reserve set on one page and invisible on the other would be a setting the
    // user cannot find again.
    await ui_.settle();
    await cfg.setReserve("gpu", 2);
    await cfg.setReserve("connected", 8);
    await cfg.setReserve("ram", 16);
    await ui_.settle();
    const panel = ui_.find("TunePanel");
    assertExists(panel["tune-reserve-gpu"]);
    assertExists(panel["tune-reserve-connected"]);
    assertExists(panel["tune-reserve-ram"]);
    assertEquals(panel["tune-reserve-gpu"].value, "2");
    // And the LAN switch, which is the same control as the all-in-one page's:
    // a setting that appears twice must be the same control twice.
    assertExists(panel["tune-lan-toggle"], "Available on LAN is here too");
    assertEquals(panel["tune-reserve-connected"].value, "8");
    assertEquals(panel["tune-reserve-ram"].value, "16");

    // A reserve larger than the card it applies to is clamped and SAID, not
    // swallowed: every placement is about to report that it does not fit, and a
    // refusal with no visible cause is the worst message this app can produce.
    await panel["tune-reserve-connected"].setValue("999");
    await ui_.expectCell(
      cfg,
      (s) => s.reserveConnectedVramB === 999 * 1024 ** 3,
    );
    await ui_.settle();
    assertExists(
      ui_.find("TunePanel")["tune-reserve-connected-over"],
      "more than the card has must say so",
    );
    await cfg.setReserve("gpu", 0);
    await cfg.setReserve("connected", 4);
  },
);

testUI(
  OnePage as never,
  "a model that is running is described, not accused",
  {
    seed: {
      // Two cards that are HOLDING our run: what they report as used is our
      // model plus a residue our own accounting does not capture (llama.cpp's
      // real overhead is larger than the estimate). That residue is what made
      // the fitter re-pack the run and come up short.
      hw: {
        lastRefresh: 1,
        mem: {
          totalB: 64 * 1024 ** 3,
          availableB: 48 * 1024 ** 3,
          usedB: 16 * 1024 ** 3,
          swapTotalB: 0,
          swapUsedB: 0,
        },
        gpus: [0, 1].map((i) => ({
          vendor: "nvidia",
          name: `Test card ${i}`,
          tempC: 50,
          utilPct: 40,
          vramTotalB: 1024 ** 3,
          vramUsedB: 540 * 1024 ** 2,
          powerW: 60,
          powerLimitW: 150,
          computeCap: 8.9,
        })),
      },
    },
  },
  async (ui_) => {
    // THE reported message: "1010 MB of layers have nowhere to go — no card has
    // room for them, however the cut is made", on the machine panel, about a
    // server that was answering prompts, with the VRAM total reading 0 over
    // capacity right beside it. The fitter's budgets hold back a safety reserve
    // and re-derive our footprint by proportion, so re-packing a LOADED model
    // came up short — a prediction contradicted by the thing it predicted.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await ui_.settle();

      const runSettings = {
        ...cfg.settings,
        ngl: 999,
        nCpuMoe: 0,
        ctxSize: 2048,
      };
      ui_.seed({
        srv: {
          status: "ready",
          pid: 4242,
          startedAt: 1,
          runModel: m.path,
          runSettings,
          rssB: 200 * 1024 ** 2,
          healthy: true,
          url: "http://127.0.0.1:8080",
        },
      });
      await ui_.settle();

      const live = currentStatePlan();
      // The scenario has to be the real one, or this test proves nothing: as a
      // PROPOSAL, on the very same machine, these settings cannot be cut across
      // the cards — while the machine is not over VRAM at all.
      const attributed = withoutOurUsage(
        hwSnapshot(),
        live.vram.usedB,
        srv.rssB,
      );
      const asProposal = computePlan(m.meta!, attributed, runSettings);
      assertEquals(
        asProposal.vram.overB,
        0,
        "the VRAM total is not the problem",
      );
      assertEquals(
        asProposal.devices.fits,
        false,
        "and the packer would refuse it — this is the reported situation",
      );
      assert(asProposal.devices.unplacedB > 0);

      // What the app must say about the same thing, now that it is running.
      assertEquals(live.devices.fits, true, "llama.cpp placed these layers");
      assertEquals(live.devices.unplacedB, 0);
      assert(
        !live.notes.some((n) => n.includes("no card that can hold them")),
        `no note may contradict the running server: ${live.notes.join(" | ")}`,
      );
      assert(
        !ui_.html().includes("nowhere to go"),
        "and the page must not print it either",
      );
    });
  },
);

testUI(
  OnePage as never,
  "OnePage: Available on LAN is off, and turning it on changes what runs",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // The switch exists because llama.cpp binds to 127.0.0.1 and is therefore
    // invisible from every other machine — the commonest reason the client
    // (client/) finds nothing on the network. It is one flag, `--host`, and it
    // is OFF by default: binding to the world is not a default.
    await ui_.settle();
    await cfg.set("host", "127.0.0.1");
    await ui_.settle();

    const page = ui_.find("OnePage");
    assertExists(page["one-lan-toggle"], "the switch is on this page");
    // Off is read from what is on screen and what would run, not from the
    // checkbox's own attribute: `checked` is only serialised when true, so its
    // absence is the state rather than a value to compare against.
    assertEquals(cfg.settings.host, "127.0.0.1", "off by default");
    assert(
      !ui_.html().includes("--host"),
      "and the command says nothing about a host it is not changing",
    );
    assert(
      !ui_.html().includes("Anyone on your network"),
      "nor warns about an exposure that has not happened",
    );

    page["one-lan-toggle"].click();
    await ui_.expectCell(cfg, (s) => s.settings.host === "0.0.0.0");
    await ui_.settle();

    // What you see is what runs: the same catalog, the same command builder.
    assertStringIncludes(ui_.html(), "--host 0.0.0.0");
    // The address another machine actually dials — 0.0.0.0 is a bind address,
    // and typing it into the client reaches nothing. Said ONCE: a fragment of
    // adjacent conditional strings rendered "Reachable at Reachable at —" with
    // the address missing, which is the trap `ReserveControls` fell into.
    assertStringIncludes(ui_.html(), "http://192.168.1.24:");
    assertEquals(
      (ui_.html().match(/Reachable at/g) ?? []).length,
      1,
      "written once, not once per re-render",
    );
    // The RISK is not repeated here: an open bind with no API key already
    // raises a red banner on this page (src/lib/stability.ts), and saying it
    // twice makes both quieter.
    assertStringIncludes(
      ui_.html(),
      "anyone on the network can use this model",
    );

    page["one-lan-toggle"].click();
    await ui_.expectCell(cfg, (s) => s.settings.host === "127.0.0.1");
    await ui_.settle();
    assert(!ui_.html().includes("Reachable at"), "and it goes quiet");
  },
);

testUI(
  OnePage as never,
  "OnePage: low priority is on by default, and the run carries it",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // llama.cpp takes every core and every spare IOPS; on the machine it runs
    // on that reads as "the computer is broken". ON by default, because a model
    // that makes the desktop stutter is a model the user turns off.
    await ui_.settle();
    const page = ui_.find("OnePage");
    assertExists(page["one-prio-toggle"], "the switch is on this page");
    assertEquals(cfg.lowPriority, true, "on by default — the PC can breathe");

    page["one-prio-toggle"].click();
    await ui_.expectCell(cfg, (s) => s.lowPriority === false);
    page["one-prio-toggle"].click();
    await ui_.expectCell(cfg, (s) => s.lowPriority === true);

    // It is NOT a llama.cpp flag: nothing about it may reach the command line,
    // because it is applied to the process after the spawn.
    await ui_.settle();
    assert(
      !ui_.html().includes("nice") || !ui_.html().includes("--nice"),
      "the argv stays the argv",
    );
  },
);

testUI(
  OnePage as never,
  "OnePage: the machine dials are coloured by how full they are",
  {
    seed: {
      hw: (() => {
        const GB = 1024 ** 3;
        const m = roomyMachine();
        // A machine under real load: 90% of the CPU, and a card with 22 of its
        // 24 GB spoken for.
        m.cpu.utilPct = 90;
        m.gpus = m.gpus.map((g, i) => ({
          ...g,
          utilPct: i === 0 ? 60 : 30,
          vramUsedB: 22 * GB,
        }));
        m.mem = { ...m.mem, usedB: 40 * GB, availableB: 88 * GB };
        return m;
      })(),
    },
  },
  async (ui_) => {
    // Four dials asked the same question — how full, how busy — answer on one
    // scale: cyan under a quarter, green to half, amber to three quarters, red
    // past that. They used to be fixed colours, so a card at 96% and a card at
    // 4% were the same shade of blue.
    await ui_.settle();
    const html = ui_.html();
    // CPU 90% → red. VRAM 44 of 48 GB → red. RAM 40 of 128 → green.
    // GPU 60% → amber.
    const dials = [...html.matchAll(/ring-wrap tone-(\w+)/g)].map((m) => m[1]);
    assertEquals(
      dials.slice(0, 4),
      ["bad", "warn", "ok", "bad"],
      `CPU, GPU, RAM, VRAM by quarter: ${dials.join(",")}`,
    );
  },
);

testUI(
  OnePage as never,
  "OnePage: one memory section — both states while choosing, only the measured one while a model runs",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // Two panels, "Current" and "Projected", asked the reader to hold two
    // pictures of the same machine side by side. While nothing is running that
    // is the point — now and next is the whole decision. Once a model is up it
    // is not: the projection's question has been answered by the machine
    // itself, and an estimate beside the measurement of the same thing invites
    // the reader to wonder which one is wrong.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await ui_.settle();

      const idle = ui_.html();
      assertStringIncludes(idle, "As it is now", "now");
      assertStringIncludes(
        idle,
        "After starting",
        "and next, while nothing is running",
      );
      assert(
        !idle.includes("Projected Memory State"),
        "one section, not two panels",
      );

      ui_.seed({
        srv: {
          status: "ready",
          pid: 4242,
          startedAt: 1,
          runModel: m.path,
          runSettings: { ...cfg.settings, ngl: 999, ctxSize: 2048 },
          rssB: 200 * 1024 ** 2,
          healthy: true,
          url: "http://127.0.0.1:8080",
        },
      });
      await ui_.settle();

      const live = ui_.html();
      assertStringIncludes(live, "As it is now", "the measurement stays");
      assert(
        !live.includes("After starting"),
        "and the forecast of the same thing goes",
      );
    });
  },
);

testUI(
  OnePage as never,
  "OnePage: while a model runs, the locked context controls give their space to the log",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // The complaint: with a server up, the server log took the whole page and
    // the panels ran past the bottom of the window. Half of it was a stray
    // element from a malformed favicon (see tests/guards.test.ts); the other
    // half is here — 180 pixels of context buttons that CANNOT BE PRESSED
    // while a model is loaded, sitting above the output everyone reads during
    // a load. One model runs at a time, so the context is fixed until it stops.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await ui_.settle();

      const idle = ui_.find("OnePage");
      assertExists(
        idle["one-ctx-bands"],
        "every control is there to choose with",
      );
      assertExists(idle["one-ctx-presets"]);

      ui_.seed({
        srv: {
          status: "ready",
          pid: 4242,
          startedAt: 1,
          runModel: m.path,
          runSettings: { ...cfg.settings, ngl: 999, ctxSize: 2048 },
          healthy: true,
          url: "http://127.0.0.1:8080",
          log: Array.from(
            { length: 300 },
            (_, i) => `llama_model_loader: ${i}`,
          ),
        },
      });
      await ui_.settle();

      const html = ui_.html();
      assert(
        !html.includes("ctx-bands") && !html.includes("ctx-presets") &&
          !html.includes("ctx-range"),
        "the controls that cannot be used are not rendered",
      );
      assertStringIncludes(
        html,
        'class="one-ctx"',
        "the number itself stays — what the running server uses is worth reading",
      );
      // And the log is on the page, in the panel that fills what is left.
      assertStringIncludes(html, "log log-fill");
      assertStringIncludes(html, "llama_model_loader: 299");
    });
  },
);

testUI(
  OnePage as never,
  "the projection is of the command Start would issue, not a stale one",
  { seed: { hw: roomyMachine() } },
  async (ui_) => {
    // The panel says "after starting". With auto-optimal on, what starts is the
    // TUNER's answer for the machine as it is now — and the tuner is suspended
    // while a server is up, because a loaded model cannot be re-placed. So the
    // settings on file drift away from what a restart would use, and projecting
    // them showed a plan for a command nobody would ever issue: on the reported
    // machine it drew gigabytes of unplaceable layers while the tuner, asked the
    // same second, had a plan that fitted.
    await ui_.settle();
    await withModel(async (dir) => {
      await models.addDir(dir);
      await models.scan();
      const m = models.items.find((x) => x.meta);
      assertExists(m, "the fixture model must parse");
      models.select(m.path);
      await cfg.setCtxOverride(0);
      await cfg.setPlacement("vram");
      await ui_.settle();

      // Settings that no longer describe what the tuner would choose — the state
      // a suspended auto-tune leaves behind.
      await cfg.apply({ ngl: 1, nCpuMoe: 0, ctxSize: 4096 });
      await ui_.settle();

      const all = placements();
      assertExists(all, "the tuner has an answer");
      const projected = projectedStatePlan();
      assertExists(projected);
      assertEquals(
        projected.ctx,
        all[cfg.placement].ctx,
        "the projection must be of the tuner's plan, not the stale settings",
      );
      assertEquals(
        projected.layersOnGpu,
        all[cfg.placement].possible
          ? computePlan(m.meta!, planningHw(), all[cfg.placement].settings)
            .layersOnGpu
          : projected.layersOnGpu,
      );

      // With auto-optimal OFF the user's own settings are what Start runs, so
      // those are what gets projected — including a stale-looking one they
      // typed on purpose.
      await cfg.toggleAutoOptimal();
      await ui_.settle();
      const own = projectedStatePlan();
      assertExists(own);
      assertEquals(own.ctx, 4096, "hand-tuned settings are projected as typed");
      // `-ngl 1` offloads one SLOT, and the output head is the last of them —
      // so no transformer layer moves. llama.cpp's arithmetic, projected as it
      // will run rather than as it reads (`src/lib/devsplit.ts:offloadRange`).
      assertEquals(own.layersOnGpu, 0);
      await cfg.toggleAutoOptimal();
    });
  },
);
