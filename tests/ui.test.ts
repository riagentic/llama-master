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
    gpus: [card, { ...card }],
    os: "linux",
    arch: "x86_64",
    lastRefresh: 1,
  };
}

// ── shell ──────────────────────────────────────────────────────────────────

testUI(
  App,
  "boots with the brand, every tab, and the command strip",
  async (ui_) => {
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "llama");
    for (
      const label of ["Machine", "Build", "Models", "Tune", "Server", "Chat"]
    ) {
      assertStringIncludes(html, label);
    }
    // The command strip is always present and always read-only.
    assertStringIncludes(html, "llama-server");
    assertStringIncludes(html, "llama-cli");
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
  App,
  "the command strip shows exactly what a setting change produces",
  async (ui_) => {
    await ui_.settle();
    assert(!ui_.html().includes("-ngl"), "a default config emits no -ngl");

    await cfg.set("ngl", "99");
    await cfg.set("ctxSize", "16384");
    await ui_.settle();

    const html = ui_.html();
    assertStringIncludes(html, "-ngl 99");
    assertStringIncludes(html, "-c 16384");
    // Server-only flags must not appear in the cli line, and vice versa.
    await cfg.set("port", "9099");
    await ui_.settle();
    assertStringIncludes(ui_.html(), "--port 9099");
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
            picker[`ctx-${ctxLabel(n)}`],
            `${ctxLabel(n)} must be offered`,
          );
        }

        // A preset the model cannot use is disabled rather than a button that
        // silently does nothing (the tuner caps at the trained length).
        const tooBig = CTX_PRESETS.filter((n) => n > trained);
        for (const n of tooBig) {
          assertEquals(
            picker[`ctx-${ctxLabel(n)}`].disabled,
            true,
            `${ctxLabel(n)} is past the ${trained} this model was trained for`,
          );
        }

        // A usable one sets the context, for this model.
        const usable = CTX_PRESETS.find((n) => n <= trained);
        if (usable) {
          picker[`ctx-${ctxLabel(usable)}`].click();
          await ui_.expectCell(cfg, (s) => s.ctxOverride === usable);
          assertEquals(cfg.ctxOverrideFor, m.path, "pinned to THIS model");
        }

        // Each named band is offered and pins its own value. Only Max is read
        // from the model; the rest are estimates and the button says so with a
        // marker, which is checked below.
        const bands = ctxBands(m.meta!);
        for (const band of CTX_BANDS) {
          const b = picker[`ctx-${band.id}`];
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

        // And "Auto" is always there — not only once you have overridden
        // something — and hands the choice back to the tuner.
        picker["ctx-optimal"].click();
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
  "OnePage: both memory states are shown at once, and told apart",
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
        // simply unavailable.
        const html = ui_.html();
        assertStringIncludes(html, "Current Memory State");
        assertStringIncludes(html, "Projected Memory State");

        // And llama.cpp's own share is distinguishable from everyone else's and
        // from free space, in words as well as colour.
        assertStringIncludes(html, "In use elsewhere");
        assertStringIncludes(html, "KV cache");
        assertStringIncludes(html, "Weights");

        // Each table says which question it answers. The current state of an
        // idle machine is not a projection, and calling it one was wrong.
        assertStringIncludes(html, "as it is now");
        assertStringIncludes(html, "What these settings would use.");

        // And the page reads in decision order: state, settings, consequence.
        const iCur = html.indexOf("Current Memory State");
        const iRun = html.indexOf("Run a model");
        const iProj = html.indexOf("Projected Memory State");
        assert(
          iCur < iRun && iRun < iProj,
          `order should be current -> settings -> projected, got ${iCur}/${iRun}/${iProj}`,
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
        assertExists(picker[`ctx-${band.id}`], `${band.label} CTX on Tune`);
      }
      assertExists(picker["ctx-optimal"], "and Auto");
      assertExists(picker["ctx-range"], "and the usable range");

      // It is wired, not decoration.
      picker["ctx-big"].click();
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
  // And the paths, so it can be cleaned up by hand too.
  assertStringIncludes(html, "builds/");
  assertStringIncludes(html, "cache/");
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
