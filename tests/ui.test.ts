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
import { plan as computePlan } from "../src/lib/plan.ts";
import { activeBuild, hwSnapshot } from "../src/ui/derive.ts";
import { chat } from "../src/cell/chat.ts";
import { startBlocker, startServer, stopServer } from "../src/ui/actions.ts";
import { updateInfo } from "../src/ui/derive.ts";
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
  "the machine tab reports this machine and its prerequisites",
  async (ui_) => {
    // Settle first: `ui.tab` is persisted and its rehydration on mount is
    // async, so a rail click issued before it lands is silently reverted.
    await ui_.settle();
    // The app opens on the all-in-one page now; this is the deep tab.
    ui_.App["tab-dashboard"].click();
    await ui_.settle();
    await prereq.scan();
    await ui_.settle();

    const html = ui_.html();
    for (
      const marker of [
        "CPU",
        "GPU",
        "Memory",
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
    ui_.App["tab-dashboard"].click();
    await ui_.settle();
    await prereq.scan();
    await ui_.settle();

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
  App,
  "the all-in-one page picks the build as well as the model",
  async (ui_) => {
    await ui_.settle();
    // Navigate explicitly: the cells are process-wide singletons, so relying on
    // the default tab makes this pass or fail on whatever ran before it.
    ui_.App["tab-one"].click();
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
      assertExists(ui_.App["one-getllama"]);
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
  try {
    await fn(bin);
  } finally {
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
function freePort(): number {
  const l = Deno.listen({ port: 0 });
  const { port } = l.addr as Deno.NetAddr;
  l.close();
  return port;
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
        for (let i = 0; i < 200 && srv.status !== "crashed"; i++) {
          await new Promise((r) => setTimeout(r, 25));
          await srv.poll();
        }
        assertEquals(srv.status, "crashed");
        await ui_.settle();

        const html = ui_.html();
        assertStringIncludes(html, "GPU ran out of memory", "the reason");
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

      await chat.clear();
      await chat.setInput("hi");
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
        const port = freePort();
        await cfg.set("port", String(port));
        // A value nobody would choose deliberately: 999 layers on a machine the
        // fixture gives no GPU. If Start honours the switch, the tuner replaces
        // it; if not, this is what would be spawned.
        await cfg.set("ngl", "999");
        assertEquals(cfg.autoOptimal, true, "on by default");
        await ui_.settle();

        await startServer();
        assert(srv.pid > 0, "it started");
        // The argv that ran is the tuned argv, and the panel agrees with it.
        assertEquals(
          Number(cfg.settings.ngl),
          0,
          "no GPU in the fixture, so the tuner offloads nothing",
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
