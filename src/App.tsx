// src/App.tsx — the shell: header, rail, panel, command strip.
//
// Reads cell state directly; AIR subscribes per component, so the header's
// live numbers re-render every second without the panel below it doing any
// work. No routing library — six panels, one `ui.tab`.

import { cfg } from "./cell/cfg.ts";
import { hw } from "./cell/hw.ts";
import { TABS, ui } from "./cell/ui.ts";
import { bytes } from "./lib/format.ts";
import { commandLine } from "./lib/command.ts";
import { Pill } from "./ui/kit.tsx";
import { MemoryMini } from "./ui/Memory.tsx";
import { Dashboard } from "./ui/Dashboard.tsx";
import { CpuPanel } from "./ui/CpuPanel.tsx";
import { GpuPanel } from "./ui/GpuPanel.tsx";
import { MemoryPanel } from "./ui/MemoryPanel.tsx";
import { BuildPanel } from "./ui/BuildPanel.tsx";
import { ModelsPanel } from "./ui/ModelsPanel.tsx";
import { TunePanel } from "./ui/TunePanel.tsx";
import { ServerPanel, StatusPill } from "./ui/ServerPanel.tsx";
import { ChatPanel } from "./ui/ChatPanel.tsx";
import { OnePage } from "./ui/OnePage.tsx";
import { About } from "./ui/About.tsx";
import { cliBin, serverBin, updateNow } from "./ui/actions.ts";
import {
  activeBuild,
  buildBusy,
  buildsSizeB,
  currentModel,
  modelsSizeB,
  updateInfo,
  vramTotalB,
  vramUsedB,
} from "./ui/derive.ts";

/** Shown only when upstream is actually ahead. One press re-acquires the build
 *  by the route it came from and puts the server back as it was. */
function UpdateButton() {
  const u = updateInfo();
  if (!u.available) return null;
  return (
    <button
      type="button"
      class="btn tiny primary"
      t="update"
      title={`${u.reason} ${u.from} → ${u.to}`}
      disabled={buildBusy()}
      onClick={() => void updateNow()}
    >
      {buildBusy() ? "Updating…" : `Update → ${u.to}`}
    </button>
  );
}

/**
 * The app mark, as a shape rather than the `◆` glyph it used to be.
 *
 * Same geometry as src/icon.svg (and so the tab icon and the packaged app
 * icon), which is the point: one shape, three places, no drift. `currentColor`
 * means it follows `--accent`, so it changes with the theme instead of being
 * pinned to the dark palette's orange.
 */
function BrandMark() {
  return (
    <svg
      class="brand-mark"
      viewBox="0 0 64 64"
      width="18"
      height="18"
      role="img"
      aria-label="llama.master"
    >
      <rect
        x="20"
        y="20"
        width="24"
        height="24"
        rx="4"
        transform="rotate(45 32 32)"
        fill="currentColor"
      />
    </svg>
  );
}

function Header() {
  const build = activeBuild();
  const model = currentModel();
  const cpu = hw.cpu;
  const gpu = hw.gpus[0];
  const mem = hw.mem;
  return (
    <header class="topbar">
      <div class="brand">
        <BrandMark />
        <span class="brand-name">
          llama<b>.master</b>
        </span>
      </div>

      <div class="topbar-chips">
        <Pill
          tone={build ? "ok" : "warn"}
          title={build?.dir ?? "No build installed"}
        >
          {build ? `${build.ref} · ${build.backend}` : "no build"}
        </Pill>
        <Pill
          tone={model ? "ok" : "warn"}
          title={model?.path ?? "No model selected"}
        >
          {model ? model.file : "no model"}
        </Pill>
        <StatusPill />
        <UpdateButton />
      </div>

      <div class="topbar-live">
        <div class="chip" title={cpu?.model ?? "CPU"}>
          <span>CPU</span>
          <b>{cpu ? `${cpu.utilPct.toFixed(0)}%` : "—"}</b>
          {cpu && cpu.tempC > 0 ? <em>{cpu.tempC.toFixed(0)}°</em> : null}
        </div>
        <div class="chip" title={gpu?.name ?? "No GPU"}>
          <span>GPU</span>
          <b>{gpu ? `${gpu.utilPct.toFixed(0)}%` : "—"}</b>
          {gpu && gpu.tempC > 0 ? <em>{gpu.tempC.toFixed(0)}°</em> : null}
        </div>
        <MemoryMini
          vramUsedB={vramUsedB()}
          vramTotalB={vramTotalB()}
          ramUsedB={mem?.usedB ?? 0}
          ramTotalB={mem?.totalB ?? 0}
        />
      </div>

      <div class="topbar-actions">
        <button
          type="button"
          class="btn tiny"
          t="zoom-out"
          title="Smaller text"
          onClick={() => ui.zoom(-1)}
        >
          A−
        </button>
        <button
          type="button"
          class="btn tiny"
          t="zoom-in"
          title="Larger text"
          onClick={() => ui.zoom(1)}
        >
          A+
        </button>
        <button
          type="button"
          class="btn tiny"
          t="theme"
          title="Switch theme"
          onClick={() => ui.toggleTheme()}
        >
          {ui.theme === "dark" ? "☾" : "☀"}
        </button>
      </div>
    </header>
  );
}

function Rail() {
  return (
    <nav class="rail">
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          t={`tab-${t.id}`}
          class={ui.tab === t.id ? "rail-item on" : "rail-item"}
          onClick={() => ui.go(t.id)}
        >
          <span class="rail-icon">{t.icon}</span>
          <span class="rail-label">{t.label}</span>
        </button>
      ))}
      <div class="rail-spacer" />
      <div class="rail-foot" title="Total size of installed builds and models">
        <div>{bytes(buildsSizeB())} builds</div>
        <div>{bytes(modelsSizeB())} models</div>
      </div>
    </nav>
  );
}

/** The command strip: always visible, always read-only, always the exact thing
 *  that will be spawned. */
function CommandStrip() {
  const model = currentModel();
  const server = commandLine("server", {
    bin: serverBin() || "llama-server",
    model: model?.path ?? "",
    settings: cfg.settings,
  });
  const cli = commandLine("cli", {
    bin: cliBin() || "llama-cli",
    model: model?.path ?? "",
    settings: cfg.settings,
  });
  return (
    <footer class={ui.showCommand ? "cmdstrip open" : "cmdstrip"}>
      <button
        type="button"
        class="cmdstrip-toggle"
        title="Show or hide the generated commands"
        onClick={() => ui.toggleCommand()}
      >
        {ui.showCommand ? "▾" : "▸"} command
      </button>
      {ui.showCommand
        ? (
          <div class="cmdstrip-body">
            <div class="cmdline">
              <span class="cmdline-tag">server</span>
              <code t="strip-server">{server}</code>
              <button
                type="button"
                class="btn tiny"
                onClick={() => void navigator.clipboard?.writeText(server)}
              >
                Copy
              </button>
            </div>
            <div class="cmdline">
              <span class="cmdline-tag">cli</span>
              <code t="strip-cli">{cli}</code>
              <button
                type="button"
                class="btn tiny"
                onClick={() => void navigator.clipboard?.writeText(cli)}
              >
                Copy
              </button>
            </div>
          </div>
        )
        : null}
    </footer>
  );
}

function Panel() {
  switch (ui.tab) {
    case "build":
      return <BuildPanel />;
    case "models":
      return <ModelsPanel />;
    case "settings":
      return <TunePanel />;
    case "server":
      return <ServerPanel />;
    case "chat":
      return <ChatPanel />;
    case "about":
      return <About />;
    case "dashboard":
      return <Dashboard />;
    case "cpu":
      return <CpuPanel />;
    case "gpu":
      return <GpuPanel />;
    case "memory":
      return <MemoryPanel />;
    default:
      return <OnePage />;
  }
}

export default function App() {
  return (
    <div
      class="app"
      data-theme={ui.theme}
      style={{ "--fs": `${ui.fontPx}px` }}
    >
      <Header />
      <div class="main">
        <Rail />
        <main class="content">
          <Panel />
        </main>
      </div>
      <CommandStrip />
    </div>
  );
}
