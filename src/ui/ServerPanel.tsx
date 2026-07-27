// src/ui/ServerPanel.tsx — start it, stop it, see that it is really up.
//
// "Running" here means the poll found the pid alive AND /health answered — not
// that a start button was pressed. The distinction is the difference between a
// status light and a decoration.

import { srv } from "../cell/srv.ts";
import { cfg } from "../cell/cfg.ts";
import { duration, stamp } from "../lib/format.ts";
import { commandBlock } from "../lib/command.ts";
import {
  cliBin,
  endpoint,
  serverBin,
  startBlocker,
  startServer,
  stopServer,
} from "./actions.ts";
import { Empty, ErrorNote, KV, LogView, Panel, Pill } from "./kit.tsx";
import { Guidance } from "./Guidance.tsx";
import { activeBuild, currentModel, serverRunning } from "./derive.ts";

/**
 * Memory held by llama-servers this app is not running.
 *
 * A model stays resident until its process exits, so "free the VRAM" and "stop
 * the process" are the same act. A crash or a killed app leaves one behind, and
 * the next Start then dies with CUDA out-of-memory on a machine that looks
 * idle — so this is loud, and one button fixes it.
 */
export function OrphanBanner() {
  const orphans = srv.orphans;
  if (orphans.length === 0) return null;
  return (
    <div class="error-note guide" t="orphans">
      <div class="guide-reason">
        {orphans.length === 1
          ? "A llama-server from an earlier session is still running and holding its memory."
          : `${orphans.length} llama-servers from earlier sessions are still running and holding their memory.`}
      </div>
      <ul class="guide-steps">
        {orphans.map((o) => (
          <li class="guide-step" key={String(o.pid)}>
            <span class="mono" title={o.argv}>
              pid {o.pid} · {o.argv.split(" -m ")[1]?.split(" ")[0]?.split("/")
                .pop() ?? "unknown model"}
            </span>
          </li>
        ))}
        <li class="guide-step">
          <span>
            Until they exit, their VRAM and RAM stay allocated and a new server
            cannot load.
          </span>
          <button
            type="button"
            class="btn tiny primary"
            t="free-memory"
            disabled={srv.freeing}
            onClick={() => srv.freeMemory()}
          >
            {srv.freeing ? "Stopping…" : "Free memory"}
          </button>
        </li>
      </ul>
    </div>
  );
}

export function StatusPill() {
  const s = srv.status;
  const tone = s === "ready"
    ? "ok"
    : s === "starting" || s === "stopping"
    ? "busy"
    : s === "crashed"
    ? "bad"
    : "idle";
  const label = s === "ready"
    ? "running"
    : s === "starting"
    ? (srv.pid ? "loading model" : "starting")
    : s;
  return <Pill tone={tone} title={srv.healthDetail}>{label}</Pill>;
}

function Props() {
  const p = srv.props;
  if (!p) return null;
  const model = String(p.model_path ?? "");
  const ctx = p.n_ctx ?? p.default_generation_settings ?? null;
  return (
    <div class="kv-grid">
      <KV k="Loaded model" v={model || "—"} mono tip={model} />
      <KV
        k="Server context"
        v={typeof ctx === "number" ? ctx.toLocaleString() : "—"}
        tip="What llama-server actually allocated, not what we asked for"
      />
      <KV k="Chat template" v={String(p.chat_template ? "embedded" : "—")} />
    </div>
  );
}

export function ServerPanel() {
  const blocker = startBlocker();
  const running = serverRunning();
  const model = currentModel();
  const build = activeBuild();
  const url = endpoint();

  return (
    <div class="tab-body">
      <OrphanBanner />
      {srv.diagnosis && srv.status === "crashed"
        ? <Guidance diagnosis={srv.diagnosis} tone="error" t="srv-failed" />
        : <ErrorNote message={srv.lastError} />}
      <div class="cols">
        <Panel
          title="llama-server"
          icon="⏻"
          wide
          right={
            <>
              <StatusPill />
              {srv.orphans.length > 0 && !running
                ? (
                  <button
                    type="button"
                    class="btn"
                    t="unload-all"
                    title="Stop every llama-server this app started and release its memory"
                    disabled={srv.freeing}
                    onClick={() => srv.freeMemory()}
                  >
                    {srv.freeing ? "Unloading…" : "Unload models"}
                  </button>
                )
                : null}
              {running
                ? (
                  <button
                    type="button"
                    class="btn danger"
                    t="stop-server"
                    onClick={() => stopServer()}
                  >
                    Stop server
                  </button>
                )
                : (
                  <button
                    type="button"
                    class="btn primary"
                    t="start-server"
                    disabled={blocker !== ""}
                    title={blocker || `Start ${serverBin()}`}
                    onClick={() => startServer()}
                  >
                    Start server
                  </button>
                )}
            </>
          }
        >
          {blocker ? <div class="warn-note">{blocker}</div> : null}
          <div class="kv-grid">
            <KV k="Build" v={build ? `${build.ref} · ${build.backend}` : "—"} />
            <KV k="Model" v={model?.file ?? "—"} tip={model?.path} />
            <KV
              k="Endpoint"
              v={running
                ? <a href={url} target="_blank" rel="noreferrer">{url}</a>
                : url}
              mono
            />
            <KV k="PID" v={srv.pid || "—"} mono />
            <KV k="Started" v={srv.startedAt ? stamp(srv.startedAt) : "—"} />
            <KV
              k="Uptime"
              v={srv.startedAt && running
                ? duration(Date.now() - srv.startedAt)
                : "—"}
            />
            <KV
              k="Health"
              v={srv.healthy
                ? <span class="ok-text">ready</span>
                : <span class="dim">{srv.healthDetail || "—"}</span>}
            />
            <KV
              k="Exit code"
              v={srv.exitCode === null ? "—" : String(srv.exitCode)}
            />
          </div>
          <Props />
        </Panel>

        <Panel
          title="Command"
          icon="›"
          right={
            <button
              type="button"
              class="btn tiny"
              title="Copy the server command"
              onClick={() => {
                void navigator.clipboard?.writeText(
                  commandBlock("server", {
                    bin: serverBin() || "llama-server",
                    model: model?.path ?? "",
                    settings: cfg.settings,
                  }).join(" ").replace(/\s+/g, " "),
                );
              }}
            >
              Copy
            </button>
          }
        >
          <pre class="cmd" t="server-command">
            {commandBlock("server", {
              bin: serverBin() || "llama-server",
              model: model?.path ?? "",
              settings: cfg.settings,
            }).join(" \\\n")}
          </pre>
          <div class="sub-label">llama-cli equivalent</div>
          <pre class="cmd dim" t="cli-command">
            {commandBlock("cli", {
              bin: cliBin() || "llama-cli",
              model: model?.path ?? "",
              settings: cfg.settings,
            }).join(" \\\n")}
          </pre>
        </Panel>

        <ServerLog rows={18} />
      </div>
    </div>
  );
}

/**
 * llama-server's own output.
 *
 * Exported because every diagnosis this app writes ends with some form of "the
 * log below" — so the log has to be below, on whichever page the diagnosis was
 * shown. The All-in-one page renders the same component for that reason.
 */
export function ServerLog(props: { rows?: number }) {
  return (
    <Panel
      title="Server log"
      icon="≡"
      wide
      right={
        <button
          type="button"
          class="btn tiny"
          onClick={() => srv.clearLog()}
          t="clear-log"
        >
          Clear
        </button>
      }
    >
      {srv.log.length === 0
        ? (
          <Empty
            icon="≡"
            title="No output yet"
            hint="Start the server to see its log."
          />
        )
        : <LogView lines={srv.log} t="server-log" rows={props.rows ?? 18} />}
    </Panel>
  );
}
