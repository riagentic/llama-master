// src/ui/Dashboard.tsx — the machine: what it is, and what it is doing.
//
// Reads `hw` directly. Every value is either measured or shown as "—"; there
// are no placeholder numbers, because a dashboard that guesses is worse than
// one that admits it cannot see a sensor.

import { hw } from "../cell/hw.ts";
import { prereq } from "../cell/prereq.ts";
import { CPU_TJMAX, GPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { describe, scriptPreview } from "../lib/fixplan.ts";
import { fixablePrereqs, fixPlanFor } from "./derive.ts";
import { bytes, duration, pctLabel, stamp } from "../lib/format.ts";
import {
  Bar,
  Empty,
  ErrorNote,
  Grid,
  KV,
  LogView,
  Panel,
  Pill,
  Ring,
  Spark,
  Stat,
} from "./kit.tsx";

/** Thermal ceilings used as the "max" end of the temperature bars. These are
 *  the throttle points, not the destruction points: a modern x86 package
 *  throttles around 95 °C and a consumer GPU around 83 °C, so the bar filling
 *  up means "about to lose clocks", which is the useful reading. */

/** Temperature as a bar against its throttle point — the kata asks for
 *  current/max, and a bare number gives no sense of how close that is. */
function TempBar(props: { label: string; c: number; max: number }) {
  if (!(props.c > 0)) {
    return (
      <KV k={props.label} v="—" tip="No sensor exposed by this hardware" />
    );
  }
  return (
    <div class="tempbar">
      <div class="tempbar-head">
        <span>{props.label}</span>
        <b class={`t-${tempTone(props.c, props.max)}`}>
          {props.c.toFixed(0)} / {props.max} °C
        </b>
      </div>
      <Bar
        value={props.c}
        max={props.max}
        tone={tempTone(props.c, props.max)}
        height={5}
      />
    </div>
  );
}
const mhz = (m: number) =>
  m > 0
    ? (m >= 1000 ? `${(m / 1000).toFixed(2)} GHz` : `${Math.round(m)} MHz`)
    : "—";

function CpuPanel() {
  const c = hw.cpu;
  return (
    <Panel
      title="CPU"
      icon="▤"
      right={c ? <Pill tone="idle">{c.threads} threads</Pill> : null}
    >
      {!c
        ? (
          <Empty
            title="No CPU telemetry"
            hint="Reading /proc/cpuinfo failed."
          />
        )
        : (
          <>
            <div class="cpu-top">
              <Ring value={c.utilPct} label="load" tone="accent" />
              <div class="cpu-facts">
                <div class="cpu-model" title={c.model}>
                  {c.model || "Unknown CPU"}
                </div>
                <div class="cpu-kv">
                  <KV
                    k="Cores"
                    v={`${c.cores} physical · ${c.threads} logical`}
                  />
                  <KV k="Clock" v={mhz(c.mhz)} />
                  <TempBar label="Package temp" c={c.tempC} max={CPU_TJMAX} />
                </div>
              </div>
            </div>
            <Spark data={hw.cpuHistory} tone="accent" height={34} />
            {c.coresUtil.length > 0
              ? (
                <div class="cores">
                  {c.coresUtil.map((u, i) => (
                    <div
                      class="core"
                      key={String(i)}
                      title={`core ${i}: ${u.toFixed(0)}%`}
                    >
                      <div
                        class="core-fill"
                        style={{ height: `${Math.max(2, u).toFixed(0)}%` }}
                      />
                    </div>
                  ))}
                </div>
              )
              : null}
          </>
        )}
    </Panel>
  );
}

function GpuPanel() {
  const gpus = hw.gpus;
  return (
    <Panel
      title="GPU"
      icon="◨"
      right={
        <Pill tone={gpus.length ? "ok" : "warn"}>{gpus.length || "none"}</Pill>
      }
    >
      {gpus.length === 0
        ? (
          <Empty
            icon="◌"
            title="No GPU detected"
            hint="CPU-only inference still works — the tuner will plan for it."
          />
        )
        : (
          <div class="gpus">
            {gpus.map((g, i) => (
              <div class="gpu" key={`${g.name}-${i}`}>
                <div class="gpu-head">
                  <span class="gpu-name" title={g.name}>{g.name}</span>
                  <Pill tone="idle">{g.vendor}</Pill>
                </div>
                <div class="gpu-rings">
                  <Ring
                    value={g.utilPct}
                    label="util"
                    tone="accent"
                    size={54}
                  />
                  <Ring
                    value={(g.vramUsedB / Math.max(1, g.vramTotalB)) * 100}
                    label="vram"
                    tone="busy"
                    size={54}
                  />
                  <div class="gpu-kv">
                    <KV
                      k="VRAM"
                      v={`${bytes(g.vramUsedB)} / ${bytes(g.vramTotalB)}`}
                    />
                    <TempBar label="Temp" c={g.tempC} max={GPU_TJMAX} />
                    <KV
                      k="Power"
                      v={g.powerW > 0
                        ? `${g.powerW.toFixed(0)} W${
                          g.powerLimitW > 0
                            ? ` / ${g.powerLimitW.toFixed(0)} W`
                            : ""
                        }`
                        : "—"}
                    />
                  </div>
                </div>
                <Bar
                  value={g.vramUsedB}
                  max={g.vramTotalB}
                  tone={g.vramUsedB / Math.max(1, g.vramTotalB) > 0.9
                    ? "bad"
                    : "busy"}
                  height={8}
                />
              </div>
            ))}
            <Spark data={hw.gpuHistory} tone="busy" height={30} />
          </div>
        )}
    </Panel>
  );
}

function MemoryPanel() {
  const m = hw.mem;
  return (
    <Panel title="Memory" icon="▥">
      {!m ? <Empty title="No memory telemetry" /> : (
        <>
          <Grid cols={3}>
            <Stat label="Total" value={bytes(m.totalB)} />
            <Stat
              label="Available"
              value={bytes(m.availableB)}
              tone={m.availableB < m.totalB * 0.1 ? "bad" : "ok"}
            />
            <Stat label="Used" value={pctLabel(m.usedB, m.totalB)} />
          </Grid>
          <Bar
            value={m.usedB}
            max={m.totalB}
            tone={m.usedB / Math.max(1, m.totalB) > 0.9 ? "bad" : "ok"}
            height={10}
          />
          {m.swapTotalB > 0
            ? (
              <>
                <div class="sub-label">
                  Swap {bytes(m.swapUsedB)} / {bytes(m.swapTotalB)}
                </div>
                <Bar
                  value={m.swapUsedB}
                  max={m.swapTotalB}
                  tone="warn"
                  height={5}
                />
              </>
            )
            : null}
        </>
      )}
    </Panel>
  );
}

/** One "Fix" per unmet prerequisite. The tooltip is the exact command that
 *  will run — nothing privileged happens without the user seeing it first. */
function FixButton(props: { id: string; found: boolean }) {
  if (props.found) return null;
  const plan = fixPlanFor(props.id);
  if (!plan) return null;
  const busy = prereq.fixing !== "" || prereq.install !== null;
  if (plan.kind === "manual") {
    // "We will not do this for you" always comes with where to read how.
    return (
      <span class="manual-fix">
        <span class="dim" title={plan.reason}>manual</span>
        {plan.docsUrl
          ? (
            <a
              href={plan.docsUrl}
              target="_blank"
              rel="noreferrer"
              title={plan.reason}
            >
              docs ↗
            </a>
          )
          : null}
      </span>
    );
  }
  return (
    <span class="script-fix">
      <button
        type="button"
        class="btn tiny"
        t={`fix-${props.id}`}
        title={describe(plan)}
        disabled={busy}
        onClick={() => prereq.fix(props.id)}
      >
        {prereq.fixing === props.id
          ? "Fixing…"
          : plan.kind === "script"
          ? `Install (${plan.steps.length} steps)`
          : "Fix"}
      </button>
      {plan.kind === "script"
        ? (
          <a
            href={plan.docsUrl}
            target="_blank"
            rel="noreferrer"
            title={plan.docsUrl}
          >
            docs ↗
          </a>
        )
        : null}
    </span>
  );
}

function PrereqPanel() {
  const items = prereq.items;
  const inst = prereq.install;
  const fixable = fixablePrereqs();
  // A scripted install (ROCm) changes the machine in ways a package install
  // does not — add a repository, install a driver — so every command is on
  // screen before the button is pressed, not only in the log afterwards.
  const script = fixable.map((f) => fixPlanFor(f.id)).find((p) =>
    p?.kind === "script"
  );
  const scriptSteps = script ? scriptPreview(script) : [];
  const scriptTitle = script?.kind === "script" ? script.title : "";
  return (
    <Panel
      title="Prerequisites"
      icon="✓"
      right={
        <>
          {fixable.length > 0
            ? (
              <button
                type="button"
                class="btn small primary"
                t="fix-all"
                title={`Install ${fixable.map((f) => f.label).join(", ")}`}
                disabled={prereq.fixing !== "" || prereq.install !== null}
                onClick={() => prereq.fixAll()}
              >
                {prereq.fixQueue.length > 0
                  ? `Fixing ${
                    fixable.length - prereq.fixQueue.length
                  }/${fixable.length}…`
                  : `Fix all (${fixable.length})`}
              </button>
            )
            : null}
          <button
            type="button"
            class="btn small"
            onClick={() => prereq.scan()}
            disabled={prereq.scanning}
          >
            {prereq.scanning ? "Scanning…" : "Re-check"}
          </button>
        </>
      }
    >
      <ErrorNote message={prereq.lastError} />
      {inst
        ? (
          <div class="install-progress">
            <div class="sub-label">{inst.label}</div>
            <Bar
              value={inst.received}
              max={inst.total ?? Math.max(1, inst.received)}
              tone="busy"
              height={8}
            />
          </div>
        )
        : null}
      <table class="table" t="prereq-table">
        <tbody>
          {items.map((p) => (
            <tr
              key={p.id}
              class={p.found ? "" : p.systemOnly ? "row-warn" : "row-bad"}
            >
              <td class="c-icon">{p.found ? "●" : "○"}</td>
              <td class="c-name" title={p.why}>{p.label}</td>
              <td class="c-ver mono" title={p.path}>
                {p.found ? p.version || "found" : "not found"}
                {p.managed ? <Pill tone="accent">app</Pill> : null}
              </td>
              <td class="c-act">
                <FixButton id={p.id} found={p.found} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {scriptSteps.length > 0
        ? (
          <details class="steps-preview" t="script-steps">
            <summary>
              {scriptTitle} — {scriptSteps.length}{" "}
              steps, shown before anything runs
            </summary>
            <pre class="cmd">{scriptSteps.join("\n\n")}</pre>
          </details>
        )
        : null}
      {prereq.fixLog.length > 0
        ? (
          <>
            <div class="sub-label">Installer output</div>
            <LogView lines={prereq.fixLog} t="fix-log" rows={10} />
          </>
        )
        : null}
      <div class="hint">
        “Fix” installs through your package manager and will ask for your
        password. Items marked “manual” need a decision the app should not make
        for you. Nothing here is required for a prebuilt release — that path
        needs no toolchain at all.
      </div>
    </Panel>
  );
}

export function Dashboard() {
  return (
    <div class="tab-body">
      <ErrorNote message={hw.lastError} />
      <div class="dash-head">
        <Stat label="OS" value={`${hw.os || "—"} · ${hw.arch || "—"}`} />
        <Stat label="Last sample" value={stamp(hw.lastRefresh)} />
        <Stat
          label="Sampler"
          value={hw.paused ? "paused" : "1 s"}
          tone={hw.paused ? "warn" : "ok"}
        />
        <Stat
          label="Uptime of app"
          value={duration(hw.cpuHistory.length * 1000)}
        />
        <div class="dash-actions">
          <button
            type="button"
            class="btn small"
            onClick={() => hw.refresh(true)}
          >
            Refresh
          </button>
          <button
            type="button"
            class="btn small"
            onClick={() => hw.togglePause()}
          >
            {hw.paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>
      <div class="cols">
        <CpuPanel />
        <GpuPanel />
        <MemoryPanel />
        <PrereqPanel />
      </div>
    </div>
  );
}
