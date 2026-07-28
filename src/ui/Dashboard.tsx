// src/ui/Dashboard.tsx — the machine: what it is, and what it is doing.
//
// Reads `hw` directly. Every value is either measured or shown as "—"; there
// are no placeholder numbers, because a dashboard that guesses is worse than
// one that admits it cannot see a sensor.

import { hw } from "../cell/hw.ts";
import { prereq } from "../cell/prereq.ts";
import { ui } from "../cell/ui.ts";
import { CPU_TJMAX, GPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { tooFullToBuild } from "../lib/disk.ts";
import { fixablePrereqs } from "./derive.ts";
import { bytes, duration, pctLabel, stamp } from "../lib/format.ts";
import {
  Bar,
  Empty,
  ErrorNote,
  Grid,
  KV,
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

/**
 * Storage — the third pool this app fills, and the one that fails a build.
 *
 * A llama.cpp checkout plus a CUDA build tree runs to several GB and models to
 * tens, so "how full is this disk" belongs on the page that summarises the
 * machine — not only on the Memory page. The Build tab already refuses when the
 * headroom is gone; this is where the user sees it coming.
 */
function DiskPanel() {
  const disks = hw.disks;
  return (
    <Panel title="Storage" icon="▣">
      {disks.length === 0
        ? <Empty title="Free space is not reported on this platform" />
        : (
          <>
            {disks.map((d) => (
              <div class="disk" key={d.mount}>
                <div class="disk-head">
                  <b class="mono">{d.mount}</b>
                  {tooFullToBuild(d)
                    ? <Pill tone="warn">tight for a build</Pill>
                    : null}
                </div>
                <Bar
                  value={d.usedB}
                  max={d.totalB}
                  tone={tooFullToBuild(d) ? "warn" : "ok"}
                  height={7}
                />
                <div class="dim mono">
                  {bytes(d.availB)} free of {bytes(d.totalB)}
                </div>
              </div>
            ))}
          </>
        )}
    </Panel>
  );
}

/** One "Fix" per unmet prerequisite. The tooltip is the exact command that
 *  will run — nothing privileged happens without the user seeing it first. */

/**
 * Software, in one line, with the way to act on it.
 *
 * The kata asks Machine for a summary of the hardware AND software; the doing
 * lives on its own page. So this counts what is present, names what is missing,
 * and hands over — a summary that silently hid two missing tools would be worse
 * than no summary.
 */
function PrereqSummary() {
  const items = prereq.items;
  const found = items.filter((i) => i.found).length;
  const missing = items.filter((i) => !i.found);
  const fixable = fixablePrereqs().length;
  return (
    <Panel
      title="Software"
      icon="✓"
      right={
        <Pill tone={missing.length === 0 ? "ok" : "warn"}>
          {items.length > 0 ? `${found} of ${items.length}` : "not scanned"}
        </Pill>
      }
    >
      {items.length === 0
        ? <Empty title="Prerequisites have not been scanned yet" />
        : (
          <>
            <div class="kv-grid">
              <KV k="Present" v={String(found)} mono />
              <KV
                k="Missing"
                v={missing.length > 0
                  ? missing.map((i) => i.label || i.id).join(", ")
                  : "none"}
                tip={missing.length > 0
                  ? "A prebuilt release needs none of these; only a source build does."
                  : undefined}
              />
            </div>
            <button
              type="button"
              class={fixable > 0 ? "btn small primary" : "btn small"}
              t="go-prereq"
              onClick={() => ui.go("prereq")}
            >
              {fixable > 0
                ? `Prerequisites — ${fixable} can be installed for you`
                : "Prerequisites"}
            </button>
          </>
        )}
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
        <DiskPanel />
        <PrereqSummary />
      </div>
    </div>
  );
}
