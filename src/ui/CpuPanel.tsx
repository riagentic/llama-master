// src/ui/CpuPanel.tsx — the CPU, in detail.
//
// The Machine page summarises; this page answers the CPU questions on their own
// terms: how many cores are there really, which of them are busy, how hot is it
// getting, and — the part that matters here — how many of them llama.cpp has
// actually been told to take.
//
// That last pairing is the reason this is a page and not a bigger card. The
// interesting fact about a CPU in this app is not its clock, it is the gap
// between what the machine has and what the current settings claim.

import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import { num } from "../lib/params.ts";
import { bytes, pctLabel } from "../lib/format.ts";
import { CPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { Bar, Empty, KV, Panel, Pill, Ring, Spark } from "./kit.tsx";
import { mappedModelB } from "./derive.ts";

/** One core's load, as a labelled bar. A grid of these is the quickest way to
 *  see whether a run is using the machine evenly or hammering two cores. */
function Core(props: { i: number; pct: number }) {
  return (
    <div class="core">
      <span class="core-n">{props.i}</span>
      <Bar value={props.pct} max={100} tone="accent" height={6} />
      <span class="core-pct">{Math.round(props.pct)}%</span>
    </div>
  );
}

export function CpuPanel() {
  const c = hw.cpu;
  const threads = num(cfg.settings, "threads");
  const threadsBatch = num(cfg.settings, "threadsBatch");
  return (
    <div class="tab-body">
      {!c
        ? (
          <Panel title="CPU" icon="⚙" wide>
            <Empty
              icon="▢"
              title="No CPU details available"
              hint="This platform does not expose them, or the first refresh has not landed."
            />
          </Panel>
        )
        : (
          <>
            <Panel
              title="CPU"
              icon="⚙"
              wide
              right={
                <Pill tone={hw.paused ? "warn" : "ok"}>
                  {hw.paused ? "paused" : "live"}
                </Pill>
              }
            >
              <div class="cpu-head">
                <Ring value={c.utilPct} label="load" tone="accent" size={72} />
                <div class="kv-grid">
                  <KV k="Model" v={c.model || "—"} />
                  <KV k="Physical cores" v={String(c.cores)} mono />
                  <KV k="Logical processors" v={String(c.threads)} mono />
                  <KV
                    k="Clock"
                    v={c.mhz > 0 ? `${(c.mhz / 1000).toFixed(2)} GHz` : "—"}
                    mono
                  />
                  <KV k="Architecture" v={`${hw.os} · ${hw.arch}`} />
                  <KV
                    k="Package temperature"
                    v={c.tempC > 0
                      ? `${c.tempC.toFixed(0)} / ${CPU_TJMAX} °C`
                      : "no sensor"}
                    mono
                  />
                </div>
              </div>
              <div class="sub-label">Load, last 60 samples</div>
              <Spark data={hw.cpuHistory} tone="accent" height={40} />
              {c.tempC > 0
                ? (
                  <Bar
                    value={c.tempC}
                    max={CPU_TJMAX}
                    tone={tempTone(c.tempC, CPU_TJMAX)}
                    height={7}
                  />
                )
                : null}
            </Panel>

            <Panel title="What llama.cpp is told to use" icon="▸" wide>
              {
                /* The comparison is the point: a thread count above the core
                   count is the single most common way to make generation
                   slower while looking like more effort. */
              }
              <div class="kv-grid">
                <KV
                  k="Generation threads (-t)"
                  v={threads > 0
                    ? `${threads} of ${c.cores} physical cores`
                    : "llama.cpp decides"}
                  mono
                />
                <KV
                  k="Prompt threads (-tb)"
                  v={threadsBatch > 0
                    ? `${threadsBatch} of ${c.cores} physical cores`
                    : "llama.cpp decides"}
                  mono
                />
                <KV
                  k="Held back for the OS"
                  v={threads > 0
                    ? `${Math.max(0, c.cores - threads)} core(s)`
                    : "—"}
                  mono
                />
                <KV
                  k="Batch / micro-batch"
                  v={`${num(cfg.settings, "batchSize")} / ${
                    num(cfg.settings, "ubatchSize")
                  }`}
                  mono
                />
              </div>
              <p class="param-tip">
                Generation is memory-bandwidth bound, so past the physical core
                count the hyper-threads mostly contend for the same bus. Two
                cores left to the OS is what keeps the window repainting while a
                model runs.
              </p>
            </Panel>

            <Panel
              title="Per-core load"
              icon="▦"
              wide
              right={
                <Pill tone="idle">
                  {c.coresUtil.length > 0
                    ? `${c.coresUtil.length} reported`
                    : "not reported"}
                </Pill>
              }
            >
              {c.coresUtil.length === 0
                ? (
                  <Empty
                    icon="▢"
                    title="Per-core load is not available here"
                    hint="It comes from /proc/stat, which only Linux provides."
                  />
                )
                : (
                  <div class="cores">
                    {c.coresUtil.map((p, i) => (
                      <Core key={String(i)} i={i} pct={p} />
                    ))}
                  </div>
                )}
            </Panel>

            {hw.mem
              ? (
                <Panel title="What it has to feed from" icon="▤" wide>
                  {
                    /* Bandwidth, not capacity, is what limits CPU inference —
                       so RAM belongs on the CPU page too, briefly. */
                  }
                  <div class="kv-grid">
                    <KV k="RAM total" v={bytes(hw.mem.totalB)} mono />
                    <KV
                      k="RAM available"
                      v={`${bytes(hw.mem.availableB)} · ${
                        pctLabel(hw.mem.availableB, hw.mem.totalB)
                      }`}
                      mono
                    />
                    {mappedModelB() > 0
                      ? (
                        <KV
                          k="Mapped model"
                          v={bytes(mappedModelB())}
                          mono
                          tip="Resident weights of the running model, memory-mapped from the file. The kernel books them as reclaimable cache, so 'available' includes them — but evicting them means re-reading from disk and generation slowing to a crawl."
                        />
                      )
                      : null}
                  </div>
                </Panel>
              )
              : null}
          </>
        )}
    </div>
  );
}
