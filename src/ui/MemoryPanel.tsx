// src/ui/MemoryPanel.tsx — all three memory pools, in detail.
//
// RAM, VRAM and storage. Storage belongs here because it is the third thing this
// app fills: a llama.cpp source tree and a cmake build directory run to several
// GB, models to tens, and "No space left on device" is a build failure this app
// already knows how to explain — better to show the number before the build than
// to diagnose it after.
//
// The All-in-one page carries the current/projected pair for the SELECTED model.
// This page is about the machine itself, whatever is loaded.

import { hw } from "../cell/hw.ts";
import { srv } from "../cell/srv.ts";
import { bytes, pctLabel } from "../lib/format.ts";
import { BUILD_NEEDS_B, tooFullToBuild } from "../lib/disk.ts";
import { Bar, Empty, KV, Panel, Pill } from "./kit.tsx";
import { MemoryDetail } from "./MemoryDetail.tsx";
import {
  currentStatePlan,
  memoryIsLive,
  vramTotalB,
  vramUsedB,
} from "./derive.ts";

export function MemoryPanel() {
  const m = hw.mem;
  const disks = hw.disks;
  const live = memoryIsLive();
  return (
    <div class="tab-body">
      <Panel
        title="System RAM"
        icon="▤"
        wide
        right={
          <Pill tone={hw.paused ? "warn" : "ok"}>
            {hw.paused ? "paused" : "live"}
          </Pill>
        }
      >
        {!m ? <Empty icon="▢" title="No memory details available" /> : (
          <>
            <div class="kv-grid">
              <KV k="Total" v={bytes(m.totalB)} mono />
              <KV
                k="Available"
                v={`${bytes(m.availableB)} · ${
                  pctLabel(m.availableB, m.totalB)
                }`}
                mono
              />
              <KV k="Used" v={bytes(m.usedB)} mono />
              <KV
                k="Swap"
                v={m.swapTotalB > 0
                  ? `${bytes(m.swapUsedB)} of ${bytes(m.swapTotalB)} used`
                  : "none configured"}
                mono
              />
            </div>
            <Bar
              value={m.usedB}
              max={m.totalB}
              tone={m.usedB / Math.max(1, m.totalB) > 0.9 ? "bad" : "ok"}
              height={8}
            />
            <p class="param-tip">
              “Available” is the number that matters, not “free”: model weights
              and the KV cache are anonymous pages the kernel cannot reclaim, so
              filling this is the OOM killer rather than merely slow.
            </p>
          </>
        )}
      </Panel>

      <Panel
        title="VRAM"
        icon="◈"
        wide
        right={
          <Pill tone="idle">
            {bytes(vramUsedB())} / {bytes(vramTotalB())}
          </Pill>
        }
      >
        {hw.gpus.length === 0
          ? (
            <Empty
              icon="▢"
              title="No GPU detected"
              hint="Everything runs in system RAM."
            />
          )
          : (
            <div class="kv-grid">
              {hw.gpus.map((g, i) => (
                <KV
                  key={`${g.name}-${i}`}
                  k={g.name}
                  v={`${bytes(g.vramUsedB)} / ${bytes(g.vramTotalB)} · ${
                    pctLabel(g.vramUsedB, g.vramTotalB)
                  }`}
                  mono
                />
              ))}
            </div>
          )}
      </Panel>

      <Panel
        title="Storage"
        icon="▣"
        wide
        right={
          <Pill tone="idle">
            {disks.length > 0 ? `${disks.length} filesystem(s)` : "unknown"}
          </Pill>
        }
      >
        {disks.length === 0
          ? (
            <Empty
              icon="▢"
              title="Free space is not reported on this platform yet"
              hint="It comes from `df`, which this app reads on Linux and macOS."
            />
          )
          : (
            <div class="disks">
              {disks.map((d) => (
                <div class="disk" key={d.mount}>
                  <div class="disk-head">
                    <b class="mono">{d.mount}</b>
                    <span class="dim">{d.filesystem}</span>
                    {tooFullToBuild(d)
                      ? (
                        <Pill tone="warn">
                          under {bytes(BUILD_NEEDS_B)} free
                        </Pill>
                      )
                      : null}
                  </div>
                  <Bar
                    value={d.usedB}
                    max={d.totalB}
                    tone={tooFullToBuild(d) ? "warn" : "ok"}
                    height={7}
                  />
                  <div class="dim mono">
                    {bytes(d.availB)} free of {bytes(d.totalB)} ·{" "}
                    {pctLabel(d.usedB, d.totalB)} used
                  </div>
                </div>
              ))}
              <p class="param-tip">
                A source build wants around {bytes(BUILD_NEEDS_B)}{" "}
                for the checkout and the cmake tree; a prebuilt release is a few
                hundred MB. Everything this app downloads lives under its own
                directory and the cache can be deleted at any time.
              </p>
            </div>
          )}
      </Panel>

      <Panel
        title="What llama.cpp is using now"
        icon="▸"
        wide
        right={
          <Pill tone={live ? "ok" : "idle"}>
            {live ? "a model is running" : "nothing running"}
          </Pill>
        }
      >
        <MemoryDetail
          plan={currentStatePlan()}
          live={live}
          mode="current"
          rssB={srv.rssB}
        />
      </Panel>
    </div>
  );
}
