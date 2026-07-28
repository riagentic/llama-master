// src/ui/GpuPanel.tsx — the GPUs, in detail.
//
// One card per device, with the numbers that decide whether a model will run on
// it: VRAM, of course, but also the compute capability (which is what makes an
// older CUDA toolkit refuse to build for a newer card) and the power headroom.
//
// The device picker lives here as well as on the Tune tab. "Which of my cards
// should llama.cpp use" is a question about the cards, and this is the page
// showing them — the control is the same component, so the two cannot drift.

import { hw } from "../cell/hw.ts";
import { bytes, pctLabel } from "../lib/format.ts";
import { GPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { devices } from "../lib/gpu.ts";
import { param } from "../lib/params.ts";
import { Bar, Empty, KV, Panel, Pill, Ring, Spark } from "./kit.tsx";
import { DevicePicker } from "./DevicePicker.tsx";
import { activeBuild, vramTotalB, vramUsedB } from "./derive.ts";
import { cfg } from "../cell/cfg.ts";
import { str } from "../lib/params.ts";

export function GpuPanel() {
  const backend = activeBuild()?.backend;
  const named = devices(backend, hw.gpus);
  const deviceParam = param("device");
  return (
    <div class="tab-body">
      {hw.gpus.length === 0
        ? (
          <Panel title="GPU" icon="◈" wide>
            <Empty
              icon="▢"
              title="No GPU detected"
              hint="llama.cpp will run on the CPU — which works, just slower. A prebuilt CPU release needs nothing installed."
            />
          </Panel>
        )
        : (
          <>
            <Panel
              title="GPUs"
              icon="◈"
              wide
              right={
                <>
                  <Pill tone="idle">
                    {bytes(vramUsedB())} / {bytes(vramTotalB())}
                  </Pill>
                  <Pill tone={hw.paused ? "warn" : "ok"}>
                    {hw.paused ? "paused" : "live"}
                  </Pill>
                </>
              }
            >
              <div class="gpu-cards">
                {hw.gpus.map((g, i) => {
                  const dev = named[i];
                  return (
                    <div class="gpu-card" key={`${g.name}-${i}`}>
                      <div class="gpu-card-head">
                        <Ring
                          value={g.utilPct}
                          label="load"
                          tone="busy"
                          size={64}
                        />
                        <div>
                          <div class="gpu-name" title={g.name}>{g.name}</div>
                          <div class="dim">
                            {g.vendor}
                            {dev ? ` · llama.cpp calls it ${dev.id}` : ""}
                            {g.computeCap > 0
                              ? ` · compute ${g.computeCap.toFixed(1)}`
                              : ""}
                          </div>
                        </div>
                      </div>
                      <div class="kv-grid">
                        <KV
                          k="VRAM"
                          v={`${bytes(g.vramUsedB)} / ${
                            bytes(g.vramTotalB)
                          } · ${pctLabel(g.vramUsedB, g.vramTotalB)}`}
                          mono
                        />
                        <KV
                          k="Temperature"
                          v={g.tempC > 0
                            ? `${g.tempC.toFixed(0)} / ${GPU_TJMAX} °C`
                            : "no sensor"}
                          mono
                        />
                        <KV
                          k="Power"
                          v={g.powerLimitW > 0
                            ? `${g.powerW.toFixed(0)} / ${
                              g.powerLimitW.toFixed(0)
                            } W`
                            : "not reported"}
                          mono
                        />
                      </div>
                      <Bar
                        value={g.vramUsedB}
                        max={g.vramTotalB}
                        tone={g.vramUsedB / Math.max(1, g.vramTotalB) > 0.9
                          ? "bad"
                          : "busy"}
                        height={7}
                      />
                      {g.tempC > 0
                        ? (
                          <Bar
                            value={g.tempC}
                            max={GPU_TJMAX}
                            tone={tempTone(g.tempC, GPU_TJMAX)}
                            height={5}
                          />
                        )
                        : null}
                    </div>
                  );
                })}
              </div>
              <div class="sub-label">First GPU load, last 60 samples</div>
              <Spark data={hw.gpuHistory} tone="busy" height={40} />
            </Panel>

            <Panel title="Which GPUs llama.cpp may use" icon="▸" wide>
              {deviceParam
                ? (
                  <>
                    <DevicePicker
                      value={str(cfg.settings, "device")}
                      p={deviceParam}
                    />
                    <p class="param-tip">{deviceParam.tip}</p>
                  </>
                )
                : null}
            </Panel>

            {backend === undefined
              ? (
                <Panel title="Addressability" icon="⚠" wide>
                  <p class="param-tip">
                    No llama.cpp build is selected, so these cards have no
                    device names yet — llama.cpp derives them from the backend,
                    and a CUDA build addresses only NVIDIA cards while a Vulkan
                    build addresses all of them. Pick a build on the Build tab.
                  </p>
                </Panel>
              )
              : null}
          </>
        )}
    </div>
  );
}
