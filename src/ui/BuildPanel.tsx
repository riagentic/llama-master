// src/ui/BuildPanel.tsx — get llama.cpp, by either route, in one click.
//
// The two routes are one control, not two screens: "Prebuilt" needs nothing
// installed and takes seconds, "From source" needs a compiler and takes
// minutes. The panel says which one is available before the user commits.

import { builds } from "../cell/builds.ts";
import { hw } from "../cell/hw.ts";
import { availableBackends, pickAsset } from "../lib/assets.ts";
import { targetReadiness } from "../lib/backend.ts";
import type { Backend } from "../lib/types.ts";
import { bytes, duration, stamp } from "../lib/format.ts";
import {
  Empty,
  ErrorNote,
  JobProgress,
  LogView,
  Panel,
  Pill,
  Segmented,
  Toggle,
} from "./kit.tsx";
import { buildBusy, buildsSizeB, foundPrereqs, prereqById } from "./derive.ts";
import { optimalForThisPc } from "./actions.ts";
import { Guidance } from "./Guidance.tsx";

const BACKENDS: readonly { id: Backend; label: string; tip: string }[] = [
  { id: "cpu", label: "CPU", tip: "Portable, no GPU runtime needed." },
  { id: "cuda", label: "CUDA", tip: "NVIDIA. Fastest where available." },
  {
    id: "vulkan",
    label: "Vulkan",
    tip: "Any modern GPU — AMD, Intel, NVIDIA.",
  },
  { id: "hip", label: "ROCm", tip: "AMD's native stack on Linux." },
  { id: "metal", label: "Metal", tip: "Apple silicon." },
];

function Chooser() {
  const source = builds.origin === "source";
  const os = hw.os || "linux";
  const arch = hw.arch || "x86_64";
  const assets = builds.assets;
  const auto = assets.length
    ? pickAsset(assets, os, arch, builds.backend)
    : null;
  const available = assets.length ? availableBackends(assets, os, arch) : null;

  // ONE question, asked for the exact route+backend the user has selected, and
  // asked before the button is enabled: will this produce a build?
  const ready = targetReadiness(builds.origin, builds.backend, {
    platform: os,
    arch,
    found: foundPrereqs(),
    availableBackends: available,
    assetCount: assets.length,
    explain: (id) => prereqById(id)?.why,
  });
  const canBuild = ready.ok;

  return (
    <div class="chooser">
      <div class="field-row">
        <label>Route</label>
        <Segmented
          value={builds.origin}
          options={[
            {
              id: "release",
              label: "Prebuilt release",
              tip: "No toolchain required.",
            },
            {
              id: "source",
              label: "Build from source",
              tip: "Needs CMake and a C++ compiler.",
            },
          ]}
          onChange={(v) => builds.setOrigin(v)}
        />
      </div>

      <div class="field-row">
        <label>Version</label>
        <div class="field-inline">
          <select
            aria-label="Version"
            value={builds.ref}
            onChange={(e) =>
              builds.setRef((e.currentTarget as HTMLSelectElement).value)}
          >
            {
              /* `selected`, not just `value` on the select: the tag list loads
                async, and a value applied before its option exists leaves the
                browser showing whichever option happens to be first — a pinned
                tag rendered as "master (latest)". Same bug 0.1.2 fixed on the
                Tune page dropdowns. */
            }
            {(builds.refs.length ? builds.refs : [builds.ref]).map((r) => (
              <option key={r} value={r} selected={r === builds.ref}>
                {r === "master" ? "master (latest)" : r}
              </option>
            ))}
          </select>
          <button
            type="button"
            class="btn small"
            onClick={() => builds.loadRefs()}
            disabled={builds.refsLoading}
          >
            {builds.refsLoading ? "Loading…" : "Fetch tags"}
          </button>
        </div>
      </div>

      <div class="field-row">
        <label>Backend</label>
        <Segmented
          value={builds.backend}
          options={BACKENDS}
          onChange={(v) => builds.setBackend(v)}
        />
      </div>

      {source
        ? (
          <div class="field-row">
            <label>Compile</label>
            <div class="field-inline">
              <input
                type="number"
                min="0"
                max="512"
                aria-label="Parallel jobs"
                value={String(builds.jobs)}
                onInput={(e) =>
                  builds.setJobs(
                    Number((e.currentTarget as HTMLInputElement).value),
                  )}
              />
              <span
                class="unit"
                title="0 = auto: every logical CPU but two, so the desktop stays usable while it compiles"
              >
                jobs (0 = auto → {Math.max(1, (hw.cpu?.threads ?? 4) - 2)} of
                {" "}
                {hw.cpu?.threads ?? "?"})
              </span>
              <Toggle
                checked={builds.native}
                label="-march=native"
                tip="Tune for THIS CPU. Faster here, not portable to another machine."
                onChange={(v) => builds.setNative(v)}
              />
            </div>
          </div>
        )
        : (
          <div class="field-row">
            <label>Asset</label>
            <div class="field-inline">
              <select
                aria-label="Asset"
                value={builds.assetName}
                onChange={(e) =>
                  builds.setAsset((e.currentTarget as HTMLSelectElement).value)}
              >
                {
                  /* `selected` for the same async-options reason as the Version
                    select above. */
                }
                <option value="" selected={builds.assetName === ""}>
                  {auto ? `auto — ${auto.name}` : "auto"}
                </option>
                {assets.map((a) => (
                  <option
                    key={a.name}
                    value={a.name}
                    selected={a.name === builds.assetName}
                  >
                    {a.name}
                    {a.sizeB > 0 ? ` · ${bytes(a.sizeB)}` : ""}
                  </option>
                ))}
              </select>
              <button
                type="button"
                class="btn small"
                onClick={() => builds.loadAssets()}
                disabled={builds.assetsLoading}
              >
                {builds.assetsLoading ? "Loading…" : "Refresh list"}
              </button>
            </div>
          </div>
        )}

      <div class="field-row">
        <label />
        <div class="field-inline">
          <button
            type="button"
            class="btn primary"
            t="get-llama"
            disabled={buildBusy() || !canBuild}
            title={canBuild
              ? undefined
              : ready.pending
              ? "Checking what is available…"
              : ready.diagnosis?.reason}
            onClick={() => builds.start()}
          >
            {source ? "Build llama.cpp" : "Install llama.cpp"}
          </button>
          {buildBusy()
            ? (
              <button type="button" class="btn" onClick={() => builds.cancel()}>
                Cancel
              </button>
            )
            : null}
          <button
            type="button"
            class="btn small"
            t="optimal-build"
            onClick={() => optimalForThisPc()}
            title="Backend for the detected hardware, -march=native for this exact CPU, and a job count that leaves the machine usable"
          >
            Optimal for this PC
          </button>
        </div>
      </div>

      {
        /* One banner, for the exact route+backend selected — asked before the
          button is enabled, so green ticks never precede a failure. */
      }
      {ready.pending
        ? (
          <div class="hint" t="checking">
            Checking which prebuilt builds exist for {os}/{arch}…
          </div>
        )
        : ready.diagnosis
        ? <Guidance diagnosis={ready.diagnosis} t="not-ready" />
        : (
          <div class="ready-note" t="ready">
            ✓ Ready — {source
              ? "every tool this build needs is installed"
              : "a prebuilt binary exists for this machine"}.
          </div>
        )}
    </div>
  );
}

function Installed() {
  const list = builds.installed;
  return (
    <Panel
      title="Installed builds"
      icon="▣"
      // A seven-column table has no business in a one-third-width grid track;
      // it wants the whole row like the chooser above it.
      wide
      right={
        <>
          <Pill tone="idle">{bytes(buildsSizeB())}</Pill>
          <button
            type="button"
            class="btn small"
            onClick={() => builds.scan()}
            disabled={builds.scanning}
          >
            Rescan
          </button>
        </>
      }
    >
      {list.length === 0
        ? (
          <Empty
            icon="▢"
            title="No llama.cpp yet"
            hint="Install a prebuilt release above — it needs nothing else on this machine."
          />
        )
        : (
          <table class="table" t="builds-table">
            <thead>
              <tr>
                <th />
                <th>Version</th>
                <th>Backend</th>
                <th>Origin</th>
                <th>Installed</th>
                <th>Size</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {list.map((b) => (
                <tr
                  key={b.id}
                  class={b.id === builds.activeId
                    ? "row-active row-pick"
                    : "row-pick"}
                  t={`build-row-${b.id}`}
                  title={`Use ${b.ref} · ${b.backend}`}
                  onClick={() => builds.setActive(b.id)}
                >
                  <td class="c-icon">
                    <input
                      type="radio"
                      aria-label={`Use ${b.id}`}
                      checked={b.id === builds.activeId}
                      onChange={() => builds.setActive(b.id)}
                    />
                  </td>
                  <td class="mono">{b.ref}</td>
                  <td>{b.backend}</td>
                  <td>{b.origin}</td>
                  <td>{stamp(b.createdAt)}</td>
                  <td class="mono">{bytes(b.sizeB)}</td>
                  <td class="c-act">
                    <button
                      type="button"
                      class="btn tiny danger"
                      title={`Delete ${b.dir}`}
                      onClick={(e) => {
                        // Without this the click also selects the row it is in.
                        e.stopPropagation();
                        builds.remove(b.id);
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </Panel>
  );
}

export function BuildPanel() {
  const job = builds.job;
  return (
    <div class="tab-body">
      <ErrorNote
        message={builds.lastError}
        onDismiss={() => builds.clearLog()}
      />
      <div class="cols">
        <Panel title="Get llama.cpp" icon="⚒" wide>
          <Chooser />
          {job
            ? (
              <div class="job-block">
                <div class="job-head">
                  <b>{job.label}</b>
                  <Pill
                    tone={job.status === "done"
                      ? "ok"
                      : job.status === "failed"
                      ? "bad"
                      : job.status === "cancelled"
                      ? "warn"
                      : "busy"}
                  >
                    {job.status}
                  </Pill>
                  <span class="dim">
                    {duration((job.endedAt ?? Date.now()) - job.startedAt)}
                  </span>
                </div>
                <JobProgress
                  steps={job.steps}
                  step={job.step}
                  progress={job.progress}
                  status={job.status}
                />
                {job.status === "failed" && builds.diagnosis
                  ? (
                    <Guidance
                      diagnosis={builds.diagnosis}
                      tone="error"
                      t="build-failed"
                    />
                  )
                  : job.error
                  ? <ErrorNote message={job.error} />
                  : null}
              </div>
            )
            : null}
          {builds.log.length > 0
            ? <LogView lines={builds.log} t="build-log" rows={16} />
            : null}
        </Panel>
        <Installed />
      </div>
    </div>
  );
}
