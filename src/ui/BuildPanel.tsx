// src/ui/BuildPanel.tsx — get llama.cpp, by either route, in one click.
//
// The two routes are one control, not two screens: "Prebuilt" needs nothing
// installed and takes seconds, "From source" needs a compiler and takes
// minutes. The panel says which one is available before the user commits.

import { builds } from "../cell/builds.ts";
import { useLocal } from "aio/air";
import {
  formatRef,
  parseForkInput,
  parsePrList,
  parseRef,
  refLabel,
  refPrs,
} from "../lib/srcref.ts";
import { hw } from "../cell/hw.ts";
import { availableBackends, isBinaryAsset, pickAsset } from "../lib/assets.ts";
import { SCHED_SPLIT_CAP, targetReadiness } from "../lib/backend.ts";
import type { Backend } from "../lib/types.ts";
import { bytes, duration, stamp } from "../lib/format.ts";
import {
  DraftInput,
  Empty,
  ErrorNote,
  JobProgress,
  LogView,
  Panel,
  Pill,
  Segmented,
  Toggle,
} from "./kit.tsx";
import {
  buildBusy,
  buildsSizeB,
  cudaMax,
  foundPrereqs,
  prereqById,
} from "./derive.ts";
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
    ? pickAsset(assets, os, arch, builds.backend, cudaMax())
    : null;
  const available = assets.length
    ? availableBackends(assets, os, arch, cudaMax())
    : null;

  // ONE question, asked for the exact route+backend the user has selected, and
  // asked before the button is enabled: will this produce a build?
  const ready = targetReadiness(builds.origin, builds.backend, {
    platform: os,
    arch,
    found: foundPrereqs(),
    availableBackends: available,
    assetCount: assets.length,
    explain: (id) => prereqById(id)?.why,
    // A pull request has no prebuilt download, so the release route must
    // refuse it BEFORE the button is enabled — the same promise this function
    // makes about a missing toolchain.
    ...(() => {
      const r = parseRef(builds.ref);
      return r.kind === "pr"
        ? { pr: r.pr }
        : r.kind === "fork"
        ? { fork: r.repo }
        : {};
    })(),
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
            {
              /* The current ref is always an option: a pull request or a
                fork is chosen below and is never in the tag list, and a
                select whose value has no option shows its FIRST option — a
                fork build displayed as "master (latest)". */
            }
            {(builds.refs.includes(builds.ref)
              ? builds.refs
              : [builds.ref, ...builds.refs]).map((r) => (
                <option key={r} value={r} selected={r === builds.ref}>
                  {refLabel(parseRef(r))}
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

      <PrPicker />

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
              <DraftInput
                type="number"
                min="0"
                max="512"
                ariaLabel="Parallel jobs"
                value={String(builds.jobs)}
                onCommit={(v) => builds.setJobs(Number(v))}
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
              <Toggle
                checked={builds.bypassSchedCap}
                label="Bypass llama.cpp's graph-split limit"
                tip={`Compile with GGML_SCHED_MAX_SPLIT_INPUTS=${SCHED_SPLIT_CAP} instead of the stock 30. llama.cpp's scheduler caps how many tensors one graph split may pull across a device boundary, and with routed experts in RAM that cap aborts extreme contexts — measured: 256k generates, 512k dies at the assert with memory to spare. Raised, contexts up to the model's full advertised length (1M) stop hitting it. Costs kilobytes of bookkeeping per split; source builds only.`}
                t="bypass-sched-cap"
                onChange={(v) => builds.setBypassSchedCap(v)}
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
                <option key="" value="" selected={builds.assetName === ""}>
                  {auto ? `auto — ${auto.name}` : "auto"}
                </option>
                {assets.filter((a) => isBinaryAsset(a.name)).map((a) => (
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
                  <td>
                    {b.backend}
                    {b.schedCap
                      ? (
                        <span
                          class="dim"
                          title={`Compiled with GGML_SCHED_MAX_SPLIT_INPUTS=${b.schedCap} (stock: 30) — the graph-split limit that aborts extreme contexts is raised in this build.`}
                        >
                          · split cap {b.schedCap}
                        </span>
                      )
                      : null}
                  </td>
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

/**
 * Build a pull request, on top of current master.
 *
 * The interesting models arrive as pull requests months before they merge, and
 * until now the only way to run one was to download its tarball by hand into
 * the source cache — which works once and then rots, because the next build
 * reuses it under a name that no longer describes it.
 *
 * GitHub publishes every PR already merged into master as its own ref
 * (`src/lib/srcref.ts`), so this is one number and a download. What the box
 * accepts is deliberately wide: people arrive here from a browser, so the URL
 * they copied is the commonest input and "#27754" is the second.
 *
 * Two things it must say out loud. The build is dated, because "master + PR"
 * means something different tomorrow. And a PR that has stopped merging is
 * REPORTED rather than attempted — GitHub withdraws the merged ref when the
 * branch and master disagree, and finding that out after a twenty-minute
 * compile is the experience this app exists to prevent.
 */
function PrPicker() {
  const cur = parseRef(builds.ref);
  const on = cur.kind === "pr" || cur.kind === "stack" || cur.kind === "fork";
  const [text, setText] = useLocal("");
  // One box for one pull request and for five: "27773, 28136, 27269" is how
  // people write a list, and three fields would be three chances to get the
  // ORDER wrong — which is load-bearing, because merging A then B is not
  // merging B then A when they touch the same lines.
  const typed = parsePrList(text);
  // The same box takes a FORK's URL: a model whose kernels live only in its
  // vendor's llama.cpp (PrismML's ternary types) arrives with a repository
  // link, and a second box for it would be a second place to look.
  const fork = typed.length === 0 ? parseForkInput(text) : null;
  const apply = () => {
    if (fork) builds.setRef(formatRef(fork));
    else if (typed.length === 1) builds.setPr(typed[0]!, "merge");
    else if (typed.length > 1) builds.setPrs(typed);
  };
  const prs = refPrs(cur);
  return (
    <div class="field-row">
      <label>PR or fork</label>
      <div class="pr-picker">
        <div class="field-inline">
          <input
            type="text"
            class="pr-input"
            t="pr-number"
            aria-label="Pull request number or URL, or a fork's URL"
            placeholder="PRs, e.g. 27773, 28136 — or a fork's URL"
            value={text}
            onInput={(e) =>
              setText((e.currentTarget as HTMLInputElement).value)}
            onKeyDown={(e) => {
              if ((e as KeyboardEvent).key !== "Enter") return;
              (e as KeyboardEvent).preventDefault();
              apply();
            }}
          />
          <button
            type="button"
            class="btn small primary"
            t="pr-use"
            disabled={typed.length === 0 && !fork}
            title={fork
              ? `Build ${refLabel(fork)} instead of upstream llama.cpp`
              : typed.length === 0
              ? "Type a pull request number, paste its URL, or paste a fork's URL"
              : typed.length === 1
              ? `Build llama.cpp master with pull request #${
                typed[0]
              } merged into it`
              : `Build llama.cpp master with pull requests ${
                typed.map((n) => `#${n}`).join(", ")
              } merged into it, in that order`}
            onClick={() => apply()}
          >
            {fork
              ? "Use this fork"
              : typed.length > 1
              ? `Use ${typed.length} with master`
              : "Use with master"}
          </button>
          {on
            ? (
              <button
                type="button"
                class="btn small"
                t="pr-clear"
                title="Go back to building a plain version"
                onClick={() => builds.setRef("master")}
              >
                Clear
              </button>
            )
            : null}
        </div>
        {on
          ? (
            <div class="pr-note" t="pr-note">
              <span class="pr-what">
                {cur.kind === "fork"
                  ? `Building ${
                    refLabel(cur)
                  } — a fork, not upstream llama.cpp. It runs what upstream cannot (a vendor's own kernels) and lags upstream in everything else.`
                  : cur.kind === "stack"
                  ? `Building master with ${
                    prs.map((n) => `#${n}`).join(", ")
                  } merged into it, in that order. More than one pull request cannot be merged by GitHub, so this one is assembled here and needs git.`
                  : cur.kind === "pr" && cur.mode === "head"
                  ? `Building pull request #${
                    prs[0]
                  } on its own — the author's branch, without master's recent changes.`
                  : `Building master with pull request #${
                    prs[0]
                  } merged into it.`}
              </span>
              {builds.prTitle
                ? <span class="pr-title">{builds.prTitle}</span>
                : null}
              <span class="pr-dated">
                {cur.kind === "fork"
                  ? "A fork build is dated: this is the fork as it stands today, and Update follows the fork, never upstream. Existing builds are kept, so nothing you already have is lost."
                  : "A pull request build is dated: this is master as it stands today, and rebuilding next week gives you a different one. Existing builds are kept, so nothing you already have is lost."}
              </span>
              {
                /* A pull request that has LANDED turns its own ref into an
                   older master — the trap that arrives on a good day. Said
                   here, with the answer, which is always "build master". */
              }
              {builds.prNotes.map((n) => (
                <span class="pr-merged" key={n} t="pr-merged">{n}</span>
              ))}
              {cur.kind === "pr" && cur.mode === "merge"
                ? (
                  <button
                    type="button"
                    class="btn tiny"
                    t="pr-head"
                    title="Use the author's branch alone. The escape hatch when a pull request no longer merges into master."
                    onClick={() => builds.setPr(cur.pr, "head")}
                  >
                    Use the branch alone instead
                  </button>
                )
                : cur.kind === "pr"
                ? (
                  <button
                    type="button"
                    class="btn tiny"
                    t="pr-merge"
                    onClick={() => builds.setPr(cur.pr, "merge")}
                  >
                    Try merging with master again
                  </button>
                )
                : null}
            </div>
          )
          : null}
      </div>
    </div>
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
