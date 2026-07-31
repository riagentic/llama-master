// src/ui/StoragePage.tsx — disks, and what this app is doing to them.
//
// Storage earns a page because llama.master is unusually heavy on it and the
// failure is unusually annoying: a llama.cpp checkout plus a CUDA build tree is
// several GB, a release archive a few hundred MB, and models tens of GB. Running
// out happens minutes into a compile, after the wait.
//
// Two questions, and the second is the one the Memory page cannot answer: how
// full is each filesystem, and **which of that is ours**. A user deciding what to
// delete needs the second one.

import { hw } from "../cell/hw.ts";
import { builds } from "../cell/builds.ts";
import { models } from "../cell/models.ts";
import { bytes, pctLabel } from "../lib/format.ts";
import { BUILD_NEEDS_B, tooFullToBuild } from "../lib/disk.ts";
import { Bar, Empty, ErrorNote, KV, Panel, Pill, Stat } from "./kit.tsx";
import { buildsSizeB, modelsSizeB } from "./derive.ts";

/** Every filesystem this app writes to, with the build-headroom warning on the
 *  ones that cannot take a source build. */
function Filesystems() {
  const disks = hw.disks;
  return (
    <Panel
      title="Filesystems"
      icon="▣"
      wide
      right={
        <Pill tone="idle">
          {disks.length > 0 ? `${disks.length} mounted` : "unknown"}
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
                  height={9}
                />
                <div class="dim mono">
                  {bytes(d.availB)} free of {bytes(d.totalB)} ·{" "}
                  {pctLabel(d.usedB, d.totalB)} used
                </div>
              </div>
            ))}
          </div>
        )}
    </Panel>
  );
}

/**
 * What llama.master itself occupies — the part a user can actually act on.
 *
 * `df` says a disk is full; it does not say that 40 GB of it is three llama.cpp
 * builds you stopped using. Each row is something that can be deleted, and says
 * where it lives so it can be deleted by hand too.
 */
function OurFootprint() {
  const bSize = buildsSizeB();
  const mSize = modelsSizeB();
  return (
    <Panel title="What llama.master is using" icon="▸" wide>
      <div class="kv-grid">
        <Stat
          label="Installed builds"
          value={bytes(bSize)}
          sub={`${builds.installed.length} build(s)`}
        />
        <Stat
          label="Models found"
          value={bytes(mSize)}
          sub={`${models.items.length} file(s)`}
        />
        <Stat label="Together" value={bytes(bSize + mSize)} />
      </div>
      <div class="kv-grid">
        {
          /* The REAL paths, from `paths()` — it honours LLAMA_MASTER_HOME, so a
            hardcoded `~/.llama-master` here could name a directory that does
            not exist on this install. */
        }
        <KV
          k="Builds live in"
          v={hw.appPaths?.builds ?? "…"}
          mono
          tip="Everything the app installed or compiled. Deleting one frees its whole tree."
        />
        <KV
          k="Downloads and source trees"
          v={hw.appPaths?.cache ?? "…"}
          mono
          tip="Regenerable: archives, checkouts, cmake output. Safe to delete at any time — the app re-fetches what it needs."
        />
        <KV
          k="Models live where you keep them"
          v={models.dirs.length > 0
            ? models.dirs.join(" · ")
            : "no search paths configured"}
          mono
          tip="llama.master never moves or copies a model; these are the directories it scans."
        />
      </div>
      <p class="param-tip">
        A source build wants around {bytes(BUILD_NEEDS_B)}{" "}
        free for the checkout and the cmake tree; a prebuilt release is a few
        hundred MB. The cache can be deleted at any time — installed builds and
        your models cannot be re-created by the app.
      </p>
    </Panel>
  );
}

export function StoragePage() {
  return (
    <div class="tab-body">
      <ErrorNote message={hw.lastError} />
      <ErrorNote message={hw.diskError} />
      <Filesystems />
      <OurFootprint />
    </div>
  );
}
