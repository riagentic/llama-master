// src/ui/About.tsx — the About page.
//
// Deliberately small. The one thing it adds beyond the credits is the
// environment block: when someone reports a problem, "what version of what, on
// what" is the first question, and it should be readable off one screen rather
// than assembled from four tabs.

import { hw } from "../cell/hw.ts";
import { ABOUT } from "../lib/about.ts";
import { activeBuild, buildsSizeB, modelsSizeB } from "./derive.ts";
import { bytes } from "../lib/format.ts";
import { KV, Panel } from "./kit.tsx";

export function About() {
  const build = activeBuild();
  const cpu = hw.cpu;
  return (
    <div class="tab-body about">
      <Panel title="About" icon="ⓘ" wide>
        <div class="about-head">
          <span class="about-mark">◆</span>
          <div>
            <h1 class="about-name">{ABOUT.name}</h1>
            <p class="about-tagline">{ABOUT.tagline}</p>
          </div>
        </div>

        <div class="kv-grid">
          <KV k="Version" v={ABOUT.version} mono />
          <KV k="Built by" v={ABOUT.author} />
          <KV k="License" v={ABOUT.license} />
          <KV
            k="GitHub"
            v={
              <a href={ABOUT.repo} target="_blank" rel="noreferrer">
                {ABOUT.repo.replace("https://", "")}
              </a>
            }
          />
        </div>

        <p class="about-note">
          llama.master is a front end. The engine is{" "}
          <a href={ABOUT.upstream} target="_blank" rel="noreferrer">
            llama.cpp
          </a>{" "}
          by Georgi Gerganov and its contributors — all inference, and every
          binary this app builds or installs, is theirs.
        </p>
      </Panel>

      <Panel title="This machine" icon="▦" wide>
        {/* The first thing anyone will be asked for in a bug report. */}
        <div class="kv-grid">
          <KV k="OS" v={`${hw.os || "—"} · ${hw.arch || "—"}`} />
          <KV k="CPU" v={cpu?.model || "—"} />
          <KV
            k="Cores"
            v={cpu ? `${cpu.cores} physical · ${cpu.threads} logical` : "—"}
          />
          <KV k="Memory" v={hw.mem ? bytes(hw.mem.totalB) : "—"} />
          <KV
            k="GPUs"
            v={hw.gpus.length === 0
              ? "none detected"
              : hw.gpus.map((g) => g.name).join(", ")}
          />
          <KV
            k="Active build"
            v={build
              ? `${build.ref} · ${build.backend} · ${build.origin}`
              : "none"}
          />
          <KV k="Builds on disk" v={bytes(buildsSizeB())} mono />
          <KV k="Models on disk" v={bytes(modelsSizeB())} mono />
        </div>
      </Panel>
    </div>
  );
}
