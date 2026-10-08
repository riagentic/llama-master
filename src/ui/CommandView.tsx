// src/ui/CommandView.tsx — the exact thing that will be spawned.
//
// This used to be a footer strip pinned under every tab: a full-width bar with
// its own toggle, its own type scale and its own copy buttons, stealing a band
// of height from every page to show two lines that wrapped anyway. It was the
// one part of the app that was not a panel, and it looked it.
//
// It is a section now, in the same shape as everything else — and the same
// shape as a code block in the chat (`.codeblock`), because that is exactly
// what it is: a named block of text with a button that takes it.
//
// What it must keep: "what you see is what runs". The argv here is composed by
// the same `src/lib/command.ts` that `srv.start` is handed, from the settings
// that will actually be used — the ones a running server was STARTED with while
// it runs (`derive.ts:shownSettings`), not whatever the panels have drifted to
// since.

import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import { ui } from "../cell/ui.ts";
import { srv } from "../cell/srv.ts";
import { argvBlock, commandBlock, droppedFlags } from "../lib/command.ts";
import { cliBin, serverBin } from "./actions.ts";
import {
  activeBuild,
  activeCaps,
  ctxOverride,
  envProblems,
  shownEnv,
  shownModel,
  shownSettings,
  strataCommand,
} from "./derive.ts";
import { LOCK_REASON, runLocked } from "./actions.ts";
import { CopyButton, DraftInput, Panel } from "./kit.tsx";

/** The argv of one target, as one pasteable line. */
function commandFor(target: "server" | "cli"): string[] {
  const model = shownModel();
  // Another engine: the argv Start spawns (or the running one's own), drawn
  // the same way. It has no chat CLI, so both targets show the server.
  const strata = srv.runModel && srv.argv.length > 0 && activeBuild()?.engine
    ? srv.argv
    : strataCommand(model?.path ?? "", shownSettings(), ctxOverride());
  if (strata) return argvBlock(strata, hw.osHome);
  const bin = target === "server"
    ? serverBin() || "llama-server"
    : cliBin() || "llama-cli";
  return commandBlock(target, {
    bin,
    model: model?.path ?? "",
    settings: shownSettings(),
    // The same probe `srv.start` composes against, so the preview cannot show
    // a flag the spawn will leave out (`command.ts:emitFor`).
    caps: activeCaps(),
    // The prefix the run carries — the working ones while nothing runs, the
    // running process's own while it is up (`derive.ts:shownEnv`).
    env: shownEnv(),
    // Display compaction only: `$HOME/...` reads shorter and pastes back to
    // the same absolute path. The spawned argv is untouched.
    home: hw.osHome,
  });
}

const NAME = { server: "llama-server", cli: "llama-cli" } as const;

/**
 * One target's argv: what it is, a way to take it, and the command.
 *
 * `bare` drops the block header — when the panel shows a single target, that
 * header said "llama-server" directly under a panel titled Command, and the
 * copy button moves up beside the fold toggle. One row of a narrow column is
 * worth more than a caption for something with one item in it.
 */
function CommandBlock(props: {
  target: "server" | "cli";
  t: string;
  bare?: boolean;
}) {
  const parts = commandFor(props.target);
  const name = NAME[props.target];
  // The env prefix, when present, is parts[0] — not an arg, so the count
  // subtracts it too.
  const env = shownEnv();
  const args = parts.length - 1 - (env.length > 0 ? 1 : 0);
  return (
    <div class="codeblock">
      {props.bare ? null : (
        <div class="codeblock-head">
          <span class="codeblock-name">{name}</span>
          <span class="codeblock-lang">{args} args</span>
          {
            /* Copied as ONE line: the display wraps it for reading, but what
               lands in a shell has to be a command — and the prefix leads it,
               so the paste reproduces the environment too. */
          }
          <CopyButton
            text={parts.join(" ").replace(/\s+/g, " ")}
            title={`Copy the ${name} command`}
            t={`${props.t}-copy`}
          />
        </div>
      )}
      <pre class="codeblock-body cmd-body" t={props.t}>{parts.join(" ")}</pre>
      <Dropped target={props.target} t={`${props.t}-dropped`} />
    </div>
  );
}

/**
 * Settings this build has no flag for, named under the command they are absent
 * from.
 *
 * The app asks each build what it accepts (`builds.probe`) and `command.ts`
 * leaves out anything it has never heard of — which is the only way a command
 * composed from a catalog can survive upstream REMOVING a flag, as it did with
 * `--mlock` and `--no-mmap`. But a setting that vanishes with no explanation is
 * a setting the user believes in that does not exist, which is the same failure
 * the environment box refuses to commit. So the panel says which, and why.
 *
 * Silent for an unprobed build: nothing is dropped there, so there is nothing
 * to report.
 */
function Dropped(props: { target: "server" | "cli"; t: string }) {
  if (activeBuild()?.engine) return null;
  const gone = droppedFlags(props.target, {
    settings: shownSettings(),
    caps: activeCaps(),
  });
  if (gone.length === 0) return null;
  const list = gone.map((g) => `${g.label} (${g.flag})`).join(" · ");
  return (
    <div class="cmd-dropped" t={props.t}>
      {`Left out — this llama.cpp build has no such flag: ${list}. Build or download a newer one to use ${
        gone.length === 1 ? "it" : "them"
      }.`}
    </div>
  );
}

/**
 * The user's own environment variables for llama-server, as one text input.
 *
 * Above the blocks it explains, because it changes them: `GGML_CUDA_DISABLE_GRAPHS=1`
 * in front of the command is how a shell reads it, and the prefix appears in
 * the block the moment the line is committed.
 *
 * Locked while a server runs, like every control that describes the next
 * start — the running process's own variables are what the block then shows.
 * A token that is not a `NAME=value` this app can honour is refused and NAMED
 * below the box rather than dropped: a setting that disappears silently is a
 * setting the user believes in that does not exist.
 */
function EnvVarsInput(props: { t: string }) {
  const id = props.t;
  const locked = runLocked();
  const problems = envProblems();
  return (
    <div class="cmd-env" t={`${id}-env`}>
      <label
        class="cmd-env-row"
        title="Environment variables the llama-server process is spawned with, exactly as a shell would read them — e.g. GGML_CUDA_DISABLE_GRAPHS=1. They appear as the prefix on the command below, and a copy-paste carries them."
      >
        <span class="cmd-env-label">Environment</span>
        <DraftInput
          type="text"
          class="cmd-env-input"
          ariaLabel="Environment variables for llama-server"
          placeholder="NAME=value, space-separated"
          t={`${id}-env-input`}
          disabled={locked}
          title={locked ? LOCK_REASON : undefined}
          value={cfg.envVars}
          onCommit={(v) => cfg.setEnvVars(v)}
        />
      </label>
      {problems.length > 0
        ? (
          <span class="cmd-env-bad" t={`${id}-env-bad`}>
            {problems.map((p) => `${p.token} — ${p.why}`).join(" · ")}
          </span>
        )
        : null}
    </div>
  );
}

/**
 * Both commands, in one panel.
 *
 * Rendered wherever the settings behind them can be changed — the all-in-one
 * page and the Tune page — because a command preview a tab away from the flag
 * it reflects is a preview nobody looks at. The fold is `ui.showCommand`, kept
 * from the strip and still persisted: on a narrow column a 30-flag command is
 * a screenful, and someone who does not want it should not have to scroll past
 * it every session.
 */
export function CommandPanel(props: {
  t?: string;
  /** Which binaries to show. The all-in-one page is about starting a SERVER
   *  and has one column for the machine, its memory and this — the `llama-cli`
   *  equivalent is a reference, and it is on the two pages that have room to
   *  be a reference (Tune, Server). */
  targets?: readonly ("server" | "cli")[];
}) {
  const id = props.t ?? "cmd";
  const targets = props.targets ?? ["server", "cli"];
  const only = targets.length === 1 ? targets[0] : null;
  return (
    <Panel
      title="Command"
      icon="›"
      right={
        <>
          {only && ui.showCommand
            ? (
              <CopyButton
                text={commandFor(only).join(" ").replace(/\s+/g, " ")}
                title={`Copy the ${NAME[only]} command`}
                t={`${id}-${only}-copy`}
              />
            )
            : null}
          <button
            type="button"
            class="btn tiny"
            t={`${id}-toggle`}
            title="Show or hide the generated commands"
            onClick={() => ui.toggleCommand()}
          >
            {ui.showCommand ? "Hide" : "Show"}
          </button>
        </>
      }
    >
      {ui.showCommand
        ? (
          <div class="cmd-blocks" t={id}>
            <EnvVarsInput key="env" t={id} />
            {targets.map((target) => (
              <CommandBlock
                key={target}
                target={target}
                t={`${id}-${target}`}
                bare={only !== null}
              />
            ))}
          </div>
        )
        : (
          <p class="dim cmd-hidden">
            {cfg.touched.length} setting{cfg.touched.length === 1 ? "" : "s"}
            {" "}
            changed from llama.cpp's defaults. Show to read the exact argv.
          </p>
        )}
    </Panel>
  );
}
