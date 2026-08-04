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
import { ui } from "../cell/ui.ts";
import { commandBlock } from "../lib/command.ts";
import { cliBin, serverBin } from "./actions.ts";
import { shownModel, shownSettings } from "./derive.ts";
import { CopyButton, Panel } from "./kit.tsx";

/** The argv of one target, as one pasteable line. */
function commandFor(target: "server" | "cli"): string[] {
  const model = shownModel();
  const bin = target === "server"
    ? serverBin() || "llama-server"
    : cliBin() || "llama-cli";
  return commandBlock(target, {
    bin,
    model: model?.path ?? "",
    settings: shownSettings(),
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
  return (
    <div class="codeblock">
      {props.bare ? null : (
        <div class="codeblock-head">
          <span class="codeblock-name">{name}</span>
          <span class="codeblock-lang">{parts.length - 1} args</span>
          {
            /* Copied as ONE line: the display wraps it for reading, but what
               lands in a shell has to be a command. */
          }
          <CopyButton
            text={parts.join(" ").replace(/\s+/g, " ")}
            title={`Copy the ${name} command`}
            t={`${props.t}-copy`}
          />
        </div>
      )}
      <pre class="codeblock-body cmd-body" t={props.t}>{parts.join(" ")}</pre>
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
