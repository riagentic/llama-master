// src/ui/LanSwitch.tsx — how the server RUNS, as opposed to what it runs.
//
// Two switches that are not llama.cpp flags in the usual sense: who may reach
// the server, and whether it yields to the desktop. They live together because
// they answer the same kind of question — not "how should the model be
// placed", but "how should this process behave on this machine".
//
// One switch over one llama.cpp flag: `--host`, off = 127.0.0.1, on = 0.0.0.0.
// It writes through the catalog like every other setting (`cfg.set`), so the
// command strip, the argv that is spawned and this switch cannot disagree —
// there is no second code path and no second source of truth.
//
// One component, both pages (all-in-one and Tune), the same rule as
// `CtxControls` and `ReserveControls`: a setting that appears twice must be the
// same control twice.
//
// What it says when it is ON is the point. llama-server has no authentication
// unless an API key is set, so this is not "expose a port" — it is "let anyone
// on this network use my GPU". That sentence belongs on screen, next to the
// switch, not in a tooltip nobody opens. And the address it then prints is the
// one another machine actually dials: `0.0.0.0` is a bind address, and typing
// it into the client reaches nothing.

import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import { isLanExposed, lanHost, lanUrl, pickLanIp } from "../lib/lan.ts";
import { num, str } from "../lib/params.ts";
import { runLocked } from "./actions.ts";
import { serverRunning } from "./derive.ts";
import { LOCK_REASON } from "./actions.ts";
import { Segmented, Toggle } from "./kit.tsx";

export function LanSwitch(props: { t?: string }) {
  const id = props.t ?? "lan";
  const on = isLanExposed(cfg.settings);
  const locked = runLocked();
  const url = lanUrl(pickLanIp(hw.lanIps), num(cfg.settings, "port"));
  // ONE text node, built here rather than interpolated as a run of sibling
  // expressions in the JSX: a fragment of adjacent conditional strings
  // re-rendered on a state change leaves the previous sentence beside the new
  // one — it rendered "Reachable at Reachable at —" with the address missing,
  // which is the same reconciliation trap `ReserveControls` fell into.
  const note = !on
    ? ""
    : url
    ? `Reachable at ${url}`
    : "Bound to every interface — this machine reports no LAN address to dial.";
  return (
    <div class="lan-switch" t={id}>
      <Toggle
        checked={on}
        label="Available on LAN"
        t={`${id}-toggle`}
        tip={locked
          ? LOCK_REASON
          : "Bind llama-server to 0.0.0.0 so other machines on your network can reach it. Off (127.0.0.1) it answers only on this machine — which is why a client elsewhere finds nothing."}
        onChange={(v) => {
          if (locked) return;
          cfg.set("host", lanHost(v));
        }}
      />
      {
        /* The risk itself is not repeated here: an open bind with no API key
           already raises a red banner on this very page
           (`src/lib/stability.ts`), and saying it twice makes both quieter.
           What the banner cannot give is the address, which is what somebody
           reaching for the client needs. */
      }
      {note ? <span class="lan-note" t={`${id}-note`}>{note}</span> : null}
    </div>
  );
}

/**
 * Run llama-server at the lowest OS priority.
 *
 * ON by default, and that is the point: llama.cpp will take every core and
 * every spare IOPS, and on the machine it runs on that reads as "the computer
 * is broken" — the pointer stutters, the editor waits. At nice 19 in the idle
 * I/O class it gets everything nobody else wants and yields the moment anybody
 * asks; generation slows by a few percent when the machine is busy and by
 * nothing when it is not (`src/lib/priority.ts`).
 *
 * Applied to the process after the spawn, so the command on screen is still
 * the command that runs. Changing it while a server is up therefore says so
 * rather than pretending: lowering a priority needs no privileges, but RAISING
 * one back does, so the switch takes effect on the next start.
 */
/**
 * Whether a reasoning model thinks before it answers.
 *
 * The strongest speed control in the app for a thinking model, and it was
 * invisible: `--reasoning` went into the catalog as an advanced server flag,
 * filed next to Prometheus metrics, where nobody looking to make a short
 * question fast would ever find it. Most of a reasoning model's tokens go into
 * the think block, so turning it off does not make the model faster per token
 * — it removes most of the tokens, which is what the user actually feels.
 *
 * Three states, not two, because `auto` is the honest default: the model's own
 * template decides, and forcing `on` for a model with no reasoning mode is a
 * request llama.cpp cannot honour. Off is a real choice a person makes for a
 * real reason; on is for the model that would otherwise answer too quickly and
 * too shallowly.
 *
 * NOT locked while a server runs, unlike the placement controls: it changes
 * the next start, and the note says so — same rule as the priority switch
 * beside it.
 */
export function ThinkSwitch(props: { t?: string }) {
  const id = props.t ?? "think";
  const mode = str(cfg.settings, "reasoning") || "auto";
  const running = serverRunning();
  return (
    <div class="lan-switch" t={id}>
      <span class="think-label">Thinking</span>
      <Segmented
        value={mode as "auto" | "on" | "off"}
        options={[
          {
            id: "auto",
            label: "Auto",
            tip:
              "The model's own chat template decides — the right answer for almost every model.",
          },
          {
            id: "on",
            label: "On",
            tip:
              "Always think first. Better answers on hard questions; several times more tokens for the same reply.",
          },
          {
            id: "off",
            label: "Off",
            tip:
              "Skip the thinking pass on models that allow it. The biggest speed win available on a reasoning model — most of its tokens are the thinking — at the cost of depth on hard questions.",
          },
        ]}
        onChange={(v) => cfg.set("reasoning", v)}
      />
      {running
        ? (
          <span class="lan-note dim" t={`${id}-note`}>
            takes effect on the next start
          </span>
        )
        : null}
    </div>
  );
}

export function PrioritySwitch(props: { t?: string }) {
  const id = props.t ?? "prio";
  const on = cfg.lowPriority;
  const running = serverRunning();
  return (
    <div class="lan-switch" t={id}>
      <Toggle
        checked={on}
        label="Low priority"
        t={`${id}-toggle`}
        tip="Run llama-server at the lowest CPU and I/O priority (nice 19, idle I/O) so the rest of the machine stays responsive while it works. It gets everything nothing else wants — which, on an idle machine, is everything."
        onChange={() => cfg.toggleLowPriority()}
      />
      {running
        ? (
          <span class="lan-note dim" t={`${id}-note`}>
            takes effect on the next start
          </span>
        )
        : null}
    </div>
  );
}
