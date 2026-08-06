// src/ui/ReserveControls.tsx — "reserve this much for me".
//
// Three numbers, asked from the two places a run is set up: the all-in-one page
// and the Tune page. One component, so they cannot disagree — the same rule as
// `CtxControls`.
//
// What it is NOT: a safety margin. Those exist already, they are fixed, and the
// user has no business tuning them (`src/lib/tune.ts:marginB`). This is the
// user's own claim on their machine — the card that also draws the desktop, the
// compile they will start while the model runs — and it is honoured by planning
// as if that memory were absent (`src/lib/reserve.ts`).
//
// Why VRAM is two boxes: the cost being defended is the DISPLAY, and a display
// lives on one card. One machine-wide figure spread over every card took memory
// from cards nobody was using for graphics; one figure applied to every card
// charged a two-card machine twice for a desktop it has once. So the user says
// which claim they mean, and the common case — 0 per card, 8 GB on the card
// with the monitor — is the default.

import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import {
  displayGpus,
  displayUnknown,
  MAX_RESERVE_GB,
  reserveGb,
  reserveOf,
  vramReserveShares,
} from "../lib/reserve.ts";
import { bytes } from "../lib/format.ts";
import { planningHw, reserveCost, vramTotalB } from "./derive.ts";

function Field(props: {
  pool: "gpu" | "connected" | "ram";
  label: string;
  unit: string;
  valueB: number;
  /** The pool this one number is charged against — one card for the VRAM
   *  figures, the whole of RAM for RAM. What it is compared to has to match
   *  what the number means, or the warning lies. */
  capacityB: number;
  t: string;
  tip: string;
}) {
  // A reserve larger than the pool is not silently ignored — it is clamped and
  // said out loud. It means every placement will report that it does not fit,
  // and the user is owed the reason rather than a mystery refusal.
  const over = props.capacityB > 0 && props.valueB > props.capacityB;
  return (
    <label class="reserve-field" title={props.tip}>
      <span class="reserve-label">{props.label}</span>
      <span class="field-inline">
        <input
          type="number"
          class="reserve-num"
          aria-label={`${props.label}, in GB`}
          t={props.t}
          min="0"
          max={String(MAX_RESERVE_GB)}
          step="1"
          value={String(reserveGb(props.valueB))}
          onInput={(e) => {
            const raw = (e.currentTarget as HTMLInputElement).value;
            // An empty box is mid-edit, not "reserve nothing". Writing 0 here
            // would hand the whole machine back the instant someone selected
            // the field to type a new number, and the plan would jump before
            // they had said anything. Typing an explicit 0 still means 0.
            if (raw.trim() === "") return;
            cfg.setReserve(props.pool, Number(raw));
          }}
        />
        <span class="unit">{props.unit}</span>
      </span>
      {over
        ? (
          <span class="reserve-over" t={`${props.t}-over`}>
            more than this machine has ({bytes(props.capacityB)}) — nothing will
            fit until you lower it
          </span>
        )
        : null}
    </label>
  );
}

/**
 * The three, plus what they cost in the terms the rest of the page uses.
 *
 * The sentence matters as much as the boxes: a user who reserves 8 GB and then
 * sees a smaller context has to be able to connect the two, and "reserved,
 * plans are made from what is left" is the whole contract in one line. It also
 * names WHICH cards the connected figure landed on — that is a reading of the
 * machine, and when the machine did not answer, the word "assuming" is the
 * difference between a fact and a guess.
 */
export function ReserveControls(props: { t?: string }) {
  const id = props.t ?? "reserve";
  const gpus = hw.gpus;
  const vramCapB = vramTotalB();
  const ramCapB = hw.mem?.totalB ?? 0;
  // The EFFECTIVE reserve — clamped to the machine, exactly as every plan on
  // this page sees it, so the summary cannot claim more is held back than is.
  const phw = planningHw();
  const held = reserveOf(phw);
  const shares = vramReserveShares(phw.gpus, held);
  const heldVramB = shares.reduce((a, b) => a + b, 0);
  const displays = displayGpus(gpus);
  const guessed = displayUnknown(gpus);
  const smallestCardB = gpus.length > 0
    ? gpus.reduce((a, g) => Math.min(a, Math.max(0, g.vramTotalB)), Infinity)
    : 0;
  const displayCardB = gpus.reduce(
    (a, g, i) => displays[i] ? Math.max(a, Math.max(0, g.vramTotalB)) : a,
    0,
  );
  const displayNames = gpus
    .map((_g, i) => displays[i] ? `GPU ${i}` : "")
    .filter(Boolean);
  const summary = heldVramB > 0 || held.ramB > 0
    ? [
      `${bytes(heldVramB)} of ${bytes(vramCapB)} VRAM and ${
        bytes(held.ramB)
      } of ${
        bytes(ramCapB)
      } reserved — every plan below is made from what is left.`,
      held.connectedB > 0 && displayNames.length > 0
        ? `Display ${displayNames.length > 1 ? "cards" : "card"}: ${
          displayNames.join(", ")
        }${guessed ? " (assumed — this machine does not report it)" : ""}.`
        : "",
      held.connectedB > 0 && displayNames.length === 0
        ? "No card reports a display attached, so the connected-GPU reserve costs nothing here."
        : "",
    ].filter(Boolean).join(" ")
    : "Nothing reserved — plans may use every byte the machine reports free.";
  // What it COSTS, in the tuner's own units. A reserve is honoured by planning
  // as if the memory were absent, which is correct and completely invisible:
  // the answer just comes back smaller, and nothing connects "my context is
  // 16k" to the 8 GB the user asked to keep. Layers and tokens, because those
  // are facts the tuner produced — a predicted tok/s here would be a guess
  // dressed as a measurement (`derive.ts:reserveCost`).
  const cost = reserveCost();
  const costNote = !cost
    ? ""
    : cost.blocks
    ? "Costing you this model entirely — it fits on this machine, but not on what is left after the reserve. Lower the numbers above."
    : [
      "Costing you",
      cost.layers > 0
        ? `${cost.layers} layer${
          cost.layers === 1 ? "" : "s"
        } of experts moved to RAM`
        : "",
      cost.layers > 0 && cost.ctxLost > 0 ? "and" : "",
      cost.ctxLost > 0
        ? `${cost.ctxLost.toLocaleString()} tokens of context (${cost.ctxWith.toLocaleString()} instead of ${
          (cost.ctxWith + cost.ctxLost).toLocaleString()
        })`
        : "",
      "— set the numbers above to 0 to get it back.",
    ].filter(Boolean).join(" ");
  return (
    <div class="reserve-controls" t={id}>
      <Field
        pool="gpu"
        label="Reserved per GPU"
        unit="GB each"
        valueB={cfg.reservePerGpuVramB}
        capacityB={smallestCardB}
        t={`${id}-gpu`}
        tip="VRAM no model may use, held back on EVERY card. For a card shared with something other than your desktop — another inference server, a render job. 0 is the right answer on a machine whose GPUs only run this app."
      />
      <Field
        pool="connected"
        label="Reserved on connected GPU"
        unit="GB"
        valueB={cfg.reserveConnectedVramB}
        capacityB={displayCardB}
        t={`${id}-connected`}
        tip="VRAM no model may use on the card(s) with a display attached — the one running your desktop, browser and games. A plan that spends the last byte of that card is a driver reset mid-generation. Held back only there, so a second, headless card keeps all of its memory."
      />
      <Field
        pool="ram"
        label="Reserved RAM"
        unit="GB"
        valueB={cfg.reserveRamB}
        capacityB={ramCapB}
        t={`${id}-ram`}
        tip="System RAM no model may use. Weights and KV cache are pages the kernel cannot reclaim, so running this pool to the edge does not mean slow, it means the OOM killer — and llama-server is the biggest process on the machine."
      />
      {
        /* ONE text node, built here rather than interpolated as a run of
           sibling expressions in the JSX. A fragment of adjacent conditional
           strings re-rendered on every keystroke, and the reconciler left the
           previous sentence in place beside the new one — the summary appeared
           twice, with two different numbers, which is the worst thing a line
           explaining a refusal can do. */
      }
      <span class="reserve-summary" t={`${id}-summary`}>{summary}</span>
      {costNote
        ? <span class="reserve-cost" t={`${id}-cost`}>{costNote}</span>
        : null}
    </div>
  );
}
