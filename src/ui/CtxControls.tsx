// src/ui/CtxControls.tsx — how long the model's memory is, in one control.
//
// Context is the setting users actually reach for, and it is asked from two
// places: the all-in-one page (where most sessions live) and the Tune page (where
// every other flag is). One component, so the two cannot disagree about what a
// band is worth or what a preset does.
//
// Three ways to say the same thing, because people arrive with different
// questions: a free number when you know what you want, the power-of-two ladder
// when you are thinking in the units models are described in, and the named bands
// when the question is "how far can THIS model actually go".

import { cfg } from "../cell/cfg.ts";
import { models } from "../cell/models.ts";
import type { ModelMeta } from "../lib/types.ts";
import {
  CTX_BANDS,
  CTX_PRESETS,
  ctxBands,
  ctxLabel,
  MIN_CTX,
} from "../lib/tune.ts";
import { LOCK_REASON, maxFor, pinMaxFor } from "./actions.ts";
import { ctxOverride } from "./derive.ts";

/**
 * The usable range, drawn.
 *
 * The bands are the point: a bar cut where quality is expected to change, with
 * a needle at what is actually configured. Seeing the needle sitting in the
 * degraded third is the fastest way to understand a setting that a number
 * alone does not explain.
 *
 * The axis is LOG₂, because that is the scale context lives on: the presets
 * double, the KV cache doubles with each step, and models describe themselves
 * in powers of two. Drawn linearly, a YaRN-stretched model (Big 65,536, Max
 * 1,048,576) was 94% "degraded" bar — technically to scale, and read by a real
 * user as "almost the whole range is bad", which is not what the bands claim.
 * On a log axis every doubling gets the same width, so Opt→Big and Big→2×Big
 * read as the equal steps they are.
 */
function CtxRange(props: { meta: ModelMeta; ctxNow: number }) {
  const b = ctxBands(props.meta);
  // 1,024 opens the axis: everything below Min is one compact "too short"
  // segment instead of an invisible sliver.
  const AXIS_MIN = 1024;
  const lg = (n: number) => Math.log2(Math.max(AXIS_MIN, n));
  const span = Math.max(lg(b.max) - lg(AXIS_MIN), 1);
  const pct = (n: number) =>
    `${Math.max(0, Math.min(100, ((lg(n) - lg(AXIS_MIN)) / span) * 100))}%`;
  const width = (from: number, to: number) =>
    `${
      Math.max(
        0,
        Math.min(100, ((lg(to) - lg(Math.max(from, AXIS_MIN))) / span) * 100),
      )
    }%`;
  const segments = [
    {
      key: "short",
      tone: "short",
      from: 0,
      to: b.min,
      what:
        `Under ${b.min.toLocaleString()} — too short to hold a conversation`,
    },
    {
      key: "full",
      tone: "full",
      from: b.min,
      to: b.opt,
      what:
        `${b.min.toLocaleString()}–${b.opt.toLocaleString()} — full quality expected (estimated)`,
    },
    {
      key: "good",
      tone: "good",
      from: b.opt,
      to: b.big,
      what:
        `${b.opt.toLocaleString()}–${b.big.toLocaleString()} — long, some quality given up (estimated)`,
    },
    {
      key: "thin",
      tone: "thin",
      from: b.big,
      to: b.max,
      what:
        `${b.big.toLocaleString()}–${b.max.toLocaleString()} — the far end of its advertised range`,
    },
  ].filter((s) => s.to > s.from);

  return (
    <div class="ctx-range" t="ctx-range">
      <div class="ctx-range-track">
        {segments.map((s) => (
          <div
            key={s.key}
            class={`ctx-band tone-${s.tone}`}
            style={{ width: width(s.from, s.to) }}
            title={s.what}
          />
        ))}
        {/* Where the current setting actually sits. */}
        <div
          class="ctx-needle"
          t="ctx-needle"
          style={{ left: pct(props.ctxNow) }}
          title={`Configured: ${props.ctxNow.toLocaleString()} tokens`}
        />
      </div>
      <div class="ctx-range-ticks">
        {CTX_BANDS.map((band) => (
          <span
            key={band.id}
            class="ctx-tick"
            style={{ left: pct(b[band.id]) }}
            title={band.tip}
          >
            {band.label}
            {band.estimated ? "≈" : ""}
          </span>
        ))}
      </div>
      <p class="param-tip">
        Log scale — every doubling is the same width. Only <b>Max</b>{" "}
        is read from the model. Min, Opt and Big are marked <b>≈</b>{" "}
        because a GGUF header carries no quality signal; they are estimated from
        published long-context measurements, not measured for this model.
      </p>
    </div>
  );
}

/**
 * Context: a number, a ladder of standard sizes, and the model's own bands.
 *
 * `Auto` is the default and is not a band — it hands the choice back to the
 * tuner, which takes the largest context the chosen placement can actually hold.
 * A pin is capped at the trained length, and a preset above it is shown disabled
 * rather than hidden: "1M exists, not for THIS model" is information, and a row
 * that changes length per model is harder to use than one that does not.
 */
export function CtxControls(
  props: {
    ctxNow: number;
    target: number;
    locked: boolean;
    meta: ModelMeta | null;
    /** Prefix for the test handles, so the two pages are addressable apart. */
    t?: string;
  },
) {
  const pinned = ctxOverride() > 0;
  const bands = props.meta ? ctxBands(props.meta) : null;
  const id = props.t ?? "ctx";
  return (
    <div class="ctx-controls">
      <div class="field-inline">
        <input
          type="number"
          class="one-ctx"
          aria-label="Context size"
          t={`${id}-value`}
          min={MIN_CTX}
          max={props.target || undefined}
          step="256"
          disabled={props.locked}
          title={props.locked
            ? LOCK_REASON
            : `Any value from ${MIN_CTX.toLocaleString()} up to the ${
              (props.target || 0).toLocaleString()
            } this model was trained for.`}
          value={String(props.ctxNow)}
          onChange={(e) =>
            cfg.setCtxOverride(
              Number((e.currentTarget as HTMLInputElement).value),
              models.selected,
            )}
        />
        <span class="unit">tokens</span>
      </div>

      <div class="ctx-bands" t={`${id}-bands`}>
        {CTX_BANDS.map((band) => {
          const n = bands?.[band.id] ?? 0;
          return (
            <button
              key={band.id}
              type="button"
              class={`btn tiny${props.ctxNow === n && pinned ? " on" : ""}`}
              t={`${id}-${band.id}`}
              disabled={props.locked || n === 0}
              title={n === 0
                ? "Select a model with a readable header first"
                : `${band.label} CTX size — ${n.toLocaleString()} tokens. ${band.tip}`}
              onClick={() => cfg.setCtxOverride(n, models.selected)}
            >
              {band.label} CTX
              {band.estimated ? <span class="est">≈</span> : null}
            </button>
          );
        })}
        <button
          type="button"
          class={`btn tiny${pinned ? "" : " on"}`}
          t={`${id}-optimal`}
          disabled={props.locked || props.target === 0}
          title={props.target > 0
            ? `Let the tuner choose: the largest context this placement can hold, up to the ${props.target.toLocaleString()} tokens this model was trained for.`
            : "Select a model with a readable header first"}
          onClick={() => cfg.setCtxOverride(0)}
        >
          Auto
        </button>
        {
          /* The priority most sessions actually have, as one click each:
            THIS placement, at the biggest context it can hold. The decision
            walk is "VRAM only? no — hybrid? no — CPU", and these two buttons
            are its first two questions, answered with the number attached. */
        }
        {(["vram", "hybrid"] as const).map((pl) => {
          const t = props.meta ? maxFor(pl) : null;
          const ok = t !== null && t.possible && t.ctx > 0;
          const label = pl === "vram" ? "Max·VRAM" : "Max·Hybrid";
          return (
            <button
              key={pl}
              type="button"
              class={`btn tiny${
                ok && pinned && props.ctxNow === t.ctx &&
                  cfg.placement === pl
                  ? " on"
                  : ""
              }`}
              t={`${id}-max-${pl}`}
              disabled={props.locked || !ok}
              title={props.locked
                ? LOCK_REASON
                : !props.meta
                ? "Select a model with a readable header first"
                : ok
                ? `${
                  pl === "vram" ? "VRAM only" : "Hybrid"
                } at the largest context it can hold here: ${t.ctx.toLocaleString()} tokens. Sets the placement and pins the context in one click.`
                : `${pl === "vram" ? "VRAM only" : "Hybrid"}: ${
                  t?.blocker || "not possible for this model here"
                }`}
              onClick={() => pinMaxFor(pl)}
            >
              {label}
              {ok ? <span class="dim">{` ${ctxLabel(t.ctx)}`}</span> : null}
            </button>
          );
        })}
      </div>

      <div class="ctx-presets" t={`${id}-presets`}>
        {CTX_PRESETS.map((n) => {
          const tooBig = props.target > 0 && n > props.target;
          return (
            <button
              key={String(n)}
              type="button"
              class={`btn tiny${props.ctxNow === n && pinned ? " on" : ""}`}
              t={`${id}-${ctxLabel(n)}`}
              disabled={props.locked || tooBig}
              title={props.locked
                ? LOCK_REASON
                : tooBig
                ? `This model was trained for ${props.target.toLocaleString()} tokens — past that, answers degrade rather than improve.`
                : `Set the context to ${n.toLocaleString()} tokens`}
              onClick={() => cfg.setCtxOverride(n, models.selected)}
            >
              {ctxLabel(n)}
            </button>
          );
        })}
      </div>

      {
        /* Some limits are llama.cpp's own and no plan arithmetic can see them:
          DeepSeek-V4 on two cards generates at a pinned 262,144 and dies at
          524,288 on a scheduler assert, with memory to spare. What the app
          DOES know is what has been measured to work — so a pin beyond it is
          flagged before Start, not discovered minutes into a load. */
      }
      {(() => {
        const proven = cfg.fitCtx[models.selected] ?? 0;
        return pinned && proven > 0 && props.ctxNow > proven
          ? (
            <p class="warn-note ctx-beyond" t="ctx-beyond-measured">
              Beyond the largest size measured to work here ({proven
                .toLocaleString()}{" "}
              tokens). The plan can price the memory, but llama.cpp also has
              internal limits it only reveals by refusing — expect the start
              itself to have the final say.
            </p>
          )
          : null;
      })()}
      {props.meta ? <CtxRange meta={props.meta} ctxNow={props.ctxNow} /> : null}
    </div>
  );
}
