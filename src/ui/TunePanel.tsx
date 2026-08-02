// src/ui/TunePanel.tsx — every llama.cpp setting, with the consequence next to
// it.
//
// The controls are generated from the catalog in src/lib/params.ts, so a flag
// cannot exist in the command line without appearing here with its explanation,
// and cannot be added here without the command builder knowing about it.
//
// The memory plan sits beside the controls and recomputes on every keystroke —
// changing the context and watching the VRAM bar move is the whole point of the
// panel.

import { cfg } from "../cell/cfg.ts";
import { GROUPS, num, PARAMS } from "../lib/params.ts";
import { DevicePicker } from "./DevicePicker.tsx";
import type { Param } from "../lib/types.ts";
import { plan as computePlan } from "../lib/plan.ts";
import { pinnedCtx, PLACEMENTS, trainedCtx } from "../lib/tune.ts";
import { applyOptimal, currentStability, runLocked } from "./actions.ts";
import { Empty, ErrorNote, Panel, Pill, Segmented, Toggle } from "./kit.tsx";
import { MemoryPlan } from "./Memory.tsx";
import { CtxControls } from "./CtxControls.tsx";
import {
  changedCount,
  ctxOverride,
  currentModel,
  isTouched,
  paramBlocker,
  planningHw,
} from "./derive.ts";

/** One control, chosen by the parameter's declared kind.
 *
 *  Exported because the all-in-one page carries the whole catalog too, and two
 *  renderings of the same flag would be two chances to disagree with the command
 *  builder. */
export function ParamControl(props: { p: Param }) {
  const p = props.p;
  const value = cfg.settings[p.key] ?? p.def;
  const touched = isTouched(p.key);
  // Some flags are only meaningful for some models — offering one this model
  // cannot honour is a load failure with the app's name on it.
  const blocker = paramBlocker(p.key);

  const control = p.kind === "bool"
    ? (
      <Toggle
        checked={value === true}
        label=""
        tip={p.tip}
        onChange={(v) => cfg.set(p.key, v)}
      />
    )
    : p.kind === "enum"
    ? (
      <select
        aria-label={p.label}
        value={String(value)}
        disabled={blocker !== ""}
        title={blocker || undefined}
        onChange={(e) =>
          cfg.set(p.key, (e.currentTarget as HTMLSelectElement).value)}
      >
        {
          /* `selected` on the option, not only `value` on the select. In a real
            client the select showed options[0] whatever the state said — Flash
            attention rendered "auto" while `-fa on` was in the command below it.
            The in-process test harness does NOT reproduce it, so this is pinned
            by a surface assertion and reported upstream. The model and build
            pickers on the all-in-one page have always used this form, which is
            why they were never wrong. */
        }
        {(p.options ?? []).map((o, i) => (
          <option
            key={o || "(default)"}
            value={o}
            selected={String(value) === o}
          >
            {p.optionLabels?.[i] ?? (o === "" ? "(default)" : o)}
          </option>
        ))}
      </select>
    )
    : p.kind === "devices"
    ? <DevicePicker value={String(value)} p={p} />
    : p.kind === "text"
    ? (
      <input
        type="text"
        aria-label={p.label}
        placeholder={p.unit ?? ""}
        value={String(value)}
        onInput={(e) =>
          cfg.set(p.key, (e.currentTarget as HTMLInputElement).value)}
      />
    )
    : (
      <input
        type="number"
        aria-label={p.label}
        min={p.min}
        max={p.max}
        step={p.step ?? (p.kind === "float" ? 0.01 : 1)}
        value={String(value)}
        onInput={(e) =>
          cfg.set(p.key, (e.currentTarget as HTMLInputElement).value)}
      />
    );

  return (
    <div class={touched ? "param touched" : "param"}>
      <div class="param-head">
        <span class="param-label" title={p.tip}>{p.label}</span>
        <code class="param-flag" title={`Emitted as ${p.flag}`}>{p.flag}</code>
        {touched
          ? (
            <button
              type="button"
              class="btn tiny"
              title={`Reset to the llama.cpp default (${String(p.def)})`}
              onClick={() => cfg.resetOne(p.key)}
            >
              ↺
            </button>
          )
          : null}
      </div>
      <div class="param-control">
        {control}
        {p.unit && p.kind !== "text"
          ? <span class="unit">{p.unit}</span>
          : null}
      </div>
      <p class="param-tip">{blocker || p.tip}</p>
    </div>
  );
}

function Group(props: { id: string; label: string }) {
  const list = PARAMS.filter(
    (p) => p.group === props.id && (cfg.advanced || !p.advanced),
  );
  if (list.length === 0) return null;
  const hidden =
    PARAMS.filter((p) => p.group === props.id && p.advanced).length;
  return (
    <Panel
      title={props.label}
      right={!cfg.advanced && hidden > 0
        ? (
          <Pill tone="idle" title="Turn on Advanced to see these">
            {`+${hidden}`}
          </Pill>
        )
        : null}
    >
      <div class="params">
        {list.map((p) => <ParamControl key={p.key} p={p} />)}
      </div>
    </Panel>
  );
}

/** The kata's rule: the user may change anything, but a change that will hurt
 *  has to say so before the server starts. */
function StabilityNote() {
  const st = currentStability();
  if (st.level === "ok") return null;
  return (
    <div
      class={st.level === "risk"
        ? "error-note stability"
        : "warn-note stability"}
      t="stability"
    >
      <div class="stability-head">
        {st.level === "risk"
          ? "⚠ This configuration will probably fail"
          : "⚠ This will run, but not well"}
      </div>
      <ul class="stability-list">
        {st.warnings.map((w) => (
          <li key={`${w.key}:${w.message}`}>
            <code>{w.key}</code> {w.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function TunePanel() {
  const m = currentModel();
  const meta = m?.meta ?? null;
  // The pin ceiling is the ADVERTISED length; the auto-tuner's native-first
  // aim stays its own business (see the same note in OnePage).
  const target = meta ? trainedCtx(meta) : 0;
  // The same clamp the tuner applies, so the number shown is the number that
  // would run (`pinnedCtx`, src/lib/tune.ts).
  const ctxNow = pinnedCtx(
    ctxOverride() || num(cfg.settings, "ctxSize"),
    target,
  );
  return (
    <div class="tab-body">
      <div class="tune-head">
        <div class="tune-model">
          {m
            ? (
              <>
                <span class="dim">Tuning</span>
                <b title={m.path}>{m.file}</b>
                {meta ? <Pill tone="accent">{meta.quant}</Pill> : null}
              </>
            )
            : <span class="dim">No model selected</span>}
        </div>
        <div class="tune-actions">
          {/* One set of optimal settings; the choice is WHERE it runs. */}
          <Segmented
            value={cfg.placement}
            options={PLACEMENTS}
            onChange={(p) => {
              cfg.setPlacement(p);
              applyOptimal();
            }}
          />
          <Toggle
            checked={cfg.autoOptimal}
            label="Optimal automatically"
            tip="Re-tune for the selected model every time the server starts. Turn it off to keep settings you have tuned by hand."
            t="auto-optimal"
            onChange={() => cfg.toggleAutoOptimal()}
          />
          <Toggle
            checked={cfg.advanced}
            label="Advanced"
            tip="Show the rarely-needed flags"
            onChange={() => cfg.toggleAdvanced()}
          />
          <Pill tone={changedCount() ? "accent" : "idle"}>
            {changedCount()} changed
          </Pill>
          <button
            type="button"
            class="btn primary"
            t="optimal"
            disabled={!meta}
            title={meta
              ? "Compute the best settings for this model on this machine"
              : "Select a model with a readable header first"}
            onClick={() => applyOptimal()}
          >
            Optimal settings
          </button>
          <button type="button" class="btn" onClick={() => cfg.reset()}>
            Reset all
          </button>
        </div>
      </div>

      <div class="tune-cols">
        <div class="tune-plan">
          <StabilityNote />
          {
            /* Context is the setting people actually reach for, so it gets the
              same control here as on the all-in-one page — bands, presets and
              the usable range — rather than being one number lost among 49. */
          }
          <Panel title="Context" icon="⇥">
            <CtxControls
              ctxNow={ctxNow}
              target={target}
              locked={runLocked()}
              meta={meta}
              t="tune-ctx"
            />
          </Panel>
          <Panel
            title="Memory plan"
            icon="▤"
            right={<Pill tone="idle">live</Pill>}
          >
            {meta
              ? (
                <MemoryPlan
                  plan={computePlan(meta, planningHw(), cfg.settings)}
                />
              )
              : (
                <Empty
                  icon="▢"
                  title="Nothing to plan"
                  hint="Select a model in the Models tab and the bars will fill in."
                />
              )}
          </Panel>
          {cfg.reasons.length > 0
            ? (
              <Panel
                title="Why these settings"
                icon="✦"
                right={
                  <button
                    type="button"
                    class="btn tiny"
                    onClick={() => cfg.clearReasons()}
                  >
                    ✕
                  </button>
                }
              >
                <ol class="reasons" t="reasons">
                  {cfg.reasons.map((r) => <li key={r}>{r}</li>)}
                </ol>
              </Panel>
            )
            : null}
        </div>
        <div class="tune-params">
          <ErrorNote
            message={m && !meta
              ? `Model header unreadable: ${m.metaError}`
              : ""}
          />
          {GROUPS.map((g) => <Group key={g.id} id={g.id} label={g.label} />)}
        </div>
      </div>
    </div>
  );
}
