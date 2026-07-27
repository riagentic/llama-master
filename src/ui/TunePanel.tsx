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
import { hw } from "../cell/hw.ts";
import { GROUPS, PARAMS } from "../lib/params.ts";
import { devices, isEnabled, toggleDevice } from "../lib/gpu.ts";
import { bytes } from "../lib/format.ts";
import type { Param } from "../lib/types.ts";
import { plan as computePlan } from "../lib/plan.ts";
import { PLACEMENTS } from "../lib/tune.ts";
import { applyOptimal, currentStability } from "./actions.ts";
import { Empty, ErrorNote, Panel, Pill, Segmented, Toggle } from "./kit.tsx";
import { MemoryPlan } from "./Memory.tsx";
import {
  activeBuild,
  changedCount,
  currentModel,
  hwSnapshot,
  isTouched,
} from "./derive.ts";

/**
 * One checkbox per GPU the active build can address.
 *
 * The kata asks that the user be able to enable or disable any GPU. Typing
 * llama.cpp's own device names into a text box is not that: it needs the naming
 * convention, the enumeration order, and the knowledge that an empty value means
 * "all". Those are the three things this control knows so the user does not have
 * to — it shows the cards by their real names, with VRAM, and writes the flag.
 */
function DevicePicker(props: { value: string; p: Param }) {
  const backend = activeBuild()?.backend;
  const list = devices(backend, hw.gpus);
  if (list.length === 0) {
    // Three different "no devices", and they need different sentences: no card
    // at all, a build that addresses none, and no build chosen yet — in which
    // case the device NAMES are unknown, because llama.cpp derives them from
    // the backend.
    return (
      <span class="unit" t="no-devices">
        {hw.gpus.length === 0
          ? "No GPU detected — llama.cpp will run on the CPU."
          : backend === undefined
          ? "Install a llama.cpp build first — which GPUs are addressable, and what they are called, depends on its backend."
          : `A ${backend.toUpperCase()} build addresses no GPU; every layer runs on the CPU.`}
      </span>
    );
  }
  const all = list.map((d) => d.id);
  const on = list.filter((d) => isEnabled(props.value, d.id)).length;
  return (
    <div class="device-picker" t="device-picker">
      {list.map((d) => (
        <label
          key={d.id}
          class={isEnabled(props.value, d.id) ? "device on" : "device"}
          title={`${d.label} — passed to llama.cpp as ${d.id}`}
        >
          <input
            type="checkbox"
            checked={isEnabled(props.value, d.id)}
            aria-label={`${d.label} (${d.id})`}
            // The device id is the stable handle: two identical cards share a
            // name, so the name alone cannot address either of them.
            t={d.id}
            // Never let the last one go: llama.cpp with no device fails at load,
            // and "use nothing" is what -ngl 0 already says, honestly.
            disabled={on === 1 && isEnabled(props.value, d.id)}
            onChange={(e) =>
              cfg.set(
                props.p.key,
                toggleDevice(
                  props.value,
                  all,
                  d.id,
                  (e.currentTarget as HTMLInputElement).checked,
                ),
              )}
          />
          <span class="device-name">{d.label}</span>
          <span class="device-id">{d.id}</span>
          <span class="device-vram">{bytes(d.gpu.vramTotalB)}</span>
        </label>
      ))}
    </div>
  );
}

/** One control, chosen by the parameter's declared kind. */
function Control(props: { p: Param }) {
  const p = props.p;
  const value = cfg.settings[p.key] ?? p.def;
  const touched = isTouched(p.key);

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
        onChange={(e) =>
          cfg.set(p.key, (e.currentTarget as HTMLSelectElement).value)}
      >
        {(p.options ?? []).map((o) => (
          <option key={o || "(default)"} value={o}>
            {o === "" ? "(default)" : o}
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
      <p class="param-tip">{p.tip}</p>
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
        {list.map((p) => <Control key={p.key} p={p} />)}
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
          <Panel
            title="Memory plan"
            icon="▤"
            right={<Pill tone="idle">live</Pill>}
          >
            {meta
              ? (
                <MemoryPlan
                  plan={computePlan(meta, hwSnapshot(), cfg.settings)}
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
