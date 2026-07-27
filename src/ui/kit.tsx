// src/ui/kit.tsx — the visual vocabulary.
//
// Every panel is built from these, so a bar means the same thing everywhere and
// the whole app changes shape from one file. Presentational only: nothing here
// reads a cell or dispatches — data comes in as props, intent goes out as
// callbacks, which is what makes each piece renderable in a test in isolation.

import type { JSX } from "aio/jsx-runtime";
import { bytes as fmtBytes, pct as pctOf } from "../lib/format.ts";
import { useStickyBottom } from "./sticky.ts";

export type Tone = "ok" | "warn" | "bad" | "idle" | "busy" | "accent";

// ── containers ─────────────────────────────────────────────────────────────

export function Panel(props: {
  title: string;
  icon?: string;
  right?: JSX.Node;
  wide?: boolean;
  children?: JSX.Node;
}) {
  return (
    <section class={props.wide ? "panel panel-wide" : "panel"}>
      <header class="panel-head">
        <h2 t={`panel-${props.title}`}>
          {props.icon ? <span class="panel-icon">{props.icon}</span> : null}
          {props.title}
        </h2>
        <div class="panel-actions">{props.right}</div>
      </header>
      <div class="panel-body">{props.children}</div>
    </section>
  );
}

export function Grid(props: { cols?: number; children?: JSX.Node }) {
  return (
    <div class="grid" style={{ "--cols": String(props.cols ?? 2) }}>
      {props.children}
    </div>
  );
}

/** Label/value line — the densest honest way to show a fact. */
export function KV(
  props: { k: string; v: JSX.Node; tip?: string; mono?: boolean },
) {
  return (
    <div class="kv" title={props.tip}>
      <span class="kv-k">{props.k}</span>
      <span class={props.mono ? "kv-v mono" : "kv-v"}>{props.v}</span>
    </div>
  );
}

export function Stat(props: {
  label: string;
  value: JSX.Node;
  sub?: JSX.Node;
  tone?: Tone;
}) {
  return (
    <div class={`stat tone-${props.tone ?? "idle"}`}>
      <div class="stat-label">{props.label}</div>
      <div class="stat-value">{props.value}</div>
      {props.sub ? <div class="stat-sub">{props.sub}</div> : null}
    </div>
  );
}

export function Pill(
  props: { tone?: Tone; title?: string; children?: JSX.Node },
) {
  return (
    <span class={`pill tone-${props.tone ?? "idle"}`} title={props.title}>
      {props.children}
    </span>
  );
}

export function Empty(
  props: { icon?: string; title: string; hint?: JSX.Node },
) {
  return (
    <div class="empty">
      <div class="empty-icon">{props.icon ?? "○"}</div>
      <div class="empty-title">{props.title}</div>
      {props.hint ? <div class="empty-hint">{props.hint}</div> : null}
    </div>
  );
}

/** The one place an error is allowed to be shown — and it is never hidden. */
export function ErrorNote(props: { message: string; onDismiss?: () => void }) {
  if (!props.message) return null;
  return (
    <div class="error-note" t="error">
      <span class="error-icon">⚠</span>
      <span class="error-text">{props.message}</span>
      {props.onDismiss
        ? (
          <button type="button" class="x" onClick={props.onDismiss}>
            ✕
          </button>
        )
        : null}
    </div>
  );
}

// ── bars and gauges ────────────────────────────────────────────────────────

/** A single-value bar. `max` of 0 renders an empty track rather than NaN. */
export function Bar(props: {
  value: number;
  max: number;
  tone?: Tone;
  label?: JSX.Node;
  height?: number;
}) {
  const p = pctOf(props.value, props.max);
  return (
    <div class="bar-wrap">
      <div class="bar-track" style={{ height: `${props.height ?? 6}px` }}>
        <div
          class={`bar-fill tone-${props.tone ?? "accent"}`}
          style={{ width: `${p.toFixed(2)}%` }}
        />
      </div>
      {props.label ? <div class="bar-label">{props.label}</div> : null}
    </div>
  );
}

export type Segment = {
  key: string;
  label: string;
  bytes: number;
  tone: Tone | "weights" | "experts" | "kv" | "compute" | "other";
};

/**
 * The memory bar: capacity as the track, each consumer as a segment, and the
 * overflow drawn OUTSIDE the track in red so "does not fit" is visible at a
 * glance instead of being clipped away.
 */
export function StackBar(props: {
  capacityB: number;
  segments: Segment[];
  overB?: number;
  height?: number;
}) {
  const cap = props.capacityB;
  const over = props.overB ?? 0;
  return (
    <div class="stack">
      <div class="stack-track" style={{ height: `${props.height ?? 14}px` }}>
        {props.segments
          .filter((s) => s.bytes > 0)
          .map((s) => (
            <div
              key={s.key}
              class={`stack-seg seg-${s.tone}`}
              style={{ width: `${pctOf(s.bytes, cap).toFixed(3)}%` }}
              title={`${s.label}: ${fmtBytes(s.bytes)}`}
            />
          ))}
        {over > 0
          ? (
            <div
              class="stack-seg seg-over"
              style={{
                width: `${Math.min(100, pctOf(over, cap)).toFixed(3)}%`,
              }}
              title={`Over capacity by ${fmtBytes(over)}`}
            />
          )
          : null}
      </div>
    </div>
  );
}

export function Legend(props: { segments: Segment[] }) {
  return (
    <div class="legend">
      {props.segments
        .filter((s) => s.bytes > 0)
        .map((s) => (
          <span class="legend-item" key={s.key}>
            <i class={`legend-dot seg-${s.tone}`} />
            {s.label}
            <b>{fmtBytes(s.bytes)}</b>
          </span>
        ))}
    </div>
  );
}

/** Circular gauge for a 0-100 reading. Percent maps 1:1 onto the dash array. */
export function Ring(props: {
  value: number;
  label: string;
  sub?: string;
  tone?: Tone;
  size?: number;
}) {
  const size = props.size ?? 62;
  const v = Math.max(0, Math.min(100, props.value));
  return (
    <div class={`ring-wrap tone-${props.tone ?? "accent"}`} title={props.label}>
      <svg class="ring" viewBox="0 0 36 36" width={size} height={size}>
        <circle
          class="ring-track"
          cx="18"
          cy="18"
          r="15.915"
          fill="none"
          stroke-width="3"
        />
        <circle
          class="ring-fill"
          cx="18"
          cy="18"
          r="15.915"
          fill="none"
          stroke-width="3"
          stroke-linecap="round"
          stroke-dasharray={`${v.toFixed(1)} ${(100 - v).toFixed(1)}`}
          stroke-dashoffset="25"
        />
      </svg>
      <div class="ring-text">
        <b>{Math.round(v)}</b>
        <small>{props.sub ?? "%"}</small>
      </div>
      <div class="ring-label">{props.label}</div>
    </div>
  );
}

/** History sparkline. Points are normalized to 0-100 on the y axis. */
export function Spark(props: { data: number[]; tone?: Tone; height?: number }) {
  const h = props.height ?? 28;
  const d = props.data;
  if (d.length < 2) {
    return <div class="spark-empty" style={{ height: `${h}px` }} />;
  }
  const step = 100 / (d.length - 1);
  const points = d
    .map((v, i) =>
      `${(i * step).toFixed(2)},${
        (100 - Math.max(0, Math.min(100, v))).toFixed(2)
      }`
    )
    .join(" ");
  return (
    <svg
      class={`spark tone-${props.tone ?? "accent"}`}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{ height: `${h}px` }}
    >
      <polyline class="spark-area" points={`0,100 ${points} 100,100`} />
      <polyline class="spark-line" points={points} />
    </svg>
  );
}

// ── job progress ───────────────────────────────────────────────────────────

/** Stepper + bar for a long job. An unknown total shows an indeterminate bar —
 *  honest about not knowing, still visibly alive. */
export function JobProgress(props: {
  steps: string[];
  step: number;
  progress: number | null;
  status: "running" | "done" | "failed" | "cancelled";
}) {
  const tone: Tone = props.status === "failed"
    ? "bad"
    : props.status === "done"
    ? "ok"
    : props.status === "cancelled"
    ? "warn"
    : "busy";
  return (
    <div class="job">
      <ol class="steps">
        {props.steps.map((label, i) => (
          <li
            key={label}
            class={i < props.step
              ? "step done"
              : i === props.step
              ? "step now"
              : "step"}
          >
            <i class="step-dot" />
            {label}
          </li>
        ))}
      </ol>
      <div class="bar-track" style={{ height: "8px" }}>
        <div
          class={props.progress === null && props.status === "running"
            ? `bar-fill tone-${tone} indeterminate`
            : `bar-fill tone-${tone}`}
          style={{
            width: props.progress === null
              ? "100%"
              : `${(props.progress * 100).toFixed(1)}%`,
          }}
        />
      </div>
    </div>
  );
}

/** Scrolling log tail. Newest at the bottom, capped upstream. */
export function LogView(props: { lines: string[]; t?: string; rows?: number }) {
  // Newest at the bottom, and the app's diagnoses all end with "the log below" —
  // so a new line must not land out of sight.
  const box = useStickyBottom(props.lines.length);
  return (
    <pre
      class="log"
      t={props.t ?? "log"}
      ref={box}
      style={{ maxHeight: `${(props.rows ?? 14) * 16}px` }}
    >
      {props.lines.length === 0 ? "—" : props.lines.join("\n")}
    </pre>
  );
}

// ── inputs ─────────────────────────────────────────────────────────────────

export function Toggle(props: {
  checked: boolean;
  label: string;
  tip?: string;
  /** Stable handle for tests, when the label is not addressable enough. */
  t?: string;
  onChange: (v: boolean) => void;
}) {
  return (
    <label class="toggle" title={props.tip}>
      <input
        type="checkbox"
        checked={props.checked}
        aria-label={props.label}
        t={props.t}
        onChange={(e) =>
          props.onChange((e.currentTarget as HTMLInputElement).checked)}
      />
      <span class="toggle-track" />
      <span class="toggle-label">{props.label}</span>
    </label>
  );
}

export function Segmented<T extends string>(props: {
  value: T;
  options: readonly { id: T; label: string; tip?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div class="segmented" role="group">
      {props.options.map((o) => (
        <button
          key={o.id}
          type="button"
          class={o.id === props.value ? "seg on" : "seg"}
          title={o.tip}
          onClick={() => props.onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
