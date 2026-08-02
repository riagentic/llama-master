// src/ui/kit.tsx — the visual vocabulary.
//
// Every panel is built from these, so a bar means the same thing everywhere and
// the whole app changes shape from one file. Presentational only: nothing here
// reads a cell or dispatches — data comes in as props, intent goes out as
// callbacks, which is what makes each piece renderable in a test in isolation.

import type { JSX } from "aio/jsx-runtime";
import {
  bytes as fmtBytes,
  pct as pctOf,
  tps as fmtTps,
} from "../lib/format.ts";
import {
  TPS_GREAT_AT,
  TPS_POOR_BELOW,
  tpsBand,
  tpsLabel,
  tpsWhy,
} from "../lib/speed.ts";
import { useStickyBottom } from "./sticky.ts";

export type Tone = "ok" | "warn" | "bad" | "idle" | "busy" | "accent";

// ── containers ─────────────────────────────────────────────────────────────

export function Panel(props: {
  title: string;
  icon?: string;
  right?: JSX.Node;
  wide?: boolean;
  /** Grow to take the remaining height of a flex column, body scrolling
   *  inside. What the all-in-one server log wants; a fixed panel elsewhere. */
  fill?: boolean;
  children?: JSX.Node;
}) {
  return (
    <section
      class={`panel${props.wide ? " panel-wide" : ""}${
        props.fill ? " panel-fill" : ""
      }`}
    >
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
/**
 * A two-part occupancy bar: what the kernel calls used, plus what it calls
 * cache while a memory-mapped model sits in it. One component, so the header
 * and the vitals cannot disagree about what the second colour means. The
 * kernel books mapped weights as reclaimable cache — every "used" meter calls
 * them free (measured: 138 of 139 GB invisible), and the model read as
 * missing from its own machine.
 */
export function MappedBar(props: {
  usedB: number;
  mappedB: number;
  totalB: number;
  height?: number;
}) {
  const pct = (n: number) =>
    `${Math.max(0, Math.min(100, (n / Math.max(1, props.totalB)) * 100))}%`;
  return (
    <div
      class="mappedbar"
      style={{ height: `${props.height ?? 4}px` }}
      title={`${fmtBytes(props.usedB)} used by everything else · ${
        fmtBytes(props.mappedB)
      } holding the memory-mapped model. The OS books the model as reclaimable cache — meters call it free, but evicting it means re-reading from disk and generation slowing to a crawl.`}
      t="mapped-bar"
    >
      <i class="mappedbar-used" style={{ width: pct(props.usedB) }} />
      <i class="mappedbar-model" style={{ width: pct(props.mappedB) }} />
    </div>
  );
}

/**
 * A reasoning model's think-first act, folded so it never drowns the answer.
 *
 * llama.cpp streams the `<think>` block as `reasoning_content`, with the
 * answer's `content` empty the whole while — on DeepSeek-V4 that is the entire
 * first half of every reply. Invisible thinking read as a broken chat; a wall
 * of raw reasoning would bury the answer. Folded is both honest and readable:
 * open and live while it is all there is, a one-line "thought first" once the
 * answer exists.
 */
export function Thinking(props: { text?: string; live?: boolean }) {
  if (!props.text) return null;
  return (
    <details class="msg-think" open={props.live} t="msg-think">
      <summary>{props.live ? "thinking…" : "thought first"}</summary>
      <div class="msg-think-body">{props.text}</div>
    </details>
  );
}

export function LogView(props: { lines: string[]; t?: string; rows?: number }) {
  // Newest at the bottom, and the app's diagnoses all end with "the log below" —
  // so a new line must not land out of sight.
  const box = useStickyBottom(props.lines.length);
  // rows 0 = fill: no cap of its own, the flex parent decides (`panel-fill`).
  const fill = props.rows === 0;
  return (
    <pre
      class={fill ? "log log-fill" : "log"}
      t={props.t ?? "log"}
      ref={box}
      style={fill ? undefined : { maxHeight: `${(props.rows ?? 14) * 16}px` }}
    >
      {props.lines.length === 0 ? "—" : props.lines.join("\n")}
    </pre>
  );
}

/**
 * Generation speed, as a number you can act on.
 *
 * A rate on its own means nothing to most people — is 7 tokens/second good? So
 * the bar is banded against READING SPEED rather than against hardware, which is
 * the question actually being asked ("will this be pleasant to use"): under ~5
 * tok/s you are waiting for the model, and from ~20 it arrives faster than you
 * can read. Colour carries the same three answers as everywhere else in this app,
 * and the text says which it is, so the meaning does not live in hue alone.
 */
export function TpsMeter(
  props: { tps: number; measured?: boolean; label?: string; t?: string },
) {
  const band = tpsBand(props.tps);
  const tone = band === "poor" ? "bad" : band === "great" ? "ok" : "warn";
  // The bar saturates at the point where more speed stops being noticeable.
  const pct = Math.max(
    0,
    Math.min(100, (props.tps / (TPS_GREAT_AT * 1.5)) * 100),
  );
  return (
    <div class="tps-meter" t={props.t ?? "tps-meter"} title={tpsWhy(band)}>
      <div class="tps-head">
        <span class="tps-label">{props.label ?? "Speed"}</span>
        <b class={`tps-value tone-${tone}`}>
          {props.tps > 0 ? fmtTps(props.tps) : "—"}
        </b>
        <span class="tps-unit">tok/s</span>
        <span class={`pill tone-${tone}`}>{tpsLabel(band)}</span>
        {props.measured === false
          ? (
            <span
              class="dim tps-est"
              title="No reply timed on this machine yet — this uses a default bandwidth. It becomes measured after one chat."
            >
              ≈ estimated
            </span>
          )
          : null}
      </div>
      <div class="tps-track">
        {/* The band boundaries, drawn on the track so the number has a scale. */}
        <div
          class="tps-zone tps-poor"
          style={{ width: `${(TPS_POOR_BELOW / (TPS_GREAT_AT * 1.5)) * 100}%` }}
        />
        <div
          class="tps-zone tps-ok"
          style={{
            width: `${
              ((TPS_GREAT_AT - TPS_POOR_BELOW) / (TPS_GREAT_AT * 1.5)) * 100
            }%`,
          }}
        />
        <div class="tps-zone tps-great" />
        {
          /* Neutral by design — the value and pill carry the tone; the needle
            just marks where on the scale this speed sits. */
        }
        <div class="tps-needle" style={{ left: `${pct}%` }} />
      </div>
    </div>
  );
}

/**
 * "It is working" — the only honest thing to show between Send and the first
 * token.
 *
 * A local model on a cold cache can take seconds to produce its first token, and
 * until it does there is nothing to render: the user sees their own message, a
 * gap, and no way to tell a slow prompt from a dead server. One component, used
 * by every chat surface, so the two cannot disagree about what waiting looks
 * like.
 */
export function Waiting() {
  return (
    <div
      class="chat-wait"
      t="chat-wait"
      role="status"
      aria-label="Waiting for a reply"
    >
      <i />
      <i />
      <i />
    </div>
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
