// client/src/ui/kit.tsx — the client's small visual vocabulary.
//
// Presentational only: props in, callbacks out, no cell reads. Deliberately a
// handful of pieces rather than a copy of the server app's kit — this app has
// one screen, and a component library for one screen is furniture nobody sits
// on.

import type { JSX } from "aio/jsx-runtime";
import { useLocal, useRef } from "aio/air";

export type Tone = "ok" | "warn" | "bad" | "idle" | "busy";

export function Pill(
  props: { tone?: Tone; title?: string; t?: string; children?: JSX.Node },
) {
  return (
    <span
      class={`pill tone-${props.tone ?? "idle"}`}
      title={props.title}
      t={props.t}
    >
      {props.children}
    </span>
  );
}

/** A label and a value, on one line. The densest honest way to show a fact. */
export function KV(
  props: { k: string; v: JSX.Node; tip?: string; t?: string },
) {
  return (
    <div class="kv" title={props.tip} t={props.t}>
      <span class="kv-k">{props.k}</span>
      <span class="kv-v">{props.v}</span>
    </div>
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

/** The one place an error is shown — and it is never hidden. */
export function ErrorNote(props: { message: string; onDismiss?: () => void }) {
  if (!props.message) return null;
  return (
    <div class="error-note" t="error">
      <span class="error-icon">⚠</span>
      <span class="error-text">{props.message}</span>
      {props.onDismiss
        ? (
          <button
            type="button"
            class="x"
            t="error-dismiss"
            aria-label="Dismiss"
            onClick={props.onDismiss}
          >
            ✕
          </button>
        )
        : null}
    </div>
  );
}

/** A 0..1 meter. `null` draws an empty track — "not reported" is not zero. */
export function Meter(
  props: { value: number | null; tone?: Tone; title?: string },
) {
  const pct = props.value === null
    ? 0
    : Math.max(0, Math.min(1, props.value)) * 100;
  return (
    <div class="meter" title={props.title}>
      <div
        class={`meter-fill tone-${props.tone ?? "busy"}`}
        style={{ width: `${pct.toFixed(1)}%` }}
      />
    </div>
  );
}

/**
 * Take this — one button, one clipboard write, one visible acknowledgement.
 *
 * The tick is the point: a copy button that looks identical before and after
 * the click leaves the user guessing, and the usual answer to that guess is to
 * press it again and paste twice.
 */
export function CopyButton(props: {
  text: string;
  title: string;
  label?: string;
  t?: string;
}) {
  const [copied, setCopied] = useLocal(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  return (
    <button
      type="button"
      class={copied ? "btn tiny copy-btn is-copied" : "btn tiny copy-btn"}
      t={props.t}
      title={props.title}
      aria-label={props.title}
      disabled={!props.text}
      onClick={() => {
        void navigator.clipboard?.writeText(props.text);
        setCopied(true);
        clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(false), 1400);
      }}
    >
      <span class="copy-icon" aria-hidden="true">{copied ? "✔" : "⧉"}</span>
      {props.label
        ? <span class="copy-label">{copied ? "copied" : props.label}</span>
        : null}
    </button>
  );
}

/** Between Send and the first token there is nothing to render, and a model
 *  thinking for eight seconds must not look like a dead server. */
export function Waiting() {
  return (
    <div class="waiting" t="waiting">
      <i class="dot" />
      <i class="dot" />
      <i class="dot" />
      <span>waiting for the first token…</span>
    </div>
  );
}

/**
 * A text or number box over a REPLICATED cell field, without the round trip
 * per keystroke.
 *
 * Bound directly (`value={cell.x}` + a dispatch on every `input`), each key is
 * a dispatch, and anything that re-renders the box meanwhile — a 1 s poll, a
 * streamed reply's flush — writes back the value the cell last acknowledged,
 * wiping keys still in flight. So the text is local while it is being edited
 * and the cell sees ONE write on `change` (blur, or Enter), the same rule as
 * llama.master's own `DraftInput` (`src/ui/kit.tsx` there; copied, not
 * imported, because the two apps share only `src/lib`).
 *
 * `onEnter` commits first and acts once that write has LANDED, because
 * `keydown` fires BEFORE the `change` that would have committed — an
 * Enter-to-connect that read the cell would connect to the previous address.
 * So `onCommit` returns its dispatch, and `onEnter` waits on it.
 */
export function DraftInput(props: {
  type: "text" | "number";
  value: string;
  onCommit: (value: string) => unknown;
  onEnter?: () => void;
  ariaLabel: string;
  class?: string;
  placeholder?: string;
  min?: number | string;
  max?: number | string;
  t?: string;
}) {
  // null = not editing: show the cell's value. A string = the user's, verbatim.
  const [draft, setDraft] = useLocal<string | null>(null);
  // What this box last handed the cell, so the `change` that follows an
  // Enter-commit is not a second, identical write.
  const sent = useRef<string | null>(null);
  const commit = (v: string): unknown => {
    setDraft(null);
    if (v === props.value || v === sent.current) return undefined;
    sent.current = v;
    return props.onCommit(v);
  };
  return (
    <input
      type={props.type}
      class={props.class}
      aria-label={props.ariaLabel}
      placeholder={props.placeholder}
      min={props.min}
      max={props.max}
      t={props.t}
      value={draft ?? props.value}
      onInput={(e) => {
        sent.current = null;
        setDraft((e.currentTarget as HTMLInputElement).value);
      }}
      onChange={(e) => commit((e.currentTarget as HTMLInputElement).value)}
      onKeyDown={(e) => {
        if ((e as KeyboardEvent).key !== "Enter" || !props.onEnter) return;
        const onEnter = props.onEnter;
        void Promise.resolve(
          commit((e.currentTarget as HTMLInputElement).value),
        ).then(() => onEnter());
      }}
    />
  );
}
