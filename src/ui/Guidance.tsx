// src/ui/Guidance.tsx — the component that makes a problem actionable.
//
// One rule, applied everywhere something can go wrong: state the reason in a
// sentence, then list the steps — and where a step is something the app can
// simply do, render it as a button instead of an instruction. "Use the source
// route" should be a click, not a scavenger hunt for the control.
//
// Presentational: it takes a Diagnosis and dispatches the actions it is given.

import { builds } from "../cell/builds.ts";
import { prereq } from "../cell/prereq.ts";
import { ui } from "../cell/ui.ts";
import type { Diagnosis, FixAction, Step } from "../lib/diagnose.ts";

/** Perform a step's action. Kept here so every place that shows guidance
 *  behaves identically. */
function run(action: FixAction): void {
  switch (action.kind) {
    case "switch-origin":
      builds.setOrigin(action.to);
      break;
    case "switch-backend":
      builds.setBackend(action.to);
      break;
    case "fix-prereq":
      ui.go("dashboard");
      prereq.fix(action.id);
      break;
    case "open-tab":
      ui.go(action.tab);
      break;
    case "open-url":
      globalThis.open?.(action.url, "_blank");
      break;
  }
}

function label(action: FixAction): string {
  switch (action.kind) {
    case "switch-origin":
      return action.to === "source"
        ? "Build from source"
        : "Use a prebuilt release";
    case "switch-backend":
      return `Switch to ${action.to.toUpperCase()}`;
    case "fix-prereq":
      return "Fix it";
    case "open-tab":
      return action.tab === "dashboard"
        ? "Open Machine"
        : action.tab === "settings"
        ? "Open Tune"
        : "Open Build";
    case "open-url":
      return "Open docs ↗";
  }
}

function StepRow(props: { step: Step }) {
  const a = props.step.action;
  return (
    <li class="guide-step">
      <span>{props.step.text}</span>
      {a
        ? (
          <button
            type="button"
            class="btn tiny primary"
            onClick={() => run(a)}
            title={props.step.text}
          >
            {label(a)}
          </button>
        )
        : null}
    </li>
  );
}

/**
 * A reason plus its steps.
 *
 * `tone` distinguishes "this will not work" from "this did not work" — the
 * content is the same shape either way, which is the point: a user should not
 * have to learn two different failure formats.
 */
export function Guidance(props: {
  diagnosis: Diagnosis;
  tone?: "error" | "warn";
  t?: string;
}) {
  const cls = props.tone === "error" ? "error-note guide" : "warn-note guide";
  return (
    <div class={cls} t={props.t ?? "guidance"}>
      <div class="guide-reason">{props.diagnosis.reason}</div>
      {props.diagnosis.steps.length > 0
        ? (
          <ul class="guide-steps">
            {props.diagnosis.steps.map((s) => (
              <StepRow
                key={s.text}
                step={s}
              />
            ))}
          </ul>
        )
        : null}
    </div>
  );
}
