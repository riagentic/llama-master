// src/ui/PrereqPage.tsx — everything llama.master needs installed, and how to get it.
//
// Its own page because it is a task, not a reading: the Machine page answers
// "what is this computer", and that is a glance. This answers "why can I not
// build yet, and what do I press" — a list with an action on most rows, the exact
// command shown before anything privileged runs, and a documentation link where
// the app cannot do it for you.
//
// The Machine page keeps a one-line summary and sends you here.

import { prereq } from "../cell/prereq.ts";
import { describe, scriptPreview } from "../lib/fixplan.ts";
import { fixablePrereqs, fixPlanFor } from "./derive.ts";
import { Bar, ErrorNote, LogView, Panel, Pill } from "./kit.tsx";

function FixButton(props: { id: string; found: boolean }) {
  if (props.found) return null;
  const plan = fixPlanFor(props.id);
  if (!plan) return null;
  const busy = prereq.fixing !== "" || prereq.install !== null;
  if (plan.kind === "manual") {
    // "We will not do this for you" always comes with where to read how.
    return (
      <span class="manual-fix">
        <span class="dim" title={plan.reason}>manual</span>
        {plan.docsUrl
          ? (
            <a
              href={plan.docsUrl}
              target="_blank"
              rel="noreferrer"
              title={plan.reason}
            >
              docs ↗
            </a>
          )
          : null}
      </span>
    );
  }
  return (
    <span class="script-fix">
      <button
        type="button"
        class="btn tiny"
        t={`fix-${props.id}`}
        title={describe(plan)}
        disabled={busy}
        onClick={() => prereq.fix(props.id)}
      >
        {prereq.fixing === props.id
          ? "Fixing…"
          : plan.kind === "script"
          ? `Install (${plan.steps.length} steps)`
          : "Fix"}
      </button>
      {plan.kind === "script"
        ? (
          <a
            href={plan.docsUrl}
            target="_blank"
            rel="noreferrer"
            title={plan.docsUrl}
          >
            docs ↗
          </a>
        )
        : null}
    </span>
  );
}

function PrereqPanel() {
  const items = prereq.items;
  const inst = prereq.install;
  const fixable = fixablePrereqs();
  // A scripted install (ROCm) changes the machine in ways a package install
  // does not — add a repository, install a driver — so every command is on
  // screen before the button is pressed, not only in the log afterwards.
  const script = fixable.map((f) => fixPlanFor(f.id)).find((p) =>
    p?.kind === "script"
  );
  const scriptSteps = script ? scriptPreview(script) : [];
  const scriptTitle = script?.kind === "script" ? script.title : "";
  return (
    <Panel
      title="Prerequisites"
      icon="✓"
      right={
        <>
          {fixable.length > 0
            ? (
              <button
                type="button"
                class="btn small primary"
                t="fix-all"
                title={`Install ${fixable.map((f) => f.label).join(", ")}`}
                disabled={prereq.fixing !== "" || prereq.install !== null}
                onClick={() => prereq.fixAll()}
              >
                {prereq.fixQueue.length > 0
                  ? `Fixing ${
                    fixable.length - prereq.fixQueue.length
                  }/${fixable.length}…`
                  : `Fix all (${fixable.length})`}
              </button>
            )
            : null}
          <button
            type="button"
            class="btn small"
            onClick={() => prereq.scan()}
            disabled={prereq.scanning}
          >
            {prereq.scanning ? "Scanning…" : "Re-check"}
          </button>
        </>
      }
    >
      <ErrorNote message={prereq.lastError} />
      {inst
        ? (
          <div class="install-progress">
            <div class="sub-label">{inst.label}</div>
            <Bar
              value={inst.received}
              max={inst.total ?? Math.max(1, inst.received)}
              tone="busy"
              height={8}
            />
          </div>
        )
        : null}
      <table class="table" t="prereq-table">
        <tbody>
          {items.map((p) => (
            <tr
              key={p.id}
              class={p.found ? "" : p.systemOnly ? "row-warn" : "row-bad"}
            >
              <td class="c-icon">{p.found ? "●" : "○"}</td>
              <td class="c-name" title={p.why}>{p.label}</td>
              <td class="c-ver mono" title={p.path}>
                {p.found ? p.version || "found" : "not found"}
                {p.managed ? <Pill tone="accent">app</Pill> : null}
              </td>
              <td class="c-act">
                <FixButton id={p.id} found={p.found} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {scriptSteps.length > 0
        ? (
          <details class="steps-preview" t="script-steps">
            <summary>
              {scriptTitle} — {scriptSteps.length}{" "}
              steps, shown before anything runs
            </summary>
            <pre class="cmd">{scriptSteps.join("\n\n")}</pre>
          </details>
        )
        : null}
      {prereq.fixLog.length > 0
        ? (
          <>
            <div class="sub-label">Installer output</div>
            <LogView lines={prereq.fixLog} t="fix-log" rows={10} />
          </>
        )
        : null}
      <div class="hint">
        “Fix” installs through your package manager and will ask for your
        password. Items marked “manual” need a decision the app should not make
        for you. Nothing here is required for a prebuilt release — that path
        needs no toolchain at all.
      </div>
    </Panel>
  );
}
export function PrereqPage() {
  // No ErrorNote here: PrereqPanel already renders `prereq.lastError`, and one
  // failure showing as two identical red boxes reads as two failures.
  return (
    <div class="tab-body">
      <PrereqPanel />
    </div>
  );
}
