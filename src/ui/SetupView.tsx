// src/ui/SetupView.tsx — the command's flags, as sentences.
//
// The command view is the exact argv; this is what it means. Both compose from
// `shownSettings()`/`shownModel()`, so while a server runs the rows describe
// the setup it was STARTED with, not what the panels have drifted to since —
// the same promise the command keeps.

import { shownSetup } from "./derive.ts";
import { KV, Panel, Pill } from "./kit.tsx";

export function SetupPanel(props: {
  t?: string;
  /**
   * The all-in-one form: one wrapping row of chips instead of labelled rows.
   * That page's machine column is a height budget, so every fact keeps its
   * fewest characters (`SetupRow.short`) and the full sentence moves into the
   * chip's hover — present, not paid for in lines.
   */
  compact?: boolean;
}) {
  const id = props.t ?? "setup";
  const rows = shownSetup();
  return (
    <Panel title="Setup" icon="≡">
      {props.compact
        ? (
          <div class="setup-chips" t={id}>
            {rows.map((r) => (
              <Pill
                key={r.label}
                title={`${r.label}: ${r.value}${r.tip ? `\n${r.tip}` : ""}`}
              >
                {r.short}
              </Pill>
            ))}
          </div>
        )
        : (
          <div class="kv-grid" t={id}>
            {rows.map((r) => (
              <KV key={r.label} k={r.label} v={r.value} tip={r.tip} />
            ))}
          </div>
        )}
    </Panel>
  );
}
