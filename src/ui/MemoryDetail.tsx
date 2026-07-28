// src/ui/MemoryDetail.tsx — every byte, in words.
//
// The bars say "roughly this much"; this says exactly what and exactly where,
// so the question "why does a 27B model need 40 GB" is answered on the page the
// user is already looking at instead of somewhere else.
//
// It has two modes, and the difference is honest rather than cosmetic:
//   RUNNING  — this describes the process that is up, computed from the command
//              it was started with, next to the RAM it actually holds and the
//              VRAM the driver actually reports.
//   PLANNED  — nothing is running; this is a projection of what would happen,
//              and it says so.

import type { Plan } from "../lib/plan.ts";
import { bytes, pctLabel } from "../lib/format.ts";

type Row = { label: string; value: string; hint?: string; strong?: boolean };

function Rows(props: { rows: Row[]; t?: string }) {
  return (
    <table class="memdetail" t={props.t}>
      <tbody>
        {props.rows.map((r) => (
          <tr key={r.label} class={r.strong ? "strong" : ""}>
            <th title={r.hint}>{r.label}</th>
            <td class="mono">{r.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** One pool, itemised: what llama.cpp puts there, what else is there, what is
 *  left. Zero-byte items are dropped — a row of "0 B" teaches nothing. */
function poolRows(
  p: Plan["vram"],
  extra: Row[] = [],
): Row[] {
  const rows: Row[] = [];
  for (const b of p.buckets) {
    if (b.bytes <= 0) continue;
    rows.push({
      label: b.label,
      value: bytes(b.bytes),
      hint: b.key === "compute"
        ? "Scratch space for the current batch. The one number here that is an estimate."
        : b.key === "kv"
        ? "Grows linearly with the context — this is what a longer conversation costs."
        : b.key === "experts"
        ? "Routed mixture-of-experts tensors, counted apart from the dense weights."
        : undefined,
    });
  }
  rows.push({ label: "llama.cpp total", value: bytes(p.usedB), strong: true });
  if (p.otherB > 0) {
    rows.push({
      label: "In use elsewhere",
      value: bytes(p.otherB),
      hint: "Other processes — the desktop, a browser, another model.",
    });
  }
  rows.push({ label: "Capacity", value: bytes(p.capacityB) });
  rows.push(
    p.overB > 0
      ? { label: "Over capacity", value: `− ${bytes(p.overB)}`, strong: true }
      : {
        label: "Free",
        value: `${bytes(p.freeB)} · ${pctLabel(p.freeB, p.capacityB)}`,
        strong: true,
      },
  );
  return [...rows, ...extra];
}

export function MemoryDetail(props: {
  plan: Plan;
  /** True when this describes a live process rather than a projection. */
  live?: boolean;
  /** Which question this table answers. "current" describes the machine as it
   *  is; "projected" describes what the settings would do. Passed rather than
   *  inferred from `live`, because an idle machine's CURRENT state is not a
   *  projection — labelling it one was simply wrong. */
  mode?: "current" | "projected";
  /** Measured RSS of the running server, when there is one. */
  rssB?: number;
}) {
  const p = props.plan;
  const measured: Row[] = props.live && (props.rssB ?? 0) > 0
    ? [{
      label: "Process RSS (measured)",
      value: bytes(props.rssB ?? 0),
      hint:
        "What the server process actually holds in RAM right now, read from /proc — not a prediction.",
      strong: true,
    }]
    : [];

  return (
    <div class="memdetail-wrap" t="memory-detail">
      <div class="memdetail-head">
        <span class={props.live ? "pill tone-ok" : "pill tone-idle"}>
          {props.live
            ? "running now"
            : props.mode === "current"
            ? "as it is now"
            : "projected"}
        </span>
        <span class="dim">
          {props.live
            ? "What the running server is using."
            : props.mode === "current"
            ? "What the machine is using, with nothing of ours loaded."
            : "What these settings would use."}
        </span>
      </div>

      <div class="memdetail-cols">
        <section>
          <h4>
            Context <span class="mono">{p.ctx.toLocaleString()} tokens</span>
          </h4>
          <Rows
            t="ctx-rows"
            rows={[
              {
                label: "KV cache total",
                value: bytes(p.kvTotalB),
                hint:
                  "Every token in the context is stored twice, once per layer — this is that.",
                strong: true,
              },
              {
                label: "per 1k tokens",
                value: bytes(p.kvPerTokenB * 1024),
                hint: "So you can price a longer context before choosing it.",
              },
              {
                label: "Layers on GPU",
                value: `${p.layersOnGpu} of ${p.nLayer}`,
              },
              ...(p.moeOnCpu > 0
                ? [{
                  label: "Expert layers in RAM",
                  value: `${p.moeOnCpu} of ${p.nLayer}`,
                }]
                : []),
            ]}
          />
        </section>

        <section>
          <h4>VRAM</h4>
          {p.vram.capacityB > 0
            ? <Rows t="vram-rows" rows={poolRows(p.vram)} />
            : <p class="dim">No GPU in use.</p>}
        </section>

        <section>
          <h4>System RAM</h4>
          <Rows t="ram-rows" rows={poolRows(p.ram, measured)} />
        </section>
      </div>

      {p.notes.length > 0
        ? (
          <ul class="memdetail-notes">
            {p.notes.map((n) => <li key={n}>{n}</li>)}
          </ul>
        )
        : null}
    </div>
  );
}
