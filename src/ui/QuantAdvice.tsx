// src/ui/QuantAdvice.tsx — "would a smaller file be faster, and by how much?"
//
// The app used to answer this only when the model did NOT fit ("try a smaller
// quantisation" is in three refusal messages). For a model that fits, the
// largest speed lever available was never mentioned: generation is bandwidth
// bound, so a file with half the bits per weight generates at close to twice
// the rate, and the app knew both numbers and said nothing.
//
// What makes this worth a table rather than a sentence is that the interesting
// answer is often not the ratio. `src/lib/quant.ts` runs every candidate
// through the real tuner on the real machine, so the row that matters is
// usually the one where the model stops spilling into system RAM — a step that
// is four times faster, not twice. A ratio could not find it and neither could
// a rule of thumb.
//
// Honesty, which this panel needs more than most: the CURRENT row is measured
// (the file is here, its bits per weight are counted off its own tensor table)
// and every other row is an estimate of a file that does not exist on this
// machine. The header says so once, plainly, rather than stamping "≈" on
// twenty numbers.

import { models } from "../cell/models.ts";
import { bytes } from "../lib/format.ts";
import { fileBpw, qualityNote } from "../lib/quant.ts";
import type { QuantOption } from "../lib/quant.ts";
import { currentModel, quantRows } from "./derive.ts";
import { Panel, Pill } from "./kit.tsx";

function placementWord(p: QuantOption): string {
  if (!p.possible) return "does not fit";
  if (p.layersInRam > 0) return `${p.layersInRam} layers in RAM`;
  return "all in VRAM";
}

/** The comparison table for the selected model. */
export function QuantAdvice() {
  const m = currentModel();
  const rows = quantRows();
  const meta = m?.meta ?? null;
  if (!meta || rows.length === 0) {
    return (
      <Panel title="Quantisation and speed" icon="⚖">
        <p class="note">
          {models.selected
            ? "This model's header could not be read, so its weight per parameter is unknown."
            : "Select a model to compare its quantisations."}
        </p>
      </Panel>
    );
  }
  const here = fileBpw(meta);
  const mine = rows.find((r) => r.current);
  return (
    <Panel
      title="Quantisation and speed"
      icon="⚖"
      right={
        <Pill tone="idle" title="Bits per weight in the file you have">
          {here.exact
            ? `${here.bpw.toFixed(2)} bits/weight`
            : `about ${here.bpw.toFixed(2)} bits/weight`}
        </Pill>
      }
    >
      <p class="note">
        Generating a token reads every active weight once, so the bits per
        weight very nearly set the speed. The row you have is measured from this
        file. The others are what a typical file of that quantisation would
        weigh, placed on this machine by the same tuner — including the step
        where a model stops spilling into system RAM, which is worth far more
        than the size alone.
      </p>
      <div class="quant-table" t="quant-table">
        <div class="quant-head" key="head">
          <span>Quant</span>
          <span>Bits</span>
          <span>Weights</span>
          <span>Placement</span>
          <span>Context</span>
          <span>Speed</span>
        </div>
        {rows.map((r) => (
          <div
            key={r.quant}
            class={r.current ? "quant-row is-current" : "quant-row"}
            t={`quant-${r.quant}`}
            title={r.current ? "The file you have" : qualityNote(r.bpw)}
          >
            <span class="quant-name">
              {r.quant}
              {r.current ? <span class="quant-tag">have</span> : null}
            </span>
            <span class="quant-bits">{r.bpw.toFixed(2)}</span>
            <span class="quant-size">{bytes(r.bytesB)}</span>
            <span class="quant-place">{placementWord(r)}</span>
            <span class="quant-ctx">
              {r.possible ? r.ctx.toLocaleString() : "—"}
            </span>
            <span class="quant-speed">
              {r.possible && r.tps > 0
                ? (
                  <>
                    {Math.round(r.tps).toLocaleString()} tok/s
                    {!r.current && r.speedup > 0
                      ? (
                        <span
                          class={r.speedup >= 1.1
                            ? "quant-gain up"
                            : r.speedup <= 0.91
                            ? "quant-gain down"
                            : "quant-gain"}
                        >
                          {r.speedup >= 1
                            ? `${r.speedup.toFixed(1)}× faster`
                            : `${(1 / r.speedup).toFixed(1)}× slower`}
                        </span>
                      )
                      : null}
                  </>
                )
                : "—"}
            </span>
          </div>
        ))}
      </div>
      {mine && !mine.possible
        ? (
          <p class="note warn">
            The file you have does not fit on this machine as configured. The
            rows above say which one would.
          </p>
        )
        : null}
      <p class="note dim">
        Speeds are this machine's measured bandwidth where the Speed panel has
        run, and a labelled default until then — the ratios between rows hold
        either way. Sizes are the weights only; the KV cache is the same at
        every quantisation and is counted separately in the memory plan.
      </p>
    </Panel>
  );
}

/** One line, for a page that has no room for the table. "" when there is
 *  nothing worth saying — which is the common case and should stay quiet. */
export function QuantLine(props: { advice: string }) {
  if (!props.advice) return null;
  return (
    <p class="note quant-line" t="quant-line">
      {props.advice}
    </p>
  );
}
