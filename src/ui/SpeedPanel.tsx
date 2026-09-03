// src/ui/SpeedPanel.tsx — measure it, rather than saying what it should be.
//
// Every speed the app shows is bytes ÷ bandwidth, and bandwidth is the one term
// that cannot be read off the machine (`src/lib/speed.ts` says why). Until now
// the only observation was whatever the user last happened to say in the chat:
// a real rate, about an unknown prompt, at an unknown context fill, with an
// unknown amount of thinking in it. Good enough to beat a default; not good
// enough to put a number beside.
//
// This is the same observation with the variables held still, against the
// server that is already up — no reload, no second process, nothing at risk
// (`src/lib/bench.ts`). Two components over one measurement, for the reason
// `CtxControls` is one component on two pages: the all-in-one page is a budget
// and takes the compact form, the Server tab has the room for the whole thing.
//
// What it shows is deliberately BOTH rates. Generation is what people mean by
// "how fast is it", but prompt processing is the wait before the first word,
// and on a long-context model it is the half that turns into minutes.

import { srv } from "../cell/srv.ts";
import { cfg } from "../cell/cfg.ts";
import { prefillNote } from "../lib/bench.ts";
import { num } from "../lib/params.ts";
import { tps as fmtTps } from "../lib/format.ts";
import { bandwidthNow, benchNow, serverRunning } from "./derive.ts";
import { ErrorNote, Panel, Pill } from "./kit.tsx";

/** The button, and what it last measured. Compact enough for a column. */
export function SpeedCheck(props: { t?: string }) {
  const id = props.t ?? "speed";
  const b = benchNow();
  const ready = srv.status === "ready";
  return (
    <div class="speed-check" t={id}>
      <button
        type="button"
        class="btn small"
        t={`${id}-run`}
        disabled={!ready || srv.benching}
        title={ready
          ? "Generate a fixed number of tokens and time it. Runs against the server that is already up — nothing is reloaded."
          : "Start the server first — this measures the model that is loaded."}
        onClick={() => srv.bench()}
      >
        {srv.benching ? "Measuring…" : "Measure speed"}
      </button>
      {b
        ? (
          <span class="speed-figures">
            <Pill tone="ok" title="Tokens generated per second, measured">
              {fmtTps(b.genTps)} tok/s
            </Pill>
            {b.promptTps > 0
              ? (
                <Pill
                  tone="idle"
                  title="Prompt processing — the wait before the first word"
                >
                  {fmtTps(b.promptTps)} tok/s prefill
                </Pill>
              )
              : null}
          </span>
        )
        : (
          <span class="speed-hint">
            {srv.benching
              ? "Generating a fixed prompt and timing it…"
              : "Not measured yet — every speed on this page is an estimate until it is."}
          </span>
        )}
    </div>
  );
}

/** The whole thing: the measurement, what it means, and what to do about it. */
export function SpeedPanel() {
  const b = benchNow();
  const band = bandwidthNow();
  const running = serverRunning();
  const prefill = b ? prefillNote(b, num(cfg.settings, "ubatchSize")) : "";
  return (
    <Panel
      title="Speed"
      icon="⚡"
      right={
        <Pill tone={b ? "ok" : "idle"}>
          {b ? "measured" : "not measured"}
        </Pill>
      }
    >
      <ErrorNote message={srv.benchError} />
      <SpeedCheck t="speed" />
      {!running
        ? (
          <p class="note">
            Start the server and press Measure. The numbers everywhere else in
            the app are arithmetic over an assumed memory bandwidth until this
            has run once — after that they are arithmetic over this machine's.
          </p>
        )
        : null}
      {b
        ? (
          <>
            <div class="speed-rows">
              <div class="speed-row">
                <span class="speed-k">Generation</span>
                <span class="speed-v">{fmtTps(b.genTps)} tokens/s</span>
                <span class="speed-n">
                  the pace the answer arrives at, over {b.genTokens} tokens
                </span>
              </div>
              <div class="speed-row">
                <span class="speed-k">Prompt</span>
                <span class="speed-v">
                  {b.promptTps > 0 ? `${fmtTps(b.promptTps)} tokens/s` : "—"}
                </span>
                <span class="speed-n">
                  the wait before the first word, over {b.promptTokens} tokens
                </span>
              </div>
              <div class="speed-row">
                <span class="speed-k">First token</span>
                <span class="speed-v">
                  {(b.latencyMs / 1000).toFixed(2)} s
                </span>
                <span class="speed-n">
                  measured here, because no server metric reports it
                </span>
              </div>
            </div>
            {prefill ? <p class="note">{prefill}</p> : null}
            {band
              ? (
                <p class={band.tone === "caution" ? "note warn" : "note"}>
                  {band.message}
                </p>
              )
              : null}
            <p class="note dim">
              Measured at a context of {b.ctx.toLocaleString()}{" "}
              tokens with the cache nearly empty. A long conversation reads the
              whole cache every token, so the same model gets slower as it fills
              — this is the fast end of the range, not the average.
            </p>
          </>
        )
        : null}
    </Panel>
  );
}
