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
//
// And it runs TWO prompts, prose and code, because one number could not answer
// the question the tuner keeps handing to this panel. Speculative decoding is
// lossless and free of memory cost and is still a bad idea about half the time,
// because a drafted token the model rejects is work thrown away — and which
// half you are in depends entirely on what you ask it to write. Measuring only
// prose measured that setting at its worst case, every time, and then printed
// one rate with nothing to read it against. Two runs against the same loaded
// server fix it: with drafting off the pair agrees to within a couple of
// percent, so the gap that opens on code is the drafter's, in tok/s, here.

import { srv } from "../cell/srv.ts";
import { cfg } from "../cell/cfg.ts";
import { prefillNote } from "../lib/bench.ts";
import type { BenchResult } from "../lib/bench.ts";
import { num } from "../lib/params.ts";
import { tps as fmtTps } from "../lib/format.ts";
import {
  bandwidthNow,
  benchCodeNow,
  benchNow,
  serverRunning,
  specVerdictNow,
} from "./derive.ts";
import { ErrorNote, Panel, Pill } from "./kit.tsx";

/** What the button says while it works. Naming the prompt in flight matters
 *  here more than it usually would: the pair takes twice as long as the old
 *  single run, and an unchanging "Measuring…" through a minute of that reads
 *  as a hang rather than as progress. */
function busyLabel(): string {
  return srv.benchKind === "code" ? "Measuring code…" : "Measuring prose…";
}

/** The button, and what it last measured. Compact enough for a column. */
export function SpeedCheck(props: { t?: string }) {
  const id = props.t ?? "speed";
  const b = benchNow();
  const c = benchCodeNow();
  const v = specVerdictNow();
  const ready = srv.status === "ready";
  return (
    <div class="speed-check" t={id}>
      <button
        type="button"
        class="btn small"
        t={`${id}-run`}
        disabled={!ready || srv.benching}
        title={ready
          ? "Generate a fixed number of tokens twice — once on prose, once on code — and time both. Runs against the server that is already up: nothing is reloaded. The pair is what says whether speculative decoding is earning its place."
          : "Start the server first — this measures the model that is loaded."}
        onClick={() => srv.benchBoth()}
      >
        {srv.benching ? busyLabel() : "Measure speed"}
      </button>
      {b
        ? (
          <span class="speed-figures">
            <Pill
              tone="ok"
              title="Tokens generated per second on the prose prompt, measured"
            >
              {fmtTps(b.genTps)} tok/s prose
            </Pill>
            {c
              ? (
                <Pill
                  tone={v?.tone === "caution"
                    ? "warn"
                    : v?.tone === "ok"
                    ? "ok"
                    : "idle"}
                  title="Tokens generated per second on the code prompt. Code repeats itself, which is what a drafter feeds on — the gap between these two is what speculative decoding is worth here."
                >
                  {fmtTps(c.genTps)} tok/s code
                </Pill>
              )
              : null}
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
              ? "Generating two fixed prompts and timing them…"
              : "Not measured yet — every speed on this page is an estimate until it is."}
          </span>
        )}
    </div>
  );
}

/** One measured row. Three of these per prompt would be six rows and a wall, so
 *  the prompt kind is the row and the three numbers are its columns. */
function Row(props: { label: string; note: string; b: BenchResult }) {
  const { b } = props;
  return (
    <div class="speed-row">
      <span class="speed-k">{props.label}</span>
      <span class="speed-v">{fmtTps(b.genTps)} tokens/s</span>
      <span class="speed-n">
        {props.note} · {b.promptTps > 0
          ? `${fmtTps(b.promptTps)} tok/s prefill`
          : "prefill not reported"} · first token {(b.latencyMs / 1000).toFixed(
            2,
          )} s
      </span>
    </div>
  );
}

/** The whole thing: the measurement, what it means, and what to do about it. */
export function SpeedPanel() {
  const b = benchNow();
  const c = benchCodeNow();
  const v = specVerdictNow();
  const band = bandwidthNow();
  const running = serverRunning();
  const prefill = b ? prefillNote(b, num(cfg.settings, "ubatchSize")) : "";
  return (
    <Panel
      title="Speed"
      icon="⚡"
      right={
        <Pill tone={b ? "ok" : "idle"}>
          {b && c ? "measured" : b ? "half measured" : "not measured"}
        </Pill>
      }
    >
      <ErrorNote message={srv.benchError} />
      <SpeedCheck t="speed" />
      {!running
        ? (
          <p class="note">
            Start the server and press Measure. It runs two fixed prompts
            against the loaded model — once on prose, once on code — because the
            gap between them is what says whether speculative decoding is
            earning its place. The numbers everywhere else in the app are
            arithmetic over an assumed memory bandwidth until this has run once;
            after that they are arithmetic over this machine's.
          </p>
        )
        : null}
      {b
        ? (
          <>
            <div class="speed-rows">
              <Row
                label="Prose"
                note={`flowing text, over ${b.genTokens} tokens`}
                b={b}
              />
              {c
                ? (
                  <Row
                    label="Code"
                    note={`repetitive text, over ${c.genTokens} tokens`}
                    b={c}
                  />
                )
                : null}
            </div>
            {v
              ? (
                <div
                  class={v.tone === "caution" ? "note warn" : "note"}
                  t="spec-verdict"
                >
                  <strong>{v.headline}</strong>
                  {v.advice ? ` ${v.advice}` : ""}
                </div>
              )
              : c
              ? null
              : (
                <p class="note dim">
                  Only the prose prompt has run. The code prompt is the other
                  half: with no drafter attached the two agree to within a few
                  percent, so the gap that opens between them is exactly what
                  speculative decoding is worth for the work you do.
                </p>
              )}
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
              — this is the fast end of the range, not the average. Bandwidth is
              calibrated from the prose run only: a drafted token is not a token
              the memory bus paid for.
            </p>
          </>
        )
        : null}
    </Panel>
  );
}
