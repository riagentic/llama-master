// src/ui/OnePage.tsx — everything that matters, on one screen.
//
// The other tabs exist because llama.cpp has depth. This one exists because
// most sessions do not need it: pick a model, see whether it fits, press start,
// talk to it. Machine vitals, the fit picture, the server switch and the chat
// are all here, and nothing on this page requires scrolling to a second panel
// to make sense of.
//
// Composed entirely from the same components the deep tabs use, so there is one
// implementation of every number and it cannot disagree with itself.

import { afterRender, useRef } from "aio/air";
import { chat } from "../cell/chat.ts";
import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import { builds } from "../cell/builds.ts";
import { models } from "../cell/models.ts";
import { srv } from "../cell/srv.ts";
import { ui } from "../cell/ui.ts";
import { CPU_TJMAX, GPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { bytes, tps } from "../lib/format.ts";
import { GROUPS, num, PARAMS } from "../lib/params.ts";
import { ParamControl } from "./TunePanel.tsx";
import {
  applyOptimal,
  betterPlacement,
  currentStability,
  endpoint,
  LOCK_REASON,
  placements,
  restartTuned,
  runLocked,
  selectModel,
  startBlocker,
  startServer,
  stopServer,
} from "./actions.ts";
import { pinnedCtx, PLACEMENTS, trainedCtx } from "../lib/tune.ts";
import { elapsedLabel } from "../lib/loadprogress.ts";
import type { Placement, Tuning } from "../lib/tune.ts";
import { CtxControls } from "./CtxControls.tsx";
import { MemoryDetail } from "./MemoryDetail.tsx";
import {
  Bar,
  Empty,
  ErrorNote,
  MappedBar,
  Panel,
  Pill,
  Ring,
  Spark,
  Thinking,
  Toggle,
  Waiting,
} from "./kit.tsx";
import { MemoryMap } from "./Memory.tsx";
import { Guidance } from "./Guidance.tsx";
import { OrphanBanner, ServerLog, StatusBig } from "./ServerPanel.tsx";
import { useStickyBottom } from "./sticky.ts";
import {
  canSend,
  changedCount,
  ctxOverride,
  currentModel,
  currentStatePlan,
  driftNow,
  headroomNow,
  loadingNow,
  mappedModelB,
  memoryIsLive,
  projectedSpeed,
  projectedStatePlan,
  serverRunning,
  shownModel,
  shownSettings,
  speedCalFromLastReply,
  vramTotalB,
  vramUsedB,
} from "./derive.ts";

function tone(pct: number) {
  return pct > 90 ? "bad" : pct > 75 ? "warn" : "ok";
}

/** Four vitals, each a number and a bar, in one row. */
function Vitals() {
  const c = hw.cpu;
  const g = hw.gpus[0];
  const m = hw.mem;
  const vramTotal = vramTotalB();
  const vramUsed = vramUsedB();
  // The mapped model, as its own colour: the kernel books it as reclaimable
  // cache, so "used" hides 100+ GB of resident weights and the model read as
  // missing from its own machine.
  const mapped = mappedModelB();
  return (
    <div class="one-vitals" t="vitals">
      <div class="vital">
        <Ring value={c?.utilPct ?? 0} label="CPU" tone="accent" size={56} />
        <div class="vital-body">
          <div class="vital-name" title={c?.model}>{c?.model || "CPU"}</div>
          <div class="vital-sub">
            {c ? `${c.cores} cores · ${c.threads} threads` : "—"}
          </div>
          <Bar
            value={c?.tempC ?? 0}
            max={CPU_TJMAX}
            tone={tempTone(c?.tempC ?? 0, CPU_TJMAX)}
            height={5}
          />
          <div class="vital-sub">
            {c && c.tempC > 0
              ? `${c.tempC.toFixed(0)} / ${CPU_TJMAX} °C`
              : "no sensor"}
          </div>
          <Spark data={hw.cpuHistory} tone="accent" height={22} />
        </div>
      </div>

      <div class="vital">
        <Ring value={g?.utilPct ?? 0} label="GPU" tone="busy" size={56} />
        <div class="vital-body">
          <div class="vital-name" title={g?.name}>{g?.name || "No GPU"}</div>
          <div class="vital-sub">
            {hw.gpus.length > 1
              ? `${hw.gpus.length} devices`
              : g
              ? g.vendor
              : "—"}
          </div>
          <Bar
            value={g?.tempC ?? 0}
            max={GPU_TJMAX}
            tone={tempTone(g?.tempC ?? 0, GPU_TJMAX)}
            height={5}
          />
          <div class="vital-sub">
            {g && g.tempC > 0
              ? `${g.tempC.toFixed(0)} / ${GPU_TJMAX} °C`
              : "no sensor"}
          </div>
          <Spark data={hw.gpuHistory} tone="busy" height={22} />
        </div>
      </div>

      <div class="vital">
        <Ring
          value={vramTotal ? (vramUsed / vramTotal) * 100 : 0}
          label="VRAM"
          tone="busy"
          size={56}
        />
        <div class="vital-body">
          <div class="vital-name">VRAM</div>
          <div class="vital-sub">{bytes(vramUsed)} / {bytes(vramTotal)}</div>
          <Bar
            value={vramUsed}
            max={vramTotal}
            tone={tone(vramTotal ? (vramUsed / vramTotal) * 100 : 0)}
            height={7}
          />
        </div>
      </div>

      <div class="vital">
        <Ring
          value={m ? ((m.usedB + mapped) / m.totalB) * 100 : 0}
          label="RAM"
          tone="ok"
          size={56}
        />
        <div class="vital-body">
          <div class="vital-name">RAM</div>
          <div class="vital-sub">
            {m ? `${bytes(m.usedB + mapped)} / ${bytes(m.totalB)}` : "—"}
          </div>
          {mapped > 0
            ? (
              <MappedBar
                usedB={m?.usedB ?? 0}
                mappedB={mapped}
                totalB={m?.totalB ?? 0}
                height={7}
              />
            )
            : (
              <Bar
                value={m?.usedB ?? 0}
                max={m?.totalB ?? 0}
                tone={tone(m ? (m.usedB / m.totalB) * 100 : 0)}
                height={7}
              />
            )}
          <div class="vital-sub">
            {m
              ? mapped > 0
                ? `${bytes(mapped)} mapped model · ${
                  bytes(Math.max(0, m.availableB - mapped))
                } truly free`
                : `${bytes(m.availableB)} free`
              : ""}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * What runs, where it runs, and how long its memory is — then Start.
 *
 * Ordered the way the decision is actually made: which llama.cpp, which model,
 * where it goes, how much context. Everything above Start is locked the moment
 * a server is up, because one model runs at a time and the command on screen
 * has to keep describing the process that is running.
 */
function RunStrip() {
  const m = currentModel();
  // Keep the settings describing what Start would actually run.
  //
  // Keyed on everything the tuning depends on rather than hooked to one
  // dropdown: a model can be selected from the Models tab, from `am`, or
  // restored from the last session, and settings tuned for a different model
  // are wrong however they got there. `afterRender` runs post-commit, so this
  // is a reaction to state rather than a side effect during render, and it
  // settles in one pass because re-tuning does not change the key.
  const tunedFor = useRef("");
  // Whether the machine has been measured is part of the key, because tuning
  // before it has is not a tuning. `models.scan()` and `hw.refresh()` race at
  // boot — measured at 45 ms against 48 ms on the author's machine, i.e. a coin
  // flip — and when models won, the tuner saw no RAM and no GPUs, fell back to
  // CPU placement, and `cfg.setPlacement` PERSISTED it. The card then sat idle
  // for the rest of the session, and forever after, with nothing on screen to
  // explain why. Booleans and a count only: `availableB` moves on every poll and
  // would re-tune once a second.
  //
  // `headroomNow()` is what makes this adaptive: a game taking 20 GB of VRAM, a
  // compile taking 8 GB of RAM, or either of those FINISHING, all change the
  // right answer — in both directions. It is deliberately coarse (eighths of each
  // pool, `src/lib/adapt.ts`) because these machines are workstations where the
  // raw numbers never hold still: keying on `availableB` itself would rewrite the
  // user's settings on every 1 s poll and fight their typing.
  const hwReady = hw.lastRefresh > 0 && hw.mem !== null;
  const key =
    `${models.selected}|${builds.activeId}|${cfg.placement}|${ctxOverride()}|${hwReady}|${hw.gpus.length}|${headroomNow()}`;
  afterRender(() => {
    if (tunedFor.current === key) return;
    if (!hwReady) return; // nothing measured yet — a tune now would be a guess
    if (!cfg.autoOptimal || serverRunning() || !currentModel()?.meta) return;
    tunedFor.current = key;
    applyOptimal();
  });
  // Write down a context that ACTUALLY generated. `proven`, not `healthy`:
  // /health only proves the weights loaded, and a DeepSeek-V4 run passed it at
  // 17,408 tokens then OOM'd on its first prompt — recording at /health wrote
  // that lie down as a fact, and rememberFit only ever grows, so it would have
  // opened every later run at a size measured to crash. For a model whose
  // buffers the planner cannot derive from its header this is the only measured
  // fact the app will ever have (`src/lib/fitladder.ts`). Keyed so it fires
  // once per successful start, not once per frame.
  const notedFit = useRef("");
  afterRender(() => {
    if (!srv.proven || !srv.runModel) return;
    const ctx = Number(srv.runSettings?.ctxSize ?? 0);
    const k = `${srv.runModel}|${ctx}|${srv.startedAt}`;
    if (notedFit.current === k || ctx <= 0) return;
    notedFit.current = k;
    // A run that walked the ladder is a measurement that the RECORD was too
    // high — the opening bid is capped at the record, and the ladder only
    // engages after that bid actually died. Replace it; growing-only would
    // re-run the crash at the top of every session.
    cfg.rememberFit({ model: srv.runModel, ctx, exact: srv.fitTries > 0 });
  });
  // Learn this machine's real bandwidth from the reply it just produced. The
  // speed estimate is bandwidth ÷ bytes-per-token, and bandwidth is the one term
  // that cannot be read off the machine — so the app ships a labelled default and
  // replaces it the first time a real generation gives it a rate to work back
  // from. Keyed on the rate so it runs once per reply, not once per frame.
  const calFor = useRef(0);
  afterRender(() => {
    if (calFor.current === chat.lastTps) return;
    calFor.current = chat.lastTps;
    const cal = speedCalFromLastReply();
    if (cal.gpuBps || cal.ramBps) cfg.setSpeedCal(cal);
  });

  const blocker = startBlocker();
  const running = serverRunning();
  const locked = runLocked();
  const st = currentStability();
  const all = placements();
  // The PIN ceiling is the advertised length (`trainedCtx`), not the tuner's
  // native-first aim: DeepSeek-V4 advertises 1,048,576 over a 65,536 native
  // range, and clamping a user's Max to 64k here contradicted the Models page
  // showing 1M for the same file. The auto-tuner still aims at the native
  // length on its own (`optimalCtx`, src/lib/tune.ts).
  const target = m?.meta ? trainedCtx(m.meta) : 0;
  // What would actually run: while a server is up, its context; otherwise the
  // context the SELECTED placement reaches. Showing `cfg.settings.ctxSize`
  // instead would display a number left over from the last model until the user
  // happened to press Optimal.
  const ctxNow = locked
    ? num(shownSettings(), "ctxSize")
    // A pinned context is clamped exactly as the tuner clamps it — showing
    // 64,000 "of 32,768 trained" was displaying a number that could never be
    // used, and the clamp is `pinnedCtx` so the two cannot drift.
    : pinnedCtx(
      ctxOverride() || all?.[cfg.placement]?.ctx ||
        num(shownSettings(), "ctxSize"),
      target,
    );

  return (
    <>
      <div class="run-grid">
        <label class="run-row">
          <span class="run-label">llama.cpp</span>
          <select
            class="one-build"
            aria-label="Build"
            disabled={locked}
            title={locked ? LOCK_REASON : undefined}
            value={builds.activeId}
            onChange={(e) => {
              builds.setActive((e.currentTarget as HTMLSelectElement).value);
            }}
          >
            {builds.installed.length === 0
              ? <option value="">No llama.cpp yet — open the Build tab</option>
              : builds.installed.map((b) => (
                <option
                  key={b.id}
                  value={b.id}
                  selected={b.id === builds.activeId}
                >
                  {b.ref} · {b.backend} · {b.origin}
                </option>
              ))}
          </select>
          {builds.installed.length === 0
            ? (
              <button
                type="button"
                class="btn small"
                t="one-getllama"
                onClick={() => ui.go("build")}
              >
                Get llama.cpp
              </button>
            )
            : null}
        </label>

        <label class="run-row">
          <span class="run-label">Model</span>
          <select
            class="one-model"
            aria-label="Model"
            disabled={locked}
            title={locked ? LOCK_REASON : undefined}
            value={models.selected}
            onChange={(e) => {
              selectModel((e.currentTarget as HTMLSelectElement).value);
              // A context pinned for the previous model means nothing for this
              // one; the re-tune itself is handled by the sync above.
              cfg.setCtxOverride(0);
            }}
          >
            {models.items.length === 0
              ? <option value="">No models found — press Detect</option>
              : models.items.map((x) => (
                // `selected` as well as the select's `value`: the value can be
                // applied before the options exist, and the browser then keeps
                // whichever option happens to be first. That showed one model
                // in the picker while the command below ran another.
                <option
                  key={x.path}
                  value={x.path}
                  selected={x.path === models.selected}
                >
                  {x.file}
                  {x.meta ? ` · ${x.meta.quant} · ${bytes(x.sizeB)}` : ""}
                </option>
              ))}
          </select>
          <button
            type="button"
            class="btn small"
            t="one-detect"
            disabled={models.scanning || locked}
            title={locked ? LOCK_REASON : undefined}
            onClick={() => models.scan()}
          >
            {models.scanning ? "Scanning…" : "Detect"}
          </button>
        </label>

        <div class="run-row">
          <span class="run-label">Runs on</span>
          <div class="ctx-controls">
            <PlacementPicker all={all} locked={locked} />
            <PlacementAdvice all={all} locked={locked} />
          </div>
        </div>

        {
          /* A div, not a <label>: CtxControls holds a dozen buttons and a
            range, and a label wrapping them all sends every click to focus
            the number input. The input carries its own aria-label. */
        }
        <div class="run-row">
          <span class="run-label">Context</span>
          <CtxControls
            ctxNow={ctxNow}
            target={target}
            locked={locked}
            meta={m?.meta ?? null}
            t="one-ctx"
          />
        </div>
      </div>

      <DriftNote />
      <LoadNote />

      <div class="run-actions">
        <StatusBig />
        <Toggle
          checked={cfg.autoOptimal}
          label="Optimal automatically"
          tip="Re-tune for the selected model every time the server starts"
          t="one-auto-optimal"
          onChange={() => cfg.toggleAutoOptimal()}
        />
        <span class="spacer" />
        {running
          ? (
            <button
              type="button"
              class="btn danger"
              t="one-stop"
              title="Stop the server and release its memory"
              onClick={() => stopServer()}
            >
              Stop
            </button>
          )
          : (
            <button
              type="button"
              class="btn primary"
              t="one-start"
              disabled={blocker !== ""}
              title={blocker || `Start ${endpoint()}`}
              onClick={() => startServer()}
            >
              Start server
            </button>
          )}
      </div>

      {m?.metaError
        ? (
          <div class="error-note" t="one-model-error">
            <b>{m.file}</b> — header could not be read:{" "}
            {m.metaError}. Nothing can be planned or started from it; pick
            another model, or re-download this one.
          </div>
        )
        : null}

      {locked
        ? (
          <div class="info-note" t="one-locked">
            {LOCK_REASON} Running <b>{shownModel()?.file ?? "a model"}</b> at
            {" "}
            {ctxNow.toLocaleString()} tokens.
          </div>
        )
        : blocker
        ? <div class="warn-note">{blocker}</div>
        : null}

      {st.level !== "ok"
        ? (
          <div
            class={st.level === "risk" ? "error-note" : "warn-note"}
            t="one-stability"
          >
            {st.warnings[0]?.message}
            {st.warnings.length > 1 ? ` (+${st.warnings.length - 1} more)` : ""}
          </div>
        )
        : null}
    </>
  );
}

/**
 * The three placements, each showing what it would actually give.
 *
 * Not a blind radio group: a placement that cannot run this model says so
 * instead of failing at Start, and one that can says how much context it
 * reaches — which is the whole basis for choosing between them.
 */
/**
 * Minutes of "loading model" with nothing moving reads as a hang.
 *
 * Everything an honest bar needs is already measured every second: the
 * device-wide VRAM drop since the spawn plus the process RSS, against the
 * plan's total for the running command; the server's own log names the phase.
 * The total is an estimate and says so — the movement is what matters.
 */
function LoadNote() {
  const lp = loadingNow();
  if (!lp) return null;
  const m = shownModel();
  return (
    <div class="info-note load-note" t="load-note">
      <div class="load-note-head">
        <span>
          Loading <b>{m?.file ?? "model"}</b> — {lp.phase}
        </span>
        <span class="spacer" />
        <span class="load-note-nums">
          {bytes(lp.loadedB)} of ~{bytes(lp.totalB)} ·{" "}
          {elapsedLabel(Date.now() - lp.startedAt)}
        </span>
      </div>
      <Bar
        value={lp.loadedB}
        max={Math.max(lp.totalB, lp.loadedB)}
        tone="accent"
        height={8}
      />
      {lp.note ? <p class="load-note-fit">{lp.note}</p> : null}
    </div>
  );
}

/**
 * The machine moved after this model was loaded.
 *
 * A running model cannot be re-placed — its weights are where they are — so the
 * only honest thing left is to say so. Two directions, both real on a workstation:
 * something else took memory this server is relying on (a game, another tool's
 * model, a compile), or enough came back that a restart would get materially more.
 *
 * Not shown while idle: there the settings simply re-tune themselves, silently and
 * correctly, and an alarm about memory that has already been adapted to would be
 * noise.
 */
function DriftNote() {
  const d = driftNow();
  if (d.kind === "none") return null;
  const squeezed = d.kind === "squeezed";
  return (
    <div
      class={squeezed ? "error-note drift-note" : "warn-note drift-note"}
      t="drift-note"
    >
      <span>
        {squeezed
          ? (
            <>
              Something else has taken memory since this model started —{" "}
              <b>
                {[
                  d.vramOverB > 0 ? `${bytes(d.vramOverB)} over on VRAM` : "",
                  d.ramOverB > 0 ? `${bytes(d.ramOverB)} over on RAM` : "",
                ].filter(Boolean).join(" and ")}
              </b>. It may fail on the next long prompt, or slow to a crawl.
            </>
          )
          : (
            <>
              Memory has come free since this model started —{" "}
              <b>
                {[
                  d.vramFreeB > 0 ? `${bytes(d.vramFreeB)} of VRAM` : "",
                  d.ramFreeB > 0 ? `${bytes(d.ramFreeB)} of RAM` : "",
                ].filter(Boolean).join(" and ")}
              </b>{" "}
              is idle. A restart would use it.
            </>
          )}
      </span>
      <span class="spacer" />
      <button
        type="button"
        class={squeezed ? "btn tiny danger" : "btn tiny"}
        t="restart-for-drift"
        disabled={srv.status === "stopping"}
        title="Stop the server, re-tune for the machine as it is now, and start again"
        onClick={() => void restartTuned()}
      >
        {srv.status === "stopping" ? "Restarting…" : "Restart for this machine"}
      </button>
    </div>
  );
}

/**
 * "Your GPU is idle and it does not have to be."
 *
 * A persisted placement outlives the reason it was chosen. When the choice on
 * file is beaten by one that actually fits, say so once, with the number that
 * makes the case and the button that takes it — and leave the choice alone
 * otherwise, because CPU only is a legitimate thing to want.
 */
function PlacementAdvice(
  props: { all: Record<Placement, Tuning> | null; locked: boolean },
) {
  const better = betterPlacement(props.all);
  if (!better || props.locked) return null;
  const label = PLACEMENTS.find((p) => p.id === better)?.label ?? better;
  const gain = props.all?.[better];
  return (
    <div class="warn-note placement-advice" t="placement-advice">
      <b>{label}</b> would run this model
      {gain && gain.ctx > 0
        ? ` at ${gain.ctx.toLocaleString()} tokens of context`
        : ""} — the current choice leaves the GPU out.
      <button
        type="button"
        class="btn tiny"
        t="use-better-placement"
        title={`Switch to ${label} and re-tune`}
        onClick={() => {
          cfg.setPlacement(better);
          applyOptimal();
        }}
      >
        Use {label}
      </button>
    </div>
  );
}

function PlacementPicker(
  props: { all: Record<Placement, Tuning> | null; locked: boolean },
) {
  // With a PINNED context a refused placement stays selectable: the refusal
  // is the compute-scratch estimate talking, the estimate is deliberately
  // pessimistic, and it has been measured wrong in this direction (a 512k
  // pin ran where the plan said no). The pin is an instruction and the
  // allocator has the final say at Start — hard-disabling the button made an
  // estimate overrule both.
  const pinned = ctxOverride() > 0;
  return (
    <div class="placements" t="placements">
      {PLACEMENTS.map((p) => {
        const t = props.all?.[p.id] ?? null;
        const dead = t !== null && !t.possible;
        const on = cfg.placement === p.id;
        return (
          <button
            key={p.id}
            type="button"
            class={`placement${on ? " on" : ""}${dead ? " dead" : ""}`}
            t={`placement-${p.id}`}
            disabled={props.locked || (dead && !pinned)}
            title={props.locked
              ? LOCK_REASON
              : dead && pinned
              ? `${t?.blocker} That is the plan's ESTIMATE — the compute scratch cannot be read from the header, and it has been measured pessimistic. You can still select this placement and Start; the allocator has the final say.`
              : t?.blocker || p.tip}
            onClick={() => cfg.setPlacement(p.id)}
          >
            <span class="placement-name">{p.label}</span>
            <span class="placement-sub">
              {t === null
                ? "select a model"
                : t.possible
                ? `${t.ctx.toLocaleString()} ctx${
                  t.ctx >= t.optimalCtx ? " · full" : ""
                }`
                : pinned
                ? "estimate says no — yours to try"
                : "does not fit"}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * The whole catalog, folded away.
 *
 * The all-in-one page is meant to hold most of the settings, not just the four
 * that matter most — but 49 flags open by default is the Tune tab, and this page
 * exists so most sessions never need it. So: closed, one line, and everything is
 * there when it is wanted. Rendered with the SAME `ParamControl` the Tune tab
 * uses, from the same catalog, so the two cannot disagree about what a flag is or
 * what it emits.
 *
 * Disabled while a server is up, for the reason the rest of the strip is: the
 * command on screen has to keep describing the process that is running.
 */
function AllSettings() {
  const changed = changedCount();
  const locked = runLocked();
  const open = ui.showAllSettings;
  return (
    <div class="one-allsettings">
      <div class="one-settings-actions">
        <button
          type="button"
          class="btn small"
          t="one-allsettings"
          onClick={() => ui.toggleAllSettings()}
        >
          {open ? "\u25be" : "\u25b8"} Every llama.cpp setting
          <span class="dim">
            {changed > 0
              ? ` \u00b7 ${changed} changed from default`
              : " \u00b7 all default"}
          </span>
        </button>
        {open
          ? (
            <>
              <Toggle
                checked={cfg.advanced}
                label="Advanced"
                tip="Show the rarely-needed flags"
                t="one-advanced"
                onChange={() => cfg.toggleAdvanced()}
              />
              <span class="spacer" />
              <button
                type="button"
                class="btn small"
                disabled={locked || changed === 0}
                title="Return every flag to the llama.cpp default"
                onClick={() => cfg.reset()}
              >
                Reset all
              </button>
            </>
          )
          : null}
      </div>
      {!open ? null : (
        <fieldset class="one-settings" disabled={locked}>
          {locked ? <p class="param-tip">{LOCK_REASON}</p> : null}
          {GROUPS.map((g) => {
            const list = PARAMS.filter(
              (p) => p.group === g.id && (cfg.advanced || !p.advanced),
            );
            if (list.length === 0) return null;
            return (
              <div class="one-settings-group" key={g.id}>
                <div class="sub-label">{g.label}</div>
                <div class="params">
                  {list.map((p) => <ParamControl key={p.key} p={p} />)}
                </div>
              </div>
            );
          })}
        </fieldset>
      )}
    </div>
  );
}

/** The same chat as the Chat tab, sized for a shared page. */
function MiniChat() {
  const ready = srv.status === "ready";
  const url = endpoint();
  // Same rule as the Chat tab: the reply has to be visible when it arrives.
  const log = useStickyBottom(chat.messages.length);
  return (
    <>
      <div class="one-chatlog" t="one-chat" ref={log}>
        {chat.messages.length === 0 && !chat.partial && !chat.streaming
          ? (
            <Empty
              icon="✉"
              title={ready ? "Ask it something" : "Start the server to chat"}
            />
          )
          : (
            <>
              {chat.messages.map((msg, i) => (
                <div class={`msg msg-${msg.role}`} key={String(i)}>
                  <div class="msg-role">
                    {msg.role}
                    {msg.tps ? ` · ${tps(msg.tps)} tok/s` : ""}
                  </div>
                  <Thinking text={msg.thinking} />
                  <div class="msg-body">
                    {msg.content ||
                      (msg.thinking
                        ? "(the reply ended while still thinking — its reasoning is above)"
                        : msg.content)}
                  </div>
                </div>
              ))}
              {chat.partial || chat.partialThink
                ? (
                  <div class="msg msg-assistant">
                    <div class="msg-role">assistant</div>
                    <Thinking text={chat.partialThink} live={!chat.partial} />
                    <div class="msg-body">{chat.partial}</div>
                  </div>
                )
                : null}
              {chat.streaming && !chat.partial && !chat.partialThink
                ? <Waiting />
                : null}
            </>
          )}
      </div>
      <form
        class="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          if (!ready) return;
          chat.send(url, {
            temp: num(cfg.settings, "temp"),
            topP: num(cfg.settings, "topP"),
          });
        }}
      >
        <input
          placeholder={ready ? "Message" : "Server is not running"}
          aria-label="Quick message"
          disabled={!ready}
          value={chat.input}
          onInput={(e) =>
            chat.setInput((e.currentTarget as HTMLInputElement).value)}
        />
        {chat.streaming
          ? (
            <button
              type="button"
              class="btn danger"
              onClick={() => chat.stop()}
            >
              Stop
            </button>
          )
          : (
            <button
              type="submit"
              class="btn primary"
              disabled={!ready || !canSend()}
            >
              Send
            </button>
          )}
      </form>
    </>
  );
}

export function OnePage() {
  const running = serverRunning();
  const live = memoryIsLive();
  // Now, and next. `currentPlan` describes the running command (or an idle
  // machine); `projected` describes the selected model with our own current
  // usage removed, so a running model is not counted twice.
  const currentPlan = currentStatePlan();
  const projected = projectedStatePlan();
  return (
    <div class="one-page" t="one-page">
      {
        /* Three columns, each one section wide, each the answer to one question:
           what the machine is doing (left), what to run on it (middle), what it
           says back (right). Every panel spans exactly its column — a panel that
           spanned two made the eye track across the fold and back, and the page
           read as a scroll instead of a glance. Each column scrolls alone, so a
           long flag catalog never pushes the chat off screen. */
      }
      <div class="one-col" t="one-col-machine">
        <Panel
          title="Machine"
          icon="▦"
          right={
            <>
              <Pill tone={hw.paused ? "warn" : "ok"}>
                {hw.paused ? "paused" : "live"}
              </Pill>
              <button
                type="button"
                class="btn tiny"
                onClick={() => ui.go("dashboard")}
              >
                Details
              </button>
            </>
          }
        >
          <Vitals />
        </Panel>

        {
          /* Both memory states are on screen at once because they answer
           different questions and the user needs both while choosing — one
           visualisation with a mode switch made whichever question you were
           not currently asking unavailable. */
        }
        <Panel
          title="Current Memory State"
          icon="▤"
          right={
            <Pill tone={live ? "ok" : "idle"}>
              {live ? "a model is running" : "nothing running"}
            </Pill>
          }
        >
          <MemoryMap plan={currentPlan} />
          <MemoryDetail
            plan={currentPlan}
            live={live}
            mode="current"
            compact
            rssB={srv.rssB}
            speed={live && chat.lastTps > 0
              ? { tps: chat.lastTps, measured: true }
              : null}
          />
        </Panel>

        <Panel
          title="Projected Memory State"
          icon="▦"
          right={
            <Pill tone="idle">
              {live ? "after replacing what runs" : "after starting"}
            </Pill>
          }
        >
          {projected
            ? (
              <>
                {live
                  ? (
                    <p class="dim projected-note">
                      Current state with the running model taken back out, plus
                      {" "}
                      <b>{currentModel()?.file}</b>{" "}
                      under these settings — one model runs at a time, so this
                      is what a swap would look like.
                    </p>
                  )
                  : null}
                <MemoryMap plan={projected} />
                <MemoryDetail
                  plan={projected}
                  mode="projected"
                  compact
                  speed={projectedSpeed()}
                />
              </>
            )
            : (
              <Empty
                icon="▢"
                title={currentModel()
                  ? "This model's header could not be read"
                  : "Select a model to see how it would fit"}
                hint={currentModel()
                  ? "Nothing can be projected without it — the Models tab shows why."
                  : "Press Detect if the list is empty."}
              />
            )}
        </Panel>
      </div>

      {
        /* The decision column: everything that changes what runs, with every
          message about the run — the orphan banner, the diagnosis, and the
          server log the diagnosis points at — in the same column, so "the log
          below" is literally below. */
      }
      <div class="one-col" t="one-col-run">
        <OrphanBanner />
        {srv.diagnosis && srv.status === "crashed"
          ? (
            <Guidance
              diagnosis={srv.diagnosis}
              tone="error"
              t="one-srv-failed"
            />
          )
          : <ErrorNote message={srv.lastError || chat.lastError} />}
        <Panel
          title={running ? "Running" : "Run a model"}
          icon="▶"
          right={chat.lastTps > 0
            ? <Pill tone="idle">{tps(chat.lastTps)} tok/s</Pill>
            : null}
        >
          <RunStrip />
          <AllSettings />
        </Panel>
        {
          /* The log takes every remaining pixel of the column — during a long
          load it IS the page, and capping it at ten rows wasted the space
          under it. */
        }
        {srv.log.length > 0 ? <ServerLog fill /> : null}
      </div>

      {
        /* Chat is a column, not a panel at the bottom of a scroll. A reply is the
          thing you are waiting for, so it should be beside the numbers that
          produced it rather than below them — and given the full height, it
          holds a real conversation instead of six lines. */
      }
      <aside class="one-side">
        <Panel title="Chat" icon="✉">
          <MiniChat />
        </Panel>
      </aside>
    </div>
  );
}
