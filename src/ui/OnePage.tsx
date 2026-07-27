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
import { plan as computePlan } from "../lib/plan.ts";
import { CPU_TJMAX, GPU_TJMAX, tempTone } from "../lib/thermal.ts";
import { bytes, tps } from "../lib/format.ts";
import { num } from "../lib/params.ts";
import {
  applyOptimal,
  currentStability,
  endpoint,
  LOCK_REASON,
  placements,
  runLocked,
  startBlocker,
  startServer,
  stopServer,
} from "./actions.ts";
import { optimalCtx, PLACEMENTS } from "../lib/tune.ts";
import type { Placement, Tuning } from "../lib/tune.ts";
import { MemoryDetail } from "./MemoryDetail.tsx";
import {
  Bar,
  Empty,
  ErrorNote,
  Panel,
  Pill,
  Ring,
  Spark,
  Toggle,
} from "./kit.tsx";
import { MemoryMap } from "./Memory.tsx";
import { Guidance } from "./Guidance.tsx";
import { OrphanBanner, ServerLog, StatusPill } from "./ServerPanel.tsx";
import { useStickyBottom } from "./sticky.ts";
import {
  canSend,
  ctxOverride,
  currentModel,
  hwSnapshot,
  memoryIsLive,
  serverRunning,
  shownModel,
  shownSettings,
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
          value={m ? (m.usedB / m.totalB) * 100 : 0}
          label="RAM"
          tone="ok"
          size={56}
        />
        <div class="vital-body">
          <div class="vital-name">RAM</div>
          <div class="vital-sub">
            {m ? `${bytes(m.usedB)} / ${bytes(m.totalB)}` : "—"}
          </div>
          <Bar
            value={m?.usedB ?? 0}
            max={m?.totalB ?? 0}
            tone={tone(m ? (m.usedB / m.totalB) * 100 : 0)}
            height={7}
          />
          <div class="vital-sub">{m ? `${bytes(m.availableB)} free` : ""}</div>
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
  const key =
    `${models.selected}|${builds.activeId}|${cfg.placement}|${ctxOverride()}`;
  afterRender(() => {
    if (tunedFor.current === key) return;
    if (!cfg.autoOptimal || serverRunning() || !currentModel()?.meta) return;
    tunedFor.current = key;
    applyOptimal();
  });
  const blocker = startBlocker();
  const running = serverRunning();
  const locked = runLocked();
  const st = currentStability();
  const all = placements();
  const target = m?.meta ? optimalCtx(m.meta) : 0;
  // What would actually run: while a server is up, its context; otherwise the
  // context the SELECTED placement reaches. Showing `cfg.settings.ctxSize`
  // instead would display a number left over from the last model until the user
  // happened to press Optimal.
  const ctxNow = locked
    ? num(shownSettings(), "ctxSize")
    // A pinned context is capped at what the model was trained for, in the
    // field as well as in the tuner — showing 64,000 "of 32,768 trained" was
    // displaying a number that could never be used.
    : Math.min(
      ctxOverride() || all?.[cfg.placement]?.ctx ||
        num(shownSettings(), "ctxSize"),
      target || Infinity,
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
              models.select((e.currentTarget as HTMLSelectElement).value);
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
          <PlacementPicker all={all} locked={locked} />
        </div>

        <label class="run-row">
          <span class="run-label">Context</span>
          <div class="field-inline">
            <input
              type="number"
              class="one-ctx"
              aria-label="Context size"
              t="one-ctx"
              min="256"
              max={target || undefined}
              step="256"
              disabled={locked}
              title={locked
                ? LOCK_REASON
                : "Tokens the model can attend to. Blank the override to go back to the model's own maximum."}
              value={String(ctxNow)}
              onChange={(e) =>
                cfg.setCtxOverride(
                  Number((e.currentTarget as HTMLInputElement).value),
                  models.selected,
                )}
            />
            {target > 0
              ? (
                <span class="unit">
                  of {target.toLocaleString()} trained
                  {ctxOverride() > 0
                    ? (
                      <>
                        {" · "}
                        <button
                          type="button"
                          class="btn tiny"
                          t="one-ctx-auto"
                          disabled={locked}
                          onClick={() => cfg.setCtxOverride(0)}
                        >
                          auto
                        </button>
                      </>
                    )
                    : null}
                </span>
              )
              : null}
          </div>
        </label>
      </div>

      <div class="run-actions">
        <StatusPill />
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
function PlacementPicker(
  props: { all: Record<Placement, Tuning> | null; locked: boolean },
) {
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
            disabled={props.locked || dead}
            title={props.locked ? LOCK_REASON : t?.blocker || p.tip}
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
                : "does not fit"}
            </span>
          </button>
        );
      })}
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
        {chat.messages.length === 0 && !chat.partial
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
                  <div class="msg-body">{msg.content}</div>
                </div>
              ))}
              {chat.partial
                ? (
                  <div class="msg msg-assistant">
                    <div class="msg-role">assistant</div>
                    <div class="msg-body">{chat.partial}</div>
                  </div>
                )
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
  // While a server is up this is the model and settings it was STARTED with.
  const shownMeta = shownModel()?.meta ?? null;
  const shownPlan = shownMeta
    ? computePlan(shownMeta, hwSnapshot(), shownSettings())
    : null;
  return (
    <div class="tab-body one-page" t="one-page">
      <OrphanBanner />
      {srv.diagnosis && srv.status === "crashed"
        ? <Guidance diagnosis={srv.diagnosis} tone="error" t="one-srv-failed" />
        : <ErrorNote message={srv.lastError || chat.lastError} />}
      <Panel
        title="Machine"
        icon="▦"
        wide
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

      <Panel
        title={running ? "Running" : "Run a model"}
        icon="▶"
        wide
        right={chat.lastTps > 0
          ? <Pill tone="idle">{tps(chat.lastTps)} tok/s</Pill>
          : null}
      >
        <RunStrip />
      </Panel>

      {
        /* Memory gets its own panel because it is the question this app exists
           to answer, and it needs the room: the picture, then every byte of it
           in words. While a server is up both describe THAT process, computed
           from the command it was started with — not from whatever has since
           been typed into the form. */
      }
      <Panel
        title="Memory"
        icon="▤"
        wide
        right={
          <Pill tone={live ? "ok" : "idle"}>
            {live ? "running now" : "projected"}
          </Pill>
        }
      >
        {shownMeta
          ? (
            <>
              <MemoryMap plan={shownPlan!} />
              <MemoryDetail plan={shownPlan!} live={live} rssB={srv.rssB} />
            </>
          )
          : (
            <Empty
              icon="▢"
              title={shownModel()
                ? "This model's header could not be read"
                : "Select a model to see how it fits"}
              hint={shownModel()
                ? "Nothing can be measured without it — the Models tab shows why."
                : "Press Detect if the list is empty."}
            />
          )}
      </Panel>

      <Panel title="Chat" icon="✉" wide>
        <MiniChat />
      </Panel>

      {
        /* The diagnosis above says "the log below" — so it has to be here, not
          one tab away. Absent until the server has actually said something. */
      }
      {srv.log.length > 0 ? <ServerLog rows={12} /> : null}
    </div>
  );
}
