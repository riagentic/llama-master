// src/ui/RunSync.tsx — the reactions a run needs, whichever page is open.
//
// Headless: it renders nothing. It exists because these four reactions used to
// live in the all-in-one page's run strip, and the shell mounts ONE tab at a
// time — so they only happened while that tab was on screen. Start a model from
// the Models tab's Run (which lands on Server) and the context it proved was
// never written down, a flag the build refused was re-learned on every start
// at the price of a reload, and every other tab's command kept showing a tuning
// for the model before. The app's memory of its own runs cannot depend on which
// page the user happened to be reading, so the shell mounts this once, always.
//
// One rule governs every key below: it is built in the RENDER BODY and closed
// over. A component re-renders only for what its render touched, so a value
// read first inside `afterRender` subscribes to nothing and the reaction fires
// once and never again (CLAUDE.md, "Rules that bite"; `RunStrip` shipped exactly
// that with `chat.lastTps`).

import { afterRender, untrack, useRef } from "aio/air";
import { builds } from "../cell/builds.ts";
import { cfg } from "../cell/cfg.ts";
import { chat } from "../cell/chat.ts";
import { hw } from "../cell/hw.ts";
import { models } from "../cell/models.ts";
import { srv } from "../cell/srv.ts";
import { applyOptimal } from "./actions.ts";
import {
  ctxOverride,
  currentModel,
  headroomNow,
  serverRunning,
  speedCalFromLastReply,
} from "./derive.ts";

export function RunSync() {
  // ── Keep the settings describing what Start would actually run.
  //
  // Keyed on everything the tuning depends on rather than hooked to one
  // dropdown: a model can be selected from the Models tab, from `am`, or
  // restored from the last session, and settings tuned for a different model
  // are wrong however they got there. `afterRender` runs post-commit, so this
  // is a reaction to state rather than a side effect during render, and it
  // settles in one pass because re-tuning does not change the key.
  //
  // Whether the machine has been measured is part of it, because tuning
  // before it has is not a tuning. `models.scan()` and `hw.refresh()` race at
  // boot — measured at 45 ms against 48 ms, a coin flip — and when models won,
  // the tuner saw no RAM and no GPUs, fell back to CPU placement, and
  // `cfg.setPlacement` PERSISTED it. Booleans and a count only: `availableB`
  // moves on every poll and would re-tune once a second.
  //
  // `headroomNow()` is what makes this adaptive: a game taking 20 GB of VRAM,
  // a compile taking 8 GB of RAM, or either of those FINISHING, all change the
  // right answer — in both directions. Coarse on purpose (eighths of each
  // pool, `src/lib/adapt.ts`), or it would rewrite the user's settings on every
  // 1 s poll and fight their typing. The reserve is in the key for the same
  // reason, in exact bytes: it only moves when someone types into the box.
  const hwReady = hw.lastRefresh > 0 && hw.mem !== null;
  const key =
    `${models.selected}|${builds.activeId}|${cfg.placement}|${ctxOverride()}|${hwReady}|${hw.gpus.length}|${headroomNow()}|${cfg.reservePerGpuVramB}:${cfg.reserveConnectedVramB}:${cfg.reserveRamB}`;
  // The gates are read HERE too, not only the key: turning auto-optimal on,
  // a server stopping, or a header finishing its read must each be able to
  // re-run this, and none of them changes the key.
  const canTune = hwReady && cfg.autoOptimal && !serverRunning() &&
    !!currentModel()?.meta;
  const tunedFor = useRef("");
  afterRender(() => {
    if (tunedFor.current === key || !canTune) return;
    tunedFor.current = key;
    // Untracked on purpose: the tuner reads half the app's state, and every
    // part of it that should re-run this is already in `key` above.
    untrack(() => applyOptimal());
  });

  // ── Write down a context that ACTUALLY generated.
  //
  // `proven`, not `healthy`: /health only proves the weights loaded, and a
  // DeepSeek-V4 run passed it at 17,408 tokens then OOM'd on its first prompt
  // — recording at /health wrote that lie down as a fact, and rememberFit
  // only ever grows. For a model whose buffers the planner cannot derive from
  // its header this is the only measured fact the app will ever have
  // (`src/lib/fitladder.ts`). Keyed so it fires once per successful start.
  //
  // A run that walked the ladder is a measurement that the RECORD was too
  // high — the opening bid is capped at the record, and the ladder only
  // engages after that bid actually died — so it replaces rather than grows.
  const fitCtx = Number(srv.runSettings?.ctxSize ?? 0);
  const fitKey = srv.proven && srv.runModel && fitCtx > 0
    ? `${srv.runModel}|${fitCtx}|${srv.startedAt}`
    : "";
  const fit = { model: srv.runModel, ctx: fitCtx, exact: srv.fitTries > 0 };
  const notedFit = useRef("");
  afterRender(() => {
    if (!fitKey || notedFit.current === fitKey) return;
    notedFit.current = fitKey;
    cfg.rememberFit(fit);
  });

  // ── Write down what the build could not do.
  //
  // Recorded the moment the ladder learns it rather than on `proven`: the
  // abort IS the proof, it names the feature in llama.cpp's own words, and it
  // holds whether or not the run that follows succeeds for some other reason.
  // Keyed on the build AND the model, so a newer llama.cpp is never held back
  // by what an older one could not do.
  const refused = srv.unsupported.slice();
  const build = builds.activeId;
  const refusedKey = refused.length > 0 && srv.runModel && build
    ? `${build}|${srv.runModel}|${refused.join(",")}`
    : "";
  const refusal = { build, model: srv.runModel, settings: refused };
  const notedRefusal = useRef("");
  afterRender(() => {
    if (!refusedKey || notedRefusal.current === refusedKey) return;
    notedRefusal.current = refusedKey;
    cfg.rememberUnsupported(refusal);
  });

  // ── Learn this machine's real bandwidth from the rate it just produced.
  //
  // The speed estimate is bandwidth ÷ bytes-per-token, and bandwidth is the
  // one term that cannot be read off the machine — so the app ships a
  // labelled default and replaces it the first time a real generation gives
  // it a rate to work back from. Keyed on BOTH observations: a chat reply and
  // a bench are two rates about the same machine, and the bench wins when it
  // applies (`speedCalFromLastReply`). `lastTps` is written once per FINISHED
  // reply, not per streamed flush, so subscribing costs one render per answer.
  const calKey = `${chat.lastTps}|${srv.lastBench.at}`;
  const calFor = useRef("");
  afterRender(() => {
    if (calFor.current === calKey) return;
    calFor.current = calKey;
    // Untracked: `calKey` is what re-runs this; the rest is the computation.
    const cal = untrack(() => speedCalFromLastReply());
    if (cal.gpuBps || cal.ramBps) cfg.setSpeedCal(cal);
  });

  return null;
}
