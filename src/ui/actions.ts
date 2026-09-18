// src/ui/actions.ts — the few multi-cell gestures the UI offers.
//
// A button that touches three cells (pick a model, tune it, start the server)
// belongs here rather than inline in a panel: the same gesture is offered from
// the models table and from the server panel, and it must mean exactly the same
// thing in both places.
//
// These are thin: they read cell state, call PURE functions from src/lib, and
// dispatch. No logic that deserves a test lives here — it lives in src/lib and
// is tested there.

import { builds } from "../cell/builds.ts";
import { cfg } from "../cell/cfg.ts";
import { chat } from "../cell/chat.ts";
import { hw } from "../cell/hw.ts";
import { models } from "../cell/models.ts";
import { srv } from "../cell/srv.ts";
import { ui } from "../cell/ui.ts";
import { argv, serverUrl } from "../lib/command.ts";
import { num, str } from "../lib/params.ts";
import { availableBackends } from "../lib/assets.ts";
import { compilableBackends, preferredBackends } from "../lib/backend.ts";
import type { Backend, ModelMeta } from "../lib/types.ts";
import { bestPlacement, PLACEMENTS, tune } from "../lib/tune.ts";
import { vetoUnsupported } from "../lib/fitladder.ts";
import type { Placement, Tuning } from "../lib/tune.ts";
import { stability } from "../lib/stability.ts";
import type { Stability } from "../lib/stability.ts";
import {
  activeBuild,
  activeCaps,
  ctxOverride,
  cudaMax,
  currentModel,
  foundPrereqs,
  hwSnapshot,
  maxTunings,
  measuredCtx,
  modelRuntime,
  paramBlocker,
  placements,
  planningHw,
  reserveCost,
  serverRunning,
  shownEnv,
  unsupportedHere,
} from "./derive.ts";

// Re-exported so panels have one import for "the current thing".
export { activeBuild, currentModel };

/**
 * Which backend this machine can actually run, so the default is not a lie.
 *
 * When the asset list has been fetched, a backend with no prebuilt binary is
 * skipped — upstream shipped CUDA for Windows only before b11039, and a
 * release without it must not send the user down a dead end.
 *
 * Which backend suits this hardware is a decision, so it lives in `src/lib`
 * where it is tested (`preferredBackends`); this only intersects it with what is
 * obtainable by the route the user has chosen.
 */
export function suggestedBackend(): Backend {
  const wish = preferredBackends(
    new Set(hw.gpus.map((g) => g.vendor)),
    hw.os || "linux",
  );

  if (builds.origin === "release" && builds.assets.length > 0) {
    const have = availableBackends(
      builds.assets,
      hw.os || "linux",
      hw.arch || "x86_64",
      cudaMax(),
    );
    return wish.find((b) => have.includes(b)) ?? "cpu";
  }
  if (builds.origin === "source") {
    // Suggesting a backend whose toolchain is missing sends the user into a
    // cmake failure four minutes from now.
    const have = compilableBackends(foundPrereqs(), hw.os || "linux");
    return wish.find((b) => have.includes(b)) ?? wish[0] ?? "cpu";
  }
  return wish[0] ?? "cpu";
}

/** One click: the backend this hardware wants, tuned for this exact CPU, with
 *  two cores left to the OS so the machine stays usable during the build. */
export function optimalForThisPc(): void {
  builds.setBackend(suggestedBackend());
  builds.setNative(true);
  builds.setJobs(0);
}

/**
 * Make the first-run default match the hardware.
 *
 * "Build with one click" and "build the optimal thing for this PC" have to be
 * the same click, and they were not: the stored default is `cpu`, so on an
 * NVIDIA machine the one-click Install fetched a CPU release and the user had to
 * know to press "Optimal for this PC" first. This runs once at boot and only
 * while the user has never picked a backend themselves — `suggestBackend`
 * enforces that, so a deliberate choice of `cpu` on a CUDA box is never
 * overridden. Skipped entirely once a build is installed: then the backend
 * follows the build that is active, which is the real state.
 */
export function seedBackend(): void {
  if (builds.installed.length > 0) return;
  builds.suggestBackend(suggestedBackend());
}

export function serverBin(): string {
  return activeBuild()?.serverBin ?? "";
}

export function cliBin(): string {
  return activeBuild()?.cliBin ?? "";
}

export function endpoint(): string {
  return serverUrl(cfg.settings);
}

/**
 * Submit whatever is in the chat box: send it, or hold it until the model is
 * free.
 *
 * A gesture that spans two cells — the sampling comes from `cfg`, the decision
 * and the text from `chat` — which is what this module is for. Both chat
 * surfaces call it, so "what Enter does" is defined once.
 *
 * The branch itself is NOT here: `chat.submit` makes it against the cell's own
 * state in a single dispatch. Deciding it here would mean reading
 * `chat.streaming` from a browser replica that can be one round trip stale, and
 * the cost of getting it wrong is a second request into a live stream.
 */
export function submitChat(url: string, text: string): Promise<boolean> {
  return chat.submit(url, text, {
    temp: num(cfg.settings, "temp"),
    topP: num(cfg.settings, "topP"),
  });
}

/**
 * Is the run configuration locked?
 *
 * One model runs at a time — the server owns the VRAM, and swapping the model
 * under a live process would mean the command on screen no longer describes
 * what is running. So while it is up, the model, the build, the placement and
 * the context are all read-only, and Stop is the way out.
 */
export function runLocked(): boolean {
  return serverRunning();
}

/** Said once, in one place, so every disabled control gives the same reason. */
export const LOCK_REASON = "Stop the server first — one model runs at a time.";

/**
 * Select a model, and void what cannot survive the switch.
 *
 * `--spec-type draft-mtp` is the tuner's decision FOR a model that ships a
 * multi-token-prediction block; against any other model llama.cpp asserts on
 * `n_layer_nextn > 0` and refuses to load. Like a pinned context, the value
 * belongs to the model it was chosen for — carrying it over would be
 * "optimal settings" that do not start. Every UI path that changes the
 * selection goes through here.
 */
export function selectModel(path: string): void {
  models.select(path);
  const meta = models.items.find((m) => m.path === path)?.meta;
  if (str(cfg.settings, "specType") !== "" && (meta?.nextnLayers ?? 0) === 0) {
    cfg.resetOne("specType");
  }
}

/** Why the app cannot start a server right now, or "" when it can. */
export function startBlocker(): string {
  if (runLocked()) return LOCK_REASON;
  if (!activeBuild()) {
    return "No llama.cpp build installed — go to the Build tab.";
  }
  if (!serverBin()) return "The active build has no llama-server binary.";
  const lib = builds.broken[builds.activeId];
  if (lib) {
    return `The active build cannot start: it cannot load ${lib}. It was built by an older version of this app that left its libraries in the build cache, and that cache has since been replaced. Rebuild it on the Build tab, or pick another build.`;
  }
  if (!currentModel()) return "No model selected — scan for models first.";
  const need = modelRuntime();
  if (need) return need.reason;
  // Backstop for a restored session or any selection path around
  // `selectModel`: spawning with a stale `--spec-type` is a server that
  // refuses to load, and with auto-optimal off nothing else would clear it
  // (when it is on, Start re-tunes and the tuner resets the flag itself).
  if (
    !cfg.autoOptimal && str(cfg.settings, "specType") !== "" &&
    paramBlocker("specType") !== ""
  ) {
    return "Speculative decoding is set, but this model ships no multi-token-prediction block — llama.cpp refuses to load. Press Optimal settings, or reset --spec-type in Tune.";
  }
  return "";
}

// `placements` and `measuredCtx` moved to `derive.ts` — they are derived
// values, not gestures, and the projection needs them without this module
// importing itself in a circle. Re-exported so callers keep one import.
export { measuredCtx, placements, reserveCost };

/**
 * A placement that would clearly beat the one currently selected, if there is
 * one.
 *
 * `cfg.placement` is persisted, which is right — it is a choice. But it means a
 * choice made under bad information OUTLIVES the information: the boot race that
 * used to degrade this to `cpu` before the hardware was read left the value
 * stored, so a machine with three idle GPUs kept running on the CPU for every
 * session afterwards with nothing on screen to explain it. The fix stopped it
 * happening; it could not un-store it.
 *
 * So this is advice, not a correction — the user is told and offered the switch,
 * and "CPU only" stays a legitimate thing to want.
 */
export function betterPlacement(
  // Callers that already ran the tuner pass its result in — `placements()` is
  // three binary searches over `plan`, and a page polling at 1 Hz should not
  // run them twice per frame for the same answer.
  all: Record<Placement, Tuning> | null = placements(),
): Placement | null {
  if (!all) return null;
  const best = bestPlacement(all);
  if (best === cfg.placement) return null;
  // Only advise upgrades. Falling back when the choice cannot run is already
  // handled at Start (`tunedForStart`), and saying so twice is nagging.
  const rank: Record<Placement, number> = { vram: 2, hybrid: 1, cpu: 0 };
  if (rank[best] <= rank[cfg.placement]) return null;
  return all[best].possible ? best : null;
}

/**
 * The tuning to actually use: the selected placement, or the fastest one that
 * can run this model when the selected one cannot.
 *
 * Both "Optimal settings" and Start go through this, so the button and the
 * spawn can never disagree about which placement was used. Silently switching
 * would be wrong, so the swap is returned as a reason and shown.
 */
function tunedForStart(): { tuning: Tuning; reasons: string[] } | null {
  const all = placements();
  if (!all) return null;
  let chosen = cfg.placement;
  const extra: string[] = [];
  // NEVER fall back off a pinned context's placement. The refusal at a pin is
  // the compute-scratch ESTIMATE talking, and the estimate has been measured
  // pessimistic (512k ran where it said no) — silently switching a pinned
  // 640k to CPU-only is the app overruling an instruction on a guess. The
  // allocator has the final say at Start; the warning says so.
  if (!all[chosen].possible && !ctxOverride()) {
    const fallback = bestPlacement(all);
    if (all[fallback].possible) {
      extra.push(
        `${PLACEMENTS.find((p) => p.id === chosen)?.label}: ${
          all[chosen].blocker
        } Switched to ${PLACEMENTS.find((p) => p.id === fallback)?.label}.`,
      );
      chosen = fallback;
      cfg.setPlacement(fallback);
    }
  }
  const tuning = all[chosen];
  // Anything this build has already refused for this model comes back out
  // before the settings are applied — otherwise auto-optimal proposes it on
  // every start and the ladder pays a whole reload to drop it again.
  const veto = vetoUnsupported(tuning.settings, unsupportedHere());
  return {
    tuning: { ...tuning, settings: veto.settings },
    reasons: [...extra, ...tuning.reasons, ...veto.reasons],
  };
}

/** Apply the tuner for the selected placement (falling back if it cannot run). */
export function applyOptimal(): void {
  const r = tunedForStart();
  if (!r) return;
  cfg.apply(r.tuning.settings, r.reasons);
}

/**
 * The largest context a placement can hold on this machine, hunted to the
 * model's ADVERTISED maximum — what the "Max on VRAM / Max on Hybrid"
 * buttons offer, computed the same way they would run.
 */
export function maxFor(placement: Placement): Tuning | null {
  if (placement !== "cpu") return maxTunings()[placement];
  const m = currentModel();
  if (!m?.meta) return null;
  return tune(
    m.meta,
    planningHw(),
    cfg.settings,
    placement,
    undefined,
    undefined,
    true,
  );
}

/**
 * One click for the priority most sessions actually have: THIS placement, at
 * the biggest context it can hold. Sets the placement, pins the hunted
 * context, and re-tunes so the settings, the projection and the command all
 * describe the same run. The pin means the ladder stays out of it — this is
 * the user stating their priority, and the measured-boundary warning on the
 * context control covers the part arithmetic cannot see.
 */
export function pinMaxFor(placement: Placement): void {
  const t = maxFor(placement);
  if (!t || !t.possible || t.ctx <= 0) return;
  cfg.setPlacement(placement);
  cfg.setCtxOverride(t.ctx, models.selected);
  applyOptimal();
}

/** Is the current configuration going to hurt? Recomputed on every render, so
 *  the warning appears the moment a control is changed. */
export function currentStability(): Stability {
  // `lowPriority` is not a llama.cpp flag, so it is not in the catalog — but it
  // decides whether "every core is claimed" is a problem or the intended
  // answer, so the check has to be told about it.
  return stability(currentModel()?.meta ?? null, planningHw(), cfg.settings, {
    lowPriority: cfg.lowPriority,
  });
}

/**
 * Start llama-server with exactly the command the UI is showing.
 *
 * When "Optimal automatically" is on (the default), the settings are re-tuned
 * for the selected model FIRST — the whole point of an auto switch is that
 * pressing Start on a new model does not run the previous model's flags. The
 * tuned values are used from the tuner's RETURN value and also published to
 * `cfg`, so the command that runs and the command on screen cannot disagree;
 * reading `cfg.settings` back after the dispatch could still see the old value
 * on a browser client.
 *
 * Returns the promise rather than dropping it: a click can ignore it, but
 * `updateNow` and the tests need to know when the spawn has actually happened.
 */
export function startServer(): Promise<void> {
  if (startBlocker()) return Promise.resolve();
  const model = currentModel();
  let settings = cfg.settings;
  if (cfg.autoOptimal && model?.meta) {
    // The same path "Optimal settings" takes, fallback included: starting must
    // not spawn a placement the tuner has already established cannot run.
    const r = tunedForStart();
    if (r) {
      settings = r.tuning.settings;
      cfg.apply(r.tuning.settings, r.reasons);
    }
  }
  const command = argv("server", {
    bin: serverBin(),
    model: model?.path ?? "",
    settings,
    // What THIS build understands. A flag it has never heard of is not an
    // ignored setting — llama-server exits with `unknown argument` before it
    // reads the model path (`command.ts:emitFor`).
    caps: activeCaps(),
  });
  return srv.start(command, serverUrl(settings), {
    model: model?.path ?? "",
    settings,
    env: shownEnv(),
    freeAtStart: freeNowB(),
    // The ladder is only for settings the APP chose. A context the user typed
    // is an instruction, and halving it because it did not fit would be the app
    // overruling them silently (`src/lib/fitladder.ts`).
    autoFit: cfg.autoOptimal && !ctxOverride(),
    // A pinned context does not pin the MICRO-BATCH. The tuner raised `-ub`
    // above 512 to spend VRAM it thought was spare; if the load then dies for
    // want of compute buffer, it may take that back without shortening the
    // context the user asked for (`fitladder.ts:autoUbatch`).
    autoUbatch: cfg.autoOptimal,
    lowPriority: cfg.lowPriority,
    shape: modelShape(model?.meta ?? null),
    // The cards llama.cpp will see, not every card the machine has. Both readers
    // of this index into it by a position that excludes the others: the fit
    // ladder reads `CUDA1` out of llama.cpp's own error, and `plan` lines it up
    // with `hwSnapshot().gpus`. Taken from the raw `hw.gpus` it was a list with
    // this machine's AMD iGPU in it — harmless while the NVIDIA cards happen to
    // sort first, and an off-by-one attributing card 1's memory to card 0 the
    // moment they do not.
    cardFreeB: hwSnapshot().gpus.map((g) =>
      Math.max(0, g.vramTotalB - g.vramUsedB)
    ),
  }).then(() => {});
}

/**
 * What the fit ladder needs to answer a weights overflow.
 *
 * The typical routed-expert weight of ONE layer, which is what `--n-cpu-moe`
 * moves per step. The median rather than the mean: a MoE model usually has a
 * few dense layers at the front with no experts at all, and averaging those in
 * would under-size every step of the ladder and turn one rung into three — each
 * of which reloads the whole model.
 */
export function modelShape(
  meta: ModelMeta | null,
): { nLayer: number; expertPerLayerB: number } {
  if (!meta) return { nLayer: 0, expertPerLayerB: 0 };
  const experts = meta.layers.map((l) => l.expert).filter((b) => b > 0).sort((
    a,
    b,
  ) => a - b);
  const mid = experts.length > 0
    ? experts[Math.floor(experts.length / 2)] ?? 0
    : 0;
  return { nLayer: meta.nLayer, expertPerLayerB: mid };
}

/** Device-wide free memory right now — the baseline a run's drift note
 *  measures against. Nothing of ours runs when this is read (Start is blocked
 *  while a server is up), so device-wide IS "everyone else". */
function freeNowB(): { vramB: number; ramB: number } {
  return {
    vramB: hw.gpus.reduce((a, g) => a + (g.vramTotalB - g.vramUsedB), 0),
    ramB: hw.mem?.availableB ?? 0,
  };
}

export function stopServer(): Promise<void> {
  return srv.stop().then(() => {});
}

/**
 * Stop, re-tune for the machine as it is now, and start again — the drift
 * note's button.
 *
 * Deliberately NOT `stop(); applyOptimal(); startServer()`: `startServer`
 * consults `startBlocker`, which reads `srv.status` — and on a browser client
 * a read straight after the awaited stop can still say "ready", turning the
 * restart into a silent no-op with the server left down. Everything here is
 * decided from return values and the tuner's output; the only status guard
 * left is `srv.start`'s own, which runs against the cell's authoritative
 * state rather than a possibly-stale replica. Explicitly re-tunes regardless
 * of the auto-optimal toggle — adapting to the machine is what the button
 * says it does.
 */
export async function restartTuned(): Promise<void> {
  const model = currentModel();
  const bin = serverBin();
  if (!model || !bin) return;
  await srv.stop();
  let settings = cfg.settings;
  const r = tunedForStart();
  if (r) {
    settings = r.tuning.settings;
    cfg.apply(r.tuning.settings, r.reasons);
  }
  const command = argv("server", {
    bin,
    model: model.path,
    settings,
    caps: activeCaps(),
  });
  await srv.start(command, serverUrl(settings), {
    model: model.path,
    settings,
    env: shownEnv(),
    freeAtStart: freeNowB(),
    // The same run context `startServer` records, for the same reasons — this
    // is the restart most likely to meet a machine that just changed, so it
    // needs the fit ladder's weights rung (`shape`) and the per-card baseline
    // (`cardFreeB`) MORE than a plain start does, not less. Without them a
    // weights overflow fell through to the context rung, which cannot move a
    // single model byte, and the live memory map lost its measured per-card
    // attribution. The ladder stays reserved for settings the APP chose: only
    // when the re-tune actually produced them, and never over a typed pin.
    autoFit: r !== null && !ctxOverride(),
    autoUbatch: r !== null,
    lowPriority: cfg.lowPriority,
    shape: modelShape(model.meta ?? null),
    cardFreeB: hwSnapshot().gpus.map((g) =>
      Math.max(0, g.vramTotalB - g.vramUsedB)
    ),
  });
}

/**
 * Take the update: rebuild or re-download the active build at the newest
 * upstream version, and put the server back exactly as it was.
 *
 * The restart is the part that makes this a button rather than a chore — the
 * binary the running server is executing is about to be replaced, so it has to
 * come down first and go back up afterwards, with the same command.
 */
export async function updateNow(): Promise<void> {
  const wasRunning = serverRunning();
  const argvBefore = srv.argv.slice();
  const urlBefore = srv.url;
  // Carry the run's identity across the restart: without it the memory view
  // stops describing the live process the moment an update brings it back up.
  // The model's shape rides along for the fit ladder's weights rung — captured
  // HERE because `srv.stop` clears `runModel` and the update takes minutes.
  const runBefore = srv.runSettings
    ? { model: srv.runModel, settings: srv.runSettings }
    : undefined;
  const shapeBefore = modelShape(
    models.items.find((m) => m.path === srv.runModel)?.meta ?? null,
  );
  // The run's own ladder policy and priority, not re-derived: the run being
  // resumed already settled whether its settings were app-chosen (ladder
  // allowed) or typed, and which priority it was started at. Omitting
  // `lowPriority` here silently defaulted the resumed run to low even when the
  // run it replaces had the switch off.
  const autoFitBefore = srv.autoFit;
  const lowPriorityBefore = srv.runLowPriority;
  // And the environment the run carried — same reason: the resumed run is the
  // run that was, and its variables are part of what it was.
  const envBefore = srv.runEnv;

  if (wasRunning) await srv.stop();
  // The RETURN value, not `builds.job`: a state read straight after an await
  // can still hold the previous value on a browser client, which would skip
  // the restart after a successful update (aiol flags exactly this).
  const status = await builds.update();

  // Only come back up if the update actually produced a working build.
  if (wasRunning && status === "done") {
    const bin = serverBin();
    if (bin && argvBefore.length > 0) {
      // The binary path changes with the ref; everything after it does not.
      // The run context is as complete as `startServer`'s: without `shape` and
      // `cardFreeB` a weights overflow on the way back up fell through to the
      // context rung, and the memory map lost its per-card measurement. Free
      // memory is read NOW — the update took minutes, and the machine moved.
      srv.start(
        [bin, ...argvBefore.slice(1)],
        urlBefore || endpoint(),
        runBefore && {
          ...runBefore,
          env: envBefore ?? undefined,
          freeAtStart: freeNowB(),
          autoFit: autoFitBefore,
          lowPriority: lowPriorityBefore,
          shape: shapeBefore,
          cardFreeB: hwSnapshot().gpus.map((g) =>
            Math.max(0, g.vramTotalB - g.vramUsedB)
          ),
        },
      );
    }
  }
}

/** Models table "Run": select, tune, start, and show the server. One gesture,
 *  because that is what "run this model" means to a user. */
export function runModel(path: string): void {
  selectModel(path);
  applyOptimal();
  ui.go("server");
  void startServer();
}
