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
import { models } from "../cell/models.ts";
import { srv } from "../cell/srv.ts";
import { ui } from "../cell/ui.ts";
import { argv, commandLine, serverUrl } from "../lib/command.ts";
import { plan as computePlan } from "../lib/plan.ts";
import type { Plan } from "../lib/plan.ts";
import { bestPlacement, PLACEMENTS, tuneAll } from "../lib/tune.ts";
import type { Placement, Tuning } from "../lib/tune.ts";
import { stability } from "../lib/stability.ts";
import type { Stability } from "../lib/stability.ts";
import {
  activeBuild,
  currentModel,
  hwSnapshot,
  serverRunning,
} from "./derive.ts";

// Re-exported so panels have one import for "the current thing".
export { activeBuild, currentModel };

export function serverBin(): string {
  return activeBuild()?.serverBin ?? "";
}

export function cliBin(): string {
  return activeBuild()?.cliBin ?? "";
}

/** The memory plan for the current model/settings/machine, or null when there
 *  is nothing to plan yet. */
export function currentPlan(): Plan | null {
  const m = currentModel();
  if (!m?.meta) return null;
  return computePlan(m.meta, hwSnapshot(), cfg.settings);
}

export function serverCommand(): string {
  return commandLine("server", {
    bin: serverBin() || "llama-server",
    model: currentModel()?.path ?? "",
    settings: cfg.settings,
  });
}

export function cliCommand(): string {
  return commandLine("cli", {
    bin: cliBin() || "llama-cli",
    model: currentModel()?.path ?? "",
    settings: cfg.settings,
  });
}

export function endpoint(): string {
  return serverUrl(cfg.settings);
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

/** Why the app cannot start a server right now, or "" when it can. */
export function startBlocker(): string {
  if (runLocked()) return LOCK_REASON;
  if (!activeBuild()) {
    return "No llama.cpp build installed — go to the Build tab.";
  }
  if (!serverBin()) return "The active build has no llama-server binary.";
  if (!currentModel()) return "No model selected — scan for models first.";
  return "";
}

/**
 * Every placement for the current model, so the UI can compare them without
 * three separate calls. Null when no model with a readable header is selected.
 */
export function placements(): Record<Placement, Tuning> | null {
  const m = currentModel();
  if (!m?.meta) return null;
  return tuneAll(
    m.meta,
    hwSnapshot(),
    cfg.settings,
    cfg.ctxOverride || undefined,
  );
}

/** What the SELECTED placement gives, or null when there is no model. */
export function currentTuning(): Tuning | null {
  return placements()?.[cfg.placement] ?? null;
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
  if (!all[chosen].possible) {
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
  return { tuning, reasons: [...extra, ...tuning.reasons] };
}

/** Apply the tuner for the selected placement (falling back if it cannot run). */
export function applyOptimal(): void {
  const r = tunedForStart();
  if (!r) return;
  cfg.apply(r.tuning.settings, r.reasons);
}

/** Is the current configuration going to hurt? Recomputed on every render, so
 *  the warning appears the moment a control is changed. */
export function currentStability(): Stability {
  return stability(currentModel()?.meta ?? null, hwSnapshot(), cfg.settings);
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
  });
  return srv.start(command, serverUrl(settings), {
    model: model?.path ?? "",
    settings,
  }).then(() => {});
}

export function stopServer(): Promise<void> {
  return srv.stop().then(() => {});
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
      srv.start([bin, ...argvBefore.slice(1)], urlBefore || endpoint());
    }
  }
}

/** Models table "Run": select, tune, start, and show the server. One gesture,
 *  because that is what "run this model" means to a user. */
export function runModel(path: string): void {
  models.select(path);
  applyOptimal();
  ui.go("server");
  void startServer();
}
