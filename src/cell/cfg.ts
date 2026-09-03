// src/cell/cfg.ts — the llama.cpp settings the user is editing.
// Browser-safe, and deliberately dumb: it holds a value map and nothing else.
//
// Everything derived from these settings — the two command lines, the memory
// plan, the tuned values — is computed by the pure functions in src/lib and
// called from the UI. No derivation lives here, so there is exactly one
// implementation of each and it is testable without booting a cell.

import { cell } from "aio";
import { coerce, defaults, param } from "../lib/params.ts";
import {
  DEFAULT_RESERVE_CONNECTED_VRAM_B,
  DEFAULT_RESERVE_PER_GPU_VRAM_B,
  DEFAULT_RESERVE_RAM_B,
  reserveBytes,
} from "../lib/reserve.ts";
import type { Placement } from "../lib/tune.ts";
import type { ParamValue, Settings } from "../lib/types.ts";

export type CfgState = {
  settings: Settings;
  /** Why the last "Optimal settings" run chose what it chose. */
  reasons: string[];
  /** Where the model runs: VRAM only, Hybrid, or CPU only. There is one set of
   *  optimal settings; this is the only placement choice. */
  placement: Placement;
  /** A context the user typed, or 0 for "aim at the model's trained maximum".
   *  Kept apart from `settings.ctxSize` so switching model does not inherit a
   *  number chosen for a different one. */
  ctxOverride: number;
  /** The model the override was chosen for. An override is an instruction about
   *  ONE model — carrying 128k onto a model trained for 32k silently caps it,
   *  with nothing on screen saying why — so it is void for any other model.
   *  Stored rather than cleared on switch because the reset has to hold however
   *  the model changed: the picker, the Models tab, `am`, or a restored session. */
  ctxOverrideFor: string;
  /** Re-tune for the selected model every time the server starts.
   *
   *  On by default: a first-time user should get good settings without knowing
   *  that a tuner exists, and the settings that are right for one model are
   *  usually wrong for the next. Switchable off, because someone who has hand-
   *  tuned a command does not want it rewritten under them. */
  autoOptimal: boolean;
  /** The largest context that has ACTUALLY started, per model path.
   *
   *  For most models the plan is arithmetic and this stays empty. For the ones
   *  whose buffers cannot be derived from the header — a sparse-attention MoE
   *  wanted a 68 GiB compute buffer where `plan.ts` predicted 730 MB — it is the
   *  only honest source: the app tries, steps down when the memory refuses, and
   *  writes down what worked so the next run of that model opens there instead
   *  of walking the ladder again (`src/lib/fitladder.ts`). A ceiling, never a
   *  target: the tuner still has to fit it in the memory that is free now. */
  fitCtx: Record<string, number>;
  /** Settings a given build has PROVEN it cannot honour for a given model.
   *
   *  Keyed `<build id>\n<model path>`, because it is a fact about the pair and
   *  about neither alone: the header genuinely declares the feature (so every
   *  other model of that architecture is unaffected), and the build genuinely
   *  lacks it (so the same model works the moment support lands, and must not
   *  be held back by something this app wrote down in August).
   *
   *  Learned the only way it can be — by asking. GLM-5.3-Flash ships a NextN
   *  block, so the tuner turned multi-token prediction on; the loader for that
   *  architecture aborted with "NextN graph not implemented yet". The ladder
   *  drops the flag and the run continues, and this is what stops the next
   *  start paying a 157 GB reload to discover it again. */
  unsupported: Record<string, string[]>;
  /** This machine's measured effective memory bandwidth, bytes/second, learned
   *  from real generation (`src/lib/speed.ts:calibrate`). 0 = never measured, in
   *  which case a labelled default is used instead. */
  gpuBps: number;
  ramBps: number;
  /** Memory to keep out of every plan, for the user's own work.
   *
   *  Not a llama.cpp flag, so deliberately not in `settings`: the catalog is the
   *  one place a flag is declared and this emits nothing on the command line. It
   *  is a claim on the machine, and it is honoured by shrinking what any plan may
   *  spend (`src/lib/reserve.ts`). Bytes, because every other memory figure in
   *  this app is bytes; the controls speak GB.
   *
   *  Two VRAM figures, and on a card that qualifies for both they ADD: one is
   *  charged to every card, the other only to the card(s) driving a display —
   *  which is where a desktop's VRAM actually goes, and is usually one card of
   *  several. Named apart from the single `reserveVramB` they replace, so a
   *  machine that had 4 GB set machine-wide does not silently reappear as 4 GB
   *  on every card. */
  reservePerGpuVramB: number;
  reserveConnectedVramB: number;
  reserveRamB: number;
  /**
   * Run llama-server at the lowest OS priority, so the machine stays usable
   * while it works. ON by default: a model that makes the desktop stutter is a
   * model the user turns off. Not a llama.cpp flag — it is applied to the
   * process after the spawn (`src/lib/priority.ts`) — so it is not in
   * `settings`, for the same reason the reserve is not.
   */
  lowPriority: boolean;
  /** Show the rarely-needed flags. */
  advanced: boolean;
  /** Settings the user has changed away from the llama.cpp default. */
  touched: string[];
};

export const cfg = cell("cfg", {
  state: {
    settings: defaults(),
    reasons: [] as string[],
    // The fastest placement, and the app corrects it the moment a model turns
    // out not to fit (see `applyOptimal`).
    placement: "vram" as Placement,
    ctxOverride: 0,
    ctxOverrideFor: "",
    autoOptimal: true,
    fitCtx: {} as Record<string, number>,
    unsupported: {} as Record<string, string[]>,
    gpuBps: 0,
    ramBps: 0,
    reservePerGpuVramB: DEFAULT_RESERVE_PER_GPU_VRAM_B,
    reserveConnectedVramB: DEFAULT_RESERVE_CONNECTED_VRAM_B,
    reserveRamB: DEFAULT_RESERVE_RAM_B,
    lowPriority: true,
    advanced: false,
    touched: [] as string[],
  } as CfgState,
  /**
   * 2 — the single machine-wide `reserveVramB` became a per-GPU figure and a
   * connected-GPU figure (`src/lib/reserve.ts`).
   *
   * A rename of a PERSISTED field is a migration whether or not one is written:
   * without this, aio deep-merges the stored blob over the defaults, keeps the
   * orphaned key forever and says so on every boot — "shape drift: 1 stored
   * field(s) no longer match the declared shape". That warning is the framework
   * asking a question, and the answer here is that the old value cannot be
   * carried forward honestly: it meant "hold this much across the whole
   * machine, divided between the cards", which is not what either new field
   * means. Writing it into `connectedB` would move memory the user never asked
   * to move onto one card; writing it into `perGpuB` would multiply it by the
   * number of cards. So it is dropped, and the new defaults (0 per card, 8 GB
   * on the display card) take over — a visible control, on two pages, that says
   * what it is holding.
   */
  version: 3,
  onMigrate(state: CfgState, from: number): CfgState {
    // Cast to delete a key that is no longer IN the type — which is the whole
    // point of a migration, and the only place in this app allowed to say it.
    if (from < 2) {
      delete (state as unknown as Record<string, unknown>).reserveVramB;
    }
    if (from < 3) {
      // Two catalog entries left: `noContextShift` (replaced by `contextShift`,
      // whose default runs the same server the old default did) and
      // `defragThold` (deprecated upstream — the value is ignored and the flag
      // only prints a warning). Not carried forward: a stored value for a flag
      // that no longer exists would sit in the map forever and never be shown.
      for (const key of ["noContextShift", "defragThold"]) {
        delete state.settings?.[key];
      }
      if (Array.isArray(state.touched)) {
        state.touched = state.touched.filter((k) =>
          k !== "noContextShift" && k !== "defragThold"
        );
      }
    }
    return state;
  },
  methods: {
    toggleAutoOptimal(s) {
      s.autoOptimal = !s.autoOptimal;
    },

    /**
     * Write down a context this model has actually GENERATED at (`srv.proven`).
     *
     * Grows by default: a run that generated at 32,768 proves 32,768 works,
     * and a later run that settled for 8,192 by choice proves nothing about
     * the model.
     *
     * `exact` replaces instead, and the caller sets it when this run WALKED
     * THE LADDER (`srv.fitTries > 0`): the ladder only engages after a start
     * actually died at the opening bid, and the opening bid is capped at the
     * recorded value — so a ladder run is a measurement that the record
     * itself is too high. Keeping the maximum then would re-run that crash at
     * the top of every session: a 17,408 recorded at /health (before `proven`
     * existed) could never answer a prompt, and grow-only kept it forever.
     */
    rememberFit(s, at: { model: string; ctx: number; exact?: boolean }) {
      if (!at.model || at.ctx <= 0) return;
      if (!at.exact && (s.fitCtx[at.model] ?? 0) >= at.ctx) return;
      s.fitCtx[at.model] = at.ctx;
    },

    /**
     * Write down a capability this build does not have for this model.
     *
     * Grow-only within its key, and the key carries the build id — so this can
     * never outlive the build that earned it. A newer llama.cpp is a different
     * id and starts with a clean sheet, which is the whole point: "not
     * implemented yet" is a statement with an expiry date.
     */
    rememberUnsupported(
      s,
      at: { build: string; model: string; settings: readonly string[] },
    ) {
      if (!at.build || !at.model || at.settings.length === 0) return;
      const key = `${at.build}\n${at.model}`;
      const had = s.unsupported[key] ?? [];
      const next = at.settings.filter((k) => !had.includes(k));
      if (next.length === 0) return;
      s.unsupported[key] = [...had, ...next];
    },

    /** Forget it, when a start fails for want of memory at that very size. */
    forgetFit(s, model: string) {
      if (model in s.fitCtx) delete s.fitCtx[model];
    },
    /**
     * Record what this machine actually achieved.
     *
     * Speed is estimated from bandwidth ÷ bytes-per-token, and bandwidth is the
     * one term that cannot be read off the machine — `nvidia-smi` does not report
     * bus width, and the achieved fraction depends on the kernel anyway. So the
     * app starts from a labelled default and replaces it the first time a real
     * reply gives it a rate to work back from. Only a run living almost entirely
     * in one pool says anything about that pool; `calibrate` returns nothing for
     * a hybrid run rather than blaming one side.
     */
    setSpeedCal(s, cal: { gpuBps?: number; ramBps?: number }) {
      if (cal.gpuBps && cal.gpuBps > 0) s.gpuBps = cal.gpuBps;
      if (cal.ramBps && cal.ramBps > 0) s.ramBps = cal.ramBps;
    },
    /** Set one parameter from a UI control. The raw value is coerced and
     *  clamped by the catalog, so state can never hold NaN or an out-of-range
     *  number no matter what the input element produces. */
    set(s, key: string, raw: string | boolean) {
      const p = param(key);
      if (!p) {
        // Fail loud: a typo'd key would otherwise write a field nothing reads.
        throw new Error(`[cfg] unknown parameter "${key}"`);
      }
      const value = coerce(p, raw);
      s.settings[key] = value;
      const isDefault = value === p.def;
      const touched = s.touched.filter((k) => k !== key);
      if (!isDefault) touched.push(key);
      s.touched = touched;
    },
    /** Apply a whole patch (the tuner's output, or a preset). */
    apply(s, patch: Settings, reasons: string[] = []) {
      const next: Settings = { ...s.settings, ...patch };
      s.settings = next;
      s.reasons = reasons;
      s.touched = Object.keys(next).filter((k) => {
        const p = param(k);
        return p ? next[k] !== p.def : false;
      });
    },
    reset(s) {
      s.settings = defaults();
      s.reasons = [];
      s.touched = [];
      // The pin is a setting the user typed, and it was just wiped out of
      // `settings` with everything else — kept here it would survive as a
      // hidden instruction, silently capping the next tune after a Reset that
      // promised a clean slate.
      s.ctxOverride = 0;
      s.ctxOverrideFor = "";
    },
    resetOne(s, key: string) {
      const p = param(key);
      if (!p) throw new Error(`[cfg] unknown parameter "${key}"`);
      s.settings[key] = p.def;
      s.touched = s.touched.filter((k) => k !== key);
    },
    setPlacement(s, placement: Placement) {
      s.placement = placement;
    },
    /** Pin the context for ONE model, or pass 0 to go back to its trained
     *  maximum. The tuner treats a pinned value as an instruction, not a
     *  suggestion — so it must not outlive the model it was typed for. */
    setCtxOverride(s, ctx: number, forModel = "") {
      s.ctxOverride = Number.isFinite(ctx) && ctx > 0 ? Math.floor(ctx) : 0;
      s.ctxOverrideFor = s.ctxOverride > 0 ? forModel : "";
      // A pin is an instruction, so it goes straight into the settings the
      // command is composed from. With auto-optimal ON the tuner re-applies
      // it anyway; with it OFF nothing else would, and the command preview,
      // the projection and the spawn all disagreed with the number on the
      // pin. Same coerce-and-track steps as `set`, so the catalog's clamp and
      // the changed-count stay honest.
      if (s.ctxOverride > 0) {
        const p = param("ctxSize");
        if (p) {
          const value = coerce(p, String(s.ctxOverride));
          s.settings.ctxSize = value;
          const touched = s.touched.filter((k) => k !== "ctxSize");
          if (value !== p.def) touched.push("ctxSize");
          s.touched = touched;
        }
      }
    },
    /**
     * Set a reserve from its control, in GB.
     *
     * Clamped here rather than at the input, for the same reason `set` coerces
     * through the catalog: a number box can produce a blank, a minus sign or
     * `1e30`, and a NaN reserve would poison every fit test it reaches — quietly,
     * because NaN comparisons are all false (`src/lib/plan.ts:whole`).
     */
    setReserve(s, pool: "gpu" | "connected" | "ram", gb: number) {
      const b = reserveBytes(gb);
      if (pool === "gpu") s.reservePerGpuVramB = b;
      else if (pool === "connected") s.reserveConnectedVramB = b;
      else s.reserveRamB = b;
    },
    toggleLowPriority(s) {
      s.lowPriority = !s.lowPriority;
    },
    toggleAdvanced(s) {
      s.advanced = !s.advanced;
    },
    clearReasons(s) {
      s.reasons = [];
    },
  },
  selectors: {
    value: (s, key: string): ParamValue =>
      s.settings[key] ?? param(key)?.def ?? "",
    isTouched: (s, key: string) => s.touched.includes(key),
    changedCount: (s) => s.touched.length,
  },
});
