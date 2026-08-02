// src/cell/cfg.ts — the llama.cpp settings the user is editing.
// Browser-safe, and deliberately dumb: it holds a value map and nothing else.
//
// Everything derived from these settings — the two command lines, the memory
// plan, the tuned values — is computed by the pure functions in src/lib and
// called from the UI. No derivation lives here, so there is exactly one
// implementation of each and it is testable without booting a cell.

import { cell } from "aio";
import { coerce, defaults, param } from "../lib/params.ts";
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
  /** This machine's measured effective memory bandwidth, bytes/second, learned
   *  from real generation (`src/lib/speed.ts:calibrate`). 0 = never measured, in
   *  which case a labelled default is used instead. */
  gpuBps: number;
  ramBps: number;
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
    gpuBps: 0,
    ramBps: 0,
    advanced: false,
    touched: [] as string[],
  } as CfgState,
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
