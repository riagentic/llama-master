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
    advanced: false,
    touched: [] as string[],
  } as CfgState,
  methods: {
    toggleAutoOptimal(s) {
      s.autoOptimal = !s.autoOptimal;
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
