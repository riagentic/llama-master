// src/lib/loadprogress.ts — what "loading model" is actually doing.
//
// A 145 GB model takes minutes to load, and a status that says "loading" for
// minutes reads as a hang. Everything needed for an honest picture is already
// measured every second: how much memory the run has taken so far (the
// device-wide VRAM drop since the spawn, plus the process RSS) against how
// much the plan says the whole run needs — and the server's own log names the
// phase it is in. Pure: numbers and lines in, a picture out.

/** The phases llama-server actually passes through, newest match wins.
 *  Ordered specific-first; scanned from the newest log line backwards. */
const PHASES: readonly { re: RegExp; label: string }[] = [
  { re: /warming up|warmup/i, label: "warming up" },
  {
    re: /compute buffer|graph splits|sched_reserve/i,
    label: "allocating compute buffers",
  },
  { re: /kv cache|KV self size|kv_cache/i, label: "allocating the KV cache" },
  {
    re: /load_tensors|loading model tensors|model buffer/i,
    label: "loading weights",
  },
  {
    re: /llama_model_loader|loading model from|load_model/i,
    label: "reading the header",
  },
];

/** The phase the newest log line places the load in. */
export function loadPhase(lines: readonly string[]): string {
  for (let i = lines.length - 1; i >= 0; i--) {
    const l = lines[i] ?? "";
    for (const p of PHASES) {
      if (p.re.test(l)) return p.label;
    }
  }
  return "starting";
}

export type LoadProgress = {
  /** 0..1 against the plan, or null when the plan has no size to offer. */
  fraction: number | null;
  /** Bytes the run has visibly taken so far — measured, not predicted. */
  loadedB: number;
  /** What the plan says the whole run needs. An estimate, and labelled one. */
  totalB: number;
  /** What the server says it is doing right now, in words. */
  phase: string;
};

/**
 * Progress from measurement, not from a timer.
 *
 * `loadedB` is the device-wide free-VRAM drop since the spawn plus the
 * process RSS: VRAM captures the weights streaming onto the cards, RSS
 * captures the host-side share (and, under mmap, the pages actually read so
 * far). It can over- or under-count transient staging buffers, which is why
 * the total is presented as an estimate — but it moves monotonically with the
 * real work, which is what a progress bar owes the user.
 */
export function loadProgress(a: {
  lines: readonly string[];
  /** Device-wide free VRAM recorded at the moment of the spawn. */
  startFreeVramB: number;
  /** Device-wide free VRAM now. */
  freeVramB: number;
  /** The process's resident set, measured by the poll. */
  rssB: number;
  /** The plan's total for this run, VRAM + RAM buckets. */
  plannedB: number;
}): LoadProgress {
  const vramDelta = a.startFreeVramB > 0
    ? Math.max(0, a.startFreeVramB - a.freeVramB)
    : 0;
  const loadedB = vramDelta + Math.max(0, a.rssB);
  const fraction = a.plannedB > 0
    ? Math.max(0, Math.min(1, loadedB / a.plannedB))
    : null;
  return { fraction, loadedB, totalB: a.plannedB, phase: loadPhase(a.lines) };
}

/** `95000` ms → `"1:35"`. The elapsed-time label. */
export function elapsedLabel(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}
