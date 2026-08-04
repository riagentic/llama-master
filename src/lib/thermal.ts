// src/lib/thermal.ts — the temperatures a bar is drawn against.
//
// These are THROTTLE points, not destruction points: a modern x86 package
// starts losing clocks around 95 °C and a consumer GPU around 83 °C. A bar that
// fills up therefore means "about to lose performance", which is the reading
// that matters when you are deciding how hard to push a model.
//
// One definition, because two copies of a threshold that decides when a bar
// turns red will drift, and a dashboard that disagrees with itself is worse
// than one that is wrong consistently.

export const CPU_TJMAX = 95;
export const GPU_TJMAX = 83;

/** Tone for a temperature reading: unknown, fine, warm, or throttling. */
export function tempTone(
  c: number,
  max: number,
): "idle" | "ok" | "warn" | "bad" {
  if (!(c > 0)) return "idle";
  const pct = (c / max) * 100;
  return pct >= 95 ? "bad" : pct >= 82 ? "warn" : "ok";
}

/**
 * Tone for a "how full / how busy" reading, in quarters.
 *
 * 0-25 cyan, 25-50 green, 50-75 amber, 75-100 red — the same four steps for
 * CPU, GPU, VRAM and RAM, because on this page they are asked the same
 * question and an eye scanning four dials should not have to learn four
 * scales. Cyan rather than green at the bottom so "idle" and "comfortable"
 * are not the same colour: on a machine that is about to load a model, the
 * difference between 5% and 40% of a card is the whole decision.
 *
 * Here, beside `tempTone`, because these are the two thresholds that decide
 * what colour a dial turns, and two copies of that rule would drift.
 */
export function loadTone(pct: number): "busy" | "ok" | "warn" | "bad" {
  if (!Number.isFinite(pct) || pct < 25) return "busy";
  if (pct < 50) return "ok";
  if (pct < 75) return "warn";
  return "bad";
}
