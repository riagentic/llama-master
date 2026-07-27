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
