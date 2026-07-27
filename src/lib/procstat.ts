// src/lib/procstat.ts — CPU utilization from two `/proc/stat` samples.
//
// Kept out of the Rust core on purpose: the counters are cumulative, so the
// calculation needs the PREVIOUS sample, which lives in cell state. Making it a
// pure function of (prev, cur) keeps the cell honest and this testable.

/** `cpu  user nice system idle iowait irq softirq steal …` → busy percentage.
 *  Returns 0 for the first sample, a restarted counter, or a malformed line. */
export function utilPct(prev: string, cur: string): number {
  const parse = (s: string) =>
    s.trim().split(/\s+/).slice(1).map((n) => Number.parseInt(n, 10) || 0);
  const a = parse(prev);
  const b = parse(cur);
  if (a.length < 4 || b.length < 4) return 0;
  const total = b.reduce((sum, v, i) => sum + (v - (a[i] ?? 0)), 0);
  if (total <= 0) return 0;
  const idle = (b[3] ?? 0) - (a[3] ?? 0);
  return Math.max(0, Math.min(100, 100 * (1 - idle / total)));
}

/** The same calculation per core, pairing `cpuN` lines by position. */
export function coresUtilPct(prev: string[], cur: string[]): number[] {
  return cur.map((line, i) => utilPct(prev[i] ?? "", line));
}

/** Append to a fixed-length ring used by the sparklines. Returns a NEW array —
 *  cell state is frozen, and a live async proxy has no `Symbol.iterator`. */
export function pushHistory(
  history: readonly number[],
  value: number,
  keep = 60,
): number[] {
  const out = history.slice(Math.max(0, history.length - (keep - 1)));
  out.push(value);
  return out;
}
