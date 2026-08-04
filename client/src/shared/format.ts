// GENERATED — do not edit. Copied from ../../../src/lib/format.ts by
// `deno task sync` (client/sync-shared.ts), because aio serves the browser
// bundle only from inside the app's own root. Edit the original.
// src/lib/format.ts — display helpers. Pure, unit-tested, used by every panel
// so one model size reads identically wherever it appears.

/** Bytes → the unit a human would say out loud. `1.5 GB`, `812 MB`, `0 B`. */
export function bytes(b: number, digits?: number): string {
  if (!Number.isFinite(b) || b <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(
    units.length - 1,
    Math.floor(Math.log(b) / Math.log(1024)),
  );
  const v = b / 1024 ** i;
  const d = digits ?? (i === 0 ? 0 : v < 10 ? 2 : v < 100 ? 1 : 0);
  return `${v.toFixed(d)} ${units[i]}`;
}

/** Bytes → GB with a fixed one-decimal scale, for bars that must line up. */
export function gb(b: number): string {
  return `${(Math.max(0, b) / 1024 ** 3).toFixed(1)}`;
}

export function pct(part: number, whole: number): number {
  if (!(whole > 0)) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}

/** `73%` — for labels; the bar itself uses the unrounded number. */
export function pctLabel(part: number, whole: number): string {
  return `${Math.round(pct(part, whole))}%`;
}

/** Seconds → `4m 12s`, `1h 03m`, `812ms`. */
export function duration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/** Epoch ms → local `YYYY-MM-DD HH:mm`; `—` when unset. */
export function stamp(ms: number): string {
  if (!ms) return "—";
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${
    p(d.getHours())
  }:${p(d.getMinutes())}`;
}

/** Shorten a path for a one-line cell, keeping the informative tail. */
export function shortPath(p: string, max = 48): string {
  if (p.length <= max) return p;
  const parts = p.split("/");
  const file = parts.pop() ?? "";
  let out = file;
  for (let i = parts.length - 1; i >= 0; i--) {
    const next = `${parts[i]}/${out}`;
    if (next.length + 2 > max) break;
    out = next;
  }
  return `…/${out}`;
}

/** Tokens/second with the precision that reads well at both 0.8 and 812. */
export function tps(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "—";
  return n < 10 ? n.toFixed(2) : n < 100 ? n.toFixed(1) : n.toFixed(0);
}
