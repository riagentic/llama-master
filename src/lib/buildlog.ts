// src/lib/buildlog.ts — read progress out of a build's own output.
//
// A cmake/ninja build already reports how far along it is; parsing that is
// strictly better than a spinner, and it costs one regex. Pure so the shape of
// every line the app relies on is pinned by a test.

/** `[ 42%] Building CXX object …` → 0.42. Null when the line has no progress. */
export function progressOf(line: string): number | null {
  const m = /^\s*\[\s*(\d{1,3})%\]/.exec(line);
  if (!m) return null;
  const pct = Number(m[1]);
  if (!Number.isFinite(pct)) return null;
  return Math.max(0, Math.min(1, pct / 100));
}

/** True for lines worth surfacing above the log tail (errors, not warnings). */
export function isError(line: string): boolean {
  return /(^|\s)(error|Error|ERROR|fatal error|undefined reference)[: ]/.test(
    line,
  );
}

/** Append to a bounded log tail. Returns a NEW array — see procstat.ts. */
export function appendLog(
  log: readonly string[],
  lines: string[],
  keep = 400,
): string[] {
  const out = log.slice(Math.max(0, log.length + lines.length - keep));
  out.push(...lines);
  return out;
}
