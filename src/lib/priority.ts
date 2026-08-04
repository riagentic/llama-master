// src/lib/priority.ts — letting the desktop go first.
//
// A model that fills the machine is the point of this app; a machine that
// stops answering the mouse while it does is not. llama-server will happily
// take every core and every spare IOPS, and on the box it is running on that
// reads as "the computer is broken" — the window manager stutters, the editor
// waits, the browser drops frames.
//
// The fix is old and boring: run it at the lowest scheduling priority and in
// the idle I/O class, so it gets everything nobody else wants and yields the
// moment anybody else asks. Generation slows by a few percent when the machine
// is otherwise busy, and by nothing at all when it is not — which is the whole
// trade, and it is why this is ON by default.
//
// How, and why not the obvious way: the priority is applied to the process
// AFTER it is spawned (`renice`/`ionice` by pid) rather than by wrapping the
// command in `nice`. Wrapping would put `/usr/bin/nice` at the front of the
// argv, which breaks two things this app has promised — the command shown on
// screen would no longer be the command that runs, and `srv.server.ts` refuses
// any binary outside the builds root, which is a sandbox rule and not a
// formality. Reniced-after-spawn keeps both, at the cost of a few milliseconds
// at normal priority during a load that takes a minute.
//
// Pure: which commands to run for a pid, and how to read the result back.

/** The politest nice value Linux has. 19 is "only when nothing else wants the
 *  CPU"; it is also the highest an unprivileged process may set, which matters
 *  because raising it back is what needs root, not lowering it. */
export const NICE_LEVEL = 19;

/** Best-effort I/O priority, worst class. 7 is the lowest band inside the
 *  best-effort class — used when the idle class is refused, which some kernels
 *  and containers still do. */
export const IO_BEST_EFFORT_LOWEST = 7;

export type PriorityStep = { cmd: string; args: string[]; what: string };

/**
 * The commands that make a running process yield to everything else.
 *
 * Two, because CPU and disk are two different queues and a model load saturates
 * the second one first: reading 145 GB of weights off an NVMe will make a
 * desktop stutter no matter what the nice value is.
 *
 * `ionice -c 3` (idle) first, `-c 2 -n 7` (best-effort, lowest) as the fallback:
 * the idle class needed CAP_SYS_ADMIN on kernels before 2.6.25 and is still
 * refused inside some containers, and a refusal must degrade rather than fail.
 */
export function prioritySteps(pid: number): PriorityStep[] {
  if (!Number.isInteger(pid) || pid <= 0) return [];
  return [
    {
      cmd: "renice",
      args: ["-n", String(NICE_LEVEL), "-p", String(pid)],
      what: `nice ${NICE_LEVEL}`,
    },
    {
      cmd: "ionice",
      args: ["-c", "3", "-p", String(pid)],
      what: "idle I/O",
    },
  ];
}

/** The fallback for a kernel that refuses the idle I/O class. */
export function ioFallback(pid: number): PriorityStep {
  return {
    cmd: "ionice",
    args: ["-c", "2", "-n", String(IO_BEST_EFFORT_LOWEST), "-p", String(pid)],
    what: `best-effort I/O ${IO_BEST_EFFORT_LOWEST}`,
  };
}

/**
 * The line the server log gets.
 *
 * Always says something: a run that was quietly left at normal priority while
 * the switch said otherwise is exactly the kind of silent disagreement this
 * app refuses elsewhere. `done` is what actually took effect.
 */
export function priorityNote(done: string[], failed: string[]): string {
  if (done.length === 0) {
    return failed.length > 0
      ? `[llama.master] could not lower the priority (${
        failed.join("; ")
      }) — llama-server runs at normal priority, so a busy generation may make the desktop sluggish`
      : "[llama.master] priority left at the system default";
  }
  const tail = failed.length > 0 ? ` (${failed.join("; ")})` : "";
  return `[llama.master] running at ${
    done.join(" + ")
  } — the desktop keeps priority${tail}`;
}

/**
 * The nice value of a process, from `/proc/<pid>/stat`.
 *
 * Field 19 (1-based) is `nice`, and the fields before it include `comm` in
 * parentheses — which can itself contain spaces and parentheses, so the split
 * has to start after the LAST `)`. This exists so the tests can assert what
 * actually happened to the process rather than that a command was issued.
 */
export function niceFromProcStat(stat: string): number | null {
  const close = stat.lastIndexOf(")");
  if (close < 0) return null;
  const fields = stat.slice(close + 1).trim().split(/\s+/);
  // After `comm`, field 1 is `state`; `nice` is field 19 overall, so index 16
  // of what is left (state=0, ppid=1, … priority=15, nice=16).
  const nice = Number(fields[16]);
  return Number.isFinite(nice) ? nice : null;
}
