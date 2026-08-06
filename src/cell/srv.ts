// src/cell/srv.ts — the llama-server lifecycle as observable state.
// Browser-safe (see the note in hw.ts).
//
// The child process itself lives in srv.server.ts; this cell is its shadow.
// `poll` is on a 1 s schedule (src/app.ts) and is the ONLY writer of liveness
// state, so "running" always means "the OS still has that pid", never "we
// dispatched start and assumed it worked".

import { cell, own } from "aio";
import type { CellEffect } from "aio";
import { appendLog } from "../lib/buildlog.ts";
import { diagnoseServerExit } from "../lib/serverlog.ts";
import {
  ctxOf,
  fitDecision,
  nCpuMoeOf,
  withCtx,
  withNCpuMoe,
} from "../lib/fitladder.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { Settings } from "../lib/types.ts";

export type ServerStatus =
  | "stopped"
  | "starting"
  | "ready"
  | "stopping"
  | "crashed";

export type SrvState = {
  status: ServerStatus;
  pid: number;
  startedAt: number;
  exitCode: number | null;
  /** The exact argv that was spawned — what the user saw in the preview. */
  argv: string[];
  /** The settings and model this process was STARTED with. Kept because the
   *  memory view must describe what is running, not what the user has since
   *  typed into the panel — those are different things the moment they edit. */
  runSettings: Settings | null;
  runModel: string;
  /** Device-wide free memory the moment this run was spawned — the baseline
   *  the "roomier" drift note measures against. Without it, "memory has come
   *  free since this model started" fired on any machine that simply had
   *  headroom to begin with. */
  startFreeVramB: number;
  startFreeRamB: number;
  /** Resident set size of the running process: measured, not predicted. */
  rssB: number;
  /** The FILE-BACKED share of that RSS — the memory-mapped model. The kernel
   *  books these pages as reclaimable cache, so every "used RAM" meter calls
   *  them free (measured: 138 of 139 GB invisible). The UI draws this share
   *  as its own colour so the model does not read as missing. */
  rssFileB: number;
  url: string;
  /** `/health` result, refreshed by the poll while running. */
  healthy: boolean;
  healthDetail: string;
  /** A probe is in flight. Polls overlap — each is an async dispatch on a 1 s
   *  schedule, and the probe takes seconds on a big model — so without this
   *  three concurrent polls each ran their own probe (observed live: three
   *  27-token generations in the server log before the first chat). */
  probing: boolean;
  /** This run has produced tokens — /health only proves the weights loaded,
   *  and a plan that is too tight can pass /health and still OOM at the first
   *  real batch, because CUDA allocates its compute scratch lazily. `ready` is
   *  only entered through a one-shot generation probe, and THIS is the flag
   *  `cfg.rememberFit` waits for: recording a context at /health once wrote
   *  down 17,408 as a working size for a model that could not answer "Hi". */
  proven: boolean;
  /** Was this run asked to yield to the desktop? Kept on the RUN rather than
   *  read from the toggle: a fit-ladder rung is the same run one number
   *  smaller, and it must be started the way the run was — not the way the
   *  switch happens to be set two minutes later. */
  runLowPriority: boolean;
  /** `/props` — what the server reports it actually loaded. */
  props: Record<string, unknown> | null;
  log: string[];
  /** Line counter from the server module; lets the poll skip a no-op update. */
  seq: number;
  /** Why it stopped, in words — never just an exit code. */
  diagnosis: Diagnosis | null;
  /** llama-servers from this app's builds directory that nothing here owns:
   *  survivors of a crash or a killed app. They hold their VRAM until stopped,
   *  which is the usual reason a fresh Start dies with CUDA out-of-memory. */
  orphans: { pid: number; argv: string }[];
  freeing: boolean;
  lastError: string;
  /** Automatic retries already spent on this Start.
   *
   *  Some models cannot be sized from their header — a sparse-attention MoE
   *  asked for a 68 GiB compute buffer where the planner predicted 730 MB — so
   *  when a run dies for want of memory the app halves the context and tries
   *  again rather than reporting a number it had no way to know
   *  (`src/lib/fitladder.ts`). Reset by a Start the user asked for, so a manual
   *  attempt never inherits an exhausted ladder. */
  fitTries: number;
  /** Whether this run may be retried at all: off when the user is driving the
   *  settings by hand, because shrinking a context they chose is not a fix. */
  autoFit: boolean;
  /** What the last automatic step-down did, in words, for the panel. */
  fitNote: string;
  /** Probes that timed out on this run. A timeout is a slow machine, not a
   *  dead process — counting it as death left a working server "starting"
   *  forever (see the probe branch in `poll`). */
  probeSlow: number;
  /** The running model's shape, kept for the ladder: a weights overflow is
   *  answered by moving experts to the host, and that needs to know how many
   *  layers there are and what one layer's experts weigh. */
  runShape: { nLayer: number; expertPerLayerB: number } | null;
  /** VRAM each card had free when this run was spawned — the denominator that
   *  turns "asked for 34.7 GB" into "short by 12.7 GB". */
  runCardFreeB: number[];
};

/**
 * How many two-minute probes may time out before the app calls it ready anyway.
 *
 * Two, so a genuinely cold start on a slow disk gets four minutes to produce
 * two tokens, and a machine that will never produce them stops being described
 * as "starting". `proven` stays false either way, so nothing is written down as
 * a measured fit off a probe that did not finish (`src/lib/fitladder.ts`).
 */
const PROBE_PATIENCE = 2;

/** The `own` slot a server process lives in. One per pid, never shared — see
 *  the note at `start`. */
function slotId(pid: number): string {
  return `srv:process:${pid}`;
}

/**
 * Take the slot for `pid`, and release the one the process it replaces held.
 *
 * Two effects rather than one `own.set` over a shared key, because `set` on a
 * held key disposes what is there as a side effect — which is how the first
 * start's teardown came to SIGTERM the second start's process. Here the release
 * names a process that has already exited, so it can only ever be a no-op.
 */
function ownProcess(
  prevPid: number,
  pid: number,
  close: () => void,
): CellEffect {
  const take = own.set(slotId(pid), () => ({ close }));
  return prevPid && prevPid !== pid
    ? [own.dispose(slotId(prevPid)), take]
    : take;
}

export const srv = cell("srv", {
  // A process cannot survive a restart of this app, so persisting its state
  // would only ever restore a lie.
  persist: "none",
  state: {
    status: "stopped" as ServerStatus,
    pid: 0,
    startedAt: 0,
    exitCode: null as number | null,
    argv: [] as string[],
    runSettings: null as Settings | null,
    runModel: "",
    startFreeVramB: 0,
    startFreeRamB: 0,
    rssB: 0,
    rssFileB: 0,
    url: "",
    healthy: false,
    healthDetail: "",
    probing: false,
    proven: false,
    runLowPriority: true,
    props: null as Record<string, unknown> | null,
    log: [] as string[],
    seq: 0,
    diagnosis: null as Diagnosis | null,
    orphans: [] as { pid: number; argv: string }[],
    freeing: false,
    lastError: "",
    fitTries: 0,
    probeSlow: 0,
    runShape: null as { nLayer: number; expertPerLayerB: number } | null,
    runCardFreeB: [] as number[],
    autoFit: false,
    fitNote: "",
  } as SrvState,
  methods: {
    /** Spawn llama-server with the exact command shown in the UI.
     *
     *  Returns an `own` effect so the runtime disposes the process on app
     *  shutdown or cell disable — the pid in state is for display, the effect
     *  is what guarantees no orphan is left behind. */
    async start(
      s,
      argv: string[],
      url: string,
      /** What this run is: the model path and the settings it was composed
       *  from, so the memory view can describe reality rather than the form —
       *  and how much was free at the moment of the spawn, so drift can tell
       *  "memory came back" apart from "there was always room". */
      run?: {
        model: string;
        settings: Settings;
        freeAtStart?: { vramB: number; ramB: number };
        /** May this run be retried at a smaller context if it dies for want of
         *  memory? On when the tuner chose the settings, off when the user did. */
        autoFit?: boolean;
        /** Set only by the retry itself, so the ladder is not reset by its own
         *  next rung. */
        retry?: boolean;
        /** Run it at the lowest OS priority, so the desktop keeps its own.
         *  Applied to the process after the spawn — the argv on screen stays
         *  the argv that ran (`src/lib/priority.ts`). */
        lowPriority?: boolean;
        /** What the ladder needs to answer a WEIGHTS overflow, which no smaller
         *  context can fix: the layer count and one layer's routed experts,
         *  both exact from the header. */
        shape?: { nLayer: number; expertPerLayerB: number };
        /** VRAM each card had to give at the moment of the spawn, so a failed
         *  allocation can be turned into a shortfall instead of being taken at
         *  face value. */
        cardFreeB?: number[];
      },
    ): Promise<CellEffect | void> {
      if (s.status === "starting" || s.status === "ready") return;
      // The pid this start REPLACES, captured before anything overwrites it —
      // its `own` slot is released below, by name (see the note at `own.set`).
      const prevPid = s.pid;
      s.status = "starting";
      if (!run?.retry) {
        s.fitTries = 0;
        s.fitNote = "";
        s.autoFit = run?.autoFit ?? false;
      }
      s.lastError = "";
      s.exitCode = null;
      s.healthy = false;
      s.healthDetail = "";
      s.proven = false;
      // Per RUN, not per ladder rung: a rung is the same machine one number
      // smaller, and it deserves its own patience.
      s.probeSlow = 0;
      s.props = null;
      s.log = [];
      s.diagnosis = null;
      s.argv = argv;
      s.url = url;
      s.runSettings = run?.settings ?? null;
      s.runModel = run?.model ?? "";
      s.runLowPriority = run?.lowPriority !== false;
      if (!run?.retry) {
        s.runShape = run?.shape ?? null;
        s.runCardFreeB = run?.cardFreeB?.slice() ?? [];
      }
      s.startFreeVramB = run?.freeAtStart?.vramB ?? 0;
      s.startFreeRamB = run?.freeAtStart?.ramB ?? 0;
      s.rssB = 0;
      s.rssFileB = 0;
      try {
        const io = await import("./srv.server.ts");
        const { pid } = io.start(argv);
        s.pid = pid;
        s.startedAt = Date.now();
        // Not awaited: two tiny subprocesses must not stand between the spawn
        // and the UI learning there is a pid. The note lands in the log through
        // `push`, which the next poll copies out like any other line.
        if (s.runLowPriority) void io.lowerPriority(pid);
        // ONE SLOT PER PROCESS, named by pid — aio's own advice for exactly
        // this bug (`docs/state/methods.md`: "if the disposer tears down
        // something the new resource needs … give each resource its own id").
        //
        // Under one shared id, `own.set` DISPOSES the resource already there as
        // a side effect of registering the new one, so the first start's
        // teardown ran against the second start's process and SIGTERMed it a
        // moment after it came up — every start after a crash died with code
        // 143. `stopOwned(pid)` was the guard that made that harmless; naming
        // the slot after the process means there is nothing to guard against,
        // and the framework stops warning that a live resource was displaced.
        //
        // The dead process's slot is released in the same breath, so a session
        // of starts and stops does not accumulate one no-op disposer per run.
        return ownProcess(prevPid, pid, () => {
          void io.stopOwned(pid);
        });
      } catch (e) {
        // A refused duplicate start is not a crash — an impatient double-click
        // used to leave the cell "crashed" with pid 0 while the first server ran
        // happily, so the panel showed a dead server that was in fact serving.
        const msg = String(e);
        // Stop won the race: the server module killed what it had just spawned
        // (see `stopGeneration`), so this is a completed cancellation.
        if (msg.includes("cancelled by stop")) {
          s.status = "stopped";
          s.pid = 0;
          return;
        }
        if (msg.includes("already running")) {
          s.status = "starting"; // the poll below will confirm it is up
          return;
        }
        s.status = "crashed";
        s.lastError = msg;
        s.pid = 0;
      }
    },

    async stop(s): Promise<CellEffect | void> {
      // Whose slot this releases, read before the fields are cleared.
      const stopping = s.pid;
      // No early return on `status === "stopped"`, however tempting: a Start
      // that has been dispatched but whose body has not run yet leaves the
      // status at "stopped", so short-circuiting here meant Stop did nothing
      // and the server the user cancelled came up a moment later and kept its
      // memory. Reaching `io.stop()` is what registers the cancellation
      // (`stopGeneration`), and it is a cheap no-op when nothing is running.
      s.status = "stopping";
      try {
        const io = await import("./srv.server.ts");
        await io.stop();
        s.status = "stopped";
        s.pid = 0;
        s.healthy = false;
        s.healthDetail = "";
        s.proven = false;
        s.props = null;
        s.runSettings = null;
        s.runModel = "";
        s.startFreeVramB = 0;
        s.startFreeRamB = 0;
        s.rssB = 0;
        s.rssFileB = 0;
        return own.dispose(slotId(stopping));
      } catch (e) {
        s.lastError = String(e);
      }
    },

    /** Single writer of liveness. Cheap when nothing changed. */
    async poll(s) {
      if (s.status === "stopped") return;
      // aiol-ok: this method IS the observer of state that changes underneath
      // it — reading the freshest status after the await is its whole job.
      const io = await import("./srv.server.ts");
      const st = io.status();

      if (st.seq !== s.seq) { // aiol-ok — see the note above
        // Replace rather than append: the server module already keeps the tail
        // bounded, and re-sending it keeps the two views identical.
        s.log = appendLog([], st.lines.slice());
        s.seq = st.seq;
      }

      if (!st.running) {
        // Read once: TypeScript narrows `s.status` through the guard above, but
        // the live proxy can change under an await, so the branch reads a copy.
        const was: ServerStatus = s.status; // aiol-ok — see the note above
        if (was === "stopping") {
          s.status = "stopped";
        } else {
          // Died on its own. The exit code is nearly always a bare 1; the
          // reason is in the output it already printed, so read that.
          s.status = "crashed";
          const d = diagnoseServerExit(st.exitCode, st.lines);
          s.diagnosis = d;
          s.lastError = d.reason;

          // ...unless a smaller context would fix it, in which case try that
          // before reporting anything. For models whose buffers cannot be
          // derived from the header this is not a fallback, it is the only way
          // the right number is ever found — the planner's estimate is the
          // opening bid and the allocator has the final say. Rewriting `-c` in
          // the argv we already ran keeps "what you see is what runs" true:
          // this IS the command, one number smaller.
          const decision = fitDecision({
            lines: st.lines,
            ctx: ctxOf(s.argv),
            tries: s.fitTries,
            auto: s.autoFit,
            nCpuMoe: nCpuMoeOf(s.argv),
            nLayer: s.runShape?.nLayer ?? 0,
            expertPerLayerB: s.runShape?.expertPerLayerB ?? 0,
            deviceFreeB: s.runCardFreeB.slice(),
          });
          if (decision.kind !== "none") {
            // Two rungs, one restart. Which number changes is the ladder's
            // decision (`fitladder.ts`); rewriting the argv that ACTUALLY ran —
            // rather than re-composing one from settings that may have moved
            // since — is what keeps "what you see is what runs" true across a
            // retry.
            const next = decision.kind === "retry"
              ? withCtx(s.argv, decision.ctx)
              : withNCpuMoe(s.argv, decision.nCpuMoe);
            s.fitTries += 1;
            s.fitNote = decision.note;
            s.argv = next;
            if (s.runSettings) {
              if (decision.kind === "retry") {
                s.runSettings.ctxSize = decision.ctx;
              } else {
                s.runSettings.nCpuMoe = decision.nCpuMoe;
                // The split went with the placement that just failed, and
                // `withNCpuMoe` dropped it from the argv — so the settings it
                // came from must stop claiming it too, or the command strip and
                // the process disagree.
                s.runSettings.tensorSplit = "";
              }
            }
            s.log = appendLog(s.log.slice(), [
              `[llama.master] ${decision.note}`,
            ]);
            s.status = "starting";
            s.diagnosis = null;
            s.lastError = "";
            s.exitCode = null;
            s.healthy = false;
            s.proven = false;
            s.rssB = 0;
            s.rssFileB = 0;
            try {
              const dead = s.pid;
              const { pid } = io.start(next);
              s.pid = pid;
              s.startedAt = Date.now();
              // Every rung is a fresh process, so every rung is reniced.
              if (s.runLowPriority) void io.lowerPriority(pid); // aiol-ok

              // Same as `start`: the rung that just died hands its slot over.
              return ownProcess(dead, pid, () => {
                void io.stopOwned(pid);
              });
            } catch (e) {
              s.status = "crashed";
              s.lastError = String(e);
            }
          }
        }
        s.pid = 0;
        s.exitCode = st.exitCode;
        s.healthy = false;
        s.proven = false;
        s.rssB = 0;
        s.rssFileB = 0;
        return;
      }

      // Measured, not predicted — and read off the poll's own await rather
      // than inside the sync status snapshot.
      const r = await io.rss();
      if (r.rssB !== s.rssB) s.rssB = r.rssB; // aiol-ok — see the note above
      if (r.fileB !== s.rssFileB) s.rssFileB = r.fileB; // aiol-ok — same

      // Single writer of liveness means the pid too: anything that leaves the
      // cell's copy stale (a refused duplicate start, a restart) is corrected
      // here rather than showing pid 0 next to a running server.
      if (st.pid !== s.pid) s.pid = st.pid; // aiol-ok — see the note above

      const h = await io.health(s.url); // aiol-ok — see the note above
      s.healthy = h.ok;
      s.healthDetail = h.detail;
      if (h.ok) {
        if (s.status !== "ready") { // aiol-ok — see the note above
          // One real forward pass before claiming ready: /health only proves
          // the weights loaded, and a too-tight plan OOMs at the first real
          // batch because CUDA allocates compute scratch lazily. If the probe
          // kills the process, the next poll's crash branch diagnoses it and
          // the fit ladder steps the context down — which is the ladder doing
          // its job on generation, not just on loading.
          //
          // One probe, not one per poll: polls overlap (the probe takes
          // seconds, the schedule fires every one), and the check-and-set is
          // adjacent synchronous statements so a second poll cannot slip
          // between them.
          if (s.probing) return; // aiol-ok — see the note above
          s.probing = true;
          s.healthDetail = "proving a first reply";
          try {
            const p = await io.probe(s.url); // aiol-ok — see the note above
            if (p.kind === "dead") {
              // The connection dropped mid-probe: the process is most likely
              // dying of the OOM the probe provoked. Say nothing yet — the
              // next poll sees the exit and owns the diagnosis.
              s.healthDetail = p.detail;
              return;
            }
            if (p.kind === "slow") {
              // Alive, just not finished. A timeout used to land in `dead`, so
              // the poll waited for an exit that was never coming and re-probed
              // every second while the panel said "proving a first reply" —
              // indefinitely, on a server that was working. The first reply
              // after a cold start runs against a page cache that is still
              // filling (measured: 23x slower prompt processing), so being slow
              // once is normal and being slow twice is the machine, not a fault.
              s.probeSlow += 1;
              s.healthDetail = p.detail;
              if (s.probeSlow < PROBE_PATIENCE) return;
              // Out of patience: READY, and honest about what was not proved.
              // Refusing to serve a working model because our own check was
              // impatient would be the worse of the two errors — and `proven`
              // stays false, so the fit ladder never records this as a fact.
              s.proven = false;
              s.status = "ready";
              s.healthDetail =
                `ready — the first reply is taking over ${PROBE_PATIENCE} minutes, so the context has not been proved at this size`;
              s.props = await io.props(s.url);
              return;
            }
            // `refused` (an old build without /completion) still becomes
            // ready — but unproven, so the fit is never recorded off it.
            s.proven = p.kind === "ok";
            s.status = "ready";
            s.healthDetail = "ready";
            s.props = await io.props(s.url);
          } finally {
            s.probing = false;
          }
        }
      } else if (s.status === "ready") { // aiol-ok — see the note above
        // Was ready, now is not: the model is reloading or the server is wedged.
        s.status = "starting";
      }
    },

    clearLog(s) {
      s.log = [];
      s.diagnosis = null;
    },

    /** Look for llama-servers this app did not start. Cheap enough for the
     *  1 s poll, and the answer is what makes a failed Start explainable. */
    async scanOrphans(s) {
      const io = await import("./srv.server.ts");
      const found = await io.findOrphans();
      // aiol-ok: the freshest list is exactly what this must compare against —
      // writing an equal value every second would churn the UI for nothing.
      const same = found.length === s.orphans.length && // aiol-ok
        found.every((o, i) => o.pid === s.orphans[i]?.pid); // aiol-ok
      if (!same) s.orphans = found;
    },

    /** Stop every stray llama-server and release the memory they hold.
     *
     *  The user-facing promise is "free the VRAM": a model stays resident until
     *  its process exits, so unloading IS stopping the process. */
    async freeMemory(s) {
      if (s.freeing) return;
      s.freeing = true;
      try {
        const io = await import("./srv.server.ts");
        const orphans = await io.findOrphans();
        for (const o of orphans) {
          try {
            await io.stopOrphan(o.pid);
          } catch (e) {
            s.lastError = String(e);
          }
        }
        s.orphans = await io.findOrphans();
        if (s.orphans.length === 0) s.lastError = "";
      } finally {
        s.freeing = false;
      }
    },
  },
  selectors: {
    running: (s) => s.status === "starting" || s.status === "ready",
    uptimeMs: (s) => (s.startedAt ? Date.now() - s.startedAt : 0),
    /** The model path llama-server reports — ground truth, not our request. */
    loadedModel: (s) => String(s.props?.model_path ?? ""),
  },
});
