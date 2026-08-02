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
import { ctxOf, fitDecision, withCtx } from "../lib/fitladder.ts";
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
};

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
    props: null as Record<string, unknown> | null,
    log: [] as string[],
    seq: 0,
    diagnosis: null as Diagnosis | null,
    orphans: [] as { pid: number; argv: string }[],
    freeing: false,
    lastError: "",
    fitTries: 0,
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
      },
    ): Promise<CellEffect | void> {
      if (s.status === "starting" || s.status === "ready") return;
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
      s.props = null;
      s.log = [];
      s.diagnosis = null;
      s.argv = argv;
      s.url = url;
      s.runSettings = run?.settings ?? null;
      s.runModel = run?.model ?? "";
      s.startFreeVramB = run?.freeAtStart?.vramB ?? 0;
      s.startFreeRamB = run?.freeAtStart?.ramB ?? 0;
      s.rssB = 0;
      s.rssFileB = 0;
      try {
        const io = await import("./srv.server.ts");
        const { pid } = io.start(argv);
        s.pid = pid;
        s.startedAt = Date.now();
        // The effect owns THIS process, by pid. Replacing the effect on the
        // next start disposes this one, and an unqualified stop() would then
        // kill the process that just replaced it.
        return own.set("srv:process", () => ({
          close: () => {
            void io.stopOwned(pid);
          },
        }));
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
        return own.dispose("srv:process");
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
          });
          if (decision.kind === "retry") {
            const next = withCtx(s.argv, decision.ctx);
            s.fitTries += 1;
            s.fitNote = decision.note;
            s.argv = next;
            if (s.runSettings) s.runSettings.ctxSize = decision.ctx;
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
              const { pid } = io.start(next);
              s.pid = pid;
              s.startedAt = Date.now();
              return own.set("srv:process", () => ({
                close: () => {
                  void io.stopOwned(pid);
                },
              }));
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
