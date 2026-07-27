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
  /** Resident set size of the running process: measured, not predicted. */
  rssB: number;
  url: string;
  /** `/health` result, refreshed by the poll while running. */
  healthy: boolean;
  healthDetail: string;
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
    rssB: 0,
    url: "",
    healthy: false,
    healthDetail: "",
    props: null as Record<string, unknown> | null,
    log: [] as string[],
    seq: 0,
    diagnosis: null as Diagnosis | null,
    orphans: [] as { pid: number; argv: string }[],
    freeing: false,
    lastError: "",
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
       *  from, so the memory view can describe reality rather than the form. */
      run?: { model: string; settings: Settings },
    ): Promise<CellEffect | void> {
      if (s.status === "starting" || s.status === "ready") return;
      s.status = "starting";
      s.lastError = "";
      s.exitCode = null;
      s.healthy = false;
      s.healthDetail = "";
      s.props = null;
      s.log = [];
      s.diagnosis = null;
      s.argv = argv;
      s.url = url;
      s.runSettings = run?.settings ?? null;
      s.runModel = run?.model ?? "";
      s.rssB = 0;
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
        s.status = "crashed";
        s.lastError = String(e);
        s.pid = 0;
      }
    },

    async stop(s): Promise<CellEffect | void> {
      if (s.status === "stopped") return;
      s.status = "stopping";
      try {
        const io = await import("./srv.server.ts");
        await io.stop();
        s.status = "stopped";
        s.pid = 0;
        s.healthy = false;
        s.healthDetail = "";
        s.props = null;
        s.runSettings = null;
        s.runModel = "";
        s.rssB = 0;
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
        }
        s.pid = 0;
        s.exitCode = st.exitCode;
        s.healthy = false;
        s.rssB = 0;
        return;
      }

      // Measured, not predicted — and read off the poll's own await rather
      // than inside the sync status snapshot.
      const rssB = await io.rss();
      if (rssB !== s.rssB) s.rssB = rssB; // aiol-ok — see the note above

      const h = await io.health(s.url); // aiol-ok — see the note above
      s.healthy = h.ok;
      s.healthDetail = h.detail;
      if (h.ok) {
        if (s.status !== "ready") { // aiol-ok — see the note above
          s.status = "ready";
          s.props = await io.props(s.url);
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
