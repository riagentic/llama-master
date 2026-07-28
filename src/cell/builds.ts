// src/cell/builds.ts — acquiring and managing llama.cpp builds.
// Browser-safe (see the note in hw.ts).
//
// One long-running method (`start`) drives both routes and reports into a `job`
// object the UI renders as a stepper plus a progress bar. Cancellation is
// declarative: `cancelOn` names the action that aborts it, and the signal is
// handed to fetch and to the child process, so "Cancel" stops a cmake run and a
// 200 MB download alike.

import { cell } from "aio";
import type { Asset } from "../lib/assets.ts";
import { appendLog } from "../lib/buildlog.ts";
import type { Upstream } from "../lib/update.ts";
import { updateFor, updateTarget } from "../lib/update.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { Backend, Build, Job } from "../lib/types.ts";
import type { MethodDraftMeta } from "aio";

export type Origin = "source" | "release";

export type BuildsState = {
  /** Release tags from GitHub, newest first, with `master` prepended. */
  refs: string[];
  refsLoading: boolean;
  ref: string;
  backend: Backend;
  /** True once the user has picked a backend by hand. Until then the boot seed
   *  may match `backend` to the hardware. */
  backendChosen: boolean;
  origin: Origin;
  /** `-j` for the compile step. 0 = auto, which is cores − 2 (see below). */
  jobs: number;
  /** `-DGGML_NATIVE`: fastest here, not portable to another CPU. */
  native: boolean;
  /** Prebuilt assets published for `ref`, once looked up. */
  assets: Asset[];
  assetName: string;
  assetsLoading: boolean;
  job: Job | null;
  /** Why the last job failed and what to do — never a raw message. */
  diagnosis: Diagnosis | null;
  log: string[];
  installed: Build[];
  /** The build every other panel uses. */
  activeId: string;
  scanning: boolean;
  /** What upstream offers, refreshed by a 5-minute poll (src/app.ts). */
  upstream: Upstream;
  checkingUpdate: boolean;
  lastError: string;
};

/** Parallel compile jobs when the user leaves it on auto: leave two cores to
 *  the operating system so the machine stays usable during a build. */
export function autoJobs(logical = navigator.hardwareConcurrency || 4): number {
  return Math.max(1, logical - 2);
}

const EMPTY_JOB = (label: string, steps: string[]): Job => ({
  id: crypto.randomUUID(),
  label,
  progress: null,
  step: 0,
  steps,
  startedAt: Date.now(),
  endedAt: null,
  status: "running",
  error: null,
});

export const builds = cell("builds", {
  // The chosen ref/backend and the active build are worth remembering; the
  // volatile fields are excluded so a restart never resumes a dead job.
  persist: {
    include: [
      "ref",
      "backend",
      "backendChosen",
      "origin",
      "jobs",
      "native",
      "activeId",
    ],
  },
  state: {
    refs: [] as string[],
    refsLoading: false,
    ref: "master",
    backend: "cpu" as Backend,
    /** Has the user picked a backend themselves? Until they have, the boot seed
     *  is free to match it to the hardware. */
    backendChosen: false,
    origin: "release" as Origin,
    jobs: 0,
    native: true,
    assets: [] as Asset[],
    assetName: "",
    assetsLoading: false,
    job: null as Job | null,
    diagnosis: null as Diagnosis | null,
    log: [] as string[],
    installed: [] as Build[],
    activeId: "",
    scanning: false,
    upstream: { latestTag: "", masterSha: "", checkedAt: 0 } as Upstream,
    checkingUpdate: false,
    lastError: "",
  } as BuildsState,
  cancelOn: {
    // Declared against the method below; `cancel` aborts a running `start`.
    start: ["builds:cancel"],
  },
  methods: {
    setRef(s, ref: string) {
      s.ref = ref;
      s.assets = [];
      s.assetName = "";
      // Readiness for the release route is unknowable without the asset list,
      // and "press List assets first" is not an experience. Fetch it.
      if (s.origin === "release") builds.loadAssets();
    },
    setBackend(s, backend: Backend) {
      s.backend = backend;
      s.backendChosen = true;
      s.assetName = "";
    },
    /**
     * Seed the backend from the hardware — but never over a deliberate choice.
     *
     * The stored default has to be *something*, and `cpu` is the only value that
     * is always installable; on a machine with a GPU that made the one-click
     * default the wrong build. This is called at boot with what the hardware
     * wants, and does nothing once the user has picked for themselves — a chosen
     * `cpu` on a CUDA box is a legitimate answer, not a stale default.
     */
    suggestBackend(s, backend: Backend) {
      if (s.backendChosen || s.backend === backend) return;
      s.backend = backend;
      s.assetName = "";
    },
    setOrigin(s, origin: Origin) {
      s.origin = origin;
      if (origin === "release" && s.assets.length === 0) builds.loadAssets();
    },
    setJobs(s, jobs: number) {
      s.jobs = Math.max(0, Math.min(512, Math.floor(jobs) || 0));
    },
    setNative(s, native: boolean) {
      s.native = native;
    },
    setAsset(s, name: string) {
      s.assetName = name;
    },
    setActive(s, id: string) {
      s.activeId = id;
    },
    clearLog(s) {
      s.log = [];
      s.lastError = "";
    },
    /** Cancels a running `start` through `cancelOn`; the state change here is
     *  only what the UI shows while the abort propagates. */
    cancel(s) {
      // Mutate the draft in place rather than spreading it: a value derived
      // from state and assigned back is rejected wholesale by the proxy.
      if (s.job && s.job.status === "running") {
        s.job.status = "cancelled";
        s.job.endedAt = Date.now();
      }
    },

    async loadRefs(s) {
      if (s.refsLoading) return;
      s.refsLoading = true;
      try {
        const io = await import("./builds.server.ts");
        s.refs = ["master", ...(await io.listRefs())];
        s.lastError = "";
      } catch (e) {
        s.lastError = `Could not list llama.cpp releases: ${e}`;
      } finally {
        s.refsLoading = false;
      }
    },

    async loadAssets(s) {
      if (s.assetsLoading) return;
      s.assetsLoading = true;
      try {
        const io = await import("./builds.server.ts");
        const { assets } = await io.listAssets(s.ref); // aiol-ok
        s.assets = assets;
        s.lastError = "";
      } catch (e) {
        s.assets = [];
        s.lastError = `Could not list assets for ${s.ref}: ${e}`;
      } finally {
        s.assetsLoading = false;
      }
    },

    /** Ask GitHub what the newest release and the current master commit are.
     *  Deliberately quiet: a failed check leaves the previous answer in place
     *  rather than flapping the Update button on a dropped connection. */
    async checkUpdates(s) {
      if (s.checkingUpdate) return;
      s.checkingUpdate = true;
      try {
        const io = await import("./builds.server.ts");
        const [latestTag, masterSha] = await Promise.all([
          io.latestTag().catch(() => ""),
          io.masterSha().catch(() => ""),
        ]);
        if (!latestTag && !masterSha) return; // offline — keep what we had
        // aiol-ok: merging the fresh answer with whatever is in state is the
        // point — a field upstream could not tell us keeps its previous value.
        s.upstream = { // aiol-ok
          latestTag: latestTag || s.upstream.latestTag, // aiol-ok
          masterSha: masterSha || s.upstream.masterSha, // aiol-ok
          checkedAt: Date.now(),
        };
      } finally {
        s.checkingUpdate = false;
      }
    },

    /** Re-acquire the active build at the newest upstream version, by whichever
     *  route it came from originally. Same button for both. */
    async update(s): Promise<Job["status"]> {
      const active = s.installed.find((b) => b.id === s.activeId) ?? null;
      if (!active || s.job?.status === "running") return "cancelled";
      const target = updateTarget(active, s.upstream);
      s.ref = target;
      s.origin = active.origin;
      s.backend = active.backend;
      s.assetName = "";
      // Hand the status back rather than making the caller read state across
      // the bridge — see `start`.
      return await builds.start();
    },

    async scan(s) {
      if (s.scanning) return;
      s.scanning = true;
      try {
        const io = await import("./builds.server.ts");
        const list = await io.listBuilds();
        s.installed = list;
        // Keep the active selection valid without silently switching away from
        // a build the user chose.
        if (!list.some((b) => b.id === s.activeId)) { // aiol-ok
          s.activeId = list[0]?.id ?? "";
        }
      } catch (e) {
        s.lastError = String(e);
      } finally {
        s.scanning = false;
      }
    },

    async remove(s, id: string) {
      try {
        const io = await import("./builds.server.ts");
        await io.removeBuild(id);
        s.installed = s.installed.filter((b) => b.id !== id);
        if (s.activeId === id) s.activeId = s.installed[0]?.id ?? "";
      } catch (e) {
        s.lastError = `Could not remove ${id}: ${e}`;
      }
    },

    /** Download a prebuilt release, or compile from source. One method, because
     *  from the user's side it is one button and one progress bar.
     *
     *  Returns the final job status. Callers must use the RETURN VALUE rather
     *  than reading `builds.job` afterwards: on a browser client the state
     *  patch may not have arrived yet, so the read sees the previous value. */
    async start(
      s: BuildsState & Partial<MethodDraftMeta>,
    ): Promise<Job["status"]> {
      if (s.job?.status === "running") return "running";
      const source = s.origin === "source";
      // The job lives in a PLAIN local object and is copied into state on every
      // update; state never round-trips back into the value we build from.
      const job: Job = EMPTY_JOB(
        source
          ? `Build ${s.ref} (${s.backend})`
          : `Install ${s.ref} (${s.backend})`,
        source
          ? ["Fetch source", "Configure", "Compile", "Install"]
          : ["Find release", "Download", "Extract", "Verify"],
      );
      s.job = { ...job };
      s.diagnosis = null;
      s.log = [];
      s.lastError = "";

      const onProgress = (p: {
        step: number;
        steps: string[];
        progress: number | null;
        lines?: string[];
      }) => {
        // Read the live status (a cancel lands in state, not in `job`), but
        // build the next value from the PLAIN local. This used to be forced:
        // spreading a value read back out of state handed the store a
        // proxy-derived object and it rejected the whole action. aio alpha38
        // lifted that, so it is now just the clearer shape — one obvious owner
        // of the job's fields, and no re-copy of state on every progress tick.
        if (s.job?.status !== "running") return;
        job.step = p.step;
        job.steps = p.steps;
        job.progress = p.progress;
        s.job = { ...job };
        if (p.lines?.length) s.log = appendLog(s.log.slice(), p.lines);
      };

      try {
        const io = await import("./builds.server.ts");
        const signal = s.$signal;
        const built = source
          ? await io.buildFromSource(
            {
              ref: s.ref,
              backend: s.backend,
              // Auto = every logical CPU but two. A compile that claims the
              // whole machine makes the desktop unusable for several minutes,
              // and the last two cores buy back almost no wall-clock.
              jobs: s.jobs || autoJobs(),
              native: s.native,
              signal,
            },
            onProgress,
          )
          : await io.installRelease(
            {
              ref: s.ref,
              backend: s.backend,
              assetName: s.assetName || undefined,
              signal,
            },
            onProgress,
          );

        const rest = s.installed.filter((b) => b.id !== built.id);
        s.installed = [built, ...rest];
        s.activeId = built.id;
        job.status = "done";
        job.progress = 1;
        job.endedAt = Date.now();
        s.job = { ...job };
        return "done";
      } catch (e) {
        const aborted = s.$signal?.aborted === true;
        job.status = aborted ? "cancelled" : "failed";
        job.endedAt = Date.now();
        if (aborted) {
          job.error = "Cancelled";
          s.job = { ...job };
          return "cancelled";
        }
        // Every failure gets an explanation and next steps. A message the user
        // cannot act on is a bug, not an error report.
        const io = await import("./builds.server.ts");
        const { diagnoseFailure } = await import("../lib/diagnose.ts");
        const message = e instanceof Error ? e.message : String(e);
        const diagnosis = e instanceof io.BuildFailure
          ? e.diagnosis
          : diagnoseFailure(
            [message, ...s.log.slice(-40)].join("\n"),
            {
              origin: s.origin,
              backend: s.backend,
              platform: navigator.platform.includes("Win")
                ? "windows"
                : "linux",
              arch: "x86_64",
            },
          );
        job.error = diagnosis.reason;
        s.job = { ...job };
        s.diagnosis = diagnosis;
        s.lastError = diagnosis.reason;
        return "failed";
      }
    },
  },
  selectors: {
    active: (s) => s.installed.find((b) => b.id === s.activeId) ?? null,
    /** Whether the active build is behind upstream, and by what. Named apart
     *  from the `update()` method: a cell's methods and selectors share one
     *  namespace, and the callable would win. */
    updateInfo: (s) =>
      updateFor(
        s.installed.find((b) => b.id === s.activeId) ?? null,
        s.upstream,
      ),
    busy: (s) => s.job?.status === "running",
    totalSizeB: (s) => s.installed.reduce((a, b) => a + b.sizeB, 0),
  },
});
