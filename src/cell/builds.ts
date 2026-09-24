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
import { SCHED_SPLIT_CAP } from "../lib/backend.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { Backend, Build, Job } from "../lib/types.ts";
import type { MethodDraftMeta } from "aio";
import { formatRef, parseRef, prStateNote, refForPrs } from "../lib/srcref.ts";

export type Origin = "source" | "release";

export type BuildsState = {
  /** Release tags from GitHub, newest first, with `master` prepended. */
  refs: string[];
  refsLoading: boolean;
  ref: string;
  /** The title of the pull request `ref` names, when it names one. Empty
   *  otherwise, and empty when GitHub could not be read — it is a label, and
   *  no build depends on it. */
  prTitle: string;
  /** Anything worth SAYING about the pull requests in `ref` — chiefly that one
   *  has been merged, which quietly turns its ref into an older master. */
  prNotes: string[];
  backend: Backend;
  /** True once the user has picked a backend by hand. Until then the boot seed
   *  may match `backend` to the hardware. */
  backendChosen: boolean;
  origin: Origin;
  /** `-j` for the compile step. 0 = auto, which is cores − 2 (see below). */
  jobs: number;
  /** `-DGGML_NATIVE`: fastest here, not portable to another CPU. */
  native: boolean;
  /** Compile with `GGML_SCHED_MAX_SPLIT_INPUTS` raised (source route only):
   *  llama.cpp's stock cap of 30 cross-device inputs per graph split aborts
   *  extreme contexts when the experts live in RAM — measured: 256k runs,
   *  512k asserts. The raised cap is what "supports up to 1M" actually needs. */
  bypassSchedCap: boolean;
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
  /** Build id → the flags that build's `llama-server --help` declares.
   *
   *  Kept because the tuner has to be able to switch something ON. Every other
   *  setting is safe to leave at a default a stale build ignores; `-bs` is a
   *  real speed lever, and a build that has never heard of it refuses to start
   *  at all. Guessing from the version string is not available either — a PR
   *  stack has no version that means anything — so the binary is asked, once,
   *  and the answer is kept (`src/lib/caps.ts`) for the life of the process —
   *  a build's flags do not change while it sits on disk. Replaced whenever
   *  a build with that id is rebuilt or removed. */
  caps: Record<string, string[]>;
  broken: Record<string, string>;
  scanning: boolean;
  /** Monotonic scan generation — the last scan to START is the one whose
   *  result may land; a superseded one discards its own (see `scan`). */
  scanEpoch: number;
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

/**
 * Look up what each pull request IS and whether it is still open.
 *
 * Two questions with one page each, and neither may block a build: the title
 * is a label, and the state decides only what the panel SAYS. The merged case
 * is the one worth the round trip — GitHub keeps a merged PR's refs pointing
 * at the merge computed back then, so the day one lands, its ref quietly
 * starts meaning "an older master, plus a change master already has".
 */
async function describePrs(
  s: BuildsState & Partial<MethodDraftMeta>,
  prs: readonly number[],
): Promise<void> {
  if (prs.length === 0) return;
  const io = await import("./builds.server.ts");
  const seen = formatRef(refForPrs(prs.slice()));
  const [titles, states] = await Promise.all([
    Promise.all(prs.map((n) => io.prTitle(n))),
    Promise.all(prs.map((n) => io.prState(n))),
  ]);
  // The ref can have moved on while those were in flight; a description of the
  // previous request beside the current one is worse than none.
  if (s.ref !== seen) return; // aiol-ok: deliberate re-read after await
  s.prTitle = titles.filter(Boolean).join(" · ");
  s.prNotes = prs
    .map((n, i) => prStateNote(n, states[i] ?? "unknown"))
    .filter((x): x is NonNullable<typeof x> => x !== null)
    .map((x) => x.message);
}

/**
 * Point the panel at a source ref, and void what cannot survive the change.
 *
 * A plain function over the draft rather than a method, so `setRef` and
 * `setPr` can share it without one dispatching the other: a nested same-cell
 * call runs against committed state and cannot see the write its caller is
 * halfway through (aiol says so, and it is right).
 */
function applyRef(s: BuildsState, ref: string): void {
  s.ref = ref;
  s.assets = [];
  s.assetName = "";
  s.prTitle = "";
  s.prNotes = [];
  // Only the source route can build a pull request — nobody publishes
  // prebuilt binaries for unmerged code — so the route follows the ref rather
  // than leaving the user on one that is about to refuse them.
  // A fork is source-only too: the release route installs upstream's own
  // releases, and PrismML's Linux CUDA download leaves out the CUDA runtime
  // it links against (measured: libcudart.so.13 / libcublas.so.13 "not
  // found"), which a source build gets from the app's own toolkit.
  const kind = parseRef(ref).kind;
  if (kind === "pr" || kind === "fork") s.origin = "source";
}

/**
 * Probe one build's flags into state, unless they are already known.
 *
 * A plain function over the draft rather than a method calling a method: a
 * cell method's first argument is the runtime's draft, so a sibling call would
 * be a second dispatch with a draft of its own — two commits where `scan`
 * wanted one sequence.
 *
 * A failed probe writes `[]` and nothing else happens: `supportsFlag` reads an
 * empty set as "switch nothing extra on", so the tuner is merely conservative.
 * Recording the failure as an error would put a red line on screen about a
 * question the user never asked.
 */
async function probeInto(
  s: BuildsState,
  id: string,
  force = false,
): Promise<void> {
  if (!id) return;
  if (!force && s.caps[id]) return;
  const bin = s.installed.find((b) => b.id === id)?.serverBin ?? "";
  if (!bin) return;
  const io = await import("./builds.server.ts");
  try {
    s.caps[id] = await io.probeCaps(bin);
    delete s.broken[id];
  } catch (e) {
    s.caps[id] = [];
    // The one failure worth a sentence: the binary cannot load a library,
    // so every Start would fail the same way (`startBlocker` reads this).
    if (e instanceof io.BrokenBuild) s.broken[id] = e.lib;
  }
}

export const builds = cell("builds", {
  // aiol: pre-alpha52 behavior pinned — remove to adopt transactions (s.$commit/s.$live)
  transaction: false,
  // A cmake build of llama.cpp takes minutes on a workstation and can take
  // hours on a laptop, and the release route downloads hundreds of MB over
  // whatever link the user has — neither has an honest ceiling. Declared here
  // rather than as a `perfBudget.methods["builds:start"].timeout` number,
  // because that key is a string in another file that no rename follows;
  // `long:` is checked against this method list at cell() time.
  long: ["start", "update"],
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
      "bypassSchedCap",
      "activeId",
    ],
  },
  state: {
    refs: [] as string[],
    refsLoading: false,
    ref: "master",
    prTitle: "",
    prNotes: [] as string[],
    backend: "cpu" as Backend,
    /** Has the user picked a backend themselves? Until they have, the boot seed
     *  is free to match it to the hardware. */
    backendChosen: false,
    origin: "release" as Origin,
    jobs: 0,
    native: true,
    bypassSchedCap: false,
    assets: [] as Asset[],
    assetName: "",
    assetsLoading: false,
    job: null as Job | null,
    diagnosis: null as Diagnosis | null,
    log: [] as string[],
    installed: [] as Build[],
    activeId: "",
    caps: {} as Record<string, string[]>,
    /** Builds whose binary cannot load a library it links against, with the
     *  library. Not persisted: a rebuild or a restored library clears it on
     *  the next probe. */
    broken: {} as Record<string, string>,
    scanning: false,
    scanEpoch: 0,
    upstream: { latestTag: "", masterSha: "", checkedAt: 0 } as Upstream,
    checkingUpdate: false,
    lastError: "",
  } as BuildsState,
  cancelOn: {
    // Declared against the method below; `cancel` aborts a running `start`.
    start: ["builds:cancel"],
    // …and against `update`, which runs `start`'s BODY inside its own action
    // (`s.$call.start()`). `$call` hands the sibling the CALLER's signal, so
    // without this line Cancel stopped a Build and did nothing at all to an
    // Update — a button that is present, enabled, and inert.
    update: ["builds:cancel"],
  },
  methods: {
    setRef(s, ref: string) {
      applyRef(s, ref);
      // Readiness for the release route is unknowable without the asset list,
      // and "press List assets first" is not an experience. Fetch it.
      const kind = parseRef(ref).kind;
      if (s.origin === "release" && kind !== "pr" && kind !== "fork") {
        builds.loadAssets();
      }
    },

    /**
     * Build the source of a pull request, on top of current master.
     *
     * GitHub publishes every PR already merged into master as its own ref and
     * serves a tarball of it, so this needs no git and no merge logic here —
     * see `src/lib/srcref.ts`. `head` is the escape hatch for a PR that no
     * longer merges: the author's branch alone, older base and all.
     *
     * Async, and it does the title lookup ITSELF rather than dispatching
     * `loadPrTitle` — a nested same-cell call runs as its own transaction
     * against COMMITTED state, so it would read the ref from before this
     * method set it and then discard the title as belonging to another
     * request. `applyRef` is the plain helper both entry points share, which
     * is what aiol asks for and the right shape anyway.
     */
    async setPr(
      s: BuildsState & Partial<MethodDraftMeta>,
      pr: number,
      mode: "merge" | "head" = "merge",
    ) {
      if (!(pr > 0)) return;
      applyRef(s, formatRef({ kind: "pr", pr, mode }));
      await describePrs(s, [pr]);
    },

    /**
     * Build master with SEVERAL pull requests merged into it, in this order.
     *
     * GitHub publishes each PR merged into master and nothing merged into
     * anything else, so this is the one shape that needs a real merge — and
     * therefore git, which `buildFromSource` checks for and names.
     */
    async setPrs(
      s: BuildsState & Partial<MethodDraftMeta>,
      prs: number[],
    ) {
      const list = prs.filter((n) => n > 0);
      applyRef(s, formatRef(refForPrs(list)));
      await describePrs(s, list);
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
      // `backend !== "cpu"` covers stores from before `backendChosen` existed:
      // the only way a pre-seed store holds a non-default backend is that the
      // user picked it, so the missing flag must not let the seed overwrite it.
      if (s.backendChosen || s.backend !== "cpu" || s.backend === backend) {
        return;
      }
      s.backend = backend;
      s.assetName = "";
    },
    setOrigin(s, origin: Origin) {
      s.origin = origin;
      // aiol-ok — `loadAssets` reads `s.ref`, which this method does not
      // write, so there is nothing of ours for it to read stale. `$call` is not
      // an option either way: an async sibling cannot be called from a sync
      // method, because there is nowhere to await it.
      if (origin === "release" && s.assets.length === 0) builds.loadAssets();
    },
    setJobs(s, jobs: number) {
      s.jobs = Math.max(0, Math.min(512, Math.floor(jobs) || 0));
    },
    setNative(s, native: boolean) {
      s.native = native;
    },
    setBypassSchedCap(s, on: boolean) {
      s.bypassSchedCap = on;
    },
    setAsset(s, name: string) {
      s.assetName = name;
    },
    async setActive(s, id: string) {
      s.activeId = id;
      await probeInto(s, id);
    },

    /**
     * Ask a build which flags it accepts, and remember the answer.
     *
     * Safe to call repeatedly: a build already probed is skipped, because the
     * flags a binary on disk accepts do not change. `force` re-asks, which is
     * what an update to the same id needs.
     */
    async probe(s, id?: string, force = false) {
      // Resolved before the await, not inside the call: `id ?? s.activeId` on
      // the same line as `await` reads as a post-await state read, and the one
      // thing this must not do is probe whichever build happened to become
      // active while it was suspended.
      const target = id ?? s.activeId;
      await probeInto(s, target, force);
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
        // A fork build is compared against the FORK, so its ref is resolved
        // before the first await — the active build may change while it is out.
        const active = s.installed.find((b) => b.id === s.activeId);
        const fork = active && parseRef(active.ref).kind === "fork"
          ? active.ref
          : "";
        const io = await import("./builds.server.ts");
        const [latestTag, masterSha, forkSha] = await Promise.all([
          io.latestTag().catch(() => ""),
          io.masterSha().catch(() => ""),
          fork ? io.movingSha(fork).catch(() => "") : Promise.resolve(""),
        ]);
        if (!latestTag && !masterSha && !forkSha) return; // offline — keep what we had
        // aiol-ok: merging the fresh answer with whatever is in state is the
        // point — a field upstream could not tell us keeps its previous value.
        s.upstream = { // aiol-ok
          latestTag: latestTag || s.upstream.latestTag, // aiol-ok
          masterSha: masterSha || s.upstream.masterSha, // aiol-ok
          forkShas: forkSha
            ? { ...s.upstream.forkShas, [fork]: forkSha } // aiol-ok
            : s.upstream.forkShas ?? {}, // aiol-ok
          checkedAt: Date.now(),
        };
      } finally {
        s.checkingUpdate = false;
      }
    },

    /** Re-acquire the active build at the newest upstream version, by whichever
     *  route it came from originally. Same button for both. */
    /**
     * Rebuild/reinstall the active build at whatever upstream offers now.
     *
     * `s.$call.start()`, never `builds.start()`.
     *
     * The ordinary spelling is a SECOND DISPATCH with its own draft, and it
     * reads state as COMMITTED — which is before the four lines above it.
     * Measured, not reasoned: `builds.start()` here set `ref` to the update
     * target and then started a job labelled `Install b1111`, the version
     * already installed. An Update button that reinstalls what you have,
     * reports success, and says it updated. `origin` and `backend` were stale
     * the same way, so an update could also change the backend under you.
     * `s.$commit()` does NOT fix it — verified: the sibling still reads the
     * pre-call snapshot. Only sharing the draft does.
     *
     * `$call` runs the sibling's body against THIS draft, in THIS commit
     * (`dep/aio/docs/state/methods.md`, "One method calling another"). Reached
     * through a local cast rather than by annotating the draft with
     * `MethodDraftCalls`: that type names the siblings, `start`'s return type
     * is being inferred from the same object literal these methods live in, and
     * TypeScript answers the circularity by widening the WHOLE method map to
     * `undefined` — which surfaces as `builds.checkUpdates is possibly
     * undefined` in app.ts, thirty lines and one file away from the cause.
     * `tests/cells.test.ts` pins the label this produces.
     */
    async update(
      s: BuildsState & Partial<MethodDraftMeta>,
    ): Promise<Job["status"]> {
      const active = s.installed.find((b) => b.id === s.activeId) ?? null;
      if (!active || s.job?.status === "running") return "cancelled";
      const target = updateTarget(active, s.upstream);
      s.ref = target;
      s.origin = active.origin;
      s.backend = active.backend;
      s.assetName = "";
      // Hand the status back rather than making the caller read state across
      // the bridge — see `start`.
      const call = (s as unknown as {
        $call: { start(): Promise<Job["status"]> };
      }).$call;
      return await call.start();
    },

    async scan(s) {
      // Newest wins, never first-wins. The old guard (`if (s.scanning) return`)
      // made a concurrent `await scan()` a silent no-op — the caller believed
      // `installed` reflected the disk and it reflected nothing — and a scan
      // that started earlier could resolve later and overwrite fresher state
      // with an older directory listing. Both are the same bug: the LAST scan
      // to start must be the one whose result stands. The epoch does that; a
      // superseded scan discards its own result instead of racing.
      const epoch = ++s.scanEpoch;
      s.scanning = true;
      try {
        const io = await import("./builds.server.ts");
        const list = await io.listBuilds();
        if (epoch !== s.scanEpoch) return; // aiol-ok — superseded, discard
        s.installed = list;
        // Keep the active selection valid without silently switching away from
        // a build the user chose.
        if (!list.some((b) => b.id === s.activeId)) { // aiol-ok
          s.activeId = list[0]?.id ?? "";
        }
        // Forget builds that are no longer on disk, so a removed-and-rebuilt
        // id cannot be answered from the flags its predecessor had.
        for (const id of Object.keys(s.caps)) { // aiol-ok
          if (!list.some((b) => b.id === id)) {
            delete s.caps[id];
            delete s.broken[id]; // aiol-ok
          }
        }
        // Only the ACTIVE build is probed. Probing all of them would spawn a
        // process per installed build on every scan for answers about builds
        // nothing is going to run; `setActive` covers the moment one becomes
        // interesting.
        await probeInto(s, s.activeId); // aiol-ok — deliberate re-read
      } catch (e) {
        s.lastError = String(e);
      } finally {
        if (epoch === s.scanEpoch) s.scanning = false; // aiol-ok — see above
      }
    },

    async remove(s, id: string) {
      try {
        const io = await import("./builds.server.ts");
        await io.removeBuild(id);
        // aiol-ok — deliberately read AFTER the disk removal, not before: the
        // list must only stop claiming a build once the files are actually
        // gone, or a failed delete leaves the UI describing a directory that is
        // still there.
        s.installed = s.installed.filter((b) => b.id !== id);
        // A rebuilt id must never be answered from its predecessor's flags.
        delete s.caps[id];
        delete s.broken[id];
        // aiol-ok — and the selection follows the list it was just filtered
        // out of; both reads are of the line above, not of a stale snapshot.
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

      // The run's parameters, gathered BEFORE the first await: these are the
      // values the user pressed the button on, and nothing that lands while
      // the build module loads may change what this job builds.
      const ref = s.ref;
      const backend = s.backend;
      const signal = s.$signal;
      // Auto = every logical CPU but two. A compile that claims the whole
      // machine makes the desktop unusable for several minutes, and the last
      // two cores buy back almost no wall-clock.
      const jobs = s.jobs || autoJobs();
      const native = s.native;
      const schedCap = s.bypassSchedCap ? SCHED_SPLIT_CAP : 0;
      const assetName = s.assetName || undefined;
      try {
        const io = await import("./builds.server.ts");
        const built = source
          ? await io.buildFromSource(
            { ref, backend, jobs, native, schedCap, signal },
            onProgress,
          )
          : await io.installRelease(
            { ref, backend, assetName, signal },
            onProgress,
          );

        // aio-ok — merged into the list as it is NOW: a scan may have landed
        // during a build that took minutes, and its answer must survive.
        const rest = s.installed.filter((b) => b.id !== built.id);
        s.installed = [built, ...rest];
        s.activeId = built.id;
        // Same id, NEW binary: the old flag list and any "broken" verdict were
        // about the build this one replaced, so ask the new one.
        delete s.caps[built.id];
        delete s.broken[built.id];
        await probeInto(s, built.id, true);
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
            // aio-ok — the log this run streamed, read back to be diagnosed.
            [message, ...s.log.slice(-40)].join("\n"),
            {
              // The run's own route and backend — the setters may have moved
              // since, and a diagnosis of a different build is no diagnosis.
              origin: source ? "source" : "release",
              backend,
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
