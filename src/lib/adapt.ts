// src/lib/adapt.ts — reacting to memory that moves underneath us.
//
// This app plans against the machine as it is, and the machine does not hold
// still. A game starts and takes 20 GB of VRAM. Another tool loads a model in
// LM Studio. A browser with ninety tabs is closed and 8 GB of RAM comes back.
// Every one of those changes the right answer, in both directions:
//
//   - memory SHRANK → the configuration on file may no longer fit, and starting
//     it would OOM or thrash. Re-plan smaller.
//   - memory GREW   → the configuration on file is now leaving the machine idle.
//     Re-plan bigger; nobody wants to keep running at a third of the card
//     because a game happened to be open when they chose.
//
// The whole difficulty is doing that WITHOUT thrashing. Telemetry arrives every
// second and `MemAvailable` moves constantly — re-tuning on every sample would
// rewrite the user's settings a hundred times a minute, fight their typing, and
// make the command strip flicker. So the trigger is a COARSE bucket: a change has
// to be worth noticing before it counts as news.
//
// A note on what is NOT here: the reserve used to be widened by this machine's
// observed memory "churn". It was removed because the only churn signal available
// is the DEVICE-WIDE usage series, and our own llama-server is inside it — so
// starting a 39 GB model registered as 39 GB of volatility and the app began
// reporting "will not fit" for models that fit. A signal that cannot separate our
// own allocation from everyone else's must not drive a refusal.
//
// Pure, so the policy is testable without a clock or a GPU.

/**
 * How much a pool has to move before it counts as a change.
 *
 * A fraction of capacity rather than an absolute, because 2 GB is a crisis on an
 * 8 GB card and a rounding error on a 180 GB host. 12.5% is chosen so that a
 * typical "a game started" (several GB on a mid-range card) always crosses a
 * boundary while fan-speed-level jitter never does — and so a pool can only
 * produce eight buckets, which bounds how often a re-tune can possibly fire.
 */
export const HEADROOM_FRACTION = 0.125;

/**
 * Which eighth of this pool is free, as an integer.
 *
 * Used as a cache key, not as a number to show anyone: two readings in the same
 * bucket are "no news", and a bucket change is "re-plan". Rounds DOWN, so the
 * boundary is crossed as memory is lost (pessimistic) and only reclaimed once
 * genuinely past the line.
 *
 * Returns 0 for a pool that does not exist or has not been read, which is a
 * stable answer rather than NaN — and one that differs from every real reading,
 * so the first real sample always counts as news.
 */
export function headroomBucket(freeB: number, capacityB: number): number {
  if (!Number.isFinite(freeB) || !Number.isFinite(capacityB)) return 0;
  if (capacityB <= 0) return 0;
  const free = Math.max(0, Math.min(freeB, capacityB));
  return Math.floor((free / capacityB) / HEADROOM_FRACTION);
}

/**
 * The one string that says "the machine's memory is materially as it was".
 *
 * Both pools, because a model is placed across both and either can move: a game
 * takes VRAM, a compile takes RAM, and each changes the answer on its own.
 */
export function headroomKey(pools: {
  vramFreeB: number;
  vramCapacityB: number;
  ramFreeB: number;
  ramCapacityB: number;
}): string {
  return `v${headroomBucket(pools.vramFreeB, pools.vramCapacityB)}:r${
    headroomBucket(pools.ramFreeB, pools.ramCapacityB)
  }`;
}

/** What changed under a running model, if anything worth saying. */
export type Drift =
  | { kind: "none" }
  /** Someone else took memory and what is loaded no longer fits. */
  | { kind: "squeezed"; vramOverB: number; ramOverB: number }
  /** Memory came back, and a restart would now get materially more. */
  | { kind: "roomier"; vramFreeB: number; ramFreeB: number };

/**
 * Has the world moved under a model that is already running?
 *
 * A loaded model cannot be re-placed — its weights are where they are — so this
 * does not re-tune. It decides whether the user should be TOLD, which is the only
 * honest option left: either something else is now competing for memory the
 * running server is relying on, or enough has been freed that restarting would
 * buy a real improvement.
 *
 * `slackB` is what has to come free before "you could do better by restarting" is
 * worth interrupting someone over — a whole extra layer at least, not 200 MB.
 */
export function drift(args: {
  /** Overflow of the CURRENT state, with our own usage already attributed to us
   *  (see `plan.ts:withoutOurUsage`) — so this is other people's pressure. */
  vramOverB: number;
  ramOverB: number;
  /** Free right now, after everything including us. */
  vramFreeB: number;
  ramFreeB: number;
  /** What the running model was given when it started. */
  startedVramB: number;
  startedRamB: number;
  /** Free JUST BEFORE this run was spawned, device-wide. 0 = not recorded,
   *  which disables the roomier signal rather than inventing a baseline. */
  vramFreeAtStartB: number;
  ramFreeAtStartB: number;
  slackFraction?: number;
}): Drift {
  const over = Math.max(0, args.vramOverB) + Math.max(0, args.ramOverB);
  if (over > 0) {
    return {
      kind: "squeezed",
      vramOverB: Math.max(0, args.vramOverB),
      ramOverB: Math.max(0, args.ramOverB),
    };
  }
  // Roomier is measured AGAINST THE MOMENT THIS RUN STARTED: what is free now,
  // minus what would be free had nobody else moved (free-at-start less what
  // this run took). Comparing free-now against the run's own size instead made
  // the note fire permanently on any machine that simply had headroom to begin
  // with — "memory has come free" while nothing had moved at all. And it still
  // has to be a real fraction of what this run is using — "you could have 3%
  // more" is not worth a restart.
  const slack = args.slackFraction ?? 0.5;
  const gained = (freeNow: number, atStart: number, started: number): number =>
    atStart > 0 ? freeNow - Math.max(0, atStart - started) : 0;
  const vGain = gained(
    args.vramFreeB,
    args.vramFreeAtStartB,
    args.startedVramB,
  );
  const rGain = gained(args.ramFreeB, args.ramFreeAtStartB, args.startedRamB);
  const vramWorth = args.startedVramB > 0 &&
    vGain >= args.startedVramB * slack;
  const ramWorth = args.startedRamB > 0 && rGain >= args.startedRamB * slack;
  if (vramWorth || ramWorth) {
    return {
      kind: "roomier",
      vramFreeB: Math.max(0, args.vramFreeB),
      ramFreeB: Math.max(0, args.ramFreeB),
    };
  }
  return { kind: "none" };
}
