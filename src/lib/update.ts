// src/lib/update.ts — is the installed llama.cpp behind upstream?
//
// Two kinds of build need two different answers. A tagged build carries its own
// version (`b10144`), so the comparison is arithmetic on the build number. A
// `master` build's ref says nothing at all — two builds a week apart are both
// "master" — so it is compared by the commit it was built from, which is why
// `Build.sourceSha` exists.
//
// Pure: upstream facts in, a decision out. The cell fetches, this decides.

import type { Build } from "./types.ts";

/** What upstream currently offers, as fetched by the update poll. */
export type Upstream = {
  /** Newest published release tag, e.g. `b10144`. */
  latestTag: string;
  /** Commit `master` points at right now. */
  masterSha: string;
  /** When this was fetched (epoch ms); 0 = never. */
  checkedAt: number;
};

export type UpdateCheck = {
  available: boolean;
  /** What is installed. */
  from: string;
  /** What it would become. */
  to: string;
  /** One line for the button's tooltip — always set, including when there is
   *  nothing to do, so the UI can explain "up to date" too. */
  reason: string;
};

const NONE: UpdateCheck = {
  available: false,
  from: "",
  to: "",
  reason: "Nothing installed yet.",
};

/** `b10144` → 10144; anything else → null. */
export function buildNumber(ref: string): number | null {
  const m = /^b(\d+)$/.exec(ref.trim());
  return m ? Number(m[1]) : null;
}

export function shortSha(sha: string): string {
  return sha.slice(0, 7);
}

/** Decide whether `build` is behind `up`. */
export function updateFor(
  build: Build | null,
  up: Upstream,
): UpdateCheck {
  if (!build) return NONE;
  if (!up.checkedAt) {
    return {
      available: false,
      from: build.ref,
      to: "",
      reason: "Upstream has not been checked yet.",
    };
  }

  if (build.ref === "master") {
    if (!up.masterSha) {
      return {
        available: false,
        from: "master",
        to: "",
        reason: "Could not read upstream master.",
      };
    }
    // A build with no recorded commit predates this check; offering an update
    // is the honest move — we cannot show it is current.
    if (!build.sourceSha) {
      return {
        available: true,
        from: "master (unknown commit)",
        to: `master ${shortSha(up.masterSha)}`,
        reason:
          "This master build did not record the commit it came from, so it cannot be shown as current.",
      };
    }
    if (build.sourceSha === up.masterSha) {
      return {
        available: false,
        from: `master ${shortSha(build.sourceSha)}`,
        to: "",
        reason: "Up to date with upstream master.",
      };
    }
    return {
      available: true,
      from: `master ${shortSha(build.sourceSha)}`,
      to: `master ${shortSha(up.masterSha)}`,
      reason: "Upstream master has moved on since this build.",
    };
  }

  const have = buildNumber(build.ref);
  const latest = buildNumber(up.latestTag);
  if (have === null || latest === null) {
    return {
      available: false,
      from: build.ref,
      to: up.latestTag,
      reason: "Cannot compare these versions.",
    };
  }
  if (latest > have) {
    return {
      available: true,
      from: build.ref,
      to: up.latestTag,
      reason: `${latest - have} llama.cpp builds newer than the installed one.`,
    };
  }
  return {
    available: false,
    from: build.ref,
    to: "",
    reason: "This is the newest published release.",
  };
}

/** The ref an update should install: the newest tag, or master again. */
export function updateTarget(build: Build | null, up: Upstream): string {
  if (!build) return "master";
  return build.ref === "master" ? "master" : up.latestTag || build.ref;
}
