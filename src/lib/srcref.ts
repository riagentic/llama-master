// src/lib/srcref.ts — which llama.cpp source to build, including one that does
// not exist yet.
//
// The app used to know two kinds of source: `master`, and a release tag like
// `b7421`. That is a real gap for the machine this app is for, because the
// interesting models arrive as PULL REQUESTS months before they merge — the
// user who prompted this had hand-downloaded a PR tarball into the cache
// directory and built it there, which works exactly once and then rots.
//
// The tempting answer is "clone and merge", and it is the wrong one here:
// source is fetched as a tarball precisely so that git is not a dependency
// (`builds.server.ts`), and adding one for this would trade a working build on
// a bare machine for a convenience.
//
// GitHub already does the merge for us. Every pull request publishes two refs:
//
//   refs/pull/<N>/head    the author's branch, as they left it — which may be
//                         based on a master from six weeks ago
//   refs/pull/<N>/merge   that branch MERGED INTO CURRENT MASTER, computed and
//                         kept up to date by GitHub
//
// and codeload serves a tarball of either. So "latest llama.cpp with this one
// PR" is a plain download, no git, no merge logic, no conflict resolution here.
//
// The merge ref's failure mode is the useful part: when a PR stops merging
// cleanly, GitHub simply does not publish it and the tarball 404s. Verified
// live against this repository on 2026-09-03 — PR 27754 answered 200 on both
// refs, PR 27742 answered 200 on `head` and 404 on `merge` (its branch has
// drifted behind master). That 404 is a FACT worth reporting rather than an
// error to swallow: it means "this PR and today's master disagree, and a human
// has to decide", which is exactly what the user needs to know before waiting
// out a twenty-minute compile.
//
// Pure: a ref string in, a URL and a directory name out.

/** What a source ref names. */
export type SrcRef =
  /** The tip of the default branch. Moves. */
  | { kind: "master" }
  /** A published build tag (`b7421`). Immutable. */
  | { kind: "tag"; tag: string }
  /** A pull request. `merge` is the PR on top of current master (moves as
   *  master moves); `head` is the author's branch alone. */
  | { kind: "pr"; pr: number; mode: "merge" | "head" }
  /**
   * Master with SEVERAL pull requests merged into it, in the order given.
   *
   * The one case GitHub cannot do for us: it publishes each PR merged into
   * master, never two of them merged into each other. So this is the only ref
   * that needs a real merge, and therefore the only one that needs git — which
   * is why it is a separate kind rather than a list on the one above. A single
   * PR keeps taking the tarball path and keeps needing nothing installed.
   */
  | { kind: "stack"; prs: number[] }
  /**
   * Another repository's llama.cpp — a FORK, built from source.
   *
   * Some model families ship before their kernels reach upstream, and the
   * only runtime that can load them is the vendor's fork: PrismML's ternary
   * `PQ2_0`/`PTQ1_0` types live in `PrismML-Eng/llama.cpp` (default branch
   * `prism`), and the one upstream PR for them is CPU-only and was asked to
   * wait for PrismML. A fork is still a codeload tarball, so it needs no git.
   * `ref` null = the fork's default branch (codeload's `HEAD`), which is what
   * a user pasting the repository's URL means.
   */
  | { kind: "fork"; repo: string; ref: string | null };

/** How a ref is spelled in `cfg`/`builds` state and in a build id. */
export function formatRef(r: SrcRef): string {
  switch (r.kind) {
    case "master":
      return "master";
    case "tag":
      return r.tag;
    case "pr":
      return r.mode === "head" ? `pr/${r.pr}@head` : `pr/${r.pr}`;
    case "stack":
      return `master+${r.prs.map((n) => `pr/${n}`).join("+")}`;
    case "fork":
      return `fork:${r.repo}${r.ref ? `@${r.ref}` : ""}`;
  }
}

/** Upstream. Every ref kind but `fork` names something in it. */
const REPO = "ggml-org/llama.cpp";

/** `owner/name` as GitHub spells it — the only shape a repo may take here,
 *  because it lands in a URL and (via `refDirName`) in a path. */
const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9-]{0,38}\/[A-Za-z0-9._-]{1,100}$/;
/** A branch or tag name, conservatively: git allows more, a URL and a path
 *  want less, and every real llama.cpp fork ref fits this. */
const GITREF_RE = /^[A-Za-z0-9][A-Za-z0-9._\/-]{0,200}$/;

function forkRef(repo: string, ref: string | null): SrcRef | null {
  const r = repo.replace(/\.git$/, "");
  if (!REPO_RE.test(r) || r.includes("..")) return null;
  if (ref !== null && (!GITREF_RE.test(ref) || ref.includes(".."))) return null;
  // The upstream repository is not a fork of itself: its refs already have
  // spellings (`master`, `b7421`), and a second one is a second cache entry.
  if (r.toLowerCase() === REPO.toLowerCase()) return null;
  return { kind: "fork", repo: r, ref };
}

/**
 * A fork out of whatever the user pasted: the repository's URL, a
 * `/tree/<branch>` URL, `owner/name`, or `owner/name@ref`. Null when it is not
 * one — and a pull request URL is deliberately NOT one, since it has its own
 * reader and meaning.
 */
export function parseForkInput(text: string): SrcRef | null {
  const s = text.trim();
  const url =
    /^(?:https?:\/\/)?(?:www\.)?github\.com\/([^/\s]+\/[^/\s#?]+)(?:\/tree\/([^\s#?]+))?\/?$/i
      .exec(s);
  if (url) return forkRef(url[1]!, url[2] ? decodeURIComponent(url[2]) : null);
  const bare = /^([^/@\s]+\/[^/@\s]+)(?:@(\S+))?$/.exec(s);
  return bare ? forkRef(bare[1]!, bare[2] ?? null) : null;
}

/** Read a stored ref back. Anything unrecognised is treated as a tag, which is
 *  what every ref was before pull requests existed here. */
export function parseRef(ref: string): SrcRef {
  const s = ref.trim();
  if (s === "master" || s === "") return { kind: "master" };
  if (s.startsWith("fork:")) {
    const f = parseForkInput(s.slice("fork:".length));
    // An unreadable fork spelling is NOT a tag — that would send a hostile
    // stored string to codeload as `refs/tags/fork:…`. Master is the one ref
    // that is always buildable.
    return f ?? { kind: "master" };
  }
  if (s.startsWith("master+")) {
    const prs = s
      .slice("master+".length)
      .split("+")
      .map((part) => /^pr\/(\d+)$/.exec(part.trim())?.[1])
      .filter((x): x is string => Boolean(x))
      .map(Number);
    // A stack of one is a stack of none is master: both are better served by
    // the refs that already exist, and having two spellings of one thing is
    // how a cache ends up with two copies of the same tree.
    if (prs.length > 1) return { kind: "stack", prs };
    if (prs.length === 1) {
      return { kind: "pr", pr: prs[0]!, mode: "merge" };
    }
    return { kind: "master" };
  }
  const m = /^pr\/(\d+)(@head)?$/.exec(s);
  if (m) {
    return { kind: "pr", pr: Number(m[1]), mode: m[2] ? "head" : "merge" };
  }
  return { kind: "tag", tag: s };
}

/** A pull request URL: the repository, then the number. */
const PR_URL = /github\.com\/([^/\s]+\/[^/\s]+)\/pull\/(\d+)/i;

/** Is this `owner/name` upstream llama.cpp? Its old home redirects to the new
 *  one with the same numbers, so a link from before the move is still ours. */
function isUpstream(repo: string): boolean {
  const r = repo.toLowerCase();
  return r === REPO.toLowerCase() || r === "ggerganov/llama.cpp";
}

/**
 * A pull request number out of whatever the user pasted.
 *
 * People arrive here from a browser, so the URL they copied is the commonest
 * input by far, and "#27754" is the second. Returning null rather than
 * guessing: a box that silently builds the wrong PR is worse than one that
 * says it did not understand.
 */
export function parsePrInput(text: string): number | null {
  const s = text.trim();
  if (!s) return null;
  const url = PR_URL.exec(s);
  // Upstream's pull requests only. A PR number means nothing without its
  // repository: `PrismML-Eng/llama.cpp/pull/12` read as a number built
  // UPSTREAM's #12 — somebody else's change, under a name that looked right.
  if (url) return isUpstream(url[1]!) ? Number(url[2]) : null;
  const bare = /^#?(\d+)$/.exec(s);
  if (!bare) return null;
  const n = Number(bare[1]);
  // A PR number is a small positive integer. Anything else is a paste error,
  // and turning it into a URL would produce a 404 the user cannot explain.
  return n > 0 && n < 10_000_000 ? n : null;
}

/**
 * Why the PR box refused something, or `null` when it did not.
 *
 * `parsePrInput` answers null for a pull request against another repository,
 * and a box that ignores a paste without saying so is the one thing worse than
 * one that refuses it: this names the repository, and what would build it.
 */
export function prInputProblem(text: string): string | null {
  for (const part of text.split(/[\s,;]+/)) {
    const url = PR_URL.exec(part);
    if (url && !isUpstream(url[1]!)) {
      return `#${url[2]} is a pull request against ${
        url[1]
      }, not ${REPO} — only upstream's pull requests can be merged into master here. To build that repository, paste its URL (github.com/${
        url[1]
      }) instead.`;
    }
  }
  return null;
}

/**
 * Every pull request number in whatever the user pasted, in order.
 *
 * One box for one PR and for five, because "27773, 28136, 27269" is how people
 * write a list and splitting it across three fields would be three chances to
 * get the ORDER wrong — and order is load-bearing here: merging A then B is
 * not merging B then A when they touch the same lines.
 *
 * Duplicates are dropped rather than merged twice, which git would refuse
 * anyway with a message about nothing to do.
 */
export function parsePrList(text: string): number[] {
  const out: number[] = [];
  for (const part of text.split(/[\s,;]+/)) {
    const n = parsePrInput(part);
    if (n !== null && !out.includes(n)) out.push(n);
  }
  return out;
}

/** The ref for a list of pull requests: master alone, one PR, or a stack. */
export function refForPrs(prs: readonly number[]): SrcRef {
  if (prs.length === 0) return { kind: "master" };
  if (prs.length === 1) return { kind: "pr", pr: prs[0]!, mode: "merge" };
  return { kind: "stack", prs: prs.slice() };
}

/** The pull requests a ref names, in order. Empty for master and tags. */
export function refPrs(r: SrcRef): number[] {
  if (r.kind === "pr") return [r.pr];
  if (r.kind === "stack") return r.prs.slice();
  return [];
}

/**
 * Does this ref point at something that MOVES?
 *
 * The load-bearing question for the source cache. A tag is immutable, so the
 * extracted tree can be reused for ever. `master` and a PR merge ref are not:
 * they are whatever those branches say today.
 *
 * This shipped wrong and it is the bug behind "I don't have the latest
 * llama.cpp updates". The cache was keyed on the ref NAME and reused whenever
 * `CMakeLists.txt` existed, so the first `master` build pinned that machine to
 * that day's master permanently — the developer's own cache held a master from
 * five weeks earlier, and every "rebuild master" since had reused it while the
 * log said "Reusing cached source". A build that silently compiles different
 * code than its name claims is the exact class of thing this app refuses.
 */
export function refMoves(r: SrcRef): boolean {
  // A fork's ref may be a branch or a tag, and nothing here can tell which
  // without asking GitHub — so it is treated as moving. Re-fetching an
  // immutable tag costs one download; reusing a moved branch costs a build
  // that is not what its name says.
  return r.kind !== "tag";
}

/**
 * Does this ref need git, and why?
 *
 * Only a stack does. GitHub publishes each pull request already merged into
 * master and serves a tarball of it, so one PR needs nothing installed; it
 * does NOT publish two of them merged into each other, and nothing but a real
 * merge can produce that. Naming the reason here rather than in the build
 * keeps the prerequisites panel honest — the tool is required for one shape of
 * request and not for the rest, and saying "git is required" flatly would be
 * false for almost everything this app does.
 */
export function refNeedsGit(r: SrcRef): boolean {
  return r.kind === "stack";
}

/** Where to download this ref's source tarball. */
export function tarballUrl(r: SrcRef, repo = REPO): string {
  const base = `https://codeload.github.com/${repo}/tar.gz`;
  switch (r.kind) {
    case "master":
      return `${base}/refs/heads/master`;
    case "tag":
      return `${base}/refs/tags/${r.tag}`;
    case "pr":
      return `${base}/refs/pull/${r.pr}/${r.mode}`;
    case "stack":
      // There is no tarball for a stack — it is assembled locally. The caller
      // checks `refNeedsGit` first; this is the honest answer to a question
      // that should not have been asked.
      return "";
    case "fork":
      // `HEAD` is the fork's default branch, whatever it is called.
      return `https://codeload.github.com/${r.repo}/tar.gz/${r.ref ?? "HEAD"}`;
  }
}

/** The clone URL a stack is assembled from. */
export function repoUrl(repo = REPO): string {
  return `https://github.com/${repo}.git`;
}

/** The git ref that carries a pull request's branch. */
export function prFetchRef(pr: number): string {
  return `refs/pull/${pr}/head`;
}

/** The pull request's page, for a human who wants to read it. */
export function prUrl(pr: number, repo = REPO): string {
  return `https://github.com/${repo}/pull/${pr}`;
}

/**
 * A directory name for this ref that is one path segment and nothing else.
 *
 * `pr/27754` has a slash in it, and both the source cache and the builds
 * registry turn a ref into a directory. Left alone that is a nested path at
 * best and an escape at worst — the same class of bug as the builds-root
 * containment check, and worth closing at the source rather than at every
 * consumer.
 */
export function refDirName(r: SrcRef): string {
  switch (r.kind) {
    case "master":
      return "master";
    case "tag":
      // Tags come from GitHub, but they land in a path, so they are filtered
      // rather than trusted: everything outside the safe set becomes `_`.
      return r.tag.replace(/[^A-Za-z0-9._-]/g, "_") || "tag";
    case "pr":
      return `pr-${r.pr}-${r.mode}`;
    case "stack":
      // Order matters — merging A then B is not merging B then A when they
      // touch the same lines — so it is part of the name, not sorted away.
      return `stack-${r.prs.join("-")}`;
    case "fork": {
      // Injective, because two refs that share a directory share a source
      // tree and a build id — one fork's build silently replaced by
      // another's. It was not: `a/b@c` and `a/b-c` were both `fork-a_b-c`,
      // and `feat/x` and `feat_x` were one branch.
      //
      // The repo half is safe as it stands — an owner has no `_`, so the first
      // one IS the slash. The ref is joined with `@`, which the repo half can
      // never contain; a ref made only of `[A-Za-z0-9.-]` is spelled as is,
      // and any other ref is filtered AND tagged with a hash of its real
      // spelling, so a filtered name can never meet a literal one (a literal
      // has no `_`) or another filtered one. A fork on its default branch
      // keeps the name it always had; one on a named ref gets a new directory
      // once — a fork is re-fetched on every build anyway (`refMoves`), so
      // the cost is one build under the old id left for the user to remove.
      const repo = r.repo.replace(/[^A-Za-z0-9._-]/g, "_");
      if (!r.ref) return `fork-${repo}`;
      const lit = r.ref.replace(/[^A-Za-z0-9.-]/g, "_");
      return `fork-${repo}@${lit === r.ref ? lit : `${lit}_${fnv1a(r.ref)}`}`;
    }
  }
}

/** FNV-1a, 32 bits, as eight hex digits — a stable name for a string, not a
 *  secret. Pure, and the same on every machine, which a directory name has to
 *  be. */
function fnv1a(text: string): string {
  let h = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    h ^= byte;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/** What to call this ref on screen. */
export function refLabel(r: SrcRef): string {
  switch (r.kind) {
    case "master":
      return "master (latest)";
    case "tag":
      return r.tag;
    case "pr":
      return r.mode === "head"
        ? `PR #${r.pr} (author's branch, older base)`
        : `master + PR #${r.pr}`;
    case "stack":
      return `master + ${r.prs.map((n) => `PR #${n}`).join(" + ")}`;
    case "fork":
      return `${r.repo}${r.ref ? ` @ ${r.ref}` : " (default branch)"}`;
  }
}

/**
 * The sentence a build's provenance deserves, for the log and the build list.
 *
 * A PR build is not reproducible from its name alone — `master + PR #27754`
 * means something different tomorrow — so the date is part of the answer, and
 * saying so is cheaper than pretending otherwise.
 */
export function refProvenance(r: SrcRef, at: number): string {
  const when = new Date(at).toISOString().slice(0, 10);
  switch (r.kind) {
    case "master":
      return `master as it stood on ${when}`;
    case "tag":
      return `release ${r.tag}`;
    case "pr":
      return r.mode === "head"
        ? `pull request #${r.pr} alone, as its author left it, fetched ${when}`
        : `master as of ${when}, with pull request #${r.pr} merged into it by GitHub`;
    case "stack":
      return `master as of ${when}, with pull requests ${
        r.prs.map((n) => `#${n}`).join(", ")
      } merged into it in that order`;
    case "fork":
      return `the fork ${r.repo} (${
        r.ref ?? "default branch"
      }) as it stood on ${when} — not upstream llama.cpp`;
  }
}

/**
 * What a pull request's own state means for a ref that names it.
 *
 * The trap this closes is the one that arrives on a GOOD day. GitHub keeps
 * `refs/pull/<N>/merge` after a pull request is merged, and it still points at
 * the merge that was computed back then — so the day #27773 lands in master,
 * `pr/27773` silently starts meaning "master as it was in September, plus a
 * change that is now in master anyway". The build would succeed, the name
 * would look right, and the binary would be older than plain master.
 *
 * So a merged pull request is reported, and the answer is always the same and
 * always simpler: build master. A closed one is reported too, because that
 * code was rejected and nobody should be running it by accident.
 */
export function prStateNote(
  pr: number,
  state: "open" | "merged" | "closed" | "unknown",
): { tone: "info" | "warn"; message: string; useMaster: boolean } | null {
  switch (state) {
    case "merged":
      return {
        tone: "warn",
        useMaster: true,
        message:
          `Pull request #${pr} has been merged into llama.cpp — it is part of master now. Building this ref would give you master as it stood when it merged, which is OLDER than master today. Build master instead and you get this change with everything since.`,
      };
    case "closed":
      return {
        tone: "warn",
        useMaster: true,
        message:
          `Pull request #${pr} was closed without being merged. Its code was not accepted upstream; it can still be built, but it will never appear in master.`,
      };
    case "open":
    case "unknown":
      return null;
  }
}

/**
 * What a 404 on a source tarball MEANS, in the user's terms.
 *
 * Never a raw status. For a PR merge ref the 404 is not "missing", it is a
 * verdict: GitHub publishes that ref only while the branch still merges into
 * master cleanly, so its absence says the two have diverged. That is worth a
 * different sentence from a mistyped tag, and it comes with the two things the
 * user can actually do.
 */
export function refNotFound(r: SrcRef): { reason: string; steps: string[] } {
  switch (r.kind) {
    case "pr":
      return r.mode === "merge"
        ? {
          reason:
            `Pull request #${r.pr} does not currently merge into llama.cpp master.`,
          steps: [
            "GitHub publishes the merged form of a pull request only while it still applies cleanly. It has stopped, which means the branch and today's master have changed the same code and a person has to decide how.",
            `Build the pull request on its own instead — it will not have master's recent changes, but it is what its author tested.`,
            `Or open ${
              prUrl(r.pr)
            } and see whether the author has been asked to rebase; when they do, this option starts working again with no change here.`,
          ],
        }
        : {
          reason: `Pull request #${r.pr} was not found on llama.cpp.`,
          steps: [
            "Check the number against the pull request's page.",
            "A pull request from a deleted fork loses its branch, and cannot be built from here.",
          ],
        };
    case "tag":
      return {
        reason: `llama.cpp has no source tagged ${r.tag}.`,
        steps: [
          "Press Fetch tags to reload the list — the tag may have been renamed or withdrawn.",
        ],
      };
    case "master":
      return {
        reason: "llama.cpp's master branch could not be downloaded.",
        steps: [
          "This is almost always a network problem rather than a missing branch. Check the connection and try again.",
        ],
      };
    case "stack":
      return {
        reason: `Could not assemble master with pull requests ${
          r.prs.map((n) => `#${n}`).join(", ")
        }.`,
        steps: [
          "One of the pull requests could not be fetched. Check each number on its own first — a stack is only as buildable as its parts.",
        ],
      };
    case "fork":
      return {
        reason: `GitHub has no ${
          r.ref ? `branch or tag "${r.ref}" in` : "public repository"
        } ${r.repo}.`,
        steps: [
          `Open https://github.com/${r.repo} and check the spelling — owner and name are exactly as the page's title shows them.`,
          r.ref
            ? "Paste the repository's URL alone to build its default branch."
            : "A private or deleted repository cannot be downloaded from here.",
        ],
      };
  }
}
