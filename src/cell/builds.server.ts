// src/cell/builds.server.ts — get llama.cpp onto this machine. SERVER ONLY.
//
// Two routes to the same result, a directory with `llama-server` and
// `llama-cli` in it:
//
//   source  — fetch the ref's tarball, cmake configure, cmake build.
//             Needs a C++ compiler; CMake is downloaded if missing.
//   release — fetch the official prebuilt asset for this platform/backend.
//             Needs nothing at all, which is what makes the app work on a
//             bare OS (kata: "no prerequisites except a running OS").
//
// Source is fetched as a tarball rather than cloned: it removes git from the
// prerequisite list, downloads ~20x less, and a specific tag is exactly what a
// user asking for "b6234" means.

import { basename, dirname, join, relative, resolve } from "@std/path";
import type { Asset } from "../lib/assets.ts";
import { availableBackends, companionAsset, pickAsset } from "../lib/assets.ts";
import { progressOf } from "../lib/buildlog.ts";
import { CAPS_TIMEOUT_MS, parseHelpFlags } from "../lib/caps.ts";
import { cudaCmakeFlags, cudaPlan } from "../lib/cuda.ts";
import { diagnoseNoAsset } from "../lib/diagnose.ts";
import type { Diagnosis } from "../lib/diagnose.ts";
import type { CudaPlan } from "../lib/cuda.ts";
import type { Backend, Build } from "../lib/types.ts";
import {
  ARCH,
  dirSize,
  download,
  ensureDir,
  exec,
  exists,
  extract,
  fetchJson,
  fetchText,
  makeExecutable,
  paths,
  PLATFORM,
  RateLimited,
  which,
} from "./host.server.ts";
import { assetsFromHtml, assetUrl, shaFromCommitsAtom } from "../lib/github.ts";
import {
  parseRef,
  prFetchRef,
  prUrl,
  refDirName,
  refLabel,
  refMoves,
  refNeedsGit,
  refNotFound,
  refProvenance,
  repoUrl,
  tarballUrl,
} from "../lib/srcref.ts";
import { resolveCmake } from "./prereq.server.ts";
import { DEMO_ENV, demoBuilds } from "../lib/demo.ts";

const REPO = "ggml-org/llama.cpp";
const API = `https://api.github.com/repos/${REPO}`;

export const BIN_SERVER = PLATFORM === "windows"
  ? "llama-server.exe"
  : "llama-server";
export const BIN_CLI = PLATFORM === "windows" ? "llama-cli.exe" : "llama-cli";

/** Progress reporting shared by both routes. `progress` is null while a step's
 *  total is genuinely unknown (cmake configure), never a fake animation. */
export type Progress = {
  step: number;
  steps: string[];
  progress: number | null;
  lines?: string[];
};
export type OnProgress = (p: Progress) => void;

/** A failure that already knows how to explain itself. The cell stores the
 *  diagnosis alongside the job so the UI can render steps and buttons rather
 *  than a wall of text. */
export class BuildFailure extends Error {
  constructor(readonly diagnosis: Diagnosis) {
    super(diagnosis.reason);
    this.name = "BuildFailure";
  }
}

// ── upstream metadata ──────────────────────────────────────────────────────

type GhCommit = { sha: string };
type GhAsset = { name: string; browser_download_url: string; size: number };
type GhRelease = { tag_name: string; assets: GhAsset[]; published_at: string };

/** A tag that names a llama.cpp BUILD — `b<number>`. Since August 2026
 *  upstream also tags semver milestones (`v0.3.0`) and per-commit
 *  `master-<sha>` snapshots; neither carries binaries (the milestone ships a
 *  single `nightly-tag.txt`), so everything below filters to build tags. */
const BUILD_TAG = /^b\d+$/;

/** Build tags from the releases atom feed, newest first — not rate limited. */
async function buildTagsFromAtom(): Promise<string[]> {
  const xml = await fetchText(`https://github.com/${REPO}/releases.atom`);
  const tags = [...xml.matchAll(/\/releases\/tag\/([^"<]+)/g)]
    .map((m) => decodeURIComponent(m[1] as string))
    .filter((t) => BUILD_TAG.test(t));
  return [...new Set(tags)];
}

/** Release tags, newest first. llama.cpp tags builds as `b<number>`.
 *
 * Sourced from the RELEASES list, not `/tags`: releases are ordered by publish
 * date, while `/tags` orders by NAME — and once upstream started tagging
 * `v0.3.0` milestones and `master-<sha>` snapshots, the first hundred tag
 * names stopped containing a single build (measured 2026-08-29: zero
 * `b<number>` entries in `/tags?per_page=100`). */
export async function listRefs(): Promise<string[]> {
  try {
    const rels = await fetchJson<GhRelease[]>(`${API}/releases?per_page=100`);
    return rels.map((r) => r.tag_name).filter((t) => BUILD_TAG.test(t));
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    return await buildTagsFromAtom();
  }
}

/** The newest published BUILD tag. Falls back to the atom feed when the API is
 *  rate limited, so the Update button keeps working.
 *
 * Not `/releases/latest`: that endpoint skips prereleases, the nightly builds
 * are marked prerelease now, and the answer it gives is the assetless
 * `v0.3.0` milestone from days earlier — an "update" that would move a build
 * BACKWARDS and then fail to download. */
export async function latestTag(): Promise<string> {
  try {
    const rels = await fetchJson<GhRelease[]>(`${API}/releases?per_page=30`);
    return rels.map((r) => r.tag_name).find((t) => BUILD_TAG.test(t)) ?? "";
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    return (await buildTagsFromAtom())[0] ?? "";
  }
}

/** The commit `master` currently points at — the only thing that distinguishes
 *  one "master" build from the next. */
export async function masterSha(): Promise<string> {
  try {
    const c = await fetchJson<GhCommit>(`${API}/commits/master`);
    return c.sha;
  } catch (e) {
    if (!(e instanceof RateLimited)) throw e;
    // Same treatment as listRefs/latestTag: without this, an exhausted quota
    // hides the Update button on every `master` build — the one case where the
    // sha IS the version. Atom feeds are not rate limited.
    const xml = await fetchText(
      `https://github.com/${REPO}/commits/master.atom`,
    );
    return shaFromCommitsAtom(xml) ?? "";
  }
}

/**
 * The commit a FORK's ref points at now — what `masterSha` is for upstream.
 * The commits atom feed, not the API: it has no quota, and `HEAD` resolves to
 * the fork's default branch without first asking what that branch is called.
 */
export async function forkSha(
  repo: string,
  ref: string | null,
): Promise<string> {
  const xml = await fetchText(
    `https://github.com/${repo}/commits/${ref ?? "HEAD"}.atom`,
  );
  return shaFromCommitsAtom(xml) ?? "";
}

/** The commit a MOVING ref was built from: the fork's own for a fork, and
 *  upstream master's for everything else (a PR build's master half). */
export async function movingSha(ref: string): Promise<string> {
  const r = parseRef(ref);
  return r.kind === "fork" ? await forkSha(r.repo, r.ref) : await masterSha();
}

export async function listAssets(
  ref: string,
): Promise<{ tag: string; assets: Asset[] }> {
  try {
    // "master" on the release route means "the newest build" — resolved
    // through the releases list, because `/releases/latest` now names an
    // assetless milestone (see `latestTag`). The newest build release that
    // actually carries assets wins; one still uploading is skipped rather
    // than reported as "no asset for your platform".
    const rel = ref === "master"
      ? (await fetchJson<GhRelease[]>(`${API}/releases?per_page=30`))
        .find((r) => BUILD_TAG.test(r.tag_name) && r.assets.length > 0)
      : await fetchJson<GhRelease>(`${API}/releases/tags/${ref}`);
    if (!rel) {
      throw new Error(
        "none of the newest thirty releases is a build with assets — GitHub may be mid-publish; retry in a minute",
      );
    }
    return {
      tag: rel.tag_name,
      assets: rel.assets.map((a) => ({
        name: a.name,
        url: a.browser_download_url,
        sizeB: a.size,
      })),
    };
  } catch (e) {
    // A rate-limited API must not stop an install: downloads were never
    // limited, and github.com's own pages carry the same facts.
    if (!(e instanceof RateLimited)) throw e;
    return await listAssetsWithoutApi(ref);
  }
}

/**
 * The same answer from plain github.com pages, which have no rate limit.
 *
 * Sizes are unknown this way (the HTML rounds them), so they come back as 0 —
 * the download reports real progress from `content-length` regardless, and the
 * picker only ever needed names.
 */
export async function listAssetsWithoutApi(
  ref: string,
): Promise<{ tag: string; assets: Asset[] }> {
  // The atom feed, not the /releases/latest redirect: that page points at the
  // assetless milestone release now (see `latestTag`).
  const tag = ref === "master" ? (await buildTagsFromAtom())[0] : ref;
  if (!tag) {
    throw new Error(
      "GitHub's API is rate limited and the latest release tag could not be resolved from the release page either.",
    );
  }
  const html = await fetchText(
    `https://github.com/${REPO}/releases/expanded_assets/${tag}`,
  );
  const names = assetsFromHtml(html, REPO);
  if (names.length === 0) {
    throw new Error(
      `GitHub's API is rate limited and no assets were listed for ${tag}.`,
    );
  }
  return {
    tag,
    assets: names.map((name) => ({
      name,
      url: assetUrl(REPO, tag, name),
      sizeB: 0,
    })),
  };
}

// ── the build registry ─────────────────────────────────────────────────────

const META = "llama-master.json";

export function buildId(
  origin: Build["origin"],
  ref: string,
  backend: Backend,
): string {
  // Through `refDirName`, because a build id becomes a DIRECTORY under the
  // builds root and `pr/27754` carries a slash. One path segment, always —
  // the containment check in `removeBuild` is the last line of defence, not
  // the first.
  return `${origin}-${refDirName(parseRef(ref))}-${backend}`;
}

async function writeMeta(dir: string, b: Build): Promise<void> {
  await Deno.writeTextFile(join(dir, META), JSON.stringify(b, null, 2));
}

/** Every build on disk. The directory IS the registry — no index file to drift
 *  out of sync with reality, and deleting a directory is a valid uninstall. */
export async function listBuilds(): Promise<Build[]> {
  // Demo mode: a build that does not exist, so the app can be shown working
  // without a llama.cpp install (src/lib/demo.ts).
  if (Deno.env.get(DEMO_ENV) === "1") return demoBuilds();
  const root = paths().builds;
  const out: Build[] = [];
  try {
    for await (const e of Deno.readDir(root)) {
      if (!e.isDirectory) continue;
      const dir = join(root, e.name);
      try {
        const meta = JSON.parse(
          await Deno.readTextFile(join(dir, META)),
        ) as Build;
        // Trust the directory over the metadata: a moved app home must not
        // leave every build pointing at paths that no longer exist.
        const serverBin = await findBinary(dir, BIN_SERVER);
        const cliBin = await findBinary(dir, BIN_CLI);
        if (!serverBin) continue;
        out.push({
          ...meta,
          dir,
          serverBin,
          cliBin: cliBin ?? "",
          sizeB: await dirSize(dir),
        });
      } catch {
        // A directory without readable metadata is a half-finished install;
        // it is skipped here and overwritten by the next install of that id.
      }
    }
  } catch {
    // No builds directory yet.
  }
  return out.sort((a, b) => b.createdAt - a.createdAt);
}

export async function removeBuild(id: string): Promise<void> {
  const root = resolve(paths().builds);
  const dir = resolve(root, id);
  // A direct child of the builds root, decided on the RESOLVED path. The text
  // test this replaced — `startsWith(root + "/")` and no `..` — accepted
  // `id = "/"`: `join(root, "/")` is `root/`, which passes both, and the
  // remove below would have taken every build at once. Same class as the
  // sandbox rule in srv.server.ts (CLAUDE.md: compare paths resolved).
  if (dirname(dir) !== root || basename(dir) !== id) {
    throw new Error(`refusing to remove ${dir}: not a build`);
  }
  await Deno.remove(dir, { recursive: true });
}

/** Depth-limited search for a named binary inside an extracted tree. */
async function findBinary(
  dir: string,
  name: string,
  depth = 3,
): Promise<string | null> {
  const direct = join(dir, name);
  if (await exists(direct)) return direct;
  const inBin = join(dir, "bin", name);
  if (await exists(inBin)) return inBin;
  if (depth <= 0) return null;
  try {
    for await (const e of Deno.readDir(dir)) {
      if (!e.isDirectory) continue;
      const found = await findBinary(join(dir, e.name), name, depth - 1);
      if (found) return found;
    }
  } catch {
    // Unreadable subtree — nothing to find here.
  }
  return null;
}

// ── route 1: prebuilt release ──────────────────────────────────────────────

/** The newest CUDA runtime every NVIDIA card's driver can run, or 0. */
async function driverCudaMax(): Promise<number> {
  try {
    const { gpus } = await import("./hw.server.ts");
    const v = (await gpus()).map((g) => g.cudaDriver ?? 0).filter((x) => x > 0);
    return v.length ? Math.min(...v) : 0;
  } catch {
    return 0;
  }
}

export async function installRelease(
  opts: {
    ref: string;
    backend: Backend;
    assetName?: string;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = ["Find release", "Download", "Extract", "Verify"];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  p(0, null, [`Looking up ${opts.ref} on ${REPO}`]);
  const { tag, assets } = await listAssets(opts.ref);
  const cudaMax = await driverCudaMax();
  const asset = opts.assetName
    ? assets.find((a) => a.name === opts.assetName) ?? null
    : pickAsset(assets, PLATFORM, ARCH, opts.backend, cudaMax);
  if (!asset) {
    // Never a filename dump: say why, and give the route that works. The
    // prerequisite state is read here so the advice is accurate — "you already
    // have nvcc, just switch route" reads very differently from "install nvcc".
    const { detect } = await import("./prereq.server.ts");
    const found = new Set(
      (await detect()).filter((i) => i.found).map((i) => i.id),
    );
    throw new BuildFailure(
      diagnoseNoAsset(
        {
          origin: "release",
          backend: opts.backend,
          platform: PLATFORM,
          arch: ARCH,
          availableBackends: availableBackends(
            assets,
            PLATFORM,
            ARCH,
            cudaMax,
          ),
          found,
        },
        assets.length,
      ),
    );
  }

  p(1, 0, [
    asset.sizeB > 0
      ? `Downloading ${asset.name} (${(asset.sizeB / 1e6).toFixed(0)} MB)`
      : `Downloading ${asset.name}`,
  ]);
  const bytes = await download(
    asset.url,
    (received, total) => p(1, total ? received / total : null),
    opts.signal,
  );

  const id = buildId("release", tag, opts.backend);
  const dest = join(paths().builds, id);
  // Staged beside its final name: the installed build of this id keeps
  // working until the new one has PROVED it runs (`promote`).
  const dir = await freshStaging(dest);
  p(2, null, [`Extracting into ${dir}`]);
  const n = await extract(
    bytes,
    dir,
    asset.name.endsWith(".zip") ? "zip" : "tar.gz",
  );

  // The CUDA runtime, when the release ships it apart from the binaries.
  // Extracted into the SAME directory: upstream's binaries carry
  // `RUNPATH $ORIGIN`, so libraries beside `llama-server` are found without
  // touching the environment — and without this the install "succeeds" and
  // the first Start fails with "libcudart.so.13: cannot open shared object".
  const rt = companionAsset(assets, asset);
  if (rt) {
    p(2, 0, [
      rt.sizeB > 0
        ? `Downloading the CUDA runtime ${rt.name} (${
          (rt.sizeB / 1e6).toFixed(0)
        } MB)`
        : `Downloading the CUDA runtime ${rt.name}`,
    ]);
    const rtBytes = await download(
      rt.url,
      (received, total) => p(2, total ? received / total : null),
      opts.signal,
    );
    const m = await extract(
      rtBytes,
      dir,
      rt.name.endsWith(".zip") ? "zip" : "tar.gz",
    );
    p(2, 1, [`${m} CUDA runtime files added`]);
  }

  p(3, null, [`${n} files extracted, checking binaries`]);
  const build = await commitBuild(
    {
      id,
      ref: tag,
      origin: "release",
      backend: opts.backend,
      dir,
    },
    dest,
    opts.signal,
  );
  p(3, 1, [`${BIN_SERVER} ready at ${build.serverBin}`]);
  return build;
}

// ── route 2: build from source ─────────────────────────────────────────────

const BACKEND_FLAGS: Record<Backend, string[]> = {
  cpu: [],
  cuda: ["-DGGML_CUDA=ON"],
  vulkan: ["-DGGML_VULKAN=ON"],
  hip: ["-DGGML_HIP=ON"],
  metal: ["-DGGML_METAL=ON"],
};

/**
 * A pull request's title and author, from the plain github.com page.
 *
 * No API call: the anonymous quota is 60/hour and it is routinely exhausted
 * (`src/lib/github.ts`), while this page is not rate limited at all — and the
 * `<title>` tag carries everything needed. It is decoration in the strict
 * sense, so every failure returns "" rather than throwing: a build must not be
 * blocked because a title could not be read.
 */
export async function prTitle(pr: number): Promise<string> {
  try {
    const html = await fetchText(prUrl(pr));
    const raw = /<title>([^<]*)<\/title>/i.exec(html)?.[1] ?? "";
    // "model: add GLM-5-Next by danielhanchen · Pull Request #27754 · ggml-org/llama.cpp · GitHub"
    const cut = raw.split(" · Pull Request")[0]?.trim() ?? "";
    return cut.length > 0 && cut.length < 300
      ? cut.replace(/&amp;/g, "&").replace(/&#39;/g, "'").replace(
        /&quot;/g,
        '"',
      )
      : "";
  } catch {
    return "";
  }
}

/**
 * A pull request's state, from the plain github.com page.
 *
 * The trap this closes arrives on a GOOD day: GitHub keeps
 * `refs/pull/<N>/merge` after a pull request lands, still pointing at the
 * merge computed back then. So the day a PR merges, its ref quietly starts
 * meaning "master as it was months ago, plus a change master already has" —
 * the build succeeds, the name looks right, and the binary is older than plain
 * master. Reading the state costs one un-rate-limited page.
 *
 * `unknown` on any failure: this decides what the panel SAYS, never whether a
 * build may proceed, so a network hiccup must not block one.
 */
export async function prState(
  pr: number,
): Promise<"open" | "merged" | "closed" | "unknown"> {
  try {
    const html = await fetchText(prUrl(pr));
    if (/"state"\s*:\s*"MERGED"/i.test(html) || />\s*Merged\s*</i.test(html)) {
      return "merged";
    }
    if (/"state"\s*:\s*"CLOSED"/i.test(html) || />\s*Closed\s*</i.test(html)) {
      return "closed";
    }
    if (/"state"\s*:\s*"OPEN"/i.test(html) || />\s*Open\s*</i.test(html)) {
      return "open";
    }
    return "unknown";
  } catch {
    return "unknown";
  }
}

/**
 * Master with several pull requests merged into it, assembled here.
 *
 * The one source shape GitHub cannot hand over ready-made: it publishes each
 * PR merged into master, never two of them merged into each other. So this is
 * the only path that needs git, and it is asked for only when the ref is a
 * stack (`refNeedsGit`).
 *
 * A shallow fetch, because the history is not wanted — only the tree. Each PR
 * is merged in the ORDER GIVEN, and a conflict stops the whole thing and names
 * the pull request and the files: "it did not build" is not an answer a person
 * can act on, and a half-merged tree left on disk would be worse than none.
 */
async function assembleStack(
  dir: string,
  prs: number[],
  say: (lines: string[]) => void,
  signal?: AbortSignal,
): Promise<void> {
  const git = await which("git");
  if (!git) {
    throw new Error(
      "Building master with more than one pull request needs git, and it is not installed. " +
        "Install git, or build a single pull request — that route needs nothing, because GitHub does the merge itself.",
    );
  }
  // ONE clone, reused by every stack, kept out of the source trees it feeds.
  //
  // Two reasons it is not inside `dir`. A build directory sits in there and
  // `git archive` gives a clean tree without one; and a clone is expensive
  // enough that throwing it away per build would make a two-line change cost a
  // full download. It also means a fetch that fails can fall back on what is
  // already here, which is the difference between "GitHub is throttling you"
  // being an inconvenience and being a dead end.
  const repo = join(paths().sources, "_gitrepo");
  const run = async (args: string[], what: string, cwd = repo) => {
    const r = await exec(git, args, { cwd, env: NO_PROMPT });
    if (r.code !== 0) {
      throw new Error(
        `${what} failed: ${(r.stderr || r.stdout).trim().slice(0, 400)}`,
      );
    }
    return r;
  };
  const tried = async (args: string[], cwd = repo) =>
    await exec(git, args, { cwd, env: NO_PROMPT });

  if (!(await exists(join(repo, "HEAD")))) {
    await ensureDir(repo);
    await run(["init", "-q", "--bare"], "git init");
    await run(["remote", "add", "origin", repoUrl()], "git remote add");
  }

  const refspecs = [
    "+refs/heads/master:refs/remotes/origin/master",
    ...prs.map((n) => `+${prFetchRef(n)}:refs/prs/${n}`),
  ];
  say([
    `Fetching master and ${prs.length} pull request${
      prs.length === 1 ? "" : "s"
    }`,
  ]);
  const fetched = await tried([
    "fetch",
    "-q",
    "--depth=300",
    "origin",
    ...refspecs,
  ]);
  const needed = [
    "refs/remotes/origin/master",
    ...prs.map((n) => `refs/prs/${n}`),
  ];
  const have: string[] = [];
  for (const ref of needed) {
    if ((await tried(["rev-parse", "--verify", "-q", ref])).code === 0) {
      have.push(ref);
    }
  }
  if (fetched.code !== 0) {
    // A fetch can fail for reasons that have nothing to do with this request —
    // GitHub throttles unauthenticated git with a 401, and a laptop goes
    // offline. If everything needed is already here, the build can go ahead;
    // what it must NOT do is go ahead quietly, because the tree may be older
    // than the name suggests, which is the one thing this app refuses.
    if (have.length !== needed.length) {
      throw new Error(
        `Could not fetch llama.cpp: ${
          (fetched.stderr || fetched.stdout).trim().slice(0, 300)
        }\nGitHub answers 401 to unauthenticated git when it is throttling an address; a GITHUB_TOKEN in the environment raises that limit, and waiting also works.`,
      );
    }
    say([
      "Could not reach GitHub, so this is built from the copy already on disk — it may be older than master is now.",
    ]);
  }
  if (signal?.aborted) throw new Error("cancelled");

  await run(
    ["branch", "-f", "work", "refs/remotes/origin/master"],
    "git branch",
  );
  for (const n of prs) {
    if (signal?.aborted) throw new Error("cancelled");
    // A bare repo has no working tree, so the merge is done with a temporary
    // one; `git merge-tree` would be neater but its writable form is newer
    // than the git on plenty of machines this has to run on.
    const wt = join(repo, `wt-${n}`);
    await Deno.remove(wt, { recursive: true }).catch(() => {});
    await run(
      ["worktree", "add", "-q", "--detach", wt, "work"],
      "git worktree add",
    );
    try {
      await exec(git, ["config", "user.email", "builds@llama.master"], {
        cwd: wt,
        env: NO_PROMPT,
      });
      await exec(git, ["config", "user.name", "llama.master"], {
        cwd: wt,
        env: NO_PROMPT,
      });
      let m = await tried(["merge", "--no-edit", "-q", `refs/prs/${n}`], wt);
      // "refusing to merge unrelated histories" from a SHALLOW clone does not
      // mean the branches are unrelated — it means the common ancestor is
      // older than the history we fetched, so git cannot see it. The fix is
      // more history, never `--allow-unrelated-histories`: that flag would
      // cheerfully splice two genuinely unrelated trees together and hand the
      // result to a compiler.
      if (m.code !== 0 && /unrelated histories|no merge base/i.test(m.stderr)) {
        say([`Fetching more history to find where #${n} branched from master`]);
        await tried(["merge", "--abort"], wt);
        const deep = await tried([
          "fetch",
          "-q",
          "--deepen=5000",
          "origin",
          ...refspecs,
        ]);
        if (deep.code !== 0) {
          throw new Error(
            `Pull request #${n} branched from a commit older than the history on disk, and more could not be fetched: ${
              (deep.stderr || deep.stdout).trim().slice(0, 200)
            }`,
          );
        }
        m = await tried(["merge", "--no-edit", "-q", `refs/prs/${n}`], wt);
      }
      if (m.code !== 0) {
        // Name the pull request AND the files. A conflict is a fact about two
        // changes touching the same lines, and the only useful report says
        // which ones.
        const files = await tried(
          ["diff", "--name-only", "--diff-filter=U"],
          wt,
        );
        const list = files.stdout.trim().split("\n").filter(Boolean);
        // No conflicted files means git refused for some OTHER reason, and
        // reporting that as a conflict sent one debugging session looking for
        // overlapping edits that did not exist. Git's own words are better
        // than a guess.
        if (list.length === 0) {
          throw new Error(
            `Could not merge pull request #${n}: ${
              (m.stderr || m.stdout).trim().slice(0, 300) ||
              "git gave no reason"
            }`,
          );
        }
        throw new Error(
          `Pull request #${n} conflicts with what is already merged.\n` +
            `Both changed: ${list.slice(0, 8).join(", ")}${
              list.length > 8 ? `, and ${list.length - 8} more` : ""
            }.\n` +
            `Drop #${n} from the list, or put it earlier — order matters when two pull requests touch the same lines.`,
        );
      }
      const head = await run(["rev-parse", "HEAD"], "git rev-parse", wt);
      await run(["branch", "-f", "work", head.stdout.trim()], "git branch");
      say([`Merged #${n}`]);
    } finally {
      await Deno.remove(wt, { recursive: true }).catch(() => {});
      await tried(["worktree", "prune"]);
    }
  }

  // `git archive` writes the merged tree with no repository in it, which keeps
  // the source directory the same shape a tarball produces — cmake, the build
  // directory and nothing else.
  await Deno.remove(dir, { recursive: true }).catch(() => {});
  await ensureDir(dir);
  const tar = await which("tar");
  if (!tar) throw new Error("tar is required to unpack the assembled source");
  const archive = join(repo, "stack.tar");
  await run(["archive", "-o", archive, "work"], "git archive");
  const untar = await exec(tar, ["xf", archive, "-C", dir]);
  await Deno.remove(archive).catch(() => {});
  if (untar.code !== 0) {
    throw new Error(`unpacking the assembled source failed: ${untar.stderr}`);
  }
}

/** Git must never stop for credentials on a public clone: a prompt in a
 *  headless build is a hang with no explanation. */
const NO_PROMPT = { GIT_TERMINAL_PROMPT: "0", GIT_ASKPASS: "true" };

/** A tarball has no git metadata, so a source build reports
 *  `version: 0 (unknown)`. When the ref is a release tag we already know the
 *  number, so hand it to cmake and let the binary tell the truth about itself. */
export function buildNumberFlags(ref: string): string[] {
  const m = /^b(\d+)$/.exec(ref);
  return m ? [`-DLLAMA_BUILD_NUMBER=${m[1]}`] : [];
}

/** What CUDA can target here — nvcc's version against the driver's report of
 *  each GPU. Read at build time so the answer is never stale. */
export async function detectCudaPlan(): Promise<
  CudaPlan & { nvcc: string; root: string }
> {
  // A toolkit this app installed WINS over whatever is on PATH. That is the
  // whole point of installing one: the system nvcc is the thing that could not
  // build for these cards, and leaving cmake to find it first would make the
  // prerequisite's Fix button do nothing visible.
  const { managedCuda } = await import("./prereq.server.ts");
  const { nvidiaSmi } = await import("./hw.server.ts");
  const own = await managedCuda();
  const [nvcc, smi] = await Promise.all([
    exec(own?.nvcc ?? "nvcc", ["--version"], { timeoutMs: 10_000 }),
    nvidiaSmi(["--query-gpu=compute_cap", "--format=csv,noheader"]),
  ]);
  const caps = smi.code === 0
    ? smi.stdout.split("\n").map((l) => Number(l.trim())).filter((n) => n > 0)
    : [];
  return {
    ...cudaPlan(nvcc.stdout || nvcc.stderr, caps),
    nvcc: own?.nvcc ?? "",
    root: own?.path ?? "",
  };
}

export async function buildFromSource(
  opts: {
    ref: string;
    backend: Backend;
    jobs: number;
    native: boolean;
    /** Raise `GGML_SCHED_MAX_SPLIT_INPUTS` to this value; 0/absent = stock. */
    schedCap?: number;
    signal?: AbortSignal;
  },
  onProgress: OnProgress,
): Promise<Build> {
  const steps = ["Fetch source", "Configure", "Compile", "Install"];
  const p = (step: number, progress: number | null, lines?: string[]) =>
    onProgress({ step, steps, progress, lines });

  const cmake = await resolveCmake();
  if (!cmake) {
    throw new Error(
      "CMake not found. Install it from the Prerequisites panel — llama.master can download it.",
    );
  }

  // Vulkan: point cmake at the app's own SPIRV-Headers when the system has
  // none. `find_package` is satisfied by CMAKE_PREFIX_PATH, but llama.cpp does
  // not link the imported target — it expects the headers on the default
  // include path — so the include directory has to be added too, or the compile
  // fails later with "'spv' has not been declared".
  // Everything that lands in CMAKE_CXX_FLAGS collects here and is emitted as
  // ONE -D: two `-DCMAKE_CXX_FLAGS=` arguments would be last-writer-wins, and
  // the Vulkan include path silently losing to the sched-cap define (or the
  // reverse) is exactly the kind of quiet breakage this app exists to refuse.
  const cxxFlags: string[] = [];
  let vulkanFlags: string[] = [];
  if (opts.backend === "vulkan") {
    const { resolveSpirvHeaders } = await import("./prereq.server.ts");
    const spirv = await resolveSpirvHeaders();
    if (!spirv) {
      throw new Error(
        "SPIRV-Headers not found. Install it from the Prerequisites panel — llama.master can download it (headers only, no root needed).",
      );
    }
    if (spirv.managed) {
      vulkanFlags = [`-DCMAKE_PREFIX_PATH=${spirv.path}`];
      cxxFlags.push(`-isystem ${join(spirv.path, "include")}`);
      p(0, null, [`Using the app's SPIRV-Headers at ${spirv.path}`]);
    }
  }

  // Raise llama.cpp's graph-split input cap (`#ifndef`-guarded, stock 30).
  // With routed experts in RAM the scheduler needs more cross-device inputs
  // per split as the context grows; measured on DeepSeek-V4: 256k generates,
  // 512k dies at the assert. The define is the supported way past it.
  if (opts.schedCap && opts.schedCap > 0) {
    cxxFlags.push(`-DGGML_SCHED_MAX_SPLIT_INPUTS=${opts.schedCap}`);
    p(0, null, [
      `Raising GGML_SCHED_MAX_SPLIT_INPUTS to ${opts.schedCap} (stock: 30) — extreme contexts with experts in RAM need more cross-device inputs per graph split than llama.cpp's default allows.`,
    ]);
  }

  // CUDA: name the architectures explicitly. Left to auto-detection, cmake asks
  // nvcc for the GPU's native arch, and an nvcc older than the card dies with
  // `Unsupported gpu architecture` several minutes into the compile.
  let cudaFlags: string[] = [];
  if (opts.backend === "cuda") {
    const plan = await detectCudaPlan();
    if (plan.mode === "impossible") {
      throw new Error(`${plan.reason} ${plan.remedy}`);
    }
    cudaFlags = cudaCmakeFlags(plan);
    if (plan.nvcc) {
      // Both, and `CMAKE_CUDA_HOST_COMPILER` left alone: cmake finds the
      // toolkit through the compiler it is given, and pinning the host
      // compiler as well is how a build starts using a gcc the system did not
      // choose. Naming the path in the log matters — a build that quietly used
      // a different compiler than the one on PATH has to say so.
      cudaFlags = [
        `-DCMAKE_CUDA_COMPILER=${plan.nvcc}`,
        `-DCUDAToolkit_ROOT=${plan.root}`,
        ...cudaFlags,
      ];
      p(0, null, [
        `Using the CUDA toolkit llama.master installed: ${plan.root}`,
      ]);
    }
    p(0, null, [plan.reason, ...(plan.remedy ? [plan.remedy] : [])]);
  }

  // 1 — source tarball.
  //
  // Cached only for a ref that CANNOT change. A tag is immutable, so its tree
  // is reused for ever; `master` and a pull request are whatever those
  // branches say today, and reusing them is how this app came to compile
  // five-week-old master under the name "master" on the developer's own
  // machine, announcing "Reusing cached source" while it did it. A build that
  // silently compiles something other than what its name claims is the exact
  // thing this app exists to refuse, so a moving ref is re-fetched every time
  // — 37 MB against a compile measured in minutes.
  const src = parseRef(opts.ref);
  const srcDir = join(paths().sources, refDirName(src));
  const moves = refMoves(src);
  const cached = await exists(join(srcDir, "CMakeLists.txt"));
  if (refNeedsGit(src) && src.kind === "stack") {
    // Always re-assembled: master moves under it, and so may every branch in
    // it. Same rule as every other moving ref, for the same reason.
    await assembleStack(
      srcDir,
      src.prs,
      (lines) => p(0, null, lines),
      opts.signal,
    );
    p(0, 1, [
      `Source assembled at ${srcDir}`,
      refProvenance(src, Date.now()),
    ]);
  } else if (moves || !cached) {
    const url = tarballUrl(src);
    p(0, 0, [
      cached && moves
        ? `${refLabel(src)} moves, so it is fetched fresh rather than reused`
        : `Fetching ${refLabel(src)}`,
      url,
    ]);
    let bytes: Uint8Array;
    try {
      bytes = await download(
        url,
        (received, total) => p(0, total ? received / total : null),
        opts.signal,
      );
    } catch (e) {
      // Never a raw 404. For a pull request's merged form the status IS the
      // verdict — GitHub publishes that ref only while the branch still
      // applies to master — and the user needs that sentence, not the number
      // (`src/lib/srcref.ts:refNotFound`).
      const msg = String(e);
      if (/→ 404\b/.test(msg)) {
        const nf = refNotFound(src);
        throw new Error(`${nf.reason}\n${nf.steps.join("\n")}`);
      }
      throw e;
    }
    await Deno.remove(srcDir, { recursive: true }).catch(() => {});
    await ensureDir(srcDir);
    const n = await extract(bytes, srcDir, "tar.gz");
    p(0, 1, [
      `${n} source files extracted to ${srcDir}`,
      refProvenance(src, Date.now()),
    ]);
  } else {
    p(0, 1, [
      `Reusing cached source at ${srcDir} — ${refLabel(src)} cannot change`,
    ]);
  }

  // 2 — configure. Nothing before this point kills a process, so a Cancel
  // pressed during extraction or toolchain detection is honoured here.
  opts.signal?.throwIfAborted();
  const buildDir = join(srcDir, `build-${opts.backend}`);
  await ensureDir(buildDir);
  const configureArgs = [
    "-S",
    srcDir,
    "-B",
    buildDir,
    "-DCMAKE_BUILD_TYPE=Release",
    // No libcurl dependency: model downloading is this app's job, not
    // llama.cpp's, and requiring libcurl-dev would break the "bare OS" promise.
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    `-DGGML_NATIVE=${opts.native ? "ON" : "OFF"}`,
    // The installed build must not need the tree it was compiled in. CMake
    // writes ABSOLUTE build-tree RUNPATHs by default, so every source build
    // resolved libllama/libggml out of `cache/sources/<ref>/build-*` — and a
    // moving ref re-extracts that directory on the next build, which is how
    // `source-master-cpu` and `-vulkan` came to fail with "libllama-server-
    // impl.so: cannot open shared object file" while listed as installed.
    // `$ORIGIN` for in-tree paths; the CUDA toolkit's own path stays absolute.
    "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
    ...BACKEND_FLAGS[opts.backend],
    ...buildNumberFlags(opts.ref),
    ...cudaFlags,
    ...vulkanFlags,
    // ALWAYS passed, even empty: the build directory is reused, and CMake
    // caches CMAKE_CXX_FLAGS — turning the sched-cap option OFF must reset
    // the cached define, or the rebuild silently keeps it while claiming
    // stock.
    `-DCMAKE_CXX_FLAGS=${cxxFlags.join(" ")}`,
  ];
  p(1, null, [`${cmake.path} ${configureArgs.join(" ")}`]);
  const cfgCode = await runStreaming(
    cmake.path,
    configureArgs,
    srcDir,
    opts.signal,
    (lines) => p(1, null, lines),
  );
  if (cfgCode !== 0) {
    throw new Error(
      `cmake configure failed (exit ${cfgCode}). The log tail above names the missing dependency.`,
    );
  }

  // 3 — compile. cmake prints `[ nn%]`, so the bar is real.
  const buildArgs = [
    "--build",
    buildDir,
    "--config",
    "Release",
    "-j",
    String(Math.max(1, opts.jobs)),
    "--target",
    "llama-server",
    "llama-cli",
  ];
  p(2, 0, [`${cmake.path} ${buildArgs.join(" ")}`]);
  let last = 0;
  const code = await runStreaming(
    cmake.path,
    buildArgs,
    srcDir,
    opts.signal,
    (lines) => {
      for (const l of lines) {
        const pr = progressOf(l);
        if (pr !== null) last = pr;
      }
      p(2, last, lines);
    },
  );
  if (code !== 0) throw new Error(`build failed (exit ${code})`);

  // 4 — install into the durable builds directory. A Cancel that lands after
  // the compile must not replace the installed build it was meant to keep.
  opts.signal?.throwIfAborted();
  const id = buildId("source", opts.ref, opts.backend);
  const dest = join(paths().builds, id);
  // Staged, like a release: a build that fails its checks below must not
  // have cost the user the working build of the same id.
  const staging = await freshStaging(dest);
  await ensureDir(join(staging, "bin"));
  p(3, null, [`Installing to ${dest}`]);
  const binSrc = join(buildDir, "bin");
  let copied = 0;
  for await (const e of Deno.readDir(binSrc)) {
    // The whole bin/ directory: the binaries need the ggml/llama shared
    // objects that sit beside them — including the SONAME symlinks
    // (`libllama.so.0` → `libllama.so.0.24.0`), which are what the loader
    // actually asks for. Skipping them (`isFile` is false for a link) left
    // every build resolving those names from the source cache instead.
    const to = join(staging, "bin", e.name);
    if (e.isSymlink) {
      await Deno.symlink(await Deno.readLink(join(binSrc, e.name)), to);
    } else if (e.isFile) {
      await Deno.copyFile(join(binSrc, e.name), to);
    } else continue;
    copied++;
  }
  p(3, 0.9, [`${copied} files installed`]);

  const build = await commitBuild(
    {
      id,
      ref: opts.ref,
      origin: "source",
      backend: opts.backend,
      dir: staging,
      // A tag identifies itself; a moving ref does not, so record what it was.
      // For a pull request this is the master it was merged INTO, which is the
      // half of "master + PR #27754" that the name cannot carry.
      // For a fork it is the FORK's commit: upstream master's sha would be a
      // fact about a tree that was never compiled.
      sourceSha: refMoves(src) ? await movingSha(opts.ref).catch(() => "") : "",
      // A build that behaves differently must say why.
      ...(opts.schedCap && opts.schedCap > 0
        ? { schedCap: opts.schedCap }
        : {}),
    },
    dest,
    opts.signal,
  );
  p(3, 1, [`${BIN_SERVER} ready at ${build.serverBin}`]);
  return build;
}

/**
 * Libraries `ldd` resolves from inside `forbidden` (the source cache), or
 * cannot resolve at all. Pure over ldd's text, so it is testable.
 */
export function borrowedLibs(ldd: string, forbidden: string): string[] {
  const out: string[] = [];
  for (const line of ldd.split("\n")) {
    const m = /^\s*(\S+)\s+=>\s+(not found|\S+)/.exec(line);
    if (!m) continue;
    const [, name, where] = m as unknown as [string, string, string];
    if (where === "not found" || where.startsWith(forbidden + "/")) {
      out.push(name);
    }
  }
  return out;
}

/** Stream a child process, batching lines so the UI gets one dispatch per tick
 *  instead of one per line — a cmake build emits thousands. */
async function runStreaming(
  bin: string,
  args: string[],
  cwd: string,
  signal: AbortSignal | undefined,
  emit: (lines: string[]) => void,
): Promise<number> {
  const { execStream } = await import("./host.server.ts");
  let pending: string[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    if (pending.length === 0) return;
    emit(pending);
    pending = [];
  };
  const code = await execStream(bin, args, { cwd, signal }, (line) => {
    pending.push(line);
    timer ??= setTimeout(flush, 100);
  });
  if (timer !== null) clearTimeout(timer);
  flush();
  return code;
}

/** Locate the binaries, make them executable, write the metadata file. */
/** An empty staging directory for `dest`, cleared of any earlier attempt. It
 *  carries no metadata until `promote`, so `listBuilds` never lists it. */
async function freshStaging(dest: string): Promise<string> {
  const staging = `${dest}.partial`;
  await Deno.remove(staging, { recursive: true }).catch(() => {});
  await ensureDir(staging);
  return staging;
}

/**
 * Swap a staged build that passed `finalize` in over the installed one.
 *
 * The metadata is written LAST, at the final path: a directory without it is
 * a half-finished install that `listBuilds` skips, so a crash at any point
 * leaves either the old build or the new one — never a refused binary listed
 * as installed. `$ORIGIN` RUNPATHs are relative, so the rename moves nothing
 * the loader depends on.
 */
async function promote(staged: Build, dest: string): Promise<Build> {
  await Deno.remove(dest, { recursive: true }).catch(() => {});
  await Deno.rename(staged.dir, dest);
  const moved = (p: string) => p && join(dest, relative(staged.dir, p));
  const build: Build = {
    ...staged,
    dir: dest,
    serverBin: moved(staged.serverBin),
    cliBin: moved(staged.cliBin),
  };
  await writeMeta(dest, build);
  return build;
}

/** Check the install staged at `base.dir` and, only if it runs, put it at
 *  `dest` in place of whatever build of that id was there. A Cancel that
 *  lands before the swap keeps the installed build. */
export async function commitBuild(
  base: Omit<Build, "serverBin" | "cliBin" | "createdAt" | "sizeB">,
  dest: string,
  signal?: AbortSignal,
): Promise<Build> {
  const staged = await finalize(base);
  if (signal?.aborted) {
    await Deno.remove(base.dir, { recursive: true }).catch(() => {});
    signal.throwIfAborted();
  }
  return await promote(staged, dest);
}

/** Check a staged install actually runs, and describe it. Throws on a build
 *  that cannot start, removing it; writes no metadata. */
async function finalize(
  base: Omit<Build, "serverBin" | "cliBin" | "createdAt" | "sizeB">,
): Promise<Build> {
  try {
    return await checkStaged(base);
  } catch (e) {
    // A refused build is not kept: the installed one is still in place, and
    // hundreds of MB of binaries that cannot run are nobody's backup.
    await Deno.remove(base.dir, { recursive: true }).catch(() => {});
    throw e;
  }
}

async function checkStaged(
  base: Omit<Build, "serverBin" | "cliBin" | "createdAt" | "sizeB">,
): Promise<Build> {
  const serverBin = await findBinary(base.dir, BIN_SERVER);
  const cliBin = await findBinary(base.dir, BIN_CLI);
  if (!serverBin) {
    throw new Error(
      `${BIN_SERVER} not found under ${base.dir} — the archive layout was not what we expected`,
    );
  }
  for (const b of [serverBin, cliBin]) if (b) await makeExecutable(b);

  const build: Build = {
    ...base,
    serverBin,
    cliBin: cliBin ?? "",
    createdAt: Date.now(),
    sizeB: await dirSize(base.dir),
  };

  // Prove it runs. A binary that cannot start (missing CUDA runtime, wrong
  // glibc) must fail here, not two clicks later when the user hits Start.
  const check = await exec(serverBin, ["--version"]);
  if (check.code === 127) {
    throw new Error(
      `installed ${BIN_SERVER} will not execute: ${
        check.stderr.trim() || "not runnable"
      }`,
    );
  }
  // …and prove it runs ON ITS OWN. `--version` passes while the tree it was
  // compiled in still exists, so a build that borrows its libraries from the
  // source cache looked healthy until the next build of that ref replaced
  // the cache. On Linux the loader can say where each library comes from.
  if (PLATFORM === "linux") {
    const leaks = borrowedLibs(
      (await exec("ldd", [serverBin])).stdout,
      paths().sources,
    );
    if (leaks.length > 0) {
      throw new Error(
        `installed ${BIN_SERVER} still loads ${
          leaks.join(", ")
        } from outside its own directory — it would break when that directory changes. This is a bug in the install step; please report it.`,
      );
    }
  }
  return build;
}

/**
 * Ask a build which flags it accepts.
 *
 * `llama-server --help` prints every option and exits, so this is a fact about
 * the binary on disk rather than an inference from its version string — which
 * is the only alternative, and which means nothing for the builds that most
 * need the answer (a PR stack, a `master` from an unknown day).
 *
 * Sandboxed like a start: only a binary under the builds root is executed,
 * compared RESOLVED rather than as text, because `<buildsRoot>/../../usr/bin/x`
 * begins with the root as a string and leaves it as a path. `--help` is not a
 * dangerous argument, but "we only ever run our own binaries" is a rule, and a
 * rule with an exception for the harmless case is not a rule.
 *
 * Returns `[]` rather than throwing when the probe cannot run or the binary
 * answers nothing usable. An unknown capability set reads as "switch nothing
 * extra on" (`caps.ts:supportsFlag`), which is the safe direction: the cost of
 * a failed probe is that the tuner is conservative, not that a server dies.
 */
/** A build whose binary cannot load a library it links against. */
export class BrokenBuild extends Error {
  constructor(readonly lib: string) {
    super(`cannot load ${lib}`);
    this.name = "BrokenBuild";
  }
}

export async function probeCaps(bin: string): Promise<string[]> {
  if (!bin) return [];
  const root = resolve(paths().builds);
  if (!resolve(bin).startsWith(root + "/")) {
    throw new Error(
      `refusing to run ${bin}: only binaries under ${root} may be probed`,
    );
  }
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), CAPS_TIMEOUT_MS);
  try {
    const out = await new Deno.Command(bin, {
      args: ["--help"],
      stdout: "piped",
      stderr: "piped",
      signal: ac.signal,
    }).output();
    const dec = new TextDecoder();
    // Both pipes: llama.cpp has printed usage to stderr in the past, and a
    // probe that reads only stdout would report a modern build as knowing
    // nothing at all — which is the one answer that is worse than no answer,
    // because it looks like a measurement.
    const help = dec.decode(out.stdout) + "\n" + dec.decode(out.stderr);
    // A binary that cannot load its own libraries is not "a build that
    // knows no flags" — it is a build that cannot start, and the one place
    // that learns it early is here. Named so the cell can say so.
    const missing = /error while loading shared libraries: ([^:]+)/.exec(help);
    if (missing) throw new BrokenBuild(missing[1] as string);
    return [...parseHelpFlags(help)].sort();
  } catch (e) {
    if (e instanceof BrokenBuild) throw e;
    return [];
  } finally {
    clearTimeout(timer);
  }
}
