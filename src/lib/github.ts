// src/lib/github.ts — talking to GitHub without depending on its API budget.
//
// The REST API allows 60 requests per hour to an unauthenticated IP. That
// sounds generous until an app polls for updates every five minutes and the
// user also has a CI checkout and two other tools on the same network — then
// the release list returns `403 Forbidden` and the app looks broken.
//
// Two answers, both here:
//   1. When the API says "rate limited", say THAT — with when it resets and how
//      to raise the limit — instead of surfacing a bare 403.
//   2. Fall back to plain github.com pages, which are not rate limited: the
//      `/releases/latest` redirect names the newest tag, and the
//      `/releases/expanded_assets/<tag>` fragment (what GitHub's own UI fetches
//      to expand the asset list) names every file. Downloads were never
//      limited, so with those two the app keeps working at 0 remaining.
//
// Pure: headers and HTML in, facts out.

/** Was this response GitHub refusing on quota rather than on permissions? */
export function isRateLimited(status: number, headers: Headers): boolean {
  if (status !== 403 && status !== 429) return false;
  const remaining = headers.get("x-ratelimit-remaining");
  return remaining === "0" || headers.has("retry-after");
}

/** A message that tells the user what happened and what to do about it. */
export function rateLimitMessage(headers: Headers, now = Date.now()): string {
  const reset = Number(headers.get("x-ratelimit-reset") ?? 0) * 1000;
  const limit = headers.get("x-ratelimit-limit") ?? "60";
  const mins = reset > now ? Math.ceil((reset - now) / 60000) : 0;
  const when = reset > now
    ? `Resets in ${mins} minute${mins === 1 ? "" : "s"} (${
      new Date(reset).toLocaleTimeString()
    }).`
    : "It should have reset — try again.";
  return `GitHub API rate limit reached (${limit} requests/hour for anonymous access). ${when} Set GITHUB_TOKEN in the environment to raise it to 5000/hour.`;
}

/** Epoch ms when the limit resets, or 0 when the header was absent. */
export function rateLimitReset(headers: Headers): number {
  return Number(headers.get("x-ratelimit-reset") ?? 0) * 1000;
}

/** `…/releases/tag/b10144` → `b10144`. Null when the URL is not a tag page. */
export function tagFromReleaseUrl(url: string): string | null {
  const m = /\/releases\/tag\/([^/?#]+)/.exec(url);
  return m ? decodeURIComponent(m[1] as string) : null;
}

/**
 * The newest commit sha out of a branch's atom feed.
 *
 * `commits/master.atom` is the only view of master's head that GitHub serves
 * without the API, and the API is the one place this app runs out of quota.
 * Entry ids look like `tag:github.com,2008:Grit::Commit/<40 hex>`, and the feed
 * is newest-first, so the first match is the head.
 */
export function shaFromCommitsAtom(xml: string): string | null {
  const m = /Grit::Commit\/([0-9a-f]{40})|\/commit\/([0-9a-f]{40})/.exec(xml);
  return (m?.[1] ?? m?.[2]) ?? null;
}

/**
 * Asset names out of the `expanded_assets` HTML fragment.
 *
 * Matching on the download href rather than on link text: the text is wrapped
 * and truncated by GitHub's own CSS classes, while the href always carries the
 * exact file name.
 */
export function assetsFromHtml(html: string, repo: string): string[] {
  const pattern = new RegExp(
    `/${repo.replace("/", "\\/")}/releases/download/[^/"]+/([^"'\\s]+)`,
    "g",
  );
  const names = new Set<string>();
  for (const m of html.matchAll(pattern)) {
    const name = m[1];
    if (name) names.add(decodeURIComponent(name));
  }
  return [...names];
}

/** Where a named asset of a given tag lives — a stable, unauthenticated URL. */
export function assetUrl(repo: string, tag: string, name: string): string {
  return `https://github.com/${repo}/releases/download/${tag}/${name}`;
}
