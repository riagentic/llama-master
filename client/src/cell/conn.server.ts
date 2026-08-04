// client/src/cell/conn.server.ts — the network. SERVER ONLY.
//
// Reached only by `await import()` inside a cell method: a static import of a
// `.server.ts` from browser-reachable code blanks the screen with "Deno is not
// defined" (dep/aio/docs/build/imports.md). Everything here is I/O; every
// decision it makes lives in `src/lib/`.

import type { Iface } from "../lib/discover.ts";
import { candidates } from "../lib/discover.ts";

/** This machine's IPv4 addresses. `Deno.networkInterfaces` is the whole of it —
 *  no shelling out, no parsing `ip addr`. */
export function interfaces(): Iface[] {
  try {
    return Deno.networkInterfaces()
      .filter((n) => n.family === "IPv4")
      .map((n) => ({
        address: n.address,
        prefix: n.cidr ? cidrBits(n.cidr) : 24,
      }));
  } catch {
    // No permission, or a platform that does not report them: discovery falls
    // back to localhost, which is still the commonest place to find a server.
    return [];
  }
}

function cidrBits(cidr: string): number {
  const n = Number(cidr.split("/")[1]);
  return Number.isInteger(n) ? n : 24;
}

/** One probe: is there a llama-server at this base URL?
 *
 *  `/props` rather than `/health`, because `/props` is the answer that
 *  IDENTIFIES it — anything can return 200 to /health, and a sweep that
 *  accepted that would list every web server on the subnet. */
export async function probe(
  base: string,
  timeoutMs = 700,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${base}/props`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) {
      await res.body?.cancel();
      return null;
    }
    const body = await res.json();
    // A llama-server always reports one of these. Without them it is something
    // else that happens to serve JSON at /props.
    const ok = body && typeof body === "object" &&
      ("model_path" in body || "default_generation_settings" in body ||
        "total_slots" in body);
    return ok ? body as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

/** GET a text endpoint (`/metrics`), or null when it is not enabled. */
export async function text(
  base: string,
  path: string,
  timeoutMs = 1500,
): Promise<string | null> {
  try {
    const res = await fetch(`${base}${path}`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) {
      await res.body?.cancel();
      return null;
    }
    return await res.text();
  } catch {
    return null;
  }
}

/** GET a JSON endpoint (`/slots`, `/health`), or null when it is not enabled. */
export async function json(
  base: string,
  path: string,
  timeoutMs = 1500,
): Promise<unknown> {
  try {
    const res = await fetch(`${base}${path}`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) {
      await res.body?.cancel();
      return null;
    }
    return await res.json();
  } catch {
    return null;
  }
}

/** How many probes are in the air at once.
 *
 *  254 addresses × 4 ports is 1,016 sockets; opened at once they exhaust the
 *  file-descriptor limit and the sweep fails as a whole rather than finding
 *  anything. 64 keeps a /24 sweep under three seconds on a quiet network while
 *  staying inside every default ulimit. */
const IN_FLIGHT = 64;

/**
 * Sweep for servers, newest answer first, reporting progress as it goes.
 *
 * Stops early once `stopAfter` servers have answered: the common case is one
 * llama.master on the subnet, and continuing to knock on 900 more doors after
 * finding it is time the user spends watching a spinner.
 */
export async function sweep(
  ifaces: readonly Iface[],
  onProgress: (done: number, total: number, found: number) => void,
  stopAfter = 4,
  timeoutMs = 700,
): Promise<{ url: string; props: Record<string, unknown> }[]> {
  const urls = candidates(ifaces);
  const found: { url: string; props: Record<string, unknown> }[] = [];
  let done = 0;
  let next = 0;
  const worker = async () => {
    while (next < urls.length && found.length < stopAfter) {
      const url = urls[next++];
      if (!url) break;
      const props = await probe(url, timeoutMs);
      done++;
      if (props) found.push({ url, props });
      onProgress(done, urls.length, found.length);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(IN_FLIGHT, urls.length) }, worker),
  );
  return found;
}
