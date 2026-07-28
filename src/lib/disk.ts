// src/lib/disk.ts — how much room is left, and where.
//
// This app writes gigabytes: a llama.cpp source tree, a cmake build directory,
// prebuilt release archives, and whatever models the user keeps. "No space left
// on device" is already one of the build failures `diagnose.ts` explains — this
// is the number that lets the app say it BEFORE the build, and the third pool
// (with RAM and VRAM) the Memory page is meant to cover.
//
// Pure: `df` output in, filesystems out. The parsing is the part worth testing,
// so it lives away from the process call.

/** One filesystem, as far as this app cares. */
export type Disk = {
  /** Device or source, e.g. `/dev/nvme0n1p4`. */
  filesystem: string;
  /** Where it is mounted, e.g. `/home`. */
  mount: string;
  totalB: number;
  usedB: number;
  /** What a non-root user can actually write — not `total - used`, which counts
   *  the reserved blocks only root may touch. */
  availB: number;
};

/**
 * Parse `df -kP` (POSIX output format, 1024-byte blocks).
 *
 * `-P` is what makes this parseable: without it a long device name wraps onto
 * its own line and the columns stop lining up. Rows are deduplicated by mount
 * point, because asking about several paths on one filesystem is the normal case
 * here (builds and models usually share a disk) and listing it twice would
 * double the apparent capacity.
 */
export function parseDf(out: string): Disk[] {
  const disks: Disk[] = [];
  const seen = new Set<string>();
  for (const line of out.split("\n").slice(1)) {
    const t = line.trim();
    if (!t) continue;
    // filesystem blocks used available capacity mount — and a mount point may
    // contain spaces, so it is everything left after the five fixed columns.
    const m = /^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%\s+(.+)$/.exec(t);
    if (!m) continue;
    const mount = (m[6] ?? "").trim();
    if (!mount || seen.has(mount)) continue;
    seen.add(mount);
    disks.push({
      filesystem: m[1] ?? "",
      mount,
      totalB: Number(m[2]) * 1024,
      usedB: Number(m[3]) * 1024,
      availB: Number(m[4]) * 1024,
    });
  }
  return disks;
}

/**
 * Is this filesystem too full to attempt a source build?
 *
 * A llama.cpp checkout plus a CUDA build tree runs to several GB, and running
 * out part-way wastes the minutes already spent. The threshold is deliberately
 * generous: the point is to warn before starting, not to refuse.
 */
export const BUILD_NEEDS_B = 8 * 1024 ** 3;

export function tooFullToBuild(d: Disk | null): boolean {
  return d !== null && d.availB < BUILD_NEEDS_B;
}
