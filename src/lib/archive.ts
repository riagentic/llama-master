// src/lib/archive.ts — tar.gz and zip readers, in ~200 lines and zero
// dependencies.
//
// Why not shell out: the app promises to work on a bare OS. `tar` exists almost
// everywhere and `unzip` does not, and neither is guaranteed on Windows. Both
// formats are simple enough that reading them here is smaller than the code
// needed to detect, locate and error-handle an external binary — and it is
// pure, so it is unit-testable against fixtures we build in the test itself.
//
// Only the two features real release archives use are implemented: tar with GNU
// long names, and zip with store/deflate entries. Anything else fails loudly
// rather than silently extracting a partial tree.

export type Entry = {
  /** Path inside the archive, `/`-separated, never absolute. */
  name: string;
  bytes: Uint8Array;
  /** Unix permission bits, when the archive records them (tar always does). */
  mode: number;
  /** Set for a symlink: the target it points at, relative to its own directory.
   *
   *  Not optional detail — llama.cpp's Linux release ships every shared object
   *  twice, as `libllama.so.0.0.10144` plus a `libllama.so.0` SONAME symlink,
   *  and the loader resolves the SONAME. Drop the links and every binary in the
   *  archive fails with "cannot open shared object file". */
  link?: string;
};

const dec = new TextDecoder();
const EMPTY = new Uint8Array(0);

function readAll(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  return new Response(stream).bytes();
}

/** gzip → raw bytes, using the runtime's own inflate. */
export async function gunzip(gz: Uint8Array): Promise<Uint8Array> {
  const src = new Blob([gz as BlobPart]).stream();
  return await readAll(src.pipeThrough(new DecompressionStream("gzip")));
}

// ── tar ────────────────────────────────────────────────────────────────────

function octal(b: Uint8Array): number {
  const s = dec.decode(b).replace(/\0.*$/, "").trim();
  return s === "" ? 0 : parseInt(s, 8) || 0;
}

function cstr(b: Uint8Array): string {
  return dec.decode(b).replace(/\0.*$/, "");
}

/**
 * Read a (already decompressed) tar archive.
 *
 * Files and symlinks are returned; directories and pax metadata are skipped
 * (directories are implied by the paths, and pax records carry no content a
 * caller needs).
 */
export function untar(tar: Uint8Array): Entry[] {
  const out: Entry[] = [];
  let p = 0;
  let longName: string | null = null;

  while (p + 512 <= tar.length) {
    const head = tar.subarray(p, p + 512);
    // Two consecutive zero blocks end the archive; one is enough to stop.
    if (head.every((b) => b === 0)) break;
    p += 512;

    const name = longName ?? cstr(head.subarray(0, 100));
    const prefix = cstr(head.subarray(345, 500));
    const size = octal(head.subarray(124, 136));
    const mode = octal(head.subarray(100, 108));
    const type = String.fromCharCode(head[156] ?? 0);
    const linkname = cstr(head.subarray(157, 257));
    longName = null;

    const dataEnd = p + size;
    if (dataEnd > tar.length) {
      throw new Error(`tar: entry "${name}" runs past the end of the archive`);
    }
    const data = tar.subarray(p, dataEnd);
    p = dataEnd + ((512 - (size % 512)) % 512);

    if (type === "L") {
      // GNU long name: the payload IS the next entry's name.
      longName = cstr(data);
      continue;
    }
    const full = prefix ? `${prefix}/${name}` : name;
    if (type === "0" || type === "\0") {
      out.push({ name: full, bytes: data.slice(), mode: mode || 0o644 });
    } else if (type === "2" && linkname) {
      out.push({
        name: full,
        bytes: EMPTY,
        mode: mode || 0o777,
        link: linkname,
      });
    }
    // '5' dir, '1' hardlink, 'x'/'g' pax — intentionally skipped.
  }
  return out;
}

export async function untargz(gz: Uint8Array): Promise<Entry[]> {
  return untar(await gunzip(gz));
}

// ── zip ────────────────────────────────────────────────────────────────────

const EOCD = 0x06054b50;
const CEN = 0x02014b50;

function u16(b: Uint8Array, o: number): number {
  return (b[o] ?? 0) | ((b[o + 1] ?? 0) << 8);
}

function u32(b: Uint8Array, o: number): number {
  return (
    ((b[o] ?? 0) |
      ((b[o + 1] ?? 0) << 8) |
      ((b[o + 2] ?? 0) << 16) |
      ((b[o + 3] ?? 0) << 24)) >>>
    0
  );
}

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const src = new Blob([data as BlobPart]).stream();
  return await readAll(src.pipeThrough(new DecompressionStream("deflate-raw")));
}

/**
 * Read a zip archive from its central directory (the only correct way — the
 * local headers may carry zeroed sizes when the writer streamed the file).
 */
export async function unzip(zip: Uint8Array): Promise<Entry[]> {
  // The EOCD lives in the last 64 KB, after a comment of unknown length.
  let eocd = -1;
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 65557); i--) {
    if (u32(zip, i) === EOCD) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("zip: no end-of-central-directory record");

  const count = u16(zip, eocd + 10);
  const cenOff = u32(zip, eocd + 16);
  if (cenOff === 0xffffffff || count === 0xffff) {
    throw new Error("zip: zip64 archives are not supported");
  }

  const out: Entry[] = [];
  let p = cenOff;
  for (let i = 0; i < count; i++) {
    if (u32(zip, p) !== CEN) throw new Error("zip: corrupt central directory");
    const method = u16(zip, p + 10);
    const compSize = u32(zip, p + 20);
    const nameLen = u16(zip, p + 28);
    const extraLen = u16(zip, p + 30);
    const commentLen = u16(zip, p + 32);
    const external = u32(zip, p + 38);
    const localOff = u32(zip, p + 42);
    const name = dec.decode(zip.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;

    if (name.endsWith("/")) continue; // directory entry

    // Re-read the local header: its name/extra lengths are authoritative for
    // where the data actually starts.
    const lNameLen = u16(zip, localOff + 26);
    const lExtraLen = u16(zip, localOff + 28);
    const start = localOff + 30 + lNameLen + lExtraLen;
    const raw = zip.subarray(start, start + compSize);

    const bytes = method === 0
      ? raw.slice()
      : method === 8
      ? await inflateRaw(raw)
      : (() => {
        throw new Error(
          `zip: unsupported compression method ${method} for ${name}`,
        );
      })();

    // Unix mode lives in the high 16 bits of the external attributes; the file
    // type is in the top four bits of that (S_IFLNK = 0xA000).
    const unix = external >>> 16;
    const mode = unix & 0o7777;
    if ((unix & 0xf000) === 0xa000) {
      out.push({ name, bytes: EMPTY, mode: 0o777, link: dec.decode(bytes) });
    } else {
      out.push({ name, bytes, mode: mode || 0o644 });
    }
  }
  return out;
}

/** Drop the single top-level directory release archives always wrap things in
 *  (`llama.cpp-b6234/…` → `…`). A no-op when there is more than one root. */
export function stripRoot(entries: Entry[]): Entry[] {
  const roots = new Set(entries.map((e) => e.name.split("/")[0] ?? ""));
  if (roots.size !== 1) return entries;
  const root = `${[...roots][0]}/`;
  return entries
    .filter((e) => e.name.startsWith(root))
    .map((e) => ({ ...e, name: e.name.slice(root.length) }));
}

/** Does this path stay inside the destination directory? */
function contained(p: string): boolean {
  if (p === "" || p.startsWith("/") || /^[A-Za-z]:/.test(p)) return false;
  return !p.split("/").includes("..");
}

/** Reject entries that would escape the destination directory (zip-slip), and
 *  symlinks that point outside it (the same attack, one indirection later). */
export function safeEntries(entries: Entry[]): Entry[] {
  return entries.filter((e) => {
    if (!contained(e.name)) return false;
    if (e.link !== undefined && !contained(e.link)) return false;
    return true;
  });
}
