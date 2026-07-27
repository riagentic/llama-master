// src/lib/ollama.ts — read an ollama model store.
//
// ollama does not keep `.gguf` files. It keeps an OCI-style store: manifests
// under `manifests/<registry>/<namespace>/<name>/<tag>` (JSON), and the actual
// weights in `blobs/sha256-<hex>` with no extension. A scanner that looks for
// `*.gguf` therefore finds nothing at all in an ollama directory, which is why
// this resolver exists: manifest in, blob path + human name out.
//
// The blob itself IS a GGUF file, so once resolved it goes through exactly the
// same header parser as every other model.
//
// Pure: strings in, a description out. The host does the reading.

/** The one layer that holds weights; the rest are templates, params, licences. */
const MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model";

export type OllamaModel = {
  /** `llama3.2:3b`, or `hf.co/user/repo:q4` for a non-library namespace. */
  name: string;
  /** `sha256-<hex>` — the file name inside `blobs/`. */
  blob: string;
  sizeB: number;
};

type Manifest = {
  layers?: { mediaType?: string; digest?: string; size?: number }[] | null;
};

/**
 * Name a model from the path of its manifest.
 *
 * `…/manifests/registry.ollama.ai/library/llama3.2/3b` → `llama3.2:3b`
 * `…/manifests/registry.ollama.ai/hf.co/user/repo/q4`  → `hf.co/user/repo:q4`
 *
 * The `library/` namespace is dropped because ollama itself hides it — a user
 * types `ollama run llama3.2:3b`, never `library/llama3.2:3b`.
 */
export function nameFromManifestPath(path: string): string | null {
  const parts = path.split("/").filter(Boolean);
  const at = parts.lastIndexOf("manifests");
  // Need at least: manifests / registry / namespace / name / tag
  if (at < 0 || parts.length - at < 5) return null;
  const after = parts.slice(at + 2); // drop "manifests" and the registry host
  const tag = after.pop() as string;
  if (after[0] === "library") after.shift();
  if (after.length === 0) return null;
  return `${after.join("/")}:${tag}`;
}

/**
 * Resolve one manifest to the blob that holds its weights.
 *
 * Returns null for a manifest with no model layer — that is the normal shape of
 * a **cloud** model (`"layers": null`), which ollama registers locally but runs
 * remotely. There is nothing on disk to load, so it must not be listed.
 */
export function resolveManifest(
  path: string,
  json: string,
): OllamaModel | null {
  let m: Manifest;
  try {
    m = JSON.parse(json) as Manifest;
  } catch {
    return null;
  }
  const layer = (m.layers ?? []).find((l) => l.mediaType === MODEL_MEDIA_TYPE);
  const digest = layer?.digest ?? "";
  if (!digest.startsWith("sha256:")) return null;

  const name = nameFromManifestPath(path);
  if (!name) return null;

  return {
    name,
    // ollama writes the digest with the colon replaced by a dash.
    blob: digest.replace(":", "-"),
    sizeB: Number(layer?.size ?? 0),
  };
}

/** Does this directory look like an ollama store? */
export function isOllamaStore(entries: readonly string[]): boolean {
  return entries.includes("manifests") && entries.includes("blobs");
}
