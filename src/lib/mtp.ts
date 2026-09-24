// src/lib/mtp.ts — find a multi-token-prediction drafter that ships beside the
// model rather than inside it.
//
// `tune.ts` switches speculative decoding on whenever `meta.nextnLayers > 0`,
// which reads the MTP block count out of the model's own header. That is right
// for every model that carries the block in the same file, and it silently
// misses the ones that do not: Gemma 4's heads are published as a SEPARATE
// GGUF next to the weights (`…-mtp-Q4_0.gguf`), and llama.cpp is told about it
// with a second model path, not with a header key.
//
// The consequence was not a crash — it was worse than a crash, because nothing
// said anything. The main model's header reports zero MTP blocks, the tuner
// concluded "this model ships no multi-token-prediction block, so there is
// nothing to draft with for free", and wrote that sentence on screen while the
// drafter sat unopened in the same directory. A 2–3x speed-up, declined on the
// strength of a fact that was true about the file and false about the model.
//
// Pure: file names in, a candidate out. No I/O — the caller already has the
// scan, and a function that reads the disk could not be unit-tested against the
// twelve naming shapes below.

/** A drafter found next to a model. */
export type MtpSibling = {
  /** Absolute path to the drafter GGUF, for `-md`. */
  path: string;
  /** Its size on disk. The memory plan does not bill a draft model, so this is
   *  the number the caller has to talk about before recommending it. */
  sizeB: number;
  /** The file name, for saying which file was found. */
  file: string;
};

/** One entry of the model scan, as much of it as this needs. */
type Candidate = { path: string; file: string; sizeB: number; dir: string };

/**
 * Is this file name marked as an MTP drafter?
 *
 * Matched as a whole token, never as a substring: `mtp` inside a word is a
 * coincidence, and a model called `Mtptune-7B` is not a drafter. The separators
 * are the three that appear in published GGUF names, plus the start and end of
 * the stem.
 */
export function isMtpName(file: string): boolean {
  return /(^|[-_. ])mtp([-_. ]|$)/i.test(stem(file));
}

/** The file name with its `.gguf` and any shard suffix removed. */
function stem(file: string): string {
  return file
    .replace(/\.gguf$/i, "")
    .replace(/-\d{5}-of-\d{5}$/i, "");
}

/**
 * The stem with the `mtp` token and every quantisation label taken out, folded
 * to a comparable key.
 *
 * Two files belong together when what is LEFT after that matches. The
 * quantisations have to go because a drafter is routinely published at a
 * different one from the model it drafts for (`…-Q8_0.gguf` beside
 * `…-mtp-Q4_0.gguf`), and requiring them to agree would reject exactly the
 * pairing that is being looked for.
 */
export function pairKey(file: string): string {
  return stem(file)
    .replace(/(^|[-_. ])mtp([-_. ]|$)/i, "$1")
    // Every family of label: legacy/K quants (`Q4_K_M`), i-quants (`IQ4_XS`,
    // `IQ3_XXS`) and ternaries (`TQ1_0`), imatrix tags (`i1`), Unsloth's
    // dynamic marker (`UD-Q4_K_XL`), and the float and FP4 types. Missing one
    // is not an edge case: Unsloth publishes almost everything as `UD-…`, and
    // a model at `IQ4_XS` beside a drafter at `Q8_0` is the ordinary pairing.
    .replace(
      /[-_. ](?:[it]?q\d+(?:_[a-z0-9]+)*|i\d+|ud|f16|bf16|f32|fp16|fp8|mxfp4|nvfp4)(?=[-_. ]|$)/gi,
      "",
    )
    .replace(/[-_. ]+/g, "")
    .toLowerCase();
}

/**
 * Find the MTP drafter published beside `modelPath`, or `null`.
 *
 * Deliberately strict, in both directions:
 *
 * - Same directory only. A drafter two folders away, matched on name, is a
 *   guess about somebody's filing rather than a fact about a download, and
 *   handing llama.cpp the wrong second model produces drafts that are never
 *   accepted — i.e. a silent slowdown, which is the worst failure available
 *   here.
 * - The pair key must match. "Any file with mtp in the name" would pair a
 *   Gemma drafter with a Qwen model on a machine that has both, and llama.cpp
 *   will load that pair: the vocabularies differ, so every draft is rejected.
 * - A model that is ITSELF a drafter gets nothing, so the drafter never drafts
 *   for itself.
 * - Only shard 1 of a split drafter is offered, because that is the path
 *   llama.cpp is given.
 */
export function findMtpSibling(
  modelPath: string,
  models: readonly Candidate[],
): MtpSibling | null {
  const me = models.find((m) => m.path === modelPath);
  if (!me) return null;
  if (isMtpName(me.file)) return null;
  const key = pairKey(me.file);
  if (!key) return null;
  const hit = models.find((m) =>
    m.path !== modelPath &&
    m.dir === me.dir &&
    isMtpName(m.file) &&
    pairKey(m.file) === key &&
    !/-(?!00001)\d{5}-of-\d{5}\.gguf$/i.test(m.file)
  );
  return hit ? { path: hit.path, sizeB: hit.sizeB, file: hit.file } : null;
}
