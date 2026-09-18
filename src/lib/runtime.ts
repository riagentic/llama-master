// src/lib/runtime.ts — which llama.cpp can run THIS model file correctly.
//
// Almost every GGUF runs on any recent llama.cpp, and the app never had to
// ask. PrismML's ternary Bonsai models broke that in the worst of both ways:
//
//   - a `PQ2_0`/`PTQ1_0` tensor is a ggml type upstream has never heard of,
//     so stock llama.cpp refuses the file — a failed start, at least loud;
//   - a Hadamard-folded file (`prism.hadamard.*` keys) of an UPSTREAM type
//     (`Q2_0`) loads fine on stock llama.cpp and then answers in garbage,
//     because nothing un-rotates the weights. Nothing fails. The user sees a
//     model that "does not work" and blames the model.
//
// The GGUF reader names the runtime a file needs (`ModelMeta.vendor`, set in
// `rust/src/gguf.rs`); this decides whether the chosen build provides it, and
// what to do when it does not. Pure: strings in, a diagnosis out.

import type { Diagnosis } from "./diagnose.ts";
import { formatRef, parseRef } from "./srcref.ts";

type Vendor = {
  /** What to call the runtime on screen. */
  label: string;
  /** GitHub owner of the fork, lower-cased for comparison. */
  owner: string;
  /** The ref that builds it (`src/lib/srcref.ts`). */
  ref: string;
  docs: string;
};

export const VENDORS: Readonly<Record<string, Vendor>> = {
  prism: {
    label: "PrismML's llama.cpp",
    owner: "prismml-eng",
    ref: formatRef({ kind: "fork", repo: "PrismML-Eng/llama.cpp", ref: null }),
    docs: "https://github.com/PrismML-Eng/Bonsai-demo",
  },
};

/**
 * What a build is, as far as a vendor's model is concerned.
 *
 * `upstream` covers master, tags, and PRs/stacks on master: none of them
 * carries a vendor's private kernels. A fork by the vendor's own owner is that
 * vendor. Any OTHER fork is `unknown` — somebody may well have merged the
 * kernels into it, and refusing a build the app cannot see into would be a
 * guess dressed as a fact.
 */
export function buildVendor(ref: string): string | "upstream" | "unknown" {
  const r = parseRef(ref);
  if (r.kind !== "fork") return "upstream";
  const owner = r.repo.split("/")[0]!.toLowerCase();
  for (const [id, v] of Object.entries(VENDORS)) {
    if (v.owner === owner) return id;
  }
  return "unknown";
}

/**
 * Why this build cannot run this model — or null when it can (or cannot be
 * known). `installed` lets the answer be "switch to the build you already
 * have" rather than "build it", which is a click instead of three minutes.
 */
export function runtimeMismatch(
  vendor: string | undefined,
  buildRef: string,
  installed: readonly { id: string; ref: string }[],
): Diagnosis | null {
  if (!vendor) return null;
  const need = VENDORS[vendor];
  // A vendor this app does not know yet: nothing honest to say.
  if (!need) return null;
  const have = buildVendor(buildRef);
  if (have === vendor || have === "unknown") return null;
  const ready = installed.find((b) => buildVendor(b.ref) === vendor);
  return {
    reason:
      `This model only runs correctly on ${need.label}. Upstream llama.cpp either refuses the file or loads it and answers in garbage (its weights are stored rotated, and only ${need.label} un-rotates them).`,
    steps: [
      ready
        ? {
          text: `Switch to the build you already have (${ready.id}).`,
          action: { kind: "use-build", id: ready.id },
        }
        : {
          text:
            `Build ${need.label} from source — about three minutes on a workstation, and it is kept beside your other builds.`,
          action: { kind: "use-ref", ref: need.ref },
        },
      {
        text: "The vendor's own notes on running these files.",
        action: { kind: "open-url", url: need.docs },
      },
    ],
  };
}
