// src/lib/about.ts — who made this and what it is.
//
// One definition, so the window title, the About page and `aio.run`'s version
// cannot drift apart. Data only: no imports, safe everywhere.

export const ABOUT = {
  name: "Llama.cpp Master",
  tagline: "User friendly application to master Llama.cpp",
  author: "riagentic",
  license: "MIT",
  repo: "https://github.com/riagentic/llama-master",
  /** Keep in step with `version` in deno.json — a guard test checks it. */
  version: "0.3.0",
  upstream: "https://github.com/ggml-org/llama.cpp",
} as const;
