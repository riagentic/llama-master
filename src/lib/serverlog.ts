// src/lib/serverlog.ts — why did llama-server stop?
//
// "exited with code 1" is not an answer. llama.cpp always says what went wrong
// on stderr before it exits, and the app already captures every line — so the
// reason is sitting in memory while the UI shows a number. This file turns the
// pair (exit code, captured log) into the sentence the user needed.
//
// Exit codes carry real information too, but mostly for signals: 137 is the
// OOM killer, 139 is a segfault, and neither will have written a tidy error
// line before dying. Those get explained from the code alone.
//
// Pure: a code and some lines in, an explanation out.

import type { Diagnosis, Step } from "./diagnose.ts";

/** llama.cpp prefixes severity after the timestamp: `0.01.234.567 E msg`. */
const ERROR_LINE = /^\s*[\d.]+\s+E\s+(.*)$/;

/** The error lines llama.cpp printed, most recent last, comments stripped. */
export function extractErrors(lines: readonly string[]): string[] {
  const out: string[] = [];
  for (const l of lines) {
    const m = ERROR_LINE.exec(l);
    if (m?.[1]) out.push(m[1].trim());
    // Some builds and the loader print bare messages with no severity column.
    else if (
      /^(error|fatal|terminate called|Segmentation fault)/i.test(l.trim())
    ) {
      out.push(l.trim());
    }
  }
  return out;
}

/** What a POSIX exit status means when the process died on a signal. */
export function signalOf(code: number): string | null {
  if (code < 128 || code > 192) return null;
  const sig = code - 128;
  const names: Record<number, string> = {
    2: "SIGINT (interrupted)",
    4: "SIGILL (illegal instruction)",
    6: "SIGABRT (aborted)",
    8: "SIGFPE (arithmetic error)",
    9: "SIGKILL (killed outright)",
    11: "SIGSEGV (segmentation fault)",
    15: "SIGTERM (asked to stop)",
  };
  return names[sig] ?? `signal ${sig}`;
}

type Sig = {
  match: RegExp;
  reason: (m: RegExpExecArray) => string;
  steps: Step[];
};

const SIGNATURES: Sig[] = [
  {
    match:
      /cudaMalloc failed: out of memory|unable to allocate CUDA\d* buffer/i,
    reason: () =>
      "The GPU ran out of memory while loading the model. Something else is already using the VRAM, or too many layers were offloaded.",
    steps: [
      {
        text:
          "If another llama-server is still running, stop it first — it holds its VRAM until it exits.",
      },
      {
        text:
          "Otherwise lower “GPU layers” on the Tune tab, or shrink the context, and watch the VRAM bar: it shows what is already in use by other processes.",
        action: { kind: "open-tab", tab: "dashboard" },
      },
    ],
  },
  {
    match:
      /failed to allocate.*buffer|ggml_backend_alloc|cannot allocate memory/i,
    reason: () =>
      "The machine ran out of memory while loading the model — system RAM this time, not VRAM.",
    steps: [
      {
        text:
          "Turn off --mlock, lower the context, or pick a smaller quantisation. The RAM bar on the Tune tab predicts this before you start.",
      },
    ],
  },
  {
    match: /bind.*(Address already in use|EADDRINUSE)|couldn't bind/i,
    reason: () =>
      "The port is already taken — most likely by another llama-server that is still running.",
    steps: [
      { text: "Stop the other server, or change the port on the Tune tab." },
    ],
  },
  {
    match: /failed to load model|error loading model|no such file/i,
    reason: (m) =>
      /no such file/i.test(m[0])
        ? "The model file is not where it was expected — it may have been moved or deleted since the last scan."
        : "llama.cpp could not load this model file. The lines above name the reason; a truncated download or an unsupported quantisation are the usual ones.",
    steps: [
      {
        text: "Re-scan for models so the list matches what is on disk.",
        action: { kind: "open-tab", tab: "build" },
      },
    ],
  },
  {
    match: /unknown argument|invalid argument|error while handling argument/i,
    reason: (m) =>
      `This build of llama-server rejected one of the flags: ${
        m[0].slice(0, 80)
      }. Older builds do not have every option.`,
    steps: [
      {
        text:
          "Reset that setting to its default on the Tune tab, or update to a newer llama.cpp build.",
      },
    ],
  },
  {
    match: /CUDA error|no CUDA-capable device|forward compatibility/i,
    reason: () =>
      "The CUDA runtime rejected the device. The build and the installed driver do not match.",
    steps: [
      {
        text:
          "Use the Vulkan build, which does not depend on a CUDA toolkit version.",
      },
      { text: "Or rebuild CUDA from source so it targets this driver." },
    ],
  },
];

/**
 * Explain why the server stopped.
 *
 * `code` is the process exit status; `lines` is everything it printed. The log
 * wins over the code whenever it says something specific, because the code is
 * almost always a bare 1.
 */
export function diagnoseServerExit(
  code: number | null,
  lines: readonly string[],
): Diagnosis {
  const errors = extractErrors(lines);
  const haystack = errors.join("\n");

  for (const s of SIGNATURES) {
    const m = s.match.exec(haystack);
    if (m) return { reason: s.reason(m), steps: s.steps };
  }

  const sig = code === null ? null : signalOf(code);
  if (sig) {
    if (code === 137) {
      return {
        reason:
          "The server was killed by the system (SIGKILL) — on Linux this is nearly always the OOM killer reclaiming memory.",
        steps: [
          {
            text:
              "Lower the context or turn off --mlock so the model needs less RAM, then start again.",
          },
        ],
      };
    }
    if (code === 143 || code === 130) {
      return {
        reason:
          `The server stopped on ${sig} — that is a normal shutdown, not a crash.`,
        steps: [],
      };
    }
    return {
      reason:
        `The server died on ${sig}. That is a crash inside llama.cpp rather than a configuration problem.`,
      steps: [
        {
          text:
            "A different backend usually avoids it — try the prebuilt CPU or Vulkan build to confirm the model itself is fine.",
          action: { kind: "open-tab", tab: "build" },
        },
        {
          text:
            "If it is reproducible, the log below is what llama.cpp's maintainers would need.",
          action: {
            kind: "open-url",
            url: "https://github.com/ggml-org/llama.cpp/issues",
          },
        },
      ],
    };
  }

  // Nothing matched: still better than a number — quote what it actually said.
  if (errors.length > 0) {
    return {
      reason: `llama-server stopped: ${errors[errors.length - 1]}`,
      steps: [
        { text: "The full output is in the log below." },
      ],
    };
  }
  return {
    reason: code === 0
      ? "llama-server exited without an error."
      : `llama-server exited with code ${code} and printed nothing useful before it did.`,
    steps: [
      {
        text:
          "Try the prebuilt CPU build with the same model — if that works, the backend is the problem, not the model.",
        action: { kind: "open-tab", tab: "build" },
      },
    ],
  };
}
