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
    } // A ggml assert prints as `<path>.cpp:1367: GGML_ASSERT(...) failed` —
    // no timestamp, no severity, and it IS the whole diagnosis. Missing it
    // reduced a named hard limit to "exited with code 134".
    else if (/GGML_ASSERT|ggml_abort|CUDA error/.test(l)) {
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
  steps: Step[] | ((m: RegExpExecArray) => Step[]);
};

/**
 * The allocation that failed, when llama.cpp named it: which card, how much.
 *
 * A failure on card 1 or later is PROOF the machine has more than one GPU, and
 * that changes the diagnosis completely. llama.cpp divides the offloaded layers
 * across cards by COUNT — free VRAM only chooses the proportions — so with
 * `--n-cpu-moe` holding the first N layers' experts in RAM, the layers that
 * still own their experts are many times heavier and all sit at the end, on the
 * last card. That put 34 GB of a 38 GB plan on one 24 GB card while the other
 * sat half empty, with 42 GB free across the machine (`src/lib/devsplit.ts`).
 *
 * Card 0 proves nothing — a single-GPU machine reports the same index — so it
 * gets the ordinary "out of VRAM" answer.
 */
const CUDA_ALLOC =
  /allocating\s+([\d.]+)\s*MiB on device (\d+):\s*cudaMalloc failed/i;

const RESTART_OTHER: Step = {
  text:
    "If another llama-server is still running, stop it first — it holds its VRAM until it exits.",
};

const SIGNATURES: Sig[] = [
  {
    match:
      /cudaMalloc failed: out of memory|unable to allocate CUDA\d* buffer/i,
    reason: (m) => {
      const a = CUDA_ALLOC.exec(m.input);
      if (!a || Number(a[2]) < 1) {
        return "The GPU ran out of memory while loading the model. Something else is already using the VRAM, or too many layers were offloaded.";
      }
      return `Card ${a[2]} was asked for ${
        (Number(a[1]) / 1024).toFixed(1)
      } GB in a single allocation and did not have it. With more than one card that usually means the layers are divided badly rather than that the model is too big: llama.cpp splits them by count, and a layer that keeps its experts is many times heavier than one that does not.`;
    },
    steps: (m) => {
      const a = CUDA_ALLOC.exec(m.input);
      if (a && Number(a[2]) >= 1) {
        return [
          {
            text:
              "Re-run the tuner — it sizes each card separately and writes an explicit “Tensor split” for this model. Left empty, llama.cpp divides the layers by count from free memory, which is the guess that fails here.",
            action: { kind: "open-tab", tab: "settings" },
          },
          RESTART_OTHER,
        ];
      }
      return [
        RESTART_OTHER,
        {
          text:
            "Otherwise lower “GPU layers” on the Tune tab, or shrink the context, and watch the VRAM bar: it shows what is already in use by other processes.",
          action: { kind: "open-tab", tab: "dashboard" },
        },
      ];
    },
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
    // AFTER the cudaMalloc signature (a named allocation is more specific) and
    // BEFORE the generic CUDA-error one: `CUDA error: out of memory` is not a
    // driver mismatch, it is VRAM running out at the first real forward pass —
    // CUDA allocates its compute scratch (activation-quantise buffers, cuBLAS
    // workspace, graphs) lazily, so a run can pass /health and die on the first
    // prompt. Telling that user "the build and driver do not match, use Vulkan"
    // sent a real user to a slower backend when a smaller context was the fix.
    match: /CUDA error: out of memory/i,
    reason: () =>
      "The GPU ran out of memory during generation, after loading fine: CUDA allocates compute scratch at the first real batch, and the card had nothing left to give. The plan was too tight, not wrong — a smaller context or one fewer GPU layer is the fix.",
    steps: [
      {
        text:
          "Press Start again with “Optimal automatically” on — the app steps the context down by itself when a run dies for want of memory, and remembers the size that works.",
      },
      {
        text:
          "Or lower the context or “GPU layers” by hand on the Tune tab, and watch the VRAM bar while generating.",
        action: { kind: "open-tab", tab: "dashboard" },
      },
    ],
  },
  {
    // Not memory at all: a hard compile-time cap in llama.cpp's backend
    // scheduler. With part of the model on the CPU (routed experts) and the
    // rest across GPUs, every step's graph is cut at the device boundaries,
    // and past a model-specific context one split needs more cross-device
    // inputs than GGML_SCHED_MAX_SPLIT_INPUTS allows. Measured on
    // DeepSeek-V4 on 2×24 GB: 262,144 generates, 524,288 trips this assert
    // during load — before any large allocation was attempted. Reporting it
    // as "crash inside llama.cpp, try another backend" sent the user counting
    // RAM they did not lack.
    match: /GGML_SCHED_MAX_SPLIT_INPUTS/,
    reason: () =>
      "llama.cpp hit a hard limit building the compute graph — not a memory shortage. With part of the model in system RAM and the rest on the GPUs, an extreme context needs more cross-device transfers per step than llama.cpp's scheduler supports, and it stops at an assert before allocating anything.",
    steps: [
      {
        text:
          "Lower the pinned context. The largest size that has actually generated on this machine is remembered and used as the automatic ceiling — the bands on the context control are safe picks.",
      },
      {
        text:
          "CPU only sidesteps the limit entirely (one device, no graph splits) — at CPU speed, which for an extreme context means very long prompt processing.",
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
    if (m) {
      return {
        reason: s.reason(m),
        steps: typeof s.steps === "function" ? s.steps(m) : s.steps,
      };
    }
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
