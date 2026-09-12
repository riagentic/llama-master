#!/usr/bin/env -S deno run -A
// test/stub-llama-server.ts — a llama-server impersonator.
//
// Speaks the endpoints llama.master actually uses (`/health`, `/props`,
// `/completion` for the readiness probe, `/v1/chat/completions` with SSE) and
// accepts llama.cpp's flags. It exists so
// the server lifecycle and the chat stream can be tested against a REAL child
// process over a REAL socket — spawn, health poll, stream, SIGTERM — instead of
// against a mock that agrees with whatever the code does.
//
// `--fail-after <ms>` makes it exit non-zero, so the crash path is testable too.

const args = Deno.args;
const flag = (name: string, fallback: string): string => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] ?? fallback : fallback;
};

// `--help` must print and exit, the way a real llama-server does — the app
// probes every build's flags this way (`src/lib/caps.ts`) and then REFUSES to
// emit anything this list does not name (`command.ts:emitFor`). A stub that
// started a server instead left the probe hanging until its timeout, once per
// test that selects a build: 23 s of suite became 95 s, which is the same bug a
// wedged binary would cause in the app.
//
// The layout is copied from real `--help` output — aliases, value placeholders,
// the two-space description margin, a continuation line — because that layout
// is exactly what `parseHelpFlags` has to survive. WHAT it lists is the whole
// catalog, because this stub stands in for a CURRENT llama.cpp: since the app
// stopped emitting flags a probed build does not declare, a short illustrative
// list turned every test build into an ancient one, and the symptom is a
// setting quietly missing from the command with nothing explaining it.
//
// It cannot import the catalog — the stub is COPIED into a temporary builds
// directory, where a relative import does not resolve — so
// `tests/guards.test.ts` fails if the two ever drift.
//
// `--old-build` answers a llama.cpp from before `-bs` and `--fit` existed, so
// the "this build cannot do that" branches still have something real to run
// against.
if (args.includes("--help")) {
  const old = args.includes("--old-build");
  const RECENT = ["-bs", "--fit"];
  const lines = [
    "----- common params -----",
    "",
    "-h,    --help, --usage                  print usage and exit",
    "-m,    --model FNAME                    model path (default: unset)",
    // Aliases in the real comma-separated shape, which is the case that a
    // by-column split and a plain two-space split both get wrong.
    "-t,    --threads N                      number of CPU threads to use during generation (default: -1)",
    "-ngl,  --gpu-layers, --n-gpu-layers N   number of layers to store in VRAM",
    "--cpu-strict <0|1>                      use strict CPU placement (default: 0)",
    "-ngl N                                  gpu layers",
    "--n-cpu-moe N                           moe layers on cpu",
    "--fit TYPE                              llama.cpp auto-fit",
    "-sm TYPE                                split mode",
    "-ts FNAME                               tensor split",
    "-mg N                                   main gpu",
    "-dev <dev1,dev2,..>                     gpus to use",
    "-nkvo                                   keep kv cache on cpu",
    "--spec-type TYPE                        speculative decoding",
    "--lazy-mode TYPE                        lazy tensor reads",
    "-md FNAME                               draft model",
    "--spec-draft-n-max N                    draft tokens (max)",
    "--spec-draft-n-min N                    draft tokens (min)",
    "-bs                                     sample on the gpu",
    "-ot FNAME                               tensor override",
    "-c N                                    context size",
    "-b N                                    batch size",
    "-ub N                                   micro-batch size",
    "-np N                                   parallel slots",
    "--context-shift                         context shift",
    "--no-context-shift                      context shift",
    "--cache-reuse N                         cache reuse",
    "--keep N                                keep tokens",
    "-t N                                    threads",
    "-tb N                                   threads (batch)",
    "-fa TYPE                                flash attention",
    "-ctk TYPE                               kv cache type (k)",
    "-ctv TYPE                               kv cache type (v)",
    "--load-mode TYPE                        model loading",
    "--numa TYPE                             numa policy",
    "--rope-scaling TYPE                     rope scaling",
    "--rope-freq-base N                      rope freq base",
    "--rope-freq-scale N                     rope freq scale",
    "--temp N                                temperature",
    "--top-k N                               top-k",
    "--top-p N                               top-p",
    "--min-p N                               min-p",
    "--repeat-penalty N                      repeat penalty",
    "--repeat-last-n N                       repeat window",
    "-s N                                    seed",
    "--host FNAME                            host",
    "--port N                                port",
    "-a FNAME                                model alias",
    "--api-key FNAME                         api key",
    "--jinja                                 jinja templates",
    "--no-jinja                              jinja templates",
    "--reasoning TYPE                        reasoning",
    "--reasoning-budget N                    reasoning budget",
    "--chat-template FNAME                   chat template",
    "--cont-batching                         continuous batching",
    "--no-cont-batching                      continuous batching",
    "--metrics                               prometheus metrics",
    "--slots                                 expose slots",
    "--no-slots                              expose slots",
    "--no-webui                              disable web ui",
    "-to N                                   read timeout",
    "-v                                      verbose log",
    "-p FNAME                                prompt",
    "-n N                                    tokens to predict",
    "-cnv                                    conversation mode",
  ];
  console.log(
    lines.filter((l) => !old || !RECENT.some((f) => l.startsWith(f + " ")))
      // One continuation line, indented, exactly as llama.cpp prints its
      // `(env: …)` notes — the parser must not read it as another option.
      .flatMap((l) =>
        /^--?[A-Za-z]/.test(l) ? [l, " ".repeat(40) + "(env: …)"] : [l]
      )
      .join("\n"),
  );
  Deno.exit(0);
}

const port = Number(flag("--port", "8080"));
const model = flag("-m", "");
const ctx = Number(flag("-c", "4096"));
const readyAfter = Number(flag("--ready-after", "0"));
const failAfter = Number(flag("--fail-after", "0"));

// `--print-env NAME` makes the stub report one environment variable on stdout
// and exit — the app's env-var feature (`src/lib/envvars.ts`) is about the
// process ENVIRONMENT, which no HTTP endpoint can observe, so the test has to
// read it the way the running process itself would.
if (args.includes("--print-env")) {
  for (const name of args.slice(args.indexOf("--print-env") + 1)) {
    if (name.startsWith("-")) break;
    console.log(`${name}=${Deno.env.get(name) ?? ""}`);
  }
  Deno.exit(0);
}

console.log(`build: 9999 (stub)`);
console.log(`llama_model_loader: loading model from ${model}`);

// `--oom` reproduces the failure the user actually hit: another process is
// holding the VRAM, so the allocation fails and llama.cpp exits 1 after saying
// exactly why on stderr. The wording is verbatim from a real run.
if (args.includes("--oom")) {
  console.error(
    "0.01.200.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating " +
      "2406.98 MiB on device 0: cudaMalloc failed: out of memory",
  );
  console.error(
    "0.01.300.000 E llama_model_load: error loading model: unable to " +
      "allocate CUDA0 buffer",
  );
  Deno.exit(1);
}

if (failAfter > 0) {
  setTimeout(() => {
    console.error("stub: simulated crash");
    Deno.exit(3);
  }, failAfter);
}

const startedAt = Date.now();
const ready = () => Date.now() - startedAt >= readyAfter;

const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`;

// `async`: the completion branch reads `n_predict` off the body so a bench
// gets back the number of tokens it asked for, the way llama.cpp does.
Deno.serve({ port, hostname: "127.0.0.1", onListen: () => {} }, async (req) => {
  const url = new URL(req.url);

  if (url.pathname === "/health") {
    return ready()
      ? Response.json({ status: "ok" })
      : Response.json({ status: "loading model" }, { status: 503 });
  }

  // The generation probe (`srv.server.ts:probe`). `--oom-on-generate`
  // reproduces the failure that motivated it, with the stderr captured from a
  // real DeepSeek-V4 run on 2×24 GB: the server loads, passes /health, and dies
  // of a pool allocation the moment it is asked to produce a token. Note the
  // word "buffer" appears nowhere — that is the point; the fit ladder must
  // recognise this shape without it.
  if (url.pathname === "/completion") {
    if (!ready()) {
      return Response.json({ error: "loading" }, { status: 503 });
    }
    // `--slow-generate` is the OTHER thing a probe can meet: a process that is
    // alive and simply has not finished. A cold start runs against a page
    // cache that is still filling — measured 23x slower prompt processing on
    // the first reply — and treating that as death left a working server
    // "starting" forever.
    if (args.includes("--slow-generate")) {
      return new Promise<Response>(() => {});
    }
    if (args.includes("--oom-on-generate")) {
      console.error(
        "/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
      );
      console.error("2.17.177.475 E CUDA error: out of memory");
      console.error(
        "2.17.177.483 E   current device: 0, in function ggml_cuda_kernel_can_use_pdl at /src/ggml-cuda/common.cuh:1622",
      );
      console.error("2.17.177.484 E   cudaFuncGetAttributes(&attr, kernel)");
      // SIGABRT's exit status, as ggml_abort produces. The handler never
      // answers — the process dies under the request and the connection drops,
      // exactly as a real abort mid-generation looks to the probe.
      setTimeout(() => Deno.exit(134), 10);
      return new Promise<Response>(() => {});
    }
    // llama.cpp reports its own `timings` on every completion, and that is
    // what the speed bench reads (`src/lib/bench.ts`). `--no-timings` is the
    // OTHER thing a bench can meet: a build old enough not to report them,
    // which has measured nothing — and must be reported as nothing rather than
    // as a machine running at 0 tokens a second.
    const asked = await req.json().catch(() => ({})) as {
      n_predict?: number;
      prompt?: string;
    };
    const predict = Number(asked.n_predict ?? 2) || 2;
    // `--drafting` models the ONE behaviour the bench pair exists to see: a
    // drafter is paid only for the tokens the full model then accepts, so it
    // roughly doubles the rate on output that repeats itself and does close to
    // nothing on prose. A stub that answered the same number to both prompts
    // would let a broken pairing (both slots fed the same prompt, or the code
    // result overwriting the prose one) pass every assertion.
    const codey = /TypeScript|JSDoc/i.test(String(asked.prompt ?? ""));
    const rate = args.includes("--drafting") && codey ? 74.5 : 37.25;
    return Response.json({
      content: " ok",
      tokens_predicted: predict,
      ...(args.includes("--no-timings") ? {} : {
        timings: {
          prompt_n: 42,
          prompt_per_second: 512.5,
          predicted_n: predict,
          predicted_per_second: rate,
        },
      }),
    });
  }

  if (url.pathname === "/props") {
    return Response.json({
      model_path: model,
      n_ctx: ctx,
      chat_template: "{{ messages }}",
    });
  }

  if (url.pathname === "/v1/chat/completions") {
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        const enc = new TextEncoder();
        // Reasoning first, exactly as llama.cpp streams a thinking model:
        // `reasoning_content` deltas with `content` absent. A client that
        // only reads `content` renders nothing for this whole phase.
        for (const word of ["Consider", " the greeting."]) {
          c.enqueue(
            enc.encode(
              sse({ choices: [{ delta: { reasoning_content: word } }] }),
            ),
          );
        }
        for (const word of ["Hello", " from", " the", " stub"]) {
          c.enqueue(
            enc.encode(sse({ choices: [{ delta: { content: word } }] })),
          );
        }
        c.enqueue(
          enc.encode(
            sse({
              choices: [{ delta: {}, finish_reason: "stop" }],
              timings: { predicted_per_second: 42.5 },
            }),
          ),
        );
        c.enqueue(enc.encode("data: [DONE]\n\n"));
        c.close();
      },
    });
    return new Response(body, {
      headers: { "content-type": "text/event-stream" },
    });
  }

  return new Response("not found", { status: 404 });
});

console.log(`main: server is listening on http://127.0.0.1:${port} - starting`);
