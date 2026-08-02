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

const port = Number(flag("--port", "8080"));
const model = flag("-m", "");
const ctx = Number(flag("-c", "4096"));
const readyAfter = Number(flag("--ready-after", "0"));
const failAfter = Number(flag("--fail-after", "0"));

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

Deno.serve({ port, hostname: "127.0.0.1", onListen: () => {} }, (req) => {
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
    return Response.json({ content: " ok", tokens_predicted: 2 });
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
