// client/tests/stub.ts — a llama-server that is not one.
//
// The client's whole contract with the far end is four endpoints and an SSE
// stream, so the tests bring their own: a real HTTP server on a real port,
// answering the way llama.cpp answers. Nothing here mocks `fetch` — the code
// under test does its own networking, and a test that stubbed that away would
// prove the parser works and nothing else.

export type StubOptions = {
  /** Serve `/metrics` (llama.cpp only does with `--metrics`). */
  metrics?: string;
  /** Serve `/slots` (only with `--slots`). */
  slots?: unknown;
  /** What `/health` says: "ok" once loaded, "loading model" before. */
  health?: string;
  /** The reply to stream, split into as many SSE events as it has entries. */
  reply?: string[];
  /** Tokens/second reported in the final event's `timings`. */
  tps?: number;
  model?: string;
};

export type Stub = {
  url: string;
  port: number;
  /** Every path the client asked for, in order — so a test can assert that
   *  `/metrics` was tried before `/slots` rather than only that the answer
   *  came out right. */
  hits: string[];
  close: () => Promise<void>;
};

/** A free port, taken by binding to 0 and letting go. A hand-picked port
 *  eventually collides with another test file and flakes the suite. */
export function freePort(): number {
  const l = Deno.listen({ port: 0 });
  const port = (l.addr as Deno.NetAddr).port;
  l.close();
  return port;
}

export function stubServer(opts: StubOptions = {}): Stub {
  const port = freePort();
  const hits: string[] = [];
  const ac = new AbortController();
  const model = opts.model ?? "/models/stub-7B-Q4_K_M.gguf";
  const reply = opts.reply ?? ["Hello", " from", " the stub"];

  const server = Deno.serve(
    { port, signal: ac.signal, onListen: () => {} },
    async (req) => {
      const path = new URL(req.url).pathname;
      hits.push(path);
      if (path === "/props") {
        return Response.json({
          model_path: model,
          total_slots: 2,
          chat_template: "{{ }}",
          default_generation_settings: { n_ctx: 8192 },
        });
      }
      if (path === "/health") {
        return Response.json({ status: opts.health ?? "ok" });
      }
      if (path === "/metrics") {
        return opts.metrics === undefined
          ? new Response("not enabled", { status: 404 })
          : new Response(opts.metrics, {
            headers: { "content-type": "text/plain" },
          });
      }
      if (path === "/slots") {
        return opts.slots === undefined
          ? new Response("not enabled", { status: 501 })
          : Response.json(opts.slots);
      }
      if (path === "/v1/chat/completions") {
        await req.body?.cancel();
        const events = reply.map((chunk) =>
          `data: ${
            JSON.stringify({ choices: [{ delta: { content: chunk } }] })
          }\n\n`
        );
        events.push(
          `data: ${
            JSON.stringify({
              choices: [{ delta: {} }],
              timings: { predicted_per_second: opts.tps ?? 12.5 },
            })
          }\n\n`,
        );
        events.push("data: [DONE]\n\n");
        return new Response(events.join(""), {
          headers: { "content-type": "text/event-stream" },
        });
      }
      return new Response("not found", { status: 404 });
    },
  );

  return {
    url: `http://127.0.0.1:${port}`,
    port,
    hits,
    close: async () => {
      ac.abort();
      await server.finished;
    },
  };
}

/** A server that answers 200 at /health but is NOT llama.cpp — the thing a
 *  sweep must not report as a find. */
export function decoyServer(): Stub {
  const port = freePort();
  const hits: string[] = [];
  const ac = new AbortController();
  const server = Deno.serve(
    { port, signal: ac.signal, onListen: () => {} },
    (req) => {
      hits.push(new URL(req.url).pathname);
      return Response.json({ hello: "I am a router admin page" });
    },
  );
  return {
    url: `http://127.0.0.1:${port}`,
    port,
    hits,
    close: async () => {
      ac.abort();
      await server.finished;
    },
  };
}
