// test/deepseek.e2e.test.ts — the whole promise, on a real 145 GB model.
//
// Everything else in `tests/` runs on fixtures in milliseconds. This one loads
// DeepSeek-V4-Flash for real, lets the retry ladder find a context that fits,
// and asks it a question. It exists because that model broke four separate
// assumptions in a row, and every one of them looked fine in the unit tests:
//
//   1. a split GGUF read as its first part (145 GB seen as 37 GB)
//   2. layers divided across two cards by count, piling 34 GB on one 24 GB card
//   3. `--mlock` and `--no-mmap` cancelling each other
//   4. a compute buffer estimated at 730 MB that llama.cpp sized at 68.5 GiB
//
// (4) is why the ladder exists: it cannot be derived from the header, only
// measured. So this test measures it, and the only way to know the app really
// starts this model is to really start it.
//
// OPT-IN, because it takes ~10 minutes and needs the model on disk:
//   LLAMA_MASTER_E2E=1 deno test -A --no-check tests/deepseek.e2e.test.ts
// Skipped otherwise, so `deno task test` stays a few seconds.

import { assert, assertEquals } from "@std/assert";
import { ctxOf, fitDecision, MAX_FIT_RETRIES } from "../src/lib/fitladder.ts";
import { diagnoseServerExit } from "../src/lib/serverlog.ts";

const MODEL =
  "/home/dev/.lmstudio/models/lmstudio-community/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf";
const BIN =
  "/home/dev/.llama-master/data/files/builds/source-master-cuda/bin/llama-server";
const PORT = 18077;

const enabled = Deno.env.get("LLAMA_MASTER_E2E") === "1";

async function have(path: string): Promise<boolean> {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

/** Spawn, and return everything the cell's `poll` reasons about. */
async function attempt(
  argv: string[],
  onLine: (l: string) => void,
): Promise<{ ok: boolean; lines: string[]; code: number | null }> {
  const child = new Deno.Command(argv[0]!, {
    args: argv.slice(1),
    stdout: "piped",
    stderr: "piped",
  }).spawn();

  const lines: string[] = [];
  const pump = async (r: ReadableStream<Uint8Array>) => {
    const dec = new TextDecoder();
    let buf = "";
    for await (const chunk of r) {
      buf += dec.decode(chunk, { stream: true });
      const parts = buf.split("\n");
      buf = parts.pop() ?? "";
      for (const p of parts) {
        lines.push(p);
        onLine(p);
      }
    }
  };
  const pumps = Promise.all([pump(child.stdout), pump(child.stderr)]);

  // Poll for health exactly as `srv.poll` does, with a ceiling per attempt.
  let ok = false;
  for (let i = 0; i < 150; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      if (res.ok) {
        const body = await res.json();
        if (body?.status === "ok") {
          ok = true;
          break;
        }
      } else {
        await res.body?.cancel();
      }
    } catch {
      // not up yet, or already dead — the loop below decides which
    }
    // Dead? `try_wait` is not exposed, so probe the process instead.
    try {
      Deno.kill(child.pid, "SIGCONT" as Deno.Signal);
    } catch {
      break; // reaped
    }
  }

  if (ok) return { ok, lines, code: null };
  try {
    child.kill("SIGKILL");
  } catch { /* already gone */ }
  const st = await child.status;
  await pumps;
  // Let the driver actually release the VRAM before the next rung asks for it.
  // Without this the next attempt races a 100 GB teardown and dies with a hard
  // CUDA error instead of a clean allocation failure — which is not a fit
  // failure, so the ladder correctly refuses to retry and the test reports
  // defeat one rung from the answer. The app never sees this: `srv.poll` only
  // starts the next attempt after the process has been reaped and a poll tick
  // has passed.
  await new Promise((r) => setTimeout(r, 20_000));
  return { ok: false, lines, code: st.code };
}

Deno.test({
  name:
    "e2e: the app finds a context DeepSeek-V4 actually runs at, and it answers",
  ignore: !enabled,
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    assert(await have(MODEL), `model not on disk: ${MODEL}`);
    assert(await have(BIN), `build not on disk: ${BIN}`);

    // The command the tuner composes for this machine. The context is the
    // model's declared 1,048,576 — which does NOT fit, and is the whole point:
    // the ladder has to discover that and come down.
    let argv = [
      BIN,
      "-m",
      MODEL,
      "-ngl",
      "999",
      "--n-cpu-moe",
      "38",
      "-ts",
      "40.5,3.5",
      "-c",
      "1048576",
      "-fa",
      "on",
      "-t",
      "14",
      "-tb",
      "14",
      "-fit",
      "off",
      "--port",
      String(PORT),
    ];

    const rungs: number[] = [];
    let started = false;
    for (let tries = 0; tries <= MAX_FIT_RETRIES; tries++) {
      const ctx = ctxOf(argv);
      rungs.push(ctx);
      console.log(`  attempt ${tries + 1}: ctx ${ctx.toLocaleString()}`);
      const r = await attempt(argv, () => {});
      if (r.ok) {
        started = true;
        break;
      }
      // Exactly the decision `srv.poll` makes, on exactly those log lines.
      const d = fitDecision({ lines: r.lines, ctx, tries, auto: true });
      if (d.kind !== "retry") {
        const why = diagnoseServerExit(r.code, r.lines);
        throw new Error(
          `gave up at ctx ${ctx}: ${why.reason}\n${
            r.lines.slice(-6).join("\n")
          }`,
        );
      }
      console.log(`    → ${d.note}`);
      argv = argv.map((t, i) => (argv[i - 1] === "-c" ? String(d.ctx) : t));
    }

    assert(started, `never started; rungs tried: ${rungs.join(" → ")}`);
    assert(rungs.length > 1, "1M was expected to fail — the ladder should run");

    try {
      // It is up. Does it actually answer? A server that loads and then produces
      // nothing is not a working model.
      const res = await fetch(
        `http://127.0.0.1:${PORT}/v1/chat/completions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: [{
              role: "user",
              content: "Reply with exactly the word: WORKING",
            }],
            max_tokens: 24,
            temperature: 0,
          }),
          signal: AbortSignal.timeout(180_000),
        },
      );
      assertEquals(res.status, 200);
      const body = await res.json();
      const text = String(body?.choices?.[0]?.message?.content ?? "");
      console.log(`  reply: ${text.trim().slice(0, 80)}`);
      assert(text.trim().length > 0, "the model produced no tokens");
      assert(
        /working/i.test(text),
        `expected the model to follow a trivial instruction, got: ${text}`,
      );
      const tps = Number(body?.timings?.predicted_per_second ?? 0);
      console.log(`  ${tps.toFixed(1)} tok/s at ctx ${ctxOf(argv)}`);
      assert(tps > 0.5, `implausibly slow: ${tps} tok/s`);
    } finally {
      // Never leave a 100 GB process behind because an assertion failed.
      for (const p of await pidsOn(PORT)) {
        try {
          Deno.kill(p, "SIGKILL");
        } catch { /* already gone */ }
      }
    }
  },
});

/** Every llama-server holding this port — the cleanup has to be certain. */
async function pidsOn(port: number): Promise<number[]> {
  const out = await new Deno.Command("bash", {
    args: ["-lc", `pgrep -f 'llama-server.*--port ${port}' || true`],
    stdout: "piped",
  }).output();
  return new TextDecoder().decode(out.stdout)
    .split("\n").map((l) => Number(l.trim())).filter((n) => n > 0);
}
