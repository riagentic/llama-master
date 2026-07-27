// test/server.test.ts — the server lifecycle and the chat stream, end to end
// against a real child process on a real socket.
//
// A mocked child process would prove that the code agrees with the mock. This
// spawns an actual binary (test/stub-llama-server.ts, which speaks llama.cpp's
// endpoints), polls it over HTTP exactly as the app does, streams a completion
// through the real SSE parser, and kills it with a real signal.
//
// Every path llama.master can put a user in is covered: it starts, it becomes
// ready, it answers, it stops — and separately, it crashes and is reported as
// crashed rather than quietly "running".

import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import { join } from "@std/path";

// Point the app home at a temp dir BEFORE anything resolves paths: `start()`
// refuses to run a binary outside its own builds directory, and this is how the
// fixture binary gets to live inside one. `bootCells` does NOT redirect the app
// home (only `testServer` does), so without this the fixtures land in the real
// ~/.llama-master — which they did until this was fixed.
const HOME = await Deno.makeTempDir({ prefix: "llama-master-srv-" });
Deno.env.set("LLAMA_MASTER_HOME", HOME);

const { paths } = await import("../src/cell/host.server.ts");
const io = await import("../src/cell/srv.server.ts");
const { srv } = await import("../src/cell/srv.ts");
const { chat } = await import("../src/cell/chat.ts");
const { bootCells } = await import("aio/testing");

/** Install the stub where a real build would be, and return its path. */
async function installStub(): Promise<string> {
  const dir = join(paths().builds, "test-build");
  await Deno.mkdir(dir, { recursive: true });
  const dest = join(dir, "llama-server");
  const src = new URL("./stub-llama-server.ts", import.meta.url);
  await Deno.copyFile(src, dest);
  await Deno.chmod(dest, 0o755);
  return dest;
}

/** A port nothing else is on. Never a constant: two test files that both
 *  hardcode 8080 flake on whichever runs second. */
function freePort(): number {
  const l = Deno.listen({ port: 0 });
  const { port } = l.addr as Deno.NetAddr;
  l.close();
  return port;
}

async function waitFor(
  predicate: () => boolean | Promise<boolean>,
  what: string,
  timeoutMs = 15_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}`);
}

Deno.test({
  name: "srv: start → ready → chat → stop, against a real process",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const bin = await installStub();
    const port = freePort();
    const url = `http://127.0.0.1:${port}`;
    using _boot = await bootCells([srv, chat]);

    await srv.start(
      [bin, "-m", "/models/fixture.gguf", "--port", String(port), "-c", "8192"],
      url,
    );
    assertEquals(srv.status, "starting");
    assert(srv.pid > 0, "a pid is recorded");

    // The poll is the single writer of liveness — drive it exactly as the
    // schedule in src/app.ts does.
    await waitFor(async () => {
      await srv.poll();
      return srv.status === "ready";
    }, "the server to report healthy");

    assertEquals(srv.healthy, true);
    assertEquals(srv.loadedModel(), "/models/fixture.gguf");
    assertEquals(srv.props?.n_ctx, 8192);
    assert(
      srv.log.some((l) => l.includes("listening")),
      `the child's stdout reaches the log; got ${JSON.stringify(srv.log)}`,
    );
    assert(
      srv.log[0]?.startsWith("$ "),
      "the log opens with the exact command",
    );

    // Chat over the same socket, through the real SSE parser.
    chat.setInput("hi");
    await chat.send(url);
    assertEquals(chat.streaming, false);
    assertEquals(chat.messages.length, 2);
    assertEquals(chat.messages[0]?.role, "user");
    assertEquals(chat.messages[1]?.content, "Hello from the stub");
    assertEquals(chat.lastTps, 42.5);

    await srv.stop();
    await srv.poll();
    assertEquals(srv.status, "stopped");
    assertEquals(srv.pid, 0);
    assertEquals(io.status().running, false, "the process really is gone");
  },
});

Deno.test({
  name: "srv: a process that dies on its own is reported as crashed",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const bin = await installStub();
    const port = freePort();
    using _boot = await bootCells([srv]);

    await srv.start(
      [bin, "-m", "/m.gguf", "--port", String(port), "--fail-after", "120"],
      `http://127.0.0.1:${port}`,
    );
    await waitFor(async () => {
      await srv.poll();
      return srv.status === "crashed";
    }, "the crash to be noticed");

    assertEquals(srv.exitCode, 3);
    assertStringIncludes(srv.lastError, "exited with code 3");
    assertEquals(srv.healthy, false);
  },
});

Deno.test({
  name: "srv: refuses to run a binary outside its own builds directory",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    using _boot = await bootCells([srv]);
    await srv.start(["/bin/sh", "-c", "echo pwned"], "http://127.0.0.1:1");
    assertEquals(srv.status, "crashed");
    assertStringIncludes(srv.lastError, "refusing to run");
    assertEquals(io.status().running, false);
  },
});

Deno.test({
  name: "srv: a second start while one is running is refused, not stacked",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const bin = await installStub();
    const port = freePort();
    const url = `http://127.0.0.1:${port}`;
    using _boot = await bootCells([srv]);

    await srv.start([bin, "-m", "/m.gguf", "--port", String(port)], url);
    const firstPid = srv.pid;
    await srv.start([bin, "-m", "/other.gguf", "--port", String(port)], url);
    assertEquals(srv.pid, firstPid, "the running server is untouched");
    await srv.stop();
  },
});

Deno.test("srv: health and props degrade quietly when nothing is listening", async () => {
  const dead = `http://127.0.0.1:${freePort()}`;
  const h = await io.health(dead, 300);
  assertEquals(h.ok, false);
  assert(h.detail.length > 0, "the reason is reported, not swallowed");
  assertEquals(await io.props(dead, 300), null);
});

// Clean up the temp app home once every test above has finished with it.
globalThis.addEventListener("unload", () => {
  try {
    Deno.removeSync(HOME, { recursive: true });
  } catch {
    // Best effort — a leftover temp dir is not worth failing a suite over.
  }
});

Deno.test({
  name: "srv: a stray llama-server is found, and stopping it releases it",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // The real scenario: a server survives the app (crash, SIGKILL, restart)
    // and keeps its VRAM. It must be findable by pid and stoppable, or the next
    // Start dies with out-of-memory on a machine that looks idle.
    const bin = await installStub();
    const port = freePort();
    const child = new Deno.Command(bin, {
      args: ["-m", "/m.gguf", "--port", String(port)],
      stdout: "null",
      stderr: "null",
      stdin: "null",
    }).spawn();

    try {
      await waitFor(
        async () => (await io.findOrphans()).some((o) => o.pid === child.pid),
        "the stray server to be detected",
      );
      const found = (await io.findOrphans()).find((o) => o.pid === child.pid);
      assert(found, "detected");
      assertStringIncludes(found.argv, "llama-server");

      await io.stopOrphan(child.pid);
      await waitFor(
        async () => !(await io.findOrphans()).some((o) => o.pid === child.pid),
        "the stray server to be gone",
      );
    } finally {
      try {
        child.kill("SIGKILL");
      } catch {
        // already stopped
      }
      await child.status;
    }
  },
});

Deno.test({
  name: "srv: a process outside the builds directory is never touched",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // A llama-server the user runs by hand from elsewhere is theirs, not ours.
    const orphans = await io.findOrphans();
    for (const o of orphans) {
      assertStringIncludes(o.argv, paths().builds);
    }
    await assertRejects(
      () => io.stopOrphan(1),
      Error,
      "not a llama-server started from this app",
    );
  },
});

Deno.test({
  name: "srv: a server that fails instantly is still diagnosed from its output",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // The race this pins: `child.status` resolves when the process is reaped,
    // which can beat the last of its stderr through the pipe. The cell
    // diagnoses an exit from exactly those lines, so reporting "not running"
    // before the pumps drained replaced the real reason with the generic
    // fallback — on precisely the failures that happen fastest.
    const bin = await installStub();
    using _boot = await bootCells([srv]);

    await srv.start([bin, "--oom"], "http://127.0.0.1:1");
    await waitFor(async () => {
      await srv.poll();
      return srv.status === "crashed";
    }, "the server to be seen as crashed");

    assert(
      srv.log.some((l) => l.includes("cudaMalloc failed")),
      `the child's stderr must be captured before the exit is reported; got ${
        JSON.stringify(srv.log)
      }`,
    );
    assertStringIncludes(srv.diagnosis?.reason ?? "", "GPU ran out of memory");
    await srv.clearLog();
  },
});
