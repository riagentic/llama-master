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
  assertThrows,
} from "@std/assert";
import { join } from "@std/path";
import type { Settings } from "../src/lib/types.ts";

// Point the app home at a temp dir BEFORE anything resolves paths: `start()`
// refuses to run a binary outside its own builds directory, and this is how the
// fixture binary gets to live inside one. This app resolves its home itself
// (`LLAMA_MASTER_HOME`, see `host.server.ts`), so redirecting it is the test's
// job — without this the fixtures land in the real ~/.llama-master, which they
// did until it was fixed.
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
    assertEquals(
      srv.proven,
      true,
      "ready was entered through a real generation, not just /health",
    );
    assert(srv.rssB > 0, "resident size is measured");
    assert(
      srv.rssFileB >= 0 && srv.rssFileB <= srv.rssB,
      `the file-backed share is sampled and sane: ${srv.rssFileB} of ${srv.rssB}`,
    );
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
    assertEquals(
      chat.messages[1]?.thinking,
      "Consider the greeting.",
      "the reasoning channel is kept — a thinking model's whole first act arrives there, and dropping it rendered 'thinking' as nothing",
    );
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
  name:
    "srv: an OOM at first generation is provoked by the probe and walks the ladder",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // The failure that motivated the probe, end to end: the server loads,
    // passes /health, and dies of a CUDA pool OOM the moment it generates —
    // measured on DeepSeek-V4, where "ready" at 17,408 tokens could not answer
    // "Hi". The probe provokes that death BEFORE ready is claimed, the crash is
    // recognised as a fit failure (its stderr never says "buffer"), and the
    // ladder steps the context down: 8192 → 4096 → 2048, then stops at MIN_CTX
    // with the honest diagnosis. `proven` must never have been set, so the
    // crashing context is never recorded as a working one.
    const bin = await installStub();
    const port = freePort();
    using _boot = await bootCells([srv]);

    await srv.start(
      [
        bin,
        "-m",
        "/m.gguf",
        "--port",
        String(port),
        "-c",
        "8192",
        "--oom-on-generate",
      ],
      `http://127.0.0.1:${port}`,
      {
        model: "/m.gguf",
        settings: { ctxSize: 8192 } as unknown as Settings,
        autoFit: true,
      },
    );

    await waitFor(
      async () => {
        await srv.poll();
        return srv.status === "crashed" && srv.diagnosis !== null;
      },
      "the ladder to run dry",
      30_000,
    );

    assertEquals(srv.fitTries, 2, "8192 → 4096 → 2048, each rung measured");
    assert(
      srv.argv.join(" ").includes("-c 2048"),
      `the final command carries the last rung; got ${srv.argv.join(" ")}`,
    );
    assertEquals(srv.runSettings?.ctxSize, 2048);
    assertEquals(
      srv.proven,
      false,
      "no rung generated, so nothing may be remembered as working",
    );
    assertStringIncludes(
      srv.diagnosis?.reason ?? "",
      "during generation",
      "the OOM is explained as memory, not as a driver mismatch",
    );
  },
});

Deno.test({
  name: "srv: a probe that times out is a slow machine, not a dead process",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // The bug: `AbortSignal.timeout` and a dropped connection both reject, and
    // both were reported as `dead`. The poll then waited for an exit that was
    // never coming and re-probed every second, so a server that was merely
    // slow sat at "proving a first reply" indefinitely. The first reply after
    // a cold start runs against a page cache still filling — measured 23x
    // slower prompt processing on a 39 GB model — so this is the normal case
    // on a big model, not an exotic one.
    const bin = await installStub();
    const port = freePort();
    const url = `http://127.0.0.1:${port}`;
    const child = new Deno.Command(bin, {
      args: ["-m", "/m.gguf", "--port", String(port), "--slow-generate"],
      stdout: "piped",
      stderr: "piped",
    }).spawn();
    try {
      await waitFor(
        async () => (await io.health(url)).ok,
        "the stub to answer /health",
        15_000,
      );
      const slow = await io.probe(url, 1_000);
      assertEquals(slow.kind, "slow", `got ${JSON.stringify(slow)}`);
      assertStringIncludes(slow.detail, "no reply");
    } finally {
      child.kill("SIGKILL");
      await child.status;
    }
    // And a process that is genuinely gone still reads as dead, which is the
    // half that must keep working: it is how the OOM the probe provokes is
    // handed to the crash path.
    const gone = await io.probe(url, 2_000);
    assertEquals(gone.kind, "dead");
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

Deno.test({
  name:
    "srv: a path that only textually starts with the builds root is refused",
  fn: () => {
    // `<buildsRoot>/../../../../usr/bin/id` starts with the root as a STRING
    // and leaves it as a PATH. The guard compared text, so the rule that reads
    // like a sandbox ("only binaries this app installed") was not one.
    const escape = join(
      paths().builds,
      "..",
      "..",
      "..",
      "..",
      "usr",
      "bin",
      "id",
    );
    assertThrows(
      () => io.start([escape]),
      Error,
      "refusing to run",
    );
    // The legitimate case still works: verified by the lifecycle tests above,
    // which spawn a stub from inside the builds directory.
  },
});

Deno.test({
  name: "srv: Stop during startup cancels it, rather than letting it come up",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // Found by probing an impatient user. `srv.stop()` early-returned when the
    // status was "stopped" — and a Start that has been dispatched but whose
    // body has not run yet leaves it exactly there. So Stop did nothing, the
    // spawn completed afterwards, and a server the user had cancelled was left
    // running and holding its memory with the UI saying "stopped".
    const bin = await installStub();
    const port = freePort();
    using _boot = await bootCells([srv]);

    const starting = srv.start(
      [bin, "--port", String(port)],
      `http://127.0.0.1:${port}`,
    );
    await srv.stop();
    await starting;
    await srv.poll();

    assertEquals(srv.status, "stopped");
    assertEquals(srv.pid, 0);
    assertEquals(io.status().running, false, "nothing may be left running");
  },
});

Deno.test({
  name: "srv: a double Start leaves one server and an accurate pid",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    // An impatient double-click: the second start is refused by the process
    // module, which used to mark the cell "crashed" with pid 0 while the first
    // server ran happily — a dead-looking panel in front of a live server.
    const bin = await installStub();
    const port = freePort();
    const url = `http://127.0.0.1:${port}`;
    using _boot = await bootCells([srv]);

    await Promise.all([
      srv.start([bin, "--port", String(port)], url),
      srv.start([bin, "--port", String(port)], url),
    ]);
    await waitFor(async () => {
      await srv.poll();
      return srv.status === "ready";
    }, "the server to report ready");

    assert(srv.pid > 0, `the pid must be accurate, got ${srv.pid}`);
    assertEquals(io.status().pid, srv.pid, "and match the real process");
    assertEquals((await io.findOrphans()).length, 0, "no second process");

    await srv.stop();
  },
});

/**
 * The priority switch, verified on the PROCESS rather than on the intent.
 *
 * The interesting failure mode is a switch that reports success and changes
 * nothing — so this reads `/proc/<pid>/stat` back and asserts the kernel agrees.
 * Skipped off Linux, where /proc is not the place to ask.
 */
Deno.test({
  name: "srv: low priority is applied to the real process, not just requested",
  ignore: Deno.build.os !== "linux",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const { niceFromProcStat } = await import("../src/lib/priority.ts");
    const bin = await installStub();
    const port = freePort();
    using _boot = await bootCells([srv]);
    try {
      await srv.start(
        [bin, "--port", String(port)],
        `http://127.0.0.1:${port}`,
      );
      await waitFor(async () => {
        await srv.poll();
        return srv.status === "ready";
      }, "the server to report ready");

      // The renice is fire-and-forget so the UI learns the pid immediately —
      // so wait for the kernel to agree rather than for a promise.
      await waitFor(async () => {
        const stat = await Deno.readTextFile(`/proc/${srv.pid}/stat`);
        return niceFromProcStat(stat) === 19;
      }, "the process to be reniced");

      const stat = await Deno.readTextFile(`/proc/${srv.pid}/stat`);
      assertEquals(
        niceFromProcStat(stat),
        19,
        "nice 19 — the politest there is",
      );
      // And it said so, in the log the app points at for everything else.
      await srv.poll();
      assert(
        srv.log.some((l) => l.includes("desktop keeps priority")),
        `the log says what happened: ${srv.log.slice(-3).join(" | ")}`,
      );
    } finally {
      await srv.stop();
    }
  },
});

/** Off means off: a run started with the switch down stays at the priority the
 *  OS gave it, and nothing in the log claims otherwise. */
Deno.test({
  name: "srv: with the switch off the process keeps the default priority",
  ignore: Deno.build.os !== "linux",
  sanitizeOps: false,
  sanitizeResources: false,
  fn: async () => {
    const { niceFromProcStat } = await import("../src/lib/priority.ts");
    const bin = await installStub();
    const port = freePort();
    using _boot = await bootCells([srv]);
    try {
      await srv.start(
        [bin, "--port", String(port)],
        `http://127.0.0.1:${port}`,
        { model: "", settings: {} as Settings, lowPriority: false },
      );
      await waitFor(async () => {
        await srv.poll();
        return srv.status === "ready";
      }, "the server to report ready");
      const stat = await Deno.readTextFile(`/proc/${srv.pid}/stat`);
      assertEquals(niceFromProcStat(stat), 0, "left where the OS put it");
      assert(!srv.log.some((l) => l.includes("desktop keeps priority")));
    } finally {
      await srv.stop();
    }
  },
});
