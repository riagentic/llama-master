// client/tests/cells.test.ts — the cells against a real server on a real port.
//
// `bootCells` runs the real dispatch loop, and the far end is `tests/stub.ts`:
// an actual HTTP server answering the way llama.cpp answers. Nothing is mocked,
// because everything that could be mocked here — fetch, the SSE framing, the
// endpoint shapes — is exactly what the client has to get right.

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { bootCells, testCell } from "aio/testing";
import { conn } from "../src/cell/conn.ts";
import { chat } from "../src/cell/chat.ts";
import { ui } from "../src/cell/ui.ts";
import { decoyServer, freePort, stubServer } from "./stub.ts";

Deno.test("conn: connecting reads what the server has loaded", async () => {
  const stub = stubServer();
  using _boot = await bootCells([conn]);
  try {
    await conn.setHost(stub.url);
    await conn.connect();
    assertEquals(conn.status, "connected");
    assertEquals(conn.info?.model, "stub-7B-Q4_K_M.gguf");
    assertEquals(conn.info?.ctx, 8192);
    assertEquals(conn.info?.slots, 2);
    // The fields fill in, so a connection is one the user can see and edit.
    assertEquals(conn.host, "127.0.0.1");
    assertEquals(conn.port, stub.port);
  } finally {
    await stub.close();
  }
});

/** The commonest failure on a LAN, and the one the user cannot guess: llama.cpp
 *  binds to 127.0.0.1 unless told otherwise, so it is invisible from every
 *  other machine. A bare "unreachable" would send them hunting the network. */
Deno.test("conn: an address that does not answer says what to check", async () => {
  using _boot = await bootCells([conn]);
  const port = freePort(); // free, therefore nothing is listening on it
  await conn.setHost(`127.0.0.1:${port}`);
  await conn.connect();
  assertEquals(conn.status, "unreachable");
  assertEquals(conn.info, null);
  assertStringIncludes(conn.lastError, "--host 0.0.0.0");
});

Deno.test("conn: nothing typed is refused with a sentence, not a silent no-op", async () => {
  using _boot = await bootCells([conn]);
  await conn.setHost("   ");
  await conn.connect();
  assertStringIncludes(conn.lastError, "Type an address first");
  assertEquals(conn.status, "idle");
});

/** Occupancy comes from `/metrics` when the server publishes it, `/slots` when
 *  it publishes that instead, and NOTHING when it publishes neither — the one
 *  case a client is tempted to render as 0%. */
Deno.test("conn: poll prefers metrics, falls back to slots, admits neither", async () => {
  const withMetrics = stubServer({
    metrics: "llamacpp:requests_processing 1\nllamacpp:requests_deferred 2\n",
  });
  using _boot = await bootCells([conn]);
  try {
    await conn.setHost(withMetrics.url);
    await conn.connect();
    await conn.poll();
    assertEquals(conn.ready, true);
    assertEquals(conn.occupancy.source, "metrics");
    assertEquals(conn.occupancy.processing, 1);
    assertEquals(conn.occupancy.queued, 2);
  } finally {
    await withMetrics.close();
  }

  const withSlots = stubServer({
    slots: [{ is_processing: true }, { is_processing: false }],
  });
  try {
    await conn.setHost(withSlots.url);
    await conn.connect();
    await conn.poll();
    assertEquals(conn.occupancy.source, "slots");
    assertEquals(conn.occupancy.processing, 1);
    assert(
      withSlots.hits.includes("/metrics"),
      "metrics is asked for first — it says more",
    );
  } finally {
    await withSlots.close();
  }

  const bare = stubServer();
  try {
    await conn.setHost(bare.url);
    await conn.connect();
    await conn.poll();
    assertEquals(conn.occupancy.source, "none");
    assertEquals(conn.occupancy.processing, null, "unknown is not zero");
  } finally {
    await bare.close();
  }
});

/** A server that answered once and stopped is a different situation from one
 *  that never answered: wait, versus check the address. */
Deno.test("conn: a server that goes away is lost, not unreachable", async () => {
  const stub = stubServer();
  using _boot = await bootCells([conn]);
  await conn.setHost(stub.url);
  await conn.connect();
  assertEquals(conn.status, "connected");
  await stub.close();
  await conn.poll();
  assertEquals(conn.status, "lost");
  assertEquals(conn.ready, false);
});

/** Health says "loading model" while the weights are still going in. The client
 *  must show the server, and must not let anyone press Send at it. */
Deno.test("conn: a loading server is connected but not usable", async () => {
  const stub = stubServer({ health: "loading model" });
  using _boot = await bootCells([conn]);
  try {
    await conn.setHost(stub.url);
    await conn.connect();
    await conn.poll();
    assertEquals(conn.status, "connected");
    assertEquals(conn.ready, false);
    assertEquals(conn.usable(), false);
    assertEquals(conn.healthDetail, "loading model");
  } finally {
    await stub.close();
  }
});

/** The identifying answer is `/props`, not a 200: a sweep that accepted any
 *  live port would list every router admin page on the subnet. */
Deno.test("conn: a decoy that answers 200 is not reported as a server", async () => {
  const decoy = decoyServer();
  using _boot = await bootCells([conn]);
  try {
    const io = await import("../src/cell/conn.server.ts");
    assertEquals(await io.probe(decoy.url, 800), null);
    assert(decoy.hits.includes("/props"), "it WAS asked");
  } finally {
    await decoy.close();
  }
});

Deno.test("conn: forgetting a server leaves nothing behind to look live", async () => {
  const stub = stubServer();
  using _boot = await bootCells([conn]);
  try {
    await conn.setHost(stub.url);
    await conn.connect();
    await conn.poll();
    await conn.forget();
    assertEquals(conn.url, "");
    assertEquals(conn.info, null);
    assertEquals(conn.status, "idle");
    assertEquals(conn.occupancy.source, "none");
    // And a poll with nothing connected is silent rather than an error.
    await conn.poll();
    assertEquals(conn.lastError, "");
  } finally {
    await stub.close();
  }
});

// ── chat ───────────────────────────────────────────────────────────────────

Deno.test("chat: a reply streams in, is committed once, and carries its rate", async () => {
  const stub = stubServer({ reply: ["Hel", "lo ", "LAN"], tps: 21.5 });
  using _boot = await bootCells([chat]);
  try {
    await chat.clear();
    await chat.setInput("hi");
    await chat.send(stub.url);
    assertEquals(chat.streaming, false);
    assertEquals(chat.messages.length, 2);
    assertEquals(chat.messages[1]?.content, "Hello LAN");
    assertEquals(chat.messages[1]?.tps, 21.5);
    assertEquals(chat.lastTps, 21.5);
    assertEquals(chat.partial, "", "nothing in flight is left behind");
    // Measured on this client, from Send to the first token — a number no
    // server-side metric reports.
    assert(chat.lastLatencyMs >= 0);
  } finally {
    await stub.close();
  }
});

/** A LAN drops. What the user gets back must say what to do about it, and must
 *  not be a raw fetch error. */
Deno.test("chat: a server that vanishes mid-conversation explains itself", async () => {
  const stub = stubServer();
  const url = stub.url;
  await stub.close();
  using _boot = await bootCells([chat]);
  await chat.clear();
  await chat.setInput("are you there");
  await chat.send(url);
  assertEquals(chat.streaming, false);
  assert(chat.lastError.length > 0);
  assertStringIncludes(chat.lastError, "Discover");
  // The question stays in the log: a message that vanishes with the error
  // makes the user type it again.
  assertEquals(chat.messages.length, 1);
  assertEquals(chat.messages[0]?.content, "are you there");
});

Deno.test("chat: clear wipes the conversation and the numbers with it", async () => {
  const stub = stubServer();
  using _boot = await bootCells([chat]);
  try {
    await chat.setInput("hi");
    await chat.send(stub.url);
    assert(chat.messages.length > 0 && chat.lastTps > 0);
    await chat.clear();
    assertEquals(chat.messages.length, 0);
    assertEquals(chat.lastTps, 0);
    assertEquals(chat.lastLatencyMs, 0);
    assertEquals(chat.lastError, "");
  } finally {
    await stub.close();
  }
});

/** The address is the one field something other than the UI can write — `am
 *  dispatch`, a script, a future deep link — and it PERSISTS. A malformed
 *  payload once put `[object Object]` in the box, where it survived a restart
 *  and no amount of clicking Connect could get past it.
 *
 *  Through `testCell` rather than `bootCells`: this is pure reducer behaviour
 *  with no I/O in it, which is exactly what that harness is for. */
testCell(
  conn,
  "the address field cannot be poisoned by a bad dispatch",
  (t) => {
    t.init();
    // deno-lint-ignore no-explicit-any -- the point is a caller that is not the UI
    t.send.setHost({ args: ["192.168.1.9"] } as any);
    t.expect.state((s) => s.host === "");
    // deno-lint-ignore no-explicit-any
    t.send.setPort("not a port" as any);
    t.expect.state((s) => s.port === 0);
    t.send.setHost("192.168.1.9");
    t.send.setPort(8080);
    t.expect.state((s) => s.host === "192.168.1.9" && s.port === 8080);
  },
);

// ── how it looks ───────────────────────────────────────────────────────────

/** Both settings are the reader's, and both persist: a person who has chosen
 *  light at 18px has chosen it, and being asked again every launch is the app
 *  disagreeing with them. */
testCell(ui, "the theme flips and the text size steps, within reason", (t) => {
  t.init();
  t.expect.state((s) => s.theme === "dark" && s.fontPx === 14);
  t.send.toggleTheme();
  t.expect.state((s) => s.theme === "light");
  t.send.toggleTheme();
  t.expect.state((s) => s.theme === "dark");

  // Clamped at both ends: below 12 the mono blocks stop being readable, above
  // 20 the connect bar wraps to three rows.
  for (let i = 0; i < 20; i++) t.send.zoom(1);
  t.expect.state((s) => s.fontPx === 20);
  for (let i = 0; i < 40; i++) t.send.zoom(-1);
  t.expect.state((s) => s.fontPx === 12);
  // And a caller that is not the UI cannot size the app to nothing.
  // deno-lint-ignore no-explicit-any
  t.send.zoom("lots" as any);
  t.expect.state((s) => s.fontPx === 12);
});
