// client/src/app.ts — boot.
//
// A standalone aio app: its own cells, its own store, its own window. It shares
// this repository with llama.master and reuses its PURE libraries (SSE parsing,
// the reply splitter, the formatters) — one implementation of "what did this
// token event contain", not two that drift — and nothing else. It cannot start,
// stop or configure a server; it finds one and talks to it.

import { aio } from "aio";
import { chat } from "./cell/chat.ts";
import { conn } from "./cell/conn.ts";
import { ui } from "./cell/ui.ts";

await aio.run({
  appId: "llama-master-client",
  appVersion: "0.1.0",
  cells: [conn, chat, ui],
  // Boot fails loudly if a cell was defined but not listed — a cell that is
  // imported and unregistered dispatches into the void.
  strictCells: true,
  perfBudget: {
    reduce: 100,
    methods: {
      // A /24 sweep is up to a thousand probes at 700 ms each, sixty-four at a
      // time: seconds, by design, and it reports progress the whole way.
      "conn:discover": { effect: 60_000, timeout: 120_000 },
      // Four small GETs against a machine that may be busy generating.
      "conn:poll": { effect: 8_000, timeout: 15_000 },
      // A reply is as long as the model needs; that is the product.
      "chat:send": { effect: 600_000, timeout: 1_800_000 },
    },
  },
  ui: {
    title: "llama.master client",
    width: 1080,
    height: 860,
    showStatus: false,
    // The mark, as a data URI. ENCODED — written with the SVG's own attributes
    // in double quotes the `href` ends at the first inner quote, the rest is
    // parsed as attributes, and `<rect>` lands in the BODY where it pushes the
    // whole app down by its own height (the same bug, found and fixed in the
    // server app: tests/guards.test.ts there).
    head:
      `<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230c0f14'/%3E%3Crect x='20' y='20' width='24' height='24' rx='4' transform='rotate(45 32 32)' fill='%234cc9f0'/%3E%3C/svg%3E">`,
  },
  schedules: [
    // 2 s: the far end's status, model and occupancy. Slower than the server
    // app's own 1 s poll because every tick is a network round trip to somebody
    // else's machine — and because nothing here changes faster than that.
    // `skipIfRunning`: a tick landing while the previous is still waiting on a
    // busy server is pure pile-up.
    {
      id: "conn.poll",
      every: 2000,
      action: conn.poll.action(),
      skipIfRunning: true,
    },
  ],
});
