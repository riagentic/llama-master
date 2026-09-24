// client/tests/ui.test.ts — the client as a user meets it.
//
// Driven through aio's semantic surface: no selectors, no sleeps. The far end
// is a real HTTP server on a real port (`tests/stub.ts`), so a green test here
// means the journey works end to end — find a server, see what it is running,
// talk to it, take the answer away.

import {
  assert,
  assertEquals,
  assertExists,
  assertStringIncludes,
} from "@std/assert";
import { testUI } from "aio/testing";
import App from "../src/App.tsx";
import { chat } from "../src/cell/chat.ts";
import { conn } from "../src/cell/conn.ts";
import { ui } from "../src/cell/ui.ts";
import { freePort, stubServer } from "./stub.ts";

testUI(App, "opens asking the one question it needs answered", async (ui_) => {
  await ui_.settle();
  const html = ui_.html();
  // The whole interface, on the first frame: an address, a Connect, a Discover.
  assertStringIncludes(html, "llama.master");
  assertExists(ui_.App.host, "an address field");
  assertExists(ui_.App.connect, "and a Connect");
  assertExists(
    ui_.App.discover,
    "and a Discover, for people who do not know it",
  );
  assertStringIncludes(html, "not connected");
  // Nothing to talk to yet, and the chat says what to do rather than sitting
  // there disabled and silent.
  assertStringIncludes(html, "Connect to a llama.master first");
  assertEquals(ui_.App.message.disabled, true);
});

testUI(
  App,
  "connect, then the strip says what is running and how fast it can answer",
  async (ui_) => {
    const stub = stubServer({
      metrics:
        "llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\nllamacpp:predicted_tokens_seconds 40\n",
    });
    try {
      await ui_.settle();
      await conn.forget();
      ui_.App.host.setValue(stub.url);
      ui_.App.connect.click();
      await ui_.expectCell(conn, (s) => s.status === "connected");
      await conn.poll();
      await ui_.settle();

      const html = ui_.html();
      assertStringIncludes(html, "connected");
      // What is running there — the question a client exists to answer.
      assertStringIncludes(html, "stub-7B-Q4_K_M.gguf");
      assertStringIncludes(html, "8,192", "the context it actually allocated");
      // How busy, from the server's own metrics, in slots rather than a number
      // nobody can place.
      assertStringIncludes(html, "1 of 2 busy");
      // And what that means for the person about to press Send.
      assertStringIncludes(html, "tok/s");
      assertStringIncludes(html, "A reply of ~256 tokens");
      assertStringIncludes(html, "≈ 6 s", "256 tokens at 40 tok/s");
    } finally {
      await stub.close();
    }
  },
);

/** A server that does not publish occupancy must not be rendered as an idle
 *  one: "not reported" is the honest answer, and it names the flag that would
 *  change it. */
testUI(App, "a server that publishes no metrics says so", async (ui_) => {
  const stub = stubServer(); // no --metrics, no --slots
  try {
    await ui_.settle();
    await conn.forget();
    ui_.App.host.setValue(stub.url);
    ui_.App.connect.click();
    await ui_.expectCell(conn, (s) => s.status === "connected");
    await conn.poll();
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "not reported");
    assertStringIncludes(html, "does not publish it");
    assert(!html.includes("0 of 2 busy"), "an absence is not an idle reading");
  } finally {
    await stub.close();
  }
});

testUI(
  App,
  "an address nobody answers explains the one thing that is usually wrong",
  async (ui_) => {
    await ui_.settle();
    await conn.forget();
    const port = freePort();
    ui_.App.host.setValue(`127.0.0.1:${port}`);
    ui_.App.connect.click();
    await ui_.expectCell(conn, (s) => s.status === "unreachable");
    await ui_.settle();
    const html = ui_.html();
    assertStringIncludes(html, "unreachable");
    // llama.cpp binds to 127.0.0.1 by default, which is invisible from every
    // other machine — the answer nobody guesses on their own.
    assertStringIncludes(html, "--host 0.0.0.0");
  },
);

testUI(
  App,
  "a whole conversation: send, stream, formatted blocks, copy, clear",
  async (ui_) => {
    const stub = stubServer({
      reply: [
        "Here is the fix:\n\n```ts src/lib/plan.ts\n",
        "const a = 1;\n",
        "```\n",
      ],
      tps: 18.25,
    });
    try {
      await ui_.settle();
      await conn.forget();
      await chat.clear();
      ui_.App.host.setValue(stub.url);
      ui_.App.connect.click();
      await ui_.expectCell(conn, (s) => s.status === "connected");
      await conn.poll();
      await ui_.settle();
      assertEquals(ui_.App.message.disabled, false, "now it can be used");

      ui_.App.message.setValue("fix it");
      ui_.App.send.click();
      // The code itself is no longer one string in the DOM — it is coloured,
      // token by token (`src/lib/highlight.ts`), which is the point.
      const KEYWORD = '<span class="tok-keyword">const</span>';
      await ui_.waitFor(
        () => ui_.html().includes(KEYWORD),
        "the reply arrives",
      );
      await ui_.settle();

      const html = ui_.html();
      // A reply is blocks, not a wall: the fence is gone, the file it named is
      // the block's header, and the block has its own copy button.
      assert(!html.includes("```"), "no raw fences on screen");
      assertStringIncludes(html, "src/lib/plan.ts");
      assertStringIncludes(html, "codeblock");
      assertStringIncludes(html, '<span class="tok-number">1</span>');
      assertExists(ui_.App["codeblock-copy"], "the block can be taken");
      assertExists(ui_.App["chat-copy"], "and so can the conversation");
      // tok/s is a measurement OF the answer, so it goes under it.
      const answer = html.indexOf(KEYWORD);
      assert(
        html.indexOf("tok/s", answer) > answer,
        "tok/s follows the answer",
      );
      // Measured here beats the server's average, and the strip says which.
      await ui_.waitFor(() => ui_.html().includes("measured here"));

      ui_.App["chat-clear"].click();
      await ui_.expectCell(chat, (s) => s.messages.length === 0);
      await ui_.settle();
      assert(!ui_.html().includes(KEYWORD), "clear means gone");
    } finally {
      await stub.close();
    }
  },
);

/** The reply is somebody else's GPU across a network: it can stop mid-sentence.
 *  What arrived must be kept, and what happened must be said. */
testUI(App, "a reply cut short keeps what arrived", async (ui_) => {
  const stub = stubServer({ reply: ["half an ans"] });
  try {
    await ui_.settle();
    await conn.forget();
    await chat.clear();
    ui_.App.host.setValue(stub.url);
    ui_.App.connect.click();
    await ui_.expectCell(conn, (s) => s.status === "connected");
    await conn.poll();
    ui_.App.message.setValue("go");
    ui_.App.send.click();
    await ui_.waitFor(() => ui_.html().includes("half an ans"));
    await ui_.settle();
    assertEquals(chat.messages.length, 2);
    assertEquals(chat.messages[1]?.content, "half an ans");
  } finally {
    await stub.close();
  }
});

/** The client observes; it does not operate. Nothing on this screen can start,
 *  stop or reconfigure the far end — that is the whole difference between this
 *  app and llama.master itself. */
testUI(App, "offers no control over the server it is watching", async (ui_) => {
  const stub = stubServer();
  try {
    await ui_.settle();
    await conn.forget();
    ui_.App.host.setValue(stub.url);
    ui_.App.connect.click();
    await ui_.expectCell(conn, (s) => s.status === "connected");
    await conn.poll();
    await ui_.settle();
    const html = ui_.html().toLowerCase();
    for (const forbidden of ["start server", "stop server", "unload", "-ngl"]) {
      assert(
        !html.includes(forbidden),
        `a client must not offer "${forbidden}"`,
      );
    }
    // And it never asks the server to do anything but answer questions.
    assert(
      stub.hits.every((p) =>
        ["/props", "/health", "/metrics", "/slots", "/v1/chat/completions"]
          .includes(p)
      ),
      `only read endpoints were used: ${[...new Set(stub.hits)].join(", ")}`,
    );
  } finally {
    await stub.close();
  }
});

/** Two settings that belong to the reader rather than to the app: which theme,
 *  and how big the text is. Both on the bar, both persisted, both applied to
 *  the whole document — the page BEHIND the app has to change with the theme,
 *  or a light window sits in a dark frame. */
testUI(
  App,
  "the reader can change the theme and the text size",
  async (ui_) => {
    await ui_.settle();
    await ui.setTheme("dark");
    await ui_.settle();

    assertExists(ui_.App.theme, "a theme toggle");
    assertExists(ui_.App["font-bigger"], "and text size, both ways");
    assertExists(ui_.App["font-smaller"]);

    ui_.App.theme.click();
    await ui_.expectCell(ui, (s) => s.theme === "light");
    await ui_.settle();
    // The attribute the stylesheet keys off, on the app itself — so the theme is
    // right on the first frame, before any effect has run.
    assertStringIncludes(ui_.html(), 'data-theme="light"');

    const before = ui.fontPx;
    ui_.App["font-bigger"].click();
    await ui_.expectCell(ui, (s) => s.fontPx === before + 1);
    await ui_.settle();
    // One control resizes everything: every size in the stylesheet is a ratio
    // of --fs.
    assertStringIncludes(ui_.html(), `--fs: ${before + 1}px`);

    ui_.App["font-smaller"].click();
    await ui_.expectCell(ui, (s) => s.fontPx === before);
    await ui.setTheme("dark");
  },
);

/** "The answer can be put to the clipboard" — the answer itself, not only the
 *  code inside it or the conversation around it. */
testUI(App, "every finished message can be copied on its own", async (ui_) => {
  const stub = stubServer({ reply: ["a whole answer, verbatim"] });
  try {
    await ui_.settle();
    await conn.forget();
    await chat.clear();
    ui_.App.host.setValue(stub.url);
    ui_.App.connect.click();
    await ui_.expectCell(conn, (s) => s.status === "connected");
    await conn.poll();
    ui_.App.message.setValue("ask");
    ui_.App.send.click();
    await ui_.waitFor(() => ui_.html().includes("a whole answer, verbatim"));
    await ui_.settle();
    // One per finished message — the question and the answer.
    assertEquals(
      (ui_.html().match(/aria-label="Copy this (message|answer)"/g) ?? [])
        .length,
      2,
      "the user's message and the model's answer each carry one",
    );
    assertStringIncludes(ui_.html(), "Copy this answer");
  } finally {
    await stub.close();
  }
});

testUI(
  App,
  "a message typed mid-reply becomes a chip, and the box stays open",
  async (ui_) => {
    // The far end streams slowly, which is the only state this feature is
    // about: before it, the box was disabled for the whole reply and the
    // follow-up had to be held in the user's head until the tokens stopped.
    const stub = stubServer({
      // Long enough that the mid-reply window is not a race: the test has to
      // connect, type and click INSIDE it, and a reply that lands first turns
      // the queue assertion below into a coin toss.
      delayMs: 250,
      reply: Array.from({ length: 12 }, (_, i) => ` word${i}`),
    });
    try {
      await ui_.settle();
      await conn.forget();
      await chat.stop();
      await chat.clear();
      await chat.clearQueue();

      ui_.App.host.setValue(stub.url);
      ui_.App.connect.click();
      await ui_.expectCell(conn, (s) => s.status === "connected");
      await conn.poll();
      await ui_.settle();

      ui_.App.message.setValue("first");
      ui_.App.send.click();
      await ui_.expectCell(chat, (s) => s.streaming);

      // The box is still usable WHILE the reply streams — the whole feature.
      assertEquals(
        ui_.App.message.disabled,
        false,
        "the input is closed during a reply — that is the bug this fixes",
      );
      ui_.App.message.setValue("second, while it is busy");
      ui_.App.send.click();
      await ui_.expectCell(chat, (s) => s.queue.length === 1);

      const html = ui_.html();
      assertStringIncludes(html, "second, while it is busy", "drawn as a chip");
      assertStringIncludes(html, "1 message", "and counted");
      assertStringIncludes(
        html,
        "this reply finishes",
        "with what will collect it — a queue nothing is coming for reads stuck",
      );
      assertEquals(
        ui_.App.message.value,
        "",
        "and the box is ready for the next thought",
      );

      // Then it sends itself, in order, with no further gesture. Two whole
      // turns of a deliberately slow stream, so the wait is given a ceiling
      // that matches what is actually being awaited.
      await ui_.waitFor(
        () => !chat.streaming && chat.queue.length === 0,
        {
          timeoutMs: 30_000,
          msg: "the queue should drain itself once the first reply lands",
        },
      );
      assertEquals(chat.messages.length, 4, "both turns completed");
      assertEquals(chat.messages[0]?.content, "first");
      assertEquals(chat.messages[2]?.content, "second, while it is busy");
    } finally {
      await chat.stop();
      await chat.clearQueue();
      await chat.clear();
      await conn.forget();
      await stub.close();
    }
  },
);

/** The system prompt, host and port boxes sit over REPLICATED fields. Bound
 *  per keystroke, anything that re-rendered them mid-edit — the 1 s poll, a
 *  streamed reply's flush — wrote the cell's last acknowledged copy back over
 *  keys still in flight. They are drafts now, committed on change. */
testUI(
  App,
  "typing survives a cell write, and lands on change",
  async (ui_) => {
    await ui_.settle();
    await chat.setSystem("");
    ui_.App.system.setValue("You are terse");
    // A chat-cell write while the user is mid-edit: the header re-renders.
    await chat.clearError();
    await chat.setSystem("");
    await ui_.settle();
    assertEquals(ui_.App.system.value, "You are terse", "the keys survived");
    assertEquals(chat.system, "", "and nothing was dispatched per keystroke");
    await ui_.App.system.blur();
    await ui_.expectCell(chat, (s) => s.system === "You are terse");
    await chat.setSystem("");
  },
);

/** An emptied port box is not a port. It once wrote 0, and the next Connect
 *  went to :80. */
testUI(App, "clearing the port box keeps the port in force", async (ui_) => {
  await ui_.settle();
  await conn.setPort(8080);
  await ui_.settle();
  ui_.App.port.setValue("");
  await ui_.App.port.blur();
  await ui_.settle();
  assertEquals(conn.port, 8080);
  assertEquals(ui_.App.port.value, "8080", "the box shows what is in force");
});

/** Enter fires BEFORE the change that commits the draft, so Enter-to-connect
 *  must commit first — or it connects to the previous address. */
testUI(
  App,
  "Enter in the address box connects to what was typed",
  async (ui_) => {
    const stub = stubServer();
    try {
      await ui_.settle();
      await conn.forget();
      await conn.setHost("192.0.2.1"); // a previous, different address
      await ui_.settle();
      ui_.App.host.setValue(stub.url);
      await ui_.App.host.press("Enter");
      await ui_.expectCell(conn, (s) => s.status === "connected");
      assertEquals(conn.url, stub.url);
    } finally {
      await conn.forget();
      await stub.close();
    }
  },
);

/** One note shows either cell's error; ✕ cleared only conn's, so a chat error
 *  stayed on screen with a button that did nothing. */
testUI(
  App,
  "dismissing the error note clears it, whichever cell said it",
  async (ui_) => {
    const stub = stubServer();
    const gone = stub.url;
    await stub.close();
    await ui_.settle();
    await conn.clearError();
    await chat.clear();
    await chat.send(gone, "anyone?");
    await ui_.settle();
    assert(chat.lastError.length > 0);
    assertExists(ui_.App["error-dismiss"]);
    ui_.App["error-dismiss"].click();
    await ui_.expectCell(chat, (s) => s.lastError === "");
    await ui_.settle();
    assert(!ui_.html().includes('t="error"'), "and the note is gone");
    await chat.clear();
  },
);
