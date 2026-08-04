// client/tests/lib.test.ts — the decisions, without a network.
//
// Everything the client concludes about a far end is arithmetic over what that
// end said: where to look for it, what its `/props` means, how busy its
// `/metrics` says it is, and how long a reply should therefore take. All of it
// is pure, so all of it is tested here rather than against a live server.

import { assert, assertEquals } from "@std/assert";
import {
  baseUrl,
  candidates,
  hasPort,
  isPrivateV4,
  KNOWN_PORTS,
  sweepHosts,
} from "../src/lib/discover.ts";
import { highlight } from "../src/lib/highlight.ts";
import {
  busyFraction,
  busyLabel,
  modelName,
  NO_OCCUPANCY,
  parseMetrics,
  parseProps,
  parseSlots,
  replySeconds,
  seconds,
} from "../src/lib/server.ts";

// ── where to look ──────────────────────────────────────────────────────────

/** A client that probes public addresses is a port scanner. The sweep is
 *  confined to the private ranges this machine is already inside. */
Deno.test("discover: only private space is swept", () => {
  assert(isPrivateV4("192.168.1.10"));
  assert(isPrivateV4("10.0.0.4"));
  assert(isPrivateV4("172.16.5.9"));
  assert(isPrivateV4("169.254.7.1"), "link-local is a direct cable");
  assert(!isPrivateV4("8.8.8.8"));
  assert(!isPrivateV4("172.32.0.1"), "just outside 172.16/12");
  assert(!isPrivateV4("not-an-ip"));
  assertEquals(sweepHosts("8.8.8.8"), [], "a public address is not swept");
});

Deno.test("discover: a sweep is one /24, and never the address it starts from", () => {
  const hosts = sweepHosts("192.168.1.10");
  assertEquals(hosts.length, 253, "254 usable, minus ourselves");
  assert(!hosts.includes("192.168.1.10"), "we are not our own server");
  assert(hosts.includes("192.168.1.1") && hosts.includes("192.168.1.254"));
  assert(!hosts.includes("192.168.1.0") && !hosts.includes("192.168.1.255"));
  // A /16 would be 65,534 probes on a button press. The manual field is what
  // reaches a bigger network.
  assert(hosts.every((h) => h.startsWith("192.168.1.")));
});

/** Order is the product: this machine first, because a llama.master on the
 *  same box answers in a millisecond and is the commonest case by far. */
Deno.test("discover: localhost is probed first, then the subnet", () => {
  const urls = candidates([{ address: "192.168.1.10", prefix: 24 }]);
  assertEquals(urls[0], `http://127.0.0.1:${KNOWN_PORTS[0]}`);
  assertEquals(urls[1], `http://127.0.0.1:${KNOWN_PORTS[1]}`);
  const firstRemote = urls.findIndex((u) => u.includes("192.168.1."));
  assert(firstRemote > 0);
  assertEquals(
    urls[firstRemote],
    `http://192.168.1.10:${KNOWN_PORTS[0]}`,
    "our own address before the rest of the subnet",
  );
  assertEquals(urls.length, (1 + 1 + 253) * KNOWN_PORTS.length);
  // Every port is tried per address, so a live host is identified without
  // waiting for 253 timeouts first.
  assert(
    urls.slice(0, KNOWN_PORTS.length).every((u) => u.includes("127.0.0.1")),
  );
});

Deno.test("discover: what the user typed reaches one server, however they typed it", () => {
  assertEquals(baseUrl("192.168.1.9", 18080), "http://192.168.1.9:18080");
  assertEquals(baseUrl(" 192.168.1.9 ", 18080), "http://192.168.1.9:18080");
  assertEquals(
    baseUrl("http://192.168.1.9:8080/", 18080),
    "http://192.168.1.9:8080",
  );
  assertEquals(baseUrl("mini.local", 8080), "http://mini.local:8080");
  assertEquals(baseUrl("https://box:9000", 8080), "https://box:9000");
  assertEquals(baseUrl("", 8080), "", "nothing typed is not a URL");
  // A port already in the text wins over the port picker — otherwise typing a
  // full address and pressing Connect would reach a different machine.
  assert(hasPort("192.168.1.9:8080"));
  assert(!hasPort("192.168.1.9"));
  assertEquals(baseUrl("192.168.1.9:8080"), "http://192.168.1.9:8080");
});

// ── what the far end says ──────────────────────────────────────────────────

Deno.test("server: /props is read whichever shape the build uses", () => {
  const modern = parseProps({
    model_path: "/models/Qwen3.6-27B-Q8_0.gguf",
    total_slots: 4,
    chat_template: "{{ ... }}",
    default_generation_settings: { n_ctx: 32768 },
  });
  assertEquals(modern?.model, "Qwen3.6-27B-Q8_0.gguf");
  assertEquals(modern?.ctx, 32768);
  assertEquals(modern?.slots, 4);
  assertEquals(modern?.chatTemplate, true);
  // Older servers put the context at the top level; reading only one of the
  // two showed "—" against a server that had plainly told us.
  assertEquals(parseProps({ model_path: "/m/a.gguf", n_ctx: 4096 })?.ctx, 4096);
  assertEquals(parseProps({})?.slots, 1, "one slot when it does not say");
  assertEquals(parseProps(null), null);
  assertEquals(modelName("/a/b/c.gguf"), "c.gguf");
});

/** Occupancy is three-valued on purpose: `--metrics` is off by default, and a
 *  client that renders "0% busy" because it could not ask is lying. */
Deno.test("server: occupancy is a reading or an absence, never an invented zero", () => {
  const m = parseMetrics(
    `# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
llamacpp:requests_deferred 1
llamacpp:kv_cache_usage_ratio 0.42
llamacpp:predicted_tokens_seconds 31.5
llamacpp:n_decode_total 1200`,
  );
  assertEquals(m.processing, 2);
  assertEquals(m.queued, 1);
  assertEquals(m.kvUsed, 0.42);
  assertEquals(m.tps, 31.5);
  assertEquals(m.source, "metrics");

  // Not enabled: 404 gives an empty body, and every field stays null.
  assertEquals(parseMetrics(""), NO_OCCUPANCY);
  assertEquals(parseMetrics("").source, "none");

  // The fallback: /slots says who is busy, and nothing else.
  const s = parseSlots([{ id: 0, is_processing: true }, {
    id: 1,
    is_processing: false,
  }]);
  assertEquals(s.processing, 1);
  assertEquals(s.queued, null, "slots do not report a queue");
  assertEquals(s.source, "slots");
  assertEquals(parseSlots({ error: "disabled" }), NO_OCCUPANCY);
});

Deno.test("server: busy is stated in slots, and says when it is not known", () => {
  assertEquals(busyFraction(NO_OCCUPANCY, 4), null, "unknown is not 0%");
  assertEquals(busyLabel(NO_OCCUPANCY, 4), "not reported");
  assertEquals(busyFraction(parseSlots([{ is_processing: true }]), 4), 0.25);
  assertEquals(busyLabel(parseSlots([]), 2), "idle · 2 slots");
  assertEquals(
    busyLabel(
      { ...NO_OCCUPANCY, processing: 2, queued: 3, source: "metrics" },
      2,
    ),
    "2 of 2 busy, 3 waiting",
  );
  // A server reporting more work than it has slots is still bounded: the bar
  // is a bar, not a claim about arithmetic.
  assertEquals(busyFraction({ ...NO_OCCUPANCY, processing: 9 }, 2), 1);
});

/** The estimate the user reads before pressing Send. It has to include the
 *  queue: "3 tok/s" and "3 tok/s once four people ahead of you are done" are
 *  different answers to the same question. */
Deno.test("server: a reply estimate counts the queue, and says nothing when it cannot", () => {
  assertEquals(Math.round(replySeconds(32, 0)), 8, "256 tokens at 32 tok/s");
  assertEquals(Math.round(replySeconds(32, 2)), 24, "three replies deep");
  assertEquals(replySeconds(0, 0), 0, "no rate, no estimate");
  assertEquals(replySeconds(Number.NaN, 0), 0);
  assertEquals(seconds(0), "—");
  assertEquals(seconds(0.4), "<1 s");
  assertEquals(seconds(8), "8 s");
  assertEquals(seconds(90), "1 m 30 s");
  assertEquals(seconds(120), "2 m");
});

// ── syntax colour ──────────────────────────────────────────────────────────

/** The one rule that is not about colour at all: a highlighter that loses a
 *  character has corrupted the answer. Every case below asserts the tokens
 *  concatenate back to the input, byte for byte. */
Deno.test("highlight: the text always survives, whatever the language", () => {
  const samples: [string, string][] = [
    ["ts", 'const x = "a\\"b"; // note\nlet y = 0x1F;'],
    ["python", "def f(a):\n    return 'x' # done"],
    ["rust", 'fn main() { let s = "hi"; /* block */ }'],
    ["bash", '# comment\nexport A=1 && echo "$A"'],
    ["json", '{"a": 1, "b": [true, null]}'],
    ["sql", "SELECT * FROM t -- all\nWHERE a = 'x'"],
    ["notalanguage", "whatever <<>> 123 'q'"],
    ["", "no language at all"],
  ];
  for (const [lang, code] of samples) {
    assertEquals(
      highlight(code, lang).map((t) => t.text).join(""),
      code,
      `${lang} must round-trip`,
    );
  }
});

Deno.test("highlight: the four distinctions, and their precedence", () => {
  const kinds = (code: string, lang = "ts") =>
    highlight(code, lang).filter((t) => t.kind !== "plain").map((t) =>
      `${t.kind}:${t.text}`
    );
  assertEquals(kinds("const a = 1;"), ["keyword:const", "number:1"]);
  // A comment marker inside a string is not a comment, and a quote inside a
  // comment does not open a string: whichever starts first consumes the other.
  assertEquals(kinds('x = "// not a comment"'), ['string:"// not a comment"']);
  assertEquals(kinds("// it's fine"), ["comment:// it's fine"]);
  // A keyword is a whole word, never the tail of an identifier.
  assertEquals(kinds("constant = 2"), ["number:2"]);
  assertEquals(kinds("a.for = 1"), ["keyword:for", "number:1"]);
  // Python and shell use their own comment marker; `//` is not one.
  assertEquals(kinds("# hi", "python"), ["comment:# hi"]);
  assertEquals(kinds("// hi", "python"), []);
  // A block that is still streaming has an unterminated string or comment; it
  // must colour to the end rather than swallowing the next block.
  assertEquals(kinds('const s = "half'), ["keyword:const", 'string:"half']);
  assertEquals(kinds("/* open"), ["comment:/* open"]);
  // An unknown language still gets its strings and numbers — but never a
  // guessed comment marker, which would colour live code as dead.
  assertEquals(kinds("value 42 'q'", "cobol"), ["number:42", "string:'q'"]);
  assertEquals(kinds("// not a comment here", "cobol"), []);
});
