// tests/guards.test.ts — structural invariants, checked against the source.
//
// Each of these encodes a mistake that is cheap to make, invisible in review,
// and expensive at runtime. A per-instance fix would leave the next one free to
// happen; a property test over the whole tree does not.

import { assert, assertEquals } from "@std/assert";
import { fromFileUrl, join, relative } from "@std/path";

const ROOT = fromFileUrl(new URL("..", import.meta.url));

async function filesUnder(dir: string, ext: string[]): Promise<string[]> {
  const out: string[] = [];
  for await (const e of Deno.readDir(dir)) {
    const p = join(dir, e.name);
    if (e.isDirectory) out.push(...(await filesUnder(p, ext)));
    else if (ext.some((x) => e.name.endsWith(x))) out.push(p);
  }
  return out;
}

const read = (p: string) => Deno.readTextFile(p);

Deno.test("guard: no server-only module is statically imported by client code", async () => {
  // The #1 aio failure mode: a `*.server.ts` reaching the browser bundle blanks
  // the screen with "Deno is not defined". Server code is only ever reached by
  // `await import()` INSIDE a method, which the bundler marks external.
  const files = (await filesUnder(join(ROOT, "src"), [".ts", ".tsx"]))
    .filter((f) => !f.endsWith(".server.ts"));
  const offenders: string[] = [];
  for (const f of files) {
    const src = await read(f);
    for (const line of src.split("\n")) {
      const isStatic = /^\s*import\s.*from\s+["'][^"']*\.server\.ts["']/.test(
        line,
      );
      const isTypeOnly = /^\s*import\s+type\s/.test(line);
      if (isStatic && !isTypeOnly) {
        offenders.push(`${relative(ROOT, f)}: ${line.trim()}`);
      }
    }
  }
  assertEquals(offenders, [], "static server-only imports blank the screen");
});

Deno.test("guard: no cell or UI file touches Deno.* outside a server module", async () => {
  const files = (await filesUnder(join(ROOT, "src"), [".ts", ".tsx"]))
    .filter((f) => !f.endsWith(".server.ts") && !f.endsWith("/app.ts"));
  const offenders: string[] = [];
  for (const f of files) {
    const src = await read(f);
    // Strip comments so prose about `Deno.*` does not trip the check.
    const code = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(
      /^\s*\/\/.*$/gm,
      "",
    );
    if (/\bDeno\./.test(code)) offenders.push(relative(ROOT, f));
  }
  assertEquals(offenders, [], "browser-reachable code must not use Deno APIs");
});

Deno.test("guard: no method assigns a value derived from its own state back into it", async () => {
  // History, because the reason changed: `s.job = { ...s.job, step }` used to be
  // REJECTED by the store — the whole action was discarded and the build panel
  // froze with no error. aio 1.0.0-alpha38 removed that restriction (verified:
  // the pattern now writes cleanly on `testServer` with `freezeState`), so this
  // is no longer a correctness rule.
  //
  // It is kept as a style rule. Building the next value from a plain local, or
  // mutating the draft in place, says plainly which fields changed; a spread of
  // the previous value hides that and re-copies state on every write. The
  // framework no longer forces the choice, and neither reading nor writing this
  // file should suggest it does.
  const files = await filesUnder(join(ROOT, "src", "cell"), [".ts"]);
  const offenders: string[] = [];
  for (const f of files) {
    if (f.endsWith(".server.ts")) continue;
    const src = await read(f);
    src.split("\n").forEach((line, i) => {
      if (line.includes("aiol-ok")) return;
      // `s.x = { ...s.` / `s.x = [ ...s.` / `s.x = {...(s.` — any shape where a
      // spread of state feeds an assignment back into state.
      if (/s\.\w+\s*=\s*[[{]\s*\.\.\.\(?s\./.test(line)) {
        offenders.push(`${relative(ROOT, f)}:${i + 1}: ${line.trim()}`);
      }
    });
  }
  assertEquals(
    offenders,
    [],
    "prefer mutating the draft or building from a plain local (style, not a runtime constraint since aio alpha38)",
  );
});

Deno.test("guard: no component calls a cell selector", async () => {
  // History: a cell SELECTOR call used to register NO reactive dependency, so a
  // component whose only read was a selector rendered once and then went stale
  // with no warning. It cost an afternoon and produced src/ui/derive.ts, which
  // re-exposes every derived value as a plain function over cell PROPERTIES.
  //
  // aio 1.0.0-alpha38 fixed it — verified: a component reading only
  // `cfg.changedCount()` now re-renders on a state change. So this is no longer
  // load-bearing either. It is kept because `derive.ts` still earns its place as
  // the one list of everything the UI derives, and because a single convention
  // beats two that both work. (Reported in dep/aio/feedback/llama.md.)
  const selectors = new Map<string, string[]>();
  for (const f of await filesUnder(join(ROOT, "src", "cell"), [".ts"])) {
    if (f.endsWith(".server.ts")) continue;
    const src = await read(f);
    const cellName = /\bcell\(\s*"([a-z0-9]+)"/.exec(src)?.[1];
    const block = /selectors:\s*\{([\s\S]*?)\n {2}\},/.exec(src)?.[1];
    if (!cellName || !block) continue;
    const names = [...block.matchAll(/^ {4}(\w+):/gm)].map((m) =>
      m[1] as string
    );
    if (names.length) selectors.set(cellName, names);
  }
  assert(
    selectors.size >= 5,
    `expected to find cell selectors, got ${selectors.size}`,
  );

  const uiFiles = [
    ...(await filesUnder(join(ROOT, "src", "ui"), [".ts", ".tsx"])),
    join(ROOT, "src", "App.tsx"),
  ].filter((f) => !f.endsWith("derive.ts"));

  const offenders: string[] = [];
  for (const f of uiFiles) {
    const src = await read(f);
    src.split("\n").forEach((line, i) => {
      const t = line.trimStart();
      if (t.startsWith("//") || t.startsWith("*")) return;
      for (const [cellName, names] of selectors) {
        for (const sel of names) {
          if (new RegExp(`\\b${cellName}\\.${sel}\\s*\\(`).test(line)) {
            offenders.push(
              `${relative(ROOT, f)}:${i + 1}: ${cellName}.${sel}()`,
            );
          }
        }
      }
    });
  }
  assertEquals(
    offenders,
    [],
    "use src/ui/derive.ts — selectors do not re-render",
  );
});

Deno.test("guard: every cell the app defines is registered in aio.run", async () => {
  // A cell that is imported but not passed to aio.run dispatches into the void:
  // green tests, dead feature. `strictCells: true` catches it at boot; this
  // catches it in CI without booting.
  const cellFiles = (await filesUnder(join(ROOT, "src", "cell"), [".ts"]))
    .filter((f) => !f.endsWith(".server.ts"));
  const names: string[] = [];
  for (const f of cellFiles) {
    const src = await read(f);
    for (const m of src.matchAll(/\bcell\(\s*"([a-z0-9]+)"/g)) {
      names.push(m[1] as string);
    }
  }
  assert(
    names.length >= 8,
    `expected the app's cells, found ${names.join(",")}`,
  );

  const app = await read(join(ROOT, "src", "app.ts"));
  const listed = /cells:\s*\[([^\]]*)\]/.exec(app)?.[1] ?? "";
  for (const n of names) {
    assert(
      new RegExp(`\\b${n}\\b`).test(listed),
      `cell "${n}" is defined but not in aio.run({ cells: [...] })`,
    );
  }
  assert(app.includes("strictCells: true"), "keep the boot-time check on too");
});

Deno.test("guard: the WASM artifact is present and current with the Rust source", async () => {
  // The .wasm is committed so a fresh clone runs without a Rust toolchain —
  // which means it can silently fall behind the .rs files it was built from.
  const wasm = await Deno.stat(join(ROOT, "src", "llama-sys.wasm"));
  assert(wasm.size > 10_000, "the wasm artifact looks truncated");

  const rustFiles = await filesUnder(join(ROOT, "rust", "src"), [".rs"]);
  const newest = Math.max(
    ...(await Promise.all(
      rustFiles.map(async (f) => (await Deno.stat(f)).mtime?.getTime() ?? 0),
    )),
  );
  assert(
    (wasm.mtime?.getTime() ?? 0) >= newest,
    "rust/src is newer than src/llama-sys.wasm — run `deno task wasm`",
  );
});

Deno.test("guard: no text in the stylesheet is a fixed pixel size", async () => {
  // Readability is the reader's call, so every size is a ratio of one root
  // variable that the A− / A+ control sets. A hard-coded px size opts that
  // element out of the whole mechanism.
  const css = await read(join(ROOT, "src", "style.css"));
  const fixed: string[] = [];
  css.split("\n").forEach((line, i) => {
    const m = /font-size:\s*([\d.]+px)/.exec(line);
    if (m) fixed.push(`style.css:${i + 1}: ${m[1]}`);
  });
  assertEquals(fixed, [], "use var(--fs-*) so the zoom control reaches it");
  assert(css.includes("--fs: 14px"), "the default root size must be legible");
});

Deno.test("guard: every source file explains itself at the top", async () => {
  // Not a style rule: these files encode decisions (why a value is estimated,
  // why an import is dynamic) that are invisible in the code itself.
  const files = await filesUnder(join(ROOT, "src"), [".ts", ".tsx"]);
  const bare: string[] = [];
  for (const f of files) {
    const first = (await read(f)).split("\n")[0] ?? "";
    if (!first.startsWith("//")) bare.push(relative(ROOT, f));
  }
  assertEquals(bare, []);
});

Deno.test("guard: the advertised version matches deno.json", async () => {
  // The About page, the window title and aio.run all read src/lib/about.ts; if
  // it drifts from deno.json the app reports a version it is not.
  const about = await read(join(ROOT, "src", "lib", "about.ts"));
  const declared = /version:\s*"([^"]+)"/.exec(about)?.[1];
  const pkg = JSON.parse(await read(join(ROOT, "deno.json"))) as {
    version: string;
  };
  assertEquals(declared, pkg.version);
});

Deno.test("guard: every append-at-the-bottom box follows its newest line", async () => {
  // The kata: "last message is always visible when it arrives". It shipped
  // violated in BOTH chat surfaces at once — the log elements were plain
  // clipping scroll containers with no scroll handling anywhere in src/ — so a
  // reply that overflowed the box stayed below the fold until the user
  // scrolled. A per-component fix is not enough: the day a third chat or log
  // surface is added it must not be able to forget, which is what this checks.
  const files = await filesUnder(join(ROOT, "src", "ui"), [".tsx"]);
  const offenders: string[] = [];
  for (const f of files) {
    const src = await read(f);
    // A scrollable, append-only box is identified by the classes the
    // stylesheet gives `overflow-y: auto`.
    const hasBox = /class="(chat-log|one-chatlog|log)"/.test(src);
    if (hasBox && !src.includes("useStickyBottom(")) {
      offenders.push(relative(ROOT, f));
    }
  }
  assertEquals(
    offenders,
    [],
    "these render a scrolling log/chat box without useStickyBottom()",
  );
});

Deno.test("guard: every chat surface shows that it is waiting", async () => {
  // The sibling of the rule above, and it shipped violated the same way — in
  // BOTH chat surfaces at once. Between Send and the first token there is
  // nothing to render, so a local model thinking for eight seconds looked
  // exactly like a dead server. Whoever adds the third chat surface must not be
  // able to forget, which is what this checks: react to `chat.streaming` and you
  // owe the user an indicator.
  const files = await filesUnder(join(ROOT, "src", "ui"), [".tsx"]);
  const offenders: string[] = [];
  for (const f of files) {
    const src = await read(f);
    const hasBox = /class="(chat-log|one-chatlog)"/.test(src);
    if (hasBox && !src.includes("<Waiting />")) {
      offenders.push(relative(ROOT, f));
    }
  }
  assertEquals(
    offenders,
    [],
    "these render a chat box with no waiting indicator",
  );
});

Deno.test("guard: the icon PNG is present and current with the SVG", async () => {
  // src/icon.svg is the source of the app mark; src/icon.png is what the
  // Electron and Android packagers actually read
  // (dep/aio/src/build/build-electron.ts). The PNG is committed so a packaged
  // build does not need a rasteriser — which means it can silently fall behind
  // the SVG, and ship the previous mark.
  const svg = join(ROOT, "src", "icon.svg");
  const png = join(ROOT, "src", "icon.png");
  const s = await Deno.stat(svg);
  const p = await Deno.stat(png);
  assert(p.size > 1_000, "the icon PNG looks truncated");
  assert(
    (p.mtime?.getTime() ?? 0) >= (s.mtime?.getTime() ?? 0),
    "src/icon.svg is newer than src/icon.png — re-export it (see README)",
  );

  // The three places the mark appears must carry the same geometry, or the tab
  // icon, the title bar and the dock icon drift apart.
  const shape = 'transform="rotate(45 32 32)"';
  for (const f of ["src/icon.svg", "src/App.tsx", "src/app.ts"]) {
    const src = await read(join(ROOT, f));
    assert(src.includes(shape), `${f} must use the same diamond geometry`);
  }
});
