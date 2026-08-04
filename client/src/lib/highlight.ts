// client/src/lib/highlight.ts — enough syntax colour to read code by.
//
// Deliberately small, and deliberately not a parser. What makes code readable at
// a glance is four distinctions — this is a comment, this is a string, this is a
// number, this is a keyword — and all four can be had from a single left-to-
// right scan. A real highlighter (tree-sitter, Prism, hljs) is a dependency,
// a bundle and a language registry, for a chat window that shows twenty lines
// at a time.
//
// What that buys, and what it costs: colour is a hint here, not an analysis. A
// keyword inside an identifier is not highlighted (word boundaries are
// respected), but a language the map does not know gets strings, comments and
// numbers only — which is still most of the benefit. It never rewrites the
// text: every token's `text` concatenates back to the input exactly, which the
// tests pin, because a highlighter that loses a character is worse than none.

export type TokenKind = "plain" | "comment" | "string" | "number" | "keyword";
export type Token = { kind: TokenKind; text: string };

/** Comment syntax by language family. `line` starts one to end of line; `block`
 *  is an open/close pair. */
type Syntax = {
  line: string[];
  block: [string, string][];
  /** Quote characters that start a string. */
  quotes: string[];
  keywords: Set<string>;
};

const WORDS = {
  js: `abstract as async await break case catch class const continue debugger
    declare default delete do else enum export extends false finally for from
    function get if implements import in instanceof interface let new null of
    private protected public readonly return satisfies set static super switch
    this throw true try type typeof undefined var void while yield`,
  py: `and as assert async await break class continue def del elif else except
    False finally for from global if import in is lambda None nonlocal not or
    pass raise return True try while with yield`,
  rust: `as async await break const continue crate dyn else enum extern false fn
    for if impl in let loop match mod move mut pub ref return self Self static
    struct super trait true type unsafe use where while`,
  go: `break case chan const continue default defer else fallthrough for func go
    goto if import interface map package range return select struct switch type
    var true false nil`,
  c: `auto bool break case char class const constexpr continue default delete do
    double else enum extern false float for friend goto if inline int long
    namespace new nullptr operator private protected public return short signed
    sizeof static struct switch template this throw true try typedef union
    unsigned using virtual void volatile while`,
  sh:
    `case do done elif else esac export fi for function if in local read return
    then until while`,
  sql: `and as asc by create delete desc distinct drop from group having insert
    into join left limit not null on or order outer right select set table union
    update values where`,
} as const;

const set = (words: string) => new Set(words.split(/\s+/).filter(Boolean));

const JS: Syntax = {
  line: ["//"],
  block: [["/*", "*/"]],
  quotes: ['"', "'", "`"],
  keywords: set(WORDS.js),
};
const PY: Syntax = {
  line: ["#"],
  block: [['"""', '"""'], ["'''", "'''"]],
  quotes: ['"', "'"],
  keywords: set(WORDS.py),
};
const RUST: Syntax = {
  line: ["//"],
  block: [["/*", "*/"]],
  quotes: ['"'],
  keywords: set(WORDS.rust),
};
const GO: Syntax = {
  line: ["//"],
  block: [["/*", "*/"]],
  quotes: ['"', "`"],
  keywords: set(WORDS.go),
};
const C: Syntax = {
  line: ["//"],
  block: [["/*", "*/"]],
  quotes: ['"', "'"],
  keywords: set(WORDS.c),
};
const SH: Syntax = {
  line: ["#"],
  block: [],
  quotes: ['"', "'"],
  keywords: set(WORDS.sh),
};
const SQL: Syntax = {
  line: ["--"],
  block: [["/*", "*/"]],
  quotes: ["'", '"'],
  keywords: set(WORDS.sql),
};
/** JSON, YAML, TOML, INI: no keywords worth the name, but strings, numbers and
 *  (outside JSON) comments carry the shape. */
const DATA: Syntax = {
  line: ["#"],
  block: [],
  quotes: ['"', "'"],
  keywords: new Set(["true", "false", "null"]),
};
/** A language the map does not know. Quotes are near-universal, so strings and
 *  numbers still carry; comment markers are not, and guessing one would colour
 *  live code as a comment. */
const PLAIN: Syntax = {
  line: [],
  block: [],
  quotes: ['"', "'"],
  keywords: new Set(),
};

const BY_LANG: Record<string, Syntax> = {
  ts: JS,
  tsx: JS,
  typescript: JS,
  js: JS,
  jsx: JS,
  javascript: JS,
  json: DATA,
  jsonc: DATA,
  py: PY,
  python: PY,
  rs: RUST,
  rust: RUST,
  go: GO,
  c: C,
  h: C,
  cpp: C,
  "c++": C,
  cuda: C,
  java: C,
  kt: C,
  kotlin: C,
  swift: C,
  "c#": C,
  cs: C,
  php: C,
  sh: SH,
  bash: SH,
  zsh: SH,
  shell: SH,
  fish: SH,
  console: SH,
  yaml: DATA,
  yml: DATA,
  toml: DATA,
  ini: DATA,
  sql: SQL,
  dockerfile: SH,
  makefile: SH,
  make: SH,
  cmake: SH,
};

/** The syntax for a language name, or the safe default: strings and numbers
 *  only. An unknown language must still be readable, never mangled. */
export function syntaxFor(lang: string): Syntax {
  return BY_LANG[lang.trim().toLowerCase()] ?? PLAIN;
}

const isWordChar = (c: string) => /[A-Za-z0-9_$]/.test(c);
const isDigit = (c: string) => c >= "0" && c <= "9";

/**
 * Split code into coloured tokens.
 *
 * One pass, no backtracking. The order of the checks is the priority: a `//`
 * inside a string is not a comment, and a quote inside a comment does not open
 * a string, because whichever starts first consumes the other.
 */
export function highlight(code: string, lang: string): Token[] {
  const sx = syntaxFor(lang);
  const out: Token[] = [];
  let plain = "";
  const flush = () => {
    if (plain) out.push({ kind: "plain", text: plain });
    plain = "";
  };
  const push = (kind: TokenKind, text: string) => {
    flush();
    out.push({ kind, text });
  };

  let i = 0;
  while (i < code.length) {
    const rest = code.slice(i);

    // Comments — to end of line, or to the closing marker (unterminated runs
    // to the end, which is what an editor does and what a streamed block needs).
    const line = sx.line.find((m) => rest.startsWith(m));
    if (line) {
      const end = code.indexOf("\n", i);
      const stop = end === -1 ? code.length : end;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }
    const block = sx.block.find(([open]) => rest.startsWith(open));
    if (block) {
      const [open, close] = block;
      const end = code.indexOf(close, i + open.length);
      const stop = end === -1 ? code.length : end + close.length;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }

    // Strings — with escapes, and unterminated ones running to the end of the
    // line so a half-streamed block does not colour the rest of the file.
    const ch = code[i] as string;
    if (sx.quotes.includes(ch)) {
      let j = i + 1;
      while (j < code.length) {
        const c = code[j] as string;
        if (c === "\\") {
          j += 2;
          continue;
        }
        if (c === ch) {
          j++;
          break;
        }
        if (c === "\n" && ch !== "`") break;
        j++;
      }
      push("string", code.slice(i, Math.min(j, code.length)));
      i = Math.min(j, code.length);
      continue;
    }

    // Numbers, including hex and decimals, but never the tail of an identifier.
    if (isDigit(ch) && !isWordChar(code[i - 1] ?? "")) {
      let j = i;
      while (j < code.length && /[0-9a-fA-FxXoObB._]/.test(code[j] as string)) {
        j++;
      }
      push("number", code.slice(i, j));
      i = j;
      continue;
    }

    // Words — a keyword only when the whole word is one.
    if (isWordChar(ch)) {
      let j = i;
      while (j < code.length && isWordChar(code[j] as string)) j++;
      const word = code.slice(i, j);
      if (sx.keywords.has(word)) push("keyword", word);
      else plain += word;
      i = j;
      continue;
    }

    plain += ch;
    i++;
  }
  flush();
  return out;
}
