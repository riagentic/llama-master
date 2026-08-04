// src/lib/richtext.ts — what a model's reply is made of.
//
// A local model answers with markdown, and the two things it answers with most
// are code and file contents. Rendered as one pre-wrapped string those are a
// wall: the fences are visible, the indentation fights the paragraph text, and
// the one thing the user wants to do with a 40-line file — take it — costs a
// careful drag-select that catches the ``` at both ends.
//
// So the reply is split into blocks before it is drawn. Pure, and here rather
// than in the component, because "where does this fence end" is a rule with
// edge cases (nested fences, a block still streaming, an info string that names
// a file) and rules with edge cases belong in `src/lib` with tests around them.
//
// Deliberately NOT a markdown renderer. Headings, lists and emphasis stay
// literal — this is a test chat, the answer is the artefact, and a half-done
// markdown pipeline that swallows an asterisk in a code comment would be worse
// than none. Blocks and inline code spans are the whole scope.

import type { ChatMessage } from "./types.ts";

/** A run of message text: prose, or an inline `code` span. */
export type Chunk = { code: boolean; text: string };

export type Block =
  | { kind: "text"; text: string; parts: Chunk[] }
  | {
    kind: "code";
    /** The block's contents, fences removed and the fence's own indent undone. */
    text: string;
    /** The language, when the info string named one ("python", "ts"). */
    lang: string;
    /** The file, when the info string named one — the header shows this over
     *  the language, because "src/lib/plan.ts" tells the reader more than
     *  "typescript" and is what makes a copy button worth pressing. */
    file: string;
    /** The closing fence has not arrived. True only for the last block of a
     *  reply that is still streaming — the code has to be readable AS it
     *  arrives, so an unterminated fence renders as a block, not as raw text
     *  that reflows into one the moment the model finishes. */
    open: boolean;
  };

const FENCE = /^([ \t]{0,3})(`{3,}|~{3,})(.*)$/;

/**
 * Split a reply into blocks.
 *
 * One pass, line by line, tracking the open fence — the same algorithm
 * CommonMark uses for fenced code, minus the parts that only matter inside a
 * full parser: a closing fence must use the same character and be at least as
 * long as the opening one, so a ```` ```` ```` block may contain ``` and a
 * shell snippet full of backticks survives.
 */
export function replyBlocks(md: string): Block[] {
  const out: Block[] = [];
  if (!md) return out;
  const lines = md.split("\n");
  let text: string[] = [];
  let code: string[] | null = null;
  let marker = "";
  let indent = "";
  let info = "";

  const flushText = () => {
    const joined = text.join("\n");
    text = [];
    if (joined.trim() === "") return;
    out.push({ kind: "text", text: joined, parts: inlineChunks(joined) });
  };
  const flushCode = (open: boolean) => {
    const body = (code ?? []).join("\n");
    const named = parseInfo(info);
    code = null;
    out.push({ kind: "code", text: body, ...named, open });
  };

  for (const line of lines) {
    const m = FENCE.exec(line);
    if (code === null) {
      if (m) {
        flushText();
        indent = m[1] ?? "";
        marker = m[2] ?? "```";
        info = m[3] ?? "";
        code = [];
      } else {
        text.push(line);
      }
      continue;
    }
    // Inside a block: only a fence of the SAME character, at least as long, and
    // with nothing after it, closes it.
    const closes = m && (m[2] ?? "").startsWith(marker[0] ?? "`") &&
      (m[2] ?? "").length >= marker.length && (m[3] ?? "").trim() === "";
    if (closes) {
      flushCode(false);
      continue;
    }
    code.push(unindent(line, indent));
  }
  if (code !== null) flushCode(true);
  else flushText();
  return out;
}

/** Drop the fence's own indentation from a content line, and never more than
 *  that: the relative shape of the code is the code. */
function unindent(line: string, indent: string): string {
  let i = 0;
  while (i < indent.length && (line[i] === " " || line[i] === "\t")) i++;
  return line.slice(i);
}

/** Extensions worth naming. Only the ones this app's users actually paste —
 *  an unknown extension is shown as itself rather than guessed at. */
const EXT_LANG: Record<string, string> = {
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  jsx: "jsx",
  py: "python",
  rs: "rust",
  go: "go",
  c: "c",
  h: "c",
  cpp: "c++",
  cc: "c++",
  hpp: "c++",
  cu: "cuda",
  java: "java",
  kt: "kotlin",
  rb: "ruby",
  php: "php",
  cs: "c#",
  swift: "swift",
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  fish: "fish",
  ps1: "powershell",
  sql: "sql",
  json: "json",
  jsonc: "json",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  ini: "ini",
  xml: "xml",
  html: "html",
  css: "css",
  scss: "scss",
  md: "markdown",
  diff: "diff",
  patch: "diff",
  dockerfile: "dockerfile",
  make: "makefile",
  mk: "makefile",
  cmake: "cmake",
  gguf: "binary",
};

/** Does this token name a file rather than a language? A slash or a real
 *  extension; `Makefile` and `Dockerfile` by name, because they have neither. */
function looksLikeFile(token: string): boolean {
  if (token.includes("/") || token.includes("\\")) return true;
  if (/^(makefile|dockerfile|cmakelists\.txt)$/i.test(token)) return true;
  return /\.[A-Za-z0-9_+-]{1,12}$/.test(token);
}

const ATTR =
  /\b(?:title|file|filename|path|name)\s*=\s*("([^"]*)"|'([^']*)'|\S+)/i;

/**
 * Read a fence's info string.
 *
 * There is no standard for naming the file a block belongs to and every tool
 * invented its own, so all the common spellings are accepted — a model trained
 * on all of them will emit all of them:
 *
 *     ```ts                       language only
 *     ```src/lib/plan.ts          file only
 *     ```ts:src/lib/plan.ts       language:file
 *     ```ts title="src/plan.ts"   attribute (title/file/filename/path/name)
 *
 * The language is filled in from the extension when only a file was named, so
 * the header can still say what it is looking at.
 */
export function parseInfo(info: string): { lang: string; file: string } {
  const raw = info.trim();
  if (!raw) return { lang: "", file: "" };
  let lang = "";
  let file = "";

  // Attributes first, because a quoted value may contain the spaces the rest
  // of the parse tokenises on. What is left is the bare part.
  const attr = ATTR.exec(raw);
  if (attr) {
    file = (attr[2] ?? attr[3] ?? attr[1] ?? "").trim();
  }
  const bare = attr ? raw.replace(attr[0], " ") : raw;

  // Then every remaining token, each classified on its own: a file is a file
  // wherever it appears, and the first token that is not one is the language.
  // ```ts, ```ts src/lib/plan.ts and ```src/lib/plan.ts are all in the wild.
  for (const token of bare.split(/\s+/).filter(Boolean)) {
    const cut = token.indexOf(":");
    if (cut > 0 && looksLikeFile(token.slice(cut + 1))) {
      if (!lang) lang = token.slice(0, cut);
      if (!file) file = token.slice(cut + 1);
    } else if (looksLikeFile(token)) {
      if (!file) file = token;
    } else if (!lang) {
      lang = token;
    }
  }
  if (!lang && file) {
    const base = file.split(/[/\\]/).pop() ?? "";
    const ext = base.includes(".") ? base.split(".").pop() ?? "" : base;
    lang = EXT_LANG[ext.toLowerCase()] ?? "";
  }
  return { lang: lang.toLowerCase(), file };
}

/**
 * Split prose into plain runs and inline `code` spans.
 *
 * Single backticks only, never across a line: a lone backtick in a sentence is
 * common and must stay a backtick rather than swallowing the rest of the reply
 * into a code span that never closes.
 */
export function inlineChunks(text: string): Chunk[] {
  const out: Chunk[] = [];
  const re = /`([^`\n]+)`/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) {
      out.push({ code: false, text: text.slice(last, m.index) });
    }
    out.push({ code: true, text: m[1] ?? "" });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ code: false, text: text.slice(last) });
  return out.length ? out : [{ code: false, text }];
}

/**
 * The whole conversation as markdown, for the copy-chat button.
 *
 * Markdown because that is what the reply already is: pasting this into an
 * issue, a file or another chat keeps every code fence intact. Nothing on
 * screen is silently dropped — a model's reasoning is quoted rather than
 * omitted, because a transcript that leaves out half of what the model said is
 * a transcript nobody can rely on. The reply still streaming is included too:
 * the button copies what the user can see.
 */
export function transcript(t: {
  system?: string;
  messages: readonly ChatMessage[];
  partial?: string;
  partialThink?: string;
}): string {
  const out: string[] = [];
  if (t.system && t.system.trim()) {
    out.push(`### system\n\n${t.system.trim()}`);
  }
  for (const m of t.messages) {
    out.push(section(m.role, m.content, m.thinking, m.tps));
  }
  if ((t.partial ?? "").trim() || (t.partialThink ?? "").trim()) {
    out.push(section("assistant", t.partial ?? "", t.partialThink, 0));
  }
  return out.join("\n\n").trimEnd() + "\n";
}

function section(
  role: string,
  content: string,
  thinking?: string,
  tps?: number,
): string {
  const head = tps && tps > 0
    ? `### ${role} · ${tps.toFixed(1)} tok/s`
    : `### ${role}`;
  const parts = [head];
  if (thinking && thinking.trim()) {
    parts.push(
      `_thought first_\n\n${
        thinking.trim().split("\n").map((l) => `> ${l}`).join("\n")
      }`,
    );
  }
  if (content.trim()) parts.push(content.trim());
  return parts.join("\n\n");
}
