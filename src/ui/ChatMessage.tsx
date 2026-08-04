// src/ui/ChatMessage.tsx — one message, drawn once.
//
// Both chat surfaces (the Chat tab and the all-in-one column) render this, for
// the same reason `CtxControls` and `ReserveControls` are shared: two copies of
// a message renderer drift, and the one that drifts is always the one the user
// is looking at. It shipped drifted — the all-in-one page put tok/s in the role
// line above the answer, the Chat tab put it under, and only one of them had
// the "ended while still thinking" fallback.
//
// Presentational: the blocks come from `src/lib/richtext.ts`, which is where
// the question "where does this fence end" is answered and tested.

import { Thinking } from "./kit.tsx";
import { CopyButton } from "./kit.tsx";
import { replyBlocks } from "../lib/richtext.ts";
import type { Block } from "../lib/richtext.ts";
import { tps as fmtTps } from "../lib/format.ts";

/**
 * A fenced block: what it is, a way to take it, and the code itself.
 *
 * The header names the FILE when the fence named one and the language
 * otherwise — "src/lib/plan.ts" tells the reader where this goes and
 * "typescript" does not. The copy button is here rather than on the message
 * because this is the unit people actually want: a 200-line reply usually
 * contains one file, and taking it should not mean selecting around the prose.
 */
function CodeBlock(props: { block: Block }) {
  const b = props.block;
  if (b.kind !== "code") return null;
  const name = b.file || b.lang || "code";
  return (
    <div class="codeblock" t="codeblock">
      <div class="codeblock-head">
        <span class="codeblock-name" title={name}>{name}</span>
        {b.file && b.lang ? <span class="codeblock-lang">{b.lang}</span> : null}
        {
          /* An unterminated fence is a block still arriving, and saying so is
             better than a header that looks final over half a file. */
        }
        {b.open ? <span class="codeblock-live">writing…</span> : null}
        <CopyButton text={b.text} title={`Copy ${name}`} t="codeblock-copy" />
      </div>
      <pre class="codeblock-body"><code>{b.text}</code></pre>
    </div>
  );
}

function TextBlock(props: { block: Block }) {
  const b = props.block;
  if (b.kind !== "text") return null;
  return (
    <div class="msg-text">
      {b.parts.map((p, i) =>
        p.code
          ? <code class="inline-code" key={String(i)}>{p.text}</code>
          : p.text
      )}
    </div>
  );
}

export function ChatMessage(props: {
  role: string;
  content: string;
  thinking?: string;
  /** The reply is still streaming — keeps the reasoning open while it is all
   *  there is, and stops an empty answer being reported as a lost one. */
  live?: boolean;
  tps?: number;
}) {
  const blocks = replyBlocks(props.content);
  return (
    <div class={`msg msg-${props.role}`}>
      <div class="msg-role">{props.role}</div>
      <Thinking text={props.thinking} live={props.live && !props.content} />
      <div class="msg-body">
        {blocks.length > 0
          ? blocks.map((b, i) =>
            b.kind === "code"
              ? <CodeBlock block={b} key={String(i)} />
              : <TextBlock block={b} key={String(i)} />
          )
          : props.thinking && !props.live
          ? "(the reply ended while still thinking — its reasoning is above)"
          : ""}
      </div>
      {
        /* At the END of the answer, where a measurement of the answer belongs.
           It used to head the message on the all-in-one page, which put a
           number the user cannot have yet above the text they are waiting for —
           and pushed the first line of every reply down a row for it. */
      }
      {props.tps ? <div class="msg-meta">{fmtTps(props.tps)} tok/s</div> : null}
    </div>
  );
}
