// client/src/ui/Message.tsx — one message, in blocks.
//
// The same renderer the server app grew, over the same pure splitter
// (`src/lib/richtext.ts`): a reply is text and fenced blocks, each block names
// its file or language and carries its own copy button, and tok/s goes UNDER
// the answer, where a measurement of the answer belongs. Sharing the splitter
// rather than the component is deliberate — the rule ("where does this fence
// end") is one implementation; the markup belongs to each app's own layout.

import { highlight } from "../lib/highlight.ts";
import { replyBlocks } from "../shared/richtext.ts";
import type { Block } from "../shared/richtext.ts";
import { tps as fmtTps } from "../shared/format.ts";
import { CopyButton } from "./kit.tsx";

function CodeBlock(props: { block: Block }) {
  const b = props.block;
  if (b.kind !== "code") return null;
  const name = b.file || b.lang || "code";
  return (
    <div class="codeblock" t="codeblock">
      <div class="codeblock-head">
        <span class="codeblock-name" title={name}>{name}</span>
        {b.file && b.lang ? <span class="codeblock-lang">{b.lang}</span> : null}
        {b.open ? <span class="codeblock-live">writing…</span> : null}
        <CopyButton text={b.text} title={`Copy ${name}`} t="codeblock-copy" />
      </div>
      {
        /* Coloured by a single left-to-right scan (`lib/highlight.ts`) — four
           distinctions (comment, string, number, keyword) are what makes code
           readable at a glance, and they cost no dependency and no bundle. The
           text is never rewritten: the tokens concatenate back to the input. */
      }
      <pre class="codeblock-body"><code>{highlight(b.text, b.lang || b.file.split(".").pop() || "").map((t, i) =>
        t.kind === "plain"
          ? t.text
          : <span class={`tok-${t.kind}`} key={String(i)}>{t.text}</span>
      )}</code></pre>
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

/** A thinking model's first act: visible so "thinking" is never a spinner over
 *  nothing, folded so it never drowns the answer. */
function Thinking(props: { text?: string; live?: boolean }) {
  if (!props.text) return null;
  return (
    <details class="msg-think" open={props.live} t="msg-think">
      <summary>{props.live ? "thinking…" : "thought first"}</summary>
      <div class="msg-think-body">{props.text}</div>
    </details>
  );
}

export function Message(props: {
  role: string;
  content: string;
  thinking?: string;
  live?: boolean;
  tps?: number;
}) {
  const blocks = replyBlocks(props.content);
  return (
    <div class={`msg msg-${props.role}`}>
      <div class="msg-role">
        {props.role}
        {
          /* The whole answer, one button. The code blocks have their own, and
             the header has one for the conversation — this is the one for the
             thing the user actually asked for. Hidden while the reply is still
             arriving: copying half an answer is not what anyone means. */
        }
        {props.content && !props.live
          ? (
            <CopyButton
              text={props.content}
              title={`Copy this ${
                props.role === "user" ? "message" : "answer"
              }`}
              t="msg-copy"
            />
          )
          : null}
      </div>
      <Thinking text={props.thinking} live={props.live && !props.content} />
      <div class="msg-body">
        {blocks.length > 0
          ? blocks.map((b, i) => b.kind === "code"
            ? <CodeBlock block={b} key={String(i)} />
            : <TextBlock block={b} key={String(i)} />
          )
          : props.thinking && !props.live
          ? "(the reply ended while still thinking — its reasoning is above)"
          : ""}
      </div>
      {props.tps ? <div class="msg-meta">{fmtTps(props.tps)} tok/s</div> : null}
    </div>
  );
}
