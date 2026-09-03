// client/src/ui/Composer.tsx — the input row, and the messages waiting behind it.
//
// The same component as llama.master's `ChatComposer`, for the same reason and
// with the same rules — a message written while the far end is answering is
// held rather than dropped, and the queue drains itself one reply at a time.
// The POLICY is shared code (`../shared/queue.ts`, synced by `deno task sync`),
// so the two apps cannot disagree about what Enter does; only the markup lives
// here, because this app has its own stylesheet and its own kit.
//
// What it draws, in the order the eye needs it:
//
//   [ what is waiting, and whether anything will collect it ]
//   [ ✕ first queued ] [ ✕ second queued ]        (oldest first — send order)
//   [ input                          ] [ Queue ] [ Stop ]
//
// The queue sits ABOVE the input rather than inside the log: nothing in it has
// been said to the server yet, and drawing it among the messages would claim
// otherwise.
//
// The text being typed lives HERE, in a browser-local signal, never on the
// wire. As a cell field (one dispatch per keystroke) the box was a controlled
// input over a replicated value: every key was a round trip, and each `partial`
// flush of a streaming reply re-rendered it with the copy the server had last
// acknowledged, wiping keys still in flight. A draft is per-window state no
// other client or restart needs.

import { signal } from "aio/air";
import { chat } from "../cell/chat.ts";
import {
  draftRows,
  queueLabel,
  queueNote,
  submitKind,
} from "../shared/queue.ts";

/** The draft, per window. */
const draft = signal("");

/** What the submit button says — "" when there is nothing it can do. The word
 *  has to change, because the two gestures have different consequences: Send
 *  starts a request, Queue writes a note for later. */
function submitLabel(text: string): string {
  switch (submitKind(text, chat.queue.length, chat.streaming)) {
    case "queue":
      return "Queue";
    case "send":
      return "Send";
    // Full is a live button that would drop the message, which is worse than a
    // dead one — the note above the input carries the reason.
    case "full":
    case null:
      return "";
  }
}

export function Composer(props: { url: string; ready: boolean }) {
  const { url, ready } = props;
  const text = draft.value;
  const label = submitLabel(text);
  const queue = chat.queue.slice(); // `.slice()` — a live async state array

  /** Enter and the button share this. The box is cleared before the dispatch
   *  resolves — `submit` runs the whole reply when it is the request — and put
   *  back only if the far end refused the text (a full queue, which the note
   *  above names). */
  const submit = () => {
    if (!ready) return;
    const t = draft.peek();
    if (submitKind(t, chat.queue.length, chat.streaming) === null) return;
    draft.set("");
    void chat.submit(url, t).then((taken) => {
      if (!taken && draft.peek() === "") draft.set(t);
    });
  };

  return (
    <>
      {queue.length > 0
        ? (
          <div class="chat-queue" t="chat-queue">
            <div class="chat-queue-head">
              <span class="chat-queue-note">
                {queueNote(chat.queue.length, chat.streaming)}
              </span>
              <button
                type="button"
                class="btn tiny"
                t="chat-queue-clear"
                title="Discard every waiting message"
                onClick={() => chat.clearQueue()}
              >
                Clear queue
              </button>
            </div>
            <div class="chat-queue-items">
              {queue.map((text, i) => (
                <span class="chat-chip" key={String(i)} title={text}>
                  <span class="chat-chip-n">{i + 1}</span>
                  <span class="chat-chip-text">{queueLabel(text)}</span>
                  <button
                    type="button"
                    class="chat-chip-x"
                    aria-label={`Remove queued message ${i + 1}`}
                    title="Remove this message"
                    onClick={() =>
                      chat.unqueue(i)}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          </div>
        )
        : null}
      <form
        class="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          t="message"
          // NOT disabled while streaming — that is the whole feature. The only
          // thing that closes this box is a far end that cannot take the
          // message at all.
          // A textarea: what gets pasted into a chat with a model that writes
          // code is code. Enter sends, Shift+Enter breaks a line.
          placeholder={ready
            ? (chat.streaming
              ? "Message — waits until this reply finishes"
              : "Message  (Shift+Enter for a new line)")
            : props.url
            ? "The server is not ready to answer yet"
            : "Not connected"}
          aria-label="Message"
          disabled={!ready}
          rows={draftRows(text)}
          value={text}
          onInput={(e) =>
            draft.set((e.currentTarget as HTMLTextAreaElement).value)}
          onKeyDown={(e) => {
            const k = e as KeyboardEvent;
            // An IME confirming a character with Enter is not a send.
            if (k.key !== "Enter" || k.shiftKey || k.isComposing) return;
            k.preventDefault();
            submit();
          }}
        />
        <button
          type="submit"
          class={`btn ${chat.streaming ? "" : "primary"}`}
          t="send"
          disabled={!ready || label === ""}
          title={chat.streaming
            ? "Hold this message and send it when the reply finishes"
            : "Send"}
        >
          {
            /* The fallback is what the button WOULD do if there were something
              to do, so a disabled button still teaches the gesture. Reading
              "Send" over a live reply promised the wrong one. */
          }
          {label || (chat.streaming ? "Queue" : "Send")}
        </button>
        {chat.streaming
          ? (
            <button
              type="button"
              class="btn danger"
              t="stop"
              title="Stop this reply. Anything waiting stays waiting."
              onClick={() => chat.stop()}
            >
              Stop
            </button>
          )
          : null}
      </form>
    </>
  );
}
