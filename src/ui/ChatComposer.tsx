// src/ui/ChatComposer.tsx — the input row, and the messages waiting behind it.
//
// ONE component for both chat surfaces, for the reason `ChatMessage` is one:
// two copies of an input row drift, and this one carries behaviour now rather
// than a text box. The Chat tab and the all-in-one page had already grown two
// copies of the same eight lines; adding a queue to each would have been the
// third and fourth.
//
// What it draws, in the order the eye needs it:
//
//   [ what is waiting, and whether anything will collect it ]
//   [ ✕ first queued ] [ ✕ second queued ]        (oldest first — send order)
//   [ input                          ] [ Queue ] [ Stop ]
//
// The queue is ABOVE the input rather than below the log, because it is not
// part of the conversation — nothing in it has been said to the model yet, and
// drawing it among the messages would claim otherwise.
//
// The text being typed lives HERE, in a browser-local signal, and nowhere on
// the wire. It was a cell field once (`chat.input`, one `setInput` dispatch per
// keystroke), which made the box a controlled input over a replicated server
// value: every key was a round trip, and while a reply streamed the 60–500 ms
// `partial` flushes re-rendered the box with whatever the server had last
// acknowledged — keys still in flight were wiped, and only while the model was
// answering. A draft is per-window state that no other client and no restart
// needs (dep/aio/docs/state/real-time.md: "if two clients or a restart never
// need to agree on it, it does not belong in a server cell").

import { signal } from "aio/air";
import { chat } from "../cell/chat.ts";
import { draftRows, queueLabel, submitKind } from "../lib/queue.ts";
import { submitChat } from "./actions.ts";
import { chatQueue, chatQueueNote, submitLabel } from "./derive.ts";

/** A queued message's key: its text, plus how many identical texts come
 *  before it — stable when earlier, different messages leave the queue. */
function chipKey(queue: readonly string[], i: number): string {
  const text = queue[i] ?? "";
  let n = 0;
  for (let k = 0; k < i; k++) if (queue[k] === text) n++;
  return `${n}\u0000${text}`;
}

/** The draft, per window. Module-level rather than `useLocal` so it survives a
 *  switch between the two surfaces that mount this — both are the same
 *  conversation, so the half-typed thought must follow. */
const draft = signal("");

export function ChatComposer(props: { url: string; ready: boolean }) {
  const { url, ready } = props;
  const text = draft.value;
  const queue = chatQueue();
  const note = chatQueueNote();
  const label = submitLabel(text);

  /** Enter and the button do the same thing, so they share it. The box is
   *  cleared BEFORE the dispatch resolves — `submit` runs the whole reply when
   *  it is the request — and put back only if the server refused the text,
   *  which it does for one reason (a full queue) that the note above names. */
  const submit = () => {
    if (!ready) return;
    const t = draft.peek();
    if (submitKind(t, chat.queue.length, chat.streaming) === null) return;
    draft.set("");
    void submitChat(url, t).then((taken) => {
      if (!taken && draft.peek() === "") draft.set(t);
    });
  };

  return (
    <>
      {queue.length > 0
        ? (
          <div class="chat-queue" t="chat-queue">
            <div class="chat-queue-head">
              <span class="chat-queue-note">{note}</span>
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
                // Keyed by CONTENT, not position: the drain shifts the head
                // off while the user looks at the rest, and a positional key
                // re-used the first chip's DOM for what had been the second.
                // `chipKey` disambiguates identical texts.
                <span class="chat-chip" key={chipKey(queue, i)} title={text}>
                  <span class="chat-chip-n">{i + 1}</span>
                  <span class="chat-chip-text">{queueLabel(text)}</span>
                  <button
                    type="button"
                    class="chat-chip-x"
                    aria-label={`Remove queued message ${i + 1}`}
                    title="Remove this message"
                    onClick={() =>
                      // The text travels with the index: the queue may have
                      // moved under this chip by the time the click lands
                      // (`chat.unqueue`).
                      chat.unqueue(i, text)}
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
          t="chat-message"
          // NOT disabled while streaming — that is the whole feature. The only
          // thing that closes this box is a server that cannot take the message
          // at all.
          //
          // A textarea, not an <input>: what gets pasted into a chat with a
          // model that writes code is code, and a one-line box turned every
          // pasted file into one line. Enter sends, Shift+Enter breaks a line —
          // the convention every chat client has taught.
          placeholder={ready
            ? (chat.streaming
              ? "Message — waits until this reply finishes"
              : "Message  (Shift+Enter for a new line)")
            : "Server is not running"}
          aria-label="Message"
          disabled={!ready}
          rows={draftRows(text)}
          value={text}
          onInput={(e) =>
            draft.set((e.currentTarget as HTMLTextAreaElement).value)}
          onKeyDown={(e) => {
            const k = e as KeyboardEvent;
            // `isComposing`: an IME confirming a character with Enter is not a
            // send, and treating it as one submits half a word.
            if (k.key !== "Enter" || k.shiftKey || k.isComposing) return;
            k.preventDefault();
            submit();
          }}
        />
        <button
          type="submit"
          class={`btn ${chat.streaming ? "" : "primary"}`}
          t="chat-submit"
          disabled={!ready || label === ""}
          title={chat.streaming
            ? "Hold this message and send it when the reply finishes"
            : "Send"}
        >
          {
            /* The fallback is what the button would do if there were something
              to do — so a disabled button still teaches the gesture. Reading
              "Send" over a live reply promised the wrong one. */
          }
          {label || (chat.streaming ? "Queue" : "Send")}
        </button>
        {chat.streaming
          ? (
            <button
              type="button"
              class="btn danger"
              t="chat-stop"
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
