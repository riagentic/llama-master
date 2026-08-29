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

import { chat } from "../cell/chat.ts";
import { queueLabel } from "../lib/queue.ts";
import { submitChat } from "./actions.ts";
import { chatQueue, chatQueueNote, submitLabel } from "./derive.ts";

export function ChatComposer(props: { url: string; ready: boolean }) {
  const { url, ready } = props;
  const queue = chatQueue();
  const note = chatQueueNote();
  const label = submitLabel();

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
          if (!ready) return;
          submitChat(url);
        }}
      >
        <input
          // NOT disabled while streaming — that is the whole feature. The only
          // thing that closes this box is a server that cannot take the message
          // at all.
          placeholder={ready
            ? (chat.streaming
              ? "Message — waits until this reply finishes"
              : "Message")
            : "Server is not running"}
          aria-label="Message"
          disabled={!ready}
          value={chat.input}
          onInput={(e) =>
            chat.setInput((e.currentTarget as HTMLInputElement).value)}
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
