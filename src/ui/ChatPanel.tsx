// src/ui/ChatPanel.tsx — talk to the server you just started.
//
// Deliberately minimal: this is a proof that the endpoint works with the
// settings on the Tune tab, not a chat product. It streams, it reports
// tokens/second, and it says plainly when the server is not up.

import { chat } from "../cell/chat.ts";
import { cfg } from "../cell/cfg.ts";
import { srv } from "../cell/srv.ts";
import { num } from "../lib/params.ts";
import { tps } from "../lib/format.ts";
import { endpoint } from "./actions.ts";
import { CopyButton, Empty, ErrorNote, Panel, Pill, Waiting } from "./kit.tsx";
import { ChatMessage } from "./ChatMessage.tsx";
import { canSend, chatHasContent, chatTranscript } from "./derive.ts";
import { useStickyBottom } from "./sticky.ts";

export function ChatPanel() {
  const ready = srv.status === "ready";
  const url = endpoint();
  const log = useStickyBottom(chat.messages.length);

  return (
    <div class="tab-body chat-tab">
      <Panel
        title="Test chat"
        icon="✉"
        wide
        right={
          <>
            <Pill tone={ready ? "ok" : "warn"} title={srv.healthDetail}>
              {ready ? url : "server not ready"}
            </Pill>
            {chat.lastTps > 0
              ? <Pill tone="idle">{tps(chat.lastTps)} tok/s</Pill>
              : null}
            <CopyButton
              text={chatTranscript()}
              title="Copy the whole conversation as markdown"
              label="Copy chat"
              t="chat-copy"
            />
            <button
              type="button"
              class="btn tiny"
              t="chat-clear"
              disabled={!chatHasContent()}
              onClick={() => chat.clear()}
            >
              Clear
            </button>
          </>
        }
      >
        <ErrorNote message={chat.lastError} />
        <input
          class="system"
          placeholder="System prompt (optional)"
          aria-label="System prompt"
          value={chat.system}
          onInput={(e) =>
            chat.setSystem((e.currentTarget as HTMLInputElement).value)}
        />
        <div class="chat-log" t="chat-log" ref={log}>
          {chat.messages.length === 0 && !chat.partial && !chat.streaming
            ? (
              <Empty
                icon="✉"
                title={ready ? "Say something" : "Start the server first"}
                hint={ready
                  ? "This talks to /v1/chat/completions on the running server."
                  : "The Server tab has the start button."}
              />
            )
            : (
              <>
                {chat.messages.map((m, i) => (
                  <ChatMessage
                    key={String(i)}
                    role={m.role}
                    content={m.content}
                    thinking={m.thinking}
                    tps={m.tps}
                  />
                ))}
                {chat.partial || chat.partialThink
                  ? (
                    <ChatMessage
                      role="assistant"
                      content={chat.partial}
                      thinking={chat.partialThink}
                      live
                    />
                  )
                  : null}
                {chat.streaming && !chat.partial && !chat.partialThink
                  ? <Waiting />
                  : null}
              </>
            )}
        </div>
        <form
          class="chat-input"
          onSubmit={(e) => {
            e.preventDefault();
            if (!ready) return;
            chat.send(url, {
              temp: num(cfg.settings, "temp"),
              topP: num(cfg.settings, "topP"),
            });
          }}
        >
          <input
            placeholder={ready ? "Message" : "Server is not running"}
            aria-label="Message"
            disabled={!ready}
            value={chat.input}
            onInput={(e) =>
              chat.setInput((e.currentTarget as HTMLInputElement).value)}
          />
          {chat.streaming
            ? (
              <button
                type="button"
                class="btn danger"
                onClick={() => chat.stop()}
              >
                Stop
              </button>
            )
            : (
              <button
                type="submit"
                class="btn primary"
                disabled={!ready || !canSend()}
              >
                Send
              </button>
            )}
        </form>
      </Panel>
    </div>
  );
}
