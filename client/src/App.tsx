// client/src/App.tsx — one screen: find a server, see what it is doing, talk to it.
//
// The whole app is a single column, in the order the questions are actually
// asked: WHICH server (the connect bar), WHAT is it doing (the status strip),
// and then the conversation, which takes every remaining pixel because it is
// the reason the app exists.
//
// Reads cell state directly; AIR subscribes per component, so the status strip
// re-renders every second without the chat below it doing any work.

import { afterRender } from "aio/air";
import { chat } from "./cell/chat.ts";
import { conn } from "./cell/conn.ts";
import { ui } from "./cell/ui.ts";
import { KNOWN_PORTS } from "./lib/discover.ts";
import {
  busyFraction,
  busyLabel,
  replySeconds,
  seconds,
  TYPICAL_REPLY_TOKENS,
} from "./lib/server.ts";
import { duration, tps as fmtTps } from "./shared/format.ts";
import { transcript } from "./shared/richtext.ts";
import { useStickyBottom } from "./ui/sticky.ts";
import {
  CopyButton,
  Empty,
  ErrorNote,
  KV,
  Meter,
  Pill,
  Waiting,
} from "./ui/kit.tsx";
import { Message } from "./ui/Message.tsx";

const STATUS: Record<
  string,
  { label: string; tone: "ok" | "warn" | "bad" | "idle" | "busy" }
> = {
  idle: { label: "not connected", tone: "idle" },
  discovering: { label: "looking…", tone: "busy" },
  connecting: { label: "connecting…", tone: "busy" },
  connected: { label: "connected", tone: "ok" },
  lost: { label: "lost", tone: "warn" },
  unreachable: { label: "unreachable", tone: "bad" },
};

/** Which server, and how to find one. */
function ConnectBar() {
  const st = STATUS[conn.status] ?? STATUS.idle!;
  return (
    <header class="bar" t="connect-bar">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">◆</span>
        <b>llama.master</b>
        <span class="dim">client</span>
      </div>

      <input
        class="host"
        t="host"
        aria-label="Server address"
        placeholder="192.168.1.20 — or a hostname"
        value={conn.host}
        onInput={(e) =>
          conn.setHost((e.currentTarget as HTMLInputElement).value)}
        onKeyDown={(e) => {
          if ((e as KeyboardEvent).key === "Enter") void conn.connect();
        }}
      />
      <input
        type="number"
        class="port"
        t="port"
        aria-label="Port"
        min="1"
        max="65535"
        value={String(conn.port)}
        onInput={(e) =>
          conn.setPort(Number((e.currentTarget as HTMLInputElement).value))}
      />
      <button
        type="button"
        class="btn primary"
        t="connect"
        disabled={conn.scanning}
        onClick={() => void conn.connect()}
      >
        Connect
      </button>
      {
        /* Discovery is a sweep of this machine's own subnets — llama-server
           does not announce itself, so there is nothing to listen for. The
           button says what it is doing while it does it. */
      }
      <button
        type="button"
        class="btn"
        t="discover"
        disabled={conn.scanning}
        title={`Look for a llama.cpp server on this network, on ports ${
          KNOWN_PORTS.join(", ")
        }`}
        onClick={() => void conn.discover()}
      >
        {conn.scanning
          ? `Looking… ${conn.progress?.done ?? 0}/${conn.progress?.total ?? 0}`
          : "Discover"}
      </button>
      <Pill tone={st.tone} title={conn.healthDetail} t="status">
        {st.label}
      </Pill>
      {
        /* The reader's own two settings, at the end of the bar and out of the
           way of the one the app is for. Text size first: someone who needs it
           larger needs it before anything else on the screen. */
      }
      <span class="looks">
        <button
          type="button"
          class="btn tiny"
          t="font-smaller"
          title="Smaller text"
          aria-label="Smaller text"
          disabled={!ui.canShrink()}
          onClick={() => ui.zoom(-1)}
        >
          A−
        </button>
        <button
          type="button"
          class="btn tiny"
          t="font-bigger"
          title="Larger text"
          aria-label="Larger text"
          disabled={!ui.canGrow()}
          onClick={() => ui.zoom(1)}
        >
          A+
        </button>
        <button
          type="button"
          class="btn tiny"
          t="theme"
          title={ui.theme === "dark" ? "Light theme" : "Dark theme"}
          aria-label={ui.theme === "dark" ? "Light theme" : "Dark theme"}
          onClick={() => ui.toggleTheme()}
        >
          {ui.theme === "dark" ? "☀" : "☾"}
        </button>
      </span>
    </header>
  );
}

/** More than one answer on the network: pick one. */
function Found() {
  if (conn.found.length < 2) return null;
  return (
    <div class="found" t="found">
      <span class="dim">{conn.found.length} servers answered:</span>
      {conn.found.map((f) => (
        <button
          key={f.url}
          type="button"
          class={`btn tiny${conn.url === f.url ? " on" : ""}`}
          onClick={() => void conn.connect(f.url)}
          title={`${f.url} — ${f.model}`}
        >
          {f.url.replace(/^https?:\/\//, "")} <span class="dim">{f.model}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * What the far end is doing: the model, how busy, how fast, how long a reply
 * should take.
 *
 * Every figure here is either measured or labelled as an estimate, and one that
 * the server does not report says "not reported" rather than showing a zero —
 * a client that invents a 0% is worse than one that admits it cannot see.
 */
function StatusStrip() {
  const info = conn.info;
  if (!conn.url) return null;
  const rate = conn.rate();
  const busy = busyFraction(conn.occupancy, info?.slots ?? 1);
  const eta = replySeconds(rate.tps, conn.occupancy.queued ?? 0);
  return (
    <section class="strip" t="strip">
      <KV
        k="Model"
        t="model"
        v={info?.model || (conn.status === "connected" ? "none loaded" : "—")}
        tip={info?.modelPath || "What llama.master has loaded right now"}
      />
      <KV
        k="Context"
        v={info?.ctx ? info.ctx.toLocaleString() : "—"}
        tip="What the server actually allocated, not what was asked for"
      />
      <div class="kv" t="busy">
        <span class="kv-k">Busy</span>
        <span class="kv-v">
          {busyLabel(conn.occupancy, info?.slots ?? 1)}
          {conn.occupancy.source === "none"
            ? (
              <span
                class="dim"
                title="Start llama-server with --metrics (or --slots) and this becomes a live reading."
              >
                {" "}(server does not publish it)
              </span>
            )
            : null}
        </span>
        <Meter
          value={busy}
          tone={busy !== null && busy > 0.75 ? "warn" : "busy"}
          title={busyLabel(conn.occupancy, info?.slots ?? 1)}
        />
      </div>
      <KV
        k="Speed"
        t="speed"
        v={rate.tps > 0
          ? (
            <>
              {fmtTps(rate.tps)} <span class="dim">tok/s</span>
              <span class="dim">
                {rate.measured ? " measured here" : " server average"}
              </span>
            </>
          )
          : "—"}
        tip={rate.measured
          ? "From the last reply this client received"
          : "Reported by the server across everyone using it"}
      />
      <KV
        k="A reply of ~256 tokens"
        t="eta"
        v={eta > 0 ? `≈ ${seconds(eta)}` : "—"}
        tip={`${TYPICAL_REPLY_TOKENS} tokens at the rate above${
          conn.occupancy.queued ? ", including the queue ahead of you" : ""
        }. An estimate, not a promise.`}
      />
      {chat.lastLatencyMs > 0
        ? (
          <KV
            k="First token"
            t="latency"
            v={duration(chat.lastLatencyMs)}
            tip="Send to first token on the last reply — prompt processing plus the network"
          />
        )
        : null}
    </section>
  );
}

function Chat() {
  const log = useStickyBottom(chat.messages.length);
  const ready = conn.usable();
  return (
    <section class="chat" t="chat">
      <div class="chat-head">
        <b>Chat</b>
        <input
          class="system"
          t="system"
          placeholder="System prompt (optional)"
          aria-label="System prompt"
          value={chat.system}
          onInput={(e) =>
            chat.setSystem((e.currentTarget as HTMLInputElement).value)}
        />
        <span class="spacer" />
        {chat.lastTps > 0
          ? <Pill tone="idle">{fmtTps(chat.lastTps)} tok/s</Pill>
          : null}
        <CopyButton
          text={transcript({
            system: chat.system,
            messages: chat.messages,
            partial: chat.partial,
            partialThink: chat.partialThink,
          })}
          title="Copy the whole conversation as markdown"
          t="chat-copy"
        />
        <button
          type="button"
          class="btn tiny"
          t="chat-clear"
          disabled={chat.messages.length === 0 && !chat.partial}
          onClick={() => chat.clear()}
        >
          Clear
        </button>
      </div>

      <div class="chat-log" t="chat-log" ref={log}>
        {chat.messages.length === 0 && !chat.partial && !chat.streaming
          ? (
            <Empty
              icon="✉"
              title={ready
                ? "Say something"
                : "Connect to a llama.master first"}
              hint={ready
                ? "This talks to /v1/chat/completions on the server above."
                : "Press Discover, or type its address and press Connect."}
            />
          )
          : (
            <>
              {chat.messages.map((m, i) => (
                <Message
                  key={String(i)}
                  role={m.role}
                  content={m.content}
                  thinking={m.thinking}
                  tps={m.tps}
                />
              ))}
              {chat.partial || chat.partialThink
                ? (
                  <Message
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
          void chat.send(conn.url);
        }}
      >
        <input
          t="message"
          placeholder={ready
            ? "Message"
            : conn.url
            ? "The server is not ready to answer yet"
            : "Not connected"}
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
              t="stop"
              onClick={() => chat.stop()}
            >
              Stop
            </button>
          )
          : (
            <button
              type="submit"
              class="btn primary"
              t="send"
              disabled={!ready || !chat.canSend()}
            >
              Send
            </button>
          )}
      </form>
    </section>
  );
}

export default function App() {
  // The rate the server reports is an average over everyone; the rate this
  // client measured is the one the user just lived through. Handing the
  // measurement to `conn` keeps both in one place, and the strip says which
  // it is showing.
  afterRender(() => {
    if (chat.lastTps > 0 && chat.lastTps !== conn.measuredTps) {
      conn.recordTps(chat.lastTps);
    }
  });
  // The theme is on this element AND on the document. On the element because
  // that is what renders — server-side, in a test harness, and on the first
  // frame before any effect has run. On the document because the page BEHIND
  // the app has to change with it (the scrollbar trough, the overscroll), and
  // `:root` is the only place a stylesheet can see that. Guarded: there is no
  // `document` in a headless render, and an effect that throws there takes the
  // whole render with it — which is how the theme button briefly stopped
  // existing at all.
  afterRender(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    if (root.dataset.theme !== ui.theme) root.dataset.theme = ui.theme;
  });
  return (
    <div
      class="app"
      t="app"
      data-theme={ui.theme}
      style={{ "--fs": `${ui.fontPx}px` }}
    >
      <ConnectBar />
      <Found />
      <ErrorNote
        message={conn.lastError || chat.lastError}
        onDismiss={() => conn.clearError()}
      />
      <StatusStrip />
      <Chat />
    </div>
  );
}
