// client/src/cell/conn.ts — which llama.master we are talking to, and how it is.
//
// The client OBSERVES a server; it never operates one. Everything here is a
// GET: `/props` to learn what is loaded, `/health` to see whether it is up,
// `/metrics` or `/slots` to see how busy it is. There is no start, no stop, no
// settings — those belong to the machine llama.master runs on, and a client
// that could reach across the network and stop somebody else's model would be
// a different and much worse product.
//
// Browser-safe: every network call is behind `await import("./conn.server.ts")`
// inside a method (dep/aio/docs/build/imports.md).

import { cell } from "aio";
import { afterSweep, baseUrl, hasPort, KNOWN_PORTS } from "../lib/discover.ts";
import {
  NO_OCCUPANCY,
  type Occupancy,
  parseMetrics,
  parseProps,
  parseSlots,
  type ServerInfo,
} from "../lib/server.ts";

export type ConnStatus =
  /** Nothing entered, nothing tried. */
  | "idle"
  /** A sweep is running. */
  | "discovering"
  /** A connect is in flight. */
  | "connecting"
  /** `/props` answered: there is a llama-server there. */
  | "connected"
  /** It answered once and has stopped answering. */
  | "lost"
  /** It was tried and did not answer. */
  | "unreachable";

export type Found = { url: string; model: string };

export type ConnState = {
  /** What the user typed, exactly as typed — never rewritten under them. */
  host: string;
  port: number;
  /** The normalised base URL currently in use. Empty until a connect. */
  url: string;
  status: ConnStatus;
  /** `/props`, parsed. Null until a server answers. */
  info: ServerInfo | null;
  /** Is the model loaded and ready to answer, per `/health`? */
  ready: boolean;
  healthDetail: string;
  occupancy: Occupancy;
  /** Tokens/second measured by THIS client on its last reply. 0 = never. */
  measuredTps: number;
  /** Servers a sweep found, so one click connects to one of them. */
  found: Found[];
  scanning: boolean;
  progress: { done: number; total: number } | null;
  lastError: string;
};

export const conn = cell("conn", {
  // aiol: pre-alpha52 behavior pinned — remove to adopt transactions (s.$commit/s.$live)
  transaction: false,
  // The address is worth keeping between sessions — it is the one thing the
  // user typed. Nothing else here survives a restart: a status, a health
  // reading or an occupancy from last week would all be fiction.
  persist: { include: ["host", "port"] },
  state: {
    host: "",
    port: KNOWN_PORTS[0] ?? 18080,
    url: "",
    status: "idle" as ConnStatus,
    info: null as ServerInfo | null,
    ready: false,
    healthDetail: "",
    occupancy: NO_OCCUPANCY as Occupancy,
    measuredTps: 0,
    found: [] as Found[],
    scanning: false,
    progress: null as { done: number; total: number } | null,
    lastError: "",
  } as ConnState,
  // No `cancelOn` for `discover`: a second one while a sweep runs is refused
  // by the `scanning` guard (and the button is disabled), and the sweep reads
  // no signal — a `["self"]` entry here promised a cancellation that did not
  // happen.
  methods: {
    // Both coerce rather than trust. The address is the one field a dispatch
    // can reach from outside the UI — `am dispatch conn:setHost …`, a script, a
    // future deep link — and an object written into it persists (`host` is in
    // the persist list), so the app reopens with `[object Object]` in the box
    // and no way to connect. Measured, not imagined: a malformed `am` payload
    // put exactly that there.
    setHost(s, host: string) {
      s.host = typeof host === "string" ? host : "";
      s.lastError = "";
    },
    // An invalid port keeps the previous one: 0 is not a port, and writing it
    // (a cleared box once did) turned the next Connect into a request to :80.
    setPort(s, port: number) {
      const n = typeof port === "number" ? port : Number(port);
      if (Number.isInteger(n) && n >= 1 && n <= 65535) s.port = n;
      s.lastError = "";
    },

    /**
     * Connect to what the user typed (or clicked in the found list).
     *
     * "Connect" is a single `/props` GET: there is no session to open, and
     * pretending otherwise would only add a state that can be wrong. What it
     * really does is decide which base URL every later poll and every message
     * goes to.
     */
    async connect(s, urlOverride = "") {
      const target = urlOverride ||
        baseUrl(s.host, hasPort(s.host) ? undefined : s.port);
      if (!target) {
        s.lastError =
          "Type an address first — an IP or a hostname, with a port if it is not the usual one.";
        return;
      }
      s.status = "connecting";
      s.lastError = "";
      s.url = target;
      const io = await import("./conn.server.ts");
      // aiol-ok: `target` was read before the await and everything below writes
      // the RESULT of probing it. The one thing re-read is `url`: a later
      // connect (or a forget) that moved it while this probe was in the air
      // owns the state now, and a slow answer from the old address must not
      // be written under the new one.
      const props = await io.probe(target, 2500);
      if (s.url !== target) return; // aiol-ok: superseded — see above
      if (!props) {
        s.status = "unreachable";
        s.info = null;
        s.ready = false;
        // The commonest cause by far, and the one the user cannot guess: a
        // llama-server bound to 127.0.0.1 is invisible from every other
        // machine, and llama.cpp binds there by default.
        // aiol-ok: the write below is the result of the probe above
        s.lastError =
          `No llama.cpp server answered at ${target}. If it is running there, check that it was started with --host 0.0.0.0 — bound to 127.0.0.1 it is only reachable from its own machine.`;
        return;
      }
      s.info = parseProps(props);
      s.status = "connected";
      // Fill the fields in, so a connection made from the found list is one the
      // user can see and edit afterwards.
      const u = new URL(target);
      s.host = u.hostname;
      // aiol-ok: the fallback is the port this connect was made with
      s.port = Number(u.port) || s.port;
    },

    /** Stop watching. Deliberately not "disconnect from the server" — there is
     *  nothing on the far end that knows we were here. */
    forget(s) {
      s.url = "";
      s.status = "idle";
      s.info = null;
      s.ready = false;
      s.healthDetail = "";
      s.occupancy = NO_OCCUPANCY;
      s.lastError = "";
    },

    /**
     * Look for servers on this machine's own subnets.
     *
     * Progress is written as it goes, because a sweep is the one thing here
     * that takes seconds and a button that goes quiet for three of them reads
     * as broken.
     */
    async discover(s) {
      if (s.scanning) return;
      // What the sweep interrupts, so it can be put back. A sweep changes which
      // servers are KNOWN, not which one is in use: leaving "discovering" behind
      // disabled the chat against a server that was still connected.
      const before = s.status;
      const wasConnected = s.url !== "";
      s.scanning = true;
      s.status = "discovering";
      s.found = [];
      s.progress = { done: 0, total: 0 };
      s.lastError = "";
      try {
        const io = await import("./conn.server.ts");
        // One interface list for both the candidates and the de-duplication:
        // this machine answers on 127.0.0.1 AND on each of its own addresses.
        const ifaces = io.interfaces();
        const hits = await io.sweep(
          ifaces,
          (done, total) => {
            s.progress = { done, total }; // aiol-ok: progress IS the state here
          },
        );
        const found = hits.map((h) => ({
          url: h.url,
          model: parseProps(h.props)?.model ?? "unknown model",
        }));
        s.found = found;
        // `url`/`status` are re-read on purpose: a connect made by hand during
        // the sweep wins over anything the sweep would do.
        // aiol-ok: re-read on purpose — see above
        const urlNow = s.url;
        // aiol-ok: re-read on purpose — see above
        const stillDiscovering = s.status === "discovering";
        const verdict = afterSweep({
          wasConnected,
          urlNow,
          stillDiscovering,
          found,
        });
        if (found.length === 0) {
          s.lastError =
            "Nothing answered on this network. A llama.cpp server is only reachable from other machines when it was started with --host 0.0.0.0; otherwise type the address and port by hand.";
        }
        if (typeof verdict === "object") {
          // aiol-ok — a second dispatch, and safe because the URL travels as an
          // ARGUMENT: `connect` reads `s.host`/`s.port` only when it is given
          // no override, so there is nothing of this method's writes for it to
          // read stale. Its own action is also what the user wants here — the
          // connect shows up in the journal as a connect.
          await conn.connect(verdict.connect);
        } else if (verdict === "restore") {
          restore(s, before);
        }
      } catch (e) {
        s.lastError = String(e);
        restore(s, before);
      } finally {
        s.scanning = false;
        s.progress = null;
      }
    },

    /**
     * One sample of the far end: is it up, what is loaded, how busy is it.
     *
     * The single writer of liveness, on a schedule — the same rule the server
     * app keeps for its own process. Cheap and silent when nothing is
     * connected.
     */
    async poll(s) {
      const url = s.url;
      if (!url || s.status === "connecting" || s.status === "discovering") {
        return;
      }
      const io = await import("./conn.server.ts");
      // aiol-ok: this method IS the observer of state that changes underneath
      // it — reading the freshest status after each await is its whole job.
      const health = await io.json(url, "/health", 1500);
      // aiol-ok: after every await — a connect or forget that moved `url`
      // meanwhile owns the state, and this sample describes the old server.
      if (s.url !== url) return;
      const alive = health !== null;
      const detail = alive
        ? String((health as Record<string, unknown>)?.status ?? "ok")
        : "no answer";

      if (!alive) {
        // Lost, not unreachable: it answered once. The difference is what the
        // user should do about it — wait, rather than re-check the address.
        // aiol-ok: this method IS the observer of state that changes under it
        s.status = s.status === "connected" || s.status === "lost"
          ? "lost"
          // aiol-ok: this method IS the observer of state that changes
          : s.status;
        s.ready = false;
        s.healthDetail = detail;
        s.occupancy = NO_OCCUPANCY;
        return;
      }

      s.status = "connected";
      s.ready = detail === "ok";
      s.healthDetail = detail;

      const props = await io.probe(url, 1500);
      if (s.url !== url) return; // aiol-ok — as above
      if (props) s.info = parseProps(props);

      // Metrics first, slots second, nothing third — and the UI says which.
      const metrics = await io.text(url, "/metrics", 1500);
      if (s.url !== url) return; // aiol-ok — as above
      if (metrics) {
        s.occupancy = parseMetrics(metrics);
      } else {
        const slots = await io.json(url, "/slots", 1500);
        if (s.url !== url) return; // aiol-ok — as above
        s.occupancy = slots ? parseSlots(slots) : NO_OCCUPANCY;
      }
    },

    /** Record a rate this client actually achieved. Measured beats the
     *  server's own average, and the UI labels which one it is showing. */
    recordTps(s, tps: number) {
      if (Number.isFinite(tps) && tps > 0) s.measuredTps = tps;
    },

    clearError(s) {
      s.lastError = "";
    },
  },
  selectors: {
    /** The rate to quote, and where it came from. */
    rate: (s): { tps: number; measured: boolean } =>
      s.measuredTps > 0
        ? { tps: s.measuredTps, measured: true }
        : { tps: s.occupancy.tps ?? 0, measured: false },
    /** Can a message be sent right now? */
    usable: (s): boolean => s.status === "connected" && s.ready,
  },
});

/** Put back the status a sweep interrupted — unless something else (a connect
 *  made by hand, a forget) already replaced "discovering", in which case that
 *  is the truth and the sweep has no business overwriting it. */
function restore(s: ConnState, before: ConnStatus): void {
  if (s.status !== "discovering") return;
  s.status = before === "discovering" ? "idle" : before;
}
