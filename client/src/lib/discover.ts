// client/src/lib/discover.ts — where a llama.master might be, in what order.
//
// There is no announcement to listen for: llama.master starts a plain
// llama-server, and llama-server does not advertise itself on the network. So
// discovery is a SWEEP — every address on the subnets this machine is already
// on, against the handful of ports llama.cpp is served on — and the identifying
// answer is the server's own `/props`, which nothing else on a LAN replies to.
//
// Pure: addresses in, candidate URLs out, ordered. The probing itself is I/O
// and lives in `cell/conn.server.ts`; keeping the arithmetic here is what makes
// "does it try the right places, in the right order" a test rather than an
// opinion.

/** Ports worth trying, best first.
 *
 *  8080 is llama.cpp's own default and what `llama-server` uses when nothing
 *  says otherwise; 18080 is llama.master's, chosen so a hand-started
 *  llama.cpp on the same machine does not collide with the one the app runs.
 *  A port the user has typed always goes first — see `candidates`. */
export const KNOWN_PORTS: readonly number[] = [18080, 8080, 8081, 11434];

/** A /24 is 254 usable addresses; anything larger is not a sweep, it is a scan,
 *  and a client that scans a /16 on a button press is a client nobody runs
 *  twice. Bigger networks are reached by typing the address. */
export const MAX_SWEEP_HOSTS = 254;

/** An IPv4 address and its prefix, as reported by the host. */
export type Iface = { address: string; prefix: number };

/** Is this a private (RFC1918) or link-local address — somewhere a llama.master
 *  could plausibly live, and somewhere we are allowed to sweep? A client that
 *  probes public addresses is a port scanner. */
export function isPrivateV4(ip: string): boolean {
  const o = octets(ip);
  if (!o) return false;
  const [a, b] = o as [number, number, number, number];
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 169 && b === 254) return true; // link-local
  return false;
}

function octets(ip: string): number[] | null {
  const parts = ip.split(".");
  if (parts.length !== 4) return null;
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return null;
  return nums;
}

/**
 * Every address on the same /24 as `ip`, excluding the network, the broadcast
 * and `ip` itself.
 *
 * /24 whatever the real prefix says: a /16 is 65,534 probes and a /8 is
 * 16 million, and neither is a thing to do because someone pressed Discover.
 * The addresses a home or office llama.master actually sits on are on the same
 * /24 as the client in every case worth automating; the rest is what the manual
 * field is for.
 */
export function sweepHosts(ip: string): string[] {
  const o = octets(ip);
  if (!o || !isPrivateV4(ip)) return [];
  const [a, b, c, self] = o as [number, number, number, number];
  const out: string[] = [];
  for (let d = 1; d <= MAX_SWEEP_HOSTS; d++) {
    if (d === self) continue;
    out.push(`${a}.${b}.${c}.${d}`);
  }
  return out;
}

/** A base URL, normalised: scheme filled in, trailing slash removed, port
 *  applied when the text did not carry one. What the user typed is never used
 *  raw — "192.168.1.9" and "http://192.168.1.9:8080/" must reach one server. */
export function baseUrl(host: string, port?: number): string {
  const text = host.trim();
  if (!text) return "";
  const withScheme = /^https?:\/\//i.test(text) ? text : `http://${text}`;
  let u: URL;
  try {
    u = new URL(withScheme);
  } catch {
    return "";
  }
  if (!u.port && port && port > 0) u.port = String(port);
  return `${u.protocol}//${u.host}`;
}

/** Does this text already name a port? Then a port picker must not override it. */
export function hasPort(host: string): boolean {
  const text = host.trim();
  if (!text) return false;
  try {
    return new URL(/^https?:\/\//i.test(text) ? text : `http://${text}`)
      .port !==
      "";
  } catch {
    return false;
  }
}

/**
 * The addresses to probe, in the order to probe them.
 *
 * Order is the whole point: the machine the client is running on first (a
 * llama.master on this very box is the commonest case and answers in a
 * millisecond), then the rest of the subnet. Ports are tried per address so a
 * responsive host is identified early rather than after 254 timeouts.
 */
export function candidates(
  ifaces: readonly Iface[],
  ports: readonly number[] = KNOWN_PORTS,
): string[] {
  const hosts: string[] = ["127.0.0.1"];
  for (const i of ifaces) {
    if (!isPrivateV4(i.address)) continue;
    if (!hosts.includes(i.address)) hosts.push(i.address);
  }
  for (const i of ifaces) {
    if (!isPrivateV4(i.address)) continue;
    for (const h of sweepHosts(i.address)) {
      if (!hosts.includes(h)) hosts.push(h);
    }
  }
  const out: string[] = [];
  for (const h of hosts) {
    for (const p of ports) out.push(`http://${h}:${p}`);
  }
  return out;
}
