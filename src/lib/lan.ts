// src/lib/lan.ts — "can the rest of the house reach this model?"
//
// llama-server binds to 127.0.0.1 unless told otherwise, which means it is
// invisible from every other machine — the commonest reason a client on the
// sofa finds nothing (`client/`, whose failure message names this file's flag).
// The fix is one llama.cpp flag, `--host 0.0.0.0`, and the whole of this module
// is about making that flag a decision rather than a piece of trivia:
//
//   • it is OFF by default, because binding to the world is not a default
//   • llama-server has NO authentication unless an API key is set, so turning
//     it on means anyone on the network can use the model — and that has to be
//     said next to the switch, not buried in a tooltip
//   • when it is on, the address other machines actually dial is worth showing;
//     "0.0.0.0" is a bind address, not a destination, and typing it into a
//     client reaches nothing
//
// Pure: the decision and the address arithmetic. The interface list is I/O and
// comes from `hw.server.ts`.

import type { Settings } from "./types.ts";

/** Bind everywhere. What the switch writes into `--host`. */
export const LAN_HOST = "0.0.0.0";
/** Bind to this machine only — llama.cpp's default, and ours. */
export const LOOPBACK = "127.0.0.1";

/** Is this configuration reachable from another machine?
 *
 *  `0.0.0.0` and `::` are the wildcard binds; anything that is not a loopback
 *  literal is a specific interface, which is also reachable. Only 127.x and
 *  `localhost` keep it to this machine. */
export function isLanExposed(settings: Settings): boolean {
  const host = String(settings.host ?? LOOPBACK).trim();
  if (!host) return false;
  if (host === "localhost" || host.startsWith("127.") || host === "::1") {
    return false;
  }
  return true;
}

/** What `--host` should become when the switch is flipped. */
export function lanHost(on: boolean): string {
  return on ? LAN_HOST : LOOPBACK;
}

/** Is this address one a llama.master would be reached on — private, routable
 *  from the same network, and not this machine's own loopback? */
export function isReachableV4(ip: string): boolean {
  const p = ip.split(".").map(Number);
  if (
    p.length !== 4 || p.some((n) => !Number.isInteger(n) || n < 0 || n > 255)
  ) {
    return false;
  }
  const [a, b] = p as [number, number, number, number];
  if (a === 127) return false; // loopback: the thing we are trying to escape
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 169 && b === 254) return true; // link-local: a direct cable counts
  return false;
}

/**
 * The address to show, out of everything the machine reports.
 *
 * A workstation has several — docker bridges, VPN tunnels, a wifi and an
 * ethernet — and the useful one is a real LAN address. 192.168/16 and 10/8
 * before link-local (169.254 only exists when DHCP failed), and the first of
 * each in the order the OS gave them, which is the order everything else on the
 * machine also prefers.
 */
export function pickLanIp(addresses: readonly string[]): string {
  const usable = addresses.filter(isReachableV4);
  const rank = (ip: string) => (ip.startsWith("169.254.") ? 1 : 0);
  return usable.sort((x, y) => rank(x) - rank(y))[0] ?? "";
}

/** The URL another machine types. Empty when there is no address to offer —
 *  better nothing than `http://0.0.0.0:8080`, which reaches nothing. */
export function lanUrl(ip: string, port: number): string {
  if (!ip || !(port > 0)) return "";
  return `http://${ip}:${port}`;
}

/**
 * What turning this on means, in one sentence.
 *
 * Shown beside the switch whenever it is on, not hidden in a tooltip: an API
 * key is the difference between "my house can use my model" and "my network
 * can use my model", and llama.cpp's default is no key at all.
 */
export function lanWarning(settings: Settings): string {
  const key = String(settings.apiKey ?? "").trim();
  return key
    ? "Anyone on your network who has the API key can use this model."
    : "Anyone on your network can use this model — llama-server has no password unless you set an API key (Tune → server → API key).";
}
