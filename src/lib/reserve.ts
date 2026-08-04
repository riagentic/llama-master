// src/lib/reserve.ts — the memory the user keeps for themselves.
//
// Every other limit in this app exists to stop llama.cpp failing: the fixed
// margins in `tune.ts` and the per-card reserve in `devsplit.ts` are there
// because an allocator that is handed the last byte of a pool dies on the next
// fragment. This one is different in kind. It is the user saying "that GPU also
// draws my desktop" or "I want to keep compiling while this runs", and the only
// honest way to honour it is to plan as if that memory were not there at all.
//
// So a reserve is not a warning and not a preference the tuner weighs. It is
// subtracted from what a plan may spend, before the plan is made — which is why
// it enters through `Hw` (`types.ts:Reserve`) and every consumer of `plan`
// inherits it without knowing it exists.
//
// Three numbers, because there are three different claims:
//
//   • per GPU     — held back on EVERY card. Default 0: a card doing nothing but
//                   inference has no second tenant, and taking bytes from all of
//                   them to protect one is how a two-card machine loses 8 GB to
//                   defend a desktop that lives on one.
//   • connected   — held back only on the card(s) with a display attached. This
//                   is where the desktop, the browser and the game actually are,
//                   and it is the reserve that is nearly always the right one.
//   • RAM         — the host pool, one number, no such distinction.
//
// Pure: clamping, selection and division only, so the policy is testable
// without a machine.

import type { Gpu, Hw, Reserve } from "./types.ts";

const GIB = 1024 ** 3;

/**
 * VRAM held back on every card by default: nothing.
 *
 * A second card is usually there to compute, and the thing worth defending —
 * the display — is defended by `DEFAULT_RESERVE_CONNECTED_VRAM_B` on the card
 * that actually has it. Charging every card for that is a refusal the user
 * never asked for. Set it when a card is shared with something else that is not
 * the desktop (another inference server, a render job).
 */
export const DEFAULT_RESERVE_PER_GPU_VRAM_B = 0;

/**
 * VRAM held back on the card that drives the display.
 *
 * 8 GB, because the pool being defended is the one with no swap and the tenants
 * are not small: a compositor, a hardware-accelerated browser with a video call
 * and a game or a second GPU tool sit well past the 2-3 GB a bare desktop costs,
 * and a card filled to the last byte by the tuner turns their next allocation
 * into a driver reset — or, worse, into llama.cpp's, mid-generation. Zero is a
 * legitimate setting for a machine whose display card is not shared, and the
 * control accepts it.
 */
export const DEFAULT_RESERVE_CONNECTED_VRAM_B = 8 * GIB;

/**
 * RAM held back by default.
 *
 * 16 GB, and the asymmetry is deliberate: host memory is where the user's own
 * work actually lives (editors, browsers, a compile, the page cache that makes
 * a mapped 145 GB model start in 6 seconds instead of 73), and llama-server is
 * always the largest process on the machine, so when this pool runs out the OOM
 * killer picks the thing the user cares about second. Weights and KV are
 * anonymous pages the kernel cannot reclaim, so "tight" here does not mean slow,
 * it means killed.
 */
export const DEFAULT_RESERVE_RAM_B = 16 * GIB;

/** The most a reserve may be set to, in GB. A number typed into a box has to
 *  have a ceiling; 1 TB is past any machine this runs on and keeps the value
 *  finite whatever the input element produces. */
export const MAX_RESERVE_GB = 1024;

/** GB off a control into bytes: finite, whole-ish, and inside the range. A
 *  blank or hostile input is 0 — "reserve nothing" — never NaN. */
export function reserveBytes(gb: number): number {
  if (!Number.isFinite(gb) || gb <= 0) return 0;
  return Math.min(gb, MAX_RESERVE_GB) * GIB;
}

/** Bytes back to the GB the control shows. One decimal, because the value is
 *  only ever entered in GB and 4 must read as `4`, not `3.9999999`. */
export function reserveGb(bytes: number): number {
  if (!Number.isFinite(bytes) || bytes <= 0) return 0;
  return Math.round((bytes / GIB) * 10) / 10;
}

/** The default reserve — what a machine gets before anyone touches a control. */
export function defaultReserve(): Reserve {
  return {
    perGpuB: DEFAULT_RESERVE_PER_GPU_VRAM_B,
    connectedB: DEFAULT_RESERVE_CONNECTED_VRAM_B,
    ramB: DEFAULT_RESERVE_RAM_B,
  };
}

/**
 * The reserve to actually plan against, clamped to the machine it applies to.
 *
 * Clamped, not validated away: a 32 GB reserve on a 24 GB card is an
 * instruction that cannot be met, and answering it by holding back the whole
 * card is the honest reading — every placement then reports that it does not
 * fit, with the numbers on screen showing why. What must never happen is a
 * reserve larger than the pool leaking through as a NEGATIVE capacity, which
 * would make every fit test downstream meaningless in the quiet way NaN does.
 *
 * The two VRAM figures are clamped to the largest card rather than to the total:
 * they are per-CARD instructions, and `vramReserveShares` clamps each card's own
 * share to that card in any case.
 */
export function reserveOf(hw: Hw): Reserve {
  const r = hw.reserve;
  if (!r) return { perGpuB: 0, connectedB: 0, ramB: 0 };
  const cardCap = hw.gpus.reduce(
    (a, g) => Math.max(a, Math.max(0, g.vramTotalB)),
    0,
  );
  const ramCap = Math.max(0, hw.mem?.totalB ?? 0);
  return {
    perGpuB: clamp(r.perGpuB, cardCap),
    connectedB: clamp(r.connectedB, cardCap),
    ramB: clamp(r.ramB, ramCap),
  };
}

function clamp(b: number, capacityB: number): number {
  if (!Number.isFinite(b) || b <= 0) return 0;
  return Math.min(b, capacityB);
}

/**
 * Which cards the connected reserve applies to.
 *
 * `Gpu.display` is a three-valued reading and all three answers matter:
 *
 * - Some card says `true` → those cards, and only those. This is the normal
 *   case and the whole point of the setting.
 * - Every card says `false` → nothing is held back. The machine has been asked
 *   and answered: it is headless, so there is no desktop to defend.
 * - Nothing is known (no card carries a reading at all) → card 0. A machine
 *   whose display card cannot be identified is far more often a workstation
 *   with one screen on the first card than a headless server, and getting this
 *   wrong the other way is a driver reset mid-generation. The UI says it is an
 *   assumption rather than pretending it was measured.
 */
export function displayGpus(gpus: readonly Gpu[]): boolean[] {
  if (gpus.length === 0) return [];
  const known = gpus.some((g) => g.display !== undefined);
  if (!known) return gpus.map((_, i) => i === 0);
  return gpus.map((g) => g.display === true);
}

/** True when no card reported whether a display is attached, so `displayGpus`
 *  is guessing at card 0 rather than reporting. The UI owes the user that word.
 */
export function displayUnknown(gpus: readonly Gpu[]): boolean {
  return gpus.length > 0 && !gpus.some((g) => g.display !== undefined);
}

/**
 * How much VRAM is held back on each card.
 *
 * The per-GPU figure on every card, plus the connected figure on the cards that
 * drive a display — added, because they are two separate claims on the same
 * card and honouring only the larger would silently drop one of them. Each
 * card's total is capped at that card: a reserve bigger than the card means the
 * card is entirely spoken for, never a negative budget.
 *
 * Note what changed here, and why. This used to divide ONE machine-wide VRAM
 * figure across the cards in proportion to their size, which kept the total
 * exactly what the user typed but spread it over cards that had no display on
 * them — protecting a desktop that is on one card by taking bytes from all of
 * them. Per-card is what the user is actually able to reason about ("this card
 * runs my screen"), so the number they type is now per card by construction.
 */
export function vramReserveShares(
  gpus: readonly Gpu[],
  reserve: Reserve,
): number[] {
  const displays = displayGpus(gpus);
  const perGpu = Math.max(0, reserve.perGpuB || 0);
  const connected = Math.max(0, reserve.connectedB || 0);
  return gpus.map((g, i) => {
    const want = perGpu + (displays[i] ? connected : 0);
    const cap = Math.max(0, g.vramTotalB);
    return Math.min(Math.max(0, want), cap);
  });
}

/** Every reserved VRAM byte on the machine — the sum of the per-card shares,
 *  which is what the VRAM pool must treat as spent. */
export function vramReserveTotal(
  gpus: readonly Gpu[],
  reserve: Reserve,
): number {
  return vramReserveShares(gpus, reserve).reduce((a, b) => a + b, 0);
}

/** Bytes as GB for the sentences the planner writes — `src/lib` has no access
 *  to the UI formatter, and one decimal is what every other reserve line uses.
 *  The VRAM figure is the machine-wide total, because that is what the plan it
 *  appears beside was denied. */
export function reserveLabel(r: Reserve, gpus: readonly Gpu[] = []): string {
  const parts: string[] = [];
  const vramB = vramReserveTotal(gpus, r);
  if (vramB > 0) parts.push(`${(vramB / GIB).toFixed(1)} GB of VRAM`);
  if (r.ramB > 0) parts.push(`${(r.ramB / GIB).toFixed(1)} GB of RAM`);
  return parts.join(" and ");
}
