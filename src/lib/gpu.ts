// src/lib/gpu.ts — which GPUs llama.cpp will use, and which the user turned off.
//
// llama.cpp restricts devices with `-dev CUDA0,CUDA1`: a comma list of names it
// assigns per BACKEND, in registration order. So a device name is not a property
// of the card — the same card is `CUDA0` to a CUDA build, `Vulkan0` to a Vulkan
// build, and invisible to a CPU build. That is why the id is computed from the
// active backend rather than stored on the Gpu.
//
// An empty value means "every device", which is llama.cpp's own default and the
// honest representation of "the user has not restricted anything" — so turning
// the last box back on returns the setting to empty rather than listing all of
// them.
//
// Pure: a backend, the detected cards and the current setting in; names, flags
// and the resulting device set out.

import { usableGpus } from "./backend.ts";
import type { Backend, Gpu } from "./types.ts";

/**
 * The prefix llama.cpp gives each backend's devices, or null when there is no
 * answer.
 *
 * Null for a CPU build (it registers no GPU device) and null for "no build
 * selected" — inventing a name there would put a flag like `-dev GPU0` on the
 * command line, which llama.cpp rejects. The device name genuinely depends on
 * the build, so with no build there is nothing honest to show.
 */
function prefixOf(backend: Backend | undefined): string | null {
  switch (backend) {
    case "cuda":
      return "CUDA";
    case "hip":
      return "ROCm";
    case "metal":
      return "Metal";
    case "vulkan":
      return "Vulkan";
    default:
      return null;
  }
}

export type Device = {
  /** What `-dev` calls it, e.g. `CUDA0`. */
  id: string;
  /** What the user calls it, e.g. `NVIDIA RTX PRO 4000 Blackwell`. */
  label: string;
  gpu: Gpu;
};

/**
 * The devices a build with this backend can address, in llama.cpp's own order.
 *
 * Indices count within the backend, not within the machine: on a CUDA build the
 * AMD iGPU is not a device at all, so the second NVIDIA card is `CUDA1` even
 * though it is the machine's third GPU.
 */
export function devices(
  backend: Backend | undefined,
  gpus: readonly Gpu[],
): Device[] {
  const prefix = prefixOf(backend);
  if (prefix === null) return [];
  return usableGpus(backend, gpus).map((gpu, i) => ({
    id: `${prefix}${i}`,
    label: gpu.name,
    gpu,
  }));
}

/** The device names in a `-dev` value. Empty value → empty list, meaning "all". */
export function parseDevices(value: string): string[] {
  return value.split(",").map((s) => s.trim()).filter(Boolean);
}

/** Is this device in use? An unrestricted setting means every device is. */
export function isEnabled(value: string, id: string): boolean {
  const list = parseDevices(value);
  return list.length === 0 || list.includes(id);
}

/**
 * The `-dev` value after switching one device on or off.
 *
 * Order follows `all` rather than the order boxes were clicked, so the flag is
 * stable and the command preview does not churn. Turning everything back on
 * yields "" — the default — rather than a list of every device.
 */
export function toggleDevice(
  value: string,
  all: readonly string[],
  id: string,
  on: boolean,
): string {
  const enabled = new Set(
    parseDevices(value).length === 0 ? all : parseDevices(value),
  );
  if (on) enabled.add(id);
  else enabled.delete(id);
  const kept = all.filter((d) => enabled.has(d));
  return kept.length === all.length ? "" : kept.join(",");
}

/**
 * The GPUs that will actually be used: the backend's devices, minus the ones
 * switched off.
 *
 * The memory planner takes this, so the picture matches what will run. Without
 * it, disabling a card left the plan still spreading the model across it.
 */
export function enabledGpus(
  backend: Backend | undefined,
  gpus: readonly Gpu[],
  value: string,
): Gpu[] {
  const named = devices(backend, gpus);
  // No nameable devices means no restriction can be expressed — with no build
  // selected yet the machine still has its VRAM, and the plan must show it.
  // (A CPU build is different: `usableGpus` already returns none for it.)
  if (named.length === 0) return usableGpus(backend, gpus);
  return named.filter((d) => isEnabled(value, d.id)).map((d) => d.gpu);
}
