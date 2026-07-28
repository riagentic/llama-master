// src/ui/DevicePicker.tsx — which GPUs llama.cpp may use.
//
// Its own module because two pages need it: the Tune tab (where the `-dev` flag
// it writes is declared) and the GPU page (where the cards it talks about are).
// Two copies of a control that writes a command-line flag is two chances to
// disagree about what the flag means.

import { cfg } from "../cell/cfg.ts";
import { hw } from "../cell/hw.ts";
import { devices, isEnabled, toggleDevice } from "../lib/gpu.ts";
import { bytes } from "../lib/format.ts";
import type { Param } from "../lib/types.ts";
import { activeBuild } from "./derive.ts";

/**
 * One checkbox per GPU the active build can address.
 *
 * The kata asks that the user be able to enable or disable any GPU. Typing
 * llama.cpp's own device names into a text box is not that: it needs the naming
 * convention, the enumeration order, and the knowledge that an empty value means
 * "all". Those are the three things this control knows so the user does not have
 * to — it shows the cards by their real names, with VRAM, and writes the flag.
 */
export function DevicePicker(props: { value: string; p: Param }) {
  const backend = activeBuild()?.backend;
  const list = devices(backend, hw.gpus);
  if (list.length === 0) {
    // Three different "no devices", and they need different sentences: no card
    // at all, a build that addresses none, and no build chosen yet — in which
    // case the device NAMES are unknown, because llama.cpp derives them from
    // the backend.
    return (
      <span class="unit" t="no-devices">
        {hw.gpus.length === 0
          ? "No GPU detected — llama.cpp will run on the CPU."
          : backend === undefined
          ? "Install a llama.cpp build first — which GPUs are addressable, and what they are called, depends on its backend."
          : `A ${backend.toUpperCase()} build addresses no GPU; every layer runs on the CPU.`}
      </span>
    );
  }
  const all = list.map((d) => d.id);
  const on = list.filter((d) => isEnabled(props.value, d.id)).length;
  return (
    <div class="device-picker" t="device-picker">
      {list.map((d) => (
        <label
          key={d.id}
          class={isEnabled(props.value, d.id) ? "device on" : "device"}
          title={`${d.label} — passed to llama.cpp as ${d.id}`}
        >
          <input
            type="checkbox"
            checked={isEnabled(props.value, d.id)}
            aria-label={`${d.label} (${d.id})`}
            // The device id is the stable handle: two identical cards share a
            // name, so the name alone cannot address either of them.
            t={d.id}
            // Never let the last one go: llama.cpp with no device fails at load,
            // and "use nothing" is what -ngl 0 already says, honestly.
            disabled={on === 1 && isEnabled(props.value, d.id)}
            onChange={(e) =>
              cfg.set(
                props.p.key,
                toggleDevice(
                  props.value,
                  all,
                  d.id,
                  (e.currentTarget as HTMLInputElement).checked,
                ),
              )}
          />
          <span class="device-name">{d.label}</span>
          <span class="device-id">{d.id}</span>
          <span class="device-vram">{bytes(d.gpu.vramTotalB)}</span>
        </label>
      ))}
    </div>
  );
}
