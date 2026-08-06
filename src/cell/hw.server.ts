// src/cell/hw.server.ts — read the machine. SERVER ONLY.
//
// Division of labour: this file does every read and spawn, the Rust core does
// every parse (rust/src/sys.rs). That is why the parsing is unit-tested and
// this file has no branching worth testing — it is I/O and nothing else.
//
// Nothing here throws. A missing sensor, an absent nvidia-smi, a non-Linux host
// all degrade to "unknown" (0 / empty), because a workstation without a GPU is
// a supported machine, not an error state.

import type { Cpu, Gpu, Mem } from "../lib/types.ts";
import { exec, PLATFORM } from "./host.server.ts";

// Re-exported so the cell can stamp them into state without importing the
// host module twice.
export { ARCH, PLATFORM } from "./host.server.ts";
import {
  DEMO_ENV,
  demoCpu,
  demoDisks,
  demoGpus,
  demoMem,
} from "../lib/demo.ts";
import { parseDf } from "../lib/disk.ts";
import type { Disk } from "../lib/disk.ts";
import { cpuJson, gpuJson, memJson } from "./wasm.server.ts";

async function read(path: string): Promise<string> {
  try {
    return await Deno.readTextFile(path);
  } catch {
    return "";
  }
}

/** `label\tmillidegrees` lines from every plausible CPU package sensor. */
async function cpuHwmon(): Promise<string> {
  const wanted = ["coretemp", "k10temp", "zenpower", "cpu_thermal"];
  const lines: string[] = [];
  try {
    for await (const e of Deno.readDir("/sys/class/hwmon")) {
      const dir = `/sys/class/hwmon/${e.name}`;
      const name = (await read(`${dir}/name`)).trim();
      if (!wanted.includes(name)) continue;
      for (let i = 1; i <= 16; i++) {
        const v = (await read(`${dir}/temp${i}_input`)).trim();
        if (!v) continue;
        const label = (await read(`${dir}/temp${i}_label`)).trim() || name;
        lines.push(`${label}\t${v}`);
      }
    }
  } catch {
    // No hwmon (container, macOS, Windows) — temperature stays 0.
  }
  return lines.join("\n");
}

export async function cpu(): Promise<Cpu | null> {
  if (PLATFORM !== "linux") return await cpuNonLinux();
  const [cpuinfo, stat, hwmon] = await Promise.all([
    read("/proc/cpuinfo"),
    read("/proc/stat"),
    cpuHwmon(),
  ]);
  if (!cpuinfo && !stat) return null;
  const j = await cpuJson(cpuinfo, stat, hwmon);
  return {
    model: String(j.model ?? ""),
    cores: Number(j.cores ?? 0),
    threads: Number(j.threads ?? 0),
    mhz: Number(j.mhz ?? 0),
    tempC: Number(j.tempC ?? 0),
    utilPct: 0, // filled by the cell from the delta of two `stat` samples
    stat: String(j.stat ?? ""),
    coreStats: (j.coreStats as string[]) ?? [],
    coresUtil: [],
  };
}

/** macOS / Windows: identity only. Utilization and temperature need per-OS
 *  privileged APIs; showing "—" is honest, inventing a number is not. */
async function cpuNonLinux(): Promise<Cpu | null> {
  const base: Cpu = {
    model: "",
    cores: navigator.hardwareConcurrency ?? 0,
    threads: navigator.hardwareConcurrency ?? 0,
    mhz: 0,
    tempC: 0,
    utilPct: 0,
    stat: "",
    coreStats: [],
    coresUtil: [],
  };
  if (PLATFORM === "darwin") {
    const brand = await exec("sysctl", ["-n", "machdep.cpu.brand_string"]);
    const phys = await exec("sysctl", ["-n", "hw.physicalcpu"]);
    base.model = brand.stdout.trim();
    base.cores = parseInt(phys.stdout.trim(), 10) || base.cores;
  } else if (PLATFORM === "windows") {
    const r = await exec("wmic", [
      "cpu",
      "get",
      "name,NumberOfCores",
      "/value",
    ]);
    base.model = /Name=(.*)/.exec(r.stdout)?.[1]?.trim() ?? "";
    base.cores =
      parseInt(/NumberOfCores=(\d+)/.exec(r.stdout)?.[1] ?? "", 10) ||
      base.cores;
  }
  return base;
}

/**
 * How much memory this process may pin, from its own limits.
 *
 * `/proc/self/limits` rather than a `ulimit` subshell: no process spawn, and it
 * is the value llama-server will INHERIT, which is the one that decides whether
 * `--mlock` does anything. "unlimited" is reported as `Infinity`; anything
 * unreadable as 0, which the tuner reads as "do not promise pinning".
 */
async function lockable(): Promise<number> {
  const text = await read("/proc/self/limits");
  if (!text) return 0;
  const line = text.split("\n").find((l) => /Max locked memory/i.test(l));
  if (!line) return 0;
  const soft = line.replace(/Max locked memory\s*/i, "").trim().split(/\s+/)[0];
  if (!soft) return 0;
  if (/unlimited/i.test(soft)) return Infinity;
  const n = Number(soft);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

export async function mem(): Promise<Mem | null> {
  if (PLATFORM === "linux") {
    const info = await read("/proc/meminfo");
    if (!info) return null;
    const j = await memJson(info);
    return {
      totalB: Number(j.totalB ?? 0),
      availableB: Number(j.availableB ?? 0),
      usedB: Number(j.usedB ?? 0),
      swapTotalB: Number(j.swapTotalB ?? 0),
      swapUsedB: Number(j.swapUsedB ?? 0),
      lockableB: await lockable(),
    };
  }
  const totalB = PLATFORM === "darwin"
    ? Number((await exec("sysctl", ["-n", "hw.memsize"])).stdout.trim()) || 0
    : Number(
      /TotalPhysicalMemory=(\d+)/.exec(
        (await exec("wmic", [
          "ComputerSystem",
          "get",
          "TotalPhysicalMemory",
          "/value",
        ]))
          .stdout,
      )?.[1] ?? 0,
    );
  return totalB
    ? {
      totalB,
      availableB: totalB,
      usedB: 0,
      swapTotalB: 0,
      swapUsedB: 0,
      // Neither macOS nor Windows exposes this the way /proc does, and a guess
      // here would be a promise about pinning we cannot keep.
      lockableB: 0,
    }
    : null;
}

const NVIDIA_QUERY = [
  "--query-gpu=name,temperature.gpu,utilization.gpu,memory.total,memory.used,power.draw,power.limit",
  "--format=csv,noheader,nounits",
];

/**
 * Which NVIDIA cards have a display attached, in `nvidia-smi` index order.
 *
 * A SECOND query rather than two more columns on `NVIDIA_QUERY`: that CSV's
 * shape is `sys::gpu`'s contract (rust/src/sys.rs), and widening it there would
 * mean a wasm rebuild to read a field the planner wants and the parser does not.
 *
 * Both fields are asked because they answer slightly different questions and
 * either one being `Enabled` means there is a screen to defend: `display_mode`
 * is "a display is connected", `display_active` is "one is initialised on this
 * GPU". `[N/A]` — an older driver, a non-display card — is `undefined`, which
 * `src/lib/reserve.ts:displayGpus` treats as "not known", not as "no".
 */
async function nvidiaDisplays(): Promise<(boolean | undefined)[]> {
  // Cached, because `snapshot()` runs every second and this one does not
  // change on that cadence — a monitor gets plugged in about as often as a
  // machine is rebooted. A second spawn of `nvidia-smi` per second to re-read a
  // constant is a cost the poll should not carry.
  const now = Date.now();
  if (nvDisplayCache && now - nvDisplayCache.at < DISPLAY_TTL_MS) {
    return nvDisplayCache.v;
  }
  const r = await exec("nvidia-smi", [
    "--query-gpu=display_mode,display_active",
    "--format=csv,noheader",
  ]);
  const v = r.code !== 0
    ? []
    : r.stdout.split("\n").filter((l) => l.trim() !== "").map((line) => {
      const f = line.split(",").map((s) => s.trim().toLowerCase());
      if (f.some((s) => s === "enabled")) return true;
      if (f.some((s) => s === "disabled")) return false;
      return undefined;
    });
  nvDisplayCache = { at: now, v };
  return v;
}

let nvDisplayCache: { at: number; v: (boolean | undefined)[] } | null = null;
const DISPLAY_TTL_MS = 30_000;

/**
 * Does this DRM card have a connected connector?
 *
 * The kernel exposes one `cardN-<connector>` directory per output with a
 * `status` file that reads `connected` or `disconnected`. That is the same
 * reading a display manager uses, and it needs no root and no vendor tool.
 * No connectors at all (a compute-only card, a headless enumeration) is
 * `undefined` rather than `false`: nothing was measured.
 */
async function drmDisplay(card: string): Promise<boolean | undefined> {
  let seen = false;
  try {
    for await (const e of Deno.readDir("/sys/class/drm")) {
      if (!e.name.startsWith(`${card}-`)) continue;
      seen = true;
      const st = (await read(`/sys/class/drm/${e.name}/status`)).trim();
      if (st === "connected") return true;
    }
  } catch {
    // No /sys/class/drm — nothing known either way.
  }
  return seen ? false : undefined;
}

/** AMD cards through sysfs: one tab-separated line per card, exactly the shape
 *  `sys::gpu` expects, plus the display reading for each in the same order. */
async function amdSysfs(): Promise<{
  text: string;
  displays: (boolean | undefined)[];
}> {
  const lines: string[] = [];
  const displays: (boolean | undefined)[] = [];
  try {
    for await (const e of Deno.readDir("/sys/class/drm")) {
      if (!/^card\d+$/.test(e.name)) continue;
      const dev = `/sys/class/drm/${e.name}/device`;
      const vendor = (await read(`${dev}/vendor`)).trim();
      if (vendor !== "0x1002") continue; // AMD
      const total = (await read(`${dev}/mem_info_vram_total`)).trim();
      if (!total) continue; // not a discrete GPU exposing VRAM counters
      const used = (await read(`${dev}/mem_info_vram_used`)).trim();
      const busy = (await read(`${dev}/gpu_busy_percent`)).trim();
      let temp = "";
      let power = "";
      try {
        for await (const h of Deno.readDir(`${dev}/hwmon`)) {
          temp = (await read(`${dev}/hwmon/${h.name}/temp1_input`)).trim();
          power = (await read(`${dev}/hwmon/${h.name}/power1_average`)).trim();
          if (temp) break;
        }
      } catch {
        // hwmon absent — temperature stays unknown.
      }
      const name = (await read(`${dev}/product_name`)).trim() ||
        `AMD GPU (${e.name})`;
      lines.push([name, temp, busy, used, total, power].join("\t"));
      displays.push(await drmDisplay(e.name));
    }
  } catch {
    // No /sys/class/drm — not Linux, or no DRM devices.
  }
  return { text: lines.join("\n"), displays };
}

/**
 * Every IPv4 address this machine has, as the OS reports them.
 *
 * For the "Available on LAN" switch: binding llama-server to 0.0.0.0 is only
 * half an answer — the other half is which address another machine should
 * dial, and `0.0.0.0` is not one (`src/lib/lan.ts`). Ordering is the OS's, and
 * the choosing is `pickLanIp`'s; this only reads.
 */
export function lanAddresses(): string[] {
  try {
    return Deno.networkInterfaces()
      .filter((n) => n.family === "IPv4")
      .map((n) => n.address);
  } catch {
    // No permission, or a platform that does not report them. The switch still
    // works; only the "reachable at …" line goes quiet, which is honest.
    return [];
  }
}

export async function gpus(): Promise<Gpu[]> {
  const [nv, nvDisplays, amd] = await Promise.all([
    exec("nvidia-smi", NVIDIA_QUERY),
    nvidiaDisplays(),
    PLATFORM === "linux"
      ? amdSysfs()
      : Promise.resolve({ text: "", displays: [] }),
  ]);
  const nvCsv = nv.code === 0 ? nv.stdout : "";
  if (!nvCsv && !amd.text) return [];
  const list = await gpuJson(nvCsv, amd.text);
  // `sys::gpu` emits the NVIDIA cards first, in CSV order, then the sysfs cards
  // in the order they were assembled above — so the display readings zip back on
  // by vendor, in order. Keeping that here rather than in the Rust core is what
  // lets a new reading ship without a wasm rebuild.
  let nvIdx = 0;
  let amdIdx = 0;
  return list.map((g) => {
    const vendor = (g.vendor as Gpu["vendor"]) ?? "nvidia";
    const display = vendor === "nvidia"
      ? nvDisplays[nvIdx++]
      : amd.displays[amdIdx++];
    return {
      vendor,
      name: String(g.name ?? "GPU"),
      tempC: Number(g.tempC ?? 0),
      utilPct: Number(g.utilPct ?? 0),
      vramTotalB: Number(g.vramTotalB ?? 0),
      vramUsedB: Number(g.vramUsedB ?? 0),
      powerW: Number(g.powerW ?? 0),
      powerLimitW: Number(g.powerLimitW ?? 0),
      computeCap: Number(g.computeCap ?? 0),
      display,
    };
  });
}

/** One shot of everything, read in parallel. */
/**
 * The filesystems this app writes to, and how much room is left on them.
 *
 * `df -kP` on POSIX; nothing on Windows yet, where the equivalent is a
 * PowerShell call and this app has never been run. Deliberately a separate
 * reader rather than part of `snapshot()`: it shells out, and disk space does
 * not change on a one-second cadence, so it gets its own slower schedule.
 */
export async function disks(paths: string[]): Promise<Disk[]> {
  if (Deno.env.get(DEMO_ENV) === "1") return demoDisks();
  if (PLATFORM === "windows") return [];
  // Only paths that exist: most of the default model locations do not, and `df`
  // exits non-zero when ANY argument is missing. Passing them all and then
  // trusting the exit code threw away every valid row — the first version of
  // this returned nothing at all on a normal machine.
  const live: string[] = [];
  for (const p of paths) {
    try {
      await Deno.stat(p);
      live.push(p);
    } catch { /* not there — nothing to measure */ }
  }
  if (live.length === 0) return [];
  const r = await exec("df", ["-kP", ...live]);
  // Parse stdout regardless of the exit code: `df` still prints the rows it
  // could resolve, and a partial answer beats none.
  return parseDf(r.stdout);
}

export async function snapshot(): Promise<{
  cpu: Cpu | null;
  mem: Mem | null;
  gpus: Gpu[];
}> {
  // Demo mode reports a machine that does not exist, so a screenshot or a bug
  // report need not carry the author's hardware (src/lib/demo.ts).
  if (Deno.env.get(DEMO_ENV) === "1") {
    return { cpu: demoCpu(), mem: demoMem(), gpus: demoGpus() };
  }
  const [c, m, g] = await Promise.all([cpu(), mem(), gpus()]);
  return { cpu: c, mem: m, gpus: g };
}
