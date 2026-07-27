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
import { DEMO_ENV, demoCpu, demoGpus, demoMem } from "../lib/demo.ts";
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
    ? { totalB, availableB: totalB, usedB: 0, swapTotalB: 0, swapUsedB: 0 }
    : null;
}

const NVIDIA_QUERY = [
  "--query-gpu=name,temperature.gpu,utilization.gpu,memory.total,memory.used,power.draw,power.limit",
  "--format=csv,noheader,nounits",
];

/** AMD cards through sysfs: one tab-separated line per card, exactly the shape
 *  `sys::gpu` expects. */
async function amdSysfs(): Promise<string> {
  const lines: string[] = [];
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
    }
  } catch {
    // No /sys/class/drm — not Linux, or no DRM devices.
  }
  return lines.join("\n");
}

export async function gpus(): Promise<Gpu[]> {
  const [nv, amd] = await Promise.all([
    exec("nvidia-smi", NVIDIA_QUERY),
    PLATFORM === "linux" ? amdSysfs() : Promise.resolve(""),
  ]);
  const nvCsv = nv.code === 0 ? nv.stdout : "";
  if (!nvCsv && !amd) return [];
  const list = await gpuJson(nvCsv, amd);
  return list.map((g) => ({
    vendor: (g.vendor as Gpu["vendor"]) ?? "nvidia",
    name: String(g.name ?? "GPU"),
    tempC: Number(g.tempC ?? 0),
    utilPct: Number(g.utilPct ?? 0),
    vramTotalB: Number(g.vramTotalB ?? 0),
    vramUsedB: Number(g.vramUsedB ?? 0),
    powerW: Number(g.powerW ?? 0),
    powerLimitW: Number(g.powerLimitW ?? 0),
    computeCap: Number(g.computeCap ?? 0),
  }));
}

/** One shot of everything, read in parallel. */
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
