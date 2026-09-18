// tests/matrix.test.ts — the build × driver × model × environment matrix.
//
// Each case is one cell of "does this combination run": a release whose CUDA
// is newer than the driver, a CUDA binary whose runtime ships in a second
// archive, a source build that links into the source cache, a vendor model on
// an upstream build, a stray LLAMA_ARG_* in the parent environment. Pure
// inputs, pure answers — the I/O around them is exercised by the app.

import { assert, assertEquals } from "@std/assert";
import {
  type Asset,
  availableBackends,
  companionAsset,
  cudaVersionOf,
  pickAsset,
} from "../src/lib/assets.ts";
import { driverCudaVersion } from "../src/lib/cuda.ts";
import { spawnEnv } from "../src/lib/envvars.ts";
import { buildVendor, runtimeMismatch, VENDORS } from "../src/lib/runtime.ts";
import { borrowedLibs } from "../src/cell/builds.server.ts";
import { argv } from "../src/lib/command.ts";
import { defaults } from "../src/lib/params.ts";

const asset = (name: string): Asset => ({ name, url: `u/${name}`, sizeB: 1 });

// The b11039 Linux release, as upstream publishes it.
const B11039 = [
  "llama-b11039-bin-ubuntu-x64.tar.gz",
  "llama-b11039-bin-ubuntu-vulkan-x64.tar.gz",
  "llama-b11039-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz",
  "cudart-llama-b11039-bin-ubuntu-cuda-12.8-x64.tar.gz",
  "cudart-llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz",
  "llama-b11039-bin-win-cuda-12.4-x64.zip",
  "cudart-llama-bin-win-cuda-12.4-x64.zip",
].map(asset);

Deno.test("assets: the CUDA version is read from the name", () => {
  assertEquals(cudaVersionOf("llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz"), 12.8);
  assertEquals(cudaVersionOf("llama-b1-bin-ubuntu-cuda-13.3-x64.tar.gz"), 13.3);
  assertEquals(cudaVersionOf("llama-b1-bin-ubuntu-vulkan-x64.tar.gz"), 0);
});

Deno.test("assets: the newest CUDA the DRIVER can run is picked, never newer", () => {
  const pick = (max: number) =>
    pickAsset(B11039, "linux", "x86_64", "cuda", max)?.name;
  // A driver that reports 13.3 gets the 13.3 build.
  assertEquals(pick(13.3), "llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz");
  // A 12.x driver cannot load a 13.3 binary — it gets 12.8, not a crash.
  assertEquals(pick(12.9), "llama-b11039-bin-ubuntu-cuda-12.8-x64.tar.gz");
  // Older than every build offered: nothing, rather than a binary that fails.
  assertEquals(pick(12.2), undefined);
  assert(
    !availableBackends(B11039, "linux", "x86_64", 12.2).includes("cuda"),
    "and the backend is not offered at all",
  );
  // Unknown driver (0): the newest is still chosen.
  assertEquals(pick(0), "llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz");
});

Deno.test("assets: a CUDA binary brings its runtime archive, a CPU one does not", () => {
  const get = (n: string) => B11039.find((a) => a.name === n)!;
  assertEquals(
    companionAsset(B11039, get("llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz"))
      ?.name,
    "cudart-llama-b11039-bin-ubuntu-cuda-13.3-x64.tar.gz",
  );
  // Windows spells it without the build number.
  assertEquals(
    companionAsset(B11039, get("llama-b11039-bin-win-cuda-12.4-x64.zip"))?.name,
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
  );
  assertEquals(
    companionAsset(B11039, get("llama-b11039-bin-ubuntu-x64.tar.gz")),
    null,
  );
  assertEquals(
    companionAsset(
      B11039,
      get("cudart-llama-b11039-bin-ubuntu-cuda-12.8-x64.tar.gz"),
    ),
    null,
    "the runtime archive has no runtime archive of its own",
  );
});

Deno.test("cuda: both nvidia-smi spellings of the driver's CUDA version are read", () => {
  assertEquals(
    driverCudaVersion("| NVIDIA-SMI 580  CUDA Version: 12.9 |"),
    12.9,
  );
  assertEquals(driverCudaVersion("CUDA UMD Version: 13.3"), 13.3);
  // `nvidia-smi --version`: a "Deprecated" line first, then the padded label.
  assertEquals(
    driverCudaVersion(
      "NVIDIA-SMI version  : 595.1\nCUDA version        : Deprecated\nCUDA UMD version    : 13.3\n",
    ),
    13.3,
  );
  assertEquals(driverCudaVersion("no gpu here"), 0);
});

Deno.test("builds: a binary linked into the source cache is not self-contained", () => {
  const ldd = [
    "\tlinux-vdso.so.1 (0x00007ffd)",
    "\tlibllama.so => /home/u/.llama-master/data/files/builds/b/libllama.so (0x7f)",
    "\tlibllama-server-impl.so => /home/u/.llama-master/cache/sources/master/build/bin/libllama-server-impl.so (0x7f)",
    "\tlibggml.so => not found",
    "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x7f)",
  ].join("\n");
  assertEquals(
    borrowedLibs(ldd, "/home/u/.llama-master/cache/sources"),
    ["libllama-server-impl.so", "libggml.so"],
  );
  // A prefix of the cache path is not the cache.
  assertEquals(
    borrowedLibs(
      "\tlibx.so => /home/u/.llama-master/cache/sources-old/libx.so (0x7f)",
      "/home/u/.llama-master/cache/sources",
    ),
    [],
  );
});

Deno.test("runtime: which build can run a vendor's model", () => {
  assertEquals(buildVendor("master"), "upstream");
  assertEquals(buildVendor("b11039"), "upstream");
  assertEquals(buildVendor("pr/29077"), "upstream");
  assertEquals(buildVendor(VENDORS.prism!.ref), "prism");
  assertEquals(buildVendor("fork:prismml-eng/llama.cpp@prism"), "prism");
  assertEquals(buildVendor("fork:someone/llama.cpp"), "unknown");

  // Plain model, any build: nothing to say.
  assertEquals(runtimeMismatch(undefined, "master", []), null);
  assertEquals(runtimeMismatch("", "master", []), null);
  // Vendor model on the vendor's build, or on a fork nobody can see into.
  assertEquals(runtimeMismatch("prism", VENDORS.prism!.ref, []), null);
  assertEquals(runtimeMismatch("prism", "fork:someone/llama.cpp", []), null);
  // A vendor this app has never heard of: no guess.
  assertEquals(runtimeMismatch("acme", "master", []), null);

  // Upstream, nothing installed: build it.
  const build = runtimeMismatch("prism", "master", [{
    id: "m",
    ref: "master",
  }]);
  assertEquals(build?.steps[0]?.action, {
    kind: "use-ref",
    ref: VENDORS.prism!.ref,
  });
  // Upstream, the fork already installed: one click, not three minutes.
  const sw = runtimeMismatch("prism", "master", [
    { id: "m", ref: "master" },
    { id: "f", ref: VENDORS.prism!.ref },
  ]);
  assertEquals(sw?.steps[0]?.action, { kind: "use-build", id: "f" });
  assert(sw!.steps.some((s) => s.action?.kind === "open-url"));
});

Deno.test("env: a stray LLAMA_ARG_* never overrides the command on screen", () => {
  const { env, dropped } = spawnEnv(
    {
      PATH: "/usr/bin",
      LLAMA_ARG_CTX_SIZE: "8192",
      LLAMA_ARG_N_GPU_LAYERS: "0",
      LLAMA_ARG_THREADS: "4",
    },
    [
      { name: "LLAMA_ARG_THREADS", value: "8" },
      { name: "GGML_CUDA_DISABLE_GRAPHS", value: "1" },
    ],
  );
  assertEquals(dropped, ["LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_N_GPU_LAYERS"]);
  // Inherited environment survives; the user's own variables win.
  assertEquals(env, {
    PATH: "/usr/bin",
    LLAMA_ARG_THREADS: "8",
    GGML_CUDA_DISABLE_GRAPHS: "1",
  });
});

Deno.test("command: the load trace is asked for, where the build knows how", () => {
  const cmd = (caps?: string[]) =>
    argv("server", { bin: "b", model: "m", settings: defaults(), caps });
  const lv = (a: string[]) => a[a.indexOf("-lv") + 1];
  // llama.cpp's default (3) hides every buffer-size and load line.
  assertEquals(lv(cmd()), "4");
  assertEquals(lv(cmd(["-m", "-ngl", "-c", "-np", "-lv"])), "4");
  // A probed build that does not declare it is never handed it.
  assert(!cmd(["-m", "-ngl", "-c", "-np"]).includes("-lv"));
});
