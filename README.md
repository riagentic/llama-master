# llama.master

A desktop app for [llama.cpp](https://github.com/ggml-org/llama.cpp): get it,
find your models, work out what will actually fit, run it, talk to it.

![llama.master](docs/screenshot.png)

<sub>Demo mode (`LLAMA_MASTER_DEMO=1`) — the machine, the models and the build
shown are fictional.</sub>

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/riagentic/aio/main/run.sh | sh -s riagentic/llama-master
```

Installs [Deno](https://deno.com) and the
[aio](https://github.com/riagentic/aio) framework if they are missing, clones,
builds and runs. Nothing else is needed: CMake and SPIRV-Headers are downloaded
into the app's own directory if a source build of llama.cpp asks for them, and a
prebuilt llama.cpp needs no toolchain.

From a clone — aio is vendored by symlink, so the two sit side by side:

```sh
git clone https://github.com/riagentic/aio.git
git clone https://github.com/riagentic/llama-master.git
cd llama-master && mkdir -p dep && ln -s ../../aio dep/aio
deno task am fix    # import paths, electron, config
deno task dev
```

`deno task compile` produces a self-contained binary in `dist/`;
`compile:electron` a desktop package.

## What it does

- **Gets llama.cpp** — prebuilt or built from source, for CPU, CUDA, Vulkan,
  ROCm or Metal. It checks the build can actually succeed before enabling the
  button, and offers to install what is missing, showing the command first.
- **Finds your models** — plain `.gguf` trees, LM Studio, and ollama's blob
  store, which has no `.gguf` files in it at all.
- **Says what will fit** — the GGUF header is parsed in Rust and walked
  tensor-by-tensor, so weights, routed experts and KV cache are exact. Only the
  compute buffer is an estimate, and it says so everywhere it appears.
- **Chooses settings** — one set of optimal settings; you choose where it runs
  (VRAM only · Hybrid · CPU only), each at the largest context that fits.
- **Runs it, and tells the truth about it** — the memory view describes the
  process that is running, not the form. A server that dies is diagnosed from
  its own output, never as a bare exit code.
- **Keeps the machine usable** — reserved VRAM and RAM the plan may not spend,
  and llama.cpp at the lowest OS priority so the desktop stays responsive.

`client/` is a second, standalone app: a chat client for a llama.master running
on another machine. It finds one on the network, shows what is loaded and how
busy it is, and chats with it — `cd client && deno task dev`.

## Status

Developed on Linux/x86_64 with NVIDIA and AMD hardware: CPU, CUDA and Vulkan
built from source and run; CPU, Vulkan and ROCm installed from prebuilt releases
and run; real models up to 145 GB. The macOS, Windows and arm64 paths are
implemented and unit-tested but have not been run by the author. Metal is
refused with an explanation off Apple hardware, and untested on it.

## Development

```sh
deno task verify                 # fmt · lint · check · test — the gate
deno task test:rust              # the Rust core
deno task am state hw.gpus       # question the running app
cd client && deno task verify    # the client has its own gate
```

Everything the app writes lives in `~/.llama-master/`; set `LLAMA_MASTER_HOME`
to move it (builds are gigabytes). The decisions worth knowing before changing
anything are in [CLAUDE.md](CLAUDE.md).

## Credits

The engine is [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi
Gerganov and its contributors — all inference, and every binary this app builds
or installs, is theirs. This is a front end. Built on
[aio](https://github.com/riagentic/aio).

## License

MIT — see [LICENSE](LICENSE).
