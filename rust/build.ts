// rust/build.ts — compile llama-sys to WASM and stage it into src/.
// Run via `deno task wasm`. The artifact (src/llama-sys.wasm) is committed so
// a fresh clone runs without a Rust toolchain; re-run this after any .rs edit.

import { dirname, fromFileUrl, join } from "@std/path";

async function run(cmd: string[], cwd: string): Promise<void> {
  const [bin, ...args] = cmd;
  if (!bin) throw new Error("run(): empty command");
  const out = await new Deno.Command(bin, {
    args,
    cwd,
    stdout: "piped",
    stderr: "piped",
  }).output();
  const dec = new TextDecoder();
  const stderr = dec.decode(out.stderr).trim();
  const stdout = dec.decode(out.stdout).trim();
  if (stdout) console.log(stdout);
  if (stderr) console.error(stderr);
  if (!out.success) throw new Error(`command failed: ${cmd.join(" ")}`);
}

if (import.meta.main) {
  const rustDir = dirname(fromFileUrl(import.meta.url));
  const root = join(rustDir, "..");
  const target = "wasm32-unknown-unknown";

  console.log(`→ cargo test (host)`);
  await run(["cargo", "test", "--quiet"], rustDir);

  console.log(`→ cargo build --release --target ${target}`);
  await run(["cargo", "build", "--release", "--target", target], rustDir);

  const artifact = join(rustDir, "target", target, "release", "llama_sys.wasm");
  const staged = join(root, "src", "llama-sys.wasm");
  await Deno.copyFile(artifact, staged);
  const { size } = await Deno.stat(staged);
  console.log(`✓ ${staged} (${(size / 1024).toFixed(1)} KiB)`);
}
