# Codacus tips for llama.master for optimal settings

- when model cannot be fit into vram optimal `--n-cpu-moe` value parameter is
  used
- kv-cache quantized to Q8 when reasonable (ie. when we need to save some VRAM)
