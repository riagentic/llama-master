## Visuals related stuff of llama.master

- appp shows two memory visualizations, "Current Memory State" and "Projecte
  Memory State"
- "Current Memory State" is the current VRAM and RAM memory state
- "Projected Memory State" is visual representation VRAM and RAM after currently
  pre-selected model will buse used with pre-selected settings (ie. how the
  memory will look like when next model applies. It is basically current memory
  model - whatever llama.master is currently running (if anything) + projected
  model memory occupation
- in memory representations, whatever is (or will be) occupied with llama.master
  models should be highlighted or somehow recognizable from the other parts of
  memory (ie. memory used by other stuff + free memory)

## Status

- for pages that can run server, there is bigger and well visible status showing
  if server is running or stopped, basically the major status of the app,
  visible at the first sight, visually distinguishible and big enough to notice
