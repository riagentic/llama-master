## lllama.master context releated stuff

- app has buttons to pre-select context size set context size to these standard
  sizes: 16k, 32k, 64k, 128k, 256k, 512k, 1M
- user can set any context size number up to fixed upper limit is 10k and lower
  limit 8k
- there is button to set context size to model optimal (whatever it is), model
  optimal is the highest context size where model still don't degrade (no
  observable context rot, etc.)

## Major context sizis

- app add multiple buttons for context settings: "Min CTX Size", "Opt CTX Size",
  "Big CTX Size", "Max CTX Size"
- Min CTX size is minimal context size that the model can still work with
- Opt CTX size is the maximum context size that model can still work with 100%
  quality (no degradation"
- Large CTX size is size where model can work on 90% of its quality without
  further context degradation"
- Max CTX size is maximum what model can handle without halucination reagrdless
  the quality degradation

## Visuals

- Thre is visual representation of usable range of context size for each model
  showing all ranges: Min, Opt, Big, Max
