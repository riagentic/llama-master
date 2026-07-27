# llama.master tune

- optimal settings tuning is smart, ie it assumes real scenairos like OS VRAM
  usage fluctuation and other scenarios
- optimal settings always leave some VRAM and RAM available for OS and apps and
  various common situation
- otpimal settings is tuned in a way it never jeopardize OS stability
- optimal settings counts everything precisely, especially differnt model
  architectures and context memory demands based on model type and other
  settings
- optimal settings always work and is realiable (model doesn't crash)
- optimal settings is fine tuned according to model selected and hw spec it's
  runing on
- settings to run optmail settings by default is visible in gui and it's turned
  on by default (can be switche off)
- when optimal settings is auto, server setting when server starts is
  automatically auto
- for bigger models (which cannot fit into VRAM), optimal hybrid settings is
  applied when optimal settings is triggered or set to auto-use
