# llama.cpp master

## General

- llama master is best app for llama.cpp written in aio framework (dep/aio)
- it provides perfect user experience for llama.cpp
  https://github.com/ggml-org/llama.cpp, resp. https://llama-cpp.com/
- app is 100% reliable, 100% correct, tested and providing optimal user
  experience

## Techstack

- It uses RS, Deno, Rust, WASM
- It is built on latest version of aio framework
- It's local electron app
- It uses whatever it needs to build llama.cpp

## Features

- It can build lama.cppp (llama-cli and llama-server)
- It provides all the settings in user-friendly way
- User can select master or specific tag version
- User can tweek llama for optimal usage on specific HW
  (https://www.youtube.com/watch?v=NIxCj-3fFNk)
- For each configuration user see visual representation of total VRAM and RAM
  and how model is placed there based on configuration

## Ui generla

- Ui is compact
- Ui has all the details
- Ui is user friendly
- Ui is intuitive
- Ui is has visual elements where suitable
- Ui is ergonomic
- Use progress bar whenever possible to inform user about the progress of any
  long-running job

## Ui

- It shows CPU details
- It shows GPU details
- It shows RAM details
- It shows VRAM details
- It shows CPU utilization (% and chart)
- It shows GPU utilization (% and chart)
- It shows CPU temperature (progress bar current/max)
- It shows GPU temperature (progress bar current/max)
- It shows RAM utilization (progress bar current/max)
- It shows VRAM utilization (progress bar current/max)
- User can build llama with one click
- User can run server with one click
- App detect available local models on disk with one button click
- Detected models can be run with llama
- With one button, optimal settings can be set
- All parameters are visible separaterly with explanatory
- Entire llama commands (server and client) are shown (ro) based on settings
- Start/stop server button is available
- Status of server runing is available
- Test chat, where user can connect and use running server is available
- Fonts are big enough so they can be read comfortably

## AI chat with running llama.cpp

- show TPS information

## Interractive Memory map

- Interactive memory map shows entire memory available RAM+VRAM
- RAM and VRAM is collored differently
- There is also legend explaining each area
- Curren tconfiguration is placed within and semi-opaque layer so it shows how
  it will run and what memory will be/is utilized by llama.cpp running selected
  model
- all is visually an aesthetically pleasing and representing real state

## Out of box experience

- App doesn't need any prerequisites except runing OS, it can download all it
  needs
- App shows used prerequisites
- App works out of box (just run the llama-master) and you are good to go, no
  preparations needed

## Build

- app is able to build llama.cpp just with one click
- app builds llama with optimal settings for specific PC it runs on
- app build is optimal for specific CPU(s), GPU(s) and overall specific PC
  setting where it is running

## Performance

- App provides optimall llama-cpp peformance
- Auto-settings has three modes: maximum performance, balanced and power mode
  (maximum performanc is default)
- Even on maximum performance, PC is usable and not frozen by llama-cpp running
- Each mode allows comfortable work during llama-cpp processing
- User is able to tweak settings futher but stability warning will be shown
- Build will run in optimal way, it will utilize as many CPU cores as possible -
  2 (those two cores should be left aside so PC is still usable during the build

## One page

- There is also one page where where all is in compact format, particurarly:
  cpu, gpu, ram, vram, model selection, representation how it fits on VRAM and
  RAM, start stop server button and simple chat interface with model and some
  TPS info, just all-in-one page

## Update

- App check (every 5 minutes) if new llama.cpp is available and if so offrers
  "Update" button to update it, if master build is built and new master is
  updated (user press "Update" button, it's automatically rebuilt. If running,
  it's also restarted.

## Models

- app detectcs lmstudio models (usually in ~/.lmstudio/models)
- app detect ollama local models (usually in /usr/share/ollama/.ollama/models)

## Builds

- llama.cpp buidls are buildable if possible (ie. there are no issues caused by
  local build setup)

## Prerequisites

- All prerequisites that are not met, there is "Fix" button next to it that will
  trigger installation, there is also "Fix all" button that will fix all unmet
  dependencies

## Builds

- User can successfully build and install CPU version (from source and prebuild
  release)
- User can successfully build and install CUDA version (from source and prebuild
  release)
- User can successfully build and install Vulkan version (from source and
  prebuild release)
- User can successfully build and install ROCm version (from source and prebuild
  release)
- User can successfully build and install Metal version (from source and
  prebuild release)

- Whatever solveable issues are occuring during the build, app provide way to
  fix it to finish the build and installation

- Whenever there is some issue that cannot be fixed automatically, app will
  provide precise and detailed explanation and steps for successful resolution
  (if possible)

## HW

- It works work with Nvidia cards
- It works with AMD cards
- It supports multiple card, for example two or more Nvidia cards
- It provides settings to enable which HW should be used. User can
  enable/disable any HW that Lllama.cpp should use
- It is able to use multiple GPUs for larger models
- It's able to use mix of GPU(s) and CPU(s)
- It suggest optimal configuration, ie configuration that will work flawlesly
  and provide maximum performance TPS
