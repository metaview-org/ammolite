# ammolite
Physically based glTF rendering with Vulkan in Rust

This repository contains highly experimental code, proceed with caution.

## Design goals

* Up-to-date real-time physically based rendering techniques
* Support for glTF
* Thin layer on top of [vulkano](https://github.com/vulkano-rs/vulkano)
* Expose the [vulkano](https://github.com/vulkano-rs/vulkano) API to allow for high customizability
* Easily switch between rendering techniques (multiple BRDF implementations)
