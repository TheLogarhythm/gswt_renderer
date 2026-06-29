# GSWT Renderer

This repository is a research extension built on top of _GSWT: Gaussian Splatting Wang Tiles_ ([Project page](https://yunfan.zone/gswt_webpage/)). Dynamic scenes use shared-LoD0 basis-bank playback produced by the current `gswt_constructor`.

## Introduction

This renderer extends a _Gaussian Splatting Wang Tiles_ renderer that runs on the web, using [wgpu](https://github.com/gfx-rs/wgpu) and WebGPU as the rendering backend. It contains specific functionalities and optimizations tailored to GSWT, including _procedural tiling_, _selective merging_, _LOD blending_, and dynamic scene playback. It takes a set of tiles produced by the GSWT constructor as input, and generates an infinitely expanding 3DGS terrain on-the-fly. The user can interact with the renderer and navigate the terrain in real time.

The original renderer discussed in the paper is developed with WebGL. This renderer is an improved version of the original one and might perform differently in certain cases.

## Getting Started

The original static GSWT renderer is available online: [Demo](https://yunfan.zone/gswt_webpage/demo/). It requires a web browser that supports WebGPU (e.g., latest Chrome).

A set of official static GSWT datasets can be found here: [Onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yzengbm_connect_ust_hk/IgB6p7U3s0FARoryHllz2jVuATc39DOnvkGZ9ieHkX_-hNw?e=Whq3DT)

To start the renderer:
1. Find a dataset and upload the zip file containing the set of tiles. After a moment of preprocessing (usually a few seconds), the config menu should show up.
2. Play with the config and click "Confirm". The renderer will switch to rendering stage and show the rendering menu.
3. Navigate the scene using **WASD** and hold **Space** to sprint. Use **IJKL** to look around.
4. There are some hotkeys to hide/unhide menus: **M** for main rendering menu and **P** for performance menu. For dynamic scenes, press **T** to freeze or resume motion playback.
5. Click "Reconfig" to go back to config menu.
6. Reload the webpage to switch to another scene.

There are quite a few config options in this renderer. The default config is used in most experiments in the paper, usually with a skybox texture and a proxy texture.

## Supported archives

Static archives contain a complete rectangular tile/LoD grid named `tile{tile}_lod{lod}.ply` or `tile{tile}_lod{lod}.splat` and no motion assets.

Dynamic archives use one shared LoD0 basis namespace and contain:

- `motion_basis_meta.bin` with `basis_scope: "shared_lod0"` and format version 1 or 2; version 2 carries a positive finite `duration_seconds`;
- exactly one `lod0_motion_basis.bin`;
- one `tile{tile}_lod{lod}_motion_basis_coeffs.bin` for every tile at every LoD;
- optionally, `motion_graph_basis.json` for graph playback and authoring features.

The metadata must describe all loaded LoDs and the current constructor policy (`basis_source_lod: 0`, direct-network teacher sampling, inclusive source times, and cubic-Hermite loop closure). Basis IDs in every coefficient file address the same shared bank directly.

An invalid optional motion graph is ignored with a warning; ordinary basis playback remains available. Required basis metadata, basis knots, and coefficient payloads are strict: missing, duplicated, mismatched, or out-of-range data stops archive initialization with an actionable error.

For current constructor version-1 archives, the renderer reads only the 28-byte `deformation_weights.bin` header and derives duration as `temporal_resolution * 2 / 30`. Version-1 basis, coefficient, and metadata versions must match. Version-2 archives use `duration_seconds` directly.

Legacy dynamic layouts are not supported. Rebuild them with the current constructor. In particular, the renderer rejects:

- per-LoD basis banks;
- dense Catmull-Rom motion archives;
- deformation-network-only archives containing `deformation_weights.bin` without shared-LoD0 basis assets;
- partially packaged basis archives.

The deformation network payload is never loaded or executed. Only its fixed header is read for version-1 duration compatibility; version-2 basis archives ignore it.
## Building locally

The renderer is written in Rust and targets WebAssembly (WASM). It is currently built with `rustc 1.92.0-nightly` and `wasm-pack 0.13.1`. They are required for building this project.


To build the project, run the following command:

```
wasm-pack build --target web
```

Then the package for release should be available in `./pkg`. It can be deployed on a web server together with `./index.html`. 

To test it locally, [sfz](https://github.com/weihanglo/sfz) is recommended. Run the following command to start a local server:

```
sfz -r --coi
```

## Credits

This renderer is derived from and heavily inspired by [Gauzilla](https://github.com/BladeTransformerLLC/gauzilla), under its [MIT License](https://github.com/BladeTransformerLLC/gauzilla?tab=MIT-1-ov-file). The egui rendering logic is derived from [this repository](https://github.com/kaphula/winit-egui-wgpu-template), under its [MIT License](https://github.com/kaphula/winit-egui-wgpu-template?tab=MIT-1-ov-file).

## BibTex

If you use the original GSWT method or datasets, please cite:

```
@inproceedings{Zeng:2025:gswt,
  author = {Zeng, Yunfan and Ma, Li and Sander, Pedro V.},
  title = {GSWT: Gaussian Splatting Wang Tiles},
  year = {2025},
  publisher = {Association for Computing Machinery},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  location = {Hong Kong, China},
  series = {SA '25}
}
```
