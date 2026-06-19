# BTQuant Vulkan Compute Shaders

Compute kernels for GPU-accelerated market data aggregation.

## Files

- `heatmap.comp` — Aggregates trade stream into a 256×256 2D heatmap
  (X-axis: time bins, Y-axis: price bins, intensity: volume).

## Build

Compiled at CMake configure time via `glslc` into `shaders/spirv/*.comp.spv`.
The `.spv` binaries are embedded as C arrays in `src/renderer/embedded_shaders.cpp`
and consumed at runtime — no runtime shader compilation needed.

If `glslc` is not found, the build fails with a clear error.

## Add a new shader

1. Write `shaders/<name>.comp` (compute shader with `#version 450`).
2. Add `SHADER_COMPILE(<name>)` line in `shaders/CMakeLists.txt`.
3. In `src/renderer/embedded_shaders.cpp`, add a `static const uint32_t
   <NAME>_SPIRV[]` array (use `tools/embed_shader.sh <name>.spv` to generate).
4. Reference the shader via `vkCreateShaderModule(device, &createInfo, ...)`
   where the SPIRV pointer is `embedded_shaders::<NAME>_SPIRV`.
