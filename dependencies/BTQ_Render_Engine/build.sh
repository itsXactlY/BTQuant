#!/bin/bash
# Compile shaders
echo "Compiling shaders..."
mkdir -p shaders
glslangValidator -V shaders/chart_lines.vert -o shaders/chart_lines.vert.spv
glslangValidator -V shaders/chart_lines.frag -o shaders/chart_lines.frag.spv
glslangValidator -V shaders/ui_vertex.vert -o shaders/ui_vertex.vert.spv
glslangValidator -V shaders/ui_fragment.frag -o shaders/ui_fragment.frag.spv
glslangValidator -V shaders/text_rendering.vert -o shaders/text_rendering.vert.spv
glslangValidator -V shaders/text_rendering.frag -o shaders/text_rendering.frag.spv

# Build project
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j$(nproc)
