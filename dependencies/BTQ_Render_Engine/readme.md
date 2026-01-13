sudo pacman -S --needed vulkan-headers vulkan-icd-loader vulkan-tools

cd ~/projects/BTQ_Render_Engine
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/hotspine_vulkan_demo