// LOB Heatmap Compute Pipeline
// Vulkan 1.3 - Market Microstructure Renderer
// Configures compute shader dispatch parameters for order book heatmap generation

#include <vulkan/vulkan.h>

// Workgroup configuration for LOB heatmap compute shader
// Matches the shader's local_size_x and local_size_y declarations
constexpr uint32_t WORKGROUP_SIZE_X = 16;
constexpr uint32_t WORKGROUP_SIZE_Y = 16;
constexpr uint32_t WORKGROUP_SIZE_Z = 1;

// Pipeline configuration structure
struct LobHeatmapPipelineConfig {
    uint32_t local_size_x = WORKGROUP_SIZE_X;
    uint32_t local_size_y = WORKGROUP_SIZE_Y;
    uint32_t local_size_z = WORKGROUP_SIZE_Z;
};
