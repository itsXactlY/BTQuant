#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <type_traits>
#include <cstdint>

namespace BTQuant {
namespace Rendering {

// 1. Entspricht exakt: struct CandlestickInstance in candlestick.vert
// Größe: 8 floats/uints * 4 Bytes = 32 Bytes pro Candle
struct alignas(16) CandlestickInstance {
    float open;
    float high;
    float low;
    float close;
    float time;
    float width;
    uint32_t color;
    uint32_t flags;
};
static_assert(sizeof(CandlestickInstance) == 32, "CandlestickInstance Memory Alignment ist falsch!");

// 2. Entspricht exakt: layout(set = 0, binding = 0) uniform Uniforms
// Standard std140 Alignment
struct alignas(16) ChartUniforms {
    glm::mat4 view_projection; // 64 Bytes (Offset 0)
    glm::vec2 resolution;      // 8 Bytes  (Offset 64)
    glm::vec2 time_range;      // 8 Bytes  (Offset 72)
    glm::vec2 price_range;     // 8 Bytes  (Offset 80)
    float candle_width;        // 4 Bytes  (Offset 88)
    float wick_width;          // 4 Bytes  (Offset 92)
    uint32_t candle_count;     // 4 Bytes  (Offset 96)
    uint32_t padding;          // 4 Bytes  (Offset 100)
};
static_assert(sizeof(ChartUniforms) == 112 || sizeof(ChartUniforms) == 128, "Uniform Buffer Alignment prüfen!");

// 3. Entspricht exakt: layout(push_constant) uniform PushConstants
// Größe: 3 * 16 + 4 * 4 = 64 Bytes. Passt perfekt in das 128-Byte Vulkan-Limit.
struct ChartPushConstants {
    glm::vec4 bull_color;
    glm::vec4 bear_color;
    glm::vec4 wick_color;
    float time_scale;
    float price_scale;
    float time_offset;
    float price_offset;
};
static_assert(sizeof(ChartPushConstants) == 64, "PushConstants Size Mismatch!");

} // namespace Rendering
} // namespace BTQuant