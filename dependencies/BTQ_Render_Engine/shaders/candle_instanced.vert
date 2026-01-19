#version 450

struct CandleData {
    float x;
    float open;
    float high;
    float low;
    float close;
    uint color;
};

layout(std430, binding = 0) readonly buffer CandleBuffer {
    CandleData candles[];
} data;

layout(push_constant) uniform PushConstants {
    mat4 projection;
    vec2 chart_min;      // camera.offset_x, camera.offset_y
    vec2 chart_max;      // offset + (range / scale)
    float candle_width;  // 10.0 * scale_x
    uint chart_offset;   // Offset in the global buffer
    vec2 viewport_size;  // Width and height of the plot in pixels
    vec2 viewport_offset; // X and Y position of the plot in the swapchain
} pc;

layout(location = 0) out vec4 outColor;

const vec2 quad_pos[6] = vec2[](
    vec2(-0.5, 0.0), vec2(0.5, 0.0), vec2(0.5, 1.0),
    vec2(-0.5, 0.0), vec2(0.5, 1.0), vec2(-0.5, 1.0)
);

void main() {
    CandleData candle = data.candles[pc.chart_offset + gl_InstanceIndex];
    vec2 range = pc.chart_max - pc.chart_min;
    
    // 1. Transform World Time/Price to [0, 1] relative to current view
    float norm_x = (candle.x - pc.chart_min.x) / range.x;
    
    // 2. Identify vertex role (0-5: Body, 6-11: Wick)
    bool is_wick = gl_VertexIndex >= 6;
    vec2 local_pos = quad_pos[gl_VertexIndex % 6];
    
    float world_y_start, world_y_end, pixel_width;
    
    if (is_wick) {
        world_y_start = candle.low;
        world_y_end = candle.high;
        pixel_width = pc.candle_width * 0.15; // Thin wick
    } else {
        world_y_start = min(candle.open, candle.close);
        world_y_end = max(candle.open, candle.close);
        pixel_width = pc.candle_width;
    }
    
    // 3. Project to pixels
    float centerX = pc.viewport_offset.x + (norm_x * pc.viewport_size.x);
    float finalX = centerX + (local_pos.x * pixel_width);
    
    // Transform Y (invert Y since Vulkan is Y-down but plots are Y-up)
    float norm_y_start = (world_y_start - pc.chart_min.y) / range.y;
    float norm_y_end = (world_y_end - pc.chart_min.y) / range.y;
    
    // Invert norm_y for screenspace
    float finalY = pc.viewport_offset.y + ((1.0 - mix(norm_y_start, norm_y_end, local_pos.y)) * pc.viewport_size.y);

    // 4. Final Position
    gl_Position = pc.projection * vec4(finalX, finalY, 0.0, 1.0);
    
    // 5. Output Color
    outColor = vec4(
        float(candle.color & 0xFF) / 255.0,
        float((candle.color >> 8) & 0xFF) / 255.0,
        float((candle.color >> 16) & 0xFF) / 255.0,
        float((candle.color >> 24) & 0xFF) / 255.0
    );
}
