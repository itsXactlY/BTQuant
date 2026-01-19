#version 450

layout(push_constant) uniform PushConstants {
    mat4 projection;
    vec2 chart_min;
    vec2 chart_max;
    float candle_width;
    uint chart_offset;
    vec2 viewport_size;
    vec2 viewport_offset;
} pc;

struct WAEData {
    float up;
    float down;
    float explosion;
    float dead_zone;
};

layout(std430, binding = 0) readonly buffer InputWAE {
    WAEData wae_data[];
};

layout(location = 0) out vec4 outColor;

void main() {
    uint candle_idx = gl_InstanceIndex;
    uint vertex_idx = gl_VertexIndex;
    
    WAEData d = wae_data[pc.chart_offset + candle_idx];
    
    // We want to render:
    // 1. Up trend bar (Green)
    // 2. Down trend bar (Red)
    // 3. Explosion line (Yellow)
    // 4. Dead zone line (White/Gray)
    
    // Since we only have one draw call, we can use vertex_idx to differentiate.
    // Let's say:
    // 0-5: Up trend bar (quad)
    // 6-11: Down trend bar (quad)
    
    // Position calculation:
    float x = float(candle_idx); // Relative X
    float range_x = pc.chart_max.x - pc.chart_min.x;
    float norm_x = (x - pc.chart_min.x) / range_x;
    float pixel_x = pc.viewport_offset.x + norm_x * pc.viewport_size.x;
    
    // Y scaling for indicators (this is tricky as they have different Y scale than price)
    // For now, let's normalize them to the bottom 30% of the viewport or use a fixed height
    float plot_height = pc.viewport_size.y * 0.3; // 30% of chart height
    float plot_bottom = pc.viewport_offset.y + pc.viewport_size.y;
    
    vec2 pos = vec2(0.0);
    vec4 color = vec4(1.0);
    
    // Bar geometry
    float half_w = pc.candle_width * 0.5;
    
    if (vertex_idx < 6) { // Up Trend Bar
        float val = d.up;
        color = vec4(0.0, 1.0, 0.0, 0.8);
        
        vec2 vertices[6] = vec2[](
            vec2(-half_w, 0.0), vec2(half_w, 0.0), vec2(-half_w, -val),
            vec2(-half_w, -val), vec2(half_w, 0.0), vec2(half_w, -val)
        );
        pos = vertices[vertex_idx % 6];
        pos.y = plot_bottom + pos.y * (plot_height / 100.0); // Assume 0-100 range for WAE
        pos.x += pixel_x;
    } else { // Down Trend Bar
        float val = d.down;
        color = vec4(1.0, 0.0, 0.0, 0.8);
        
        vec2 vertices[6] = vec2[](
            vec2(-half_w, 0.0), vec2(half_w, 0.0), vec2(-half_w, -val),
            vec2(-half_w, -val), vec2(half_w, 0.0), vec2(half_w, -val)
        );
        pos = vertices[vertex_idx % 6];
        pos.y = plot_bottom + pos.y * (plot_height / 100.0);
        pos.x += pixel_x;
    }
    
    gl_Position = pc.projection * vec4(pos, 0.0, 1.0);
    outColor = color;
}
