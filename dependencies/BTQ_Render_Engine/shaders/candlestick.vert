#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require

/**
 * candlestick.vert - Vertex Shader for Hardware-Instanced Candlestick Rendering
 * 
 * This shader renders candlesticks using hardware instancing. Each instance
 * represents one candlestick with OHLC data passed as instance attributes.
 * 
 * The shader generates:
 * - Candle body (rectangle between open and close)
 * - Upper wick (line from high to max(open, close))
 * - Lower wick (line from low to min(open, close))
 */

// Push constants for per-frame updates
layout(push_constant) uniform PushConstants {
    vec4 bull_color;       // Bullish candle color (green)
    vec4 bear_color;       // Bearish candle color (red)
    vec4 wick_color;       // Wick color
    float time_scale;      // Time axis scale
    float price_scale;     // Price axis scale
    float time_offset;     // Time axis offset
    float price_offset;    // Price axis offset
} pc;

// Uniform buffer for view-projection matrix
layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view_projection;
    vec2 resolution;
    vec2 time_range;
    vec2 price_range;
    float candle_width;
    float wick_width;
    uint candle_count;
    uint padding;
} u;

// Instance data (per-candlestick)
struct CandlestickInstance {
    float open;
    float high;
    float low;
    float close;
    float time;
    float width;
    uint color;
    uint flags;
};

layout(set = 0, binding = 1) readonly buffer InstanceBuffer {
    CandlestickInstance instances[];
};

// Vertex output
layout(location = 0) out VertexOutput {
    vec4 color;
    vec2 uv;
    flat uint flags;
} vout;

// Vertex indices for a candlestick (6 vertices = 2 triangles for body)
// We use a unit quad and transform it based on OHLC data
const vec2 VERTICES[6] = vec2[](
    vec2(-0.5, 0.0),  // Bottom-left
    vec2(0.5, 0.0),   // Bottom-right
    vec2(-0.5, 1.0),  // Top-left
    vec2(-0.5, 1.0),  // Top-left
    vec2(0.5, 0.0),   // Bottom-right
    vec2(0.5, 1.0)    // Top-right
);

void main() {
    uint instance_idx = gl_InstanceIndex;
    CandlestickInstance candle = instances[instance_idx];
    
    // Determine if bullish or bearish
    bool is_bullish = candle.close >= candle.open;
    
    // Calculate body bounds
    float body_bottom = min(candle.open, candle.close);
    float body_top = max(candle.open, candle.close);
    float body_height = body_top - body_bottom;
    
    // Prevent zero-height bodies
    if (body_height < 0.0001) {
        body_height = 0.0001;
    }
    
    // Get vertex position (unit quad)
    vec2 vertex = VERTICES[gl_VertexIndex];
    
    // Transform vertex to candlestick position
    vec2 position;
    
    // X position: time coordinate mapped to screen space
    // candle.time is world time. We subtract offset and multiply by scale.
    float center_x = (candle.time - pc.time_offset) * pc.time_scale;
    // The width is also in time units? Usually candle width is a fraction of the timeframe.
    // If width is in world units (e.g., seconds), we scale it. Wait, the generic setup is just scaling unit width.
    float width_screen = candle.width * pc.time_scale;
    // if width is given in world units, (vertex.x) * width_screen gives the pixel width.
    // let's assume candle.width is in world units (e.g. 0.8 * timeframe)
    position.x = center_x + (vertex.x) * width_screen;
    
    // Y position: price coordinate mapped to screen space
    float bottom_screen = (body_bottom - pc.price_offset) * pc.price_scale;
    float height_screen = body_height * pc.price_scale;
    position.y = bottom_screen + vertex.y * height_screen;
    
    // Apply view-projection transformation (Maps screen pixels -> NDC)
    vec4 clip_pos = u.view_projection * vec4(position, 0.0, 1.0);
    
    gl_Position = clip_pos;
    
    // Pass color to fragment shader
    vout.color = is_bullish ? pc.bull_color : pc.bear_color;
    vout.uv = vertex;
    vout.flags = candle.flags;
}
