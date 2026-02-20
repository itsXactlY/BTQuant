#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require

/**
 * wick.vert - Vertex Shader for Candlestick Wick Rendering
 * 
 * This shader renders the wicks (shadows) of candlesticks as thin lines
 * extending from the body to the high and low prices.
 */

// Push constants
layout(push_constant) uniform PushConstants {
    vec4 bull_color;
    vec4 bear_color;
    vec4 wick_color;
    float time_scale;
    float price_scale;
    float time_offset;
    float price_offset;
} pc;

// Uniform buffer
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

// Instance data
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
    float alpha;
} vout;

// Wick vertices: 4 vertices per wick (2 for upper, 2 for lower)
// Each candlestick has 4 vertices total for both wicks
// Vertex 0: Upper wick start (body top)
// Vertex 1: Upper wick end (high)
// Vertex 2: Lower wick start (body bottom)
// Vertex 3: Lower wick end (low)

void main() {
    uint instance_idx = gl_InstanceIndex;
    CandlestickInstance candle = instances[instance_idx];
    
    bool is_bullish = candle.close >= candle.open;
    float body_top = max(candle.open, candle.close);
    float body_bottom = min(candle.open, candle.close);
    
    // Center X position
    float center_x = candle.time + candle.width * 0.5;
    
    vec2 position;
    
    // X position mapped to screen space
    float px_center_x = (center_x - pc.time_offset) * pc.time_scale;
    
    // Determine which wick vertex this is
    if (gl_VertexIndex == 0) {
        // Upper wick start (at body top)
        position = vec2(px_center_x, (body_top - pc.price_offset) * pc.price_scale);
    } else if (gl_VertexIndex == 1) {
        // Upper wick end (at high)
        position = vec2(px_center_x, (candle.high - pc.price_offset) * pc.price_scale);
    } else if (gl_VertexIndex == 2) {
        // Lower wick start (at body bottom)
        position = vec2(px_center_x, (body_bottom - pc.price_offset) * pc.price_scale);
    } else {
        // Lower wick end (at low)
        position = vec2(px_center_x, (candle.low - pc.price_offset) * pc.price_scale);
    }
    
    gl_Position = u.view_projection * vec4(position, 0.0, 1.0);
    
    // Use dedicated wick color or candle color
    vout.color = pc.wick_color.a > 0.0 ? pc.wick_color : (is_bullish ? pc.bull_color : pc.bear_color);
    vout.alpha = 1.0;
}
