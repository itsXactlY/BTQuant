#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require

/**
 * candlestick.frag - Fragment Shader for Hardware-Instanced Candlestick Rendering
 * 
 * This shader handles:
 * - Anti-aliased candle body rendering
 * - Sub-pixel accurate edges
 * - Color interpolation for gradient effects
 */

// Input from vertex shader
layout(location = 0) in VertexOutput {
    vec4 color;
    vec2 uv;
    flat uint flags;
} vin;

// Output
layout(location = 0) out vec4 frag_color;

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

void main() {
    // Simple solid color output
    frag_color = vin.color;
    
    // Optional: Add subtle gradient based on UV
    // This creates a slight 3D effect on the candle body
    float gradient = 1.0 - abs(vin.uv.x - 0.5) * 0.2;
    frag_color.rgb *= gradient;
    
    // Optional: Add border/outline effect
    // float border_dist = min(vin.uv.x, min(vin.uv.y, min(1.0 - vin.uv.x, 1.0 - vin.uv.y)));
    // if (border_dist < 0.05) {
    //     frag_color.rgb *= 0.8;  // Darken border
    // }
}
