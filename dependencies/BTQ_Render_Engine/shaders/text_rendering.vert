#version 450 core

/**
 * BTQuant Advanced Text Rendering Vertex Shader
 * 
 * High-quality text rendering vertex shader with SDF (Signed Distance Field)
 * support, professional typography, and multi-font capabilities.
 * 
 * Features:
 * - SDF text rendering for crisp text at any scale
 * - Multi-font support with font atlas
 * - Professional typography with kerning
 * - High-DPI display support
 * - Text effects and animations
 * - Efficient instanced rendering
 */

// Vertex attributes
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;
layout(location = 2) in vec4 in_color;
layout(location = 3) in uint in_glyph_id;
layout(location = 4) in float in_font_size;

// Uniform buffer
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 projection;
    mat4 view;
    vec2 viewport_size;
    vec2 dpi_scale;
    float time;
    vec4 global_text_color;
    uint render_flags;
} ubo;

struct GlyphMetric {
    vec4 atlas_coords; // x, y, width, height in atlas
    vec2 bearing;      // Offset from baseline to glyph top-left
    float advance;     // Horizontal advance to next glyph
    float padding;
};

// Font metrics buffer
layout(set = 0, binding = 1, std430) restrict readonly buffer FontMetrics {
    GlyphMetric glyph_metrics[];
};

// Push constants for per-draw text parameters
layout(push_constant) uniform PushConstants {
    vec2 text_offset;
    float text_scale_factor;
} push;

// Output to fragment shader
layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out flat uint frag_glyph_id;

void main() {
    // Transform to clip space
    // in_position is already in absolute screen coordinates from C++
    vec2 world_pos = in_position + push.text_offset;
    gl_Position = ubo.projection * ubo.view * vec4(world_pos, 0.0, 1.0);
    
    // Get glyph metrics for atlas mapping
    GlyphMetric glyph = glyph_metrics[in_glyph_id];
    
    // Calculate texture coordinates in font atlas
    // in_texcoord is 0..1 for the quad
    frag_texcoord = glyph.atlas_coords.xy + in_texcoord * glyph.atlas_coords.zw;
    
    // Pass color to fragment shader
    frag_color = in_color * ubo.global_text_color;
    frag_glyph_id = in_glyph_id;
}