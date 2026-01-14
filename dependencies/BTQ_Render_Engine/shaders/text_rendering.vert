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
layout(location = 5) in uint in_font_id;

// Instance attributes for batch text rendering
layout(location = 6) in vec2 in_text_position;
layout(location = 7) in float in_text_scale;
layout(location = 8) in vec4 in_text_color;
layout(location = 9) in float in_text_rotation;
layout(location = 10) in uint in_text_flags;

// Uniform buffer
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 projection;
    mat4 view;
    vec2 viewport_size;
    vec2 dpi_scale;
    float time;
    vec4 global_text_color;
    vec2 shadow_offset;
    vec4 shadow_color;
    float outline_width;
    vec4 outline_color;
    uint render_flags;
} ubo;

// Font metrics buffer
layout(set = 0, binding = 1, std430) restrict readonly buffer FontMetrics {
    struct GlyphMetric {
        vec4 atlas_coords; // x, y, width, height in atlas
        vec2 bearing;      // Offset from baseline to glyph top-left
        float advance;     // Horizontal advance to next glyph
        float padding;
    } glyph_metrics[];
};

// Push constants for per-draw text parameters
layout(push_constant) uniform PushConstants {
    vec2 text_offset;
    float text_scale_factor;
    vec4 text_tint;
    uint text_effects;
    float effect_intensity;
    float kerning_adjustment;
    float line_height;
} push;

// Output to fragment shader
layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec2 frag_world_pos;
layout(location = 3) out float frag_font_size;
layout(location = 4) out flat uint frag_glyph_id;
layout(location = 5) out float frag_sdf_scale;
layout(location = 6) out vec2 frag_glyph_size;
layout(location = 7) out float frag_outline_width;
layout(location = 8) out vec4 frag_outline_color;

// Text effect functions
vec2 rotate2D(vec2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

float easeInOutQuad(float t) {
    return t < 0.5 ? 2.0 * t * t : 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0;
}

void main() {
    // Get glyph metrics
    GlyphMetric glyph = glyph_metrics[in_glyph_id];
    
    // Calculate base position and size
    vec2 glyph_size = glyph.atlas_coords.zw;
    vec2 bearing = glyph.bearing;
    
    // Apply DPI scaling
    float effective_font_size = in_font_size * ubo.dpi_scale.x * in_text_scale * push.text_scale_factor;
    vec2 scaled_size = glyph_size * effective_font_size;
    vec2 scaled_bearing = bearing * effective_font_size;
    
    // Calculate local glyph position
    vec2 local_pos = in_position * scaled_size + scaled_bearing;
    
    // Apply text rotation
    if (in_text_rotation != 0.0) {
        local_pos = rotate2D(local_pos, in_text_rotation);
    }
    
    // Calculate world position
    vec2 world_pos = in_text_position + local_pos + push.text_offset;
    
    // Apply text effects
    if ((in_text_flags & 0x1u) != 0u) { // Typewriter effect
        float typewriter_progress = push.effect_intensity;
        float char_reveal = smoothstep(0.0, 0.1, typewriter_progress - float(gl_InstanceIndex) * 0.05);
        world_pos.y += (1.0 - char_reveal) * 20.0;
    }
    
    if ((in_text_flags & 0x2u) != 0u) { // Wave effect
        float wave_offset = sin(ubo.time * 3.0 + world_pos.x * 0.01) * 5.0 * push.effect_intensity;
        world_pos.y += wave_offset;
    }
    
    if ((in_text_flags & 0x4u) != 0u) { // Shake effect
        vec2 shake = vec2(
            sin(ubo.time * 20.0 + world_pos.x * 0.1) * 2.0,
            cos(ubo.time * 25.0 + world_pos.y * 0.1) * 2.0
        ) * push.effect_intensity;
        world_pos += shake;
    }
    
    // Transform to clip space
    vec4 clip_pos = ubo.projection * ubo.view * vec4(world_pos, 0.0, 1.0);
    gl_Position = clip_pos;
    
    // Calculate texture coordinates in font atlas
    vec2 atlas_uv = glyph.atlas_coords.xy + in_texcoord * glyph.atlas_coords.zw;
    
    // Pass data to fragment shader
    frag_texcoord = atlas_uv;
    frag_color = in_color * in_text_color * push.text_tint * ubo.global_text_color;
    frag_world_pos = world_pos;
    frag_font_size = effective_font_size;
    frag_glyph_id = in_glyph_id;
    
    // Calculate SDF scale for proper anti-aliasing
    vec2 unit_range = vec2(2.0) / glyph_size;
    vec2 screen_tex_size = vec2(1.0) / fwidth(atlas_uv);
    frag_sdf_scale = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    
    frag_glyph_size = scaled_size;
    frag_outline_width = ubo.outline_width;
    frag_outline_color = ubo.outline_color;
    
    // Apply fade effects for smooth text animations
    if ((in_text_flags & 0x8u) != 0u) { // Fade in effect
        float fade_progress = easeInOutQuad(clamp(push.effect_intensity, 0.0, 1.0));
        frag_color.a *= fade_progress;
    }
    
    // Ensure color is in valid range
    frag_color = clamp(frag_color, 0.0, 1.0);
}