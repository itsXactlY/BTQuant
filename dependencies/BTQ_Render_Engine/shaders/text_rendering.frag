#version 450 core

/**
 * BTQuant Advanced Text Rendering Fragment Shader
 * 
 * High-quality SDF text rendering with professional typography effects,
 * outlines, shadows, and multi-channel distance field support.
 * 
 * Features:
 * - Multi-channel SDF for crisp text rendering
 * - Professional text effects (outline, shadow, glow)
 * - Sub-pixel rendering for LCD displays
 * - Gamma correction and proper anti-aliasing
 * - Real-time text animations and effects
 */

// Input from vertex shader
layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec2 frag_world_pos;
layout(location = 3) in float frag_font_size;
layout(location = 4) in flat uint frag_glyph_id;
layout(location = 5) in float frag_sdf_scale;
layout(location = 6) in vec2 frag_glyph_size;
layout(location = 7) in float frag_outline_width;
layout(location = 8) in vec4 frag_outline_color;

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

// Font atlas texture
layout(set = 0, binding = 2) uniform sampler2D font_atlas;

// Push constants
layout(push_constant) uniform PushConstants {
    vec2 text_offset;
    float text_scale_factor;
    vec4 text_tint;
    uint text_effects;
    float effect_intensity;
    float kerning_adjustment;
    float line_height;
} push;

// Output
layout(location = 0) out vec4 out_color;

// Render flags
const uint RENDER_SHADOW = 0x1u;
const uint RENDER_OUTLINE = 0x2u;
const uint RENDER_GLOW = 0x4u;
const uint RENDER_SUBPIXEL = 0x8u;
const uint RENDER_GAMMA_CORRECT = 0x10u;

// Text effects flags
const uint EFFECT_TYPEWRITER = 0x1u;
const uint EFFECT_WAVE = 0x2u;
const uint EFFECT_SHAKE = 0x4u;
const uint EFFECT_FADE = 0x8u;
const uint EFFECT_PULSE = 0x10u;
const uint EFFECT_RAINBOW = 0x20u;

// Utility functions
float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

// Multi-channel SDF sampling
float sample_sdf(vec2 uv) {
    vec3 sdf_sample = texture(font_atlas, uv).rgb;
    return median(sdf_sample.r, sdf_sample.g, sdf_sample.b);
}

// Screen pixel distance calculation for proper anti-aliasing
float screen_pixel_distance(float sdf, vec2 uv) {
    vec2 unit_range = vec2(2.0) / textureSize(font_atlas, 0);
    vec2 screen_tex_size = vec2(1.0) / fwidth(uv);
    float screen_pixel_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    return screen_pixel_range * (sdf - 0.5);
}

// Sub-pixel rendering for LCD displays
vec3 sample_subpixel(vec2 uv) {
    vec2 pixel_size = 1.0 / textureSize(font_atlas, 0);
    
    // Sample at sub-pixel offsets for RGB
    float r = sample_sdf(uv + vec2(-pixel_size.x / 3.0, 0.0));
    float g = sample_sdf(uv);
    float b = sample_sdf(uv + vec2(pixel_size.x / 3.0, 0.0));
    
    return vec3(r, g, b);
}

// Gamma correction
vec3 gamma_correct(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

// Rainbow color effect
vec3 rainbow_color(float t) {
    t = fract(t);
    vec3 color;
    
    if (t < 1.0/6.0) {
        color = vec3(1.0, 6.0 * t, 0.0);
    } else if (t < 2.0/6.0) {
        color = vec3(2.0 - 6.0 * t, 1.0, 0.0);
    } else if (t < 3.0/6.0) {
        color = vec3(0.0, 1.0, 6.0 * t - 2.0);
    } else if (t < 4.0/6.0) {
        color = vec3(0.0, 4.0 - 6.0 * t, 1.0);
    } else if (t < 5.0/6.0) {
        color = vec3(6.0 * t - 4.0, 0.0, 1.0);
    } else {
        color = vec3(1.0, 0.0, 6.0 - 6.0 * t);
    }
    
    return color;
}

void main() {
    vec4 final_color = vec4(0.0);
    
    // Sample the SDF
    float sdf = sample_sdf(frag_texcoord);
    float screen_dist = screen_pixel_distance(sdf, frag_texcoord);
    
    // Base text alpha
    float text_alpha = clamp(screen_dist + 0.5, 0.0, 1.0);
    
    // Render shadow if enabled
    if ((ubo.render_flags & RENDER_SHADOW) != 0u && length(ubo.shadow_offset) > 0.0) {
        vec2 shadow_uv = frag_texcoord + ubo.shadow_offset / frag_glyph_size;
        float shadow_sdf = sample_sdf(shadow_uv);
        float shadow_dist = screen_pixel_distance(shadow_sdf, shadow_uv);
        float shadow_alpha = clamp(shadow_dist + 0.5, 0.0, 1.0);
        
        // Blend shadow
        vec4 shadow = ubo.shadow_color * shadow_alpha;
        final_color = mix(final_color, shadow, shadow.a);
    }
    
    // Render outline if enabled
    if ((ubo.render_flags & RENDER_OUTLINE) != 0u && frag_outline_width > 0.0) {
        float outline_threshold = 0.5 - frag_outline_width / frag_sdf_scale;
        float outline_alpha = clamp(screen_dist + 0.5 - outline_threshold, 0.0, 1.0);
        
        // Blend outline
        vec4 outline = frag_outline_color * outline_alpha;
        final_color = mix(final_color, outline, outline.a * (1.0 - text_alpha));
    }
    
    // Main text color
    vec4 text_color = frag_color;
    
    // Apply text effects
    if ((push.text_effects & EFFECT_PULSE) != 0u) {
        float pulse = sin(ubo.time * 4.0) * 0.5 + 0.5;
        text_color.rgb *= (0.7 + pulse * 0.6);
    }
    
    if ((push.text_effects & EFFECT_RAINBOW) != 0u) {
        float rainbow_t = ubo.time * 0.5 + frag_world_pos.x * 0.01;
        text_color.rgb = rainbow_color(rainbow_t);
    }
    
    if ((push.text_effects & EFFECT_FADE) != 0u) {
        text_color.a *= push.effect_intensity;
    }
    
    // Sub-pixel rendering for LCD displays
    if ((ubo.render_flags & RENDER_SUBPIXEL) != 0u) {
        vec3 subpixel_alpha = sample_subpixel(frag_texcoord);
        
        // Convert to screen distances
        subpixel_alpha.r = clamp(screen_pixel_distance(subpixel_alpha.r, frag_texcoord) + 0.5, 0.0, 1.0);
        subpixel_alpha.g = clamp(screen_pixel_distance(subpixel_alpha.g, frag_texcoord) + 0.5, 0.0, 1.0);
        subpixel_alpha.b = clamp(screen_pixel_distance(subpixel_alpha.b, frag_texcoord) + 0.5, 0.0, 1.0);
        
        // Apply sub-pixel alpha
        text_color.rgb *= subpixel_alpha;
        text_alpha = (subpixel_alpha.r + subpixel_alpha.g + subpixel_alpha.b) / 3.0;
    }
    
    // Blend main text
    text_color.a *= text_alpha;
    final_color = mix(final_color, text_color, text_color.a);
    
    // Render glow effect if enabled
    if ((ubo.render_flags & RENDER_GLOW) != 0u) {
        float glow_size = 4.0; // Glow radius in pixels
        float glow_threshold = 0.5 - glow_size / frag_sdf_scale;
        float glow_alpha = clamp((screen_dist - glow_threshold) / glow_size, 0.0, 1.0);
        glow_alpha = 1.0 - glow_alpha;
        glow_alpha = pow(glow_alpha, 2.0); // Smooth falloff
        
        vec3 glow_color = text_color.rgb * 0.8;
        final_color.rgb += glow_color * glow_alpha * 0.3;
    }
    
    // Apply gamma correction if enabled
    if ((ubo.render_flags & RENDER_GAMMA_CORRECT) != 0u) {
        final_color.rgb = gamma_correct(final_color.rgb, 2.2);
    }
    
    // Ensure alpha is in valid range
    final_color.a = clamp(final_color.a, 0.0, 1.0);
    
    // Discard fully transparent pixels for better performance
    if (final_color.a < 0.01) {
        discard;
    }
    
    out_color = final_color;
}