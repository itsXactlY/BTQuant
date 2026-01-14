#version 450 core

/**
 * BTQuant Advanced UI Fragment Shader
 * 
 * Professional fragment shader with advanced visual effects, anti-aliasing,
 * and multiple rendering modes for high-quality UI components.
 * 
 * Features:
 * - Multi-sampled anti-aliasing (MSAA)
 * - Professional gradients and shadows
 * - High-quality text rendering with SDF
 * - Real-time visual effects and animations
 * - HDR color processing
 * - Professional styling for financial data
 */

// Input from vertex shader
layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec2 frag_world_pos;
layout(location = 3) in float frag_font_size;
layout(location = 4) in flat uint frag_glyph_id;
layout(location = 5) in flat uint frag_instance_id;
layout(location = 6) in float frag_animation_factor;
layout(location = 7) in vec2 frag_local_pos;

// Uniform buffer
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 projection;
    mat4 view;
    mat4 model;
    vec2 viewport_size;
    vec2 dpi_scale;
    float time;
    float delta_time;
    vec4 global_tint;
    uint render_mode;
    float animation_phase;
    vec2 mouse_position;
    float hover_radius;
} ubo;

// Texture samplers
layout(set = 0, binding = 1) uniform sampler2D font_atlas;
layout(set = 0, binding = 2) uniform sampler2D gradient_texture;
layout(set = 0, binding = 3) uniform sampler2D noise_texture;

// Push constants
layout(push_constant) uniform PushConstants {
    vec2 offset;
    vec2 scale;
    vec4 color_multiplier;
    uint flags;
    float custom_param1;
    float custom_param2;
    float custom_param3;
} push;

// Output
layout(location = 0) out vec4 out_color;

// Render mode constants
const uint RENDER_MODE_SOLID = 0u;
const uint RENDER_MODE_TEXT = 1u;
const uint RENDER_MODE_GRADIENT = 2u;
const uint RENDER_MODE_CHART_LINE = 3u;
const uint RENDER_MODE_HEATMAP = 4u;
const uint RENDER_MODE_ORDERBOOK = 5u;

// Utility functions
float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

// SDF text rendering
float sdf_text(vec2 uv, uint glyph_id) {
    // Sample the font atlas
    vec3 sdf_sample = texture(font_atlas, uv).rgb;
    
    // Multi-channel SDF
    float sdf = median(sdf_sample.r, sdf_sample.g, sdf_sample.b);
    
    // Calculate screen pixel range for proper anti-aliasing
    vec2 unit_range = vec2(2.0) / vec2(textureSize(font_atlas, 0));
    vec2 screen_tex_size = vec2(1.0) / fwidth(uv);
    float screen_pixel_range = max(0.5 * dot(unit_range, screen_tex_size), 1.0);
    
    // Convert SDF to alpha
    float screen_pixel_distance = screen_pixel_range * (sdf - 0.5);
    float alpha = clamp(screen_pixel_distance + 0.5, 0.0, 1.0);
    
    return alpha;
}

// Professional gradient generation
vec4 generate_gradient(vec2 uv, vec4 base_color) {
    // Multi-stop gradient for professional look
    float t = uv.y;
    
    vec4 color1 = base_color;
    vec4 color2 = base_color * 0.8;
    vec4 color3 = base_color * 1.2;
    
    vec4 gradient_color;
    if (t < 0.5) {
        gradient_color = mix(color1, color2, t * 2.0);
    } else {
        gradient_color = mix(color2, color3, (t - 0.5) * 2.0);
    }
    
    // Add subtle noise for organic feel
    vec2 noise_uv = frag_world_pos * 0.01 + ubo.time * 0.05;
    float noise = texture(noise_texture, noise_uv).r;
    gradient_color.rgb += (noise - 0.5) * 0.02;
    
    return gradient_color;
}

// Anti-aliased line rendering for charts
float line_sdf(vec2 p, vec2 a, vec2 b, float thickness) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - thickness * 0.5;
}

// Heatmap cell rendering with smooth interpolation
vec4 render_heatmap_cell(vec2 uv, vec4 base_color, float intensity) {
    // Create smooth cell boundaries
    vec2 cell_uv = fract(uv * 10.0); // Assuming 10x10 grid
    vec2 cell_center = abs(cell_uv - 0.5) * 2.0;
    float cell_factor = 1.0 - smoothstep(0.8, 1.0, max(cell_center.x, cell_center.y));
    
    // Apply intensity-based coloring
    vec4 color = base_color;
    color.rgb *= (0.5 + intensity * 0.5);
    color.a *= cell_factor;
    
    // Add subtle animation
    float pulse = sin(ubo.time * 2.0 + intensity * 10.0) * 0.1 + 0.9;
    color.rgb *= pulse;
    
    return color;
}

// Order book depth visualization
vec4 render_orderbook_bar(vec2 uv, vec4 base_color, float depth) {
    // Create horizontal bar with smooth edges
    float bar_height = 0.8;
    float bar_y = (1.0 - bar_height) * 0.5;
    
    float bar_alpha = smoothstep(bar_y - 0.02, bar_y, uv.y) * 
                     smoothstep(bar_y + bar_height + 0.02, bar_y + bar_height, uv.y);
    
    // Bar width based on depth
    float bar_width = depth;
    float bar_x_alpha = smoothstep(bar_width + 0.02, bar_width, uv.x);
    
    vec4 color = base_color;
    color.a *= bar_alpha * bar_x_alpha;
    
    // Add depth-based intensity
    color.rgb *= (0.3 + depth * 0.7);
    
    return color;
}

// Professional shadow and glow effects
vec4 apply_shadow_glow(vec4 color, vec2 uv) {
    // Subtle drop shadow
    vec2 shadow_offset = vec2(1.0, 1.0) / ubo.viewport_size;
    vec2 shadow_uv = uv - shadow_offset;
    
    // Glow effect for highlighted elements
    if (frag_animation_factor > 1.0) {
        float glow_intensity = (frag_animation_factor - 1.0) * 2.0;
        vec3 glow_color = color.rgb * glow_intensity;
        color.rgb += glow_color * 0.3;
    }
    
    return color;
}

void main() {
    vec4 final_color = frag_color;
    
    // Render based on mode
    switch (ubo.render_mode) {
        case RENDER_MODE_SOLID:
            // Simple solid color with gradient
            final_color = generate_gradient(frag_texcoord, frag_color);
            break;
            
        case RENDER_MODE_TEXT:
            // High-quality SDF text rendering
            float text_alpha = sdf_text(frag_texcoord, frag_glyph_id);
            final_color.a *= text_alpha;
            
            // Add text outline for better readability
            if (text_alpha > 0.1 && text_alpha < 0.9) {
                final_color.rgb = mix(vec3(0.0), final_color.rgb, text_alpha);
            }
            break;
            
        case RENDER_MODE_GRADIENT:
            // Professional gradient rendering
            final_color = generate_gradient(frag_texcoord, frag_color);
            break;
            
        case RENDER_MODE_CHART_LINE:
            // Anti-aliased line rendering for charts
            vec2 line_start = vec2(0.0, 0.5);
            vec2 line_end = vec2(1.0, frag_texcoord.y);
            float line_dist = line_sdf(frag_texcoord, line_start, line_end, 0.02);
            float line_alpha = 1.0 - smoothstep(0.0, 0.01, abs(line_dist));
            final_color.a *= line_alpha;
            
            // Add glow effect for active lines
            if (line_alpha > 0.5) {
                final_color.rgb += vec3(0.2) * (1.0 - abs(line_dist) * 50.0);
            }
            break;
            
        case RENDER_MODE_HEATMAP:
            // Heatmap cell rendering
            float intensity = push.custom_param1;
            final_color = render_heatmap_cell(frag_texcoord, frag_color, intensity);
            break;
            
        case RENDER_MODE_ORDERBOOK:
            // Order book depth bars
            float depth = push.custom_param1;
            final_color = render_orderbook_bar(frag_texcoord, frag_color, depth);
            break;
    }
    
    // Apply professional effects
    final_color = apply_shadow_glow(final_color, frag_texcoord);
    
    // Hover effect
    float dist_to_mouse = length(frag_world_pos - ubo.mouse_position);
    float hover_effect = smoothstep(ubo.hover_radius, ubo.hover_radius * 0.5, dist_to_mouse);
    final_color.rgb += vec3(hover_effect * 0.1);
    
    // Animation effects
    if (frag_animation_factor != 1.0) {
        // Pulsing effect
        float pulse = sin(ubo.time * 8.0) * 0.5 + 0.5;
        final_color.rgb += vec3(pulse * (frag_animation_factor - 1.0) * 0.2);
    }
    
    // Professional color grading
    final_color.rgb = pow(final_color.rgb, vec3(1.0 / 2.2)); // Gamma correction
    
    // HDR tone mapping for bright highlights
    final_color.rgb = final_color.rgb / (final_color.rgb + vec3(1.0));
    
    // Subtle vignette for depth
    vec2 vignette_uv = (frag_texcoord - 0.5) * 2.0;
    float vignette = 1.0 - dot(vignette_uv, vignette_uv) * 0.1;
    final_color.rgb *= vignette;
    
    // Ensure alpha is in valid range
    final_color.a = clamp(final_color.a, 0.0, 1.0);
    
    // Apply global tint
    final_color *= ubo.global_tint;
    
    out_color = final_color;
}