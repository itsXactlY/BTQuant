#version 450 core

/**
 * BTQuant Advanced Chart Line Fragment Shader
 * 
 * High-quality fragment shader for chart line rendering with anti-aliasing,
 * professional styling, and advanced visual effects.
 * 
 * Features:
 * - Perfect anti-aliasing for smooth lines
 * - Dashed and dotted line patterns
 * - Gradient effects and glow
 * - Real-time data highlighting
 * - Professional styling for financial charts
 */

// Input from vertex shader
layout(location = 0) in vec2 frag_line_coord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec2 frag_world_pos;
layout(location = 3) in float frag_thickness;
layout(location = 4) in float frag_distance_along_line;
layout(location = 5) in flat uint frag_line_id;
layout(location = 6) in vec2 frag_line_direction;
layout(location = 7) in float frag_glow_intensity;

// Uniform buffer
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 projection;
    mat4 view;
    vec2 viewport_size;
    vec2 chart_bounds_min;
    vec2 chart_bounds_max;
    vec2 data_range;
    float time;
    float line_thickness_scale;
    float anti_alias_width;
    uint render_mode;
    vec4 gradient_colors[4];
    float animation_phase;
} ubo;

// Push constants
layout(push_constant) uniform PushConstants {
    vec2 chart_offset;
    vec2 chart_scale;
    vec4 color_multiplier;
    float thickness_multiplier;
    uint line_style;
    float dash_pattern;
    float glow_intensity;
} push;

// Output
layout(location = 0) out vec4 out_color;

// Line style constants
const uint LINE_STYLE_SOLID = 0u;
const uint LINE_STYLE_DASHED = 1u;
const uint LINE_STYLE_DOTTED = 2u;
const uint LINE_STYLE_GRADIENT = 3u;

// Utility functions
float smoothstep_custom(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Anti-aliased line distance function
float line_distance(vec2 coord, float thickness) {
    float distance_from_center = abs(coord.y);
    float half_thickness = thickness * 0.5;
    
    // Anti-aliased edge
    return 1.0 - smoothstep(half_thickness - ubo.anti_alias_width, 
                           half_thickness + ubo.anti_alias_width, 
                           distance_from_center);
}

// Dashed line pattern
float dash_pattern(float distance, float dash_length, float gap_length) {
    float total_length = dash_length + gap_length;
    float position = mod(distance, total_length);
    return smoothstep(0.0, 1.0, step(position, dash_length));
}

// Dotted line pattern
float dot_pattern(float distance, float dot_spacing) {
    float position = mod(distance, dot_spacing);
    float dot_radius = dot_spacing * 0.2;
    float center_distance = abs(position - dot_spacing * 0.5);
    return 1.0 - smoothstep(dot_radius - 1.0, dot_radius + 1.0, center_distance);
}

// Glow effect
float glow_effect(vec2 coord, float thickness, float glow_size) {
    float distance_from_center = abs(coord.y);
    float glow_radius = thickness * 0.5 + glow_size;
    
    float glow = 1.0 - smoothstep(0.0, glow_radius, distance_from_center);
    return pow(glow, 2.0); // Smooth falloff
}

void main() {
    vec4 final_color = frag_color;
    
    // Calculate base line alpha using anti-aliased distance
    float line_alpha = line_distance(frag_line_coord, frag_thickness);
    
    // Apply line style patterns
    float pattern_alpha = 1.0;
    
    switch (push.line_style) {
        case LINE_STYLE_SOLID:
            // No pattern modification needed
            break;
            
        case LINE_STYLE_DASHED:
            pattern_alpha = dash_pattern(frag_distance_along_line, 
                                       push.dash_pattern * 10.0, 
                                       push.dash_pattern * 5.0);
            break;
            
        case LINE_STYLE_DOTTED:
            pattern_alpha = dot_pattern(frag_distance_along_line, 
                                      push.dash_pattern * 8.0);
            break;
            
        case LINE_STYLE_GRADIENT:
            // Gradient along the line
            float gradient_t = mod(frag_distance_along_line * 0.01, 1.0);
            vec4 gradient_color = mix(ubo.gradient_colors[0], ubo.gradient_colors[1], gradient_t);
            final_color = mix(final_color, gradient_color, 0.5);
            break;
    }
    
    // Combine line alpha with pattern alpha
    final_color.a *= line_alpha * pattern_alpha;
    
    // Add glow effect if enabled
    if (frag_glow_intensity > 0.0) {
        float glow = glow_effect(frag_line_coord, frag_thickness, frag_glow_intensity * 5.0);
        vec3 glow_color = final_color.rgb * 0.8;
        final_color.rgb += glow_color * glow * frag_glow_intensity * 0.3;
    }
    
    // Add subtle animation effects
    float animation_intensity = sin(ubo.time * 3.0 + frag_distance_along_line * 0.01) * 0.5 + 0.5;
    final_color.rgb += vec3(animation_intensity * 0.05);
    
    // Professional edge enhancement for crisp lines
    if (line_alpha > 0.1 && line_alpha < 0.9) {
        // Add subtle edge highlighting
        float edge_factor = 1.0 - abs(line_alpha - 0.5) * 2.0;
        final_color.rgb += vec3(edge_factor * 0.1);
    }
    
    // Ensure color is in valid range
    final_color = clamp(final_color, 0.0, 1.0);
    
    // Discard fully transparent pixels for better performance
    if (final_color.a < 0.01) {
        discard;
    }
    
    out_color = final_color;
}