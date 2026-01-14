#version 450 core

/**
 * BTQuant Advanced Chart Line Vertex Shader
 * 
 * Specialized vertex shader for high-quality chart line rendering with
 * anti-aliasing, smooth curves, and professional visual effects.
 * 
 * Features:
 * - Anti-aliased line rendering with proper thickness
 * - Smooth curve interpolation for price charts
 * - Real-time data streaming support
 * - Professional styling with gradients
 * - Performance optimized for large datasets
 * - Multi-line support with different styles
 */

// Vertex attributes
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_direction;
layout(location = 2) in float in_thickness;
layout(location = 3) in vec4 in_color;
layout(location = 4) in float in_distance;
layout(location = 5) in uint in_line_id;

// Instance attributes for multi-line rendering
layout(location = 6) in vec2 in_line_offset;
layout(location = 7) in float in_line_scale;
layout(location = 8) in vec4 in_line_color;
layout(location = 9) in uint in_line_flags;

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

// Push constants for per-draw parameters
layout(push_constant) uniform PushConstants {
    vec2 chart_offset;
    vec2 chart_scale;
    vec4 color_multiplier;
    float thickness_multiplier;
    uint line_style;
    float dash_pattern;
    float glow_intensity;
} push;

// Output to fragment shader
layout(location = 0) out vec2 frag_line_coord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec2 frag_world_pos;
layout(location = 3) out float frag_thickness;
layout(location = 4) out float frag_distance_along_line;
layout(location = 5) out flat uint frag_line_id;
layout(location = 6) out vec2 frag_line_direction;
layout(location = 7) out float frag_glow_intensity;

// Line style constants
const uint LINE_STYLE_SOLID = 0u;
const uint LINE_STYLE_DASHED = 1u;
const uint LINE_STYLE_DOTTED = 2u;
const uint LINE_STYLE_GRADIENT = 3u;

// Utility functions
vec2 rotate2D(vec2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Smooth step function for animations
float smoothstep_custom(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

void main() {
    // Calculate effective line thickness
    float effective_thickness = in_thickness * ubo.line_thickness_scale * 
                               push.thickness_multiplier * in_line_scale;
    
    // Calculate line direction and normal
    vec2 line_direction = normalize(in_direction);
    vec2 line_normal = vec2(-line_direction.y, line_direction.x);
    
    // Calculate vertex position along the line
    vec2 local_pos = in_position;
    
    // Expand vertices perpendicular to line direction for thickness
    float half_thickness = effective_thickness * 0.5;
    vec2 thickness_offset = line_normal * half_thickness;
    
    // Add anti-aliasing padding
    float aa_padding = ubo.anti_alias_width;
    vec2 aa_offset = line_normal * aa_padding;
    
    // Determine which side of the line this vertex is on
    float side = sign(dot(local_pos, line_normal));
    vec2 expanded_pos = local_pos + thickness_offset * side + aa_offset * side;
    
    // Apply chart transformations
    vec2 chart_pos = expanded_pos * push.chart_scale + push.chart_offset + in_line_offset;
    
    // Map to chart bounds
    vec2 normalized_pos = (chart_pos - ubo.chart_bounds_min) / 
                         (ubo.chart_bounds_max - ubo.chart_bounds_min);
    
    // Transform to clip space
    vec4 clip_pos = ubo.projection * ubo.view * vec4(normalized_pos, 0.0, 1.0);
    gl_Position = clip_pos;
    
    // Calculate line-local coordinates for fragment shader
    frag_line_coord = vec2(in_distance, side * (half_thickness + aa_padding));
    
    // Color calculation based on line style and data
    vec4 line_color = in_color * in_line_color * push.color_multiplier;
    
    // Apply gradient coloring if enabled
    if ((in_line_flags & 0x1u) != 0u) {
        float gradient_t = (chart_pos.y - ubo.data_range.x) / (ubo.data_range.y - ubo.data_range.x);
        gradient_t = clamp(gradient_t, 0.0, 1.0);
        
        // Interpolate between gradient colors
        vec4 gradient_color;
        if (gradient_t < 0.33) {
            gradient_color = mix(ubo.gradient_colors[0], ubo.gradient_colors[1], gradient_t * 3.0);
        } else if (gradient_t < 0.66) {
            gradient_color = mix(ubo.gradient_colors[1], ubo.gradient_colors[2], (gradient_t - 0.33) * 3.0);
        } else {
            gradient_color = mix(ubo.gradient_colors[2], ubo.gradient_colors[3], (gradient_t - 0.66) * 3.0);
        }
        
        line_color = mix(line_color, gradient_color, 0.7);
    }
    
    // Apply animation effects
    if ((in_line_flags & 0x2u) != 0u) {
        float animation_factor = sin(ubo.time * 2.0 + in_distance * 0.01) * 0.5 + 0.5;
        line_color.rgb *= (0.8 + animation_factor * 0.4);
    }
    
    // Apply real-time data highlighting
    if ((in_line_flags & 0x4u) != 0u) {
        float highlight_intensity = exp(-ubo.time * 2.0) * sin(ubo.time * 10.0);
        line_color.rgb += vec3(highlight_intensity * 0.3);
    }
    
    frag_color = line_color;
    frag_world_pos = normalized_pos;
    frag_thickness = effective_thickness;
    frag_distance_along_line = in_distance;
    frag_line_id = in_line_id;
    frag_line_direction = line_direction;
    frag_glow_intensity = push.glow_intensity;
    
    // Ensure color is in valid range
    frag_color = clamp(frag_color, 0.0, 1.0);
}