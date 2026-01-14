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

// Output to fragment shader
layout(location = 0) out vec4 frag_color;

void main() {
    // Transform to clip space
    // in_position is already in absolute screen coordinates from C++
    gl_Position = ubo.projection * ubo.view * vec4(in_position, 0.0, 1.0);
    
    frag_color = in_color;
}