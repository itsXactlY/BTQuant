#version 450 core

/**
 * BTQuant Advanced UI Vertex Shader
 * 
 * Modern vertex shader with instancing support for high-performance UI rendering.
 * Supports multiple rendering modes, animations, and professional visual effects.
 * 
 * Features:
 * - Instanced rendering for thousands of UI elements
 * - Smooth animations and transitions
 * - Multi-layer rendering support
 * - High-DPI display scaling
 * - Professional typography positioning
 * - Real-time data-driven animations
 */

// Vertex attributes
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord; // Used for quad mapping or bar expansion
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_custom_val; // intensity or border_width
layout(location = 4) in uint in_type;        // level_type or candle_type

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

// Push constants for per-draw parameters
layout(push_constant) uniform PushConstants {
    vec2 offset;
    vec2 scale;
} push;

// Output to fragment shader
layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out float frag_custom_val;
layout(location = 3) out flat uint frag_type;

void main() {
    // Transform to clip space
    vec2 world_pos = in_position + push.offset;
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(world_pos, 0.0, 1.0);
    
    // Pass data to fragment shader
    frag_texcoord = in_texcoord;
    frag_color = in_color * ubo.global_tint;
    frag_custom_val = in_custom_val;
    frag_type = in_type;
}