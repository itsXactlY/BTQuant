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
layout(location = 1) in vec2 in_texcoord;
layout(location = 2) in vec4 in_color;
layout(location = 3) in uint in_instance_id;
layout(location = 4) in float in_font_size;
layout(location = 5) in uint in_glyph_id;

// Instance attributes (for instanced rendering)
layout(location = 6) in vec2 in_instance_position;
layout(location = 7) in vec2 in_instance_size;
layout(location = 8) in vec4 in_instance_color;
layout(location = 9) in float in_instance_rotation;
layout(location = 10) in float in_instance_scale;
layout(location = 11) in uint in_instance_flags;

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
    vec4 color_multiplier;
    uint flags;
    float custom_param1;
    float custom_param2;
    float custom_param3;
} push;

// Output to fragment shader
layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec2 frag_world_pos;
layout(location = 3) out float frag_font_size;
layout(location = 4) out uint frag_glyph_id;
layout(location = 5) out uint frag_instance_id;
layout(location = 6) out float frag_animation_factor;
layout(location = 7) out vec2 frag_local_pos;

// Animation and effect functions
float easeInOutCubic(float t) {
    return t < 0.5 ? 4.0 * t * t * t : 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0;
}

float easeOutElastic(float t) {
    const float c4 = (2.0 * 3.14159265359) / 3.0;
    return t == 0.0 ? 0.0 : t == 1.0 ? 1.0 : pow(2.0, -10.0 * t) * sin((t * 10.0 - 0.75) * c4) + 1.0;
}

vec2 rotate2D(vec2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Noise function for subtle animations
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Calculate instance transform
    vec2 instance_pos = in_instance_position;
    vec2 instance_size = in_instance_size;
    vec4 instance_color = in_instance_color;
    float instance_rotation = in_instance_rotation;
    float instance_scale = in_instance_scale;
    
    // Apply DPI scaling
    vec2 dpi_adjusted_size = instance_size * ubo.dpi_scale;
    
    // Calculate local position within the instance
    vec2 local_pos = in_position * dpi_adjusted_size;
    
    // Apply instance rotation
    if (instance_rotation != 0.0) {
        local_pos = rotate2D(local_pos, instance_rotation);
    }
    
    // Apply instance scaling
    local_pos *= instance_scale;
    
    // Calculate world position
    vec2 world_pos = instance_pos + local_pos + push.offset;
    
    // Apply push constant scaling
    world_pos = world_pos * push.scale;
    
    // Animation effects based on render mode and flags
    float animation_factor = 1.0;
    
    // Hover effect
    float dist_to_mouse = length(world_pos - ubo.mouse_position);
    float hover_effect = smoothstep(ubo.hover_radius, ubo.hover_radius * 0.5, dist_to_mouse);
    
    // Pulsing animation for highlighted elements
    if ((in_instance_flags & 0x1u) != 0u) {
        float pulse = sin(ubo.time * 4.0 + float(in_instance_id) * 0.1) * 0.5 + 0.5;
        animation_factor = 1.0 + pulse * 0.1;
        world_pos *= animation_factor;
    }
    
    // Smooth entry animation
    if ((in_instance_flags & 0x2u) != 0u) {
        float entry_time = mod(ubo.time + float(in_instance_id) * 0.05, 2.0);
        float entry_factor = easeOutElastic(clamp(entry_time, 0.0, 1.0));
        world_pos = mix(instance_pos + vec2(0.0, -50.0), world_pos, entry_factor);
        animation_factor = entry_factor;
    }
    
    // Data-driven animation (for real-time updates)
    if ((in_instance_flags & 0x4u) != 0u) {
        float data_animation = push.custom_param1; // Data change intensity
        float flash_effect = exp(-data_animation * 5.0) * sin(data_animation * 20.0);
        animation_factor += flash_effect * 0.2;
    }
    
    // Apply hover scaling
    world_pos += (world_pos - ubo.mouse_position) * hover_effect * 0.05;
    
    // Transform to clip space
    vec4 clip_pos = ubo.projection * ubo.view * vec4(world_pos, 0.0, 1.0);
    
    // Output vertex position
    gl_Position = clip_pos;
    
    // Pass data to fragment shader
    frag_texcoord = in_texcoord;
    frag_color = in_color * instance_color * push.color_multiplier * ubo.global_tint;
    frag_world_pos = world_pos;
    frag_font_size = in_font_size * ubo.dpi_scale.x;
    frag_glyph_id = in_glyph_id;
    frag_instance_id = in_instance_id;
    frag_animation_factor = animation_factor;
    frag_local_pos = local_pos;
    
    // Apply hover brightness
    frag_color.rgb += vec3(hover_effect * 0.1);
    
    // Apply animation brightness
    if (animation_factor > 1.0) {
        frag_color.rgb += vec3((animation_factor - 1.0) * 0.5);
    }
    
    // Subtle noise for organic feel
    float subtle_noise = noise(world_pos * 0.01 + ubo.time * 0.1) * 0.02;
    frag_color.rgb += vec3(subtle_noise);
    
    // Ensure color stays in valid range
    frag_color = clamp(frag_color, 0.0, 1.0);
}