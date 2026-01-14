#version 450 core

layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in flat uint frag_glyph_id;

layout(set = 0, binding = 2) uniform sampler2D font_atlas;

layout(location = 0) out vec4 out_color;

void main() {
    // Sample texture and multiply by vertex color
    vec4 tex_color = texture(font_atlas, frag_texcoord);
    
    // For debugging: Use tex_color if it's not zero, otherwise just frag_color
    // But force alpha to 1.0
    vec3 final_rgb = frag_color.rgb * tex_color.rgb;
    if (tex_color.a < 0.01) {
        final_rgb = frag_color.rgb; // Fallback to solid color if texture is empty
    }
    
    out_color = vec4(final_rgb, 1.0);
}