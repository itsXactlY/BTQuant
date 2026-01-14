#version 450 core

layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in float frag_custom_val;
layout(location = 3) in flat uint frag_type;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 final_color = frag_color;
    
    // Add a very subtle horizontal gradient to bars
    if (frag_type == 0 || frag_type == 1) { // Size bar or Candlestick Body
        final_color.rgb *= (0.9 + 0.2 * frag_texcoord.x);
    }
    
    // Force alpha to 1.0
    out_color = vec4(final_color.rgb, 1.0);
}