#version 450 core

layout(location = 0) in vec2 frag_texcoord;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in float frag_custom_val;
layout(location = 3) in flat uint frag_type;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 final_color = frag_color;
    
    // Add professional styling based on type
    if (frag_type == 0) { // Candlestick Body
        // Vertical gradient for bodies
        final_color.rgb *= (0.85 + 0.3 * frag_texcoord.y);
    } else if (frag_type == 1) { // Wick
        // No modification for wicks
    } else if (frag_type == 2) { // Volume Bar
        // Subtle glow effect
        float glow = exp(-2.0 * abs(frag_texcoord.x - 0.5));
        final_color.rgb *= (0.7 + 0.3 * glow);
    } else if (frag_type == 3) { // Size Bar (Orderbook)
        // Horizontal gradient from center
        final_color.rgb *= (0.9 + 0.2 * frag_texcoord.x);
    } else if (frag_type == 4) { // Crosshair
        // Dashing effect based on texture coordinates or just simple alpha
        final_color.a *= 0.7;
    }
    
    out_color = vec4(final_color.rgb, final_color.a);
}