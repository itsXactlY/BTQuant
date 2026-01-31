// Volumetric Footprint (Candle Cluster) Fragment Shader
// Vulkan 1.3 - Market Microstructure Renderer
// Handles cell coloring and SDF text rendering

#version 460 core

// ========================================
// Inputs from Vertex Shader
// ========================================

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec2 fragCenter;
layout(location = 3) in float fragSize;

// ========================================
// Uniforms
// ========================================

layout(set = 2, binding = 0) uniform sampler2D sdfFontAtlas;

layout(std140, set = 2, binding = 1) uniform TextParams {
    vec2 atlasSize;          // Font atlas texture dimensions
    float fontSize;          // Font size in pixels
    float sdfSpread;         // SDF spread parameter (0-1)
    vec4 textColor;          // Text color
    bool showLabels;         // Enable/disable text rendering
} text;

// ========================================
// Output
// ========================================

layout(location = 0) out vec4 outColor;

// ========================================
// SDF Text Rendering Helper
// ========================================

float sdfSample(vec2 uv) {
    vec2 texCoords = uv / text.atlasSize;
    return texture(sdfFontAtlas, texCoords).r;
}

vec4 renderText(vec2 uv, vec3 color) {
    float distance = sdfSample(uv);
    float alpha = smoothstep(0.5 - text.sdfSpread, 0.5 + text.sdfSpread, distance);
    return vec4(color, alpha);
}

// ========================================
// Main Fragment Shader Entry
// ========================================

void main() {
    // Base cell color
    outColor = fragColor;
    
    // Add text rendering if enabled and cell is large enough
    if (text.showLabels && fragSize > 10.0) {
        vec4 textSample = renderText(fragTexCoord * text.atlasSize, text.textColor.rgb);
        outColor = mix(outColor, textSample, textSample.a);
    }
    
    // Add subtle border for cell separation
    float border = 0.02;
    if (fragTexCoord.x < border || fragTexCoord.x > 1.0 - border ||
        fragTexCoord.y < border || fragTexCoord.y > 1.0 - border) {
        outColor.rgb *= 0.8;
    }
}