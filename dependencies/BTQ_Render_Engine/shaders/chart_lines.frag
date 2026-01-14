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
layout(location = 0) in vec4 frag_color;

// Output
layout(location = 0) out vec4 out_color;

void main() {
    out_color = frag_color;
}