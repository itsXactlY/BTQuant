#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV; // Used to identify body vs wick

// Instance data
layout(location = 2) in vec4 inCandleData; // open, high, low, close
layout(location = 3) in float inTimestampOffset;

layout(binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    vec2 viewport_size;
    vec2 chart_bounds_min; // [t_min, price_min]
    vec2 chart_bounds_max; // [t_max, price_max]
} ubo;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outIsBullish;

void main() {
    float open = inCandleData.x;
    float high = inCandleData.y;
    float low = inCandleData.z;
    float close = inCandleData.w;
    
    bool isBullish = close >= open;
    outIsBullish = isBullish ? 1.0 : 0.0;
    
    // Calculate color based on type
    vec4 bullish_color = vec4(0.0, 1.0, 0.6, 1.0); // Teal Street Green
    vec4 bearish_color = vec4(1.0, 0.2, 0.3, 1.0); // Teal Street Red
    outColor = isBullish ? bullish_color : bearish_color;

    // Mapping logic
    float x_range = ubo.chart_bounds_max.x - ubo.chart_bounds_min.x;
    float y_range = ubo.chart_bounds_max.y - ubo.chart_bounds_min.y;
    
    float x = (inTimestampOffset / x_range) * ubo.viewport_size.x;
    
    float candle_width = 5.0; // Base width in pixels, should probably be a uniform
    float half_width = candle_width * 0.5;
    
    float final_x = x;
    float final_y = 0.0;
    
    if (inUV.x > 0.5) { // Wick
        final_x = x + (inPos.x - 0.5) * (half_width * 0.2);
        final_y = ( (low + inPos.y * (high - low)) - ubo.chart_bounds_min.y) / y_range * ubo.viewport_size.y;
    } else { // Body
        float body_min = min(open, close);
        float body_max = max(open, close);
        final_x = x + (inPos.x - 0.5) * half_width;
        final_y = ( (body_min + inPos.y * (body_max - body_min)) - ubo.chart_bounds_min.y) / y_range * ubo.viewport_size.y;
        
        // Add minimal height for flat candles
        if (abs(body_max - body_min) < 0.0001) {
             // ...
        }
    }
    
    gl_Position = ubo.projection * vec4(final_x, final_y, 0.0, 1.0);
}
