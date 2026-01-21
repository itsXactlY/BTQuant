// Volumetric Footprint (Candle Cluster) Vertex Shader
// Vulkan 1.3 - Market Microstructure Renderer
// Handles instanced rendering of footprint cells

#version 460 core

// ========================================
// Structures (std430 aligned)
// ========================================

struct CandleCluster {
    float centerX;           // Center X coordinate (time)
    float centerY;           // Center Y coordinate (price)
    float width;             // Cluster width (time duration)
    float height;            // Cluster height (price range)
    uint32_t bidVolume;      // Total bid volume
    uint32_t askVolume;      // Total ask volume
    uint32_t tradeCount;     // Number of trades
    float vwap;              // Volume-weighted average price
    bool hasTrades;          // Trade activity indicator
};

// ========================================
// Inputs
// ========================================

layout(location = 0) in vec2 position; // Quad corner position

layout(std430, set = 0, binding = 0) readonly buffer CandleClusters {
    CandleCluster clusters[];
} clusterData;

// ========================================
// Uniforms
// ========================================

layout(std140, set = 1, binding = 0) uniform ViewParams {
    mat4 projectionMatrix;
    mat4 viewMatrix;
    vec2 viewportSize;
    float timeRange;
    float priceRange;
} view;

// ========================================
// Outputs to Fragment Shader
// ========================================

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec2 fragCenter;
layout(location = 3) out float fragSize;

// ========================================
// Main Vertex Shader Entry
// ========================================

void main() {
    uint instanceID = gl_InstanceID;
    CandleCluster cluster = clusterData.clusters[instanceID];
    
    // Calculate quad corner position
    vec2 cornerPosition = vec2(
        cluster.centerX + position.x * cluster.width,
        cluster.centerY + position.y * cluster.height
    );
    
    // Transform to clip space
    gl_Position = view.projectionMatrix * view.viewMatrix * vec4(cornerPosition, 0.0, 1.0);
    
    // Calculate volume delta for coloring
    int delta = int(cluster.askVolume) - int(cluster.bidVolume);
    float normalizedDelta = clamp(float(delta) / max(float(cluster.askVolume + cluster.bidVolume), 1.0), -1.0, 1.0);
    
    // Bid-Ask color gradient
    vec3 bidColor = vec3(0.0, 0.5, 0.0);    // Green for bids
    vec3 askColor = vec3(1.0, 0.0, 0.0);    // Red for asks
    vec3 neutralColor = vec3(0.2, 0.2, 0.2); // Neutral gray
    
    if (normalizedDelta < 0.0) {
        fragColor.rgb = mix(neutralColor, bidColor, -normalizedDelta);
    } else {
        fragColor.rgb = mix(neutralColor, askColor, normalizedDelta);
    }
    fragColor.a = 0.8;
    
    // Texture coordinates for SDF text rendering
    fragTexCoord = position * 0.5 + vec2(0.5);
    fragCenter = cornerPosition;
    fragSize = min(cluster.width, cluster.height);
}