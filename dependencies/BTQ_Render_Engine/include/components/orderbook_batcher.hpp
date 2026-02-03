#pragma once

#include <imgui.h>
#include <vector>
#include <array>
#include <functional>

namespace BTQuant {

// Structure to hold batched geometry data for order book rendering
struct OrderbookBatchVertex {
    ImVec2 pos;
    ImVec2 uv;  // Texture coordinates if needed
    ImU32 col;

    // Constructor for easy initialization
    OrderbookBatchVertex() : pos(0, 0), uv(0, 0), col(0) {}
    OrderbookBatchVertex(const ImVec2& position, const ImVec2& texture_coords, ImU32 color)
        : pos(position), uv(texture_coords), col(color) {}
};

struct OrderbookBatchElement {
    std::vector<OrderbookBatchVertex> vertices;
    std::vector<ImDrawIdx> indices;
    ImTextureID texture;  // For textured elements
    int primitive_type;   // ImGui's primitive type (ImDrawList flags)

    OrderbookBatchElement() : texture((ImTextureID)0), primitive_type(0) {}
};

// Enum for different types of orderbook elements
enum class OrderbookElementType {
    RECT_FILLED,
    LINE,
    CIRCLE_FILLED,
    TEXT
};

// Structure for line data
struct LineData {
    ImVec2 p1, p2;
};

// Structure for rectangle data
struct RectData {
    ImVec2 min, max;
};

// Structure for orderbook element data
struct OrderbookElementData {
    OrderbookElementType type;
    ImU32 color;
    ImTextureID texture;
    float thickness;  // For lines
    union {
        LineData line;
        RectData rect;
    };

    // Constructor for rectangle
    OrderbookElementData(ImVec2 min, ImVec2 max, ImU32 col, ImTextureID tex = (ImTextureID)0)
        : type(OrderbookElementType::RECT_FILLED), color(col), texture(tex), thickness(1.0f) {
        rect.min = min;
        rect.max = max;
    }

    // Constructor for line
    OrderbookElementData(ImVec2 p1, ImVec2 p2, ImU32 col, float thick, ImTextureID tex = (ImTextureID)0)
        : type(OrderbookElementType::LINE), color(col), texture(tex), thickness(thick) {
        line.p1 = p1;
        line.p2 = p2;
    }
};

// Batched geometry for order book rendering
class OrderbookBatcher {
public:
    OrderbookBatcher();
    ~OrderbookBatcher();

    // Clear all batched geometry
    void clear();

    // Add a rectangle to the batch
    void addRectFilled(const ImVec2& min, const ImVec2& max, ImU32 col);

    // Add a circle to the batch
    void addCircleFilled(const ImVec2& center, float radius, ImU32 col);

    // Add a line to the batch
    void addLine(const ImVec2& p1, const ImVec2& p2, ImU32 col, float thickness = 1.0f);

    // Add text to the batch (simplified)
    void addText(const ImVec2& pos, ImU32 col, const char* text);

    // Add multiple rectangles to the batch (for better batching of similar elements)
    void addRectanglesFilled(const std::vector<std::pair<ImVec2, ImVec2>>& rect_pairs,
                           const std::vector<ImU32>& colors);

    // Add multiple rectangles to the batch with optimized grouping for maximum batching
    void addRectanglesFilledOptimized(const std::vector<std::pair<ImVec2, ImVec2>>& rect_pairs,
                                    const std::vector<ImU32>& colors);

    // Enhanced method to batch multiple similar elements together for maximum efficiency
    void batchSimilarElements(const std::vector<std::function<void(OrderbookBatchElement*)>>& element_adders);

    // Advanced method for order book specific batching - combines multiple elements with intelligent grouping
    void addOrderbookElements(const std::vector<OrderbookElementData>& elements);

    // Ultra-efficient method for order book specific batching - maximizes batching by using ultra-compatible grouping
    void addOrderbookElementsUltra(const std::vector<OrderbookElementData>& elements);

    // Super-efficient method for order book specific batching - maximizes batching by using super-compatible grouping
    void addOrderbookElementsSuper(const std::vector<OrderbookElementData>& elements);

    // Ultra-performance method for order book specific batching - uses ultra-fast grouping and batching
    void addOrderbookElementsUltraPerformance(const std::vector<OrderbookElementData>& elements);

    // Submit all batched geometry to the draw list
    void submit(ImDrawList* draw_list);

    // Optimized submit method that reduces GPU overhead by minimizing draw calls and memory operations
    void submitOptimized(ImDrawList* draw_list);

    // Get the number of batched elements
    size_t getBatchCount() const { return batches_.size(); }

    // Check if there are any batched elements
    bool isEmpty() const { return batches_.empty(); }

    // Advanced batch optimization that minimizes draw calls by maximizing batch sizes
    void advancedBatchOptimization();

    // GPU-optimized batching that focuses on reducing draw calls and memory allocations
    void gpuOptimizedBatching();

    // Memory-efficient batching that reduces memory fragmentation and allocation overhead
    void memoryEfficientBatching();

    // Batch multiple similar draw calls together to reduce GPU overhead
    void batchMultipleDrawCalls(const std::vector<std::function<void(OrderbookBatchElement*)>>& draw_calls);

    // Ultra-efficient method to batch multiple similar draw calls together with maximum performance
    void batchMultipleDrawCallsUltra(const std::vector<std::function<void(OrderbookBatchElement*)>>& draw_calls);

    // Enhanced batch geometry method that combines multiple similar elements into single draw calls
    void batchGeometryEnhanced(const std::vector<OrderbookElementData>& elements);

    // Highly optimized batch method that combines similar elements with minimal overhead
    void batchGeometryHighPerformance(const std::vector<OrderbookElementData>& elements);

    // Ultra-high performance batch method that combines similar elements with maximum efficiency and minimal overhead
    void batchGeometryUltraHighPerformance(const std::vector<OrderbookElementData>& elements);

    // Maximum performance batch method that combines similar elements with ultimate efficiency and minimal GPU overhead
    void batchGeometryMaximumPerformance(const std::vector<OrderbookElementData>& elements);

    // Ultra-fast batch consolidation that uses a more efficient algorithm to reduce draw calls and GPU overhead
    void ultraFastConsolidateBatches();

    // Smart batch optimization that uses intelligent grouping to minimize GPU overhead
    void smartOptimizeBatches();

    // Advanced batch optimization that uses hierarchical grouping to minimize GPU overhead
    void hierarchicalOptimizeBatches();

private:
    std::vector<OrderbookBatchElement> batches_;

    // Find or create a compatible batch for the given parameters
    OrderbookBatchElement* findOrCreateCompatibleBatch(ImTextureID texture, ImU32 col);

    // Find or create the best compatible batch considering multiple factors
    OrderbookBatchElement* findOrCreateBestCompatibleBatch(ImTextureID texture, ImU32 col);

    // Ultra-performance version that prioritizes maximum batching with minimum computational overhead
    OrderbookBatchElement* findOrCreateBestCompatibleBatchUltraPerformance(ImTextureID texture, ImU32 col);

    // Ultra-compatible version that prioritizes maximum batching with ultra-relaxed color matching
    OrderbookBatchElement* findOrCreateBestCompatibleBatchUltra(ImTextureID texture, ImU32 col);

    // Super-compatible version that prioritizes maximum batching with super-relaxed color matching
    OrderbookBatchElement* findOrCreateBestCompatibleBatchSuper(ImTextureID texture, ImU32 col);

    // Initialize a batch with optimal memory allocation
    void initializeBatch(OrderbookBatchElement& batch, ImTextureID texture);

    // Initialize a batch with expected size for optimal memory allocation
    void initializeBatchWithExpectedSize(OrderbookBatchElement& batch, ImTextureID texture,
                                       size_t expected_vertices, size_t expected_indices);

    // Resize batch capacity intelligently based on usage
    void resizeBatchIfNeeded(OrderbookBatchElement& batch);

    // Optimize batches by merging compatible ones to reduce draw calls
    void optimizeBatches();

    // Advanced optimization that groups batches by texture first, then by color similarity
    void advancedOptimizeBatches();

    // Ultra-optimized optimization that aggressively combines batches to minimize draw calls
    void ultraOptimizeBatches();

    // Super-optimized optimization that super-aggressively combines batches to minimize draw calls even further
    void superOptimizeBatches();

    // Ultra-performance optimization that uses the fastest possible merging to minimize GPU overhead
    void ultraPerformanceOptimizeBatches();

    // Efficient method to combine similar batches and reduce draw calls
    void combineSimilarBatches();

    // Enhanced method to batch geometry with maximum efficiency for order book rendering
    void batchGeometry(const std::vector<OrderbookElementData>& elements);

    // Fast batch consolidation to reduce GPU overhead
    void fastConsolidateBatches();

    // Maximum optimization that uses the ultimate approach to minimize GPU overhead
    void maximumOptimizeBatches();

    // Ultimate optimization that uses the most comprehensive approach to minimize GPU overhead
    void ultimateOptimizeBatches();
};

} // namespace BTQuant