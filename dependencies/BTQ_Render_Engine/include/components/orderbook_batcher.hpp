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

    // Submit all batched geometry to the draw list
    void submit(ImDrawList* draw_list);

    // Get the number of batched elements
    size_t getBatchCount() const { return batches_.size(); }

    // Check if there are any batched elements
    bool isEmpty() const { return batches_.empty(); }

private:
    std::vector<OrderbookBatchElement> batches_;

    // Find or create a compatible batch for the given parameters
    OrderbookBatchElement* findOrCreateCompatibleBatch(ImTextureID texture, ImU32 col);

    // Find or create the best compatible batch considering multiple factors
    OrderbookBatchElement* findOrCreateBestCompatibleBatch(ImTextureID texture, ImU32 col);

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
};

} // namespace BTQuant