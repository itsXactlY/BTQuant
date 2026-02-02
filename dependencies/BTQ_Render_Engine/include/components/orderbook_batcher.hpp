#pragma once

#include <imgui.h>
#include <vector>
#include <array>

namespace BTQuant {

// Structure to hold batched geometry data for order book rendering
struct OrderbookBatchVertex {
    ImVec2 pos;
    ImU32 col;
    ImVec2 uv;  // Texture coordinates if needed
};

struct OrderbookBatchElement {
    std::vector<OrderbookBatchVertex> vertices;
    std::vector<ImDrawIdx> indices;
    ImTextureID texture;  // For textured elements
    int primitive_type;   // ImGui's primitive type (ImDrawList flags)

    OrderbookBatchElement() : texture((ImTextureID)0), primitive_type(0) {}
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

    // Submit all batched geometry to the draw list
    void submit(ImDrawList* draw_list);

    // Get the number of batched elements
    size_t getBatchCount() const { return batches_.size(); }

    // Check if there are any batched elements
    bool isEmpty() const { return batches_.empty(); }

private:
    std::vector<OrderbookBatchElement> batches_;

    // Find or create a compatible batch for the given parameters
    OrderbookBatchElement* findOrCreateCompatibleBatch(ImTextureID texture, int primitive_type);

    // Optimize batches by merging compatible ones to reduce draw calls
    void optimizeBatches();
};

} // namespace BTQuant