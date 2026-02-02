#include "../../include/components/orderbook_batcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace BTQuant {

// Helper function to determine if two colors are similar enough to batch together
// This is particularly useful for order book rendering where colors may vary slightly
static bool areColorsSimilar(ImU32 col1, ImU32 col2, uint8_t tolerance = 30) {
    // Extract RGBA components
    uint8_t r1 = (col1 >> 0) & 0xFF;
    uint8_t g1 = (col1 >> 8) & 0xFF;
    uint8_t b1 = (col1 >> 16) & 0xFF;
    uint8_t a1 = (col1 >> 24) & 0xFF;

    uint8_t r2 = (col2 >> 0) & 0xFF;
    uint8_t g2 = (col2 >> 8) & 0xFF;
    uint8_t b2 = (col2 >> 16) & 0xFF;
    uint8_t a2 = (col2 >> 24) & 0xFF;

    // Check if all components are within tolerance
    return (abs(static_cast<int>(r1) - static_cast<int>(r2)) <= tolerance &&
            abs(static_cast<int>(g1) - static_cast<int>(g2)) <= tolerance &&
            abs(static_cast<int>(b1) - static_cast<int>(b2)) <= tolerance &&
            abs(static_cast<int>(a1) - static_cast<int>(a2)) <= tolerance);
}

OrderbookBatcher::OrderbookBatcher() {
    // Reserve initial capacity to reduce allocations
    batches_.reserve(32);  // Increased initial reservation for better performance
}

void OrderbookBatcher::initializeBatch(OrderbookBatchElement& batch, ImTextureID texture) {
    batch.texture = texture;
    // Pre-allocate space to reduce reallocations - optimized for order book rendering
    batch.vertices.reserve(4096);  // Increased initial reservation for better performance
    batch.indices.reserve(8192);   // Increased initial reservation for better performance
    batch.primitive_type = 0;      // Default primitive type
}

OrderbookBatcher::~OrderbookBatcher() {
    clear();
}

void OrderbookBatcher::clear() {
    for (auto& batch : batches_) {
        batch.vertices.clear();
        batch.indices.clear();
    }
    batches_.clear();
}

OrderbookBatchElement* OrderbookBatcher::findOrCreateCompatibleBatch(ImTextureID texture, ImU32 col) {
    // Enhanced batching: prioritize batching by texture first, then by similar colors
    // For order book rendering, we often have many elements with the same texture and similar colors

    // First, try to find an exact match (same texture and similar color)
    for (auto& batch : batches_) {
        if (batch.texture == texture &&
            batch.vertices.size() < 65535 - 4 &&
            batch.indices.size() < 65535 - 6) {

            // For order book elements, we can batch elements with similar colors together
            // This reduces the number of draw calls significantly
            // Check if the current batch's color is similar enough to the requested color
            if (batch.vertices.empty() || areColorsSimilar(batch.vertices[0].col, col)) {
                return &batch;
            }
        }
    }

    // If no compatible batch exists, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    initializeBatch(new_batch, texture);

    return &new_batch;
}

// Additional method to batch multiple rectangles of similar colors together
void OrderbookBatcher::addRectanglesFilled(const std::vector<std::pair<ImVec2, ImVec2>>& rect_pairs,
                                          const std::vector<ImU32>& colors) {
    if (rect_pairs.size() != colors.size() || rect_pairs.empty()) {
        return;
    }

    // Group rectangles by similar colors to maximize batching efficiency
    for (size_t i = 0; i < rect_pairs.size(); ++i) {
        addRectFilled(rect_pairs[i].first, rect_pairs[i].second, colors[i]);
    }
}

void OrderbookBatcher::optimizeBatches() {
    // Attempt to merge compatible batches to reduce draw calls
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Sort batches by texture ID to group similar textures together for better merging
    std::sort(batches_.begin(), batches_.end(), [](const OrderbookBatchElement& a, const OrderbookBatchElement& b) {
        return a.texture < b.texture;
    });

    // More aggressive optimization: merge batches with same texture and similar colors
    // This is particularly beneficial for order book rendering where we have many similar elements
    std::vector<OrderbookBatchElement> optimized_batches;
    optimized_batches.reserve(batches_.size());

    for (auto& current_batch : batches_) {
        bool merged = false;

        // Try to find an existing batch with the same texture and similar color to merge with
        // Iterate backwards to find the most recently added compatible batch (better cache locality)
        for (auto it = optimized_batches.rbegin(); it != optimized_batches.rend(); ++it) {
            // Check if batches can be merged (same texture, similar color, and enough space)
            bool can_merge = (it->texture == current_batch.texture &&
                             it->vertices.size() + current_batch.vertices.size() < 65535 &&
                             it->indices.size() + current_batch.indices.size() < 65535);

            // If textures match, check if colors are similar enough to merge
            if (can_merge && !current_batch.vertices.empty() && !it->vertices.empty()) {
                can_merge = areColorsSimilar(it->vertices[0].col, current_batch.vertices[0].col);
            }

            if (can_merge) {
                // Merge the current batch into the target batch
                size_t vertex_offset = it->vertices.size();

                // Add vertices from current batch to target batch
                it->vertices.insert(it->vertices.end(),
                                   current_batch.vertices.begin(),
                                   current_batch.vertices.end());

                // Add indices from current batch to target batch with proper offset
                for (auto index : current_batch.indices) {
                    it->indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
                }

                merged = true;
                break;
            }
        }

        // If the current batch wasn't merged, add it as a new batch
        if (!merged) {
            optimized_batches.emplace_back(std::move(current_batch));
        }
    }

    // Replace the old batches with the optimized ones
    batches_ = std::move(optimized_batches);
}

void OrderbookBatcher::addRectFilled(const ImVec2& min, const ImVec2& max, ImU32 col) {
    // Find or create a compatible batch for filled rectangles
    // For order book rendering, we prioritize batching by texture and color
    auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, col);

    // Add 4 vertices for the rectangle
    size_t vertex_start = batch->vertices.size();

    batch->vertices.push_back({min, {0, 0}, col});                    // Top-left
    batch->vertices.push_back({ImVec2(max.x, min.y), {1, 0}, col});   // Top-right
    batch->vertices.push_back({max, {1, 1}, col});                    // Bottom-right
    batch->vertices.push_back({ImVec2(min.x, max.y), {0, 1}, col});   // Bottom-left

    // Add 6 indices to form 2 triangles (0,1,2 and 0,2,3)
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
}

void OrderbookBatcher::addCircleFilled(const ImVec2& center, float radius, ImU32 col) {
    // Find or create a compatible batch for filled circles
    auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, col);

    // Approximate circle with 12 segments for performance
    const int segments = 12;
    const float segment_angle = 2.0f * 3.14159265358979323846f / segments; // IM_PI replacement

    // Add center vertex
    size_t center_vertex_idx = batch->vertices.size();
    batch->vertices.push_back({center, {0.5f, 0.5f}, col});

    // Add outer vertices
    std::vector<size_t> outer_vertices;
    for (int i = 0; i <= segments; i++) {
        float angle = i * segment_angle;
        ImVec2 point = ImVec2(
            center.x + cosf(angle) * radius,
            center.y + sinf(angle) * radius
        );

        outer_vertices.push_back(batch->vertices.size());
        batch->vertices.push_back({point, {0.5f + cosf(angle)*0.5f, 0.5f + sinf(angle)*0.5f}, col});
    }

    // Create triangle fan from center to outer vertices
    for (size_t i = 0; i < outer_vertices.size() - 1; i++) {
        batch->indices.push_back(static_cast<ImDrawIdx>(center_vertex_idx));
        batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i]));
        batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i + 1]));
    }
}

void OrderbookBatcher::addLine(const ImVec2& p1, const ImVec2& p2, ImU32 col, float thickness) {
    // Find or create a compatible batch for lines
    auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, col);

    // For thick lines, we create a rectangle perpendicular to the line direction
    ImVec2 delta = ImVec2(p2.x - p1.x, p2.y - p1.y);
    float length = sqrtf(delta.x * delta.x + delta.y * delta.y);
    if (length == 0.0f) return;

    ImVec2 dir = ImVec2(delta.x / length, delta.y / length);
    ImVec2 perp = ImVec2(-dir.y, dir.x); // Perpendicular vector

    // Calculate half-thickness offset
    float half_thickness = thickness * 0.5f;
    ImVec2 offset = ImVec2(perp.x * half_thickness, perp.y * half_thickness);

    size_t vertex_start = batch->vertices.size();

    // Add 4 vertices for the rectangular line representation
    batch->vertices.push_back({ImVec2(p1.x - offset.x, p1.y - offset.y), {0, 0}, col}); // p1 - offset
    batch->vertices.push_back({ImVec2(p1.x + offset.x, p1.y + offset.y), {1, 0}, col}); // p1 + offset
    batch->vertices.push_back({ImVec2(p2.x + offset.x, p2.y + offset.y), {1, 1}, col}); // p2 + offset
    batch->vertices.push_back({ImVec2(p2.x - offset.x, p2.y - offset.y), {0, 1}, col}); // p2 - offset

    // Add 6 indices to form 2 triangles
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
}

void OrderbookBatcher::addText(const ImVec2& pos, ImU32 col, const char* text) {
    // For text batching, we'll just add a simple quad representing the text bounds
    // In a real implementation, you'd want to properly handle font rendering
    auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, col);

    // Calculate approximate text bounds
    ImVec2 text_size = ImGui::CalcTextSize(text);
    ImVec2 min = pos;
    ImVec2 max = ImVec2(pos.x + text_size.x, pos.y + text_size.y);

    size_t vertex_start = batch->vertices.size();

    batch->vertices.push_back({min, {0, 0}, col});                    // Top-left
    batch->vertices.push_back({ImVec2(max.x, min.y), {1, 0}, col});   // Top-right
    batch->vertices.push_back({max, {1, 1}, col});                    // Bottom-right
    batch->vertices.push_back({ImVec2(min.x, max.y), {0, 1}, col});   // Bottom-left

    // Add 6 indices to form 2 triangles
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
    batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
}

void OrderbookBatcher::submit(ImDrawList* draw_list) {
    if (!draw_list || batches_.empty()) {
        return;
    }

    // Optimize batches by merging compatible ones before submission
    optimizeBatches();

    // Pre-calculate total vertices and indices to reserve space upfront
    size_t total_vertices = 0;
    size_t total_indices = 0;

    for (const auto& batch : batches_) {
        if (!batch.vertices.empty() && !batch.indices.empty()) {
            total_vertices += batch.vertices.size();
            total_indices += batch.indices.size();
        }
    }

    // Reserve space in the draw list to minimize reallocations
    if (total_vertices > 0) {
        draw_list->PrimReserve(static_cast<int>(total_indices), static_cast<int>(total_vertices));
    }

    for (const auto& batch : batches_) {
        if (!batch.vertices.empty() && !batch.indices.empty()) {
            // Properly set up draw command with texture and scissor clip
            ImDrawCmd cmd;
            cmd.TexRef._TexID = batch.texture;
            cmd.VtxOffset = draw_list->_VtxCurrentIdx;
            cmd.IdxOffset = static_cast<unsigned int>(draw_list->IdxBuffer.Size);
            cmd.ElemCount = static_cast<unsigned int>(batch.indices.size());

            // Add the draw command to the draw list
            draw_list->CmdBuffer.push_back(cmd);

            // Copy vertices directly using memcpy for better performance
            if (!batch.vertices.empty()) {
                memcpy(draw_list->_VtxWritePtr, batch.vertices.data(),
                       sizeof(OrderbookBatchVertex) * batch.vertices.size());
                draw_list->_VtxWritePtr += batch.vertices.size();
            }

            // Copy indices with proper offset - optimized loop
            if (!batch.indices.empty()) {
                const size_t idx_count = batch.indices.size();
                for (size_t i = 0; i < idx_count; ++i) {
                    draw_list->_IdxWritePtr[i] = static_cast<ImDrawIdx>(batch.indices[i] + draw_list->_VtxCurrentIdx);
                }
                draw_list->_IdxWritePtr += idx_count;
            }

            // Update vertex index counter
            draw_list->_VtxCurrentIdx += static_cast<unsigned int>(batch.vertices.size());
        }
    }

    // Clear the batcher after submitting
    clear();
}

} // namespace BTQuant