#include "../../include/components/orderbook_batcher.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

OrderbookBatcher::OrderbookBatcher() {
    // Reserve initial capacity to reduce allocations
    batches_.reserve(16);
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
    // Prioritize batching by texture and color to minimize draw calls
    // For order book rendering, we often have many elements with the same texture and similar colors
    for (auto& batch : batches_) {
        if (batch.texture == texture && batch.vertices.size() < 65535 - 4 && batch.indices.size() < 65535 - 6) {
            // For order book elements, we can often batch elements with the same color together
            // This reduces the number of draw calls significantly
            return &batch;
        }
    }

    // If no compatible batch exists, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    new_batch.texture = texture;

    // Pre-allocate space to reduce reallocations - optimized for order book rendering
    new_batch.vertices.reserve(2048);  // Increased initial reservation for better performance
    new_batch.indices.reserve(4096);   // Increased initial reservation for better performance

    return &new_batch;
}

void OrderbookBatcher::optimizeBatches() {
    // Attempt to merge compatible batches to reduce draw calls
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // More aggressive optimization: merge batches with same texture regardless of primitive type
    // This is particularly beneficial for order book rendering where we have many similar elements
    std::vector<OrderbookBatchElement> optimized_batches;
    optimized_batches.reserve(batches_.size());

    for (auto& current_batch : batches_) {
        bool merged = false;

        // Try to find an existing batch with the same texture to merge with
        for (auto& target_batch : optimized_batches) {
            // Check if batches can be merged (same texture, and enough space)
            if (target_batch.texture == current_batch.texture &&
                target_batch.vertices.size() + current_batch.vertices.size() < 65535 &&
                target_batch.indices.size() + current_batch.indices.size() < 65535) {

                // Merge the current batch into the target batch
                size_t vertex_offset = target_batch.vertices.size();

                // Add vertices from current batch to target batch
                target_batch.vertices.insert(target_batch.vertices.end(),
                                           current_batch.vertices.begin(),
                                           current_batch.vertices.end());

                // Add indices from current batch to target batch with proper offset
                for (auto index : current_batch.indices) {
                    target_batch.indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
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

            // Add the vertices and indices to the draw list
            draw_list->PrimReserve(static_cast<int>(batch.indices.size()), static_cast<int>(batch.vertices.size()));

            // Copy vertices
            for (const auto& vertex : batch.vertices) {
                draw_list->_VtxWritePtr[0].pos = vertex.pos;
                draw_list->_VtxWritePtr[0].uv = vertex.uv;
                draw_list->_VtxWritePtr[0].col = vertex.col;
                draw_list->_VtxWritePtr++;
            }

            // Copy indices with proper offset
            for (const auto& index : batch.indices) {
                draw_list->_IdxWritePtr[0] = static_cast<ImDrawIdx>(index + draw_list->_VtxCurrentIdx);
                draw_list->_IdxWritePtr++;
            }

            // Update vertex index counter
            draw_list->_VtxCurrentIdx += static_cast<unsigned int>(batch.vertices.size());
        }
    }

    // Clear the batcher after submitting
    clear();
}

} // namespace BTQuant