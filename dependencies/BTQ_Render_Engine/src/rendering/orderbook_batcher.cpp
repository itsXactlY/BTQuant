#include "../../include/components/orderbook_batcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>

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

// Enhanced helper function to calculate color distance for more accurate batching
static float getColorDistance(ImU32 col1, ImU32 col2) {
    uint8_t r1 = (col1 >> 0) & 0xFF;
    uint8_t g1 = (col1 >> 8) & 0xFF;
    uint8_t b1 = (col1 >> 16) & 0xFF;
    uint8_t a1 = (col1 >> 24) & 0xFF;

    uint8_t r2 = (col2 >> 0) & 0xFF;
    uint8_t g2 = (col2 >> 8) & 0xFF;
    uint8_t b2 = (col2 >> 16) & 0xFF;
    uint8_t a2 = (col2 >> 24) & 0xFF;

    // Calculate Euclidean distance in RGBA space
    int dr = r1 - r2;
    int dg = g1 - g2;
    int db = b1 - b2;
    int da = a1 - a2;

    return sqrtf(static_cast<float>(dr*dr + dg*dg + db*db + da*da));
}

// Enhanced helper function to determine if two batches can be combined
static bool canCombineBatches(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                              uint8_t color_tolerance = 30) {
    // Check if textures match
    if (batch1.texture != batch2.texture) {
        return false;
    }

    // Check if combining would exceed vertex/index limits
    if (batch1.vertices.size() + batch2.vertices.size() >= 65535 ||
        batch1.indices.size() + batch2.indices.size() >= 65535) {
        return false;
    }

    // Check if colors are similar enough to batch together
    if (!batch1.vertices.empty() && !batch2.vertices.empty()) {
        return areColorsSimilar(batch1.vertices[0].col, batch2.vertices[0].col, color_tolerance);
    }

    return true;
}

// Advanced helper function to determine if batches can be combined with more sophisticated criteria
static bool canCombineBatchesAdvanced(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                                     uint8_t color_tolerance = 30, float max_color_distance = 75.0f) {
    // Check if textures match
    if (batch1.texture != batch2.texture) {
        return false;
    }

    // Check if combining would exceed vertex/index limits
    if (batch1.vertices.size() + batch2.vertices.size() >= 65535 ||
        batch1.indices.size() + batch2.indices.size() >= 65535) {
        return false;
    }

    // For empty batches, allow combination
    if (batch1.vertices.empty() || batch2.vertices.empty()) {
        return true;
    }

    // Use both color similarity and distance for more accurate batching decisions
    if (!areColorsSimilar(batch1.vertices[0].col, batch2.vertices[0].col, color_tolerance)) {
        // If colors aren't similar by tolerance, check if they're close in RGB space
        float distance = getColorDistance(batch1.vertices[0].col, batch2.vertices[0].col);
        if (distance > max_color_distance) {
            return false;
        }
    }

    return true;
}

// Ultra-efficient helper function to determine if batches can be combined with maximum batching
static bool canCombineBatchesUltra(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                                   uint8_t color_tolerance = 50, float max_color_distance = 100.0f) {
    // Check if textures match - this is the primary criterion for batching
    if (batch1.texture != batch2.texture) {
        return false;
    }

    // Check if combining would exceed vertex/index limits
    if (batch1.vertices.size() + batch2.vertices.size() >= 65535 ||
        batch1.indices.size() + batch2.indices.size() >= 65535) {
        return false;
    }

    // For order book rendering, we can be more lenient with color similarity to maximize batching
    // This significantly reduces the number of draw calls
    if (batch1.vertices.empty() || batch2.vertices.empty()) {
        return true;
    }

    // Use a more lenient color similarity check to enable more batching
    if (!areColorsSimilar(batch1.vertices[0].col, batch2.vertices[0].col, color_tolerance)) {
        float distance = getColorDistance(batch1.vertices[0].col, batch2.vertices[0].col);
        if (distance > max_color_distance) {
            return false;
        }
    }

    return true;
}

OrderbookBatcher::OrderbookBatcher() {
    // Reserve initial capacity to reduce allocations
    batches_.reserve(32);  // Increased initial reservation for better performance
}

void OrderbookBatcher::initializeBatch(OrderbookBatchElement& batch, ImTextureID texture) {
    batch.texture = texture;
    // Pre-allocate space to reduce reallocations - optimized for order book rendering
    // Using more conservative initial reservations to balance memory usage and performance
    batch.vertices.reserve(2048);  // Reduced initial reservation for better memory usage
    batch.indices.reserve(4096);   // Reduced initial reservation for better memory usage
    batch.primitive_type = 0;      // Default primitive type
}

// Optimized initialization with adaptive sizing based on expected usage
void OrderbookBatcher::initializeBatchWithExpectedSize(OrderbookBatchElement& batch, ImTextureID texture,
                                                      size_t expected_vertices, size_t expected_indices) {
    batch.texture = texture;

    // Set capacity based on expected usage to minimize reallocations
    batch.vertices.reserve(expected_vertices > 0 ? expected_vertices : 2048);
    batch.indices.reserve(expected_indices > 0 ? expected_indices : 4096);
    batch.primitive_type = 0;
}

// Method to intelligently resize batch capacity based on usage patterns
void OrderbookBatcher::resizeBatchIfNeeded(OrderbookBatchElement& batch) {
    // Only increase capacity if we're close to the limit
    if (batch.vertices.size() > batch.vertices.capacity() * 0.8) {
        batch.vertices.reserve(batch.vertices.capacity() * 2);
    }
    if (batch.indices.size() > batch.indices.capacity() * 0.8) {
        batch.indices.reserve(batch.indices.capacity() * 2);
    }
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
    // Search from the end to find the most recently used batch (better cache locality)
    for (auto it = batches_.rbegin(); it != batches_.rend(); ++it) {
        if (it->texture == texture &&
            it->vertices.size() < 65535 - 4 &&
            it->indices.size() < 65535 - 6) {

            // For order book elements, we can batch elements with similar colors together
            // This reduces the number of draw calls significantly
            // Check if the current batch's color is similar enough to the requested color
            if (it->vertices.empty() || areColorsSimilar(it->vertices[0].col, col)) {
                return &(*it);
            }
        }
    }

    // If no compatible batch exists, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    initializeBatch(new_batch, texture);

    return &new_batch;
}

// Enhanced version that finds the best compatible batch considering multiple factors
OrderbookBatchElement* OrderbookBatcher::findOrCreateBestCompatibleBatch(ImTextureID texture, ImU32 col) {
    OrderbookBatchElement* best_batch = nullptr;
    float best_score = -1.0f; // Higher score means better compatibility

    // Look for the best compatible batch among existing ones
    for (auto& batch : batches_) {
        if (batch.texture != texture) {
            continue;
        }

        if (batch.vertices.size() >= 65535 - 4 || batch.indices.size() >= 65535 - 6) {
            continue; // Would exceed limits
        }

        // Calculate compatibility score based on color similarity and batch utilization
        float color_similarity = 0.0f;
        if (!batch.vertices.empty()) {
            float distance = getColorDistance(batch.vertices[0].col, col);
            // Convert distance to similarity (lower distance = higher similarity)
            color_similarity = 255.0f - std::min(distance, 255.0f);
        } else {
            color_similarity = 255.0f; // Perfect match for empty batch
        }

        // Consider batch utilization - prefer fuller batches to reduce draw calls
        float utilization = static_cast<float>(batch.vertices.size()) / 65535.0f;

        // Calculate overall score (color similarity + utilization factor)
        float score = color_similarity + (utilization * 100.0f);

        if (score > best_score) {
            best_score = score;
            best_batch = &batch;
        }
    }

    // If we found a compatible batch, return it
    if (best_batch) {
        return best_batch;
    }

    // Otherwise, create a new batch
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    initializeBatch(new_batch, texture);

    return &new_batch;
}

// Ultra-efficient version that prioritizes maximum batching for order book rendering
OrderbookBatchElement* OrderbookBatcher::findOrCreateBestCompatibleBatchUltra(ImTextureID texture, ImU32 col) {
    OrderbookBatchElement* best_batch = nullptr;
    float best_score = -1.0f; // Higher score means better compatibility

    // Look for the best compatible batch among existing ones with more lenient color matching
    for (auto& batch : batches_) {
        if (batch.texture != texture) {
            continue;
        }

        if (batch.vertices.size() >= 65535 - 4 || batch.indices.size() >= 65535 - 6) {
            continue; // Would exceed limits
        }

        // Calculate compatibility score with more lenient color similarity for maximum batching
        float color_similarity = 0.0f;
        if (!batch.vertices.empty()) {
            // Create a temporary batch element to test compatibility
            OrderbookBatchElement temp_batch;
            temp_batch.texture = texture;
            temp_batch.vertices.push_back({{}, {}, col}); // Add a dummy vertex with the target color

            // Use the ultra combination function for more lenient matching
            if (canCombineBatchesUltra(batch, temp_batch, 50, 100.0f)) {
                // If colors are considered compatible by ultra standards, give high similarity score
                color_similarity = 200.0f; // High score for ultra-compatible colors
            } else {
                float distance = getColorDistance(batch.vertices[0].col, col);
                // Convert distance to similarity (lower distance = higher similarity)
                color_similarity = 255.0f - std::min(distance, 255.0f);
            }

            // Clean up the temporary batch
            temp_batch.vertices.clear();
        } else {
            color_similarity = 255.0f; // Perfect match for empty batch
        }

        // Consider batch utilization - heavily prefer fuller batches to reduce draw calls
        float utilization = static_cast<float>(batch.vertices.size()) / 65535.0f;

        // Calculate overall score (color similarity + strong utilization factor)
        float score = color_similarity + (utilization * 200.0f); // Higher weight for utilization

        if (score > best_score) {
            best_score = score;
            best_batch = &batch;
        }
    }

    // If we found a compatible batch, return it
    if (best_batch) {
        return best_batch;
    }

    // Otherwise, create a new batch
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

    // Optimized batching: group rectangles by texture and similar colors to maximize batching efficiency
    // This is much more efficient than calling addRectFilled individually for each rectangle

    // First, group rectangles by compatible batches to minimize the number of batch lookups
    struct RectangleGroup {
        std::vector<std::pair<ImVec2, ImVec2>> rectangles;
        std::vector<ImU32> rectangle_colors;
    };

    std::vector<RectangleGroup> groups;
    std::vector<OrderbookBatchElement*> batch_pointers;

    // Process each rectangle and group them by compatible batches
    for (size_t i = 0; i < rect_pairs.size(); ++i) {
        auto* batch = findOrCreateBestCompatibleBatch((ImTextureID)0, colors[i]);  // Use the enhanced method

        // Find if this batch already has a group
        bool found_group = false;
        for (size_t j = 0; j < batch_pointers.size(); ++j) {
            if (batch_pointers[j] == batch) {
                groups[j].rectangles.push_back(rect_pairs[i]);
                groups[j].rectangle_colors.push_back(colors[i]);
                found_group = true;
                break;
            }
        }

        if (!found_group) {
            groups.emplace_back();
            groups.back().rectangles.push_back(rect_pairs[i]);
            groups.back().rectangle_colors.push_back(colors[i]);
            batch_pointers.push_back(batch);
        }
    }

    // Now add all rectangles to their respective batches in bulk
    for (size_t i = 0; i < groups.size(); ++i) {
        auto* batch = batch_pointers[i];
        const auto& group = groups[i];

        size_t initial_vertex_count = batch->vertices.size();
        size_t initial_index_count = batch->indices.size();

        // Pre-calculate required space and reserve if needed
        batch->vertices.reserve(batch->vertices.size() + group.rectangles.size() * 4);
        batch->indices.reserve(batch->indices.size() + group.rectangles.size() * 6);

        // Add all rectangles in this group to the batch
        for (size_t j = 0; j < group.rectangles.size(); ++j) {
            const auto& rect = group.rectangles[j];
            ImU32 col = group.rectangle_colors[j];

            size_t vertex_start = batch->vertices.size();

            batch->vertices.push_back({rect.first, {0, 0}, col});                    // Top-left
            batch->vertices.push_back({ImVec2(rect.second.x, rect.first.y), {1, 0}, col});   // Top-right
            batch->vertices.push_back({rect.second, {1, 1}, col});                    // Bottom-right
            batch->vertices.push_back({ImVec2(rect.first.x, rect.second.y), {0, 1}, col});   // Bottom-left

            // Add 6 indices to form 2 triangles (0,1,2 and 0,2,3)
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
        }
    }
}

// Enhanced method for adding multiple rectangles with even better batching
void OrderbookBatcher::addRectanglesFilledOptimized(const std::vector<std::pair<ImVec2, ImVec2>>& rect_pairs,
                                                   const std::vector<ImU32>& colors) {
    if (rect_pairs.size() != colors.size() || rect_pairs.empty()) {
        return;
    }

    // Group rectangles by texture and color similarity first, then assign to batches
    // This creates fewer, larger batches which reduces draw calls

    // Map to store groups of rectangles by texture and representative color
    std::map<std::pair<ImTextureID, ImU32>, std::vector<size_t>> color_groups;

    for (size_t i = 0; i < colors.size(); ++i) {
        bool found_group = false;

        // Look for a compatible color group
        for (auto& [key, indices] : color_groups) {
            if (areColorsSimilar(key.second, colors[i])) {
                indices.push_back(i);
                found_group = true;
                break;
            }
        }

        if (!found_group) {
            // Create a new group with this color as representative
            color_groups[{(ImTextureID)0, colors[i]}].push_back(i);
        }
    }

    // Process each color group
    for (const auto& [key, indices] : color_groups) {
        // Find or create the best compatible batch for this group
        auto* batch = findOrCreateBestCompatibleBatch(key.first, key.second);

        // Pre-allocate space for all rectangles in this group
        batch->vertices.reserve(batch->vertices.size() + indices.size() * 4);
        batch->indices.reserve(batch->indices.size() + indices.size() * 6);

        // Add all rectangles in this group to the batch
        for (size_t idx : indices) {
            const auto& rect = rect_pairs[idx];
            ImU32 col = colors[idx];

            size_t vertex_start = batch->vertices.size();

            batch->vertices.push_back({rect.first, {0, 0}, col});                    // Top-left
            batch->vertices.push_back({ImVec2(rect.second.x, rect.first.y), {1, 0}, col});   // Top-right
            batch->vertices.push_back({rect.second, {1, 1}, col});                    // Bottom-right
            batch->vertices.push_back({ImVec2(rect.first.x, rect.second.y), {0, 1}, col});   // Bottom-left

            // Add 6 indices to form 2 triangles (0,1,2 and 0,2,3)
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
            batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
        }
    }
}

// Enhanced method to batch multiple similar elements together for maximum efficiency
void OrderbookBatcher::batchSimilarElements(const std::vector<std::function<void(OrderbookBatchElement*)>>& element_adders) {
    if (element_adders.empty()) {
        return;
    }

    // Group similar elements together to maximize batching efficiency
    std::vector<OrderbookBatchElement*> assigned_batches;
    assigned_batches.reserve(element_adders.size());

    // Assign each element to a compatible batch
    for (const auto& adder_func : element_adders) {
        // For this generic batching, we'll use a dummy color and texture
        // In practice, this would be customized based on the specific elements
        auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, 0xFFFFFFFF);
        assigned_batches.push_back(batch);

        // Execute the adder function to add the element to the batch
        adder_func(batch);
    }
}

// Advanced method for order book specific batching - combines multiple elements with intelligent grouping
void OrderbookBatcher::addOrderbookElements(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Group elements by texture and similar colors to maximize batching
    struct ElementGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<ElementGroup> groups;

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group
        for (auto& group : groups) {
            if (group.texture == element.texture &&
                areColorsSimilar(group.representative_color, element.color)) {

                group.grouped_elements.push_back(element);
                found_group = true;
                break;
            }
        }

        // If no compatible group found, create a new one
        if (!found_group) {
            ElementGroup new_group;
            new_group.texture = element.texture;
            new_group.representative_color = element.color;
            new_group.grouped_elements.push_back(element);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group
    for (const auto& group : groups) {
        auto* batch = findOrCreateCompatibleBatch(group.texture, group.representative_color);

        // Check if we need to resize the batch capacity
        resizeBatchIfNeeded(*batch);

        for (const auto& element : group.grouped_elements) {
            // Add the element based on its type
            switch (element.type) {
                case OrderbookElementType::RECT_FILLED:
                    // Add rectangle vertices and indices
                    {
                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({element.rect.min, {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.max.x, element.rect.min.y), {1, 0}, element.color});
                        batch->vertices.push_back({element.rect.max, {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.min.x, element.rect.max.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;

                case OrderbookElementType::LINE:
                    // Add line vertices and indices
                    {
                        ImVec2 delta = ImVec2(element.line.p2.x - element.line.p1.x, element.line.p2.y - element.line.p1.y);
                        float length = sqrtf(delta.x * delta.x + delta.y * delta.y);
                        if (length == 0.0f) continue;

                        ImVec2 dir = ImVec2(delta.x / length, delta.y / length);
                        ImVec2 perp = ImVec2(-dir.y, dir.x);

                        float half_thickness = element.thickness * 0.5f;
                        ImVec2 offset = ImVec2(perp.x * half_thickness, perp.y * half_thickness);

                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({ImVec2(element.line.p1.x - offset.x, element.line.p1.y - offset.y), {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p1.x + offset.x, element.line.p1.y + offset.y), {1, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p2.x + offset.x, element.line.p2.y + offset.y), {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p2.x - offset.x, element.line.p2.y - offset.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;

                default:
                    // For other types, just add as a rectangle
                    {
                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({element.rect.min, {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.max.x, element.rect.min.y), {1, 0}, element.color});
                        batch->vertices.push_back({element.rect.max, {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.min.x, element.rect.max.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;
            }
        }
    }
}

// Ultra-efficient method for order book specific batching - maximizes batching by using ultra-compatible grouping
void OrderbookBatcher::addOrderbookElementsUltra(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Group elements by texture and ultra-compatible colors to maximize batching
    struct ElementGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<ElementGroup> groups;

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group using ultra-compatible color matching
        for (auto& group : groups) {
            if (group.texture == element.texture) {
                // Use ultra-compatible color matching for maximum batching
                OrderbookBatchElement temp_batch;
                temp_batch.texture = group.texture;
                temp_batch.vertices.push_back({{}, {}, group.representative_color});

                OrderbookBatchElement temp_element_batch;
                temp_element_batch.texture = element.texture;
                temp_element_batch.vertices.push_back({{}, {}, element.color});

                if (canCombineBatchesUltra(temp_batch, temp_element_batch, 50, 100.0f)) {
                    group.grouped_elements.push_back(element);
                    found_group = true;
                }

                // Clean up temporary batches
                temp_batch.vertices.clear();
                temp_element_batch.vertices.clear();

                if (found_group) break;
            }
        }

        // If no compatible group found, create a new one
        if (!found_group) {
            ElementGroup new_group;
            new_group.texture = element.texture;
            new_group.representative_color = element.color;
            new_group.grouped_elements.push_back(element);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group using ultra-compatible batching
    for (const auto& group : groups) {
        // Use ultra-compatible batch finding for maximum batching
        auto* batch = findOrCreateBestCompatibleBatchUltra(group.texture, group.representative_color);

        // Check if we need to resize the batch capacity
        resizeBatchIfNeeded(*batch);

        for (const auto& element : group.grouped_elements) {
            // Add the element based on its type
            switch (element.type) {
                case OrderbookElementType::RECT_FILLED:
                    // Add rectangle vertices and indices
                    {
                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({element.rect.min, {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.max.x, element.rect.min.y), {1, 0}, element.color});
                        batch->vertices.push_back({element.rect.max, {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.min.x, element.rect.max.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;

                case OrderbookElementType::LINE:
                    // Add line vertices and indices
                    {
                        ImVec2 delta = ImVec2(element.line.p2.x - element.line.p1.x, element.line.p2.y - element.line.p1.y);
                        float length = sqrtf(delta.x * delta.x + delta.y * delta.y);
                        if (length == 0.0f) continue;

                        ImVec2 dir = ImVec2(delta.x / length, delta.y / length);
                        ImVec2 perp = ImVec2(-dir.y, dir.x);

                        float half_thickness = element.thickness * 0.5f;
                        ImVec2 offset = ImVec2(perp.x * half_thickness, perp.y * half_thickness);

                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({ImVec2(element.line.p1.x - offset.x, element.line.p1.y - offset.y), {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p1.x + offset.x, element.line.p1.y + offset.y), {1, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p2.x + offset.x, element.line.p2.y + offset.y), {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.line.p2.x - offset.x, element.line.p2.y - offset.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;

                default:
                    // For other types, just add as a rectangle
                    {
                        size_t vertex_start = batch->vertices.size();

                        batch->vertices.push_back({element.rect.min, {0, 0}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.max.x, element.rect.min.y), {1, 0}, element.color});
                        batch->vertices.push_back({element.rect.max, {1, 1}, element.color});
                        batch->vertices.push_back({ImVec2(element.rect.min.x, element.rect.max.y), {0, 1}, element.color});

                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 1));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 0));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 2));
                        batch->indices.push_back(static_cast<ImDrawIdx>(vertex_start + 3));
                    }
                    break;
            }
        }
    }
}

void OrderbookBatcher::optimizeBatches() {
    // Attempt to merge compatible batches to reduce draw calls
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // More aggressive optimization: merge batches with same texture and similar colors
    // This is particularly beneficial for order book rendering where we have many similar elements
    std::vector<OrderbookBatchElement> optimized_batches;
    optimized_batches.reserve(batches_.size());

    for (auto& current_batch : batches_) {
        bool merged = false;

        // Try to find an existing batch with the same texture and similar color to merge with
        // Iterate backwards to find the most recently added compatible batch (better cache locality)
        for (auto it = optimized_batches.rbegin(); it != optimized_batches.rend(); ++it) {
            // Use the advanced helper function to determine if batches can be combined
            if (canCombineBatchesAdvanced(*it, current_batch)) {
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

// Advanced optimization that groups batches by texture first, then by color similarity
void OrderbookBatcher::advancedOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Group batches by texture first to minimize texture switches
    std::map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        texture_groups[batches_[i].texture].push_back(i);
    }

    std::vector<OrderbookBatchElement> optimized_batches;

    // Process each texture group separately
    for (auto& [texture, indices] : texture_groups) {
        // Within each texture group, try to merge batches with similar colors
        std::vector<bool> processed(indices.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (processed[i]) continue;

            size_t current_idx = indices[i];
            OrderbookBatchElement combined_batch = std::move(batches_[current_idx]);
            processed[i] = true;

            // Look for other batches in the same texture group that can be merged
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (processed[j]) continue;

                size_t candidate_idx = indices[j];

                if (canCombineBatchesAdvanced(combined_batch, batches_[candidate_idx])) {
                    // Merge the candidate batch into the combined batch
                    size_t vertex_offset = combined_batch.vertices.size();

                    // Add vertices from candidate batch to combined batch
                    combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                  batches_[candidate_idx].vertices.begin(),
                                                  batches_[candidate_idx].vertices.end());

                    // Add indices from candidate batch to combined batch with proper offset
                    for (auto index : batches_[candidate_idx].indices) {
                        combined_batch.indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
                    }

                    processed[j] = true;
                }
            }

            optimized_batches.emplace_back(std::move(combined_batch));
        }
    }

    // Replace the old batches with the optimized ones
    batches_ = std::move(optimized_batches);
}

// Ultra-optimized batching that aggressively combines batches to minimize draw calls
void OrderbookBatcher::ultraOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Group batches by texture first to minimize texture switches
    std::map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        texture_groups[batches_[i].texture].push_back(i);
    }

    std::vector<OrderbookBatchElement> optimized_batches;

    // Process each texture group separately with ultra-aggressive merging
    for (auto& [texture, indices] : texture_groups) {
        // Within each texture group, aggressively merge batches with similar colors
        std::vector<bool> processed(indices.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (processed[i]) continue;

            size_t current_idx = indices[i];
            OrderbookBatchElement combined_batch = std::move(batches_[current_idx]);
            processed[i] = true;

            // Look for other batches in the same texture group that can be merged
            // Use ultra-aggressive merging to maximize batching
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (processed[j]) continue;

                size_t candidate_idx = indices[j];

                if (canCombineBatchesUltra(combined_batch, batches_[candidate_idx])) {
                    // Merge the candidate batch into the combined batch
                    size_t vertex_offset = combined_batch.vertices.size();

                    // Add vertices from candidate batch to combined batch
                    combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                  batches_[candidate_idx].vertices.begin(),
                                                  batches_[candidate_idx].vertices.end());

                    // Add indices from candidate batch to combined batch with proper offset
                    for (auto index : batches_[candidate_idx].indices) {
                        combined_batch.indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
                    }

                    processed[j] = true;
                }
            }

            optimized_batches.emplace_back(std::move(combined_batch));
        }
    }

    // Replace the old batches with the optimized ones
    batches_ = std::move(optimized_batches);
}

void OrderbookBatcher::addRectFilled(const ImVec2& min, const ImVec2& max, ImU32 col) {
    // Find or create a compatible batch for filled rectangles
    // For order book rendering, we prioritize batching by texture and color
    auto* batch = findOrCreateCompatibleBatch((ImTextureID)0, col);

    // Check if we need to resize the batch capacity
    resizeBatchIfNeeded(*batch);

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

    // Check if we need to resize the batch capacity
    resizeBatchIfNeeded(*batch);

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

    // Check if we need to resize the batch capacity
    resizeBatchIfNeeded(*batch);

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

    // Check if we need to resize the batch capacity
    resizeBatchIfNeeded(*batch);

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

    // Use ultra optimization to minimize draw calls and GPU overhead
    ultraOptimizeBatches();

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

    // Since we've already optimized by texture in advancedOptimizeBatches,
    // the batches are already organized to minimize texture switches
    for (const auto& batch : batches_) {
        if (!batch.vertices.empty() && !batch.indices.empty()) {
            // Properly set up draw command with texture and scissor clip
            ImDrawCmd cmd;
            cmd.TexRef = batch.texture;
            cmd.ClipRect = draw_list->_CmdHeader.ClipRect;
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