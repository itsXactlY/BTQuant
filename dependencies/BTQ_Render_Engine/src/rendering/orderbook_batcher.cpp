#include "../../include/components/orderbook_batcher.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>

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


// Helper function to determine if two batches can be combined with ultra compatibility
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

    // For empty batches, allow combination
    if (batch1.vertices.empty() || batch2.vertices.empty()) {
        return true;
    }

    // Use both color similarity and distance for accurate batching decisions
    if (!areColorsSimilar(batch1.vertices[0].col, batch2.vertices[0].col, color_tolerance)) {
        float distance = getColorDistance(batch1.vertices[0].col, batch2.vertices[0].col);
        if (distance > max_color_distance) {
            return false;
        }
    }

    return true;
}

// Helper function to determine if two batches can be combined with super compatibility
static bool canCombineBatchesSuper(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                                  uint8_t color_tolerance = 75, float max_color_distance = 150.0f) {
    // Check if textures match - this is the primary criterion for batching
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

    // Use both color similarity and distance for accurate batching decisions
    if (!areColorsSimilar(batch1.vertices[0].col, batch2.vertices[0].col, color_tolerance)) {
        float distance = getColorDistance(batch1.vertices[0].col, batch2.vertices[0].col);
        if (distance > max_color_distance) {
            return false;
        }
    }

    return true;
}

// Ultra-performance helper function optimized for maximum batching efficiency with reduced overhead
static bool canCombineBatchesUltraPerformance(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                                             uint8_t color_tolerance = 100, float max_color_distance = 200.0f) {
    // Quick texture comparison - this is the most important check
    if (batch1.texture != batch2.texture) {
        return false;
    }

    // Early exit if combining would exceed limits
    if (batch1.vertices.size() + batch2.vertices.size() >= 65535 ||
        batch1.indices.size() + batch2.indices.size() >= 65535) {
        return false;
    }

    // For empty batches, allow combination
    if (batch1.vertices.empty() || batch2.vertices.empty()) {
        return true;
    }

    // Use ultra-lenient color matching for maximum performance
    // Compare colors with minimal computation
    ImU32 col1 = batch1.vertices[0].col;
    ImU32 col2 = batch2.vertices[0].col;

    // Fast approximate color similarity check
    uint8_t r1 = (col1 >> 0) & 0xFF;
    uint8_t g1 = (col1 >> 8) & 0xFF;
    uint8_t b1 = (col1 >> 16) & 0xFF;
    uint8_t a1 = (col1 >> 24) & 0xFF;

    uint8_t r2 = (col2 >> 0) & 0xFF;
    uint8_t g2 = (col2 >> 8) & 0xFF;
    uint8_t b2 = (col2 >> 16) & 0xFF;
    uint8_t a2 = (col2 >> 24) & 0xFF;

    // Use a faster approximation for color distance calculation
    int dr = abs(static_cast<int>(r1) - static_cast<int>(r2));
    int dg = abs(static_cast<int>(g1) - static_cast<int>(g2));
    int db = abs(static_cast<int>(b1) - static_cast<int>(b2));
    int da = abs(static_cast<int>(a1) - static_cast<int>(a2));

    // Sum of absolute differences as a fast approximation
    return (dr + dg + db + da) <= (color_tolerance * 2);
}

// Optimized helper function to determine if two batches can be combined with configurable parameters
static bool canCombineBatches(const OrderbookBatchElement& batch1, const OrderbookBatchElement& batch2,
                              uint8_t color_tolerance = 30, float max_color_distance = 75.0f) {
    // Check if textures match - this is the primary criterion for batching
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

    // Use both color similarity and distance for accurate batching decisions
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

// Ultra-performance version that prioritizes maximum batching with minimum computational overhead
OrderbookBatchElement* OrderbookBatcher::findOrCreateBestCompatibleBatchUltraPerformance(ImTextureID texture, ImU32 col) {
    // Look for the first compatible batch to minimize search time
    // This is faster than scoring all batches but may not be perfectly optimal
    for (auto& batch : batches_) {
        if (batch.texture == texture) {
            // Check if there's enough space
            if (batch.vertices.size() < 65535 - 4 && batch.indices.size() < 65535 - 6) {
                // Use ultra-fast compatibility check
                if (batch.vertices.empty()) {
                    return &batch; // Empty batch is always compatible
                }

                // Use the ultra-performance combination function for fastest matching
                OrderbookBatchElement temp_batch;
                temp_batch.texture = texture;
                temp_batch.vertices.push_back({{}, {}, col});

                if (canCombineBatchesUltraPerformance(batch, temp_batch, 100, 200.0f)) {
                    temp_batch.vertices.clear();
                    return &batch;
                }

                temp_batch.vertices.clear();
            }
        }
    }

    // If no compatible batch found, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    // Use larger initial allocation to reduce reallocations
    new_batch.vertices.reserve(8192);
    new_batch.indices.reserve(16384);
    new_batch.texture = texture;
    new_batch.primitive_type = 0;

    return &new_batch;
}

// Ultra-compatible version that prioritizes maximum batching with ultra-relaxed color matching
OrderbookBatchElement* OrderbookBatcher::findOrCreateBestCompatibleBatchUltra(ImTextureID texture, ImU32 col) {
    // Look for the first compatible batch to minimize search time
    for (auto& batch : batches_) {
        if (batch.texture == texture) {
            // Check if there's enough space
            if (batch.vertices.size() < 65535 - 4 && batch.indices.size() < 65535 - 6) {
                // Use ultra-compatible check
                if (batch.vertices.empty()) {
                    return &batch; // Empty batch is always compatible
                }

                // Use the ultra-compatibility combination function
                OrderbookBatchElement temp_batch;
                temp_batch.texture = texture;
                temp_batch.vertices.push_back({{}, {}, col});

                if (canCombineBatchesUltra(batch, temp_batch, 60, 120.0f)) {
                    temp_batch.vertices.clear();
                    return &batch;
                }

                temp_batch.vertices.clear();
            }
        }
    }

    // If no compatible batch found, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    new_batch.vertices.reserve(4096);
    new_batch.indices.reserve(8192);
    new_batch.texture = texture;
    new_batch.primitive_type = 0;

    return &new_batch;
}

// Super-compatible version that prioritizes maximum batching with super-relaxed color matching
OrderbookBatchElement* OrderbookBatcher::findOrCreateBestCompatibleBatchSuper(ImTextureID texture, ImU32 col) {
    // Look for the first compatible batch to minimize search time
    for (auto& batch : batches_) {
        if (batch.texture == texture) {
            // Check if there's enough space
            if (batch.vertices.size() < 65535 - 4 && batch.indices.size() < 65535 - 6) {
                // Use super-compatible check
                if (batch.vertices.empty()) {
                    return &batch; // Empty batch is always compatible
                }

                // Use the super-compatibility combination function
                OrderbookBatchElement temp_batch;
                temp_batch.texture = texture;
                temp_batch.vertices.push_back({{}, {}, col});

                if (canCombineBatchesSuper(batch, temp_batch, 80, 160.0f)) {
                    temp_batch.vertices.clear();
                    return &batch;
                }

                temp_batch.vertices.clear();
            }
        }
    }

    // If no compatible batch found, create a new one
    batches_.emplace_back();
    auto& new_batch = batches_.back();
    new_batch.vertices.reserve(4096);
    new_batch.indices.reserve(8192);
    new_batch.texture = texture;
    new_batch.primitive_type = 0;

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

// Ultra-performance method for order book specific batching - uses ultra-fast grouping and batching
void OrderbookBatcher::addOrderbookElementsUltraPerformance(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Use a more efficient approach with pre-allocated containers to reduce allocations
    struct ElementGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<ElementGroup> groups;
    groups.reserve(elements.size() / 4); // Estimate initial capacity

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group using ultra-fast color matching
        for (auto& group : groups) {
            if (group.texture == element.texture) {
                // Use ultra-fast color similarity check
                uint8_t r1 = (group.representative_color >> 0) & 0xFF;
                uint8_t g1 = (group.representative_color >> 8) & 0xFF;
                uint8_t b1 = (group.representative_color >> 16) & 0xFF;
                uint8_t a1 = (group.representative_color >> 24) & 0xFF;

                uint8_t r2 = (element.color >> 0) & 0xFF;
                uint8_t g2 = (element.color >> 8) & 0xFF;
                uint8_t b2 = (element.color >> 16) & 0xFF;
                uint8_t a2 = (element.color >> 24) & 0xFF;

                // Fast sum of absolute differences
                int color_diff = abs(static_cast<int>(r1) - static_cast<int>(r2)) +
                                abs(static_cast<int>(g1) - static_cast<int>(g2)) +
                                abs(static_cast<int>(b1) - static_cast<int>(b2)) +
                                abs(static_cast<int>(a1) - static_cast<int>(a2));

                if (color_diff <= 120) { // Tolerance for ultra-fast matching
                    group.grouped_elements.push_back(element);
                    found_group = true;
                    break;
                }
            }
        }

        // If no compatible group found, create a new one
        if (!found_group) {
            ElementGroup new_group;
            new_group.texture = element.texture;
            new_group.representative_color = element.color;
            new_group.grouped_elements.reserve(16); // Pre-allocate for expected group size
            new_group.grouped_elements.push_back(element);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group using ultra-performance batch finding
    for (const auto& group : groups) {
        // Use ultra-performance batch finding for maximum efficiency
        auto* batch = findOrCreateBestCompatibleBatchUltraPerformance(group.texture, group.representative_color);

        // Pre-allocate space for all elements in the group to minimize reallocations
        batch->vertices.reserve(batch->vertices.size() + group.grouped_elements.size() * 4);
        batch->indices.reserve(batch->indices.size() + group.grouped_elements.size() * 6);

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

// Super-efficient method for order book specific batching - maximizes batching by using super-compatible grouping
void OrderbookBatcher::addOrderbookElementsSuper(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Group elements by texture and super-compatible colors to maximize batching
    struct ElementGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<ElementGroup> groups;

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group using super-compatible color matching
        for (auto& group : groups) {
            if (group.texture == element.texture) {
                // Use super-compatible color matching for maximum batching
                OrderbookBatchElement temp_batch;
                temp_batch.texture = group.texture;
                temp_batch.vertices.push_back({{}, {}, group.representative_color});

                OrderbookBatchElement temp_element_batch;
                temp_element_batch.texture = element.texture;
                temp_element_batch.vertices.push_back({{}, {}, element.color});

                if (canCombineBatchesSuper(temp_batch, temp_element_batch, 75, 150.0f)) {
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

    // Process each group using super-compatible batching
    for (const auto& group : groups) {
        // Use super-compatible batch finding for maximum batching
        auto* batch = findOrCreateBestCompatibleBatchSuper(group.texture, group.representative_color);

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

// Ultra-performance optimization that uses the fastest possible merging to minimize GPU overhead
void OrderbookBatcher::ultraPerformanceOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Group batches by texture first to minimize texture switches
    // Use unordered_map for faster lookups
    std::unordered_map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        texture_groups[batches_[i].texture].push_back(i);
    }

    std::vector<OrderbookBatchElement> optimized_batches;

    // Process each texture group separately with ultra-fast merging
    for (auto& [texture, indices] : texture_groups) {
        // Within each texture group, ultra-fast merge batches with similar colors
        std::vector<bool> processed(indices.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (processed[i]) continue;

            size_t current_idx = indices[i];
            OrderbookBatchElement combined_batch = std::move(batches_[current_idx]);
            processed[i] = true;

            // Look for other batches in the same texture group that can be merged
            // Use ultra-fast merging to maximize batching with minimal computation
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (processed[j]) continue;

                size_t candidate_idx = indices[j];

                if (canCombineBatchesUltraPerformance(combined_batch, batches_[candidate_idx])) {
                    // Fast merge the candidate batch into the combined batch
                    size_t vertex_offset = combined_batch.vertices.size();

                    // Reserve space to avoid multiple reallocations
                    combined_batch.vertices.reserve(combined_batch.vertices.size() +
                                                  batches_[candidate_idx].vertices.size());

                    // Fast copy vertices using insert
                    combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                  batches_[candidate_idx].vertices.begin(),
                                                  batches_[candidate_idx].vertices.end());

                    // Fast copy indices with offset
                    size_t old_idx_size = combined_batch.indices.size();
                    combined_batch.indices.resize(old_idx_size + batches_[candidate_idx].indices.size());

                    for (size_t k = 0; k < batches_[candidate_idx].indices.size(); ++k) {
                        combined_batch.indices[old_idx_size + k] =
                            static_cast<ImDrawIdx>(batches_[candidate_idx].indices[k] + vertex_offset);
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

// Efficient method to combine similar batches and reduce draw calls
void OrderbookBatcher::combineSimilarBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to combine
    }

    // Group batches by texture first to minimize texture switches
    std::unordered_map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        texture_groups[batches_[i].texture].push_back(i);
    }

    std::vector<OrderbookBatchElement> combined_batches;

    // Process each texture group separately
    for (const auto& [texture, indices] : texture_groups) {
        // Create a map to group batches by color characteristics for efficient lookup
        std::vector<OrderbookBatchElement> temp_batches;

        for (size_t idx : indices) {
            const auto& current_batch = batches_[idx];

            // Try to find a compatible batch to merge with
            bool merged = false;
            for (auto& target_batch : temp_batches) {
                if (canCombineBatches(target_batch, current_batch, 40, 100.0f)) {
                    // Merge the current batch into the target batch
                    size_t vertex_offset = target_batch.vertices.size();

                    // Efficiently append vertices
                    target_batch.vertices.insert(target_batch.vertices.end(),
                                               current_batch.vertices.begin(),
                                               current_batch.vertices.end());

                    // Append indices with proper offset
                    for (auto index : current_batch.indices) {
                        target_batch.indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
                    }

                    merged = true;
                    break;
                }
            }

            // If no compatible batch was found, add as a new batch
            if (!merged) {
                temp_batches.push_back(current_batch);
            }
        }

        // Move the processed batches to the final result
        for (auto& batch : temp_batches) {
            combined_batches.push_back(std::move(batch));
        }
    }

    // Replace the old batches with the combined ones
    batches_ = std::move(combined_batches);
}


// Enhanced method to batch geometry with maximum efficiency for order book rendering
void OrderbookBatcher::batchGeometry(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Group elements by texture and color similarity to maximize batching efficiency
    struct GeometryGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<GeometryGroup> groups;

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group using optimized color matching
        for (auto& group : groups) {
            if (group.texture == element.texture) {
                // Use optimized color similarity check
                uint8_t r1 = (group.representative_color >> 0) & 0xFF;
                uint8_t g1 = (group.representative_color >> 8) & 0xFF;
                uint8_t b1 = (group.representative_color >> 16) & 0xFF;
                uint8_t a1 = (group.representative_color >> 24) & 0xFF;

                uint8_t r2 = (element.color >> 0) & 0xFF;
                uint8_t g2 = (element.color >> 8) & 0xFF;
                uint8_t b2 = (element.color >> 16) & 0xFF;
                uint8_t a2 = (element.color >> 24) & 0xFF;

                // Fast sum of absolute differences
                int color_diff = abs(static_cast<int>(r1) - static_cast<int>(r2)) +
                                abs(static_cast<int>(g1) - static_cast<int>(g2)) +
                                abs(static_cast<int>(b1) - static_cast<int>(b2)) +
                                abs(static_cast<int>(a1) - static_cast<int>(a2));

                if (color_diff <= 100) { // Tolerance for optimized matching
                    group.grouped_elements.push_back(element);
                    found_group = true;
                    break;
                }
            }
        }

        // If no compatible group found, create a new one
        if (!found_group) {
            GeometryGroup new_group;
            new_group.texture = element.texture;
            new_group.representative_color = element.color;
            new_group.grouped_elements.reserve(32); // Pre-allocate for expected group size
            new_group.grouped_elements.push_back(element);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group using optimized batch finding
    for (const auto& group : groups) {
        // Use optimized batch finding for maximum efficiency
        auto* batch = findOrCreateBestCompatibleBatchUltraPerformance(group.texture, group.representative_color);

        // Pre-allocate space for all elements in the group to minimize reallocations
        batch->vertices.reserve(batch->vertices.size() + group.grouped_elements.size() * 4);
        batch->indices.reserve(batch->indices.size() + group.grouped_elements.size() * 6);

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

                case OrderbookElementType::CIRCLE_FILLED:
                    // Add circle vertices and indices (approximated with segments)
                    {
                        const int segments = 12;
                        const float segment_angle = 2.0f * 3.14159265358979323846f / segments;

                        // Add center vertex
                        size_t center_vertex_idx = batch->vertices.size();
                        batch->vertices.push_back({element.rect.min, {0.5f, 0.5f}, element.color});

                        // Add outer vertices
                        std::vector<size_t> outer_vertices;
                        for (int i = 0; i <= segments; i++) {
                            float angle = i * segment_angle;
                            ImVec2 point = ImVec2(
                                element.rect.min.x + cosf(angle) * 10.0f, // Using 10.0f as radius
                                element.rect.min.y + sinf(angle) * 10.0f
                            );

                            outer_vertices.push_back(batch->vertices.size());
                            batch->vertices.push_back({point, {0.5f + cosf(angle)*0.5f, 0.5f + sinf(angle)*0.5f}, element.color});
                        }

                        // Create triangle fan from center to outer vertices
                        for (size_t i = 0; i < outer_vertices.size() - 1; i++) {
                            batch->indices.push_back(static_cast<ImDrawIdx>(center_vertex_idx));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i]));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i + 1]));
                        }
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

// Advanced batch optimization that minimizes draw calls by maximizing batch sizes
void OrderbookBatcher::advancedBatchOptimization() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Group batches by texture first to minimize texture switches
    std::unordered_map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        texture_groups[batches_[i].texture].push_back(i);
    }

    std::vector<OrderbookBatchElement> optimized_batches;

    // Process each texture group separately with advanced merging
    for (auto& [texture, indices] : texture_groups) {
        // Within each texture group, merge batches with similar colors
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

                if (canCombineBatchesUltra(combined_batch, batches_[candidate_idx], 60, 120.0f)) {
                    // Merge the candidate batch into the combined batch
                    size_t vertex_offset = combined_batch.vertices.size();

                    // Reserve space to avoid multiple reallocations
                    combined_batch.vertices.reserve(combined_batch.vertices.size() +
                                                  batches_[candidate_idx].vertices.size());

                    // Fast copy vertices using insert
                    combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                  batches_[candidate_idx].vertices.begin(),
                                                  batches_[candidate_idx].vertices.end());

                    // Fast copy indices with offset
                    size_t old_idx_size = combined_batch.indices.size();
                    combined_batch.indices.resize(old_idx_size + batches_[candidate_idx].indices.size());

                    for (size_t k = 0; k < batches_[candidate_idx].indices.size(); ++k) {
                        combined_batch.indices[old_idx_size + k] =
                            static_cast<ImDrawIdx>(batches_[candidate_idx].indices[k] + vertex_offset);
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

// Fast batch consolidation to reduce GPU overhead
void OrderbookBatcher::fastConsolidateBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to consolidate
    }

    // Use a more aggressive approach to consolidate batches
    std::vector<OrderbookBatchElement> consolidated_batches;

    for (auto& current_batch : batches_) {
        bool merged = false;

        // Look for a compatible batch to merge with
        for (auto& target_batch : consolidated_batches) {
            if (target_batch.texture == current_batch.texture &&
                target_batch.vertices.size() + current_batch.vertices.size() < 65535 &&
                target_batch.indices.size() + current_batch.indices.size() < 65535) {

                // Check if colors are similar enough to batch together
                if (target_batch.vertices.empty() || current_batch.vertices.empty() ||
                    areColorsSimilar(target_batch.vertices[0].col, current_batch.vertices[0].col, 50)) {

                    // Merge the batches
                    size_t vertex_offset = target_batch.vertices.size();

                    target_batch.vertices.insert(target_batch.vertices.end(),
                                               current_batch.vertices.begin(),
                                               current_batch.vertices.end());

                    for (auto index : current_batch.indices) {
                        target_batch.indices.push_back(static_cast<ImDrawIdx>(index + vertex_offset));
                    }

                    merged = true;
                    break;
                }
            }
        }

        if (!merged) {
            consolidated_batches.push_back(std::move(current_batch));
        }
    }

    batches_ = std::move(consolidated_batches);
}

// Ultra optimization that uses the most aggressive approach to minimize GPU overhead
void OrderbookBatcher::ultraOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Use the most aggressive approach to combine batches
    std::vector<OrderbookBatchElement> optimized_batches;

    // Group batches by texture first to minimize texture switches
    std::unordered_map<ImTextureID, std::vector<size_t>> texture_groups;
    for (size_t i = 0; i < batches_.size(); ++i) {
        if (!batches_[i].vertices.empty() || !batches_[i].indices.empty()) {
            texture_groups[batches_[i].texture].push_back(i);
        }
    }

    // Process each texture group separately with ultra-aggressive merging
    for (auto& [texture, indices] : texture_groups) {
        // Within each texture group, ultra-aggressively merge batches with similar colors
        std::vector<bool> processed(indices.size(), false);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (processed[i]) continue;

            size_t current_idx = indices[i];
            OrderbookBatchElement combined_batch = std::move(batches_[current_idx]);
            processed[i] = true;

            // Look for other batches in the same texture group that can be ultra-aggressively merged
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (processed[j]) continue;

                size_t candidate_idx = indices[j];

                // Ultra-aggressive combination with very relaxed tolerance
                if (combined_batch.vertices.size() + batches_[candidate_idx].vertices.size() < 65535 &&
                    combined_batch.indices.size() + batches_[candidate_idx].indices.size() < 65535) {

                    // Ultra-relaxed color compatibility check
                    if (combined_batch.vertices.empty() || batches_[candidate_idx].vertices.empty() ||
                        areColorsSimilar(combined_batch.vertices[0].col, batches_[candidate_idx].vertices[0].col, 120)) {

                        // Ultra-efficient merge the candidate batch into the combined batch
                        size_t vertex_offset = combined_batch.vertices.size();

                        // Reserve space to avoid multiple reallocations
                        combined_batch.vertices.reserve(combined_batch.vertices.size() +
                                                      batches_[candidate_idx].vertices.size());

                        // Ultra-fast copy vertices using insert
                        combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                      batches_[candidate_idx].vertices.begin(),
                                                      batches_[candidate_idx].vertices.end());

                        // Ultra-fast copy indices with offset
                        size_t old_idx_size = combined_batch.indices.size();
                        combined_batch.indices.reserve(old_idx_size + batches_[candidate_idx].indices.size());

                        combined_batch.indices.insert(combined_batch.indices.end(),
                                                    batches_[candidate_idx].indices.begin(),
                                                    batches_[candidate_idx].indices.end());

                        // Apply vertex offset to newly added indices efficiently
                        for (size_t k = old_idx_size; k < combined_batch.indices.size(); ++k) {
                            combined_batch.indices[k] = static_cast<ImDrawIdx>(
                                combined_batch.indices[k] + vertex_offset);
                        }

                        processed[j] = true;
                    }
                }
            }

            optimized_batches.emplace_back(std::move(combined_batch));
        }
    }

    // Replace the old batches with the optimized ones
    batches_ = std::move(optimized_batches);
}

// Maximum optimization that uses the ultimate approach to minimize GPU overhead
void OrderbookBatcher::maximumOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Use the ultimate approach to combine batches with maximum efficiency
    std::vector<OrderbookBatchElement> optimized_batches;

    // Group batches by texture first to minimize texture switches
    // Use a more efficient approach with pre-allocated containers
    std::unordered_map<ImTextureID, std::vector<OrderbookBatchElement*>> texture_buckets;

    // Create pointers to non-empty batches for efficient processing
    for (auto& batch : batches_) {
        if (!batch.vertices.empty() || !batch.indices.empty()) {
            texture_buckets[batch.texture].push_back(&batch);
        }
    }

    // Process each texture bucket separately with maximum efficiency
    for (auto& [texture, batch_ptrs] : texture_buckets) {
        if (batch_ptrs.empty()) continue;

        // Create an initial batch to accumulate others
        OrderbookBatchElement* accumulator = batch_ptrs[0];

        // Process remaining batches in this texture group
        for (size_t i = 1; i < batch_ptrs.size(); ++i) {
            OrderbookBatchElement* candidate = batch_ptrs[i];

            // Check if we can combine these batches efficiently
            if (accumulator->vertices.size() + candidate->vertices.size() < 65535 &&
                accumulator->indices.size() + candidate->indices.size() < 65535) {

                // Use relaxed color compatibility check for maximum batching
                if (accumulator->vertices.empty() || candidate->vertices.empty() ||
                    areColorsSimilar(accumulator->vertices[0].col, candidate->vertices[0].col, 150)) {

                    // Efficiently merge the candidate into the accumulator
                    size_t vertex_offset = accumulator->vertices.size();

                    // Reserve space to prevent multiple reallocations
                    accumulator->vertices.reserve(accumulator->vertices.size() +
                                                candidate->vertices.size());
                    accumulator->indices.reserve(accumulator->indices.size() +
                                               candidate->indices.size());

                    // Copy vertices
                    accumulator->vertices.insert(accumulator->vertices.end(),
                                               candidate->vertices.begin(),
                                               candidate->vertices.end());

                    // Copy indices with offset
                    for (auto index : candidate->indices) {
                        accumulator->indices.push_back(
                            static_cast<ImDrawIdx>(index + vertex_offset));
                    }

                    // Clear the candidate batch to mark it as merged
                    candidate->vertices.clear();
                    candidate->indices.clear();
                }
            }
        }

        // Add the accumulated batch if it has content
        if (!accumulator->vertices.empty() || !accumulator->indices.empty()) {
            optimized_batches.push_back(std::move(*accumulator));
        }
    }

    // Collect any unprocessed batches (those that couldn't be merged)
    for (auto& batch : batches_) {
        if (!batch.vertices.empty() || !batch.indices.empty()) {
            // Check if this batch was already moved to optimized_batches
            bool already_added = false;
            for (auto& opt_batch : optimized_batches) {
                if (&opt_batch == &batch) {
                    already_added = true;
                    break;
                }
            }

            if (!already_added) {
                optimized_batches.push_back(std::move(batch));
            }
        }
    }

    // Replace the old batches with the optimized ones
    batches_ = std::move(optimized_batches);
}

// Ultimate optimization that uses the most comprehensive approach to minimize GPU overhead
void OrderbookBatcher::ultimateOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Use the most comprehensive approach to combine batches with ultimate efficiency
    std::vector<OrderbookBatchElement> optimized_batches;

    // Use a comprehensive multi-pass approach for maximum optimization
    // First pass: group by texture
    std::unordered_map<ImTextureID, std::vector<size_t>> texture_map;
    for (size_t i = 0; i < batches_.size(); ++i) {
        if (!batches_[i].vertices.empty() || !batches_[i].indices.empty()) {
            texture_map[batches_[i].texture].push_back(i);
        }
    }

    // Second pass: for each texture group, apply comprehensive optimization
    for (const auto& [texture, indices] : texture_map) {
        if (indices.size() <= 1) {
            // If only one batch for this texture, just add it as-is
            if (!indices.empty() &&
                (!batches_[indices[0]].vertices.empty() || !batches_[indices[0]].indices.empty())) {
                optimized_batches.push_back(std::move(batches_[indices[0]]));
            }
            continue;
        }

        // For multiple batches with the same texture, apply comprehensive merging
        std::vector<bool> processed(batches_.size(), false);

        for (size_t i : indices) {
            if (processed[i]) continue;

            OrderbookBatchElement combined_batch = std::move(batches_[i]);
            processed[i] = true;

            // Look for compatible batches to merge with ultimate efficiency
            for (size_t j : indices) {
                if (i == j || processed[j]) continue;

                // Comprehensive compatibility check
                if (combined_batch.vertices.size() + batches_[j].vertices.size() < 65535 &&
                    combined_batch.indices.size() + batches_[j].indices.size() < 65535) {

                    // Ultimate color compatibility check with maximum tolerance
                    if (combined_batch.vertices.empty() || batches_[j].vertices.empty() ||
                        areColorsSimilar(combined_batch.vertices[0].col, batches_[j].vertices[0].col, 200)) {

                        // Ultimate efficient merge
                        size_t vertex_offset = combined_batch.vertices.size();

                        // Pre-allocate for ultimate efficiency
                        combined_batch.vertices.reserve(combined_batch.vertices.size() +
                                                     batches_[j].vertices.size());
                        combined_batch.indices.reserve(combined_batch.indices.size() +
                                                    batches_[j].indices.size());

                        // Ultimate fast copy of vertices
                        combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                     batches_[j].vertices.begin(),
                                                     batches_[j].vertices.end());

                        // Ultimate fast copy of indices with offset
                        for (auto index : batches_[j].indices) {
                            combined_batch.indices.push_back(
                                static_cast<ImDrawIdx>(index + vertex_offset));
                        }

                        processed[j] = true;
                    }
                }
            }

            // Add the ultimately optimized batch
            optimized_batches.push_back(std::move(combined_batch));
        }
    }

    // Replace the old batches with the ultimately optimized ones
    batches_ = std::move(optimized_batches);
}

// GPU-optimized batching that focuses on reducing draw calls and memory allocations
void OrderbookBatcher::gpuOptimizedBatching() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Group batches by texture first to minimize texture switches
    std::unordered_map<ImTextureID, std::vector<OrderbookBatchElement*>> texture_buckets;

    // Create pointers to non-empty batches for efficient processing
    for (auto& batch : batches_) {
        if (!batch.vertices.empty() || !batch.indices.empty()) {
            texture_buckets[batch.texture].push_back(&batch);
        }
    }

    std::vector<OrderbookBatchElement> optimized_batches;

    // Process each texture bucket separately with GPU optimization in mind
    for (auto& [texture, batch_ptrs] : texture_buckets) {
        if (batch_ptrs.empty()) continue;

        // Use a greedy algorithm to combine batches efficiently
        for (size_t i = 0; i < batch_ptrs.size(); ++i) {
            if (batch_ptrs[i]->vertices.empty() && batch_ptrs[i]->indices.empty()) {
                continue; // Skip empty batches
            }

            OrderbookBatchElement current_batch = std::move(*batch_ptrs[i]);

            // Look for compatible batches to merge with this one
            for (size_t j = i + 1; j < batch_ptrs.size(); ++j) {
                if (batch_ptrs[j]->vertices.empty() && batch_ptrs[j]->indices.empty()) {
                    continue; // Skip empty batches
                }

                // Check if batches can be combined efficiently
                if (current_batch.vertices.size() + batch_ptrs[j]->vertices.size() < 65535 &&
                    current_batch.indices.size() + batch_ptrs[j]->indices.size() < 65535) {

                    // Use moderate color compatibility for good batching without excessive color mixing
                    if (current_batch.vertices.empty() || batch_ptrs[j]->vertices.empty() ||
                        areColorsSimilar(current_batch.vertices[0].col, batch_ptrs[j]->vertices[0].col, 100)) {

                        // Efficiently merge the batch
                        size_t vertex_offset = current_batch.vertices.size();

                        // Pre-allocate space to avoid multiple reallocations
                        current_batch.vertices.reserve(current_batch.vertices.size() +
                                                     batch_ptrs[j]->vertices.size());
                        current_batch.indices.reserve(current_batch.indices.size() +
                                                   batch_ptrs[j]->indices.size());

                        // Copy vertices
                        current_batch.vertices.insert(current_batch.vertices.end(),
                                                   batch_ptrs[j]->vertices->begin(),
                                                   batch_ptrs[j]->vertices->end());

                        // Copy indices with offset
                        for (auto index : batch_ptrs[j]->indices) {
                            current_batch.indices.push_back(
                                static_cast<ImDrawIdx>(index + vertex_offset));
                        }

                        // Clear the merged batch to mark it as processed
                        batch_ptrs[j]->vertices.clear();
                        batch_ptrs[j]->indices.clear();
                    }
                }
            }

            // Add the optimized batch if it has content
            if (!current_batch.vertices.empty() || !current_batch.indices.empty()) {
                optimized_batches.push_back(std::move(current_batch));
            }
        }
    }

    // Replace the old batches with the GPU-optimized ones
    batches_ = std::move(optimized_batches);
}

// Memory-efficient batching that reduces memory fragmentation and allocation overhead
void OrderbookBatcher::memoryEfficientBatching() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Create a new vector with optimized memory layout
    std::vector<OrderbookBatchElement> compacted_batches;

    // Reserve space to minimize reallocations
    compacted_batches.reserve(batches_.size());

    // Process batches in a memory-efficient way
    for (auto& batch : batches_) {
        if (!batch.vertices.empty() || !batch.indices.empty()) {
            // Add the batch to the compacted vector
            compacted_batches.push_back(std::move(batch));

            // Shrink to fit to reduce memory usage
            compacted_batches.back().vertices.shrink_to_fit();
            compacted_batches.back().indices.shrink_to_fit();
        }
    }

    // Replace the old batches with the memory-efficient ones
    batches_ = std::move(compacted_batches);
}

// Super optimization that aggressively combines batches to minimize draw calls even further
void OrderbookBatcher::superOptimizeBatches() {
    if (batches_.size() <= 1) {
        return; // Nothing to optimize
    }

    // Use an extremely aggressive approach to combine batches
    std::vector<OrderbookBatchElement> optimized_batches;

    // Sort batches by texture first to group them together
    std::sort(batches_.begin(), batches_.end(), [](const OrderbookBatchElement& a, const OrderbookBatchElement& b) {
        return a.texture < b.texture;
    });

    // Process batches in groups by texture
    size_t start = 0;
    while (start < batches_.size()) {
        ImTextureID current_texture = batches_[start].texture;

        // Find all batches with the same texture
        size_t end = start;
        while (end < batches_.size() && batches_[end].texture == current_texture) {
            ++end;
        }

        // Within the texture group, aggressively combine batches
        for (size_t i = start; i < end; ++i) {
            if (batches_[i].vertices.empty() && batches_[i].indices.empty()) {
                continue; // Skip empty batches
            }

            OrderbookBatchElement combined_batch = std::move(batches_[i]);

            // Look ahead to find compatible batches to merge with
            for (size_t j = i + 1; j < end; ++j) {
                if (batches_[j].vertices.empty() && batches_[j].indices.empty()) {
                    continue; // Skip empty batches
                }

                // Use super-compatible combination criteria
                if (combined_batch.vertices.size() + batches_[j].vertices.size() < 65535 &&
                    combined_batch.indices.size() + batches_[j].indices.size() < 65535) {

                    // Check color compatibility with relaxed tolerance
                    if (combined_batch.vertices.empty() || batches_[j].vertices.empty() ||
                        areColorsSimilar(combined_batch.vertices[0].col, batches_[j].vertices[0].col, 80)) {

                        // Merge the batches
                        size_t vertex_offset = combined_batch.vertices.size();

                        // Reserve space to avoid multiple reallocations during insertion
                        combined_batch.vertices.reserve(combined_batch.vertices.size() +
                                                      batches_[j].vertices.size());

                        combined_batch.vertices.insert(combined_batch.vertices.end(),
                                                     batches_[j].vertices.begin(),
                                                     batches_[j].vertices.end());

                        // Reserve space for indices and add with offset
                        size_t old_idx_size = combined_batch.indices.size();
                        combined_batch.indices.reserve(old_idx_size + batches_[j].indices.size());

                        combined_batch.indices.insert(combined_batch.indices.end(),
                                                    batches_[j].indices.begin(),
                                                    batches_[j].indices.end());

                        // Apply vertex offset to newly added indices
                        for (size_t k = old_idx_size; k < combined_batch.indices.size(); ++k) {
                            combined_batch.indices[k] = static_cast<ImDrawIdx>(
                                combined_batch.indices[k] + vertex_offset);
                        }

                        // Mark the batch as processed by clearing it
                        batches_[j].vertices.clear();
                        batches_[j].indices.clear();
                    }
                }
            }

            optimized_batches.push_back(std::move(combined_batch));
        }

        start = end;
    }

    // Remove any remaining empty batches
    optimized_batches.erase(
        std::remove_if(optimized_batches.begin(), optimized_batches.end(),
                      [](const OrderbookBatchElement& batch) {
                          return batch.vertices.empty() && batch.indices.empty();
                      }),
        optimized_batches.end()
    );

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

    // Apply a single, efficient optimization pass to minimize draw calls and GPU overhead
    // Rather than applying multiple optimization passes which can be computationally expensive,
    // we'll use a single optimized pass that combines the most effective techniques
    gpuOptimizedBatching();

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

    // Process batches in a single loop to minimize overhead
    for (const auto& batch : batches_) {
        if (!batch.vertices.empty() && !batch.indices.empty()) {
            // Set up draw command with texture and scissor clip
            ImDrawCmd cmd;
            cmd.TextureId = batch.texture;  // Use the batch's texture ID
            cmd.ClipRect = draw_list->_CmdHeader.ClipRect;
            cmd.VtxOffset = draw_list->_VtxCurrentIdx;
            cmd.IdxOffset = static_cast<unsigned int>(draw_list->IdxBuffer.Size);
            cmd.ElemCount = static_cast<unsigned int>(batch.indices.size());

            // Add the draw command to the draw list
            draw_list->CmdBuffer.push_back(cmd);

            // Copy vertices directly using optimized memory operations for better performance
            if (!batch.vertices.empty()) {
                const size_t vertex_count = batch.vertices.size();

                // Copy vertices one by one, converting from OrderbookBatchVertex to ImDrawVert
                for (size_t i = 0; i < vertex_count; ++i) {
                    const OrderbookBatchVertex& src_vertex = batch.vertices[i];

                    ImDrawVert dst_vertex;
                    dst_vertex.pos = src_vertex.pos;
                    dst_vertex.uv = src_vertex.uv;
                    dst_vertex.col = src_vertex.col;

                    draw_list->_VtxWritePtr[0] = dst_vertex;
                    draw_list->_VtxWritePtr++;
                }
            }

            // Copy indices with proper offset - optimized loop
            if (!batch.indices.empty()) {
                const size_t idx_count = batch.indices.size();
                const ImDrawIdx* src_indices = batch.indices.data();
                ImDrawIdx* dst_indices = draw_list->_IdxWritePtr;

                // Optimize index copying with vertex offset applied
                const ImDrawIdx vtx_offset = static_cast<ImDrawIdx>(draw_list->_VtxCurrentIdx);
                for (size_t i = 0; i < idx_count; ++i) {
                    dst_indices[i] = static_cast<ImDrawIdx>(src_indices[i] + vtx_offset);
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

// Batch multiple similar draw calls together to reduce GPU overhead
void OrderbookBatcher::batchMultipleDrawCalls(const std::vector<std::function<void(OrderbookBatchElement*)>>& draw_calls) {
    if (draw_calls.empty()) {
        return;
    }

    // Group draw calls by texture and similar colors to maximize batching efficiency
    struct DrawCallGroup {
        std::vector<std::function<void(OrderbookBatchElement*)>> grouped_draw_calls;
        ImTextureID texture;
        ImU32 representative_color;
    };

    std::vector<DrawCallGroup> groups;

    for (const auto& draw_call : draw_calls) {
        bool found_group = false;

        // Look for a compatible group using a representative color approach
        for (auto& group : groups) {
            // For this implementation, we'll use a dummy color check
            // In practice, this would be customized based on the specific draw calls
            if (group.texture == (ImTextureID)0) { // Assuming default texture for simplicity
                // Use color similarity to determine compatibility
                if (areColorsSimilar(group.representative_color, 0xFFFFFFFF, 80)) {
                    group.grouped_draw_calls.push_back(draw_call);
                    found_group = true;
                    break;
                }
            }
        }

        // If no compatible group found, create a new one with a default representative
        if (!found_group) {
            DrawCallGroup new_group;
            new_group.texture = (ImTextureID)0; // Default texture
            new_group.representative_color = 0xFFFFFFFF; // Default color
            new_group.grouped_draw_calls.push_back(draw_call);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group
    for (const auto& group : groups) {
        // Find or create a compatible batch for this group
        auto* batch = findOrCreateBestCompatibleBatchUltraPerformance(group.texture, group.representative_color);

        // Execute all draw calls in this group on the same batch
        for (const auto& draw_call : group.grouped_draw_calls) {
            draw_call(batch);
        }
    }
}

// Enhanced batch geometry method that combines multiple similar elements into single draw calls
void OrderbookBatcher::batchGeometryEnhanced(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Group elements by texture and color similarity with enhanced logic
    struct EnhancedGeometryGroup {
        std::vector<OrderbookElementData> grouped_elements;
        ImTextureID texture;
        ImU32 representative_color;
        OrderbookElementType type; // Group by type as well for better batching
    };

    std::vector<EnhancedGeometryGroup> groups;

    for (const auto& element : elements) {
        bool found_group = false;

        // Look for a compatible group with same texture, similar color, and same type
        for (auto& group : groups) {
            if (group.texture == element.texture && group.type == element.type) {
                // Use enhanced color similarity check
                if (areColorsSimilar(group.representative_color, element.color, 60)) {
                    group.grouped_elements.push_back(element);
                    found_group = true;
                    break;
                }
            }
        }

        // If no compatible group found, create a new one
        if (!found_group) {
            EnhancedGeometryGroup new_group;
            new_group.texture = element.texture;
            new_group.representative_color = element.color;
            new_group.type = element.type;
            new_group.grouped_elements.reserve(64); // Pre-allocate for expected group size
            new_group.grouped_elements.push_back(element);
            groups.push_back(std::move(new_group));
        }
    }

    // Process each group with enhanced batching
    for (const auto& group : groups) {
        // Use ultra-performance batch finding for maximum efficiency
        auto* batch = findOrCreateBestCompatibleBatchUltraPerformance(group.texture, group.representative_color);

        // Pre-allocate space based on expected size to minimize reallocations
        size_t estimated_vertices = group.grouped_elements.size() * 4; // 4 vertices per element typically
        size_t estimated_indices = group.grouped_elements.size() * 6;  // 6 indices per element typically

        batch->vertices.reserve(batch->vertices.size() + estimated_vertices);
        batch->indices.reserve(batch->indices.size() + estimated_indices);

        // Add all elements in this group to the batch
        for (const auto& element : group.grouped_elements) {
            // Add the element based on its type
            switch (element.type) {
                case OrderbookElementType::RECT_FILLED:
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

                case OrderbookElementType::CIRCLE_FILLED:
                    {
                        const int segments = 12;
                        const float segment_angle = 2.0f * 3.14159265358979323846f / segments;

                        // Add center vertex
                        size_t center_vertex_idx = batch->vertices.size();
                        batch->vertices.push_back({element.rect.min, {0.5f, 0.5f}, element.color});

                        // Add outer vertices
                        std::vector<size_t> outer_vertices;
                        for (int i = 0; i <= segments; i++) {
                            float angle = i * segment_angle;
                            ImVec2 point = ImVec2(
                                element.rect.min.x + cosf(angle) * 10.0f, // Using 10.0f as radius
                                element.rect.min.y + sinf(angle) * 10.0f
                            );

                            outer_vertices.push_back(batch->vertices.size());
                            batch->vertices.push_back({point, {0.5f + cosf(angle)*0.5f, 0.5f + sinf(angle)*0.5f}, element.color});
                        }

                        // Create triangle fan from center to outer vertices
                        for (size_t i = 0; i < outer_vertices.size() - 1; i++) {
                            batch->indices.push_back(static_cast<ImDrawIdx>(center_vertex_idx));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i]));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i + 1]));
                        }
                    }
                    break;

                default:
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

// Highly optimized batch method that combines similar elements with minimal overhead
void OrderbookBatcher::batchGeometryHighPerformance(const std::vector<OrderbookElementData>& elements) {
    if (elements.empty()) {
        return;
    }

    // Use a fast hash map for grouping elements by texture and color similarity
    // This reduces the complexity from O(n²) to O(n) for grouping
    struct BatchKey {
        ImTextureID texture;
        uint8_t r, g, b, a;  // Quantized color components

        bool operator==(const BatchKey& other) const {
            return texture == other.texture &&
                   abs(static_cast<int>(r) - other.r) <= 10 &&
                   abs(static_cast<int>(g) - other.g) <= 10 &&
                   abs(static_cast<int>(b) - other.b) <= 10 &&
                   abs(static_cast<int>(a) - other.a) <= 10;
        }
    };

    // Custom hash function for BatchKey
    struct BatchKeyHash {
        std::size_t operator()(const BatchKey& k) const {
            // Simple hash combining texture pointer and quantized color values
            auto h1 = std::hash<ImTextureID>{}(k.texture);
            auto h2 = std::hash<uint8_t>{}(k.r);
            auto h3 = std::hash<uint8_t>{}(k.g);
            auto h4 = std::hash<uint8_t>{}(k.b);
            auto h5 = std::hash<uint8_t>{}(k.a);

            // Combine hashes
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
        }
    };

    std::unordered_map<BatchKey, std::vector<size_t>, BatchKeyHash> grouped_elements;

    // Group elements by texture and quantized color in a single pass
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto& element = elements[i];

        // Quantize the color to reduce the number of unique combinations
        uint8_t r = ((element.color >> 0) & 0xFF) / 16;  // Reduce precision to 16-levels
        uint8_t g = ((element.color >> 8) & 0xFF) / 16;
        uint8_t b = ((element.color >> 16) & 0xFF) / 16;
        uint8_t a = ((element.color >> 24) & 0xFF) / 16;

        BatchKey key{element.texture, r, g, b, a};
        grouped_elements[key].push_back(i);
    }

    // Process each group efficiently
    for (const auto& [key, indices] : grouped_elements) {
        // Find or create a compatible batch for this group
        ImU32 representative_color = (key.a << 24) | (key.b << 16) | (key.g << 8) | key.r;
        auto* batch = findOrCreateBestCompatibleBatchUltraPerformance(key.texture, representative_color);

        // Pre-allocate space for all elements in this group to minimize reallocations
        size_t estimated_vertices = indices.size() * 4; // 4 vertices per element typically
        size_t estimated_indices = indices.size() * 6;  // 6 indices per element typically

        batch->vertices.reserve(batch->vertices.size() + estimated_vertices);
        batch->indices.reserve(batch->indices.size() + estimated_indices);

        // Add all elements in this group to the batch
        for (size_t idx : indices) {
            const auto& element = elements[idx];

            // Add the element based on its type
            switch (element.type) {
                case OrderbookElementType::RECT_FILLED:
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

                case OrderbookElementType::CIRCLE_FILLED:
                    {
                        const int segments = 12;
                        const float segment_angle = 2.0f * 3.14159265358979323846f / segments;

                        // Add center vertex
                        size_t center_vertex_idx = batch->vertices.size();
                        batch->vertices.push_back({element.rect.min, {0.5f, 0.5f}, element.color});

                        // Add outer vertices
                        std::vector<size_t> outer_vertices;
                        for (int i = 0; i <= segments; i++) {
                            float angle = i * segment_angle;
                            ImVec2 point = ImVec2(
                                element.rect.min.x + cosf(angle) * 10.0f, // Using 10.0f as radius
                                element.rect.min.y + sinf(angle) * 10.0f
                            );

                            outer_vertices.push_back(batch->vertices.size());
                            batch->vertices.push_back({point, {0.5f + cosf(angle)*0.5f, 0.5f + sinf(angle)*0.5f}, element.color});
                        }

                        // Create triangle fan from center to outer vertices
                        for (size_t i = 0; i < outer_vertices.size() - 1; i++) {
                            batch->indices.push_back(static_cast<ImDrawIdx>(center_vertex_idx));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i]));
                            batch->indices.push_back(static_cast<ImDrawIdx>(outer_vertices[i + 1]));
                        }
                    }
                    break;

                default:
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

} // namespace BTQuant