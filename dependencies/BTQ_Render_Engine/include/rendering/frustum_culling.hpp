#pragma once

/**
 * @file frustum_culling.hpp
 * @brief Frustum Culling for Candlesticks (AABB Bounding Boxes)
 * 
 * This implementation provides:
 * - Axis-Aligned Bounding Box (AABB) culling
 * - View frustum extraction from view-projection matrix
 * - Efficient batch culling for thousands of candlesticks
 * - SIMD-optimized culling where available
 * - Hierarchical culling for large datasets
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <vector>

namespace btq {
namespace rendering {

/**
 * @brief Axis-Aligned Bounding Box
 */
struct AABB {
    glm::vec2 min;  // Minimum corner (x = time, y = low price)
    glm::vec2 max;  // Maximum corner (x = time + width, y = high price)
    
    /**
     * @brief Check if this AABB intersects another
     */
    bool intersects(const AABB& other) const {
        return (min.x <= other.max.x && max.x >= other.min.x) &&
               (min.y <= other.max.y && max.y >= other.min.y);
    }
    
    /**
     * @brief Check if a point is inside this AABB
     */
    bool contains(const glm::vec2& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y;
    }
    
    /**
     * @brief Get the center of the AABB
     */
    glm::vec2 center() const {
        return (min + max) * 0.5f;
    }
    
    /**
     * @brief Get the half-extents of the AABB
     */
    glm::vec2 extents() const {
        return (max - min) * 0.5f;
    }
    
    /**
     * @brief Expand the AABB to include a point
     */
    void expand(const glm::vec2& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
    
    /**
     * @brief Expand the AABB to include another AABB
     */
    void expand(const AABB& other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }
};

/**
 * @brief View frustum for 2D culling
 */
struct ViewFrustum2D {
    // Frustum planes (left, right, bottom, top)
    // Each plane is stored as (normal_x, normal_y, distance)
    std::array<glm::vec3, 4> planes;
    
    /**
     * @brief Extract frustum from view-projection matrix
     */
    static ViewFrustum2D extractFromMatrix(const glm::mat4& view_proj) {
        ViewFrustum2D frustum;
        
        // Left plane
        frustum.planes[0] = glm::vec3(
            view_proj[0][3] + view_proj[0][0],
            view_proj[1][3] + view_proj[1][0],
            view_proj[3][3] + view_proj[3][0]
        );
        
        // Right plane
        frustum.planes[1] = glm::vec3(
            view_proj[0][3] - view_proj[0][0],
            view_proj[1][3] - view_proj[1][0],
            view_proj[3][3] - view_proj[3][0]
        );
        
        // Bottom plane
        frustum.planes[2] = glm::vec3(
            view_proj[0][3] + view_proj[0][1],
            view_proj[1][3] + view_proj[1][1],
            view_proj[3][3] + view_proj[3][1]
        );
        
        // Top plane
        frustum.planes[3] = glm::vec3(
            view_proj[0][3] - view_proj[0][1],
            view_proj[1][3] - view_proj[1][1],
            view_proj[3][3] - view_proj[3][1]
        );
        
        // Normalize planes
        for (auto& plane : frustum.planes) {
            float length = std::sqrt(plane.x * plane.x + plane.y * plane.y);
            if (length > 0.0f) {
                plane /= length;
            }
        }
        
        return frustum;
    }
    
    /**
     * @brief Check if a point is inside the frustum
     */
    bool contains(const glm::vec2& point) const {
        for (const auto& plane : planes) {
            float distance = plane.x * point.x + plane.y * point.y + plane.z;
            if (distance < 0.0f) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Check if an AABB intersects the frustum
     */
    bool intersectsAABB(const AABB& aabb) const {
        // Test each plane
        for (const auto& plane : planes) {
            // Find the positive vertex (furthest along plane normal)
            glm::vec2 positive_vertex;
            positive_vertex.x = (plane.x >= 0.0f) ? aabb.max.x : aabb.min.x;
            positive_vertex.y = (plane.y >= 0.0f) ? aabb.max.y : aabb.min.y;
            
            // If positive vertex is outside plane, AABB is outside frustum
            float distance = plane.x * positive_vertex.x + 
                            plane.y * positive_vertex.y + plane.z;
            
            if (distance < 0.0f) {
                return false;
            }
        }
        
        return true;
    }
};

/**
 * @brief Simple rectangle-based culling for 2D charts
 */
struct ViewBounds {
    float min_x, max_x;  // Time range
    float min_y, max_y;  // Price range
    
    /**
     * @brief Check if an AABB is visible
     */
    bool isVisible(const AABB& aabb) const {
        return aabb.max.x >= min_x && aabb.min.x <= max_x &&
               aabb.max.y >= min_y && aabb.min.y <= max_y;
    }
    
    /**
     * @brief Check if a point is visible
     */
    bool isVisible(const glm::vec2& point) const {
        return point.x >= min_x && point.x <= max_x &&
               point.y >= min_y && point.y <= max_y;
    }
    
    /**
     * @brief Get as AABB
     */
    AABB toAABB() const {
        return {{min_x, min_y}, {max_x, max_y}};
    }
};

/**
 * @brief Candlestick culling result
 */
struct CullingResult {
    std::vector<size_t> visible_indices;  // Indices of visible candlesticks
    size_t total_candlesticks = 0;
    size_t visible_count = 0;
    size_t culled_count = 0;
    float culling_time_ms = 0.0f;
};

/**
 * @brief Frustum culler for candlesticks
 */
class CandlestickCuller {
public:
    CandlestickCuller() = default;
    
    /**
     * @brief Build AABBs for candlesticks
     */
    void buildAABBs(
        const std::vector<float>& times,
        const std::vector<float>& highs,
        const std::vector<float>& lows,
        float candle_width)
    {
        const size_t n = times.size();
        aabbs_.resize(n);
        
        // Build AABBs in parallel
        std::transform(
            std::execution::par_unseq,
            times.begin(), times.end(),
            highs.begin(),
            lows.begin(),
            aabbs_.begin(),
            [candle_width](float time, float high, float low) {
                return AABB{
                    {time, low},
                    {time + candle_width, high}
                };
            });
    }
    
    /**
     * @brief Cull candlesticks against view bounds
     */
    CullingResult cull(const ViewBounds& bounds) const {
        CullingResult result;
        result.total_candlesticks = aabbs_.size();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Reserve space for visible indices
        result.visible_indices.reserve(aabbs_.size());
        
        // Cull each AABB
        for (size_t i = 0; i < aabbs_.size(); ++i) {
            if (bounds.isVisible(aabbs_[i])) {
                result.visible_indices.push_back(i);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.culling_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        result.visible_count = result.visible_indices.size();
        result.culled_count = result.total_candlesticks - result.visible_count;
        
        return result;
    }
    
    /**
     * @brief Cull candlesticks against view bounds (parallel version)
     */
    CullingResult cullParallel(const ViewBounds& bounds) const {
        CullingResult result;
        result.total_candlesticks = aabbs_.size();
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create visibility flags
        std::vector<bool> visible(aabbs_.size());
        
        std::transform(
            std::execution::par_unseq,
            aabbs_.begin(), aabbs_.end(),
            visible.begin(),
            [&bounds](const AABB& aabb) { return bounds.isVisible(aabb); });
        
        // Collect visible indices
        result.visible_indices.reserve(aabbs_.size());
        for (size_t i = 0; i < visible.size(); ++i) {
            if (visible[i]) {
                result.visible_indices.push_back(i);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.culling_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        result.visible_count = result.visible_indices.size();
        result.culled_count = result.total_candlesticks - result.visible_count;
        
        return result;
    }
    
    /**
     * @brief Get the stored AABBs
     */
    const std::vector<AABB>& getAABBs() const { return aabbs_; }

private:
    std::vector<AABB> aabbs_;
};

/**
 * @brief Hierarchical culling using a bounding volume hierarchy (BVH)
 */
class HierarchicalCuller {
public:
    /**
     * @brief BVH node
     */
    struct BVHNode {
        AABB bounds;
        int32_t left_child = -1;   // Index of left child, -1 if leaf
        int32_t right_child = -1;  // Index of right child, -1 if leaf
        int32_t start_index = -1;  // Start index in leaf, -1 if internal
        int32_t count = 0;         // Number of elements in leaf
        bool is_leaf = false;
    };
    
    /**
     * @brief Build BVH from AABBs
     */
    void build(const std::vector<AABB>& aabbs) {
        if (aabbs.empty()) return;
        
        nodes_.clear();
        leaf_indices_.clear();
        
        // Create index array
        std::vector<size_t> indices(aabbs.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Build recursively
        buildNode(aabbs, indices, 0, indices.size());
    }
    
    /**
     * @brief Cull against view bounds using BVH
     */
    CullingResult cull(const ViewBounds& bounds) const {
        CullingResult result;
        result.total_candlesticks = leaf_indices_.size();
        
        if (nodes_.empty()) {
            return result;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Traverse BVH
        std::vector<size_t> stack;
        stack.push_back(0);  // Start with root
        
        while (!stack.empty()) {
            size_t node_idx = stack.back();
            stack.pop_back();
            
            const BVHNode& node = nodes_[node_idx];
            
            // Test node bounds against view
            if (!bounds.isVisible(node.bounds)) {
                continue;  // Cull entire subtree
            }
            
            if (node.is_leaf) {
                // Add all elements in leaf
                for (int32_t i = 0; i < node.count; ++i) {
                    result.visible_indices.push_back(
                        leaf_indices_[node.start_index + i]);
                }
            } else {
                // Push children
                if (node.left_child >= 0) {
                    stack.push_back(node.left_child);
                }
                if (node.right_child >= 0) {
                    stack.push_back(node.right_child);
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.culling_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        result.visible_count = result.visible_indices.size();
        result.culled_count = result.total_candlesticks - result.visible_count;
        
        return result;
    }

private:
    int32_t buildNode(
        const std::vector<AABB>& aabbs,
        std::vector<size_t>& indices,
        size_t start,
        size_t end)
    {
        int32_t node_idx = static_cast<int32_t>(nodes_.size());
        nodes_.push_back(BVHNode{});
        
        BVHNode& node = nodes_.back();
        
        // Calculate bounds for this node
        node.bounds = aabbs[indices[start]];
        for (size_t i = start + 1; i < end; ++i) {
            node.bounds.expand(aabbs[indices[i]]);
        }
        
        size_t count = end - start;
        
        // Leaf node threshold
        if (count <= 4) {
            node.is_leaf = true;
            node.start_index = static_cast<int32_t>(leaf_indices_.size());
            node.count = static_cast<int32_t>(count);
            
            for (size_t i = start; i < end; ++i) {
                leaf_indices_.push_back(indices[i]);
            }
            
            return node_idx;
        }
        
        // Find split axis (longest dimension)
        glm::vec2 extent = node.bounds.extents();
        int axis = (extent.x > extent.y) ? 0 : 1;
        
        // Sort indices along axis
        std::sort(
            indices.begin() + start,
            indices.begin() + end,
            [&aabbs, axis](size_t a, size_t b) {
                return aabbs[a].center()[axis] < aabbs[b].center()[axis];
            });
        
        // Split in middle
        size_t mid = start + count / 2;
        
        node.is_leaf = false;
        node.left_child = buildNode(aabbs, indices, start, mid);
        node.right_child = buildNode(aabbs, indices, mid, end);
        
        return node_idx;
    }
    
    std::vector<BVHNode> nodes_;
    std::vector<size_t> leaf_indices_;
};

/**
 * @brief Utility functions for culling
 */
namespace culling_utils {

/**
 * @brief Calculate view bounds from camera parameters
 */
inline ViewBounds calculateViewBounds(
    const glm::vec2& camera_position,
    const glm::vec2& camera_scale,
    const glm::vec2& viewport_size)
{
    ViewBounds bounds;
    
    float half_width = viewport_size.x / (2.0f * camera_scale.x);
    float half_height = viewport_size.y / (2.0f * camera_scale.y);
    
    bounds.min_x = camera_position.x - half_width;
    bounds.max_x = camera_position.x + half_width;
    bounds.min_y = camera_position.y - half_height;
    bounds.max_y = camera_position.y + half_height;
    
    return bounds;
}

/**
 * @brief Calculate visible candle range
 */
inline std::pair<size_t, size_t> calculateVisibleRange(
    const std::vector<float>& candle_times,
    float candle_width,
    const ViewBounds& bounds)
{
    if (candle_times.empty()) {
        return {0, 0};
    }
    
    // Binary search for first visible candle
    auto start_it = std::lower_bound(
        candle_times.begin(), candle_times.end(),
        bounds.min_x - candle_width);
    
    // Binary search for last visible candle
    auto end_it = std::upper_bound(
        candle_times.begin(), candle_times.end(),
        bounds.max_x);
    
    size_t start_idx = std::distance(candle_times.begin(), start_it);
    size_t end_idx = std::distance(candle_times.begin(), end_it);
    
    return {start_idx, end_idx};
}

/**
 * @brief Quick visibility test for a single candlestick
 */
inline bool isCandleVisible(
    float time,
    float high,
    float low,
    float width,
    const ViewBounds& bounds)
{
    return (time + width >= bounds.min_x) && (time <= bounds.max_x) &&
           (high >= bounds.min_y) && (low <= bounds.max_y);
}

} // namespace culling_utils

} // namespace rendering
} // namespace btq
