/**
 * BTQuant Performance Optimization Engine
 * 
 * Advanced performance optimization system for professional trading dashboard
 * with dynamic LOD, frustum culling, adaptive quality scaling, and GPU optimization.
 */

#include "../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>

namespace BTQuant {

// ============================================================================
// Level of Detail (LOD) System
// ============================================================================

class LODManager {
public:
    enum class LODLevel {
        High = 0,    // Full detail
        Medium = 1,  // Reduced detail
        Low = 2,     // Minimal detail
        Culled = 3   // Not rendered
    };
    
    struct LODSettings {
        float high_distance = 100.0f;
        float medium_distance = 500.0f;
        float low_distance = 1000.0f;
        float cull_distance = 2000.0f;
        
        // Performance thresholds
        float target_fps = 60.0f;
        float fps_tolerance = 5.0f;
        
        // Quality settings per LOD level
        struct QualitySettings {
            float geometry_detail = 1.0f;
            float texture_quality = 1.0f;
            float effect_quality = 1.0f;
            int max_elements = 1000;
            bool enable_shadows = true;
            bool enable_reflections = true;
        };
        
        std::array<QualitySettings, 4> quality_levels;
    };
    
    LODManager() {
        initialize_default_settings();
    }
    
    void update(float delta_time, float current_fps, const glm::vec3& camera_position) {
        current_fps_ = current_fps;
        camera_position_ = camera_position;
        
        // Adaptive LOD based on performance
        if (current_fps < settings_.target_fps - settings_.fps_tolerance) {
            // Performance is poor, reduce quality
            global_lod_bias_ = std::min(global_lod_bias_ + 0.1f, 2.0f);
        } else if (current_fps > settings_.target_fps + settings_.fps_tolerance) {
            // Performance is good, increase quality
            global_lod_bias_ = std::max(global_lod_bias_ - 0.05f, 0.0f);
        }
        
        update_lod_levels();
    }
    
    LODLevel calculate_lod_level(const glm::vec3& object_position, float object_size = 1.0f) const {
        float distance = glm::length(object_position - camera_position_);
        
        // Adjust distance based on object size
        distance /= std::max(object_size, 0.1f);
        
        // Apply global LOD bias
        distance *= (1.0f + global_lod_bias_);
        
        if (distance > settings_.cull_distance) {
            return LODLevel::Culled;
        } else if (distance > settings_.low_distance) {
            return LODLevel::Low;
        } else if (distance > settings_.medium_distance) {
            return LODLevel::Medium;
        } else {
            return LODLevel::High;
        }
    }
    
    const LODSettings::QualitySettings& get_quality_settings(LODLevel level) const {
        return settings_.quality_levels[static_cast<int>(level)];
    }
    
    void set_settings(const LODSettings& settings) {
        settings_ = settings;
    }
    
    float get_global_lod_bias() const { return global_lod_bias_; }
    
    struct LODStats {
        int high_detail_objects = 0;
        int medium_detail_objects = 0;
        int low_detail_objects = 0;
        int culled_objects = 0;
        float average_distance = 0.0f;
        float performance_impact = 0.0f;
    };
    
    LODStats get_stats() const { return current_stats_; }
    
private:
    LODSettings settings_;
    glm::vec3 camera_position_{0.0f};
    float current_fps_ = 60.0f;
    float global_lod_bias_ = 0.0f;
    LODStats current_stats_;
    
    void initialize_default_settings() {
        // High quality settings
        settings_.quality_levels[0] = {
            .geometry_detail = 1.0f,
            .texture_quality = 1.0f,
            .effect_quality = 1.0f,
            .max_elements = 1000,
            .enable_shadows = true,
            .enable_reflections = true
        };
        
        // Medium quality settings
        settings_.quality_levels[1] = {
            .geometry_detail = 0.7f,
            .texture_quality = 0.8f,
            .effect_quality = 0.7f,
            .max_elements = 500,
            .enable_shadows = true,
            .enable_reflections = false
        };
        
        // Low quality settings
        settings_.quality_levels[2] = {
            .geometry_detail = 0.4f,
            .texture_quality = 0.5f,
            .effect_quality = 0.3f,
            .max_elements = 200,
            .enable_shadows = false,
            .enable_reflections = false
        };
        
        // Culled (not rendered)
        settings_.quality_levels[3] = {
            .geometry_detail = 0.0f,
            .texture_quality = 0.0f,
            .effect_quality = 0.0f,
            .max_elements = 0,
            .enable_shadows = false,
            .enable_reflections = false
        };
    }
    
    void update_lod_levels() {
        // Update statistics
        current_stats_ = LODStats{};
        current_stats_.performance_impact = global_lod_bias_;
    }
};

// ============================================================================
// Frustum Culling System
// ============================================================================

class FrustumCuller {
public:
    struct Plane {
        glm::vec3 normal;
        float distance;
        
        float distance_to_point(const glm::vec3& point) const {
            return glm::dot(normal, point) + distance;
        }
    };
    
    struct Frustum {
        std::array<Plane, 6> planes; // left, right, bottom, top, near, far
    };
    
    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        
        glm::vec3 center() const { return (min + max) * 0.5f; }
        glm::vec3 extents() const { return (max - min) * 0.5f; }
    };
    
    FrustumCuller() = default;
    
    void update_frustum(const glm::mat4& view_projection_matrix) {
        extract_frustum_planes(view_projection_matrix, current_frustum_);
    }
    
    bool is_box_in_frustum(const BoundingBox& box) const {
        return test_box_against_frustum(box, current_frustum_);
    }
    
    bool is_sphere_in_frustum(const glm::vec3& center, float radius) const {
        return test_sphere_against_frustum(center, radius, current_frustum_);
    }
    
    bool is_point_in_frustum(const glm::vec3& point) const {
        return test_point_against_frustum(point, current_frustum_);
    }
    
    struct CullingStats {
        int total_objects = 0;
        int visible_objects = 0;
        int culled_objects = 0;
        float culling_efficiency = 0.0f;
        std::chrono::microseconds culling_time{0};
    };
    
    CullingStats get_stats() const { return current_stats_; }
    
    void begin_culling_frame() {
        culling_start_time_ = std::chrono::high_resolution_clock::now();
        current_stats_ = CullingStats{};
    }
    
    void end_culling_frame() {
        auto end_time = std::chrono::high_resolution_clock::now();
        current_stats_.culling_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - culling_start_time_);
        
        if (current_stats_.total_objects > 0) {
            current_stats_.culling_efficiency = 
                static_cast<float>(current_stats_.culled_objects) / current_stats_.total_objects;
        }
    }
    
    void record_object_test(bool visible) {
        current_stats_.total_objects++;
        if (visible) {
            current_stats_.visible_objects++;
        } else {
            current_stats_.culled_objects++;
        }
    }
    
private:
    Frustum current_frustum_;
    CullingStats current_stats_;
    std::chrono::high_resolution_clock::time_point culling_start_time_;
    
    void extract_frustum_planes(const glm::mat4& mvp, Frustum& frustum) {
        // Extract frustum planes from view-projection matrix
        // Left plane
        frustum.planes[0].normal.x = mvp[0][3] + mvp[0][0];
        frustum.planes[0].normal.y = mvp[1][3] + mvp[1][0];
        frustum.planes[0].normal.z = mvp[2][3] + mvp[2][0];
        frustum.planes[0].distance = mvp[3][3] + mvp[3][0];
        
        // Right plane
        frustum.planes[1].normal.x = mvp[0][3] - mvp[0][0];
        frustum.planes[1].normal.y = mvp[1][3] - mvp[1][0];
        frustum.planes[1].normal.z = mvp[2][3] - mvp[2][0];
        frustum.planes[1].distance = mvp[3][3] - mvp[3][0];
        
        // Bottom plane
        frustum.planes[2].normal.x = mvp[0][3] + mvp[0][1];
        frustum.planes[2].normal.y = mvp[1][3] + mvp[1][1];
        frustum.planes[2].normal.z = mvp[2][3] + mvp[2][1];
        frustum.planes[2].distance = mvp[3][3] + mvp[3][1];
        
        // Top plane
        frustum.planes[3].normal.x = mvp[0][3] - mvp[0][1];
        frustum.planes[3].normal.y = mvp[1][3] - mvp[1][1];
        frustum.planes[3].normal.z = mvp[2][3] - mvp[2][1];
        frustum.planes[3].distance = mvp[3][3] - mvp[3][1];
        
        // Near plane
        frustum.planes[4].normal.x = mvp[0][3] + mvp[0][2];
        frustum.planes[4].normal.y = mvp[1][3] + mvp[1][2];
        frustum.planes[4].normal.z = mvp[2][3] + mvp[2][2];
        frustum.planes[4].distance = mvp[3][3] + mvp[3][2];
        
        // Far plane
        frustum.planes[5].normal.x = mvp[0][3] - mvp[0][2];
        frustum.planes[5].normal.y = mvp[1][3] - mvp[1][2];
        frustum.planes[5].normal.z = mvp[2][3] - mvp[2][2];
        frustum.planes[5].distance = mvp[3][3] - mvp[3][2];
        
        // Normalize planes
        for (auto& plane : frustum.planes) {
            float length = glm::length(plane.normal);
            plane.normal /= length;
            plane.distance /= length;
        }
    }
    
    bool test_box_against_frustum(const BoundingBox& box, const Frustum& frustum) const {
        glm::vec3 center = box.center();
        glm::vec3 extents = box.extents();
        
        for (const auto& plane : frustum.planes) {
            float distance = plane.distance_to_point(center);
            float radius = glm::dot(extents, glm::abs(plane.normal));
            
            if (distance < -radius) {
                return false; // Box is completely outside this plane
            }
        }
        
        return true; // Box is at least partially inside frustum
    }
    
    bool test_sphere_against_frustum(const glm::vec3& center, float radius, const Frustum& frustum) const {
        for (const auto& plane : frustum.planes) {
            float distance = plane.distance_to_point(center);
            if (distance < -radius) {
                return false; // Sphere is completely outside this plane
            }
        }
        
        return true; // Sphere is at least partially inside frustum
    }
    
    bool test_point_against_frustum(const glm::vec3& point, const Frustum& frustum) const {
        for (const auto& plane : frustum.planes) {
            if (plane.distance_to_point(point) < 0) {
                return false; // Point is outside this plane
            }
        }
        
        return true; // Point is inside frustum
    }
};

// ============================================================================
// Adaptive Quality Scaling System
// ============================================================================

class AdaptiveQualityScaler {
public:
    struct QualitySettings {
        float render_scale = 1.0f;          // 0.5 to 2.0
        int msaa_samples = 4;               // 1, 2, 4, 8
        bool enable_bloom = true;
        bool enable_ssao = true;
        bool enable_motion_blur = false;
        float shadow_quality = 1.0f;        // 0.0 to 1.0
        float texture_quality = 1.0f;       // 0.0 to 1.0
        int max_lights = 8;                 // 1 to 16
        bool enable_vsync = true;
    };
    
    struct PerformanceMetrics {
        float current_fps = 60.0f;
        float average_fps = 60.0f;
        float frame_time_ms = 16.67f;
        float gpu_usage_percent = 50.0f;
        float vram_usage_percent = 50.0f;
        float cpu_usage_percent = 30.0f;
        int dropped_frames = 0;
    };
    
    AdaptiveQualityScaler() {
        initialize_quality_presets();
        current_settings_ = quality_presets_["High"];
        target_fps_ = 60.0f;
        fps_tolerance_ = 5.0f;
    }
    
    void update(const PerformanceMetrics& metrics) {
        current_metrics_ = metrics;
        
        // Update FPS history
        fps_history_.push_back(metrics.current_fps);
        if (fps_history_.size() > fps_history_size_) {
            fps_history_.pop_front();
        }
        
        // Calculate average FPS
        float avg_fps = 0.0f;
        for (float fps : fps_history_) {
            avg_fps += fps;
        }
        avg_fps /= fps_history_.size();
        
        // Determine if quality adjustment is needed
        if (should_decrease_quality(avg_fps)) {
            decrease_quality();
        } else if (should_increase_quality(avg_fps)) {
            increase_quality();
        }
        
        // Apply emergency quality reduction if needed
        if (metrics.current_fps < target_fps_ * 0.5f) {
            apply_emergency_quality_reduction();
        }
    }
    
    const QualitySettings& get_current_settings() const {
        return current_settings_;
    }
    
    void set_target_fps(float target_fps) {
        target_fps_ = target_fps;
    }
    
    void set_quality_preset(const std::string& preset_name) {
        auto it = quality_presets_.find(preset_name);
        if (it != quality_presets_.end()) {
            current_settings_ = it->second;
            current_preset_ = preset_name;
        }
    }
    
    std::vector<std::string> get_available_presets() const {
        std::vector<std::string> presets;
        for (const auto& pair : quality_presets_) {
            presets.push_back(pair.first);
        }
        return presets;
    }
    
    struct QualityStats {
        std::string current_preset;
        float quality_score = 1.0f;
        int quality_adjustments = 0;
        float performance_gain = 0.0f;
        bool emergency_mode = false;
    };
    
    QualityStats get_stats() const {
        QualityStats stats;
        stats.current_preset = current_preset_;
        stats.quality_score = calculate_quality_score();
        stats.quality_adjustments = quality_adjustments_;
        stats.performance_gain = performance_gain_;
        stats.emergency_mode = emergency_mode_;
        return stats;
    }
    
private:
    std::unordered_map<std::string, QualitySettings> quality_presets_;
    QualitySettings current_settings_;
    std::string current_preset_ = "High";
    
    PerformanceMetrics current_metrics_;
    std::deque<float> fps_history_;
    static constexpr size_t fps_history_size_ = 60; // 1 second at 60 FPS
    
    float target_fps_ = 60.0f;
    float fps_tolerance_ = 5.0f;
    
    int quality_adjustments_ = 0;
    float performance_gain_ = 0.0f;
    bool emergency_mode_ = false;
    
    std::chrono::high_resolution_clock::time_point last_adjustment_time_;
    static constexpr auto min_adjustment_interval_ = std::chrono::seconds(2);
    
    void initialize_quality_presets() {
        // Ultra quality preset
        quality_presets_["Ultra"] = {
            .render_scale = 1.5f,
            .msaa_samples = 8,
            .enable_bloom = true,
            .enable_ssao = true,
            .enable_motion_blur = true,
            .shadow_quality = 1.0f,
            .texture_quality = 1.0f,
            .max_lights = 16,
            .enable_vsync = true
        };
        
        // High quality preset
        quality_presets_["High"] = {
            .render_scale = 1.0f,
            .msaa_samples = 4,
            .enable_bloom = true,
            .enable_ssao = true,
            .enable_motion_blur = false,
            .shadow_quality = 1.0f,
            .texture_quality = 1.0f,
            .max_lights = 8,
            .enable_vsync = true
        };
        
        // Medium quality preset
        quality_presets_["Medium"] = {
            .render_scale = 0.8f,
            .msaa_samples = 2,
            .enable_bloom = true,
            .enable_ssao = false,
            .enable_motion_blur = false,
            .shadow_quality = 0.7f,
            .texture_quality = 0.8f,
            .max_lights = 4,
            .enable_vsync = true
        };
        
        // Low quality preset
        quality_presets_["Low"] = {
            .render_scale = 0.6f,
            .msaa_samples = 1,
            .enable_bloom = false,
            .enable_ssao = false,
            .enable_motion_blur = false,
            .shadow_quality = 0.3f,
            .texture_quality = 0.5f,
            .max_lights = 2,
            .enable_vsync = false
        };
        
        // Performance preset
        quality_presets_["Performance"] = {
            .render_scale = 0.5f,
            .msaa_samples = 1,
            .enable_bloom = false,
            .enable_ssao = false,
            .enable_motion_blur = false,
            .shadow_quality = 0.0f,
            .texture_quality = 0.3f,
            .max_lights = 1,
            .enable_vsync = false
        };
    }
    
    bool should_decrease_quality(float avg_fps) {
        if (emergency_mode_) return false; // Already at minimum
        
        auto now = std::chrono::high_resolution_clock::now();
        if (now - last_adjustment_time_ < min_adjustment_interval_) {
            return false; // Too soon since last adjustment
        }
        
        return avg_fps < target_fps_ - fps_tolerance_;
    }
    
    bool should_increase_quality(float avg_fps) {
        if (current_preset_ == "Ultra") return false; // Already at maximum
        
        auto now = std::chrono::high_resolution_clock::now();
        if (now - last_adjustment_time_ < min_adjustment_interval_ * 2) {
            return false; // Wait longer before increasing quality
        }
        
        return avg_fps > target_fps_ + fps_tolerance_ * 2;
    }
    
    void decrease_quality() {
        std::vector<std::string> quality_order = {"Ultra", "High", "Medium", "Low", "Performance"};
        
        auto it = std::find(quality_order.begin(), quality_order.end(), current_preset_);
        if (it != quality_order.end() && it + 1 != quality_order.end()) {
            set_quality_preset(*(it + 1));
            quality_adjustments_++;
            last_adjustment_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    void increase_quality() {
        std::vector<std::string> quality_order = {"Performance", "Low", "Medium", "High", "Ultra"};
        
        auto it = std::find(quality_order.begin(), quality_order.end(), current_preset_);
        if (it != quality_order.end() && it + 1 != quality_order.end()) {
            set_quality_preset(*(it + 1));
            quality_adjustments_++;
            last_adjustment_time_ = std::chrono::high_resolution_clock::now();
        }
    }
    
    void apply_emergency_quality_reduction() {
        if (!emergency_mode_) {
            emergency_mode_ = true;
            
            // Apply emergency settings
            current_settings_.render_scale = 0.4f;
            current_settings_.msaa_samples = 1;
            current_settings_.enable_bloom = false;
            current_settings_.enable_ssao = false;
            current_settings_.enable_motion_blur = false;
            current_settings_.shadow_quality = 0.0f;
            current_settings_.texture_quality = 0.2f;
            current_settings_.max_lights = 1;
            current_settings_.enable_vsync = false;
            
            current_preset_ = "Emergency";
        }
    }
    
    float calculate_quality_score() const {
        float score = 0.0f;
        
        score += current_settings_.render_scale * 0.3f;
        score += (current_settings_.msaa_samples / 8.0f) * 0.2f;
        score += (current_settings_.enable_bloom ? 0.1f : 0.0f);
        score += (current_settings_.enable_ssao ? 0.1f : 0.0f);
        score += current_settings_.shadow_quality * 0.15f;
        score += current_settings_.texture_quality * 0.1f;
        score += (current_settings_.max_lights / 16.0f) * 0.05f;
        
        return std::clamp(score, 0.0f, 1.0f);
    }
};

// ============================================================================
// GPU Command Buffer Optimizer
// ============================================================================

class CommandBufferOptimizer {
public:
    struct DrawCall {
        VkPipeline pipeline;
        VkDescriptorSet descriptor_set;
        VkBuffer vertex_buffer;
        VkBuffer index_buffer;
        uint32_t index_count;
        uint32_t instance_count = 1;
        uint32_t first_index = 0;
        int32_t vertex_offset = 0;
        uint32_t first_instance = 0;
        
        // Sorting key for state changes
        uint64_t sort_key = 0;
    };
    
    CommandBufferOptimizer() = default;
    
    void begin_frame() {
        draw_calls_.clear();
        state_changes_ = 0;
        batched_draws_ = 0;
    }
    
    void add_draw_call(const DrawCall& draw_call) {
        DrawCall optimized_call = draw_call;
        optimized_call.sort_key = generate_sort_key(draw_call);
        draw_calls_.push_back(optimized_call);
    }
    
    void optimize_and_submit(VkCommandBuffer cmd) {
        // Sort draw calls to minimize state changes
        std::sort(draw_calls_.begin(), draw_calls_.end(),
                 [](const DrawCall& a, const DrawCall& b) {
                     return a.sort_key < b.sort_key;
                 });
        
        // Submit optimized draw calls
        VkPipeline current_pipeline = VK_NULL_HANDLE;
        VkDescriptorSet current_descriptor_set = VK_NULL_HANDLE;
        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;
        
        for (const auto& draw_call : draw_calls_) {
            // Bind pipeline if changed
            if (draw_call.pipeline != current_pipeline) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_call.pipeline);
                current_pipeline = draw_call.pipeline;
                state_changes_++;
            }
            
            // Bind descriptor set if changed
            if (draw_call.descriptor_set != current_descriptor_set) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                       VK_NULL_HANDLE, 0, 1, &draw_call.descriptor_set, 0, nullptr);
                current_descriptor_set = draw_call.descriptor_set;
                state_changes_++;
            }
            
            // Bind vertex buffer if changed
            if (draw_call.vertex_buffer != current_vertex_buffer) {
                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &draw_call.vertex_buffer, &offset);
                current_vertex_buffer = draw_call.vertex_buffer;
                state_changes_++;
            }
            
            // Bind index buffer if changed
            if (draw_call.index_buffer != current_index_buffer) {
                vkCmdBindIndexBuffer(cmd, draw_call.index_buffer, 0, VK_INDEX_TYPE_UINT32);
                current_index_buffer = draw_call.index_buffer;
                state_changes_++;
            }
            
            // Submit draw call
            if (draw_call.instance_count > 1) {
                vkCmdDrawIndexed(cmd, draw_call.index_count, draw_call.instance_count,
                               draw_call.first_index, draw_call.vertex_offset, draw_call.first_instance);
            } else {
                vkCmdDrawIndexed(cmd, draw_call.index_count, 1,
                               draw_call.first_index, draw_call.vertex_offset, 0);
            }
        }
    }
    
    struct OptimizationStats {
        int total_draw_calls = 0;
        int state_changes = 0;
        int batched_draws = 0;
        float optimization_ratio = 0.0f;
    };
    
    OptimizationStats get_stats() const {
        OptimizationStats stats;
        stats.total_draw_calls = static_cast<int>(draw_calls_.size());
        stats.state_changes = state_changes_;
        stats.batched_draws = batched_draws_;
        
        if (stats.total_draw_calls > 0) {
            stats.optimization_ratio = 1.0f - (static_cast<float>(stats.state_changes) / stats.total_draw_calls);
        }
        
        return stats;
    }
    
private:
    std::vector<DrawCall> draw_calls_;
    int state_changes_ = 0;
    int batched_draws_ = 0;
    
    uint64_t generate_sort_key(const DrawCall& draw_call) {
        // Generate sort key to minimize state changes
        // Higher priority for pipeline changes, then descriptor sets, then buffers
        
        uint64_t key = 0;
        
        // Pipeline (highest priority)
        key |= (reinterpret_cast<uint64_t>(draw_call.pipeline) & 0xFFFF) << 48;
        
        // Descriptor set
        key |= (reinterpret_cast<uint64_t>(draw_call.descriptor_set) & 0xFFFF) << 32;
        
        // Vertex buffer
        key |= (reinterpret_cast<uint64_t>(draw_call.vertex_buffer) & 0xFFFF) << 16;
        
        // Index buffer
        key |= (reinterpret_cast<uint64_t>(draw_call.index_buffer) & 0xFFFF);
        
        return key;
    }
};

// ============================================================================
// Memory Pool Optimizer
// ============================================================================

class MemoryPoolOptimizer {
public:
    struct PoolStats {
        size_t total_size = 0;
        size_t used_size = 0;
        size_t free_size = 0;
        size_t largest_free_block = 0;
        float fragmentation_ratio = 0.0f;
        int allocation_count = 0;
        int deallocation_count = 0;
    };
    
    MemoryPoolOptimizer() = default;
    
    void optimize_pools(GPUMemoryManager& memory_manager) {
        // Analyze memory usage patterns
        auto stats = memory_manager.get_memory_stats();
        
        // Trigger garbage collection if fragmentation is high
        if (calculate_fragmentation_ratio() > 0.3f) {
            trigger_garbage_collection(memory_manager);
        }
        
        // Defragment memory pools if needed
        if (should_defragment()) {
            defragment_pools(memory_manager);
        }
        
        // Resize pools based on usage patterns
        resize_pools_if_needed(memory_manager);
    }
    
    void record_allocation(size_t size, const std::string& pool_name) {
        allocation_history_.push_back({size, pool_name, std::chrono::high_resolution_clock::now()});
        
        // Keep only recent history
        auto cutoff = std::chrono::high_resolution_clock::now() - std::chrono::minutes(5);
        allocation_history_.erase(
            std::remove_if(allocation_history_.begin(), allocation_history_.end(),
                          [cutoff](const AllocationRecord& record) {
                              return record.timestamp < cutoff;
                          }),
            allocation_history_.end());
    }
    
    PoolStats get_pool_stats(const std::string& pool_name) const {
        auto it = pool_stats_.find(pool_name);
        if (it != pool_stats_.end()) {
            return it->second;
        }
        return PoolStats{};
    }
    
    struct OptimizationReport {
        float memory_efficiency = 0.0f;
        float fragmentation_ratio = 0.0f;
        int garbage_collections = 0;
        int defragmentations = 0;
        size_t memory_saved = 0;
    };
    
    OptimizationReport get_optimization_report() const {
        OptimizationReport report;
        report.memory_efficiency = calculate_memory_efficiency();
        report.fragmentation_ratio = calculate_fragmentation_ratio();
        report.garbage_collections = garbage_collections_;
        report.defragmentations = defragmentations_;
        report.memory_saved = memory_saved_;
        return report;
    }
    
private:
    struct AllocationRecord {
        size_t size;
        std::string pool_name;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::vector<AllocationRecord> allocation_history_;
    std::unordered_map<std::string, PoolStats> pool_stats_;
    
    int garbage_collections_ = 0;
    int defragmentations_ = 0;
    size_t memory_saved_ = 0;
    
    float calculate_fragmentation_ratio() const {
        // Calculate overall fragmentation across all pools
        size_t total_free = 0;
        size_t largest_free = 0;
        
        for (const auto& pair : pool_stats_) {
            total_free += pair.second.free_size;
            largest_free = std::max(largest_free, pair.second.largest_free_block);
        }
        
        if (total_free == 0) return 0.0f;
        
        return 1.0f - (static_cast<float>(largest_free) / total_free);
    }
    
    float calculate_memory_efficiency() const {
        size_t total_size = 0;
        size_t used_size = 0;
        
        for (const auto& pair : pool_stats_) {
            total_size += pair.second.total_size;
            used_size += pair.second.used_size;
        }
        
        if (total_size == 0) return 0.0f;
        
        return static_cast<float>(used_size) / total_size;
    }
    
    bool should_defragment() const {
        return calculate_fragmentation_ratio() > 0.5f;
    }
    
    void trigger_garbage_collection(GPUMemoryManager& memory_manager) {
        // Implement garbage collection logic
        garbage_collections_++;
    }
    
    void defragment_pools(GPUMemoryManager& memory_manager) {
        // Implement memory defragmentation
        defragmentations_++;
    }
    
    void resize_pools_if_needed(GPUMemoryManager& memory_manager) {
        // Analyze allocation patterns and resize pools accordingly
        for (const auto& record : allocation_history_) {
            // Implementation would analyze patterns and suggest pool resizing
        }
    }
};

// ============================================================================
// Main Performance Optimizer
// ============================================================================

class PerformanceOptimizer {
public:
    PerformanceOptimizer() = default;
    
    void initialize(VulkanCore* vulkan_core) {
        vulkan_core_ = vulkan_core;
        
        // Initialize subsystems
        lod_manager_ = std::make_unique<LODManager>();
        frustum_culler_ = std::make_unique<FrustumCuller>();
        quality_scaler_ = std::make_unique<AdaptiveQualityScaler>();
        command_optimizer_ = std::make_unique<CommandBufferOptimizer>();
        memory_optimizer_ = std::make_unique<MemoryPoolOptimizer>();
        
        // Start performance monitoring thread
        start_monitoring_thread();
    }
    
    void shutdown() {
        stop_monitoring_thread();
    }
    
    void update(float delta_time) {
        // Update performance metrics
        update_performance_metrics();
        
        // Update LOD system
        lod_manager_->update(delta_time, current_metrics_.fps, camera_position_);
        
        // Update adaptive quality scaling
        AdaptiveQualityScaler::PerformanceMetrics metrics;
        metrics.current_fps = current_metrics_.fps;
        metrics.frame_time_ms = current_metrics_.frame_time_ms;
        metrics.gpu_usage_percent = current_metrics_.gpu_usage;
        metrics.vram_usage_percent = current_metrics_.vram_usage;
        metrics.cpu_usage_percent = current_metrics_.cpu_usage;
        
        quality_scaler_->update(metrics);
        
        // Optimize memory pools periodically
        if (should_optimize_memory()) {
            memory_optimizer_->optimize_pools(vulkan_core_->get_memory_manager());
        }
    }
    
    void begin_frame(const glm::mat4& view_projection_matrix, const glm::vec3& camera_position) {
        camera_position_ = camera_position;
        
        // Update frustum culling
        frustum_culler_->update_frustum(view_projection_matrix);
        frustum_culler_->begin_culling_frame();
        
        // Begin command buffer optimization
        command_optimizer_->begin_frame();
        
        frame_start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void end_frame() {
        frustum_culler_->end_culling_frame();
        
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end_time - frame_start_time_);
        
        current_metrics_.frame_time_ms = frame_duration.count() / 1000.0f;
        current_metrics_.fps = 1000.0f / current_metrics_.frame_time_ms;
        
        // Update frame time history
        frame_times_.push_back(current_metrics_.frame_time_ms);
        if (frame_times_.size() > 60) {
            frame_times_.pop_front();
        }
    }
    
    // Subsystem access
    LODManager& get_lod_manager() { return *lod_manager_; }
    FrustumCuller& get_frustum_culler() { return *frustum_culler_; }
    AdaptiveQualityScaler& get_quality_scaler() { return *quality_scaler_; }
    CommandBufferOptimizer& get_command_optimizer() { return *command_optimizer_; }
    MemoryPoolOptimizer& get_memory_optimizer() { return *memory_optimizer_; }
    
    struct OverallPerformanceStats {
        float current_fps = 60.0f;
        float average_fps = 60.0f;
        float frame_time_ms = 16.67f;
        float cpu_usage = 30.0f;
        float gpu_usage = 50.0f;
        float vram_usage = 50.0f;
        
        LODManager::LODStats lod_stats;
        FrustumCuller::CullingStats culling_stats;
        AdaptiveQualityScaler::QualityStats quality_stats;
        CommandBufferOptimizer::OptimizationStats command_stats;
        MemoryPoolOptimizer::OptimizationReport memory_report;
        
        float overall_optimization_score = 0.0f;
    };
    
    OverallPerformanceStats get_performance_stats() const {
        OverallPerformanceStats stats;
        
        stats.current_fps = current_metrics_.fps;
        stats.average_fps = calculate_average_fps();
        stats.frame_time_ms = current_metrics_.frame_time_ms;
        stats.cpu_usage = current_metrics_.cpu_usage;
        stats.gpu_usage = current_metrics_.gpu_usage;
        stats.vram_usage = current_metrics_.vram_usage;
        
        stats.lod_stats = lod_manager_->get_stats();
        stats.culling_stats = frustum_culler_->get_stats();
        stats.quality_stats = quality_scaler_->get_stats();
        stats.command_stats = command_optimizer_->get_stats();
        stats.memory_report = memory_optimizer_->get_optimization_report();
        
        stats.overall_optimization_score = calculate_optimization_score();
        
        return stats;
    }
    
private:
    VulkanCore* vulkan_core_ = nullptr;
    
    // Subsystems
    std::unique_ptr<LODManager> lod_manager_;
    std::unique_ptr<FrustumCuller> frustum_culler_;
    std::unique_ptr<AdaptiveQualityScaler> quality_scaler_;
    std::unique_ptr<CommandBufferOptimizer> command_optimizer_;
    std::unique_ptr<MemoryPoolOptimizer> memory_optimizer_;
    
    // Performance metrics
    struct PerformanceMetrics {
        float fps = 60.0f;
        float frame_time_ms = 16.67f;
        float cpu_usage = 30.0f;
        float gpu_usage = 50.0f;
        float vram_usage = 50.0f;
    } current_metrics_;
    
    std::deque<float> frame_times_;
    glm::vec3 camera_position_{0.0f};
    
    // Monitoring thread
    std::thread monitoring_thread_;
    std::atomic<bool> monitoring_active_{false};
    
    std::chrono::high_resolution_clock::time_point frame_start_time_;
    std::chrono::high_resolution_clock::time_point last_memory_optimization_;
    
    void start_monitoring_thread() {
        monitoring_active_ = true;
        monitoring_thread_ = std::thread([this]() {
            while (monitoring_active_) {
                monitor_system_performance();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    
    void stop_monitoring_thread() {
        monitoring_active_ = false;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }
    
    void monitor_system_performance() {
        // Monitor CPU usage
        current_metrics_.cpu_usage = get_cpu_usage();
        
        // Monitor GPU usage (platform-specific implementation needed)
        current_metrics_.gpu_usage = get_gpu_usage();
        
        // Monitor VRAM usage
        current_metrics_.vram_usage = get_vram_usage();
    }
    
    void update_performance_metrics() {
        // Update metrics from monitoring thread data
        // This is called from the main thread
    }
    
    float get_cpu_usage() {
        // Platform-specific CPU usage monitoring
        return 30.0f; // Placeholder
    }
    
    float get_gpu_usage() {
        // Platform-specific GPU usage monitoring
        return 50.0f; // Placeholder
    }
    
    float get_vram_usage() {
        // Get VRAM usage from Vulkan memory manager
        auto memory_stats = vulkan_core_->get_memory_manager().get_memory_stats();
        
        // Calculate total usage percentage
        float total_used = memory_stats.vertex_pool_used + 
                          memory_stats.uniform_pool_used + 
                          memory_stats.storage_pool_used;
        
        // Estimate total VRAM (this would need to be queried from Vulkan)
        float estimated_total_vram = 1024 * 1024 * 1024; // 1GB placeholder
        
        return (total_used / estimated_total_vram) * 100.0f;
    }
    
    float calculate_average_fps() const {
        if (frame_times_.empty()) return 60.0f;
        
        float total_time = 0.0f;
        for (float time : frame_times_) {
            total_time += time;
        }
        
        float average_frame_time = total_time / frame_times_.size();
        return 1000.0f / average_frame_time;
    }
    
    bool should_optimize_memory() {
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_last = now - last_memory_optimization_;
        
        // Optimize memory every 30 seconds
        if (time_since_last > std::chrono::seconds(30)) {
            last_memory_optimization_ = now;
            return true;
        }
        
        return false;
    }
    
    float calculate_optimization_score() const {
        float score = 0.0f;
        
        // FPS score (target 60 FPS)
        float fps_score = std::min(current_metrics_.fps / 60.0f, 1.0f);
        score += fps_score * 0.4f;
        
        // Resource usage score (lower is better)
        float resource_score = 1.0f - ((current_metrics_.cpu_usage + current_metrics_.gpu_usage) / 200.0f);
        score += std::max(resource_score, 0.0f) * 0.3f;
        
        // Quality score
        score += quality_scaler_->get_stats().quality_score * 0.2f;
        
        // Memory efficiency score
        score += memory_optimizer_->get_optimization_report().memory_efficiency * 0.1f;
        
        return std::clamp(score, 0.0f, 1.0f);
    }
};

} // namespace BTQuant