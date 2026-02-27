#pragma once

#include "rendering/auto_quality.hpp"
#include <chrono>

namespace RenderEngine {

/**
 * @brief Integration helper for auto-quality system with rendering pipeline
 * 
 * This class provides utilities to integrate the auto-quality system with
 * the main rendering loop to automatically adjust quality settings based on
 * performance metrics.
 */
class AutoQualityRenderer {
public:
    explicit AutoQualityRenderer(const AutoQualityConfig& config);
    ~AutoQualityRenderer() = default;

    /**
     * @brief Called at the beginning of each frame to start timing
     */
    void beginFrame();

    /**
     * @brief Called at the end of each frame to record frame time and adjust quality
     */
    void endFrame();

    /**
     * @brief Get the current rendering settings based on auto-quality adjustments
     */
    const QualitySettings& getQualitySettings() const;

    /**
     * @brief Get the current auto-quality controller for direct access
     */
    AutoQualityController& getController();

    /**
     * @brief Get the current auto-quality controller for direct access (const)
     */
    const AutoQualityController& getController() const;

private:
    AutoQualityController controller_;
    std::chrono::high_resolution_clock::time_point frame_start_time_;
    QualitySettings current_settings_;
};

inline AutoQualityRenderer::AutoQualityRenderer(const AutoQualityConfig& config)
    : controller_(config)
    , frame_start_time_(std::chrono::high_resolution_clock::now())
    , current_settings_(controller_.getCurrentQualitySettings())
{
}

inline void AutoQualityRenderer::beginFrame() {
    frame_start_time_ = std::chrono::high_resolution_clock::now();
}

inline void AutoQualityRenderer::endFrame() {
    auto frame_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        frame_end_time - frame_start_time_);
    double frame_time_ms = static_cast<double>(duration.count()) / 1000.0;

    // Record frame time for auto-quality analysis
    controller_.recordFrameTime(frame_time_ms);

    // Update current settings if quality level changed
    current_settings_ = controller_.getRecommendedRenderingSettings();
}

inline const QualitySettings& AutoQualityRenderer::getQualitySettings() const {
    return current_settings_;
}

inline AutoQualityController& AutoQualityRenderer::getController() {
    return controller_;
}

inline const AutoQualityController& AutoQualityRenderer::getController() const {
    return controller_;
}

} // namespace RenderEngine