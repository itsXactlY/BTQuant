#pragma once

/**
 * @file fibonacci_retracement.hpp
 * @brief Fibonacci Retracement Drawing Tools with Dynamic Text Labels
 * 
 * This implementation provides:
 * - Fibonacci retracement level calculation
 * - Dynamic text label positioning
 * - Customizable level colors and styles
 * - Hit-testing for selection and modification
 * - Extension lines and price labels
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace btq {
namespace drawing {

/**
 * @brief Fibonacci level definition
 */
struct FibonacciLevel {
    double ratio;           // Fibonacci ratio (e.g., 0.236, 0.382, 0.5, 0.618, 0.786)
    std::string label;      // Display label (e.g., "23.6%", "38.2%")
    glm::vec4 color;        // Line color
    float line_width;       // Line width
    bool is_extended;       // Extend to right edge
    bool show_label;        // Show text label
};

/**
 * @brief Default Fibonacci levels
 */
inline std::vector<FibonacciLevel> getDefaultFibonacciLevels() {
    return {
        {0.000, "0%",    {0.0f, 1.0f, 0.0f, 1.0f}, 1.5f, true, true},   // 0% - Green
        {0.236, "23.6%", {0.5f, 0.5f, 0.5f, 0.8f}, 1.0f, true, true},   // 23.6%
        {0.382, "38.2%", {0.5f, 0.5f, 0.5f, 0.8f}, 1.0f, true, true},   // 38.2%
        {0.500, "50%",   {1.0f, 1.0f, 0.0f, 1.0f}, 1.5f, true, true},   // 50% - Yellow
        {0.618, "61.8%", {0.5f, 0.5f, 0.5f, 0.8f}, 1.0f, true, true},   // 61.8% - Golden ratio
        {0.786, "78.6%", {0.5f, 0.5f, 0.5f, 0.8f}, 1.0f, true, true},   // 78.6%
        {1.000, "100%",  {1.0f, 0.0f, 0.0f, 1.0f}, 1.5f, true, true}    // 100% - Red
    };
}

/**
 * @brief Fibonacci retracement drawing
 */
class FibonacciRetracement {
public:
    FibonacciRetracement() = default;
    
    /**
     * @brief Construct from two points
     */
    FibonacciRetracement(
        const glm::vec2& start_point,
        const glm::vec2& end_point,
        const std::vector<FibonacciLevel>& levels = getDefaultFibonacciLevels())
        : start_point_(start_point)
        , end_point_(end_point)
        , levels_(levels)
    {
        calculateLevels();
    }
    
    /**
     * @brief Calculate all level prices
     */
    void calculateLevels() {
        double high = std::max(start_point_.y, end_point_.y);
        double low = std::min(start_point_.y, end_point_.y);
        double range = high - low;
        
        // Determine if uptrend or downtrend
        is_uptrend_ = end_point_.y > start_point_.y;
        
        level_prices_.clear();
        level_prices_.reserve(levels_.size());
        
        for (const auto& level : levels_) {
            double price;
            if (is_uptrend_) {
                // Uptrend: levels go from high (0%) to low (100%)
                price = high - (range * level.ratio);
            } else {
                // Downtrend: levels go from low (0%) to high (100%)
                price = low + (range * level.ratio);
            }
            level_prices_.push_back(price);
        }
    }
    
    /**
     * @brief Get the price at a specific Fibonacci level
     */
    double getPriceAtLevel(size_t level_index) const {
        if (level_index < level_prices_.size()) {
            return level_prices_[level_index];
        }
        return 0.0;
    }
    
    /**
     * @brief Get all level prices
     */
    const std::vector<double>& getAllLevelPrices() const {
        return level_prices_;
    }
    
    /**
     * @brief Get the Fibonacci level closest to a price
     */
    size_t getNearestLevelIndex(double price) const {
        size_t nearest = 0;
        double min_dist = std::abs(price - level_prices_[0]);
        
        for (size_t i = 1; i < level_prices_.size(); ++i) {
            double dist = std::abs(price - level_prices_[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = i;
            }
        }
        
        return nearest;
    }
    
    /**
     * @brief Hit-test: check if a point is near any level line
     * @param point Point to test in chart coordinates
     * @param threshold Distance threshold in pixels
     * @param pixels_per_price Pixels per price unit
     * @return Level index if hit, -1 otherwise
     */
    int hitTest(const glm::vec2& point, float threshold, float pixels_per_price) const {
        // Check X range first
        double min_x = std::min(start_point_.x, end_point_.x);
        double max_x = std::max(start_point_.x, end_point_.x);
        
        if (point.x < min_x || point.x > max_x) {
            return -1;
        }
        
        // Check each level
        for (size_t i = 0; i < level_prices_.size(); ++i) {
            double price = level_prices_[i];
            double distance = std::abs(point.y - price) * pixels_per_price;
            
            if (distance <= threshold) {
                return static_cast<int>(i);
            }
        }
        
        return -1;
    }
    
    /**
     * @brief Hit-test for the start/end handles
     */
    int hitTestHandle(const glm::vec2& point, float threshold, float pixels_per_price, float pixels_per_time) const {
        // Check start point handle
        double dist_start = std::sqrt(
            std::pow((point.x - start_point_.x) * pixels_per_time, 2) +
            std::pow((point.y - start_point_.y) * pixels_per_price, 2)
        );
        
        if (dist_start <= threshold) {
            return 0;  // Start handle
        }
        
        // Check end point handle
        double dist_end = std::sqrt(
            std::pow((point.x - end_point_.x) * pixels_per_time, 2) +
            std::pow((point.y - end_point_.y) * pixels_per_price, 2)
        );
        
        if (dist_end <= threshold) {
            return 1;  // End handle
        }
        
        return -1;  // No hit
    }
    
    /**
     * @brief Move a handle to a new position
     */
    void moveHandle(int handle_index, const glm::vec2& new_position) {
        if (handle_index == 0) {
            start_point_ = new_position;
        } else if (handle_index == 1) {
            end_point_ = new_position;
        }
        calculateLevels();
    }
    
    /**
     * @brief Move the entire drawing
     */
    void move(const glm::vec2& delta) {
        start_point_ += delta;
        end_point_ += delta;
        calculateLevels();
    }
    
    /**
     * @brief Get start point
     */
    const glm::vec2& getStartPoint() const { return start_point_; }
    
    /**
     * @brief Get end point
     */
    const glm::vec2& getEndPoint() const { return end_point_; }
    
    /**
     * @brief Get levels configuration
     */
    const std::vector<FibonacciLevel>& getLevels() const { return levels_; }
    
    /**
     * @brief Set levels configuration
     */
    void setLevels(const std::vector<FibonacciLevel>& levels) {
        levels_ = levels;
        calculateLevels();
    }
    
    /**
     * @brief Check if uptrend
     */
    bool isUptrend() const { return is_uptrend_; }
    
    /**
     * @brief Get unique ID
     */
    uint64_t getId() const { return id_; }
    
    /**
     * @brief Set unique ID
     */
    void setId(uint64_t id) { id_ = id; }
    
    /**
     * @brief Check if visible
     */
    bool isVisible() const { return is_visible_; }
    
    /**
     * @brief Set visibility
     */
    void setVisible(bool visible) { is_visible_ = visible; }
    
    /**
     * @brief Check if locked
     */
    bool isLocked() const { return is_locked_; }
    
    /**
     * @brief Set locked state
     */
    void setLocked(bool locked) { is_locked_ = locked; }

private:
    glm::vec2 start_point_{0.0f, 0.0f};
    glm::vec2 end_point_{0.0f, 0.0f};
    std::vector<FibonacciLevel> levels_;
    std::vector<double> level_prices_;
    bool is_uptrend_ = true;
    bool is_visible_ = true;
    bool is_locked_ = false;
    uint64_t id_ = 0;
};

/**
 * @brief Fibonacci extension drawing
 */
class FibonacciExtension {
public:
    /**
     * @brief Construct from three points
     */
    FibonacciExtension(
        const glm::vec2& point1,
        const glm::vec2& point2,
        const glm::vec2& point3,
        const std::vector<FibonacciLevel>& levels = getDefaultFibonacciLevels())
        : point1_(point1)
        , point2_(point2)
        , point3_(point3)
        , levels_(levels)
    {
        calculateLevels();
    }
    
    void calculateLevels() {
        // Calculate the swing and projection
        double swing_high = std::max(point1_.y, point2_.y);
        double swing_low = std::min(point1_.y, point2_.y);
        double swing_range = swing_high - swing_low;
        
        level_prices_.clear();
        level_prices_.reserve(levels_.size());
        
        for (const auto& level : levels_) {
            double price;
            if (point3_.y < swing_low) {
                // Extension downward
                price = point3_.y - (swing_range * level.ratio);
            } else {
                // Extension upward
                price = point3_.y + (swing_range * level.ratio);
            }
            level_prices_.push_back(price);
        }
    }
    
    const std::vector<double>& getAllLevelPrices() const {
        return level_prices_;
    }

private:
    glm::vec2 point1_;
    glm::vec2 point2_;
    glm::vec2 point3_;
    std::vector<FibonacciLevel> levels_;
    std::vector<double> level_prices_;
};

/**
 * @brief Fibonacci time zones drawing
 */
class FibonacciTimeZones {
public:
    FibonacciTimeZones(
        const glm::vec2& start_point,
        int num_zones = 13)
        : start_point_(start_point)
        , num_zones_(num_zones)
    {
        calculateTimeZones();
    }
    
    void calculateTimeZones() {
        // Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233
        time_zones_.clear();
        time_zones_.reserve(num_zones_);
        
        int a = 1, b = 1;
        for (int i = 0; i < num_zones_; ++i) {
            time_zones_.push_back(a);
            int next = a + b;
            a = b;
            b = next;
        }
    }
    
    const std::vector<int>& getTimeZones() const {
        return time_zones_;
    }
    
    const glm::vec2& getStartPoint() const { return start_point_; }

private:
    glm::vec2 start_point_;
    int num_zones_;
    std::vector<int> time_zones_;
};

/**
 * @brief Fibonacci fan drawing
 */
class FibonacciFan {
public:
    FibonacciFan(
        const glm::vec2& start_point,
        const glm::vec2& end_point)
        : start_point_(start_point)
        , end_point_(end_point)
    {
        calculateFanLines();
    }
    
    void calculateFanLines() {
        double time_diff = end_point_.x - start_point_.x;
        double price_diff = end_point_.y - start_point_.y;
        
        // Fan angles based on Fibonacci ratios
        std::vector<double> ratios = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};
        
        fan_lines_.clear();
        fan_lines_.reserve(ratios.size());
        
        for (double ratio : ratios) {
            glm::vec2 end;
            end.x = end_point_.x;
            end.y = start_point_.y + price_diff * ratio;
            fan_lines_.push_back({start_point_, end, ratio});
        }
    }
    
    const std::vector<std::tuple<glm::vec2, glm::vec2, double>>& getFanLines() const {
        return fan_lines_;
    }

private:
    glm::vec2 start_point_;
    glm::vec2 end_point_;
    std::vector<std::tuple<glm::vec2, glm::vec2, double>> fan_lines_;
};

/**
 * @brief Fibonacci arc drawing
 */
class FibonacciArc {
public:
    FibonacciArc(
        const glm::vec2& center,
        const glm::vec2& end_point)
        : center_(center)
        , end_point_(end_point)
    {
        calculateArcs();
    }
    
    void calculateArcs() {
        double radius = std::sqrt(
            std::pow(end_point_.x - center_.x, 2) +
            std::pow(end_point_.y - center_.y, 2)
        );
        
        std::vector<double> ratios = {0.382, 0.5, 0.618, 1.0};
        
        arc_radii_.clear();
        arc_radii_.reserve(ratios.size());
        
        for (double ratio : ratios) {
            arc_radii_.push_back(radius * ratio);
        }
    }
    
    const std::vector<double>& getArcRadii() const {
        return arc_radii_;
    }
    
    const glm::vec2& getCenter() const { return center_; }

private:
    glm::vec2 center_;
    glm::vec2 end_point_;
    std::vector<double> arc_radii_;
};

/**
 * @brief Drawing manager for Fibonacci tools
 */
class FibonacciDrawingManager {
public:
    /**
     * @brief Add a new Fibonacci retracement
     */
    uint64_t addRetracement(
        const glm::vec2& start_point,
        const glm::vec2& end_point)
    {
        uint64_t id = next_id_++;
        FibonacciRetracement retracement(start_point, end_point);
        retracement.setId(id);
        retracements_.push_back(std::move(retracement));
        return id;
    }
    
    /**
     * @brief Remove a Fibonacci retracement
     */
    bool removeRetracement(uint64_t id) {
        auto it = std::find_if(retracements_.begin(), retracements_.end(),
            [id](const FibonacciRetracement& r) { return r.getId() == id; });
        
        if (it != retracements_.end()) {
            retracements_.erase(it);
            return true;
        }
        return false;
    }
    
    /**
     * @brief Get a Fibonacci retracement by ID
     */
    FibonacciRetracement* getRetracement(uint64_t id) {
        auto it = std::find_if(retracements_.begin(), retracements_.end(),
            [id](const FibonacciRetracement& r) { return r.getId() == id; });
        
        return it != retracements_.end() ? &(*it) : nullptr;
    }
    
    /**
     * @brief Hit-test all retracements
     */
    int64_t hitTestRetracement(
        const glm::vec2& point,
        float threshold,
        float pixels_per_price,
        float pixels_per_time) const
    {
        for (const auto& retracement : retracements_) {
            if (!retracement.isVisible()) continue;
            
            int handle_hit = retracement.hitTestHandle(point, threshold, pixels_per_price, pixels_per_time);
            if (handle_hit >= 0) {
                return static_cast<int64_t>(retracement.getId() * 10 + handle_hit);
            }
            
            int level_hit = retracement.hitTest(point, threshold, pixels_per_price);
            if (level_hit >= 0) {
                return static_cast<int64_t>(retracement.getId());
            }
        }
        
        return -1;
    }
    
    /**
     * @brief Get all retracements
     */
    const std::vector<FibonacciRetracement>& getAllRetracements() const {
        return retracements_;
    }
    
    /**
     * @brief Clear all drawings
     */
    void clear() {
        retracements_.clear();
    }

private:
    std::vector<FibonacciRetracement> retracements_;
    uint64_t next_id_ = 1;
};

} // namespace drawing
} // namespace btq
