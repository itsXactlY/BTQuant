#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

// Define a simple 2D vector type if GLM is not available
#ifndef USE_GLM
struct SimpleVec2 {
    float x = 0.0f;
    float y = 0.0f;

    SimpleVec2(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}

    SimpleVec2 operator+(const SimpleVec2& other) const {
        return SimpleVec2(x + other.x, y + other.y);
    }

    SimpleVec2 operator-(const SimpleVec2& other) const {
        return SimpleVec2(x - other.x, y - other.y);
    }

    SimpleVec2 operator*(float scalar) const {
        return SimpleVec2(x * scalar, y * scalar);
    }
};

// Use our simple vector instead of GLM
using glm = struct {
    using vec2 = SimpleVec2;
};
#else
#include <glm/glm.hpp>
#endif

namespace BTQuant {
namespace UI {

// ============================================================================
// Animation System
// ============================================================================

enum class EasingFunction {
  Linear,
  EaseInQuad,
  EaseOutQuad,
  EaseInOutQuad,
  EaseInCubic,
  EaseOutCubic,
  EaseInOutCubic,
  EaseInQuart,
  EaseOutQuart,
  EaseInOutQuart,
  EaseInQuint,
  EaseOutQuint,
  EaseInOutQuint,
  EaseInSine,
  EaseOutSine,
  EaseInOutSine,
  EaseInExpo,
  EaseOutExpo,
  EaseInOutExpo,
  EaseInCirc,
  EaseOutCirc,
  EaseInOutCirc,
  Spring,
  Elastic,
  Bounce
};

class AnimationSystem {
public:
  static AnimationSystem& getInstance() {
    static AnimationSystem instance;
    return instance;
  }

  AnimationSystem(const AnimationSystem&) = delete;
  AnimationSystem& operator=(const AnimationSystem&) = delete;

  struct Animation {
    float start_time = 0.0f;
    float duration = 0.0f;
    EasingFunction easing = EasingFunction::EaseOutQuad;

    // Value-based animation
    float start_value = 0.0f;
    float end_value = 0.0f;
    float* target_value = nullptr;

    // Vector-based animation
    glm::vec2 start_vec2 = glm::vec2(0.0f);
    glm::vec2 end_vec2 = glm::vec2(0.0f);
    glm::vec2* target_vec2 = nullptr;

    // Callback-based animation
    std::function<void(float)> on_update;
    std::function<void()> on_complete;

    bool is_playing = true;
    bool is_completed = false;
    bool loop = false;

    // Animation ID for identification
    int id = 0;
  };

  // Create a new animation and return its ID
  int create_animation(float duration, EasingFunction easing = EasingFunction::EaseOutQuad);

  // Animate a float value
  int animate_value(float* target, float from, float to, float duration,
                    EasingFunction easing = EasingFunction::EaseOutQuad);

  // Animate a vec2 value
  int animate_vector(glm::vec2* target, glm::vec2 from, glm::vec2 to, float duration,
                     EasingFunction easing = EasingFunction::EaseOutQuad);

  // Animate with callback
  int animate_callback(std::function<void(float)> on_update, float duration,
                       std::function<void()> on_complete = nullptr,
                       EasingFunction easing = EasingFunction::EaseOutQuad);

  // Update all active animations
  void update(float delta_time);

  // Stop a specific animation
  void stop_animation(int id);

  // Stop all animations
  void stop_all_animations();

  // Pause/resume animations
  void pause_animation(int id);
  void resume_animation(int id);

  // Get animation progress (0.0 to 1.0)
  float get_animation_progress(int id) const;

  // Check if animation is playing
  bool is_animation_playing(int id) const;

private:
  AnimationSystem() = default;  // Private constructor for singleton
  ~AnimationSystem() = default; // Private destructor for singleton

  std::vector<Animation> animations_;
  int next_id_ = 1;

  float apply_easing(EasingFunction func, float t) const;
  void update_single_animation(Animation& anim, float delta_time);
};

// Predefined animation presets
class AnimationPresets {
public:
  // Fade in/out animations
  static int fade_in(void* target, float duration = 0.3f);
  static int fade_out(void* target, float duration = 0.3f);

  // Slide animations
  static int slide_in_left(void* target, float duration = 0.4f);
  static int slide_in_right(void* target, float duration = 0.4f);
  static int slide_in_up(void* target, float duration = 0.4f);
  static int slide_in_down(void* target, float duration = 0.4f);

  // Scale animations
  static int scale_in(void* target, float duration = 0.3f);
  static int scale_out(void* target, float duration = 0.3f);

  // Pulse animation
  static int pulse(void* target, float duration = 0.5f);

  // Shake animation
  static int shake(void* target, float intensity = 10.0f, float duration = 0.5f);

  // Trading-specific animations
  static int price_change_highlight(float* target_alpha, float duration = 0.2f);
  static int panel_slide_in(void* panel, float offset_x, float offset_y, float duration = 0.3f);
  static int data_refresh_pulse(void* element, float duration = 0.5f);
  static int notification_slide_in(void* notification, float duration = 0.3f);
  static int symbol_switch_transition(void* chart_element, float duration = 0.4f);
};

} // namespace UI
} // namespace BTQuant