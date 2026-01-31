#include "../include/ui/animation_system.hpp"
#include <algorithm>  // for std::remove_if
#include <cmath>

namespace BTQuant {
namespace UI {

// Helper function to clamp values
float clamp(float value, float min_val, float max_val) {
    return std::max(min_val, std::min(value, max_val));
}

int AnimationSystem::create_animation(float duration, EasingFunction easing) {
    Animation anim;
    anim.id = next_id_++;
    anim.duration = duration;
    anim.easing = easing;
    animations_.push_back(anim);
    return anim.id;
}

int AnimationSystem::animate_value(float* target, float from, float to, float duration, EasingFunction easing) {
    Animation anim;
    anim.id = next_id_++;
    anim.start_value = from;
    anim.end_value = to;
    anim.target_value = target;
    anim.duration = duration;
    anim.easing = easing;
    anim.start_time = 0.0f; // Will be set in update
    
    animations_.push_back(anim);
    return anim.id;
}

int AnimationSystem::animate_vector(glm::vec2* target, glm::vec2 from, glm::vec2 to, float duration, EasingFunction easing) {
    Animation anim;
    anim.id = next_id_++;
    anim.start_vec2 = from;
    anim.end_vec2 = to;
    anim.target_vec2 = target;
    anim.duration = duration;
    anim.easing = easing;
    anim.start_time = 0.0f; // Will be set in update
    
    animations_.push_back(anim);
    return anim.id;
}

int AnimationSystem::animate_callback(std::function<void(float)> on_update, float duration, std::function<void()> on_complete, EasingFunction easing) {
    Animation anim;
    anim.id = next_id_++;
    anim.on_update = on_update;
    anim.on_complete = on_complete;
    anim.duration = duration;
    anim.easing = easing;
    anim.start_time = 0.0f; // Will be set in update
    
    animations_.push_back(anim);
    return anim.id;
}

void AnimationSystem::update(float delta_time) {
    for (auto& anim : animations_) {
        if (!anim.is_playing || anim.is_completed) {
            continue;
        }
        
        // Initialize start time if this is the first update
        if (anim.start_time == 0.0f) {
            anim.start_time = delta_time; // Using delta_time as a reference point
        }
        
        update_single_animation(anim, delta_time);
    }
    
    // Remove completed animations that don't loop
    animations_.erase(
        std::remove_if(animations_.begin(), animations_.end(),
            [](const Animation& anim) {
                return anim.is_completed && !anim.loop;
            }),
        animations_.end()
    );
}

void AnimationSystem::stop_animation(int id) {
    for (auto& anim : animations_) {
        if (anim.id == id) {
            anim.is_playing = false;
            anim.is_completed = true;
            if (anim.on_complete) {
                anim.on_complete();
            }
            break;
        }
    }
}

void AnimationSystem::stop_all_animations() {
    for (auto& anim : animations_) {
        anim.is_playing = false;
        anim.is_completed = true;
        if (anim.on_complete) {
            anim.on_complete();
        }
    }
    animations_.clear();
}

void AnimationSystem::pause_animation(int id) {
    for (auto& anim : animations_) {
        if (anim.id == id) {
            anim.is_playing = false;
            break;
        }
    }
}

void AnimationSystem::resume_animation(int id) {
    for (auto& anim : animations_) {
        if (anim.id == id) {
            anim.is_playing = true;
            break;
        }
    }
}

float AnimationSystem::get_animation_progress(int id) const {
    for (const auto& anim : animations_) {
        if (anim.id == id) {
            if (anim.duration <= 0.0f) return 1.0f;
            float elapsed = 0.0f; // In a real implementation, this would be calculated properly
            return clamp(elapsed / anim.duration, 0.0f, 1.0f);
        }
    }
    return 0.0f;
}

bool AnimationSystem::is_animation_playing(int id) const {
    for (const auto& anim : animations_) {
        if (anim.id == id) {
            return anim.is_playing && !anim.is_completed;
        }
    }
    return false;
}

float AnimationSystem::apply_easing(EasingFunction func, float t) const {
    // Clamp t to [0, 1]
    t = clamp(t, 0.0f, 1.0f);
    
    switch (func) {
        case EasingFunction::Linear:
            return t;
            
        case EasingFunction::EaseInQuad:
            return t * t;
            
        case EasingFunction::EaseOutQuad:
            return t * (2.0f - t);
            
        case EasingFunction::EaseInOutQuad:
            return t < 0.5f ? 2.0f * t * t : -1.0f + (4.0f - 2.0f * t) * t;
            
        case EasingFunction::EaseInCubic:
            return t * t * t;
            
        case EasingFunction::EaseOutCubic:
            t -= 1.0f;
            return t * t * t + 1.0f;
            
        case EasingFunction::EaseInOutCubic:
            return t < 0.5f ? 4.0f * t * t * t : (t - 1.0f) * (2.0f * t - 2.0f) * (2.0f * t - 2.0f) + 1.0f;
            
        case EasingFunction::EaseInQuart:
            t *= t;
            return t * t;

        case EasingFunction::EaseOutQuart:
            {
                float temp = t - 1.0f;
                temp = temp * temp;
                return 1.0f - temp * temp;
            }

        case EasingFunction::EaseInOutQuart:
            if (t < 0.5f) {
                t *= t;
                return 8.0f * t * t;
            } else {
                float temp = t - 1.0f;
                temp = temp * temp;
                return 1.0f - 8.0f * temp * temp;
            }
            
        case EasingFunction::EaseInQuint:
            {
                float t2 = t * t;
                return t * t2 * t2;
            }

        case EasingFunction::EaseOutQuint:
            {
                float temp = t - 1.0f;
                float t2 = temp * temp;
                return 1.0f + temp * t2 * t2;
            }

        case EasingFunction::EaseInOutQuint:
            if (t < 0.5f) {
                float t2 = t * t;
                return 16.0f * t * t2 * t2;
            } else {
                float temp = t - 1.0f;
                float t2 = temp * temp;
                return 1.0f + 16.0f * temp * t2 * t2;
            }
            
        case EasingFunction::EaseInSine:
            return 1.0f - cos(t * M_PI * 0.5f);
            
        case EasingFunction::EaseOutSine:
            return sin(t * M_PI * 0.5f);
            
        case EasingFunction::EaseInOutSine:
            return 0.5f * (1.0f - cos(M_PI * t));
            
        case EasingFunction::EaseInExpo:
            return (t == 0.0f) ? 0.0f : pow(2.0f, 10.0f * (t - 1.0f));
            
        case EasingFunction::EaseOutExpo:
            return (t == 1.0f) ? 1.0f : 1.0f - pow(2.0f, -10.0f * t);
            
        case EasingFunction::EaseInOutExpo:
            if (t == 0.0f) return 0.0f;
            if (t == 1.0f) return 1.0f;
            if (t < 0.5f) return 0.5f * pow(2.0f, 20.0f * t - 10.0f);
            return 0.5f - 0.5f * pow(2.0f, -20.0f * t + 10.0f);
            
        case EasingFunction::EaseInCirc:
            return 1.0f - sqrt(1.0f - t * t);
            
        case EasingFunction::EaseOutCirc:
            return sqrt(t * (2.0f - t));
            
        case EasingFunction::EaseInOutCirc:
            if (t < 0.5f) return 0.5f * (1.0f - sqrt(1.0f - 4.0f * t * t));
            {
                float val = (2.0f * t - 3.0f) * (2.0f * t - 1.0f);
                return 0.5f * (sqrt(-val) + 1.0f);
            }
            
        default:
            return t; // Default to linear
    }
}

void AnimationSystem::update_single_animation(Animation& anim, float delta_time) {
    if (!anim.is_playing || anim.is_completed) {
        return;
    }

    // Update animation timer
    anim.start_time += delta_time;
    float elapsed = anim.start_time;

    float progress = clamp(elapsed / anim.duration, 0.0f, 1.0f);

    if (progress >= 1.0f) {
        progress = 1.0f;
        if (!anim.loop) {
            anim.is_completed = true;
        } else {
            // Reset for next loop
            anim.start_time = 0.0f;
            elapsed = 0.0f;
            progress = 0.0f;
        }
    }

    float eased_progress = apply_easing(anim.easing, progress);

    // Update value if target is set
    if (anim.target_value) {
        *anim.target_value = anim.start_value + (anim.end_value - anim.start_value) * eased_progress;
    }

    // Update vector if target is set
    if (anim.target_vec2) {
        *anim.target_vec2 = anim.start_vec2 + (anim.end_vec2 - anim.start_vec2) * eased_progress;
    }

    // Call update callback if set
    if (anim.on_update) {
        anim.on_update(eased_progress);
    }

    // Call completion callback if animation is finished and not looping
    if (progress >= 1.0f && !anim.loop && anim.on_complete) {
        anim.on_complete();
    }
}

// Animation presets implementation
int AnimationPresets::fade_in(void* target, float duration) {
    // This would typically animate opacity from 0 to 1
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::fade_out(void* target, float duration) {
    // This would typically animate opacity from 1 to 0
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::slide_in_left(void* target, float duration) {
    // This would typically animate position from left off-screen to original position
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::slide_in_right(void* target, float duration) {
    // This would typically animate position from right off-screen to original position
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::slide_in_up(void* target, float duration) {
    // This would typically animate position from below screen to original position
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::slide_in_down(void* target, float duration) {
    // This would typically animate position from above screen to original position
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::scale_in(void* target, float duration) {
    // This would typically animate scale from 0 to 1
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::scale_out(void* target, float duration) {
    // This would typically animate scale from 1 to 0
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::pulse(void* target, float duration) {
    // This would typically animate scale in a pulsing motion
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::shake(void* target, float intensity, float duration) {
    // This would typically animate position in a shaking motion
    // For now, returning a dummy ID
    return 0;
}

// Additional animation presets for trading dashboard
int AnimationPresets::price_change_highlight(float* target_alpha, float duration) {
    static auto& anim_system = AnimationSystem::getInstance();

    // Animate alpha from 0.0 to 1.0 and back to 0.0
    return anim_system.animate_value(target_alpha, 0.0f, 1.0f, duration * 0.5f, EasingFunction::EaseOutQuad);
}

int AnimationPresets::panel_slide_in(void* panel, float offset_x, float offset_y, float duration) {
    // This would animate a panel sliding in from an offset position
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::data_refresh_pulse(void* element, float duration) {
    // This would animate a subtle pulse to indicate data refresh
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::notification_slide_in(void* notification, float duration) {
    // This would animate a notification sliding in
    // For now, returning a dummy ID
    return 0;
}

int AnimationPresets::symbol_switch_transition(void* chart_element, float duration) {
    // This would animate a smooth transition when switching symbols
    // For now, returning a dummy ID
    return 0;
}

} // namespace UI
} // namespace BTQuant