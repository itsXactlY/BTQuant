#include "../include/ui/haptic_feedback.hpp"

#include <chrono>
#include <iostream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <unistd.h>
// On Linux, we might use libudev or other system-specific haptic APIs
// For now, we'll simulate haptic feedback
#elif __APPLE__
#include <CoreHaptics/CoreHaptics.h>
// On macOS, we'll use CoreHaptics framework
#endif

namespace BTQuant {
namespace UI {

void HapticFeedback::initialize() {
  // Platform-specific initialization
#ifdef _WIN32
  // Windows haptic initialization would go here
  // For now, just log that initialization happened
  std::cout << "Windows haptic feedback initialized\n";
#elif __linux__
  // Linux haptic initialization would go here
  // For now, just log that initialization happened
  std::cout << "Linux haptic feedback initialized\n";
#elif __APPLE__
  // macOS haptic initialization would go here
  std::cout << "macOS haptic feedback initialized\n";
#endif
}

void HapticFeedback::trigger(FeedbackType type) {
  if (!enabled_) {
    return;
  }

  float intensity = getIntensityForType(type);
  triggerWithIntensity(intensity);
}

void HapticFeedback::triggerWithIntensity(float intensity) {
  if (!enabled_) {
    return;
  }

  // Apply global intensity multiplier
  float final_intensity = intensity * intensity_;

  // Call custom handler if registered
  if (custom_handler_) {
    custom_handler_(final_intensity);
    return;
  }

  // Platform-specific haptic feedback implementation
#ifdef _WIN32
  // On Windows, we could use XInput or other haptic APIs
  // For now, just log (non-blocking)
  std::cout << "Windows haptic feedback triggered with intensity: " << final_intensity << "\n";

#elif __linux__
  // On Linux, we could use libudev or other haptic APIs
  // For now, just log (non-blocking)
  std::cout << "Linux haptic feedback simulated with intensity: " << final_intensity << "\n";

  // Optionally, we could use system beep or other audio cues
  // This is a placeholder for actual haptic hardware integration

#elif __APPLE__
  // On macOS, we could use CoreHaptics
  // For now, just log (non-blocking)
  std::cout << "macOS haptic feedback simulated with intensity: " << final_intensity << "\n";

  // Placeholder for actual CoreHaptics implementation
  // CHHapticEngine would be used in a real implementation
#endif
}

void HapticFeedback::setIntensity(float intensity) {
  // Clamp intensity between 0.0 and 1.0
  intensity_ = std::max(0.0f, std::min(1.0f, intensity));
}

float HapticFeedback::getIntensity() const { return intensity_; }

void HapticFeedback::setEnabled(bool enabled) { enabled_ = enabled; }

bool HapticFeedback::isEnabled() const { return enabled_; }

void HapticFeedback::registerHandler(std::function<void(float)> handler) {
  custom_handler_ = handler;
}

void HapticFeedback::triggerForImportantInteraction() {
  if (!enabled_) {
    return;
  }

  // Use heavy click feedback for important interactions
  trigger(FeedbackType::HeavyClick);
}

void HapticFeedback::triggerForSubtleInteraction() {
  if (!enabled_) {
    return;
  }

  // Use light click feedback for subtle interactions
  trigger(FeedbackType::LightClick);
}

float HapticFeedback::getIntensityForType(FeedbackType type) const {
  switch (type) {
    case FeedbackType::LightClick:
      return 0.2f;  // Very subtle
    case FeedbackType::MediumClick:
      return 0.5f;  // Standard
    case FeedbackType::HeavyClick:
      return 0.8f;  // Strong
    case FeedbackType::Selection:
      return 0.3f;  // Light feedback for selection
    case FeedbackType::Warning:
      return 0.6f;  // Medium-strong for warnings
    case FeedbackType::Success:
      return 0.4f;  // Light-medium for success
    case FeedbackType::Error:
      return 0.7f;  // Strong for errors
    case FeedbackType::Notification:
      return 0.3f;  // Light for notifications
    case FeedbackType::DragStart:
      return 0.2f;  // Very light for drag start
    case FeedbackType::DragEnd:
      return 0.2f;  // Very light for drag end
    case FeedbackType::ScrollBump:
      return 0.3f;  // Light bump for scroll boundaries
    default:
      return 0.5f;  // Default to medium
  }
}

}  // namespace UI
}  // namespace BTQuant