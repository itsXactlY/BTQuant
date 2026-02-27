#pragma once

#include <functional>
#include <string>

namespace BTQuant {
namespace UI {

// ============================================================================
// Haptic Feedback System
// ============================================================================

class HapticFeedback {
 public:
  static HapticFeedback& getInstance() {
    static HapticFeedback instance;
    return instance;
  }

  HapticFeedback(const HapticFeedback&) = delete;
  HapticFeedback& operator=(const HapticFeedback&) = delete;

  // Haptic feedback types for different interaction contexts
  enum class FeedbackType {
    LightClick,      // Subtle feedback for minor interactions
    MediumClick,     // Standard feedback for button presses
    HeavyClick,      // Strong feedback for important actions
    Selection,       // Feedback for selecting items
    Warning,         // Feedback for warnings or alerts
    Success,         // Feedback for successful operations
    Error,           // Feedback for errors or failed operations
    Notification,    // Feedback for notifications
    DragStart,       // Feedback when starting drag operations
    DragEnd,         // Feedback when ending drag operations
    ScrollBump       // Feedback when scrolling hits boundaries
  };

  // Initialize the haptic feedback system
  void initialize();

  // Trigger haptic feedback based on type
  void trigger(FeedbackType type);

  // Trigger haptic feedback with custom intensity
  void triggerWithIntensity(float intensity);

  // Set global intensity multiplier (0.0 to 1.0)
  void setIntensity(float intensity);

  // Get current intensity
  float getIntensity() const;

  // Enable/disable haptic feedback globally
  void setEnabled(bool enabled);

  // Check if haptic feedback is enabled
  bool isEnabled() const;

  // Register a custom haptic feedback handler
  void registerHandler(std::function<void(float)> handler);

  // Perform haptic feedback for important interactions
  void triggerForImportantInteraction();

  // Perform haptic feedback for subtle interactions
  void triggerForSubtleInteraction();

 private:
  HapticFeedback() = default;   // Private constructor for singleton
  ~HapticFeedback() = default;  // Private destructor for singleton

  float intensity_ = 0.5f;  // Default medium intensity
  bool enabled_ = true;     // Enabled by default
  std::function<void(float)> custom_handler_ = nullptr;

  // Map feedback types to intensities
  float getIntensityForType(FeedbackType type) const;
};

}  // namespace UI
}  // namespace BTQuant