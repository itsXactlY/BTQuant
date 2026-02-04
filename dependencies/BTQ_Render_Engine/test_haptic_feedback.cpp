#include "../include/ui/haptic_feedback.hpp"
#include <iostream>

int main() {
    auto& haptic = BTQuant::UI::HapticFeedback::getInstance();
    
    std::cout << "Testing Haptic Feedback System...\n";
    
    // Initialize the system
    haptic.initialize();
    
    // Test default intensity
    std::cout << "Default intensity: " << haptic.getIntensity() << std::endl;
    
    // Test setting intensity
    haptic.setIntensity(0.7f);
    std::cout << "Set intensity to 0.7, current: " << haptic.getIntensity() << std::endl;
    
    // Test different feedback types
    std::cout << "\nTesting different feedback types:\n";
    haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::LightClick);
    haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::MediumClick);
    haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::HeavyClick);
    
    // Test important/subtle interactions
    std::cout << "\nTesting important/subtle interactions:\n";
    haptic.triggerForImportantInteraction();
    haptic.triggerForSubtleInteraction();
    
    // Test with custom handler
    std::cout << "\nTesting custom handler:\n";
    haptic.registerHandler([](float intensity) {
        std::cout << "Custom handler called with intensity: " << intensity << std::endl;
    });
    haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::Success);
    
    return 0;
}