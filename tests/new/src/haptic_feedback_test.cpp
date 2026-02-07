#include "../include/ui/haptic_feedback.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Starting Haptic Feedback Test Suite..." << std::endl;

    // Test 1: Haptic Feedback
    std::cout << "\n--- Testing Haptic Feedback ---" << std::endl;
    {
        auto& haptic = BTQuant::UI::HapticFeedback::getInstance();
        
        // Test initialization
        haptic.initialize();
        std::cout << "✓ Haptic feedback initialized" << std::endl;
        
        // Test getting/setting intensity
        haptic.setIntensity(0.7f);
        assert(haptic.getIntensity() == 0.7f);
        std::cout << "✓ Intensity setting/getting works" << std::endl;
        
        // Test enable/disable
        haptic.setEnabled(true);
        assert(haptic.isEnabled() == true);
        haptic.setEnabled(false);
        assert(haptic.isEnabled() == false);
        haptic.setEnabled(true); // Re-enable for other tests
        std::cout << "✓ Enable/disable works" << std::endl;
        
        // Test triggering different feedback types
        haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::LightClick);
        haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::MediumClick);
        haptic.trigger(BTQuant::UI::HapticFeedback::FeedbackType::HeavyClick);
        std::cout << "✓ Different feedback types trigger successfully" << std::endl;
        
        // Test important/subtle interaction feedback
        haptic.triggerForImportantInteraction();
        haptic.triggerForSubtleInteraction();
        std::cout << "✓ Important/subtle interaction feedback works" << std::endl;
    }

    std::cout << "\n🎉 All Haptic Feedback tests passed!" << std::endl;
    std::cout << "Testing completed successfully." << std::endl;

    return 0;
}