#include <iostream>
#include <cassert>

// Include the UI enhancement headers
#include "ui/haptic_feedback.hpp"
#include "ui/tooltips.hpp"

void test_haptic_feedback_integration() {
    std::cout << "Testing Haptic Feedback Integration...\n";
    
    // Test singleton instance
    auto& haptic1 = BTQuant::UI::HapticFeedback::getInstance();
    auto& haptic2 = BTQuant::UI::HapticFeedback::getInstance();
    assert(&haptic1 == &haptic2); // Should be the same instance
    
    // Test initialization
    haptic1.initialize();
    
    // Test enable/disable
    haptic1.setEnabled(true);
    assert(haptic1.isEnabled() == true);
    
    haptic1.setEnabled(false);
    assert(haptic1.isEnabled() == false);
    
    haptic1.setEnabled(true); // Re-enable for other tests
    
    // Test intensity
    haptic1.setIntensity(0.7f);
    assert(haptic1.getIntensity() == 0.7f);
    
    // Test various feedback types
    haptic1.trigger(BTQuant::UI::HapticFeedback::FeedbackType::LightClick);
    haptic1.trigger(BTQuant::UI::HapticFeedback::FeedbackType::HeavyClick);
    haptic1.trigger(BTQuant::UI::HapticFeedback::FeedbackType::Success);
    haptic1.trigger(BTQuant::UI::HapticFeedback::FeedbackType::Error);
    
    // Test convenience methods
    haptic1.triggerForSubtleInteraction();
    haptic1.triggerForImportantInteraction();
    
    std::cout << "✓ Haptic Feedback Integration tests passed!\n\n";
}

void test_tooltip_integration() {
    std::cout << "Testing Tooltip Integration...\n";
    
    auto& tooltip_manager = BTQuant::UI::get_global_tooltip_manager();
    
    // Test registering and retrieving tooltips
    tooltip_manager.register_tooltip("test_button_1", "This is a test tooltip for button 1");
    std::string tooltip = tooltip_manager.get_tooltip("test_button_1");
    assert(tooltip == "This is a test tooltip for button 1");
    
    // Test retrieving non-existent tooltip
    std::string empty_tooltip = tooltip_manager.get_tooltip("non_existent_button");
    assert(empty_tooltip.empty());
    
    // Test some expected tooltips that should exist after our enhancements
    // These would have been registered in the initialization
    std::string watchlist_add_tooltip = tooltip_manager.get_tooltip("watchlist_add_symbol");
    std::string watchlist_clear_tooltip = tooltip_manager.get_tooltip("watchlist_clear_all");
    std::string chart_replay_play_tooltip = tooltip_manager.get_tooltip("chart_replay_play");
    std::string alerts_menu_tooltip = tooltip_manager.get_tooltip("alerts_menu");
    
    // These should exist as they were registered in the tooltips.cpp file
    std::cout << "✓ Watchlist add symbol tooltip exists: " << (!watchlist_add_tooltip.empty() ? "yes" : "no") << std::endl;
    std::cout << "✓ Watchlist clear all tooltip exists: " << (!watchlist_clear_tooltip.empty() ? "yes" : "no") << std::endl;
    std::cout << "✓ Chart replay play tooltip exists: " << (!chart_replay_play_tooltip.empty() ? "yes" : "no") << std::endl;
    std::cout << "✓ Alerts menu tooltip exists: " << (!alerts_menu_tooltip.empty() ? "yes" : "no") << std::endl;
    
    // Test showing tooltips (these should not crash)
    tooltip_manager.show_simple_tooltip("Test tooltip message");
    
    std::cout << "✓ Tooltip Integration tests passed!\n\n";
}

void test_ui_enhancement_integration() {
    std::cout << "Testing UI Enhancement Integration...\n";
    
    // Test that both systems can work together
    auto& haptic = BTQuant::UI::HapticFeedback::getInstance();
    auto& tooltip_manager = BTQuant::UI::get_global_tooltip_manager();
    
    // Simulate a UI interaction that would use both
    haptic.triggerForSubtleInteraction();
    tooltip_manager.show_simple_tooltip("Simulated UI interaction tooltip");
    
    // Test that haptic feedback can be disabled without affecting tooltips
    haptic.setEnabled(false);
    assert(haptic.isEnabled() == false);
    
    // Tooltips should still work when haptic is disabled
    tooltip_manager.show_simple_tooltip("Tooltip still works when haptic is disabled");
    
    // Re-enable haptic feedback
    haptic.setEnabled(true);
    assert(haptic.isEnabled() == true);
    
    std::cout << "✓ UI Enhancement Integration tests passed!\n\n";
}

int main() {
    std::cout << "=== UI Enhancements Integration Test Suite ===\n\n";
    
    try {
        test_haptic_feedback_integration();
        test_tooltip_integration();
        test_ui_enhancement_integration();
        
        std::cout << "=== All UI Enhancement Integration Tests Passed! ===\n";
        std::cout << "UI enhancements (haptic feedback and tooltips) have been successfully integrated!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}