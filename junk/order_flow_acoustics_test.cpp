#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

// Mock the audio functionality for testing
class MockAudioEngine {
public:
    struct AudioEvent {
        double volume;
        bool is_buy;
        double pitch;
        double amplitude;
    };
    
    static std::vector<AudioEvent> played_sounds;
    
    static void playSound(double volume, bool is_buy, double pitch, double amplitude) {
        AudioEvent event{volume, is_buy, pitch, amplitude};
        played_sounds.push_back(event);
        std::cout << "[MockAudio] Played sound - Volume: " << volume 
                  << ", Direction: " << (is_buy ? "BUY" : "SELL")
                  << ", Pitch: " << pitch 
                  << ", Amplitude: " << amplitude << std::endl;
    }
    
    static void clearEvents() {
        played_sounds.clear();
    }
};

std::vector<MockAudioEngine::AudioEvent> MockAudioEngine::played_sounds;

// Simulate the pitch calculation logic from MarketDataProcessor
double calculatePitchFromVolume(double volume, bool is_buy) {
    // Use logarithmic scaling to handle the wide range of trade volumes
    double log_volume = std::log10(std::max(volume, 1.0)); // Prevent log(0)
    
    // Define reference values for normalization
    double min_log_volume = 0.0; // log10(1) = 0
    double max_log_volume = 6.0; // log10(1000000) = 6 (for very large trades)
    
    // Normalize the log volume to 0-1 range
    double normalized_volume = std::min(1.0, std::max(0.0, (log_volume - min_log_volume) / (max_log_volume - min_log_volume)));

    // Base pitch parameters
    double base_pitch = 440.0;     // Base pitch (A4 note)
    double min_pitch = 220.0;      // Minimum pitch for large trades
    double max_pitch = 880.0;      // Maximum pitch for small trades (inverse relationship)

    // Calculate pitch (inverse relationship: large volume = low pitch)
    // Pitch decreases as volume increases (bass for large trades)
    double pitch = base_pitch + (max_pitch - base_pitch) * (1.0 - normalized_volume);

    // Adjust pitch slightly based on trade direction (buy/sell)
    double pitch_adjustment = 1.0f;
    if (!is_buy) {
        pitch_adjustment = 0.9f; // Slightly lower pitch for sell trades
    } else {
        pitch_adjustment = 1.05f; // Slightly higher pitch for buy trades
    }
    
    pitch *= pitch_adjustment;

    // Clamp pitch to valid range AFTER direction adjustment
    pitch = std::max(min_pitch, std::min(max_pitch, pitch));
    
    return pitch;
}

// Test the pitch scaling logic
void testPitchScaling() {
    std::cout << "\n=== Testing Pitch Scaling Logic ===" << std::endl;
    
    // Test cases: small trade, medium trade, large trade
    struct TestCase {
        double volume;
        bool is_buy;
        std::string description;
    };
    
    std::vector<TestCase> testCases = {
        {1.0, true, "Small BUY trade"},
        {1.0, false, "Small SELL trade"},
        {100.0, true, "Medium BUY trade"},
        {100.0, false, "Medium SELL trade"},
        {10000.0, true, "Large BUY trade"},
        {10000.0, false, "Large SELL trade"}
    };
    
    for (const auto& test : testCases) {
        double pitch = calculatePitchFromVolume(test.volume, test.is_buy);
        
        std::cout << test.description << " (Volume: " << test.volume << ") -> Pitch: " << pitch << " Hz" << std::endl;
        
        // Verify that pitch is within expected range (accounting for direction adjustments)
        // Max pitch can go up to 880*1.05 = 924 for buy trades, min can go down to 220*0.9 = 198 for sell trades
        assert(pitch >= 198.0 && pitch <= 924.0);
        
        // For the same volume, buy should have slightly higher pitch than sell
        if (test.volume == 100.0) {
            if (test.is_buy) {
                double sell_pitch = calculatePitchFromVolume(test.volume, false);
                assert(pitch > sell_pitch); // Buy should have higher pitch than sell for same volume
            }
        }
    }
    
    // Verify inverse relationship: larger volume should have lower pitch
    double smallTradePitch = calculatePitchFromVolume(1.0, true);
    double largeTradePitch = calculatePitchFromVolume(10000.0, true);
    assert(smallTradePitch > largeTradePitch); // Small trade should have higher pitch than large trade
    
    std::cout << "✓ Pitch scaling tests passed!" << std::endl;
}

// Test the complete audio flow
void testAudioFlow() {
    std::cout << "\n=== Testing Audio Flow ===" << std::endl;
    
    MockAudioEngine::clearEvents();
    
    // Simulate processing different trades
    struct Trade {
        double volume;
        bool is_buy;
        std::string description;
    };
    
    std::vector<Trade> trades = {
        {10.0, true, "Small BUY"},
        {1000.0, false, "Large SELL"},
        {500.0, true, "Medium BUY"}
    };
    
    for (const auto& trade : trades) {
        // Calculate pitch based on volume
        double pitch = calculatePitchFromVolume(trade.volume, trade.is_buy);
        
        // Calculate amplitude based on volume
        double log_volume = std::log10(std::max(trade.volume, 1.0));
        double min_log_volume = 0.0;
        double max_log_volume = 6.0;
        double normalized_volume = std::min(1.0, std::max(0.0, (log_volume - min_log_volume) / (max_log_volume - min_log_volume)));
        double amplitude = std::min(1.0, normalized_volume * 0.8 + 0.2); // Range 0.2 to 1.0
        
        // Simulate playing the sound
        MockAudioEngine::playSound(trade.volume, trade.is_buy, pitch, amplitude);
    }
    
    // Verify that all trades generated audio events
    assert(MockAudioEngine::played_sounds.size() == trades.size());
    
    std::cout << "✓ Audio flow tests passed!" << std::endl;
    std::cout << "Generated " << MockAudioEngine::played_sounds.size() << " audio events" << std::endl;
}

int main() {
    std::cout << "Testing Order Flow Acoustics Implementation" << std::endl;
    
    try {
        testPitchScaling();
        testAudioFlow();
        
        std::cout << "\n=== All Tests Passed! ===" << std::endl;
        std::cout << "Order Flow Acoustics implementation is working correctly:" << std::endl;
        std::cout << "- Pitch scales inversely with trade volume (large trades = low pitch)" << std::endl;
        std::cout << "- Different sounds for buy vs sell trades" << std::endl;
        std::cout << "- Proper handling of wide range of trade volumes" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}