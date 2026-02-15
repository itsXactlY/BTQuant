#include "audio_engine_integration.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstring>  // For malloc/free

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AudioIntegration {

TradeAudioNotifier::TradeAudioNotifier()
    : pitch_shifter_(440.0, 1.0, 10000.0)  // Default pitch settings
    , audio_engine_(nullptr)
    , audio_initialized_(false)
{
}

TradeAudioNotifier::~TradeAudioNotifier() {
    shutdownAudio();
}

bool TradeAudioNotifier::initializeAudio() {
    // For the stub implementation, just return true
    audio_initialized_ = true;
    std::cout << "[Audio] Audio initialized (stub implementation)" << std::endl;
    return true;
}

void TradeAudioNotifier::connectToTradeTable(RawTradeTable& trade_table) {
    // Set up the callback to be called when new trades are added
    trade_table.set_trade_notification_callback(
        [this](const RawTrade& trade) {
            this->onTradeDetected(trade);
        }
    );
}

void TradeAudioNotifier::onTradeDetected(const RawTrade& trade) {
    std::cout << "[Audio] New trade detected - Volume: " << trade.volume 
              << ", Price: " << trade.price 
              << ", Side: " << trade.side << std::endl;
              
    // Determine if it's a buy or sell trade
    bool is_buy = (trade.side == 'B' || trade.side == 'b');
    
    // Play the trade sound based on the trade characteristics
    playTradeSound(trade.volume, is_buy);
}

void TradeAudioNotifier::playTradeSound(double volume, bool is_buy) {
    if (!audio_initialized_) {
        std::cout << "[Audio] Cannot play sound - audio not initialized" << std::endl;
        return;
    }

    // Calculate pitch based on volume (inverse relationship - large trades = low pitch)
    double pitch = pitch_shifter_.calculate_pitch(volume);

    // Determine amplitude based on volume
    double normalized_volume = std::min(1.0, std::max(0.1, volume / 10000.0)); // Normalize to 0.1-1.0 range
    float amplitude = static_cast<float>(normalized_volume * 0.5 + 0.1); // Range 0.1 to 0.6

    // Enhance bass effect for large trades
    bool is_large_trade = normalized_volume > 0.7; // Large trade threshold
    float bass_boost = is_large_trade ? 1.5f : 1.0f; // Boost amplitude for large trades

    // For the stub implementation, just print what would be played
    std::cout << "[Audio] Would play trade sound - Volume: " << volume
              << ", Pitch: " << pitch
              << ", Side: " << (is_buy ? "BUY" : "SELL")
              << ", Large Trade: " << (is_large_trade ? "YES" : "NO") << std::endl;
}

void TradeAudioNotifier::shutdownAudio() {
    if (audio_initialized_) {
        audio_initialized_ = false;
        std::cout << "[Audio] Audio engine shut down" << std::endl;
    }
}

} // namespace AudioIntegration