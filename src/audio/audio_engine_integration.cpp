#include "audio_engine_integration.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace AudioIntegration {

TradeAudioNotifier::TradeAudioNotifier() 
    : pitch_shifter_(440.0, 1.0, 10000.0)  // Default pitch settings
#ifdef MINIAUDIO_IMPLEMENTATION
    , audio_engine_(nullptr)
    , audio_initialized_(false)
#else
    , audio_engine_(nullptr)
    , audio_initialized_(false)
#endif
{
}

TradeAudioNotifier::~TradeAudioNotifier() {
    shutdownAudio();
}

bool TradeAudioNotifier::initializeAudio() {
#ifdef MINIAUDIO_IMPLEMENTATION
    // Initialize the miniaudio engine
    audio_engine_ = malloc(sizeof(ma_engine));
    if (!audio_engine_) {
        std::cerr << "[Audio] Failed to allocate memory for audio engine" << std::endl;
        return false;
    }
    
    ma_engine* engine = static_cast<ma_engine*>(audio_engine_);
    ma_engine_config config = ma_engine_config_init();
    ma_result result = ma_engine_init(&config, engine);
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to initialize audio engine: " << result << std::endl;
        free(audio_engine_);
        audio_engine_ = nullptr;
        audio_initialized_ = false;
        return false;
    }
    
    audio_initialized_ = true;
    std::cout << "[Audio] Audio engine initialized successfully" << std::endl;
    return true;
#else
    // For the stub implementation, just return true
    audio_initialized_ = true;
    std::cout << "[Audio] Audio initialized (stub implementation)" << std::endl;
    return true;
#endif
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

#ifdef MINIAUDIO_IMPLEMENTATION
    ma_engine* engine = static_cast<ma_engine*>(audio_engine_);

    // For ma_engine_play_sound, we need to play a pre-existing audio file
    // In a real implementation, we would have pre-made sound files that we can play
    // with different parameters based on trade characteristics
    
    // For this implementation, we'll create a simple approach to trigger ma_engine_play_sound
    // In a real application, you would have pre-generated sound files or use a different approach
    // to play dynamic sounds based on trade characteristics
    
    // For now, let's use a placeholder sound file that would represent the trade sound
    // In a real implementation, you would have different sound files for different trade types
    const char* sound_file = "assets/trade_beep.wav";  // Placeholder for actual sound file
    
    // Play the sound using ma_engine_play_sound
    ma_result result = ma_engine_play_sound(engine, sound_file, NULL);
    if (result != MA_SUCCESS) {
        // Note: This is expected if the sound file doesn't exist
        // In a real implementation, you would have actual sound assets
        std::cout << "[Audio] Sound file not found or error playing sound - would play trade sound with ma_engine_play_sound - Volume: " << volume
                  << ", Pitch: " << pitch
                  << ", Side: " << (is_buy ? "BUY" : "SELL")
                  << ", Large Trade: " << (is_large_trade ? "YES" : "NO") << std::endl;
    } else {
        std::cout << "[Audio] Played trade sound with ma_engine_play_sound - Volume: " << volume
                  << ", Pitch: " << pitch
                  << ", Side: " << (is_buy ? "BUY" : "SELL")
                  << ", Large Trade: " << (is_large_trade ? "YES" : "NO") << std::endl;
    }

#else
    // For the stub implementation, just print what would be played
    std::cout << "[Audio] Would play trade sound - Volume: " << volume
              << ", Pitch: " << pitch
              << ", Side: " << (is_buy ? "BUY" : "SELL")
              << ", Large Trade: " << (is_large_trade ? "YES" : "NO") << std::endl;
#endif
}

void TradeAudioNotifier::shutdownAudio() {
#ifdef MINIAUDIO_IMPLEMENTATION
    if (audio_initialized_ && audio_engine_) {
        ma_engine* engine = static_cast<ma_engine*>(audio_engine_);
        ma_engine_uninit(engine);
        free(audio_engine_);
        audio_engine_ = nullptr;
        audio_initialized_ = false;
        std::cout << "[Audio] Audio engine shut down" << std::endl;
    }
#endif
}

} // namespace AudioIntegration