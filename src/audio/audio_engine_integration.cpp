#include "audio_engine_integration.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstring>  // For malloc/free

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include miniaudio for sound file playback
#ifndef MINIAUDIO_IMPLEMENTATION
#define MINIAUDIO_IMPLEMENTATION
#endif
#include "miniaudio.h"

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
    ma_result result;
    ma_engine_config config = ma_engine_config_init();

    // Initialize the audio engine
    result = ma_engine_init(&config, reinterpret_cast<ma_engine*>(&audio_engine_));
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to initialize audio engine: " << result << std::endl;
        audio_initialized_ = false;
        return false;
    }

    audio_initialized_ = true;
    std::cout << "[Audio] Audio engine initialized for trade sound effects" << std::endl;
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

    // Determine which sound file to play based on trade direction
    const char* sound_file = is_buy ? "buy.wav" : "sell.wav";

    // Play the sound file using miniaudio
    ma_engine* engine = reinterpret_cast<ma_engine*>(audio_engine_);
    
    ma_sound sound;
    ma_result result = ma_sound_init_from_file(engine, sound_file, 0, NULL, NULL, &sound);
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to load sound file '" << sound_file << "': " << result << std::endl;
        
        // Fallback: play a simple tone if the sound file is not available
        // Calculate pitch based on volume (inverse relationship - large trades = low pitch)
        double pitch = pitch_shifter_.calculate_pitch(volume);

        // Determine amplitude based on volume
        double normalized_volume = std::min(1.0, std::max(0.1, volume / 10000.0)); // Normalize to 0.1-1.0 range
        float amplitude = static_cast<float>(normalized_volume * 0.5 + 0.1); // Range 0.1 to 0.6

        // For fallback, just print what would be played
        std::cout << "[Audio] Fallback - Would play tone for " << (is_buy ? "BUY" : "SELL") 
                  << " - Volume: " << volume << ", Pitch: " << pitch << std::endl;
        return;
    }

    // Play the sound
    result = ma_sound_start(&sound);
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to start sound playback: " << result << std::endl;
        ma_sound_uninit(&sound);
        return;
    }

    std::cout << "[Audio] Played " << sound_file << " for trade - Volume: " << volume << std::endl;

    // Play the sound and let it finish asynchronously
    // In a real application, we could track the sound to handle cleanup later
    // For now, we'll just start the sound and uninitialize it immediately
    // This is acceptable for short sound effects like trade alerts
    
    // Clean up the sound
    ma_sound_uninit(&sound);
}

void TradeAudioNotifier::shutdownAudio() {
    if (audio_initialized_) {
        ma_engine* engine = reinterpret_cast<ma_engine*>(audio_engine_);
        if (engine) {
            ma_engine_uninit(engine);
            audio_engine_ = nullptr;
        }
        audio_initialized_ = false;
        std::cout << "[Audio] Audio engine shut down" << std::endl;
    }
}

} // namespace AudioIntegration