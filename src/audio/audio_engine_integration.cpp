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

    // Duration based on trade size (larger trades might have longer sounds)
    float duration = 0.05f + static_cast<float>(normalized_volume) * 0.1f; // 50-150ms

    // Enhance bass effect for large trades
    bool is_large_trade = normalized_volume > 0.7; // Large trade threshold
    float bass_boost = is_large_trade ? 1.5f : 1.0f; // Boost amplitude for large trades

#ifdef MINIAUDIO_IMPLEMENTATION
    ma_engine* engine = static_cast<ma_engine*>(audio_engine_);

    // Generate and play a tone using miniaudio
    ma_uint32 sampleRate = 44100;
    ma_uint32 channels = 1; // Mono
    ma_uint32 totalFrames = static_cast<ma_uint32>(sampleRate * duration);

    if (totalFrames == 0) {
        return; // Nothing to play
    }

    // Allocate memory for the audio data
    float* pAudioData = static_cast<float*>(malloc(totalFrames * channels * sizeof(float)));
    if (pAudioData == nullptr) {
        std::cerr << "[Audio] Failed to allocate memory for audio data" << std::endl;
        return;
    }

    // Generate a richer waveform with harmonics for bass effect
    for (ma_uint32 i = 0; i < totalFrames; i++) {
        float t = static_cast<float>(i) / sampleRate; // Time in seconds
        
        // Base frequency (the pitch-shifted note)
        float base_sample = amplitude * sinf(2.0f * static_cast<float>(M_PI) * static_cast<float>(pitch) * t);
        
        // Add sub-harmonic for deep bass effect on large trades
        float bass_sample = 0.0f;
        if (is_large_trade) {
            bass_sample = (amplitude * 0.3f * bass_boost) * 
                         sinf(2.0f * static_cast<float>(M_PI) * static_cast<float>(pitch * 0.5f) * t); // Octave lower
        }
        
        // Add some harmonic content for richness
        float harmonic_sample = 0.0f;
        if (pitch > 110.0) { // Only add harmonics for audible pitches
            harmonic_sample = (amplitude * 0.2f) * 
                             sinf(2.0f * static_cast<float>(M_PI) * static_cast<float>(pitch * 2.0f) * t); // Octave higher
        }
        
        float sample = base_sample + bass_sample + harmonic_sample;

        // Add slight variation based on trade direction (buy/sell)
        if (!is_buy) {
            // Slightly different waveform for sell trades
            sample *= 0.9f;
        }

        pAudioData[i] = sample;
    }

    // Create an audio buffer and play it using the engine
    ma_sound sound;
    ma_audio_buffer_config bufferConfig = ma_audio_buffer_config_init(
        ma_format_f32,    // Format
        channels,         // Channels
        totalFrames,      // Size in frames
        pAudioData        // Data pointer
    );

    ma_audio_buffer buffer;
    ma_result result = ma_audio_buffer_init(&bufferConfig, &buffer);
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to initialize audio buffer: " << result << std::endl;
        free(pAudioData);
        return;
    }

    // Initialize the sound with the audio buffer
    result = ma_sound_init_from_data_source(engine, &buffer.data_source, 0, NULL, &sound);
    if (result != MA_SUCCESS) {
        std::cerr << "[Audio] Failed to initialize sound: " << result << std::endl;
        ma_audio_buffer_uninit(&buffer);
        free(pAudioData);
        return;
    }

    // Set volume and play the sound
    ma_sound_set_volume(&sound, amplitude * bass_boost);
    ma_sound_start(&sound);

    // Wait briefly to allow the sound to play
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000)));

    // Clean up resources after playback
    ma_sound_uninit(&sound);
    ma_audio_buffer_uninit(&buffer);
    free(pAudioData);

    std::cout << "[Audio] Played trade sound - Volume: " << volume
              << ", Pitch: " << pitch
              << ", Side: " << (is_buy ? "BUY" : "SELL")
              << ", Large Trade: " << (is_large_trade ? "YES" : "NO") << std::endl;
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