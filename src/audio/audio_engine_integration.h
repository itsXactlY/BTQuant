#ifndef AUDIO_ENGINE_INTEGRATION_H
#define AUDIO_ENGINE_INTEGRATION_H

#include "../analytics/rawtradetable.h"
#include "pitch_shifter.h"
#include <memory>

// Forward declaration of miniaudio types
#ifdef MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#endif

namespace AudioIntegration {

/**
 * @brief Class to integrate audio engine with trade detection
 * This class connects the RawTradeTable to the audio system to trigger sounds on new trades
 */
class TradeAudioNotifier {
private:
    PitchShifter pitch_shifter_;
#ifdef MINIAUDIO_IMPLEMENTATION
    ma_engine* audio_engine_;
    bool audio_initialized_;
#else
    void* audio_engine_;  // Placeholder when miniaudio is not available
    bool audio_initialized_;
#endif

public:
    TradeAudioNotifier();
    ~TradeAudioNotifier();
    
    /**
     * @brief Initialize the audio engine
     * @return true if initialization was successful, false otherwise
     */
    bool initializeAudio();
    
    /**
     * @brief Connect to a RawTradeTable to receive trade notifications
     * @param trade_table Reference to the RawTradeTable to monitor
     */
    void connectToTradeTable(RawTradeTable& trade_table);
    
    /**
     * @brief Callback function to be called when a new trade is detected
     * @param trade The new trade that was detected
     */
    void onTradeDetected(const RawTrade& trade);
    
    /**
     * @brief Play a sound based on trade characteristics
     * @param volume The volume of the trade (used to determine pitch)
     * @param is_buy Whether the trade was a buy (affects sound characteristics)
     */
    void playTradeSound(double volume, bool is_buy);
    
    /**
     * @brief Shutdown the audio engine
     */
    void shutdownAudio();
};

} // namespace AudioIntegration

#endif // AUDIO_ENGINE_INTEGRATION_H