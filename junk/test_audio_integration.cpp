#include "src/audio/audio_engine_integration.h"
#include "src/analytics/rawtradetable.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Testing Audio Integration for Trade Sounds..." << std::endl;
    
    // Create a raw trade table
    RawTradeTable trade_table(1000);
    
    // Initialize audio integration to trigger sounds on new trades
    std::cout << "Initializing audio integration for trade notifications...\n";
    AudioIntegration::TradeAudioNotifier audio_notifier;
    if (audio_notifier.initializeAudio()) {
        // Connect the audio notifier to the trade table
        audio_notifier.connectToTradeTable(trade_table);
        std::cout << "Audio integration connected to trade table\n";
    } else {
        std::cout << "Warning: Could not initialize audio, continuing without sound\n";
        return 1;
    }

    // Generate some sample trades to test the audio
    auto trade_base_time = std::chrono::system_clock::now();
    
    std::cout << "\nAdding BUY trade to test sound...\n";
    RawTrade buy_trade;
    buy_trade.timestamp = trade_base_time;
    buy_trade.price = 100.50;
    buy_trade.volume = 10.0;
    buy_trade.side = 'B';  // Buy
    buy_trade.trade_id = "TEST_BUY_001";
    
    trade_table.add_trade(buy_trade);
    
    // Wait a moment to hear the sound
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "\nAdding SELL trade to test sound...\n";
    RawTrade sell_trade;
    sell_trade.timestamp = trade_base_time + std::chrono::milliseconds(100);
    sell_trade.price = 100.45;
    sell_trade.volume = 15.0;
    sell_trade.side = 'S';  // Sell
    sell_trade.trade_id = "TEST_SELL_001";
    
    trade_table.add_trade(sell_trade);
    
    // Wait a moment to hear the sound
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Clean up audio resources
    audio_notifier.shutdownAudio();
    
    std::cout << "\nAudio integration test completed successfully!" << std::endl;
    
    return 0;
}