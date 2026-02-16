#include "trading/trade_command_queue.hpp"
#include <iostream>

namespace BTQuant {
namespace RenderEngine {

TradeCommandQueue::TradeCommandQueue(size_t capacity)
    : queue_(capacity) {
    std::cout << "[TradeCommandQueue] Initialized with dedicated SPSC ring buffer (capacity: " << capacity << ")" << std::endl;
}

TradeCommandQueue::~TradeCommandQueue() {
    stop_execution_thread();
    std::cout << "[TradeCommandQueue] Shutdown complete" << std::endl;
}

bool TradeCommandQueue::push(const TradeCommand& command) {
    return queue_.push(command);
}

bool TradeCommandQueue::push(TradeCommand&& command) {
    return queue_.push(std::move(command));
}

bool TradeCommandQueue::try_pop(TradeCommand& command) {
    auto result = queue_.try_pop();
    if (result.has_value()) {
        command = std::move(result.value());
        return true;
    }
    return false;
}

size_t TradeCommandQueue::size_approx() const {
    return queue_.size();
}

bool TradeCommandQueue::empty() const {
    return queue_.empty();
}

void TradeCommandQueue::set_result_callback(TradeResultCallback callback) {
    result_callback_ = std::move(callback);
}

void TradeCommandQueue::report_result(const TradeResult& result) {
    if (result_callback_) {
        result_callback_(result);
    }
}

uint64_t TradeCommandQueue::next_command_id() {
    return next_command_id_.fetch_add(1, std::memory_order_relaxed);
}

void TradeCommandQueue::start_execution_thread(std::function<void(TradeCommand&, TradeCommandQueue&)> executor) {
    if (running_.load(std::memory_order_acquire)) {
        std::cout << "[TradeCommandQueue] Execution thread already running" << std::endl;
        return;
    }
    
    running_.store(true, std::memory_order_release);
    
    execution_thread_ = std::thread([this, executor = std::move(executor)]() {
        std::cout << "[TradeCommandQueue] Execution thread started" << std::endl;
        
        while (running_.load(std::memory_order_acquire)) {
            TradeCommand command;
            if (try_pop(command)) {
                try {
                    // Execute the command
                    executor(command, *this);
                } catch (const std::exception& e) {
                    std::cerr << "[TradeCommandQueue] Exception in executor: " << e.what() << std::endl;
                    
                    // Report failure
                    TradeResult result;
                    result.command_id = command.command_id;
                    result.success = false;
                    result.error_message = e.what();
                    result.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()
                    ).count();
                    report_result(result);
                }
            } else {
                // No commands available, sleep briefly
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        
        std::cout << "[TradeCommandQueue] Execution thread stopped" << std::endl;
    });
}

void TradeCommandQueue::stop_execution_thread() {
    running_.store(false, std::memory_order_release);
    
    if (execution_thread_.joinable()) {
        execution_thread_.join();
    }
}

}  // namespace RenderEngine
}  // namespace BTQuant
