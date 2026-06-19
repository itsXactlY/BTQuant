#ifndef BTQUANT_RING_BUFFER_HPP
#define BTQUANT_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace btquant::data {

template<typename T, size_t Capacity>
class RingBuffer {
public:
    RingBuffer() : m_buffer(std::make_unique<T[]>(Capacity)) {}

    bool push(const T& item) {
        size_t writePos = m_writePos.load(std::memory_order_relaxed);
        size_t nextPos = (writePos + 1) % Capacity;
        if (nextPos == m_readPos.load(std::memory_order_acquire)) {
            return false;  // Full
        }
        m_buffer[writePos] = item;
        m_writePos.store(nextPos, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t readPos = m_readPos.load(std::memory_order_relaxed);
        if (readPos == m_writePos.load(std::memory_order_acquire)) {
            return false;  // Empty
        }
        item = m_buffer[readPos];
        m_readPos.store((readPos + 1) % Capacity, std::memory_order_release);
        return true;
    }

    bool isEmpty() const {
        return m_readPos.load(std::memory_order_acquire) == m_writePos.load(std::memory_order_acquire);
    }

    bool isFull() const {
        return ((m_writePos.load(std::memory_order_acquire) + 1) % Capacity) == m_readPos.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t w = m_writePos.load(std::memory_order_acquire);
        size_t r = m_readPos.load(std::memory_order_acquire);
        return (w >= r) ? (w - r) : (Capacity - r + w);
    }

private:
    std::unique_ptr<T[]> m_buffer;
    std::atomic<size_t> m_writePos{0};
    std::atomic<size_t> m_readPos{0};
};

} // namespace btquant::data

#endif
