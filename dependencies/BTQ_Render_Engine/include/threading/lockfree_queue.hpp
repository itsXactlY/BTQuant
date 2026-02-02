#ifndef BTQ_LOCKFREE_QUEUE_HPP
#define BTQ_LOCKFREE_QUEUE_HPP

#include <atomic>
#include <memory>
#include <thread>
#include <new>

namespace btq {
namespace threading {

template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        T data{};

        Node() = default;
        explicit Node(const T& value) : data(value) {}
        explicit Node(T&& value) : data(std::move(value)) {}
    };

    static constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size

    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail_;

    // Padding to avoid false sharing
    char padding_[CACHE_LINE_SIZE - sizeof(std::atomic<Node*>)];

    // For memory management
    static constexpr size_t DEFAULT_POOL_SIZE = 1024;
    Node* pool_;
    std::atomic<size_t> pool_index_{0};
    size_t pool_size_;

public:
    explicit LockFreeQueue(size_t pool_size = DEFAULT_POOL_SIZE)
        : pool_size_(pool_size) {
        // Allocate a pool of nodes upfront
        pool_ = new Node[pool_size_];

        // Initialize with a dummy node
        Node* dummy = &pool_[0];
        pool_index_.store(1);

        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        // Clean up remaining nodes
        Node* current = head_.load(std::memory_order_relaxed);
        while (current != nullptr) {
            Node* next = reinterpret_cast<Node*>(current->next.load(std::memory_order_relaxed));
            // Don't delete pool nodes, they're managed separately
            current = next;
        }
        delete[] pool_;
    }

    void push(const T& new_value) {
        Node* new_node = allocate_node(new_value);
        if (!new_node) return; // Pool exhausted

        Node* prev_tail = tail_.load(std::memory_order_relaxed);
        while (true) {
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = reinterpret_cast<Node*>(tail_copy->next.load(std::memory_order_acquire));

            if (tail_copy == tail_.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    // Try to link the new node
                    if (tail_copy->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                        break; // Successfully linked
                    }
                } else {
                    // Move tail forward
                    tail_.compare_exchange_weak(tail_copy, next, std::memory_order_release);
                }
            }
        }

        // Now move the tail pointer to the new node
        tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release);
    }

    void push(T&& new_value) {
        Node* new_node = allocate_node(std::move(new_value));
        if (!new_node) return; // Pool exhausted

        Node* prev_tail = tail_.load(std::memory_order_relaxed);
        while (true) {
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = reinterpret_cast<Node*>(tail_copy->next.load(std::memory_order_acquire));

            if (tail_copy == tail_.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    // Try to link the new node
                    if (tail_copy->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                        break; // Successfully linked
                    }
                } else {
                    // Move tail forward
                    tail_.compare_exchange_weak(tail_copy, next, std::memory_order_release);
                }
            }
        }

        // Now move the tail pointer to the new node
        tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release);
    }

    std::shared_ptr<T> pop() {
        Node* head_copy = head_.load(std::memory_order_acquire);
        Node* next = reinterpret_cast<Node*>(head_copy->next.load(std::memory_order_acquire));

        if (next == nullptr) {
            return nullptr; // Queue is empty
        }

        // Move the head forward
        if (head_.compare_exchange_strong(head_copy, next, std::memory_order_release)) {
            T data = std::move(next->data);
            // Note: In a real implementation, we'd return the node to a freelist
            // For this implementation, we'll just return the data
            return std::make_shared<T>(std::move(data));
        }

        return nullptr; // Failed to pop
    }

    // Non-blocking try_pop
    bool try_pop(T& value) {
        Node* head_copy = head_.load(std::memory_order_acquire);
        Node* next = reinterpret_cast<Node*>(head_copy->next.load(std::memory_order_acquire));

        if (next == nullptr) {
            return false; // Queue is empty
        }

        // Move the head forward
        if (head_.compare_exchange_strong(head_copy, next, std::memory_order_release)) {
            value = std::move(next->data);
            // Note: In a real implementation, we'd return the node to a freelist
            return true;
        }

        return false; // Failed to pop
    }

    bool empty() const {
        Node* head_copy = head_.load(std::memory_order_acquire);
        Node* next = head_copy->next.load(std::memory_order_acquire);
        return next == nullptr;
    }

    // Note: size() is not lock-free and should be used carefully in concurrent environments
    size_t size_approx() const {
        size_t count = 0;
        Node* current = head_.load(std::memory_order_acquire);
        Node* tail_copy = tail_.load(std::memory_order_acquire);

        while (current != tail_copy) {
            Node* next = current->next.load(std::memory_order_acquire);
            if (next != nullptr) {
                current = next;
                count++;
            } else {
                break;
            }
        }
        // Don't subtract 1 for dummy node since we're counting elements after dummy
        return count;
    }

    // For compatibility with existing interface
    size_t size() const {
        return size_approx();
    }

private:
    Node* allocate_node(const T& value) {
        size_t idx = pool_index_.fetch_add(1, std::memory_order_acquire);
        if (idx >= pool_size_) {
            return nullptr; // Pool exhausted
        }
        new (&pool_[idx]) Node(value);
        return &pool_[idx];
    }

    Node* allocate_node(T&& value) {
        size_t idx = pool_index_.fetch_add(1, std::memory_order_acquire);
        if (idx >= pool_size_) {
            return nullptr; // Pool exhausted
        }
        new (&pool_[idx]) Node(std::move(value));
        return &pool_[idx];
    }
};

} // namespace threading
} // namespace btq

#endif // BTQ_LOCKFREE_QUEUE_HPP