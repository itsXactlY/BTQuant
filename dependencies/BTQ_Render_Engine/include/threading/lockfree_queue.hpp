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

public:
    explicit LockFreeQueue() {
        // Initialize with a dummy sentinel node to simplify the algorithm
        Node* sentinel = new Node();
        head_.store(sentinel, std::memory_order_relaxed);
        tail_.store(sentinel, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        // Clean up all nodes including the sentinel
        while (Node* const old_head = head_.load(std::memory_order_relaxed)) {
            head_.store(old_head->next.load(std::memory_order_relaxed), std::memory_order_relaxed);
            delete old_head;
        }
    }

    void push(const T& new_value) {
        Node* new_node = new Node(new_value);

        // Atomically get the current tail
        Node* prev_tail = tail_.load(std::memory_order_relaxed);
        Node* next = nullptr;

        while (true) {
            // Load the next pointer of the current tail
            next = prev_tail->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            if (prev_tail != tail_snapshot) {
                // Another thread advanced tail, update our view
                prev_tail = tail_snapshot;
                continue;
            }

            if (next == nullptr) {
                // Tail was pointing to the last node, try to link our new node
                if (prev_tail->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                    // Successfully added the node, now advance the tail
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_release, std::memory_order_acquire);
            }

            // Update prev_tail for the next iteration
            prev_tail = tail_.load(std::memory_order_relaxed);
        }
    }

    void push(T&& new_value) {
        Node* new_node = new Node(std::move(new_value));

        // Atomically get the current tail
        Node* prev_tail = tail_.load(std::memory_order_relaxed);
        Node* next = nullptr;

        while (true) {
            // Load the next pointer of the current tail
            next = prev_tail->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            if (prev_tail != tail_snapshot) {
                // Another thread advanced tail, update our view
                prev_tail = tail_snapshot;
                continue;
            }

            if (next == nullptr) {
                // Tail was pointing to the last node, try to link our new node
                if (prev_tail->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                    // Successfully added the node, now advance the tail
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_release, std::memory_order_acquire);
            }

            // Update prev_tail for the next iteration
            prev_tail = tail_.load(std::memory_order_relaxed);
        }
    }

    std::shared_ptr<T> pop() {
        Node* prev_head = head_.load(std::memory_order_relaxed);

        while (true) {
            Node* head_snapshot = head_.load(std::memory_order_acquire);
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            Node* next = head_snapshot->next.load(std::memory_order_acquire);

            // Check if head is still pointing to the same node
            if (head_snapshot != head_.load(std::memory_order_acquire)) {
                continue; // Another thread modified head, retry
            }

            if (head_snapshot == tail_snapshot) {
                // Queue is empty or tail is falling behind
                if (next == nullptr) {
                    return nullptr; // Queue is actually empty
                }
                // Tail is falling behind, try to advance it
                tail_.compare_exchange_strong(tail_snapshot, next, std::memory_order_release, std::memory_order_acquire);
                continue;
            } else {
                if (next == nullptr) {
                    // This shouldn't happen in a consistent state, but handle it
                    return nullptr;
                }

                // Try to advance the head to the next node
                if (head_.compare_exchange_weak(head_snapshot, next, std::memory_order_release, std::memory_order_acquire)) {
                    // Successfully dequeued, extract the data
                    T data = std::move(next->data);

                    // Delete the old head (sentinel node), but only if it's not the initial sentinel
                    // The initial sentinel node will be deleted in the destructor
                    if (head_snapshot != prev_head && head_snapshot != head_.load(std::memory_order_relaxed)) {
                        delete head_snapshot;
                    }

                    return std::make_shared<T>(std::move(data));
                }
                // If compare_exchange failed, continue loop to try again
            }
        }
    }

    // Non-blocking try_pop
    bool try_pop(T& value) {
        Node* prev_head = head_.load(std::memory_order_relaxed);

        while (true) {
            Node* head_snapshot = head_.load(std::memory_order_acquire);
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            Node* next = head_snapshot->next.load(std::memory_order_acquire);

            // Check if head is still pointing to the same node
            if (head_snapshot != head_.load(std::memory_order_acquire)) {
                continue; // Another thread modified head, retry
            }

            if (head_snapshot == tail_snapshot) {
                // Queue is empty or tail is falling behind
                if (next == nullptr) {
                    return false; // Queue is actually empty
                }
                // Tail is falling behind, try to advance it
                tail_.compare_exchange_strong(tail_snapshot, next, std::memory_order_release, std::memory_order_acquire);
                continue;
            } else {
                if (next == nullptr) {
                    // This shouldn't happen in a consistent state, but handle it
                    return false;
                }

                // Try to advance the head to the next node
                if (head_.compare_exchange_weak(head_snapshot, next, std::memory_order_release, std::memory_order_acquire)) {
                    // Successfully dequeued, extract the data
                    value = std::move(next->data);

                    // Delete the old head (sentinel node), but only if it's not the initial sentinel
                    // The initial sentinel node will be deleted in the destructor
                    if (head_snapshot != prev_head && head_snapshot != head_.load(std::memory_order_relaxed)) {
                        delete head_snapshot;
                    }

                    return true;
                }
                // If compare_exchange failed, continue loop to try again
            }
        }
    }

    bool empty() const {
        Node* head_snapshot = head_.load(std::memory_order_acquire);
        Node* tail_snapshot = tail_.load(std::memory_order_acquire);
        Node* next = head_snapshot->next.load(std::memory_order_acquire);

        if (head_snapshot == tail_snapshot) {
            return (next == nullptr);
        }
        return false; // There are definitely elements in the queue
    }

    // Note: size() is not lock-free and should be used carefully in concurrent environments
    size_t size_approx() const {
        size_t count = 0;
        Node* current = head_.load(std::memory_order_acquire)->next.load(std::memory_order_acquire);

        while (current != nullptr) {
            current = current->next.load(std::memory_order_acquire);
            count++;
        }
        return count;
    }

    // For compatibility with existing interface
    size_t size() const {
        return size_approx();
    }

    // Additional utility methods for thread safety
    void clear() {
        while (pop() != nullptr) {
            // Keep popping until queue is empty
        }
    }
};

} // namespace threading
} // namespace btq

#endif // BTQ_LOCKFREE_QUEUE_HPP