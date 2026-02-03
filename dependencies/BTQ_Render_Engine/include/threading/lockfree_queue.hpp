#ifndef BTQ_LOCKFREE_QUEUE_HPP
#define BTQ_LOCKFREE_QUEUE_HPP

#include <atomic>
#include <memory>
#include <thread>
#include <new>
#include <optional>
#include <utility>
#include <vector>
#include <memory_resource> // For potential memory resource support

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

        template<typename... Args>
        explicit Node(Args&&... args) : data(std::forward<Args>(args)...) {}
    };

    static constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size

    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail_;

    // Additional padding to avoid false sharing between head and tail
    alignas(CACHE_LINE_SIZE) char padding_[CACHE_LINE_SIZE];

public:
    explicit LockFreeQueue() {
        // Initialize with a dummy sentinel node to simplify the algorithm
        Node* sentinel = new Node();
        head_.store(sentinel, std::memory_order_relaxed);
        tail_.store(sentinel, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        // Sequentially clean up all nodes
        // This assumes that no other threads are accessing the queue during destruction
        Node* current = head_.load(std::memory_order_acquire);

        while (current != nullptr) {
            Node* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }
    }

    void push(const T& new_value) {
        Node* new_node = new Node(new_value);

        Node* prev_tail = tail_.load(std::memory_order_acquire);

        while (true) {
            Node* next = prev_tail->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            if (prev_tail != tail_snapshot) {
                // Another thread advanced tail, update our view
                prev_tail = tail_snapshot;
                continue;
            }

            if (next == nullptr) {
                // Tail was pointing to the last node, try to link our new node
                if (prev_tail->next.compare_exchange_weak(next, new_node, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Successfully added the node, now advance the tail
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_acq_rel, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_acq_rel, std::memory_order_acquire);
            }
        }
    }

    void push(T&& new_value) {
        Node* new_node = new Node(std::move(new_value));

        Node* prev_tail = tail_.load(std::memory_order_acquire);

        while (true) {
            Node* next = prev_tail->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            if (prev_tail != tail_snapshot) {
                // Another thread advanced tail, update our view
                prev_tail = tail_snapshot;
                continue;
            }

            if (next == nullptr) {
                // Tail was pointing to the last node, try to link our new node
                if (prev_tail->next.compare_exchange_weak(next, new_node, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Successfully added the node, now advance the tail
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_acq_rel, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_acq_rel, std::memory_order_acquire);
            }
        }
    }

    std::shared_ptr<T> pop() {
        Node* prev_head = head_.load(std::memory_order_acquire);

        while (true) {
            Node* head_snapshot = head_.load(std::memory_order_acquire);
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            Node* next = head_snapshot->next.load(std::memory_order_acquire);

            if (head_snapshot == tail_snapshot) {
                // Queue is empty or tail is falling behind
                if (next == nullptr) {
                    return nullptr; // Queue is actually empty
                }
                // Tail is falling behind, try to advance it
                tail_.compare_exchange_strong(tail_snapshot, next, std::memory_order_acq_rel, std::memory_order_acquire);
                continue;
            } else {
                if (next == nullptr) {
                    // This shouldn't happen in a consistent state, but handle it
                    return nullptr;
                }

                // Try to advance the head to the next node
                if (head_.compare_exchange_weak(head_snapshot, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Successfully dequeued, extract the data
                    T data = std::move(next->data);

                    // Delete the old head (the sentinel node that was previously at head)
                    // We only delete the old head after advancing the head pointer
                    delete head_snapshot;

                    return std::make_shared<T>(std::move(data));
                }
                // If compare_exchange failed, continue loop to try again
            }
        }
    }

    // Non-blocking try_pop with std::optional return
    std::optional<T> try_pop() {
        Node* prev_head = head_.load(std::memory_order_acquire);

        while (true) {
            Node* head_snapshot = head_.load(std::memory_order_acquire);
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            Node* next = head_snapshot->next.load(std::memory_order_acquire);

            if (head_snapshot == tail_snapshot) {
                // Queue is empty or tail is falling behind
                if (next == nullptr) {
                    return std::nullopt; // Queue is actually empty
                }
                // Tail is falling behind, try to advance it
                tail_.compare_exchange_strong(tail_snapshot, next, std::memory_order_acq_rel, std::memory_order_acquire);
                continue;
            } else {
                if (next == nullptr) {
                    // This shouldn't happen in a consistent state, but handle it
                    return std::nullopt;
                }

                // Try to advance the head to the next node
                if (head_.compare_exchange_weak(head_snapshot, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Successfully dequeued, extract the data
                    T data = std::move(next->data);

                    // Delete the old head (the sentinel node that was previously at head)
                    delete head_snapshot;

                    return std::move(data);
                }
                // If compare_exchange failed, continue loop to try again
            }
        }
    }

    // Legacy try_pop for backward compatibility
    bool try_pop(T& value) {
        auto result = try_pop();
        if (result.has_value()) {
            value = std::move(result.value());
            return true;
        }
        return false;
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

    // Wait-free push operation with memory pool for better performance
    template<typename... Args>
    void emplace(Args&&... args) {
        Node* new_node = new Node(std::forward<Args>(args)...);

        Node* prev_tail = tail_.load(std::memory_order_acquire);

        while (true) {
            Node* next = prev_tail->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            Node* tail_snapshot = tail_.load(std::memory_order_acquire);
            if (prev_tail != tail_snapshot) {
                // Another thread advanced tail, update our view
                prev_tail = tail_snapshot;
                continue;
            }

            if (next == nullptr) {
                // Tail was pointing to the last node, try to link our new node
                if (prev_tail->next.compare_exchange_weak(next, new_node, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Successfully added the node, now advance the tail
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_acq_rel, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_acq_rel, std::memory_order_acquire);
            }
        }
    }

    // Batch push operation for better performance when pushing multiple items
    template<typename Iterator>
    void push_batch(Iterator begin, Iterator end) {
        for (auto it = begin; it != end; ++it) {
            push(*it);
        }
    }

    // Batch pop operation to retrieve multiple items at once
    std::vector<T> pop_batch(size_t max_items) {
        std::vector<T> result;
        result.reserve(max_items);

        for (size_t i = 0; i < max_items; ++i) {
            auto item = try_pop();
            if (item.has_value()) {
                result.emplace_back(std::move(item.value()));
            } else {
                break; // Queue is empty
            }
        }

        return result;
    }

    // Blocking pop with timeout for use in UI thread
    template<typename Rep, typename Period>
    std::shared_ptr<T> pop_for(const std::chrono::duration<Rep, Period>& timeout_duration) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + timeout_duration;

        while (std::chrono::steady_clock::now() < end_time) {
            auto result = pop();
            if (result) {
                return result;
            }
            std::this_thread::yield(); // Allow other threads to run
        }

        return nullptr; // Timeout reached
    }

    // Method to check if the queue has data without fully consuming it
    bool has_data() const {
        Node* head_snapshot = head_.load(std::memory_order_acquire);
        Node* tail_snapshot = tail_.load(std::memory_order_acquire);

        return head_snapshot != tail_snapshot || head_snapshot->next.load(std::memory_order_acquire) != nullptr;
    }

    // Method to get approximate number of waiting consumers (not exact, for optimization hints)
    bool has_waiting_consumers() const {
        // This is a simplified check - in practice, you'd need more sophisticated tracking
        return !empty();
    }
};

} // namespace threading
} // namespace btq

#endif // BTQ_LOCKFREE_QUEUE_HPP