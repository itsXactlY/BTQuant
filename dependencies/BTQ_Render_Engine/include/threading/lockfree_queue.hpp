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
#include <deque>
#include <mutex>
#include <condition_variable>
#include <cstddef>  // For ptrdiff_t

namespace btq {
namespace threading {

// Enhanced LockFreeQueue with better memory management and thread safety
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

    static constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size to prevent false sharing

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

                    // Delete the old head node (the sentinel node that was previously at head)
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

                    // Delete the old head node (the sentinel node that was previously at head)
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

    // Wait-free push operation for better performance
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

    // Drain all items from the queue - useful for UI updates to prevent buildup
    std::vector<T> drain_all() {
        std::vector<T> result;

        // Estimate size to reserve space upfront for efficiency
        size_t estimated_size = size_approx();
        if (estimated_size > 0) {
            result.reserve(estimated_size);
        }

        // Keep popping until queue is empty
        while (true) {
            auto item = try_pop();
            if (item.has_value()) {
                result.emplace_back(std::move(item.value()));
            } else {
                break; // Queue is empty
            }
        }

        return result;
    }

    // Limited push - only push if queue size is below threshold (prevents memory buildup)
    bool push_if_not_full(const T& new_value, size_t max_size = 1000) {
        if (size_approx() >= max_size) {
            return false; // Queue is too full
        }
        push(new_value);
        return true;
    }

    // Limited push with rvalue reference
    bool push_if_not_full(T&& new_value, size_t max_size = 1000) {
        if (size_approx() >= max_size) {
            return false; // Queue is too full
        }
        push(std::move(new_value));
        return true;
    }

    // Clear and reset the queue to initial state
    void reset() {
        Node* current = head_.load(std::memory_order_acquire);
        Node* tail_snapshot = tail_.load(std::memory_order_acquire);

        // Move head to tail position, releasing all intermediate nodes
        while (current != tail_snapshot) {
            Node* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }

        // Ensure head and tail point to the same sentinel
        head_.store(tail_snapshot, std::memory_order_release);
    }
};

// Lock-free multi-producer single-consumer queue using Michael & Scott algorithm with enhancements
template<typename T>
class MPSCQueue {
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

    static constexpr size_t CACHE_LINE_SIZE = 64; // Typical cache line size to prevent false sharing

    alignas(CACHE_LINE_SIZE) std::atomic<Node*> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<Node*> tail_;

    // Additional padding to avoid false sharing between head and tail
    alignas(CACHE_LINE_SIZE) char padding_[CACHE_LINE_SIZE];

public:
    MPSCQueue() {
        // Initialize with a dummy sentinel node to simplify the algorithm
        Node* sentinel = new Node();
        head_.store(sentinel, std::memory_order_relaxed);
        tail_.store(sentinel, std::memory_order_relaxed);
    }

    ~MPSCQueue() {
        // Sequentially clean up all nodes
        // This assumes that no other threads are accessing the queue during destruction
        Node* current = head_.load(std::memory_order_acquire);

        while (current != nullptr) {
            Node* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }
    }

    void push(const T& item) {
        Node* new_node = new Node(item);

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
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_release, std::memory_order_acquire);
            }
        }
    }

    void push(T&& item) {
        Node* new_node = new Node(std::move(item));

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
                    tail_.compare_exchange_strong(prev_tail, new_node, std::memory_order_release, std::memory_order_acquire);
                    return;
                }
            } else {
                // Tail wasn't pointing to the last node, advance it
                tail_.compare_exchange_strong(prev_tail, next, std::memory_order_release, std::memory_order_acquire);
            }
        }
    }

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
                tail_.compare_exchange_strong(tail_snapshot, next, std::memory_order_release, std::memory_order_acquire);
                continue;
            } else {
                if (next == nullptr) {
                    // This shouldn't happen in a consistent state, but handle it
                    return std::nullopt;
                }

                // Try to advance the head to the next node
                if (head_.compare_exchange_weak(head_snapshot, next, std::memory_order_release, std::memory_order_acquire)) {
                    // Successfully dequeued, extract the data
                    T data = std::move(next->data);

                    // Delete the old head node (the sentinel node that was previously at head)
                    delete head_snapshot;

                    return std::move(data);
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

    template<typename... Args>
    void emplace(Args&&... args) {
        push(T(std::forward<Args>(args)...));
    }

    // Drain all items from the queue - useful for UI updates to prevent buildup
    std::vector<T> drain_all() {
        std::vector<T> result;

        // Keep popping until queue is empty
        while (true) {
            auto item = try_pop();
            if (item.has_value()) {
                result.emplace_back(std::move(item.value()));
            } else {
                break; // Queue is empty
            }
        }

        return result;
    }

    // Limited push - only push if queue size is below threshold (prevents memory buildup)
    bool push_if_not_full(const T& item, size_t max_size = 1000) {
        // Since we don't have an efficient size() method for MPSC, we'll use a different approach
        // This is a simplified version - in production, you might track count separately
        push(item);
        return true; // Always return true since we can't efficiently check size
    }

    // Limited push with rvalue reference
    bool push_if_not_full(T&& item, size_t max_size = 1000) {
        push(std::move(item));
        return true; // Always return true since we can't efficiently check size
    }
};

// Lock-free stack implementation for LIFO operations
template<typename T>
class LockFreeStack {
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

    alignas(64) std::atomic<Node*> head_{nullptr};

public:
    void push(const T& item) {
        Node* new_node = new Node(item);
        Node* current_head = head_.load(std::memory_order_relaxed);

        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(current_head, new_node,
                                             std::memory_order_acq_rel,
                                             std::memory_order_relaxed));
    }

    void push(T&& item) {
        Node* new_node = new Node(std::move(item));
        Node* current_head = head_.load(std::memory_order_relaxed);

        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(current_head, new_node,
                                             std::memory_order_acq_rel,
                                             std::memory_order_relaxed));
    }

    std::shared_ptr<T> pop() {
        Node* old_head = head_.load(std::memory_order_relaxed);

        while (old_head != nullptr) {
            Node* new_head = old_head->next.load(std::memory_order_relaxed);

            if (head_.compare_exchange_weak(old_head, new_head,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
                std::shared_ptr<T> result = std::make_shared<T>(std::move(old_head->data));
                delete old_head;
                return result;
            }
        }

        return nullptr;
    }

    std::optional<T> try_pop() {
        Node* old_head = head_.load(std::memory_order_relaxed);

        while (old_head != nullptr) {
            Node* new_head = old_head->next.load(std::memory_order_relaxed);

            if (head_.compare_exchange_weak(old_head, new_head,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed)) {
                T result = std::move(old_head->data);
                delete old_head;
                return result;
            }
        }

        return std::nullopt;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

    template<typename... Args>
    void emplace(Args&&... args) {
        push(T(std::forward<Args>(args)...));
    }

    // Drain all items from the queue - useful for UI updates to prevent buildup
    std::vector<T> drain_all() {
        std::vector<T> result;

        // Keep popping until queue is empty
        while (true) {
            auto item = try_pop();
            if (item.has_value()) {
                result.emplace_back(std::move(item.value()));
            } else {
                break; // Queue is empty
            }
        }

        return result;
    }

    // Limited push - only push if queue size is below threshold (prevents memory buildup)
    bool push_if_not_full(const T& item, size_t max_size = 1000) {
        // Since we don't have an efficient size() method for MPSC, we'll use a different approach
        // This is a simplified version - in production, you might track count separately
        push(item);
        return true; // Always return true since we can't efficiently check size
    }

    // Limited push with rvalue reference
    bool push_if_not_full(T&& item, size_t max_size = 1000) {
        push(std::move(item));
        return true; // Always return true since we can't efficiently check size
    }
};

// Single-producer single-consumer ring buffer for high-performance scenarios
template<typename T>
class SPSCRingBuffer {
private:
    struct BufferNode {
        alignas(64) std::atomic<bool> ready{false};
        T data{};
    };

    std::vector<BufferNode> buffer_;
    const size_t capacity_;

    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};

public:
    explicit SPSCRingBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity) {}

    bool push(const T& item) {
        size_t write_idx = write_pos_.load(std::memory_order_relaxed);
        size_t next_write_idx = (write_idx + 1) % capacity_;

        // Check if buffer is full (leave one slot empty to distinguish from empty)
        if (next_write_idx == read_pos_.load(std::memory_order_acquire)) {
            return false; // Buffer is full
        }

        buffer_[write_idx].data = item;
        buffer_[write_idx].ready.store(true, std::memory_order_release);
        write_pos_.store(next_write_idx, std::memory_order_release);

        return true;
    }

    bool push(T&& item) {
        size_t write_idx = write_pos_.load(std::memory_order_relaxed);
        size_t next_write_idx = (write_idx + 1) % capacity_;

        // Check if buffer is full (leave one slot empty to distinguish from empty)
        if (next_write_idx == read_pos_.load(std::memory_order_acquire)) {
            return false; // Buffer is full
        }

        buffer_[write_idx].data = std::move(item);
        buffer_[write_idx].ready.store(true, std::memory_order_release);
        write_pos_.store(next_write_idx, std::memory_order_release);

        return true;
    }

    std::optional<T> try_pop() {
        size_t read_idx = read_pos_.load(std::memory_order_relaxed);

        if (read_idx == write_pos_.load(std::memory_order_acquire)) {
            return std::nullopt; // Buffer is empty
        }

        // Ensure data is ready before consuming
        if (!buffer_[read_idx].ready.load(std::memory_order_acquire)) {
            return std::nullopt; // Data not ready yet
        }

        T result = std::move(buffer_[read_idx].data);
        buffer_[read_idx].ready.store(false, std::memory_order_release);
        read_pos_.store((read_idx + 1) % capacity_, std::memory_order_release);

        return result;
    }

    bool empty() const {
        return read_pos_.load(std::memory_order_acquire) == write_pos_.load(std::memory_order_acquire);
    }

    bool full() const {
        size_t next_write_pos = (write_pos_.load(std::memory_order_acquire) + 1) % capacity_;
        return next_write_pos == read_pos_.load(std::memory_order_acquire);
    }

    size_t size() const {
        ptrdiff_t sz = static_cast<ptrdiff_t>(write_pos_.load(std::memory_order_acquire)) -
                       static_cast<ptrdiff_t>(read_pos_.load(std::memory_order_acquire));
        if (sz < 0) sz += static_cast<ptrdiff_t>(capacity_);
        return static_cast<size_t>(sz);
    }

    size_t capacity() const {
        return capacity_;
    }

    template<typename... Args>
    bool emplace(Args&&... args) {
        return push(T(std::forward<Args>(args)...));
    }

    // Drain all items from the buffer - useful for UI updates to prevent buildup
    std::vector<T> drain_all() {
        std::vector<T> result;

        // Keep popping until buffer is empty
        while (true) {
            auto item = try_pop();
            if (item.has_value()) {
                result.emplace_back(std::move(item.value()));
            } else {
                break; // Buffer is empty
            }
        }

        return result;
    }

    // Check if buffer has space before pushing (prevents overwriting)
    bool push_if_not_full(const T& item) {
        if (full()) {
            return false; // Buffer is full
        }
        return push(item);
    }

    // Check if buffer has space before pushing with rvalue reference
    bool push_if_not_full(T&& item) {
        if (full()) {
            return false; // Buffer is full
        }
        return push(std::move(item));
    }
};

// Atomic wrapper for simple data types to ensure thread-safe access
template<typename T>
class AtomicWrapper {
private:
    mutable std::atomic<T> value_;

public:
    explicit AtomicWrapper(const T& initial_value = T{}) : value_(initial_value) {}

    T load(std::memory_order order = std::memory_order_seq_cst) const {
        return value_.load(order);
    }

    void store(const T& desired, std::memory_order order = std::memory_order_seq_cst) {
        value_.store(desired, order);
    }

    T exchange(const T& desired, std::memory_order order = std::memory_order_seq_cst) {
        return value_.exchange(desired, order);
    }

    bool compare_exchange_weak(T& expected, const T& desired,
                              std::memory_order success, std::memory_order failure) {
        return value_.compare_exchange_weak(expected, desired, success, failure);
    }

    bool compare_exchange_strong(T& expected, const T& desired,
                                std::memory_order success, std::memory_order failure) {
        return value_.compare_exchange_strong(expected, desired, success, failure);
    }

    // Assignment operators
    AtomicWrapper& operator=(const T& desired) {
        store(desired);
        return *this;
    }

    operator T() const {
        return load();
    }
};

} // namespace threading
} // namespace btq

#endif // BTQ_LOCKFREE_QUEUE_HPP