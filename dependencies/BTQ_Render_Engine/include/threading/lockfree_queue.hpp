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
        // Initialize with a dummy node to simplify the algorithm
        Node* dummy = new Node();
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        // Clean up remaining nodes
        Node* current = head_.load(std::memory_order_acquire);
        while (current != nullptr) {
            Node* next = current->next.load(std::memory_order_acquire);
            delete current;
            current = next;
        }
    }

    void push(const T& new_value) {
        Node* new_node = new Node(new_value);

        // Keep trying until we successfully add the node
        while (true) {
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = tail_copy->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            if (tail_copy == tail_.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    // Tail was pointing to the last node, try to link our new node
                    if (tail_copy->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                        // Successfully added the node, now advance the tail
                        tail_.compare_exchange_strong(tail_copy, new_node, std::memory_order_release, std::memory_order_acquire);
                        return;
                    }
                } else {
                    // Tail wasn't pointing to the last node, advance it
                    tail_.compare_exchange_strong(tail_copy, next, std::memory_order_release, std::memory_order_acquire);
                }
            }
        }
    }

    void push(T&& new_value) {
        Node* new_node = new Node(std::move(new_value));

        // Keep trying until we successfully add the node
        while (true) {
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = tail_copy->next.load(std::memory_order_acquire);

            // Check if tail is still pointing to the same node
            if (tail_copy == tail_.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    // Tail was pointing to the last node, try to link our new node
                    if (tail_copy->next.compare_exchange_weak(next, new_node, std::memory_order_release)) {
                        // Successfully added the node, now advance the tail
                        tail_.compare_exchange_strong(tail_copy, new_node, std::memory_order_release, std::memory_order_acquire);
                        return;
                    }
                } else {
                    // Tail wasn't pointing to the last node, advance it
                    tail_.compare_exchange_strong(tail_copy, next, std::memory_order_release, std::memory_order_acquire);
                }
            }
        }
    }

    std::shared_ptr<T> pop() {
        // Keep trying until we successfully remove a node
        while (true) {
            Node* head_copy = head_.load(std::memory_order_acquire);
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = head_copy->next.load(std::memory_order_acquire);

            // Check if head is still pointing to the same node
            if (head_copy == head_.load(std::memory_order_acquire)) {
                if (head_copy == tail_copy) {
                    // Queue is empty or tail is falling behind
                    if (next == nullptr) {
                        return nullptr; // Queue is actually empty
                    }
                    // Tail is falling behind, try to advance it
                    tail_.compare_exchange_strong(tail_copy, next, std::memory_order_release, std::memory_order_acquire);
                    // Continue loop to try again
                } else {
                    if (next == nullptr) {
                        // This shouldn't happen in a consistent state, but handle it
                        return nullptr;
                    }

                    // Try to advance the head
                    if (head_.compare_exchange_weak(head_copy, next, std::memory_order_release)) {
                        T data = std::move(next->data);

                        // Successfully dequeued the node, delete the old head
                        delete head_copy;

                        return std::make_shared<T>(std::move(data));
                    }
                    // If compare_exchange failed, continue loop to try again
                }
            }
        }
    }

    // Non-blocking try_pop
    bool try_pop(T& value) {
        // Keep trying until we successfully remove a node
        while (true) {
            Node* head_copy = head_.load(std::memory_order_acquire);
            Node* tail_copy = tail_.load(std::memory_order_acquire);
            Node* next = head_copy->next.load(std::memory_order_acquire);

            // Check if head is still pointing to the same node
            if (head_copy == head_.load(std::memory_order_acquire)) {
                if (head_copy == tail_copy) {
                    // Queue is empty or tail is falling behind
                    if (next == nullptr) {
                        return false; // Queue is actually empty
                    }
                    // Tail is falling behind, try to advance it
                    tail_.compare_exchange_strong(tail_copy, next, std::memory_order_release, std::memory_order_acquire);
                    // Continue loop to try again
                } else {
                    if (next == nullptr) {
                        // This shouldn't happen in a consistent state, but handle it
                        return false;
                    }

                    // Try to advance the head
                    if (head_.compare_exchange_weak(head_copy, next, std::memory_order_release)) {
                        value = std::move(next->data);

                        // Successfully dequeued the node, delete the old head
                        delete head_copy;

                        return true;
                    }
                    // If compare_exchange failed, continue loop to try again
                }
            }
        }
    }

    bool empty() const {
        Node* head_copy = head_.load(std::memory_order_acquire);
        Node* tail_copy = tail_.load(std::memory_order_acquire);
        Node* next = head_copy->next.load(std::memory_order_acquire);

        if (head_copy == tail_copy) {
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
};

} // namespace threading
} // namespace btq

#endif // BTQ_LOCKFREE_QUEUE_HPP