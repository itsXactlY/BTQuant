#ifndef INCLUDE_CCAPI_CPP_CCAPI_QUEUE_H_
#define INCLUDE_CCAPI_CPP_CCAPI_QUEUE_H_
#include <vector>
#include "concurrentqueue.h"

#include "ccapi_cpp/ccapi_logger.h"

namespace ccapi {

/**
 * This class represents a generic FIFO queue.
 */

template <typename T>
class Queue {
 public:
  inline static const std::string EXCEPTION_QUEUE_FULL = "queue is full";
  inline static const std::string EXCEPTION_QUEUE_EMPTY = "queue is empty";

  explicit Queue(const size_t maxSize = 0) : maxSize(maxSize) {}

  void pushBack(const T& t) {
    // Note: ConcurrentQueue doesn't have a built-in max size check, so we approximate
    if (this->maxSize <= 0 || this->moody_queue.size_approx() < this->maxSize) {
      CCAPI_LOGGER_TRACE("this->queue.size() = " + size_tToString(this->moody_queue.size_approx()));
      this->moody_queue.enqueue(t);
    } else {
      throw std::runtime_error(EXCEPTION_QUEUE_FULL);
    }
  }

  void pushBack(T&& t) {
    // Note: ConcurrentQueue doesn't have a built-in max size check, so we approximate
    if (this->maxSize <= 0 || this->moody_queue.size_approx() < this->maxSize) {
      CCAPI_LOGGER_TRACE("this->queue.size() = " + size_tToString(this->moody_queue.size_approx()));
      this->moody_queue.enqueue(std::move(t));
    } else {
      throw std::runtime_error(EXCEPTION_QUEUE_FULL);
    }
  }

  T popBack() {
    T t;
    if (this->moody_queue.try_dequeue(t)) {
      return t;
    } else {
      throw std::runtime_error(EXCEPTION_QUEUE_EMPTY);
    }
  }

  std::vector<T> purge() {
    std::vector<T> p;
    T item;
    while (this->moody_queue.try_dequeue(item)) {
      p.push_back(std::move(item));
    }
    return p;
  }

  void removeAll(std::vector<T>& c) {
    T item;
    if (c.empty()) {
      while (this->moody_queue.try_dequeue(item)) {
        c.push_back(std::move(item));
      }
    } else {
      c.reserve(c.size() + this->moody_queue.size_approx());
      while (this->moody_queue.try_dequeue(item)) {
        c.push_back(std::move(item));
      }
    }
  }

  size_t size() const {
    return this->moody_queue.size_approx();
  }

  bool empty() const {
    return this->moody_queue.size_approx() == 0;
  }
#ifndef CCAPI_EXPOSE_INTERNAL

 private:
#endif
  mutable moodycamel::ConcurrentQueue<T> moody_queue;
  size_t maxSize{};
};

} /* namespace ccapi */
#endif  // INCLUDE_CCAPI_CPP_CCAPI_QUEUE_H_
