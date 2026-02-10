#ifndef INCLUDE_CCAPI_CPP_CCAPI_EVENT_DISPATCHER_H_
#define INCLUDE_CCAPI_CPP_CCAPI_EVENT_DISPATCHER_H_
#include <stddef.h>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include "concurrentqueue.h"

#include "ccapi_cpp/ccapi_logger.h"
#include "ccapi_cpp/ccapi_util_private.h"

namespace ccapi {

/**
 * Dispatches events from one or more Sessions through callbacks. EventDispatcher objects are optionally specified when Session objects are constructed. A
 * single EventDispatcher can be shared by multiple Session objects. The EventDispatcher provides an event-driven interface, generating callbacks from one or
 * more internal threads for one or more sessions.
 */

class EventDispatcher {
 public:
  explicit EventDispatcher(const int numDispatcherThreads = 1) : numDispatcherThreads(numDispatcherThreads) {
    CCAPI_LOGGER_FUNCTION_ENTER;
    CCAPI_LOGGER_TRACE("numDispatcherThreads = " + size_tToString(numDispatcherThreads));
    this->start();
    CCAPI_LOGGER_FUNCTION_EXIT;
  }

  ~EventDispatcher() {
    CCAPI_LOGGER_FUNCTION_ENTER;
    CCAPI_LOGGER_FUNCTION_EXIT;
  }

  void dispatch(const std::function<void()>& op) {
    CCAPI_LOGGER_FUNCTION_ENTER;
    if (this->shouldContinue.load()) {
      CCAPI_LOGGER_TRACE("start to dispatch an operation");
      this->queue.enqueue(op);
    } else {
      CCAPI_LOGGER_WARN("dispatching of events were paused");
    }
    CCAPI_LOGGER_FUNCTION_EXIT;
  }

  void start() {
    this->shouldContinue = true;
    for (size_t i = 0; i < numDispatcherThreads; i++) {
      this->dispatcherThreads.push_back(std::thread(&EventDispatcher::dispatch_thread_handler, this));
    }
  }

  void resume() { this->shouldContinue = true; }

  void pause() { this->shouldContinue = false; }

  void stop() {
    this->quit = true;
    for (auto& dispatcherThread : this->dispatcherThreads) {
      dispatcherThread.join();
    }
  }
#ifndef CCAPI_EXPOSE_INTERNAL

 private:
#endif
  void dispatch_thread_handler() {
    CCAPI_LOGGER_FUNCTION_ENTER;
    do {
      std::function<void()> op;
      if (this->queue.try_dequeue(op)) {
        op();
      } else {
        // No task available, sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } while (!this->quit);
    CCAPI_LOGGER_FUNCTION_EXIT;
  }

  size_t numDispatcherThreads{};
  std::atomic<bool> shouldContinue{};
  std::vector<std::thread> dispatcherThreads;
  moodycamel::ConcurrentQueue<std::function<void()>> queue;
  std::atomic<bool> quit{};
};

} /* namespace ccapi */
#endif  // INCLUDE_CCAPI_CPP_CCAPI_EVENT_DISPATCHER_H_
