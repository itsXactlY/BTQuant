#include "hotspine_reader_c_interface.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "hotspine_reader.hpp"

// Global registry to manage HotSpineReader instances
// This is needed because C++ objects can't be directly exposed to C
static std::unordered_map<HotSpineReaderHandle, std::unique_ptr<HotSpine::HotSpineReader>> reader_registry;
static std::mutex registry_mutex;
static uint64_t next_handle_id = 1;

extern "C" {

HotSpineReaderHandle hotspine_reader_create(const char* shm_name) {
  try {
    std::lock_guard<std::mutex> lock(registry_mutex);

    // Create the reader
    auto reader = std::make_unique<HotSpine::HotSpineReader>(shm_name);

    // Generate a unique handle
    uint64_t handle = next_handle_id++;

    // Store in registry
    reader_registry[reinterpret_cast<HotSpineReaderHandle>(handle)] = std::move(reader);

    return reinterpret_cast<HotSpineReaderHandle>(handle);
  } catch (const std::exception& e) {
    fprintf(stderr, "hotspine_reader_create failed: %s\n", e.what());
    return nullptr;
  }
}

void hotspine_reader_destroy(HotSpineReaderHandle reader) {
  if (!reader) return;

  std::lock_guard<std::mutex> lock(registry_mutex);
  reader_registry.erase(reader);
}

int hotspine_reader_poll_trade(HotSpineReaderHandle reader, HotSpineTradeC* trade) {
  // V3 Migration: pollTrade is deprecated/removed.
  // This interface needs to be updated to pollLatestViewport if still needed.
  return 0;
}

uint64_t hotspine_reader_get_lost_count(HotSpineReaderHandle reader) {
  if (!reader) {
    return 0;
  }

  // V3: Infinite Canvas doesn't track lost trades in this way.
  return 0;
}

int hotspine_reader_is_healthy(HotSpineReaderHandle reader) {
  if (!reader) {
    return 0;
  }

  try {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = reader_registry.find(reader);

    if (it == reader_registry.end()) {
      return 0;
    }

    return it->second->isAttached() ? 1 : 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "hotspine_reader_is_healthy failed: %s\n", e.what());
    return 0;
  }
}

}  // extern "C"
