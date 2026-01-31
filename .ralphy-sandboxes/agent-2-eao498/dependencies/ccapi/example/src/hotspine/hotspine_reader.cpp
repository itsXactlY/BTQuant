#include "hotspine_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace HotSpine {

HotSpineReader::HotSpineReader(const std::string& shm_name) : shm_name_(shm_name) {
  if (!attachToSharedMemory()) {
    std::cerr << "HotSpineReader: Warning: Failed to attach to " << shm_name_ << " (Writer might not be ready yet)" << std::endl;
  }
}

HotSpineReader::~HotSpineReader() { detachFromSharedMemory(); }

bool HotSpineReader::isAttached() const { return layout_ != nullptr; }

bool HotSpineReader::pollLatestViewport(HotSpine::V3::ClusterColumn& out_viewport) {
  if (!isAttached()) {
    // Try to re-attach loosely if not attached
    if (!attachToSharedMemory()) return false;
  }

  // Lock-Free Reading Pattern
  uint64_t seq_start;

  // We want the latest HEAD.
  // Ideally we copy the slot at HEAD.

  // However, head_index moves.
  // We should probably read head_index first inside the lock check or optimistic read?
  // User logic 3.3: Write lock around updates.
  // If global_lock protects head_index AND the slot data (conceptually, global write lock),
  // then reading global_lock protects everything.

  int max_retries = 100;
  while (max_retries-- > 0) {
    seq_start = layout_->header.global_lock.read_begin();

    uint64_t head = layout_->header.head_index.load(std::memory_order_relaxed);
    // If head is 0, maybe no data yet?
    // head points to NEXT write slot? 3.3 says: `viewport = history[head % 1024]; write; head++`.
    // So the LATEST WRITTEN slot is `head - 1`.

    if (head == 0) {
      // specific logic for empty?
      // If we want to return something, maybe empty column.
      // But if head=0, no writes happened (unless wrapped around UINT64_MAX, unlikely).
      // Assume empty.
      if (layout_->header.global_lock.read_retry(seq_start)) continue;  // Retry if locked
      return false;
    }

    uint64_t target_idx = head - 1;
    const auto& src = layout_->history[target_idx % 1024];

    // Copy data
    // VolumeNode is POD but custom copy assignment is implicit default.
    // ClusterColumn is POD.
    memcpy(&out_viewport, &src, sizeof(HotSpine::V3::ClusterColumn));

    if (!layout_->header.global_lock.read_retry(seq_start)) {
      return true;  // Success
    }
    // else retry
  }

  return false;  // Contention too high
}

bool HotSpineReader::attachToSharedMemory() {
  if (layout_) return true;  // Already attached

  shm_fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0666);
  if (shm_fd_ == -1) {
    // Silent fail as reader often starts before writer
    return false;
  }

  struct stat st;
  if (fstat(shm_fd_, &st) == -1) {
    close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }

  if (st.st_size < static_cast<off_t>(sizeof(HotSpine::V3::SharedMemoryLayoutV3))) {
    // Size mismatch or not fully initialized
    close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }

  shm_ptr_ = mmap(nullptr, sizeof(HotSpine::V3::SharedMemoryLayoutV3), PROT_READ, MAP_SHARED, shm_fd_, 0);
  if (shm_ptr_ == MAP_FAILED) {
    close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }

  layout_ = static_cast<HotSpine::V3::SharedMemoryLayoutV3*>(shm_ptr_);

  // Check Magic
  if (layout_->header.magic != 0x42545133) {  // BTQ3
    std::cerr << "HotSpineReader: Magic mismatch: " << std::hex << layout_->header.magic << std::endl;
    detachFromSharedMemory();
    return false;
  }

  return true;
}

bool HotSpineReader::detachFromSharedMemory() {
  if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
    munmap(shm_ptr_, sizeof(HotSpine::V3::SharedMemoryLayoutV3));
  }
  if (shm_fd_ != -1) {
    close(shm_fd_);
  }

  shm_ptr_ = nullptr;
  shm_fd_ = -1;
  layout_ = nullptr;

  return true;
}

}  // namespace HotSpine
