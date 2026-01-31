#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../../../BTQ_Render_Engine/include/hotspine_layout_v3.hpp"

namespace HotSpine {

class HotSpineReader {
 public:
  explicit HotSpineReader(const std::string& shm_name = "/BTQU_V3");
  ~HotSpineReader();

  bool isAttached() const;

  // V3: Poll the latest viewport state
  // Returns true if successful, false if not attached
  bool pollLatestViewport(HotSpine::V3::ClusterColumn& out_viewport);

  HotSpineReader(const HotSpineReader&) = delete;
  HotSpineReader& operator=(const HotSpineReader&) = delete;

 private:
  std::string shm_name_;
  int shm_fd_{-1};
  void* shm_ptr_{nullptr};
  HotSpine::V3::SharedMemoryLayoutV3* layout_{nullptr};

  bool attachToSharedMemory();
  bool detachFromSharedMemory();
};

using HotSpineReaderPtr = std::shared_ptr<HotSpineReader>;

}  // namespace HotSpine
