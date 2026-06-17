// Phase 7 Adaptive LOD — STUB
// Real implementation deferred. This file was an orphan experiment that
// referenced types (LODLevel, LODRenderSettings, FootprintLOD, FootprintCell)
// that don't exist in the codebase. The stub satisfies the linker.

#include "rendering/footprint_lod.hpp"

#include <cstddef>

namespace BTQuant {

// All Phase 7.3 adaptive-LOD entry points are no-ops in this build.
// FootprintPanel and tpo_panel.cpp already implement their own per-row
// text-skip heuristic (cell_height_px < 4.0f).

}  // namespace BTQuant
