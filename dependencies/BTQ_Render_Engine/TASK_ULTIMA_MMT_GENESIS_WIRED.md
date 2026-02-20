# TASK_ULTIMA_MMT_GENESIS — WIRED SPECIFICATION v2.0
> Rebuilt against the actual BTQuant codebase. Every task references a real file, real type, or real method. No invented paths.

---

## CODEBASE AUDIT SUMMARY

Before any phase begins, internalize this map of what already exists:

| Existing Asset | File | Status vs MMT Spec |
|---|---|---|
| `SpscRingBuffer<TradeData, 65536>` | `market_data_processor.hpp` | ✅ Exists. Used by `MarketDataProcessor`. Needs Tape consumer path added. |
| `TradeData`, `OrderBookSnapshot` | `data/core_types.hpp` (referenced, not uploaded) | ⚠️ Must verify `alignas(64)`. `BTQuant::OrderBookSnapshot` in `hotspine_data_bridge.hpp` is NOT `alignas(64)`. |
| `HotSpine::V3::ClusterColumn` | `hotspine_layout_v3.hpp` | ✅ Already `alignas(64)`. Has `VolumeNode rows[256]`. Ring buffer of 1024 columns. |
| `HotSpine::V3::VolumeNode` | `hotspine_layout_v3.hpp` | ✅ `alignas(16)`, 16 bytes. `buy_vol`, `sell_vol`, `trade_count`, `tpo_bits`. Matches `std430`. |
| `HotSpine::V3::SeqLock` | `hotspine_layout_v3.hpp` | ✅ Writer/reader seq-lock for `SharedMemoryLayoutV3`. |
| `GPUMemoryManager` | `vulkan_base_types.hpp` | ✅ `allocate_storage_buffer(size)` exists. Pool: 64MB. Use for SSBO. |
| `VulkanCore` | `vulkan_base_types.hpp` | ✅ `get_compute_queue()`, `begin_single_time_commands()`, `get_device()`. All needed Vulkan handles present. |
| `GPUMemoryManager::add_texture()` | `vulkan_base_types.hpp` | ✅ Returns `CachedTexture{descriptor_set, image_view, sampler}`. This IS the `ImGui_ImplVulkan_AddTexture()` wrapper. |
| `DashboardConfig::ThemeConfig` | `dashboard_config.hpp` | ⚠️ Wrong colors. `background_color = {0.1f,0.1f,0.1f,1.0f}`. Must be patched to `#0B0E11`. |
| `ChartMath::ViewPort`, `MapToScreen()` | `ChartMath.hpp` | ✅ Full viewport math. Use for DOM auto-centering and crosshair projection. |
| `InstrumentStore::updateVolumeProfile()` | `hotspine_data_bridge.hpp` | ✅ O(1) bin-index write. `VOL_PROFILE_BINS = 1000`. Needs CAS upgrade for thread safety. |
| `InstrumentStore::latest_snapshot` | `hotspine_data_bridge.hpp` | ✅ `std::atomic<OrderBookSnapshot*>`. This is the DOM's atomic book pointer. |
| `MarketDataProcessor::get_active_orderbook()` | `market_data_processor.hpp` | ✅ Returns `OrderBookSnapshot*` via `memory_order_acquire`. DOM reads from here. |
| `VolumeNode::tpo_bits` | `hotspine_layout_v3.hpp` | ✅ `uint16_t` bitmask — 16 half-hour brackets. TPOEngine writes here directly. |
| `shader_spirv.hpp` | `shader_spirv.hpp` | ⚠️ Has vertex + fragment SPIR-V only. Must ADD `LOB_HEATMAP_COMPUTE_SPIRV[]`. |
| `TaskScheduler` | `task_scheduler.hpp` | ✅ `calculate_cumulative_volume_delta_async()` exists. Wire to Phase 5 CVD. |
| `PerformanceMonitor` + `g_performance_monitor` | `performance_monitor.hpp` | ✅ Frame timing, FPS, extern global. Wire to Phase 7 TSC telemetry. |
| `CacheManager` | `cache_manager.hpp` | ✅ LRU cache for profiles/indicators. Wire to Phase 5 cluster binning cache. |
| `TapeComponent`, `DOMSurfacePanel` | `vulkan_dashboard_advanced.hpp` | ⚠️ Forward-declared only. Implementations are the deliverables of Phases 3–4. |
| `MemoryArena` | — | ❌ Does not exist. Phase 0.3 is a greenfield addition. |
| `lob_heatmap.comp` SPIR-V | — | ❌ Does not exist. Phase 1 deliverable. |
| `std::atomic<int64_t> g_cvd` | — | ❌ Does not exist. Phase 5 deliverable. |
| `std::atomic<double> g_crosshair_price` | — | ❌ Does not exist. Phase 7 deliverable. |

---

## DEPENDENCY GRAPH

```
Phase 0 (SPSC spine, alignas, arena)
    └─► Phase 1 (SSBO mapped from arena, ClusterColumn→GPU)
            └─► Phase 2 (ImGui theme + docking, heatmap texture hook)
                    ├─► Phase 3 (Tape reads SpscRingBuffer, ImGuiListClipper)
                    ├─► Phase 4 (DOM reads atomic_book + heatmap texture)
                    │       └─► Phase 5 (Footprint/CVD feeds DOM clusters)
                    │               └─► Phase 6 (TPO writes tpo_bits in VolumeNode)
                    └─► Phase 7 (Global sync: crosshair atomic, frame pacer, TSC telemetry)
```

**Hard rule:** Do not start Phase N+1 until all `[ACCEPTANCE]` criteria for Phase N pass.

---

## SELF-FEED LOOP PROTOCOL

At the end of each phase, re-read this document from the top and verify:

1. All `[ACCEPTANCE]` criteria for the completed phase are ✅.
2. No new code broke the interface contracts of prior phases.
3. If a criterion is ❌, fix it before proceeding.
4. Append a `## PHASE N AUDIT LOG` section to this document after each phase completes, recording actual file diffs, any contract deviations, and the timestamp.

---

## PHASE 0 — THE BARE METAL SPINE
**Target Files:**
- `include/data/core_types.hpp` — modify
- `include/threading/lockfree_queue.hpp` — verify/modify
- `include/memory/memory_arena.hpp` — **NEW FILE**

### 0.1 — Harden `TradeData` and `OrderBookSnapshot`

Open `include/data/core_types.hpp`. Add the following static assertions immediately after each struct definition:

```cpp
// TradeData must be padded to eliminate false sharing across the SpscRingBuffer slots
struct alignas(64) TradeData {
    double   price;           // 8
    double   volume;          // 8
    uint64_t timestamp_us;    // 8
    uint32_t symbol_id;       // 4
    uint8_t  side;            // 0=Buy, 1=Sell
    uint8_t  _pad[31];        // pad to 64 bytes
};
static_assert(sizeof(TradeData) == 64);
static_assert(std::is_trivial_v<TradeData>);
static_assert(std::is_standard_layout_v<TradeData>);

// BTQuant::OrderBookSnapshot (hotspine_data_bridge.hpp) currently lacks alignas(64).
// Add it. The struct is 200*2*16 + 24 = 3224 bytes → pad to 3264 (next multiple of 64).
// Do NOT change HotSpine::V3::ClusterColumn — it is already alignas(64).
```

**CONFLICT NOTE:** `BTQuant::OrderBookSnapshot` in `hotspine_data_bridge.hpp` and `HotSpine::OrderBookSnapshot` in `hotspine_reader.hpp` are two different types. The DOM uses `BTQuant::OrderBookSnapshot`. Apply `alignas(64)` only to `BTQuant::OrderBookSnapshot`.

### 0.2 — Verify `SpscRingBuffer` in `include/threading/lockfree_queue.hpp`

The ring buffer is already used as `SpscRingBuffer<TradeData, 65536>` in `MarketDataProcessor`. Verify the implementation satisfies:

```cpp
template<typename T, size_t Capacity>
// REQUIRED: Capacity must be power-of-2. Static assert this.
static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

// REQUIRED: writer uses std::memory_order_release on head_
// REQUIRED: reader uses std::memory_order_acquire on head_, memory_order_release on tail_
// REQUIRED: index masking uses (idx & (Capacity - 1)), NOT modulo
```

If any of these fail, patch the implementation. Do not change the template signature — `MarketDataProcessor` depends on it.

**ADD a `peek(size_t n, T* out) -> size_t` method** — non-consuming read of the last N items. The Tape panel (`src/components/tape_panel.cpp`) uses this to read without draining the queue that `process_queues()` also drains:

```cpp
// Returns up to `n` most-recently-written items WITHOUT advancing tail.
// Thread-safe for single reader. Copies items into `out[0..return_value-1]`.
size_t peek(size_t n, T* out) const noexcept;
```

### 0.3 — `MemoryArena` (New File: `include/memory/memory_arena.hpp`)

```cpp
namespace BTQuant {

class MemoryArena {
public:
    // Single 1GB reservation on construction. mmap(MAP_ANON) on Linux,
    // VirtualAlloc(MEM_RESERVE|MEM_COMMIT) on Windows.
    explicit MemoryArena(size_t size_bytes = 1ULL << 30);
    ~MemoryArena();

    // O(1) bump allocator for the hot path. Returns aligned pointer.
    // alignment must be power-of-2. Returns nullptr if arena is exhausted.
    [[nodiscard]] void* acquire(size_t bytes, size_t alignment = 64) noexcept;

    // Returns memory to the lock-free free-list (O(1) CAS).
    // Only call with pointers originally returned from acquire().
    void release(void* ptr, size_t bytes) noexcept;

    size_t used_bytes() const noexcept;
    size_t capacity_bytes() const noexcept;

private:
    uint8_t*                   base_     = nullptr;
    size_t                     capacity_ = 0;
    std::atomic<size_t>        offset_   = 0;   // bump pointer

    // Free-list: intrusive stack via CAS
    struct FreeNode { FreeNode* next; size_t size; };
    std::atomic<FreeNode*>     free_head_ = nullptr;
};

// Global singleton — initialized in main() before any subsystem
extern MemoryArena g_arena;

} // namespace BTQuant
```

**Implementation rules:**
- `acquire()`: Try free-list first (CAS pop). If no matching block, bump `offset_` atomically with `compare_exchange_weak`. Alignment: round up `offset_` to next multiple.
- `release()`: CAS push onto `free_head_`. Zero the memory on debug builds only.
- Do NOT replace `VulkanCore`'s `MemoryPool` (which uses `VkDeviceMemory`) with `g_arena`. The arena is CPU-side only, for `TradeData` arrays, `OrderBookSnapshot` buffers, and analytics structs.

### 0.4 — Wire `g_arena` into `MarketDataProcessor`

In `market_data_processor.hpp`, the `symbol_trades_` map currently uses heap `std::vector<TradeData>`. This is the one place where the arena must be threaded in:

```cpp
// Replace:
std::unordered_map<uint32_t, std::vector<TradeData>> symbol_trades_;

// With a fixed-capacity ring per symbol backed by g_arena:
// Allocate during MarketDataProcessor construction, not at first trade.
struct TradeRing {
    TradeData* buf   = nullptr;  // g_arena.acquire(1024 * sizeof(TradeData), 64)
    uint32_t   head  = 0;
    uint32_t   count = 0;
    static constexpr uint32_t CAP = 1024;
};
TradeRing symbol_rings_[MAX_SYMBOLS];  // stack allocation, 100 symbols
```

**[ACCEPTANCE — Phase 0]**
- [ ] `static_assert(sizeof(TradeData) == 64)` compiles without error.
- [ ] `static_assert(std::is_trivial_v<TradeData>)` passes.
- [ ] `SpscRingBuffer::peek()` compiles and unit-test returns last N items without modifying tail.
- [ ] `MemoryArena` constructed with 1GB. `g_arena.used_bytes()` starts at 0.
- [ ] `MemoryArena::acquire(64, 64)` returns a 64-byte-aligned pointer on first call.
- [ ] `MarketDataProcessor` constructor allocates all `TradeRing` buffers from `g_arena` at startup without `new`.

---

## PHASE 1 — VULKAN COMPUTE & GPU HEATMAP
**Target Files:**
- `src/vulkan/lob_heatmap_compute_pipeline.cpp` — **NEW FILE**
- `src/vulkan/ssbo_snapshot_updater.cpp` — **NEW FILE**
- `include/shader_spirv.hpp` — modify (add compute SPIR-V)

### 1.1 — SSBO Layout Contract

The GPU buffer mirrors `HotSpine::V3::SharedMemoryLayoutV3::history[1024]`. Each `ClusterColumn` contains `VolumeNode rows[256]`. The SSBO element type in the compute shader must match `VolumeNode` exactly:

```glsl
// lob_heatmap.comp — std430 binding
struct VolumeNode {
    float    buy_vol;      // offset 0
    float    sell_vol;     // offset 4
    uint16_t trade_count;  // offset 8  — declared as uint in GLSL (std430 packs uint16 as uint32)
    uint16_t tpo_bits;     // offset 10
    uint8_t  padding[4];   // offset 12
};
// sizeof = 16 bytes. Matches HotSpine::V3::VolumeNode exactly. ✅
```

**GLSL binding:**
```glsl
layout(set = 0, binding = 0, std430) readonly buffer ClusterSSBO {
    VolumeNode nodes[];  // Flat array: column * 256 + row
} ssbo;
layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D heatmap_out;
```

### 1.2 — SSBO Allocation via Existing `GPUMemoryManager`

In `ssbo_snapshot_updater.cpp`, do NOT call `vkCreateBuffer` directly. Use the existing pool:

```cpp
// In SsboSnapshotUpdater::initialize(GPUMemoryManager& mem, VulkanCore& core):
ssbo_alloc_ = mem.allocate_storage_buffer(
    1024 * 256 * sizeof(HotSpine::V3::VolumeNode)  // = 1024 * 4096 = 4MB
);
// ssbo_alloc_.mapped_ptr is the persistent CPU-side write pointer.
// GPUMemoryManager allocates storage pool with HOST_VISIBLE | HOST_COHERENT.
// No explicit flush needed — coherent mapping.
```

The `GPUMemoryManager::MemoryPool` constructor takes `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` for the storage pool. Verify this in `vulkan_base_types.hpp` before proceeding. If the storage pool was created as `DEVICE_LOCAL` only, add a dedicated `HOST_COHERENT` storage pool for the SSBO.

### 1.3 — Compute Pipeline Construction (`lob_heatmap_compute_pipeline.cpp`)

```cpp
class LobHeatmapComputePipeline {
public:
    void initialize(VkDevice device, VkDescriptorPool pool);
    void dispatch(VkCommandBuffer cmd, uint32_t columns, uint32_t rows);
    VkImageView get_output_image_view() const;
    void destroy(VkDevice device);

private:
    VkPipeline        pipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout  layout_         = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout_  = VK_NULL_HANDLE;
    VkDescriptorSet   descriptor_set_ = VK_NULL_HANDLE;
    VkImage           output_image_   = VK_NULL_HANDLE;
    VkImageView       output_view_    = VK_NULL_HANDLE;
    VkDeviceMemory    output_memory_  = VK_NULL_HANDLE;
};
```

**Workgroup configuration** — wired to actual data dimensions:
- `ClusterColumn history[1024]` → X axis = 1024 columns
- `VolumeNode rows[256]` → Y axis = 256 rows
- `local_size_x = 16, local_size_y = 16`
- `vkCmdDispatch(1024/16, 256/16, 1)` = `vkCmdDispatch(64, 16, 1)`

### 1.4 — Color Mapping in Compute Shader

Map `buy_vol` and `sell_vol` to the MMT gradient. The normalization denominator comes from the `ClusterColumn` with highest `total_volume` (computed in a prior reduce pass or passed as push constant):

```glsl
layout(push_constant) uniform PushConstants {
    float max_volume;   // Running max from CPU side
    float alpha;        // Overall heatmap opacity (0.3 default)
} pc;

// Per-node color logic:
float bid_intensity = node.buy_vol  / max(pc.max_volume, 1.0);
float ask_intensity = node.sell_vol / max(pc.max_volume, 1.0);
vec4 color = vec4(0.0); // #0B0E11 void
if (bid_intensity > 0.0) color = mix(color, vec4(0.0, 0.9, 0.4, pc.alpha), bid_intensity); // Neon Mint
if (ask_intensity > 0.0) color = mix(color, vec4(1.0, 0.5, 0.0, pc.alpha), ask_intensity); // Orange
imageStore(heatmap_out, ivec2(gl_GlobalInvocationID.xy), color);
```

### 1.5 — Pipeline Barrier

Before handing the texture to ImGui, issue the transition inside `VulkanCore::RecordCommandBuffer()`:

```cpp
// After vkCmdDispatch, before main render pass:
VkImageMemoryBarrier barrier{};
barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
barrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
barrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
barrier.image               = heatmap_pipeline_.get_output_image();
// ... subresource range
vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    0, 0, nullptr, 0, nullptr, 1, &barrier);
```

### 1.6 — ImGui Texture Registration

Use the **existing** `GPUMemoryManager::add_texture()` — it wraps `ImGui_ImplVulkan_AddTexture()`:

```cpp
// In LobHeatmapComputePipeline::initialize(), after creating VkSampler:
CachedTexture tex = mem_manager.add_texture(
    output_view_,
    sampler_,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL  // post-barrier layout
);
heatmap_imgui_descriptor_ = tex.descriptor_set;
// Store tex.im_texture_id for ImGui::Image() calls in DOM and Chart panels.
```

**[ACCEPTANCE — Phase 1]**
- [ ] `SsboSnapshotUpdater::update()` copies one full `SharedMemoryLayoutV3::history` slice into the SSBO `mapped_ptr` in under 500μs (measure with `__rdtsc()`).
- [ ] `vkCmdDispatch(64, 16, 1)` executes without validation layer errors.
- [ ] `GPUMemoryManager::add_texture()` returns a non-null `descriptor_set`.
- [ ] `ImGui::Image((ImTextureID)tex.im_texture_id, size)` renders a colored heatmap (not a black rect).
- [ ] Pipeline barrier: Vulkan validation layer produces zero layout-transition warnings.

---

## PHASE 2 — DEEP VOID AESTHETIC & DOCKING
**Target Files:**
- `src/ui/unified_theme_system.cpp` — **NEW FILE** (replaces scattered `DashboardConfig` color patches)
- `include/dashboard_config.hpp` — modify `ThemeConfig` defaults

### 2.1 — Patch `ThemeConfig` Defaults

In `dashboard_config.hpp`, `ThemeConfig` struct:

```cpp
// REPLACE:
ColorRGBA background_color = {0.1f, 0.1f, 0.1f, 1.0f};

// WITH (MMT Deep Void — #0B0E11):
ColorRGBA background_color = {0.043f, 0.055f, 0.067f, 1.0f};

// ADD:
ColorRGBA child_bg_color   = {0.082f, 0.098f, 0.118f, 1.0f}; // #15191E
ColorRGBA header_bg_color  = {0.055f, 0.067f, 0.082f, 1.0f}; // #0E111A
```

### 2.2 — `unified_theme_system.cpp` — ImGui Style Application

This file owns the single call point for all ImGui style mutations. Call `UnifiedThemeSystem::apply()` once, immediately after `ImGui::CreateContext()` inside `VulkanCore::init_imgui()`.

```cpp
namespace BTQuant {
class UnifiedThemeSystem {
public:
    // Reads from DashboardConfig::ThemeConfig, applies to ImGui::GetStyle()
    static void apply(const RenderEngine::ThemeConfig& cfg);
    static void apply_mmt_void();  // Hardcoded MMT preset, bypasses config

private:
    static void apply_borders(ImGuiStyle& s);
    static void apply_colors(ImGuiStyle& s, const RenderEngine::ThemeConfig& cfg);
    static void apply_fonts();
};
} // namespace BTQuant
```

**Exact style mutations in `apply_borders()`:**
```cpp
s.WindowBorderSize  = 0.0f;
s.ChildBorderSize   = 0.0f;
s.FrameBorderSize   = 0.0f;
s.WindowRounding    = 0.0f;
s.FrameRounding     = 0.0f;
s.PopupRounding     = 0.0f;
s.TabRounding       = 0.0f;
s.ScrollbarRounding = 0.0f;
s.GrabRounding      = 0.0f;
```

**Exact font config in `apply_fonts()`:**
```cpp
ImFontConfig cfg;
cfg.OversampleH = 4;
cfg.OversampleV = 4;
cfg.PixelSnapH  = false;
// Path: embed JetBrains Mono TTF as a static uint8_t array in font_data.hpp
// to avoid runtime file dependency.
ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
    (void*)JetBrainsMonoTTF, sizeof(JetBrainsMonoTTF), 11.0f, &cfg);
// Merge FontAwesome 6 icons into same atlas:
cfg.MergeMode = true;
cfg.GlyphMinAdvanceX = 11.0f;
static const ImWchar fa_ranges[] = { 0xe000, 0xf8ff, 0 };
ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
    (void*)FontAwesome6TTF, sizeof(FontAwesome6TTF), 11.0f, &cfg, fa_ranges);
```

### 2.3 — Programmatic Docking

In `VulkanDashboard::init_components()`, after ImGui dockspace creation:

```cpp
// One-time layout setup (guard with static bool first_time)
ImGuiID dockspace_id = ImGui::GetID("MMT_DockSpace");
ImGui::DockBuilderRemoveNode(dockspace_id);
ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

ImGuiID left, center, right;
ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left,  0.08f, &left,   &center);
ImGui::DockBuilderSplitNode(center,        ImGuiDir_Right, 0.35f, &right,  &center);

ImGuiID right_top, right_bottom;
ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.40f, &right_bottom, &right_top);

ImGui::DockBuilderDockWindow("Sidebar",    left);
ImGui::DockBuilderDockWindow("Charts",     center);
ImGui::DockBuilderDockWindow("DOM",        right_top);
ImGui::DockBuilderDockWindow("Tape",       right_bottom);
ImGui::DockBuilderFinish(dockspace_id);
```

**[ACCEPTANCE — Phase 2]**
- [ ] `ImGui::GetStyle().WindowBorderSize == 0.0f` at runtime.
- [ ] `ImGui::GetStyle().Colors[ImGuiCol_WindowBg]` equals `{0.043f, 0.055f, 0.067f, 1.0f}`.
- [ ] JetBrains Mono 11px renders in the terminal/tape panel without atlas overflow.
- [ ] On first launch, all four dock windows (Sidebar, Charts, DOM, Tape) appear in the correct quadrants without user drag.

---

## PHASE 3 — THE TAPE
**Target Files:**
- `src/components/tape_panel.cpp` — **NEW FILE** (implements `TapeComponent` declared in `vulkan_dashboard_advanced.hpp`)

### 3.1 — Binding to the Ring Buffer

`TapeComponent` must NOT drain `MarketDataProcessor::trade_queue_` — that is `process_queues()`'s job. Instead, use the `SpscRingBuffer::peek()` method added in Phase 0.2.

```cpp
class TapeComponent : public UIComponent {
public:
    explicit TapeComponent(MarketDataProcessor& mdp,
                           const glm::vec2& pos, const glm::vec2& size);
    void update(float dt) override;
    void render_gui() override;
    void clear_data() override;
    void initialize_vulkan_resources(VulkanCore*) override {}

private:
    MarketDataProcessor& mdp_;
    std::array<TradeData, 500> recent_trades_;  // From g_arena in ctor
    size_t           trade_count_ = 0;
    float            size_filter_ = 0.0f;       // UI slider threshold
    uint32_t         active_symbol_id_ = 0;
};
```

In `update()`, call:
```cpp
trade_count_ = mdp_.get_trade_ring(active_symbol_id_).peek(500, recent_trades_.data());
```
(The `TradeRing::peek()` wraps the ring buffer in `MarketDataProcessor`, added in Phase 0.4.)

### 3.2 — Virtualized List

```cpp
void TapeComponent::render_gui() {
    ImGui::Begin("Tape");
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(trade_count_));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const TradeData& t = recent_trades_[trade_count_ - 1 - i]; // newest first
            if (t.volume < size_filter_) continue;
            render_tape_row(t, i);
        }
    }
    ImGui::End();
}
```

### 3.3 — Alpha Percentile Mapping

```cpp
// Computed once per update(), not per row:
float compute_size_percentile(const TradeData& t) const {
    // Count how many of recent_trades_[0..499] have volume <= t.volume
    // Return ratio in [0,1]. Map to alpha [0.05, 0.50].
    size_t count = 0;
    for (size_t i = 0; i < std::min(trade_count_, size_t(500)); ++i)
        count += (recent_trades_[i].volume <= t.volume) ? 1 : 0;
    return 0.05f + 0.45f * (static_cast<float>(count) / 500.0f);
}
```

Use the alpha as `ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(r,g,b,alpha*255))`.

### 3.4 — Aggression Colors

```cpp
// side == 0 → Market Buy (Ask hit) → Neon Mint
const ImVec4 BUY_COLOR  = ImVec4(0.0f, 0.90f, 0.40f, 1.0f);  // #00E566
// side == 1 → Market Sell (Bid hit) → Crimson
const ImVec4 SELL_COLOR = ImVec4(0.90f, 0.10f, 0.15f, 1.0f);  // #E61926
```

### 3.5 — Sweep Bracket Detection

```cpp
// In render_tape_row(), compare with previous row:
if (i + 1 < trade_count_) {
    const TradeData& prev = recent_trades_[trade_count_ - i - 2];
    uint64_t delta_us = t.timestamp_us - prev.timestamp_us;
    bool is_sweep = (delta_us < 50000) && (t.price != prev.price);
    if (is_sweep) {
        // Draw 1px white vertical bracket on left edge of row
        ImVec2 p_min = ImGui::GetItemRectMin();
        ImVec2 p_max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddLine(
            {p_min.x, p_min.y}, {p_min.x, p_max.y},
            IM_COL32(255, 255, 255, 200), 1.0f);
    }
}
```

### 3.6 — Size Filter Slider

```cpp
ImGui::SliderFloat("Min Size", &size_filter_, 0.0f, 100.0f, "%.1f");
```

**[ACCEPTANCE — Phase 3]**
- [ ] Tape renders at 60fps with 500 visible rows and zero frame drops (verify via `g_performance_monitor.get_frame_time_ms() < 16.67f`).
- [ ] Buy rows are Neon Mint, Sell rows are Crimson. Verify with 10 manual test trades.
- [ ] Sweep brackets appear when two consecutive trades are within 50ms and at different prices.
- [ ] Setting size filter to 50 hides rows with volume < 50.

---

## PHASE 4 — THE DOM (DEPTH OF MARKET)
**Target Files:**
- `src/components/dom_surface_panel.cpp` — **NEW FILE** (implements `DOMSurfacePanel` + `HeatmapComponent` from `vulkan_dashboard_advanced.hpp`)

### 4.1 — Table Layout

```cpp
// 5-column transparent table
ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, IM_COL32(0,0,0,0));
ImGui::PushStyleColor(ImGuiCol_TableBorderLight,  IM_COL32(0,0,0,0));
if (ImGui::BeginTable("DOM", 5, ImGuiTableFlags_NoSavedSettings)) {
    ImGui::TableSetupColumn("Buys",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Asks",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Bids",  ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    // ... render rows
    ImGui::EndTable();
}
```

### 4.2 — Heatmap Backdrop

The heatmap texture is registered in Phase 1.6 as `CachedTexture::im_texture_id`. Render it before the table using the same draw list:

```cpp
ImDrawList* dl = ImGui::GetWindowDrawList();
ImVec2 dom_min = ImGui::GetCursorScreenPos();
ImVec2 dom_max = {dom_min.x + panel_width_, dom_min.y + panel_height_};
dl->AddImage(
    (ImTextureID)heatmap_imgui_descriptor_,
    dom_min, dom_max,
    {0,0}, {1,1},
    IM_COL32(255,255,255,180)  // 70% opacity overlay
);
```

### 4.3 — Auto-Centering

Use `ChartMath::MapToScreen()` to convert `live_price` to a Y pixel offset. The auto-center threshold is 5 ticks:

```cpp
void DomSurfacePanel::update_center(double live_price, double tick_size) {
    double center_price = viewport_.minPrice +
        (viewport_.maxPrice - viewport_.minPrice) * 0.5;
    double delta = std::abs(live_price - center_price);
    if (delta > tick_size * 5.0) {
        // Smoothly interpolate: move 10% of delta per frame
        double target_shift = (live_price - center_price) * 0.10;
        viewport_.minPrice += target_shift;
        viewport_.maxPrice += target_shift;
        viewport_.clamp(); // ChartMath::ViewPort::clamp() already exists
    }
}
```

`live_price` is read from `MarketDataProcessor::get_latest_price(symbol_id)` — O(1), lock-free.

### 4.4 — Instanced Volume Bars

Do NOT call `ImGui::AddRectFilled()` in a loop. Build a vertex array:

```cpp
// One rect = 2 triangles = 6 vertices. Pre-allocate on stack.
const BTQuant::OrderBookSnapshot* book =
    mdp_.get_active_orderbook(active_symbol_id_);
if (!book) return;

// Construct vertex buffer for all bid bars in one call:
std::array<ImDrawVert, 200 * 6> verts;
uint32_t vert_count = 0;
for (int i = 0; i < book->bids_count; ++i) {
    // ... fill verts[vert_count..vert_count+5] with rect geometry
    vert_count += 6;
}
dl->AddDrawCmd(); // Flush prior commands
// Add raw vtx/idx directly via PrimReserve:
dl->PrimReserve(vert_count, vert_count);
memcpy(dl->_VtxWritePtr, verts.data(), vert_count * sizeof(ImDrawVert));
dl->_VtxWritePtr += vert_count;
dl->_IdxWritePtr += vert_count; // Indices are sequential for non-indexed quads
```

### 4.5 — Execution Bubbles

Pull from `TapeComponent::recent_trades_` (shared via pointer passed in constructor). For each trade where `abs(trade.price - dom_price_level) < tick_size * 0.5`, draw a circle on the Price column:

```cpp
ImVec2 price_col_x = ...; // Center X of Price column
float y = ChartMath::MapToScreen({0, (float)trade.price}, viewport_,
                                  0, panel_height_).y + dom_min.y;
dl->AddCircleFilled({price_col_x, y}, 4.0f,
    trade.side == 0 ? IM_COL32(0,230,102,200) : IM_COL32(230,25,38,200));
```

**[ACCEPTANCE — Phase 4]**
- [ ] DOM table renders 200 bid + 200 ask levels without performance regression.
- [ ] Heatmap texture is visible behind the transparent table.
- [ ] Scrolling 10+ ticks away from live price triggers auto-center within 500ms (10% lerp per frame × 30fps).
- [ ] Volume bars are drawn as a single `AddDrawCmd()` call per side (verify via ImGui draw call counter).
- [ ] Execution bubbles appear at the correct price level within one render frame of the trade arriving in the Tape.

---

## PHASE 5 — FOOTPRINT & VOLUME PROFILE
**Target Files:**
- `src/analytics/cluster_engine.cpp` — **NEW FILE**
- `src/components/footprint_panel.cpp` — **NEW FILE**
- `src/components/volume_profile_panel.cpp` — **NEW FILE** (wraps existing `InstrumentStore::updateVolumeProfile()`)

### 5.1 — O(1) Cluster Binning

`InstrumentStore::updateVolumeProfile()` already implements O(1) bin-index math:
```cpp
size_t bin = static_cast<size_t>((price - vol_profile_min_price_) / vol_profile_bin_size_);
```
**The ClusterEngine extends this** to track buy/sell separately using `HotSpine::V3::VolumeNode`:

```cpp
class ClusterEngine {
public:
    void initialize(double day_low, double tick_size, size_t max_bins = 1000);

    // O(1). Called for every trade from MarketDataProcessor::process_queues().
    void ingest(const TradeData& t) noexcept;

    const HotSpine::V3::VolumeNode* get_bins() const noexcept { return bins_; }
    size_t get_bin_count() const noexcept { return bin_count_; }
    double get_day_low() const noexcept { return day_low_; }
    double get_tick_size() const noexcept { return tick_size_; }

    // CVD: updated atomically in ingest()
    int64_t get_cvd() const noexcept {
        return cvd_.load(std::memory_order_acquire);
    }

private:
    HotSpine::V3::VolumeNode* bins_      = nullptr; // From g_arena
    size_t                    bin_count_ = 0;
    double                    day_low_   = 0.0;
    double                    tick_size_ = 0.0;
    std::atomic<int64_t>      cvd_       = 0;  // Phase 5.5 — CVD
};
```

### 5.2 — CAS Volume Accumulation

`HotSpine::V3::VolumeNode::buy_vol` and `sell_vol` are `float`. Standard `atomic<float>` CAS:

```cpp
void ClusterEngine::ingest(const TradeData& t) noexcept {
    size_t bin = static_cast<size_t>((t.price - day_low_) / tick_size_);
    if (bin >= bin_count_) return;

    auto* node = &bins_[bin];
    auto& target = t.side == 0 ? node->buy_vol : node->sell_vol;

    // CAS loop on float (stored as uint32_t for atomic ops)
    auto* raw = reinterpret_cast<std::atomic<uint32_t>*>(&target);
    uint32_t old_bits, new_bits;
    do {
        old_bits = raw->load(std::memory_order_relaxed);
        float new_val = std::bit_cast<float>(old_bits) + static_cast<float>(t.volume);
        new_bits = std::bit_cast<uint32_t>(new_val);
    } while (!raw->compare_exchange_weak(old_bits, new_bits,
                 std::memory_order_release, std::memory_order_relaxed));

    // CVD: atomic add (Phase 5.5)
    int64_t delta = static_cast<int64_t>(t.volume * 100); // Scale to avoid float
    if (t.side == 0) cvd_.fetch_add( delta, std::memory_order_relaxed);
    else             cvd_.fetch_add(-delta, std::memory_order_relaxed);
}
```

### 5.3 — Diagonal Imbalance Detection

In `footprint_panel.cpp`, after reading the cluster array:

```cpp
// Compare bin[i].buy_vol vs bin[i+1].sell_vol
for (size_t i = 0; i + 1 < engine.get_bin_count(); ++i) {
    float bid_vol = bins[i].buy_vol;
    float ask_vol = bins[i+1].sell_vol;
    if (ask_vol > 0 && bid_vol / ask_vol > 3.0f) {
        // Draw 2px boundary box around row i
        ImVec2 rect_min = ...; ImVec2 rect_max = ...;
        dl->AddRect(rect_min, rect_max, IM_COL32(255,200,0,220), 0.0f, 0, 2.0f);
    }
}
```

### 5.4 — Dynamic POC

Track during `ClusterEngine::ingest()`. No sort required:

```cpp
// In ClusterEngine, add:
std::atomic<size_t> poc_bin_{0};

// In ingest(), after updating buy/sell vol:
float total = bins_[bin].buy_vol + bins_[bin].sell_vol;
float poc_total = bins_[poc_bin_.load(std::memory_order_relaxed)].buy_vol
                + bins_[poc_bin_.load(std::memory_order_relaxed)].sell_vol;
if (total > poc_total) {
    poc_bin_.store(bin, std::memory_order_release);
}
```

### 5.5 — CVD Panel

Wire `ClusterEngine::get_cvd()` to a running line chart in `volume_profile_panel.cpp`. Use `CacheManager` (already exists in `cache_manager.hpp`) to cache profile snapshots per bar timestamp:

```cpp
// After each bar close, cache the snapshot:
cache_manager_.cacheAggregatedData(
    symbol_id, timeframe, bar_timestamp,
    "cvd", {static_cast<double>(engine.get_cvd())});
```

**[ACCEPTANCE — Phase 5]**
- [ ] `ClusterEngine::ingest()` benchmarks at < 200ns per call (measure with `__rdtsc()`).
- [ ] Two concurrent threads calling `ingest()` with overlapping `symbol_id`s produce no torn `buy_vol` reads (verified via ThreadSanitizer).
- [ ] Diagonal imbalance boxes appear at known 3:1 ratio levels in a replayed dataset.
- [ ] POC bin index is always the bin with highest `buy_vol + sell_vol` (spot-check 100 trades).
- [ ] CVD increments on buys, decrements on sells, observable in real-time in the profile panel.

---

## PHASE 6 — MARKET PROFILE (TPO)
**Target Files:**
- `src/analytics/tpoengine.cpp` — **NEW FILE**
- `src/components/tpo_panel.cpp` — **NEW FILE**

### 6.1 — TPO Bracket Mapping

The 30-min brackets map directly to `VolumeNode::tpo_bits` (uint16_t, 16 bits = 16 half-hour slots, covering 8 hours = one full session):

```cpp
// Map UTC microsecond timestamp to bracket index (0–15)
uint8_t TPOEngine::timestamp_to_bracket(uint64_t ts_us) {
    // Minutes since session open (session_open_us_ set at day start)
    uint64_t minutes = (ts_us - session_open_us_) / (60ULL * 1000000ULL);
    return static_cast<uint8_t>(std::min(minutes / 30, uint64_t(15)));
}
// Set bit in VolumeNode::tpo_bits atomically:
auto* raw = reinterpret_cast<std::atomic<uint16_t>*>(&bins_[bin].tpo_bits);
raw->fetch_or(uint16_t(1 << bracket), std::memory_order_relaxed);
```

The ASCII character mapping (A=bracket 0, B=bracket 1, etc.) used for rendering:
```cpp
char bracket_to_char(uint8_t bracket) {
    return bracket < 26 ? ('A' + bracket) : ('a' + bracket - 26);
}
```

### 6.2 — TPO Matrix Accumulation

```cpp
// TPOEngine holds a flat view into ClusterEngine's bins_:
// tpo_bits already updated in Phase 5 CAS ingest. No separate storage needed.
// TPOPanel reads directly from ClusterEngine::get_bins().
```

### 6.3 — Raw Text Rendering

```cpp
// In tpo_panel.cpp, iterate bins from high price to low:
for (size_t i = engine.get_bin_count(); i-- > 0;) {
    const VolumeNode& node = bins[i];
    double price = engine.get_day_low() + i * engine.get_tick_size();
    float y = ChartMath::MapToScreen({0, (float)price}, viewport_,
                                     0, panel_height_).y + panel_origin_.y;
    float x = panel_origin_.x;
    for (int bit = 0; bit < 16; ++bit) {
        if (node.tpo_bits & (1 << bit)) {
            char c[2] = { bracket_to_char(bit), '\0' };
            dl->AddText({x + bit * 8.0f, y}, IM_COL32(180,200,255,255), c);
        }
    }
}
```

`AddText()` with a pre-loaded glyph atlas (Phase 2.2) is the fastest text path — no layout overhead.

### 6.4 — Single Print Detection

```cpp
bool is_single_print = (__builtin_popcount(node.tpo_bits) == 1);
bool adjacent_above = (i+1 < bin_count && __builtin_popcount(bins[i+1].tpo_bits) > 1);
bool adjacent_below = (i > 0           && __builtin_popcount(bins[i-1].tpo_bits) > 1);
if (is_single_print && adjacent_above && adjacent_below) {
    dl->AddRectFilled(row_min, row_max, IM_COL32(74, 144, 255, 60));
}
```

### 6.5 — Value Area (VA) Math

```cpp
// Called once per bar close. bins sorted by total popcount descending.
// Expand outward from POC until 68% of total bracket appearances enclosed.
size_t total_chars = 0;
for (size_t i = 0; i < bin_count; ++i)
    total_chars += __builtin_popcount(bins[i].tpo_bits);

size_t va_target = static_cast<size_t>(total_chars * 0.68);
size_t poc = engine.get_poc_bin();
size_t lo = poc, hi = poc;
size_t enclosed = __builtin_popcount(bins[poc].tpo_bits);

while (enclosed < va_target && (lo > 0 || hi < bin_count - 1)) {
    size_t add_lo = lo > 0           ? __builtin_popcount(bins[lo-1].tpo_bits) : 0;
    size_t add_hi = hi < bin_count-1 ? __builtin_popcount(bins[hi+1].tpo_bits) : 0;
    if (add_lo >= add_hi && lo > 0)        { lo--; enclosed += add_lo; }
    else if (hi < bin_count - 1)           { hi++; enclosed += add_hi; }
    else break;
}
va_low_bin_ = lo;
va_high_bin_ = hi;
```

**[ACCEPTANCE — Phase 6]**
- [ ] TPO characters A–P appear at correct price levels after 8 hours of simulated data.
- [ ] `tpo_bits` bit 0 is set only during bracket 0 (first 30 minutes). Verify with unit test.
- [ ] Single print highlighting fires only on levels bracketed by multi-TPO levels above AND below.
- [ ] Value Area covers 68% ± 2% of total TPO characters.

---

## PHASE 7 — GLOBAL SYNC & PERFORMANCE
**Target Files:**
- `src/components/quant_workspace_component.cpp` — modify
- `src/rendering/frame_pacer.cpp` — **NEW FILE**
- `src/performance/regression_detector.cpp` — **NEW FILE**

### 7.1 — Atomic Crosshair

```cpp
// In a new header: include/sync/global_state.hpp
namespace BTQuant {
    inline std::atomic<double> g_crosshair_price{0.0};
    inline std::atomic<int32_t> g_crosshair_symbol_id{-1}; // -1 = no crosshair active
}
```

All panels (Charts, DOM, Tape, TPO) read this in their `render_gui()`:
```cpp
double ch_price = g_crosshair_price.load(std::memory_order_acquire);
if (ch_price > 0.0 && g_crosshair_symbol_id.load() == active_symbol_id_) {
    float y = ChartMath::MapToScreen({0,(float)ch_price}, viewport_, 0, h).y;
    // Draw 1px dashed horizontal line using AddLine() segments:
    for (float x = panel_min.x; x < panel_max.x; x += 6.0f)
        dl->AddLine({x, y}, {x + 3.0f, y}, IM_COL32(255,255,255,120), 1.0f);
}
```

When any panel's mouse hovers, write:
```cpp
g_crosshair_price.store(hovered_price, std::memory_order_release);
g_crosshair_symbol_id.store(active_symbol_id_, std::memory_order_release);
```

### 7.2 — Frame Pacer

Wire into `VulkanDashboard::render_frame()`. The existing `PerformanceMonitor::start_frame()` / `end_frame()` measures elapsed time. Extend:

```cpp
// src/rendering/frame_pacer.cpp
class FramePacer {
public:
    // Call after vkQueuePresentKHR returns.
    // If frame completed in < budget_us microseconds, yield to ingestion thread.
    void pace(uint64_t budget_us = 6944) { // 6944μs = 144fps budget
        using Clock = std::chrono::high_resolution_clock;
        auto now = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - frame_start_).count();
        if (elapsed < static_cast<int64_t>(budget_us)) {
            // std::this_thread::yield() lets the OS schedule the network/ingestion thread
            std::this_thread::yield();
        }
        frame_start_ = Clock::now();
    }
    void mark_frame_start() { frame_start_ = std::chrono::high_resolution_clock::now(); }
private:
    std::chrono::high_resolution_clock::time_point frame_start_;
};
extern FramePacer g_frame_pacer;
```

### 7.3 — Adaptive LOD

In `footprint_panel.cpp` and `tpo_panel.cpp`, before the text render loops:

```cpp
float cell_height_px = ChartMath::MapToScreen(
    {0, (float)(engine.get_day_low() + engine.get_tick_size())}, viewport_, 0, panel_height_).y
  - ChartMath::MapToScreen(
    {0, (float)engine.get_day_low()}, viewport_, 0, panel_height_).y;
cell_height_px = std::abs(cell_height_px);

const bool render_text = (cell_height_px >= 4.0f);
// If !render_text: draw colored rects only (no AddText calls)
```

### 7.4 — TSC Telemetry

Wire into `TelemetryCollector` (already exists in `telemetry_collector.h`):

```cpp
// In network ingestion callback (wherever HotSpineReader feeds MarketDataProcessor):
uint64_t tsc_ingress = __rdtsc();

// ... trade flows through SpscRingBuffer ...

// At end of TapeComponent::update() (UI render complete):
uint64_t tsc_render = __rdtsc();
double latency_us = static_cast<double>(tsc_render - tsc_ingress) / tsc_freq_mhz_;

btq::TelemetryCollector::getInstance().recordPerformanceMetric(
    "network_to_ui_latency_us", latency_us, "us");

// P99 alert:
if (latency_us > 10000.0) { // 10ms
    BTQ_LOG_WARNING("P99 latency breach: " + std::to_string(latency_us) + "us");
    // Optionally trigger PerformanceMonitor alert:
    // g_performance_monitor thresholds are already set in PerformanceConfig::latency_alert_threshold_ms
}
```

**TSC calibration** — compute `tsc_freq_mhz_` once at startup:
```cpp
auto t0 = std::chrono::high_resolution_clock::now();
uint64_t r0 = __rdtsc();
std::this_thread::sleep_for(std::chrono::milliseconds(100));
uint64_t r1 = __rdtsc();
auto t1 = std::chrono::high_resolution_clock::now();
double us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
tsc_freq_mhz_ = static_cast<double>(r1 - r0) / us;
```

**[ACCEPTANCE — Phase 7]**
- [ ] `g_crosshair_price` dashed line appears simultaneously in Charts, DOM, and TPO panels when hovering any one panel.
- [ ] `FramePacer::pace()` results in `g_performance_monitor.get_fps() >= 60` under full data load.
- [ ] Footprint text rendering is suppressed when cell height < 4px (zoom out until tick rows are < 4px).
- [ ] `TelemetryCollector` records `network_to_ui_latency_us`. P99 over 10,000 samples is < 10,000μs.
- [ ] `BTQ_LOG_WARNING` fires within one render frame of a latency breach.

---

## INTERFACE CONTRACT REGISTRY

The following interfaces must never be broken by any phase. If a change requires modifying these, update ALL dependent call sites first.

| Interface | File | Consumers |
|---|---|---|
| `MarketDataProcessor::get_latest_price(uint32_t)` | `market_data_processor.hpp` | Phase 4 DOM auto-center |
| `MarketDataProcessor::get_active_orderbook(uint32_t)` | `market_data_processor.hpp` | Phase 4 DOM volume bars |
| `SpscRingBuffer::peek(size_t, T*)` | `threading/lockfree_queue.hpp` | Phase 3 Tape |
| `GPUMemoryManager::add_texture(VkImageView, VkSampler, VkImageLayout)` | `vulkan_base_types.hpp` | Phase 1 heatmap, Phase 4 backdrop |
| `GPUMemoryManager::allocate_storage_buffer(VkDeviceSize)` | `vulkan_base_types.hpp` | Phase 1 SSBO |
| `VulkanCore::begin_single_time_commands()` | `vulkan_base_types.hpp` | Phase 1 image transitions |
| `ChartMath::MapToScreen(glm::vec2, ViewPort, float, float)` | `ChartMath.hpp` | Phases 4, 6, 7 |
| `ClusterEngine::ingest(const TradeData&)` | `cluster_engine.cpp` | Phase 5, 6 |
| `HotSpine::V3::VolumeNode` layout (16 bytes, std430) | `hotspine_layout_v3.hpp` | Phase 1 compute shader SSBO |
| `CacheManager::cacheAggregatedData()` | `cache_manager.hpp` | Phase 5 CVD persistence |
| `TelemetryCollector::recordPerformanceMetric()` | `telemetry_collector.h` | Phase 7 TSC latency |

---

## AUDIT LOG (Append here after each phase)

```
## PHASE 0 AUDIT LOG
Date:
Files modified:
Static assertions passing: [ ]
Arena allocation verified: [ ]
peek() unit test result:
Contract deviations:

## PHASE 1 AUDIT LOG
...
```
