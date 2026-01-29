# BTQ Render Engine - Quantower Clone Implementation
## Product Requirements Document (C++/Vulkan)

**Repository:** `dependencies/BTQ_Render_Engine`  
**Target:** Quantower-style crypto trading terminal  
**Tech Stack:** C++17/20, Vulkan, ImGui  
**Reference:** https://help.quantower.com/quantower/analytics-panels/chart/volume-analysis-tools

---

## ⚠️ CRITICAL CONSTRAINTS

1. **ONLY work in:** `dependencies/BTQ_Render_Engine` and subfolders
2. **NO breaking changes** to existing Vulkan pipeline
3. **Incremental additions** - extend, don't rewrite
4. **Test each widget** in isolation before integration

---

## Phase 1: Codebase Analysis & Architecture

### Task 1.1: Audit Existing Renderer Infrastructure
**Goal:** Understand current Vulkan setup without modifying anything

**Files to examine:**
```
dependencies/BTQ_Render_Engine/
├── src/renderer/
│   ├── VulkanContext.cpp
│   ├── SwapChain.cpp
│   ├── Pipeline.cpp
│   └── CommandBuffer.cpp
├── include/renderer/
│   └── *.h
└── shaders/
    ├── *.vert
    └── *.frag
```

**Deliverable:** `dependencies/BTQ_Render_Engine/docs/RENDERER_ANALYSIS.md`

**Document:**
1. **Device Setup:** Physical device selection, queue families (graphics, compute, transfer)
2. **Swap Chain:** Format (BGRA8? RGBA16F?), Present mode (FIFO, Mailbox?), Buffer count
3. **Command Buffer Strategy:** Single-threaded or multi-threaded submission?
4. **Descriptor Management:** Pool sizes, set layouts
5. **Memory Allocation:** VMA (Vulkan Memory Allocator) usage? Custom allocator?
6. **Sync Primitives:** Fences, semaphores for frame pacing

**Acceptance Criteria:**
- [ ] Complete Vulkan initialization sequence documented
- [ ] Identified thread model (single/multi-threaded)
- [ ] Memory allocation strategy clear
- [ ] No assumptions - only facts from code

---

### Task 1.2: ImGui Integration Assessment
**Goal:** Verify ImGui docking setup and custom rendering capabilities

**Files to check:**
```
dependencies/BTQ_Render_Engine/src/ui/
├── ImGuiLayer.cpp
├── ImGuiVulkanBackend.cpp
└── widgets/
    └── (existing widgets)
```

**Verify:**
1. ImGui version and docking branch enabled
2. Vulkan backend initialization (`ImGui_ImplVulkan_Init`)
3. Font loading (monospace for prices/volumes)
4. Custom draw list usage (`ImDrawList* draw_list = ImGui::GetWindowDrawList()`)
5. Multi-viewport support (floating windows)

**Deliverable:** `dependencies/BTQ_Render_Engine/docs/IMGUI_STATUS.md`

**Must include:**
- ImGui version
- Enabled features (docking, viewports, tables)
- Font atlas size and fonts loaded
- Custom rendering examples (if any exist)
- Performance: draw calls per frame, vertex count

**Acceptance Criteria:**
- [ ] Docking confirmed enabled
- [ ] Custom draw lists possible
- [ ] Font suitable for financial data (monospace)
- [ ] No performance red flags

---

### Task 1.3: Data Pipeline Architecture Review
**Goal:** Map data flow from WebSocket → Widgets without changing code

**Files to analyze:**
```
dependencies/BTQ_Render_Engine/
├── src/data/
│   ├── MarketDataManager.cpp
│   ├── WebSocketClient.cpp
│   ├── OrderBook.cpp
│   └── TradeAggregator.cpp
├── include/data/
│   ├── OrderBook.h
│   ├── Trade.h
│   └── OHLCV.h
```

**Map out:**
1. **WebSocket → Parser:** How are messages decoded? (JSON? FlatBuffers?)
2. **Order Book Updates:** How are bid/ask levels stored? (`std::map`? Custom structure?)
3. **Trade Aggregation:** How are trades collected for footprint/TPO?
4. **Historical Data:** SQLite? In-memory ring buffers? File-based?
5. **Thread Safety:** Mutexes? Lock-free queues? Single-threaded?
6. **Symbol Switching:** How is the active symbol changed globally?

**Deliverable:** `dependencies/BTQ_Render_Engine/docs/DATA_FLOW.md`

**Must include:**
- Sequence diagram: WebSocket message → Widget update
- Data structures (OrderBook, Trade, Candle)
- Threading model
- Bottlenecks (locks, copies, allocations in hot path)

**Acceptance Criteria:**
- [ ] Complete data flow documented
- [ ] Thread-safety issues identified
- [ ] Performance bottlenecks noted
- [ ] Recommendations for improvements (but don't implement yet)

---

## Phase 2: Volume Analysis Tools (Core Features)

### Task 2.1: Order Book Widget (DOM - Depth of Market)
**Reference:** Screenshot - Left panel showing price levels with bid/ask volumes

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/widgets/OrderBookWidget.cpp
dependencies/BTQ_Render_Engine/include/widgets/OrderBookWidget.h
```

**Visual Reference (from screenshot):**
```
Order Book (Live)
Price    Center   Multi
91270    23.3            1.1
91260    7.1             6.4
91250    9.8             1.9
91240    80.8            0.5
...
```

**Data Structure:**
```cpp
namespace BTQ {

struct OrderBookLevel {
    double price;
    double bidVolume;
    double askVolume;
    int bidOrders;    // number of orders at this level
    int askOrders;
};

class OrderBookWidget {
public:
    void render();
    void update(const OrderBook& book);
    
private:
    std::vector<OrderBookLevel> levels_;
    
    // View settings
    int numLevels_ = 20;         // visible levels each side
    double tickGrouping_ = 1.0;  // group by N ticks
    
    // Visual settings
    bool showCenter_ = true;     // center column (total volume)
    bool showMulti_ = true;      // multi-exchange aggregation
    bool highlightLargeOrders_ = true;
};

} // namespace BTQ
```

**Rendering (ImGui):**
```cpp
void OrderBookWidget::render() {
    ImGui::Begin("Order Book (Live)");
    
    if (ImGui::BeginTable("orderbook", 4, ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Price");
        ImGui::TableSetupColumn("Center");
        ImGui::TableSetupColumn("Multi");
        ImGui::TableSetupColumn("Settings");
        ImGui::TableHeadersRow();
        
        for (const auto& level : levels_) {
            ImGui::TableNextRow();
            
            // Price column
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", level.price);
            
            // Center column (total bid volume)
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", level.bidVolume);
            
            // Multi column (ask volume)
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", level.askVolume);
            
            // Highlight large orders
            if (level.bidVolume > averageVolume_ * 5.0) {
                // Draw background highlight
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 min = ImGui::GetItemRectMin();
                ImVec2 max = ImGui::GetItemRectMax();
                draw_list->AddRectFilled(min, max, IM_COL32(255, 255, 0, 50));
            }
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}
```

**Features:**
1. **Price levels:** Show N levels above/below mid price
2. **Volume display:** Bid and Ask volumes per level
3. **Tick grouping:** Aggregate by 1, 5, 10, 50 ticks
4. **Large order highlight:** Color/background when volume > threshold
5. **Multi-exchange:** Aggregate order book from multiple exchanges
6. **Settings popup:** Right-click to configure

**Acceptance Criteria:**
- [ ] Displays current order book with configurable levels
- [ ] Updates in real-time (<100ms latency)
- [ ] Large orders visually distinct
- [ ] Tick grouping selector works
- [ ] Performance: 60+ FPS with 100 levels

---

### Task 2.2: Footprint Chart (Cluster Chart / OrderFlow)
**Reference:** Screenshot - Center chart with green/red candles and volume histogram  
**Quantower Docs:** https://help.quantower.com/quantower/analytics-panels/chart/volume-analysis-tools/cluster-chart

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/widgets/FootprintChart.cpp
dependencies/BTQ_Render_Engine/include/widgets/FootprintChart.h
dependencies/BTQ_Render_Engine/shaders/footprint.frag
```

**What is Footprint Chart?**
A footprint chart displays **volume traded at each price level** within each time bar, showing bid vs ask volume to reveal order flow.

**Visual Layout:**
```
Time →
Price ↓   10:00    10:30    11:00    11:30
91300     150      200      180      220
          120↓80↑  150↓50↑  100↓80↑  180↓40↑
                   (more sellers)
91290     100      120      90       110
          60↓40↑   70↓50↑   50↓40↑   60↓50↑
```

**Data Structure:**
```cpp
struct FootprintCell {
    double priceLevel;       // rounded to tick size
    uint64_t timeBucket;     // bar start time (e.g., 10:00:00)
    double buyVolume;        // total buy volume at this price in this bar
    double sellVolume;       // total sell volume
    double delta;            // buyVolume - sellVolume
    int numBuyTrades;
    int numSellTrades;
    double maxSingleTrade;   // largest single trade
};

class FootprintChart {
public:
    void render();
    void update(const std::vector<Trade>& trades);
    
    // Settings
    enum class DataType {
        Trades,              // number of trades
        Volume,              // total volume
        BuySellVolume,       // buy and sell volume separately
        Delta,               // buy - sell volume
        DeltaPercent,        // delta / total volume * 100
        AverageSize,         // average trade size
        MaxOneTrade          // largest single trade
    };
    
    void setDataType(DataType type);
    void setTimeframe(int minutes);  // bar size: 1, 5, 15, 30, 60
    
private:
    std::map<uint64_t, std::map<double, FootprintCell>> grid_;
    DataType dataType_ = DataType::BuySellVolume;
    int timeframeMinutes_ = 30;
};
```

**Rendering Strategy:**

**Option A: ImGui Table (Simple, but slow for >1000 cells)**
```cpp
void FootprintChart::render() {
    ImGui::Begin("Footprint Chart");
    
    if (ImGui::BeginTable("footprint", numTimeBuckets + 1)) {
        // Header row (times)
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); // empty corner
        for (auto time : timeBuckets_) {
            ImGui::TableNextColumn();
            ImGui::Text("%02d:%02d", getHour(time), getMinute(time));
        }
        
        // Price rows
        for (double price : priceLevels_) {
            ImGui::TableNextRow();
            
            // Price label
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", price);
            
            // Cells
            for (auto time : timeBuckets_) {
                ImGui::TableNextColumn();
                
                auto& cell = grid_[time][price];
                
                // Background color based on delta
                ImU32 bgColor = getHeatmapColor(cell.delta);
                ImVec2 min = ImGui::GetCursorScreenPos();
                ImVec2 max = ImVec2(min.x + cellWidth, min.y + cellHeight);
                ImGui::GetWindowDrawList()->AddRectFilled(min, max, bgColor);
                
                // Text (buy/sell volumes)
                ImGui::Text("%d↓ %d↑", (int)cell.sellVolume, (int)cell.buyVolume);
            }
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}
```

**Option B: Vulkan Instanced Rendering (Fast, for >10,000 cells)**
```cpp
// Each cell is a quad (2 triangles)
struct FootprintVertex {
    glm::vec2 position;  // screen space
    glm::vec4 color;     // heatmap color
    glm::vec2 texCoord;  // for text atlas
};

// Vertex shader
#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}

// Fragment shader (footprint.frag)
#version 450
layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;  // heatmap color
}
```

**Heatmap Color Calculation:**
```cpp
ImU32 FootprintChart::getHeatmapColor(double delta) const {
    // Normalize delta to [-1, 1]
    double maxDelta = 1000.0;  // configurable
    double normalized = std::clamp(delta / maxDelta, -1.0, 1.0);
    
    if (normalized > 0) {
        // Positive delta (more buyers) → green gradient
        int green = static_cast<int>(255 * normalized);
        return IM_COL32(0, green, 0, 200);
    } else {
        // Negative delta (more sellers) → red gradient
        int red = static_cast<int>(255 * -normalized);
        return IM_COL32(red, 0, 0, 200);
    }
}
```

**Data Types (from Quantower):**
1. **Trades:** Number of trades at each price/time
2. **Volume:** Total volume (buy + sell)
3. **Buy/Sell Volume:** Separate columns or split display
4. **Delta:** buyVolume - sellVolume
5. **Delta %:** (delta / totalVolume) * 100
6. **Average Size:** avg trade size at price/time
7. **Max One Trade:** largest single trade

**Acceptance Criteria:**
- [ ] Renders grid with time × price cells
- [ ] Heatmap color reflects delta (red=sellers, green=buyers)
- [ ] Shows buy/sell volume in each cell
- [ ] Updates in real-time as trades arrive
- [ ] Performance: 200 time buckets × 100 price levels at 60 FPS
- [ ] Data type selector works (Trades, Volume, Delta, etc.)

---

### Task 2.3: Volume Profile (Step, Right, Left, Custom)
**Reference:** Screenshot - Right side of chart showing horizontal volume histogram  
**Quantower Docs:** https://help.quantower.com/quantower/analytics-panels/chart/volume-analysis-tools/volume-profiles

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/widgets/VolumeProfile.cpp
dependencies/BTQ_Render_Engine/include/widgets/VolumeProfile.h
```

**Types of Volume Profiles:**

1. **Right Profile:** Attached to right edge of chart (visible timeframe)
2. **Left Profile:** Attached to left edge
3. **Step Profile:** Separate profile for each bar
4. **Custom Profile:** User-defined time range

**Visual (from screenshot - right side):**
```
Price    Volume Bar
91300 ▓▓▓▓▓▓▓▓▓▓░░░░
91280 ▓▓▓▓▓▓▓░░░░░░░  <- Volume Profile
91260 ▓▓▓▓▓▓▓▓▓▓▓▓▓
91240 ▓▓▓▓░░░░░░░░░░
```

**Data Structure:**
```cpp
struct VolumeProfileNode {
    double priceLevel;      // bin center
    double totalVolume;
    double buyVolume;
    double sellVolume;
    double delta;
    int numTrades;
};

struct VolumeProfileData {
    std::vector<VolumeProfileNode> nodes;
    double POC;   // Point of Control (price with highest volume)
    double VAH;   // Value Area High (top of 70% volume zone)
    double VAL;   // Value Area Low (bottom of 70% volume zone)
    double totalVolume;
};

class VolumeProfile {
public:
    enum class Type { Right, Left, Step, Custom };
    enum class DataType {
        Volume,
        BuyVolume,
        SellVolume,
        BuySellVolume,  // split bar
        Delta,
        DeltaPercent,
        Trades
    };
    
    void render();
    void calculate(const std::vector<Trade>& trades, 
                   uint64_t startTime, 
                   uint64_t endTime);
    
    void setType(Type type);
    void setDataType(DataType dataType);
    void setCustomStep(double ticks);  // bin size
    
private:
    VolumeProfileData profile_;
    Type type_ = Type::Right;
    DataType dataType_ = DataType::BuySellVolume;
    double customStepTicks_ = 1.0;
};
```

**Calculation Algorithm:**
```cpp
VolumeProfileData VolumeProfile::calculate(
    const std::vector<Trade>& trades,
    uint64_t startTime,
    uint64_t endTime
) {
    // 1. Determine price range
    double minPrice = std::numeric_limits<double>::max();
    double maxPrice = std::numeric_limits<double>::lowest();
    for (const auto& trade : trades) {
        if (trade.timestamp >= startTime && trade.timestamp <= endTime) {
            minPrice = std::min(minPrice, trade.price);
            maxPrice = std::max(maxPrice, trade.price);
        }
    }
    
    // 2. Create price bins (e.g., every 1 tick)
    double tickSize = 0.1;  // BTC-USDT tick size
    double binSize = customStepTicks_ * tickSize;
    int numBins = static_cast<int>((maxPrice - minPrice) / binSize) + 1;
    
    std::vector<VolumeProfileNode> nodes(numBins);
    for (int i = 0; i < numBins; ++i) {
        nodes[i].priceLevel = minPrice + i * binSize;
    }
    
    // 3. Aggregate trades into bins
    for (const auto& trade : trades) {
        if (trade.timestamp < startTime || trade.timestamp > endTime) continue;
        
        int binIndex = static_cast<int>((trade.price - minPrice) / binSize);
        if (binIndex >= 0 && binIndex < numBins) {
            nodes[binIndex].totalVolume += trade.volume;
            if (trade.side == TradeSide::Buy) {
                nodes[binIndex].buyVolume += trade.volume;
            } else {
                nodes[binIndex].sellVolume += trade.volume;
            }
            nodes[binIndex].numTrades++;
        }
    }
    
    // 4. Calculate POC (Point of Control)
    double maxVolume = 0;
    double POC = 0;
    for (const auto& node : nodes) {
        if (node.totalVolume > maxVolume) {
            maxVolume = node.totalVolume;
            POC = node.priceLevel;
        }
    }
    
    // 5. Calculate Value Area (70% of volume)
    double totalVolume = 0;
    for (const auto& node : nodes) {
        totalVolume += node.totalVolume;
    }
    double targetVolume = totalVolume * 0.70;
    
    // Find VAH and VAL by expanding from POC
    // (implementation details omitted for brevity)
    
    VolumeProfileData result;
    result.nodes = std::move(nodes);
    result.POC = POC;
    result.VAH = VAH;  // calculated above
    result.VAL = VAL;  // calculated above
    result.totalVolume = totalVolume;
    
    return result;
}
```

**Rendering:**
```cpp
void VolumeProfile::render() {
    ImGui::Begin("Volume Profile");
    
    // Settings
    if (ImGui::BeginCombo("Type", getTypeName(type_))) {
        if (ImGui::Selectable("Right")) type_ = Type::Right;
        if (ImGui::Selectable("Left")) type_ = Type::Left;
        if (ImGui::Selectable("Step")) type_ = Type::Step;
        if (ImGui::Selectable("Custom")) type_ = Type::Custom;
        ImGui::EndCombo();
    }
    
    if (ImGui::BeginCombo("Data Type", getDataTypeName(dataType_))) {
        // ... selectable for each DataType
        ImGui::EndCombo();
    }
    
    // Render histogram
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    
    // Find max volume for scaling
    double maxVolume = 0;
    for (const auto& node : profile_.nodes) {
        maxVolume = std::max(maxVolume, node.totalVolume);
    }
    
    // Draw horizontal bars
    float barHeight = canvas_size.y / profile_.nodes.size();
    for (size_t i = 0; i < profile_.nodes.size(); ++i) {
        const auto& node = profile_.nodes[i];
        
        float y = canvas_pos.y + i * barHeight;
        float barWidth = (node.totalVolume / maxVolume) * canvas_size.x;
        
        // Draw bid volume (green, left side)
        float bidWidth = (node.buyVolume / maxVolume) * canvas_size.x;
        draw_list->AddRectFilled(
            ImVec2(canvas_pos.x, y),
            ImVec2(canvas_pos.x + bidWidth, y + barHeight),
            IM_COL32(0, 255, 0, 150)
        );
        
        // Draw ask volume (red, right side of bid)
        float askWidth = (node.sellVolume / maxVolume) * canvas_size.x;
        draw_list->AddRectFilled(
            ImVec2(canvas_pos.x + bidWidth, y),
            ImVec2(canvas_pos.x + bidWidth + askWidth, y + barHeight),
            IM_COL32(255, 0, 0, 150)
        );
        
        // Highlight POC
        if (std::abs(node.priceLevel - profile_.POC) < 0.01) {
            draw_list->AddLine(
                ImVec2(canvas_pos.x, y + barHeight/2),
                ImVec2(canvas_pos.x + canvas_size.x, y + barHeight/2),
                IM_COL32(255, 255, 0, 255),
                3.0f
            );
        }
        
        // Shade Value Area
        if (node.priceLevel >= profile_.VAL && node.priceLevel <= profile_.VAH) {
            draw_list->AddRectFilled(
                ImVec2(canvas_pos.x, y),
                ImVec2(canvas_pos.x + canvas_size.x, y + barHeight),
                IM_COL32(255, 255, 255, 30)
            );
        }
    }
    
    ImGui::End();
}
```

**Acceptance Criteria:**
- [ ] Calculates volume profile for visible time range
- [ ] POC line clearly visible
- [ ] Value Area (70%) shaded
- [ ] Buy/Sell volume split display works
- [ ] Type selector (Right, Left, Step, Custom) functional
- [ ] Data type selector works
- [ ] Performance: <10ms calculation for 10,000 trades

---

### Task 2.4: Candlestick Chart with Overlays
**Reference:** Screenshot - Main chart area with green/red candles

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/widgets/CandlestickChart.cpp
dependencies/BTQ_Render_Engine/include/widgets/CandlestickChart.h
dependencies/BTQ_Render_Engine/shaders/candle.vert
dependencies/BTQ_Render_Engine/shaders/candle.frag
```

**Data Structure:**
```cpp
struct OHLCV {
    uint64_t timestamp;  // bar open time
    double open;
    double high;
    double low;
    double close;
    double volume;
    double buyVolume;   // for coloring volume bars
    double sellVolume;
};

class CandlestickChart {
public:
    void render();
    void update(const std::vector<OHLCV>& candles);
    
    // Zoom/Pan
    void handleMouseWheel(double delta);
    void handleMouseDrag(double dx, double dy);
    void resetView();
    
    // Overlays
    void attachVolumeProfile(VolumeProfile* profile);
    void attachFootprint(FootprintChart* footprint);
    
private:
    std::vector<OHLCV> candles_;
    
    // View transform
    double xScale_;   // pixels per millisecond
    double yScale_;   // pixels per price unit
    double xOffset_;  // pan offset
    double yOffset_;
    
    // Attached widgets
    VolumeProfile* volumeProfile_ = nullptr;
    FootprintChart* footprint_ = nullptr;
};
```

**Rendering (Vulkan):**

Each candle is 2 quads:
1. **Wick** (high-low line): thin vertical line
2. **Body** (open-close): thick rectangle

```cpp
struct CandleVertex {
    glm::vec2 position;  // screen space
    glm::vec4 color;     // green or red
};

// Generate vertices for each candle
std::vector<CandleVertex> generateCandleVertices(const OHLCV& candle) {
    std::vector<CandleVertex> vertices;
    
    // Determine color
    glm::vec4 color = (candle.close >= candle.open) 
        ? glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)  // green
        : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // red
    
    // Convert price to screen Y
    auto priceToY = [](double price) { /* transform */ };
    auto timeToX = [](uint64_t time) { /* transform */ };
    
    float x = timeToX(candle.timestamp);
    float highY = priceToY(candle.high);
    float lowY = priceToY(candle.low);
    float openY = priceToY(candle.open);
    float closeY = priceToY(candle.close);
    
    // Wick (thin line)
    float wickWidth = 2.0f;
    vertices.push_back({{x - wickWidth/2, highY}, color});
    vertices.push_back({{x + wickWidth/2, highY}, color});
    vertices.push_back({{x + wickWidth/2, lowY}, color});
    vertices.push_back({{x - wickWidth/2, lowY}, color});
    
    // Body (thick rectangle)
    float bodyWidth = 8.0f;
    float bodyTop = std::min(openY, closeY);
    float bodyBottom = std::max(openY, closeY);
    vertices.push_back({{x - bodyWidth/2, bodyTop}, color});
    vertices.push_back({{x + bodyWidth/2, bodyTop}, color});
    vertices.push_back({{x + bodyWidth/2, bodyBottom}, color});
    vertices.push_back({{x - bodyWidth/2, bodyBottom}, color});
    
    return vertices;
}
```

**Zoom/Pan Implementation:**
```cpp
void CandlestickChart::handleMouseWheel(double delta) {
    // Zoom centered on mouse cursor
    ImVec2 mousePos = ImGui::GetMousePos();
    
    // Convert mouse position to price/time
    double mouseTime = screenToTime(mousePos.x);
    double mousePrice = screenToPrice(mousePos.y);
    
    // Zoom
    double zoomFactor = 1.0 + delta * 0.1;
    xScale_ *= zoomFactor;
    yScale_ *= zoomFactor;
    
    // Adjust offset to keep mouse position fixed
    xOffset_ = mouseTime - screenToTime(mousePos.x);
    yOffset_ = mousePrice - screenToPrice(mousePos.y);
}

void CandlestickChart::handleMouseDrag(double dx, double dy) {
    // Pan by dragging
    xOffset_ += dx / xScale_;
    yOffset_ += dy / yScale_;
}
```

**Volume Bars (below chart):**
```cpp
void renderVolumeBars() {
    for (const auto& candle : candles_) {
        float x = timeToX(candle.timestamp);
        float height = (candle.volume / maxVolume_) * volumeBarMaxHeight;
        
        // Color matches candle
        glm::vec4 color = (candle.close >= candle.open)
            ? glm::vec4(0.0f, 1.0f, 0.0f, 0.5f)
            : glm::vec4(1.0f, 0.0f, 0.0f, 0.5f);
        
        draw_list->AddRectFilled(
            ImVec2(x - barWidth/2, volumeBaseY),
            ImVec2(x + barWidth/2, volumeBaseY - height),
            ImColor(color)
        );
    }
}
```

**Acceptance Criteria:**
- [ ] Candlesticks render correctly (OHLC)
- [ ] Green candles for close > open, red for close < open
- [ ] Volume bars below chart, color-coded
- [ ] Zoom centers on mouse cursor
- [ ] Pan works smoothly
- [ ] Performance: 1000 candles at 144 FPS
- [ ] Volume profile can be overlaid on right edge

---

### Task 2.5: Time & Sales (Trades List)
**Reference:** Screenshot - Bottom-left panel "Trades (Live)"

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/widgets/TimeAndSales.cpp
dependencies/BTQ_Render_Engine/include/widgets/TimeAndSales.h
```

**Visual (from screenshot):**
```
Trades (Live) | 3 perp, 1 spot | $100k+ | Filters
9.7%                                              83%
44 Trades 0:55.5M $$:82.0M

Time         Exch    Price       Qty      Value      S
17:02:47     ⚡      90988.10    1.30     $118K      B
17:02:44     🔸      90988.00    1.11     $101K      B
```

**Data Structure:**
```cpp
struct Trade {
    uint64_t timestamp;  // microseconds
    std::string exchange;
    double price;
    double quantity;
    double value;  // price * quantity
    TradeSide side;  // BUY, SELL
    bool isLargeTrade;
};

class TimeAndSales {
public:
    void render();
    void addTrade(const Trade& trade);
    
    // Filters
    void setMinValue(double minValue);  // only show trades > $X
    void setExchangeFilter(const std::vector<std::string>& exchanges);
    
private:
    std::deque<Trade> trades_;  // ring buffer
    int maxTrades_ = 1000;
    
    // Filters
    double minValue_ = 0;
    std::set<std::string> activeExchanges_;
    
    // UI state
    bool autoScroll_ = true;
};
```

**Rendering (ImGui Table):**
```cpp
void TimeAndSales::render() {
    ImGui::Begin("Trades (Live)");
    
    // Header with stats
    ImGui::Text("44 Trades | $55.5M | $$:82.0M");
    
    // Filters
    if (ImGui::Button("Filters")) {
        ImGui::OpenPopup("TradeFilters");
    }
    
    if (ImGui::BeginPopup("TradeFilters")) {
        ImGui::InputDouble("Min Value ($)", &minValue_);
        // Exchange checkboxes...
        ImGui::EndPopup();
    }
    
    // Table
    if (ImGui::BeginTable("trades", 6, 
                          ImGuiTableFlags_ScrollY | 
                          ImGuiTableFlags_Borders)) {
        ImGui::TableSetupColumn("Time");
        ImGui::TableSetupColumn("Exch");
        ImGui::TableSetupColumn("Price");
        ImGui::TableSetupColumn("Qty");
        ImGui::TableSetupColumn("Value");
        ImGui::TableSetupColumn("S");  // Side
        ImGui::TableSetupScrollFreeze(0, 1);  // freeze header
        ImGui::TableHeadersRow();
        
        // Virtualized rendering
        ImGuiListClipper clipper;
        clipper.Begin(trades_.size());
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const auto& trade = trades_[i];
                
                // Skip filtered trades
                if (trade.value < minValue_) continue;
                if (!activeExchanges_.count(trade.exchange)) continue;
                
                ImGui::TableNextRow();
                
                // Time
                ImGui::TableNextColumn();
                ImGui::Text("%s", formatTime(trade.timestamp).c_str());
                
                // Exchange
                ImGui::TableNextColumn();
                ImGui::Text("%s", trade.exchange.c_str());
                
                // Price
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", trade.price);
                
                // Quantity
                ImGui::TableNextColumn();
                ImGui::Text("%.2f", trade.quantity);
                
                // Value
                ImGui::TableNextColumn();
                ImGui::Text("$%.0fK", trade.value / 1000.0);
                if (trade.isLargeTrade) {
                    // Highlight background
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    ImVec2 min = ImGui::GetItemRectMin();
                    ImVec2 max = ImGui::GetItemRectMax();
                    dl->AddRectFilled(min, max, IM_COL32(255, 255, 0, 50));
                }
                
                // Side
                ImGui::TableNextColumn();
                if (trade.side == TradeSide::Buy) {
                    ImGui::TextColored(ImVec4(0,1,0,1), "B");
                } else {
                    ImGui::TextColored(ImVec4(1,0,0,1), "S");
                }
            }
        }
        
        // Auto-scroll to bottom
        if (autoScroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
        
        // Pause auto-scroll on manual scroll
        if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseWheel != 0) {
            autoScroll_ = false;
        }
        
        ImGui::EndTable();
    }
    
    ImGui::End();
}
```

**Large Trade Detection:**
```cpp
void TimeAndSales::addTrade(const Trade& trade) {
    // Calculate rolling average trade size
    double avgSize = calculateAverageSize();  // last 100 trades
    
    Trade processedTrade = trade;
    processedTrade.isLargeTrade = (trade.value > avgSize * 5.0);
    
    trades_.push_back(processedTrade);
    
    // Keep buffer size limited
    if (trades_.size() > maxTrades_) {
        trades_.pop_front();
    }
}
```

**Acceptance Criteria:**
- [ ] Displays live trade feed
- [ ] Color-coded by side (green=buy, red=sell)
- [ ] Large trades highlighted
- [ ] Filters work (min value, exchange)
- [ ] Auto-scroll to latest trade
- [ ] Pause auto-scroll on manual scroll
- [ ] Virtualized rendering (handles 10,000+ trades)
- [ ] Performance: 60+ FPS

---

### Task 2.6: Custom VWAP (Anchored VWAP)
**Reference:** Quantower Docs  
**Quantower Docs:** https://help.quantower.com/quantower/analytics-panels/chart/volume-analysis-tools/anchored-vwap

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/indicators/AnchoredVWAP.cpp
dependencies/BTQ_Render_Engine/include/indicators/AnchoredVWAP.h
```

**What is Anchored VWAP?**
VWAP (Volume Weighted Average Price) starting from a user-selected bar, used to identify support/resistance levels from key events.

**Formula:**
```
VWAP = Σ(Price × Volume) / Σ(Volume)
```

**Data Structure:**
```cpp
class AnchoredVWAP {
public:
    void setAnchorBar(uint64_t timestamp);
    void calculate(const std::vector<OHLCV>& candles);
    
    double getVWAP(uint64_t timestamp) const;
    double getUpperBand(uint64_t timestamp) const;  // VWAP + stddev
    double getLowerBand(uint64_t timestamp) const;  // VWAP - stddev
    
private:
    uint64_t anchorTimestamp_;
    std::map<uint64_t, double> vwapValues_;
    std::map<uint64_t, double> upperBands_;
    std::map<uint64_t, double> lowerBands_;
};
```

**Calculation:**
```cpp
void AnchoredVWAP::calculate(const std::vector<OHLCV>& candles) {
    double cumulativePV = 0;  // price × volume
    double cumulativeVolume = 0;
    
    for (const auto& candle : candles) {
        if (candle.timestamp < anchorTimestamp_) continue;
        
        double typicalPrice = (candle.high + candle.low + candle.close) / 3.0;
        cumulativePV += typicalPrice * candle.volume;
        cumulativeVolume += candle.volume;
        
        double vwap = cumulativePV / cumulativeVolume;
        vwapValues_[candle.timestamp] = vwap;
        
        // Calculate standard deviation for bands
        // (implementation omitted)
    }
}
```

**Rendering on Chart:**
```cpp
void CandlestickChart::renderAnchoredVWAP() {
    if (!anchoredVWAP_) return;
    
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    
    std::vector<ImVec2> vwapPoints;
    for (const auto& candle : candles_) {
        double vwap = anchoredVWAP_->getVWAP(candle.timestamp);
        if (vwap > 0) {
            float x = timeToX(candle.timestamp);
            float y = priceToY(vwap);
            vwapPoints.push_back(ImVec2(x, y));
        }
    }
    
    // Draw VWAP line
    if (vwapPoints.size() > 1) {
        draw_list->AddPolyline(
            vwapPoints.data(), 
            vwapPoints.size(), 
            IM_COL32(255, 255, 0, 255),  // yellow
            ImDrawFlags_None,
            2.0f  // thickness
        );
    }
    
    // Draw bands (optional)
    // ...
}
```

**Acceptance Criteria:**
- [ ] User can anchor VWAP to any bar (click to set anchor)
- [ ] VWAP line renders from anchor point forward
- [ ] Standard deviation bands optional
- [ ] Multiple anchored VWAPs can exist simultaneously
- [ ] Calculation correct (matches TradingView)
- [ ] Performance: <5ms calculation for 1000 bars

---

## Phase 3: Integration & Dashboard

### Task 3.1: Dashboard Layout System
**Reference:** Screenshot - Multi-panel layout with docking

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/ui/Dashboard.cpp
dependencies/BTQ_Render_Engine/include/ui/Dashboard.h
```

**Features:**
1. **Dockable Panels:** Drag panels to reposition
2. **Resizable:** Drag panel edges to resize
3. **Tabbed Windows:** Multiple charts in tabs
4. **Floating Windows:** Detach panels
5. **Layout Presets:** Save/load configurations

**ImGui Docking API:**
```cpp
void Dashboard::render() {
    // Enable docking
    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), dockspace_flags);
    
    // First time setup: create default layout
    static bool first_time = true;
    if (first_time) {
        first_time = false;
        createDefaultLayout(dockspace_id);
    }
    
    // Render all panels
    orderBookWidget_->render();
    candlestickChart_->render();
    footprintChart_->render();
    volumeProfile_->render();
    timeAndSales_->render();
    watchlist_->render();
}

void Dashboard::createDefaultLayout(ImGuiID dockspace_id) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);
    
    // Split layout
    ImGuiID dock_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.2f, nullptr, &dockspace_id);
    ImGuiID dock_right = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.25f, nullptr, &dockspace_id);
    ImGuiID dock_bottom = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.3f, nullptr, &dockspace_id);
    
    // Dock windows
    ImGui::DockBuilderDockWindow("Order Book (Live)", dock_left);
    ImGui::DockBuilderDockWindow("Footprint Chart", dockspace_id);  // center
    ImGui::DockBuilderDockWindow("Volume Profile", dock_right);
    ImGui::DockBuilderDockWindow("Trades (Live)", dock_bottom);
    ImGui::DockBuilderDockWindow("Watchlist", dock_bottom);
    
    ImGui::DockBuilderFinish(dockspace_id);
}
```

**Layout Saving:**
```cpp
void Dashboard::saveLayout(const std::string& name) {
    ImGui::SaveIniSettingsToDisk((name + ".ini").c_str());
}

void Dashboard::loadLayout(const std::string& name) {
    ImGui::LoadIniSettingsFromDisk((name + ".ini").c_str());
}
```

**Acceptance Criteria:**
- [ ] All widgets dockable
- [ ] Drag-and-drop panel repositioning works
- [ ] Resize handles functional
- [ ] Tabs work for multi-chart setups
- [ ] Floating windows supported
- [ ] Layout save/load functional
- [ ] Default layout matches Quantower style

---

### Task 3.2: Global Symbol Switching
**Goal:** Change symbol in one panel updates all panels

**Files to modify:**
```
dependencies/BTQ_Render_Engine/src/ui/Dashboard.cpp
dependencies/BTQ_Render_Engine/src/data/MarketDataManager.cpp
```

**Implementation:**
```cpp
class Dashboard {
public:
    void setActiveSymbol(const std::string& symbol);
    const std::string& getActiveSymbol() const { return activeSymbol_; }
    
private:
    std::string activeSymbol_ = "BTC-USDT";
    
    // Callbacks for widgets
    std::vector<std::function<void(const std::string&)>> symbolChangeCallbacks_;
};

void Dashboard::setActiveSymbol(const std::string& symbol) {
    activeSymbol_ = symbol;
    
    // Notify all widgets
    for (auto& callback : symbolChangeCallbacks_) {
        callback(symbol);
    }
    
    // Trigger data reload
    marketDataManager_->switchSymbol(symbol);
}

// Widgets register callbacks
orderBookWidget_->onSymbolChange([this](const std::string& symbol) {
    dashboard_->setActiveSymbol(symbol);
});
```

**Acceptance Criteria:**
- [ ] Clicking symbol in watchlist updates all panels
- [ ] Symbol selector in header updates all panels
- [ ] Data reloads for new symbol (<500ms)
- [ ] No stale data from previous symbol

---

### Task 3.3: Performance Monitoring Overlay
**Reference:** Screenshot - Bottom-left "FPS: 120 | Idle | Mode: Demo"

**Files to create:**
```
dependencies/BTQ_Render_Engine/src/ui/PerformanceOverlay.cpp
```

**Metrics to Display:**
1. **FPS:** Frames per second
2. **Frame Time:** ms per frame
3. **CPU Usage:** per widget
4. **Memory Usage:** Vulkan allocations
5. **Data Latency:** WebSocket → Render
6. **Network Stats:** Messages/sec, bandwidth

**Implementation:**
```cpp
class PerformanceOverlay {
public:
    void render();
    void update(float deltaTime);
    
private:
    struct FrameStats {
        float fps;
        float frameTime;
        float cpuTime;
        float gpuTime;
        size_t memoryUsage;
    };
    
    std::deque<FrameStats> history_;  // last 120 frames
};

void PerformanceOverlay::render() {
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 50));
    ImGui::SetNextWindowSize(ImVec2(400, 40));
    ImGui::Begin("Performance", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | 
                 ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove);
    
    float avgFps = calculateAverageFPS();
    float avgFrameTime = calculateAverageFrameTime();
    size_t memUsageMB = getCurrentMemoryUsage() / (1024 * 1024);
    
    ImGui::Text("FPS: %.0f | Frame: %.1fms | Mem: %zuMB | Latency: %.0fms",
                avgFps, avgFrameTime, memUsageMB, getDataLatency());
    
    ImGui::End();
}
```

**Acceptance Criteria:**
- [ ] FPS displayed and accurate
- [ ] Frame time graph optional
- [ ] Memory usage tracked
- [ ] Data latency measured (WebSocket to render)
- [ ] Toggle on/off with hotkey (F3)

---

## Phase 4: Optimization & Polish

### Task 4.1: Rendering Performance Optimization
**Goal:** Achieve 144 FPS with all widgets active

**Strategies:**
1. **Culling:** Don't render off-screen panels
2. **LOD:** Reduce detail when zoomed out
3. **Instancing:** Batch similar geometry (footprint cells)
4. **Caching:** Cache volume profile calculations
5. **Multi-threading:** Calculate indicators on worker threads

**Example: Culling**
```cpp
void Dashboard::render() {
    for (auto& widget : widgets_) {
        if (!widget->isVisible()) continue;  // culling
        widget->render();
    }
}
```

**Example: LOD for Candlesticks**
```cpp
void CandlestickChart::render() {
    int numVisibleCandles = calculateVisibleCandles();
    
    if (numVisibleCandles > 500) {
        // High LOD: show wicks and bodies
        renderDetailedCandles();
    } else if (numVisibleCandles > 100) {
        // Medium LOD: show bodies only
        renderSimplifiedCandles();
    } else {
        // Low LOD: show lines only
        renderCandleLines();
    }
}
```

**Acceptance Criteria:**
- [ ] Maintains 144 FPS with all widgets visible
- [ ] CPU usage <30% on modern CPU
- [ ] GPU usage <40%
- [ ] Memory usage <500MB
- [ ] No stuttering during data updates

---

### Task 4.2: Memory Management
**Goal:** No memory leaks, efficient buffer usage

**Strategies:**
1. **Ring Buffers:** Limit historical data
2. **VMA:** Use Vulkan Memory Allocator
3. **Pooling:** Reuse allocations
4. **Profiling:** Valgrind, Tracy Profiler

**Example: Ring Buffer for Trades**
```cpp
class TimeAndSales {
private:
    static constexpr size_t MAX_TRADES = 10000;
    std::deque<Trade> trades_;
    
public:
    void addTrade(const Trade& trade) {
        trades_.push_back(trade);
        if (trades_.size() > MAX_TRADES) {
            trades_.pop_front();  // automatic memory management
        }
    }
};
```

**Acceptance Criteria:**
- [ ] No memory leaks (Valgrind clean)
- [ ] Memory usage stable over 24 hours
- [ ] Vulkan buffers reused (no re-allocation per frame)
- [ ] Trade buffer limited to 10k entries

---

### Task 4.3: Visual Theme & Polish
**Reference:** Screenshot - Dark theme with cyan/magenta accents

**Colors from Screenshot:**
- Background: `#0a0e1a` (very dark blue)
- Panel Background: `#12171f`
- Text: `#e0e0e0` (light gray)
- Bid/Green: `#00ff88`
- Ask/Red: `#ff0066`
- Volume Profile: Cyan `#00d4ff`, Magenta `#ff00aa`

**ImGui Style Setup:**
```cpp
void applyQuantowerTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Window
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameRounding = 2.0f;
    
    // Colors
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.07f, 0.09f, 0.12f, 1.0f);
    colors[ImGuiCol_Border] = ImVec4(0.2f, 0.2f, 0.3f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.1f, 0.1f, 0.15f, 1.0f);
    colors[ImGuiCol_Text] = ImVec4(0.88f, 0.88f, 0.88f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.15f, 0.2f, 0.3f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.2f, 0.3f, 0.4f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.3f, 0.4f, 0.5f, 1.0f);
    
    // Tabs
    colors[ImGuiCol_Tab] = ImVec4(0.1f, 0.15f, 0.2f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.2f, 0.3f, 0.4f, 1.0f);
    colors[ImGuiCol_TabActive] = ImVec4(0.15f, 0.25f, 0.35f, 1.0f);
}
```

**Font Loading:**
```cpp
ImGuiIO& io = ImGui::GetIO();
io.Fonts->AddFontFromFileTTF("fonts/RobotoMono-Regular.ttf", 14.0f);  // monospace
io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 14.0f);  // UI
```

**Acceptance Criteria:**
- [ ] Theme matches Quantower dark mode
- [ ] Monospace font for prices/volumes
- [ ] Smooth gradients (no banding)
- [ ] Consistent spacing/padding
- [ ] High contrast for readability

---

## Phase 5: Testing & Documentation

### Task 5.1: Unit Tests
**Goal:** Test calculations independently

**Files to create:**
```
dependencies/BTQ_Render_Engine/tests/
├── test_volume_profile.cpp
├── test_footprint.cpp
├── test_vwap.cpp
└── CMakeLists.txt
```

**Example Test:**
```cpp
#include <gtest/gtest.h>
#include "widgets/VolumeProfile.h"

TEST(VolumeProfile, CalculatesPOCCorrectly) {
    VolumeProfile profile;
    
    std::vector<Trade> trades = {
        {0, "okx", 91000, 1.0, 91000, TradeSide::Buy},
        {1, "okx", 91000, 2.0, 182000, TradeSide::Sell},  // highest volume
        {2, "okx", 91010, 0.5, 45505, TradeSide::Buy},
    };
    
    profile.calculate(trades, 0, 100);
    
    EXPECT_DOUBLE_EQ(profile.getPOC(), 91000.0);
}
```

**Acceptance Criteria:**
- [ ] 80%+ code coverage for calculations
- [ ] All volume analysis tools tested
- [ ] Edge cases covered (empty data, single trade)
- [ ] Tests run in CI

---

### Task 5.2: Performance Benchmarks
**Goal:** Ensure widgets meet performance targets

**Benchmarks:**
```cpp
#include <benchmark/benchmark.h>

static void BM_FootprintCalculation(benchmark::State& state) {
    std::vector<Trade> trades = generateRandomTrades(10000);
    FootprintChart chart;
    
    for (auto _ : state) {
        chart.calculate(trades);
    }
}
BENCHMARK(BM_FootprintCalculation);

static void BM_VolumeProfileRender(benchmark::State& state) {
    VolumeProfile profile;
    // ... setup
    
    for (auto _ : state) {
        profile.render();
    }
}
BENCHMARK(BM_VolumeProfileRender);
```

**Acceptance Criteria:**
- [ ] Footprint calculation: <50ms for 10k trades
- [ ] Volume profile calculation: <10ms for 10k trades
- [ ] Widget render: <5ms per widget
- [ ] Full dashboard render: <16ms (60 FPS target)

---

### Task 5.3: Documentation
**Goal:** Comprehensive usage and API docs

**Files to create:**
```
dependencies/BTQ_Render_Engine/docs/
├── USER_GUIDE.md
├── API_REFERENCE.md
├── ARCHITECTURE.md
└── PERFORMANCE_TUNING.md
```

**USER_GUIDE.md Contents:**
- How to add/remove panels
- How to configure data types
- How to save/load layouts
- Keyboard shortcuts
- Troubleshooting

**API_REFERENCE.md Contents:**
- Class documentation (Doxygen)
- Function signatures
- Usage examples
- Thread-safety notes

**Acceptance Criteria:**
- [ ] All public APIs documented
- [ ] User guide with screenshots
- [ ] Architecture diagrams
- [ ] Performance tuning guide

---

## Summary of Deliverables

| Phase | Tasks | Files Created | Testing |
|-------|-------|---------------|---------|
| 1 | 3 | 3 docs | Analysis only |
| 2 | 6 | 12+ .cpp/.h, 2+ shaders | Unit tests |
| 3 | 3 | 4 .cpp/.h | Integration tests |
| 4 | 3 | Optimizations | Benchmarks |
| 5 | 3 | Tests + Docs | CI/CD |

**Total:** ~25-30 source files, ~10 documentation files

---

## Technical Notes

### Quantower Data Types Mapping
```cpp
enum class VolumeDataType {
    Trades,              // Count of trades
    BuyTrades,           // Count of buy trades
    SellTrades,          // Count of sell trades
    Volume,              // Total volume
    BuyVolume,           // Buy volume only
    SellVolume,          // Sell volume only
    BuyVolumePercent,    // (Buy / Total) * 100
    SellVolumePercent,   // (Sell / Total) * 100
    BuySellVolume,       // Both displayed
    Delta,               // Buy - Sell
    DeltaPercent,        // (Delta / Total) * 100
    CumulativeDelta,     // Running sum of delta
    AverageSize,         // Avg trade size
    AverageBuySize,      // Avg buy trade size
    AverageSellSize,     // Avg sell trade size
    MaxOneTradeVolume,   // Largest single trade
    FilteredVolume       // Volume > threshold
};
```

### Performance Targets
- **FPS:** 144 (gaming monitor standard)
- **Frame Time:** <7ms
- **Data Latency:** <50ms (WebSocket → Render)
- **Memory:** <500MB total
- **CPU:** <30% on 8-core CPU

### Testing Strategy
1. **Unit Tests:** Calculations (volume profile, VWAP, etc.)
2. **Integration Tests:** Data flow (WebSocket → Widget)
3. **Rendering Tests:** Screenshot comparison
4. **Performance Tests:** Benchmarks, profiling
5. **Stress Tests:** 24-hour runs, memory leaks

---

## FAQ

**Q: Do I need to implement all data types at once?**  
A: No. Start with `Volume`, `BuySellVolume`, and `Delta`. Add others incrementally.

**Q: Should I use Vulkan or ImGui for rendering widgets?**  
A: Use ImGui for UI (tables, buttons, text). Use Vulkan for high-performance visualizations (footprint grid with 10k+ cells, candlesticks).

**Q: How do I handle real-time data updates?**  
A: Use a thread-safe queue. WebSocket thread pushes data, render thread pulls and updates widgets. Use `std::mutex` or lock-free queue.

**Q: What if performance target is missed?**  
A: Profile with Tracy/Optick, identify bottleneck, optimize (reduce draw calls, use instancing, cache calculations).

**Q: How to test without live WebSocket?**  
A: Create mock data generator. Load historical trades from file. Replay at real-time speed.

---

## Next Steps

1. **Start with Task 1.1** - Audit existing code (no modifications)
2. **Read Quantower docs** - Understand expected behavior
3. **Implement one widget at a time** - Test before moving on
4. **Use existing Vulkan pipeline** - Don't reinvent the wheel
5. **Ask for help** - If stuck, check existing code or documentation

Good luck! 🚀