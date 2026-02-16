# TASK_CHART_ANATOMY.md: The Quantower Chart Clone

**Objective:** Transform `ChartPanel` into a 5-region, lock-free trading interface.
**Core Principle:** UI layout must use `ImGui::BeginChild` for rigid regional division. All price/volume data MUST be read via `std::memory_order_relaxed` from the `MarketDataProcessor` atomic snapshots.

---

## Phase 1: The Architectural Skeleton (5-Region Split)
**Goal:** Divide the ImGui window into the 5 distinct anatomical regions before filling them with logic.

- [ ] **1.1: Define Layout Constants (`include/components/chart_panel.hpp`)**
    - Add layout dimensions to the class:
      ```cpp
      static constexpr float TOP_BAR_HEIGHT = 32.0f;
      static constexpr float BOTTOM_BAR_HEIGHT = 32.0f;
      static constexpr float LEFT_SIDEBAR_WIDTH = 45.0f;
      static constexpr float RIGHT_SIDEBAR_WIDTH = 220.0f;
      ```

- [ ] **1.2: Implement the Regional Grid (`src/components/chart_panel.cpp`)**
    - Inside `ChartPanel::render()`, replace the generic render logic with the following structural scaffold:
      ```cpp
      // 1. Top Toolbar (Spans full width)
      ImGui::BeginChild("ChartTopBar", ImVec2(0, TOP_BAR_HEIGHT), false, ImGuiWindowFlags_NoScrollbar);
      render_top_toolbar();
      ImGui::EndChild();

      // Middle Section (Contains Left Bar, Chart, Right Bar)
      float middle_height = ImGui::GetContentRegionAvail().y - BOTTOM_BAR_HEIGHT;
      ImGui::BeginChild("ChartMiddleRegion", ImVec2(0, middle_height), false, ImGuiWindowFlags_NoScrollbar);
      
      // 2. Left Sidebar
      ImGui::BeginChild("ChartLeftSidebar", ImVec2(LEFT_SIDEBAR_WIDTH, 0), true, ImGuiWindowFlags_NoScrollbar);
      render_left_sidebar();
      ImGui::EndChild();
      ImGui::SameLine();

      // 3. Main Chart Area (Dynamic width)
      float chart_width = ImGui::GetContentRegionAvail().x - RIGHT_SIDEBAR_WIDTH;
      ImGui::BeginChild("ChartMainArea", ImVec2(chart_width, 0), false);
      render_main_chart();
      ImGui::EndChild();
      ImGui::SameLine();

      // 4. Right Sidebar (Order Entry)
      ImGui::BeginChild("ChartOrderEntry", ImVec2(RIGHT_SIDEBAR_WIDTH, 0), true);
      render_order_entry();
      ImGui::EndChild();

      ImGui::EndChild(); // End Middle Region

      // 5. Bottom Toolbar
      ImGui::BeginChild("ChartBottomBar", ImVec2(0, BOTTOM_BAR_HEIGHT), false, ImGuiWindowFlags_NoScrollbar);
      render_bottom_toolbar();
      ImGui::EndChild();
      ```

---

## Phase 2: Toolbars & Sidebars (The Controls)
**Goal:** Populate the outer regions with interactive components.

- [ ] **2.1: Top Toolbar (`render_top_toolbar`)**
    - Add `ImGui::InputText` for Symbol Lookup (wires to `processor_->set_active_symbol()`).
    - Add `ImGui::Combo` for all available Timeframes (1ms to 1m, 5m, 15m, 1H, 1D).
    - Add `ImGui::Combo` for Chart Style (Candles, Bars, Line, Quantower Style).
    - Add `ImGui::Button` toggle for "Keyboard Trading" vs "Mouse Trading".

- [ ] **2.2: Left Sidebar (`render_left_sidebar`)**
    - Render icon buttons for Drawing Tools (Crosshair, Trendline, Fibonacci).
    - **Favorite System:** Add a right-click context menu to these buttons (`ImGui::BeginPopupContextItem`). Allow the user to toggle an `is_favorite` boolean. Draw a small ImGui star icon next to favorited tools.

- [ ] **2.3: Bottom Toolbar (`render_bottom_toolbar`)**
    - Render toggle buttons (`ImGui::Selectable` or custom styled buttons) for Volume Analysis overlays: "Volume Profile", "Delta", "Cumulative Delta".

---

## Phase 3: Main Chart Area (ImPlot & Viewport Math)
**Goal:** Implement the 4 Quantower price centering modes and the "Snap to Last" logic.

- [ ] **3.1: Price Scale Modes (`render_main_chart`)**
    - Define `enum class PriceScaleMode { AUTO, AUTO_CENTERED, KEEP_IN_VIEW, MANUAL };`.
    - Retrieve the latest atomic price: 
      `double last_price = processor_->get_atomic_snapshot(active_sym)->price.load(std::memory_order_relaxed);`
    - **Logic Implementation before `ImPlot::BeginPlot`:**
        - `AUTO`: Use `ImPlotAxisFlags_AutoFit`.
        - `AUTO_CENTERED`: Manually calculate Y limits: `ImPlot::SetupAxisLimits(ImAxis_Y1, last_price - range, last_price + range, ImPlotCond_Always);`.
        - `KEEP_IN_VIEW`: Check current ImPlot Y limits. If `last_price` > `Y_max` or < `Y_min`, shift the limits.
        - `MANUAL`: Use `ImPlotCond_Once`.
    - **Drag Detection:** If `ImPlot::IsPlotHovered()` and `ImGui::IsMouseDragging(1)`, force `mode = PriceScaleMode::MANUAL`.

- [ ] **3.2: "Snap to Last" Button**
    - Calculate current X-axis bounds via `ImPlot::GetPlotLimits().X.Max`.
    - Compare with the `latest_timestamp` from the atomic snapshot.
    - If `X.Max` < `latest_timestamp` (user scrolled left), render a floating `ImGui::Button("Snap to Last")` in the bottom right corner of the chart area (`ImGui::SetCursorPos`).
    - On click: Force X-axis limits to `latest_timestamp` and add the configured right-margin offset.

---

## Phase 4: Right Sidebar (Lock-Free Order Entry)
**Goal:** Implement the quick trading interface using pure atomic reads and SPSC queue writes.

- [ ] **4.1: Atomic Price Feeds (`render_order_entry`)**
    - Read best bid/ask immediately without locking:
      ```cpp
      auto* snap = processor_->get_atomic_snapshot(active_sym);
      double bid = snap->best_bid.load(std::memory_order_relaxed);
      double ask = snap->best_ask.load(std::memory_order_relaxed);
      ```
    - Render massive "BUY MKT" (Ask Price, Green) and "SELL MKT" (Bid Price, Red) buttons using `ImGui::PushStyleColor`.

- [ ] **4.2: Asynchronous Trade Command Routing**
    - Add `ImGui::InputInt("Qty", &order_qty);` and `ImGui::Combo("TIF", &order_tif, ...);`.
    - **The Lock-Free Write:** When a button is clicked, push the command to the execution queue:
      ```cpp
      if (ImGui::Button("BUY MKT")) {
          trade_command_queue_.enqueue({
              .type = OrderType::MARKET,
              .side = OrderSide::BUY,
              .qty = order_qty,
              .symbol_id = active_sym
          });
      }
      ```
    - *Crucial:* The UI thread MUST NOT execute network calls or database writes here. It fires the struct into the `trade_command_queue_` and continues rendering.