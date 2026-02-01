# Click-to-Switch Functionality Analysis

## Status: ALREADY IMPLEMENTED

The click-to-switch functionality described in the task is already fully implemented in the codebase.

## Implementation Details

### 1. Watchlist Panel (`dependencies/BTQ_Render_Engine/src/components/watchlist_panel.cpp`)
- The `WatchlistPanel` class has a `SymbolSelectedCallback` mechanism
- The `render_table_row` method handles clicks on symbol rows
- When a symbol is clicked, it triggers the callback with the symbol ID and name

### 2. Callback Registration (`dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp`)
- In the `PanelManager`, when a `WatchlistPanel` is created, the callback is registered:
```cpp
watchlist->set_symbol_selected_callback(
    [this](uint32_t symbol_id, const std::string& symbol_name) {
      this->set_active_symbol(symbol_id, symbol_name);
    });
```

### 3. Symbol Propagation (`dependencies/BTQ_Render_Engine/src/components/panel_manager.cpp`)
- The `set_active_symbol` method updates all relevant panels with the new symbol:
  - Charts
  - Order books
  - Time & Sales
  - Volume Profiles
  - Depth Charts
  - Footprint Charts
  - TPO Profiles
  - DOM Surfaces
  - And other symbol-dependent panels

## How It Works
1. User clicks on any symbol in the watchlist
2. The click event triggers the `SymbolSelectedCallback`
3. The callback calls `PanelManager::set_active_symbol()` with the selected symbol
4. `set_active_symbol()` updates the symbol for all panels that support symbol-specific data
5. All relevant panels switch to display the selected symbol

## Conclusion
The functionality described in the task "Implement click-to-switch: clicking any symbol in watchlist changes all panels to display that symbol" is already fully implemented and operational.