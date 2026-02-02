# Per-Panel Settings Implementation Documentation

## Overview
This implementation adds per-panel settings functionality to the trading terminal application. Each panel type can now have its own dedicated settings modal accessible via a settings button in the panel header or through a right-click context menu.

## Files Added

### 1. `panel_settings_interface.hpp`
- Defines the base interface for panel-specific settings (`PanelSettingsInterface`)
- Provides a base class (`BasePanelSettings`) with common functionality

### 2. `chart_panel_settings.hpp/cpp`
- Implements settings for the ChartPanel with comprehensive indicator configuration options
- Includes persistence functionality using JSON files

## Files Modified

### 1. `panel_base.hpp/cpp`
- Added `get_settings_interface()` and `open_settings()` virtual methods
- Added `render_context_menu()` virtual method
- Enhanced `render_panel_header()` to include a settings button (⚙) when available
- Added right-click context menu support in the panel header

### 2. `chart_panel.hpp/cpp`
- Integrated `ChartPanelSettings` as a member
- Implemented the settings interface methods
- Added settings button functionality to the chart panel

## Features Implemented

### 1. Settings Button in Panel Header
- A gear icon (⚙) appears in the panel header when the panel supports settings
- Clicking the button opens the panel-specific settings modal

### 2. Context Menu Support
- Right-clicking on the panel header opens a context menu
- The context menu contains a "Panel Settings" option

### 3. Chart Panel Settings
- Comprehensive indicator configuration (SMA, EMA, RSI, MACD, etc.)
- Individual parameter controls for each indicator
- Apply, Cancel, and Reset functionality

### 4. Settings Persistence
- Settings are saved to JSON files in a `settings/` directory
- Each panel gets a unique settings file based on its memory address
- Settings are automatically loaded when the panel is created

## Technical Details

### Architecture
```
PanelSettingsInterface (Base interface)
    ↓
BasePanelSettings (Common implementation)
    ↓
ChartPanelSettings (Panel-specific implementation)
```

### Integration Points
- `PanelBase::render_panel_header()` - Adds settings button
- `PanelBase::render()` - Renders settings modal if available
- `PanelBase::render_context_menu()` - Virtual method for context menu
- `ChartPanel` - Integrates settings functionality

### Persistence
- Settings are stored in JSON format
- File naming convention: `settings/chart_panel_[memory_address].json`
- Automatic save/load on settings apply/close and panel initialization

## Usage

### For Developers Adding New Panel Types
1. Create a new settings class inheriting from `BasePanelSettings`
2. Override the `render()` method to implement the UI
3. Add a member variable in your panel class to hold the settings instance
4. Override `get_settings_interface()` and `open_settings()` methods
5. Optionally override `render_context_menu()` for custom context menu items

### For Users
1. Click the gear icon (⚙) in any panel header to open settings
2. Alternatively, right-click the panel header and select "Panel Settings"
3. Adjust settings as needed
4. Click "Apply" to save and apply changes, or "Cancel" to discard

## Future Enhancements
- Add settings for other panel types (OrderBook, Watchlist, etc.)
- Implement a centralized settings management system
- Add import/export functionality for settings
- Create backup/restore mechanisms for settings