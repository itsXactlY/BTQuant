#pragma once

#include <string>
#include <functional>

#include "imgui.h"

namespace BTQuant {

// Interface for panel-specific settings
class PanelSettingsInterface {
public:
    virtual ~PanelSettingsInterface() = default;
    
    // Render the settings UI
    virtual void render() = 0;
    
    // Get the title for the settings modal
    virtual std::string get_title() const = 0;
    
    // Check if settings modal is open
    virtual bool is_open() const = 0;
    
    // Open the settings modal
    virtual void open() = 0;
    
    // Close the settings modal
    virtual void close() = 0;
    
    // Toggle the settings modal
    virtual void toggle() = 0;
};

// Base class for panel-specific settings implementations
class BasePanelSettings : public PanelSettingsInterface {
protected:
    bool is_modal_open_ = false;
    std::string title_;
    
public:
    explicit BasePanelSettings(const std::string& title) : title_(title) {}
    
    bool is_open() const override { return is_modal_open_; }
    void open() override { is_modal_open_ = true; }
    void close() override { is_modal_open_ = false; }
    void toggle() override { is_modal_open_ = !is_modal_open_; }
    std::string get_title() const override { return title_; }
};

} // namespace BTQuant