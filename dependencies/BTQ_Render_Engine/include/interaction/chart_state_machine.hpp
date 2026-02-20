#pragma once

/**
 * @file chart_state_machine.hpp
 * @brief Event-Driven State Machine for Chart Interactions
 * 
 * This implementation provides:
 * - State machine for chart interaction modes
 * - Event handling for mouse, keyboard, and touch input
 * - Transition management between states
 * - Support for panning, drawing, modifying, and zooming
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace btq {
namespace interaction {

/**
 * @brief Interaction states
 */
enum class InteractionState : uint8_t {
    IDLE,               // No active interaction
    PANNING,            // Dragging to pan the chart
    ZOOMING,            // Zooming with scroll wheel or pinch
    DRAWING,            // Drawing a new tool
    MODIFYING,          // Modifying an existing drawing
    SELECTING,          // Selecting a region or tool
    CROSSHAIR_TRACKING, // Tracking crosshair position
    WAITING_FOR_CLICK,  // Waiting for user click (e.g., for order entry)
    CONTEXT_MENU,       // Context menu is open
    MEASURING           // Measuring distance between two points
};

/**
 * @brief Drawing tool types
 */
enum class DrawingToolType : uint8_t {
    NONE,
    TRENDLINE,
    HORIZONTAL_LINE,
    VERTICAL_LINE,
    RECTANGLE,
    FIBONACCI_RETRACEMENT,
    TEXT_LABEL,
    ARROW,
    PARALLEL_CHANNEL,
    ANDREW_PITCHFORK
};

/**
 * @brief Input event types
 */
struct MouseEvent {
    enum class Type : uint8_t {
        PRESS,
        RELEASE,
        MOVE,
        DRAG,
        SCROLL,
        ENTER,
        LEAVE
    };
    
    Type type;
    glm::vec2 position;         // Screen position
    glm::vec2 chart_position;   // Chart coordinates (time, price)
    int button;                 // 0=left, 1=middle, 2=right
    int modifiers;              // Shift, Ctrl, Alt flags
    glm::vec2 scroll_delta;     // For scroll events
    bool is_double_click = false;
};

struct KeyEvent {
    enum class Type : uint8_t {
        PRESS,
        RELEASE,
        REPEAT
    };
    
    Type type;
    int key;
    int scancode;
    int modifiers;
    bool is_repeat = false;
};

struct TouchEvent {
    enum class Type : uint8_t {
        BEGIN,
        MOVE,
        END,
        CANCEL
    };
    
    Type type;
    std::vector<glm::vec2> positions;  // Multiple touch points
    float scale = 1.0f;                // Pinch scale factor
    float rotation = 0.0f;             // Rotation angle
};

using InputEvent = std::variant<MouseEvent, KeyEvent, TouchEvent>;

/**
 * @brief State transition result
 */
struct TransitionResult {
    InteractionState new_state;
    bool handled;
    bool redraw_required;
    std::string status_message;
};

/**
 * @brief State handler function type
 */
using StateHandler = std::function<TransitionResult(const InputEvent&, InteractionState)>;

/**
 * @brief Drawing context for current drawing operation
 */
struct DrawingContext {
    DrawingToolType tool_type = DrawingToolType::NONE;
    glm::vec2 start_point;      // Starting point in chart coordinates
    glm::vec2 current_point;    // Current point in chart coordinates
    glm::vec2 end_point;        // End point (when completed)
    bool is_snapping = false;   // Magnet mode enabled
    int snap_candle_index = -1; // Candle index for snapping
    float snap_price = 0.0f;    // Snapped price
    std::vector<glm::vec2> intermediate_points;  // For multi-point tools
};

/**
 * @brief Selection context for current selection
 */
struct SelectionContext {
    glm::vec2 start_point;
    glm::vec2 end_point;
    bool is_active = false;
    int selected_tool_id = -1;  // ID of selected drawing tool
    int selected_handle = -1;   // Handle index for modification
};

/**
 * @brief Event-driven state machine for chart interactions
 */
class ChartStateMachine {
public:
    ChartStateMachine() {
        initializeStateHandlers();
    }
    
    /**
     * @brief Get the current state
     */
    InteractionState getCurrentState() const {
        return current_state_.load();
    }
    
    /**
     * @brief Get the previous state
     */
    InteractionState getPreviousState() const {
        return previous_state_;
    }
    
    /**
     * @brief Process an input event
     */
    TransitionResult processEvent(const InputEvent& event) {
        auto it = state_handlers_.find(current_state_);
        if (it != state_handlers_.end()) {
            auto result = it->second(event, current_state_);
            
            if (result.new_state != current_state_) {
                previous_state_ = current_state_;
                current_state_.store(result.new_state);
                
                // Call state change callback
                if (state_change_callback_) {
                    state_change_callback_(previous_state_, result.new_state);
                }
            }
            
            return result;
        }
        
        return {current_state_, false, false, ""};
    }
    
    /**
     * @brief Set the active drawing tool
     */
    void setActiveDrawingTool(DrawingToolType tool) {
        active_tool_ = tool;
        if (tool != DrawingToolType::NONE && current_state_ == InteractionState::IDLE) {
            current_state_.store(InteractionState::DRAWING);
        }
    }
    
    /**
     * @brief Get the active drawing tool
     */
    DrawingToolType getActiveDrawingTool() const {
        return active_tool_;
    }
    
    /**
     * @brief Get the drawing context
     */
    DrawingContext& getDrawingContext() {
        return drawing_context_;
    }
    
    /**
     * @brief Get the selection context
     */
    SelectionContext& getSelectionContext() {
        return selection_context_;
    }
    
    /**
     * @brief Set state change callback
     */
    void setStateChangeCallback(std::function<void(InteractionState, InteractionState)> callback) {
        state_change_callback_ = std::move(callback);
    }
    
    /**
     * @brief Force a state transition
     */
    void forceState(InteractionState new_state) {
        previous_state_ = current_state_;
        current_state_.store(new_state);
        
        if (state_change_callback_) {
            state_change_callback_(previous_state_, new_state);
        }
    }
    
    /**
     * @brief Check if currently drawing
     */
    bool isDrawing() const {
        return current_state_ == InteractionState::DRAWING;
    }
    
    /**
     * @brief Check if currently panning
     */
    bool isPanning() const {
        return current_state_ == InteractionState::PANNING;
    }
    
    /**
     * @brief Check if currently modifying
     */
    bool isModifying() const {
        return current_state_ == InteractionState::MODIFYING;
    }
    
    /**
     * @brief Cancel current operation
     */
    void cancel() {
        drawing_context_ = DrawingContext{};
        selection_context_ = SelectionContext{};
        active_tool_ = DrawingToolType::NONE;
        current_state_.store(InteractionState::IDLE);
    }

private:
    void initializeStateHandlers() {
        // IDLE state handler
        state_handlers_[InteractionState::IDLE] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::IDLE, false, false, ""};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::PRESS:
                        if (mouse.button == 0) {  // Left click
                            if (active_tool_ != DrawingToolType::NONE) {
                                result.new_state = InteractionState::DRAWING;
                                drawing_context_.tool_type = active_tool_;
                                drawing_context_.start_point = mouse.chart_position;
                                drawing_context_.current_point = mouse.chart_position;
                                result.redraw_required = true;
                            } else if (mouse.modifiers & 0x01) {  // Shift
                                result.new_state = InteractionState::SELECTING;
                                selection_context_.start_point = mouse.chart_position;
                                selection_context_.is_active = true;
                            } else {
                                // Check if clicking on existing tool
                                if (selection_context_.selected_tool_id >= 0) {
                                    result.new_state = InteractionState::MODIFYING;
                                } else {
                                    result.new_state = InteractionState::PANNING;
                                }
                            }
                        } else if (mouse.button == 2) {  // Right click
                            result.new_state = InteractionState::CONTEXT_MENU;
                        }
                        result.handled = true;
                        break;
                        
                    case MouseEvent::Type::SCROLL:
                        result.new_state = InteractionState::ZOOMING;
                        result.handled = true;
                        result.redraw_required = true;
                        break;
                        
                    case MouseEvent::Type::MOVE:
                        result.new_state = InteractionState::CROSSHAIR_TRACKING;
                        result.handled = true;
                        result.redraw_required = true;
                        break;
                        
                    default:
                        break;
                }
            }
            
            return result;
        };
        
        // PANNING state handler
        state_handlers_[InteractionState::PANNING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::PANNING, true, true, "Panning"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::RELEASE:
                        if (mouse.button == 0) {
                            result.new_state = InteractionState::IDLE;
                        }
                        break;
                        
                    case MouseEvent::Type::DRAG:
                        // Pan delta is calculated from position change
                        result.redraw_required = true;
                        break;
                        
                    default:
                        break;
                }
            }
            
            return result;
        };
        
        // ZOOMING state handler
        state_handlers_[InteractionState::ZOOMING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::ZOOMING, true, true, "Zooming"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                if (mouse.type == MouseEvent::Type::SCROLL) {
                    // Zoom is processed, return to idle
                    result.new_state = InteractionState::IDLE;
                }
            }
            
            return result;
        };
        
        // DRAWING state handler
        state_handlers_[InteractionState::DRAWING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::DRAWING, true, true, "Drawing"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::MOVE:
                    case MouseEvent::Type::DRAG:
                        drawing_context_.current_point = mouse.chart_position;
                        result.redraw_required = true;
                        break;
                        
                    case MouseEvent::Type::RELEASE:
                        if (mouse.button == 0) {
                            drawing_context_.end_point = mouse.chart_position;
                            // Drawing completed, return to idle
                            result.new_state = InteractionState::IDLE;
                            result.status_message = "Drawing completed";
                            
                            // Keep tool active for multiple drawings
                            if (mouse.modifiers & 0x01) {  // Shift held
                                result.new_state = InteractionState::DRAWING;
                                drawing_context_ = DrawingContext{};
                                drawing_context_.tool_type = active_tool_;
                            }
                        }
                        break;
                        
                    case MouseEvent::Type::PRESS:
                        if (mouse.button == 2) {  // Right click cancels
                            cancel();
                            result.new_state = InteractionState::IDLE;
                            result.status_message = "Drawing cancelled";
                        }
                        break;
                        
                    default:
                        break;
                }
            } else if (std::holds_alternative<KeyEvent>(event)) {
                const auto& key = std::get<KeyEvent>(event);
                if (key.type == KeyEvent::Type::PRESS && key.key == 256) {  // Escape
                    cancel();
                    result.new_state = InteractionState::IDLE;
                    result.status_message = "Drawing cancelled";
                }
            }
            
            return result;
        };
        
        // MODIFYING state handler
        state_handlers_[InteractionState::MODIFYING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::MODIFYING, true, true, "Modifying"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::DRAG:
                        // Update tool position
                        result.redraw_required = true;
                        break;
                        
                    case MouseEvent::Type::RELEASE:
                        result.new_state = InteractionState::IDLE;
                        result.status_message = "Modification completed";
                        break;
                        
                    default:
                        break;
                }
            }
            
            return result;
        };
        
        // SELECTING state handler
        state_handlers_[InteractionState::SELECTING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::SELECTING, true, true, "Selecting"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::DRAG:
                        selection_context_.end_point = mouse.chart_position;
                        result.redraw_required = true;
                        break;
                        
                    case MouseEvent::Type::RELEASE:
                        selection_context_.end_point = mouse.chart_position;
                        result.new_state = InteractionState::IDLE;
                        result.status_message = "Selection completed";
                        break;
                        
                    default:
                        break;
                }
            }
            
            return result;
        };
        
        // CROSSHAIR_TRACKING state handler
        state_handlers_[InteractionState::CROSSHAIR_TRACKING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::CROSSHAIR_TRACKING, true, true, ""};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                if (mouse.type == MouseEvent::Type::MOVE) {
                    result.redraw_required = true;
                } else if (mouse.type == MouseEvent::Type::LEAVE) {
                    result.new_state = InteractionState::IDLE;
                } else if (mouse.type == MouseEvent::Type::PRESS) {
                    // Transition to appropriate state based on button
                    if (mouse.button == 0) {
                        result.new_state = InteractionState::PANNING;
                    }
                }
            }
            
            return result;
        };
        
        // CONTEXT_MENU state handler
        state_handlers_[InteractionState::CONTEXT_MENU] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::CONTEXT_MENU, true, false, ""};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                if (mouse.type == MouseEvent::Type::PRESS || 
                    mouse.type == MouseEvent::Type::RELEASE) {
                    result.new_state = InteractionState::IDLE;
                }
            }
            
            return result;
        };
        
        // MEASURING state handler
        state_handlers_[InteractionState::MEASURING] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::MEASURING, true, true, "Measuring"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                switch (mouse.type) {
                    case MouseEvent::Type::MOVE:
                        result.redraw_required = true;
                        break;
                        
                    case MouseEvent::Type::RELEASE:
                        result.new_state = InteractionState::IDLE;
                        break;
                        
                    default:
                        break;
                }
            }
            
            return result;
        };
        
        // WAITING_FOR_CLICK state handler
        state_handlers_[InteractionState::WAITING_FOR_CLICK] = [this](const InputEvent& event, InteractionState) {
            TransitionResult result{InteractionState::WAITING_FOR_CLICK, true, false, "Click to confirm"};
            
            if (std::holds_alternative<MouseEvent>(event)) {
                const auto& mouse = std::get<MouseEvent>(event);
                
                if (mouse.type == MouseEvent::Type::PRESS && mouse.button == 0) {
                    result.new_state = InteractionState::IDLE;
                    result.status_message = "Action confirmed";
                } else if (mouse.type == MouseEvent::Type::PRESS && mouse.button == 2) {
                    result.new_state = InteractionState::IDLE;
                    result.status_message = "Action cancelled";
                }
            }
            
            return result;
        };
    }
    
    std::atomic<InteractionState> current_state_{InteractionState::IDLE};
    InteractionState previous_state_ = InteractionState::IDLE;
    
    DrawingToolType active_tool_ = DrawingToolType::NONE;
    DrawingContext drawing_context_;
    SelectionContext selection_context_;
    
    std::unordered_map<InteractionState, StateHandler> state_handlers_;
    std::function<void(InteractionState, InteractionState)> state_change_callback_;
};

/**
 * @brief Input modifier flags
 */
namespace modifiers {
    constexpr int SHIFT = 0x01;
    constexpr int CTRL = 0x02;
    constexpr int ALT = 0x04;
    constexpr int SUPER = 0x08;
}

/**
 * @brief Mouse button constants
 */
namespace mouse_button {
    constexpr int LEFT = 0;
    constexpr int MIDDLE = 1;
    constexpr int RIGHT = 2;
}

} // namespace interaction
} // namespace btq