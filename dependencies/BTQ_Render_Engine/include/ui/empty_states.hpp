#ifndef BTQ_RENDER_ENGINE_EMPTY_STATES_HPP
#define BTQ_RENDER_ENGINE_EMPTY_STATES_HPP

#include <string>
#include <vector>
#include <functional>
#include <imgui.h>

namespace btq {
namespace ui {

/**
 * @brief Represents an empty state with helpful message and suggested next steps
 */
struct EmptyStateConfig {
    std::string title;              ///< Main title for the empty state
    std::string message;            ///< Detailed message explaining the empty state
    std::vector<std::string> suggestions;  ///< List of suggested next steps
    std::string icon;               ///< Optional icon character or emoji
    ImVec4 color;                   ///< Color for the empty state elements
    float scale;                    ///< Scale factor for the empty state UI
};

/**
 * @brief Manages and renders empty states in the UI
 *
 * This class provides functionality to display helpful messages when no data is available,
 * along with suggestions for next steps to guide the user.
 */
class EmptyStateManager {
public:
    /**
     * @brief Construct a new Empty State Manager object
     */
    EmptyStateManager();

    /**
     * @brief Destroy the Empty State Manager object
     */
    ~EmptyStateManager();

    /**
     * @brief Render an empty state with the given configuration
     * @param config Configuration for the empty state
     */
    void renderEmptyState(const EmptyStateConfig& config);

    /**
     * @brief Render a simple empty state with just a message
     * @param message Message to display
     * @param title Optional title for the empty state
     */
    void renderSimpleEmptyState(const std::string& message, const std::string& title = "No Data");

    /**
     * @brief Render a trading-specific empty state
     * @param context Context for the empty state (e.g., "chart", "orders", "positions")
     */
    void renderTradingEmptyState(const std::string& context);

    /**
     * @brief Render an analytical empty state
     * @param context Context for the empty state (e.g., "indicators", "analysis", "reports")
     */
    void renderAnalyticalEmptyState(const std::string& context);

    /**
     * @brief Render a data feed empty state
     * @param context Context for the empty state (e.g., "market data", "news", "alerts")
     */
    void renderDataFeedEmptyState(const std::string& context);

    /**
     * @brief Render an empty state with action buttons
     * @param config Configuration for the empty state
     * @param button_labels Labels for action buttons to display
     * @param button_callbacks Callbacks for each button
     */
    void renderActionableEmptyState(
        const EmptyStateConfig& config,
        const std::vector<std::string>& button_labels,
        const std::vector<std::function<void()>>& button_callbacks);

private:
    /**
     * @brief Render the icon for the empty state
     * @param icon Icon character or emoji to display
     * @param color Color for the icon
     * @param scale Scale factor for the icon
     */
    void renderIcon(const std::string& icon, const ImVec4& color, float scale);

    /**
     * @brief Render the title for the empty state
     * @param title Title text to display
     * @param color Color for the title
     */
    void renderTitle(const std::string& title, const ImVec4& color);

    /**
     * @brief Render the main message for the empty state
     * @param message Message text to display
     * @param color Color for the message
     */
    void renderMessage(const std::string& message, const ImVec4& color);

    /**
     * @brief Render suggestions for next steps
     * @param suggestions List of suggestion texts
     * @param color Color for the suggestions
     */
    void renderSuggestions(const std::vector<std::string>& suggestions, const ImVec4& color);
};

} // namespace ui
} // namespace btq

#endif // BTQ_RENDER_ENGINE_EMPTY_STATES_HPP