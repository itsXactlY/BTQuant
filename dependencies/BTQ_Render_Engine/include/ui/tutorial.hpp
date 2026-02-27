#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace BTQuant {
namespace UI {

/**
 * @brief Represents a single step in the tutorial
 */
struct TutorialStep {
    std::string title;              ///< Title of the tutorial step
    std::string description;        ///< Detailed description of the step
    std::string target_control_id;  ///< ID of the UI control being highlighted (optional)
    glm::vec2 highlight_position;   ///< Position to highlight on screen
    glm::vec2 highlight_size;       ///< Size of the highlight area

    /**
     * @brief Constructor for TutorialStep
     * @param title Title of the step
     * @param description Description of the step
     * @param target_control_id Control ID to highlight
     * @param highlight_position Position to highlight
     * @param highlight_size Size of highlight area
     */
    TutorialStep(const std::string& title,
                 const std::string& description,
                 const std::string& target_control_id = "",
                 const glm::vec2& highlight_position = glm::vec2(0.0f, 0.0f),
                 const glm::vec2& highlight_size = glm::vec2(0.0f, 0.0f));
};

/**
 * @brief Manages the first-run tutorial experience
 * 
 * The TutorialManager handles the guided tour for new users, walking them
 * through key features of the application step-by-step. It manages the
 * tutorial state, UI rendering, and progression through tutorial steps.
 */
class TutorialManager {
public:
    /**
     * @brief Constructor that initializes default tutorial steps
     */
    TutorialManager();

    /**
     * @brief Start the tutorial
     */
    void start_tutorial();

    /**
     * @brief Stop the tutorial
     */
    void stop_tutorial();

    /**
     * @brief Move to the next tutorial step
     */
    void next_step();

    /**
     * @brief Move to the previous tutorial step
     */
    void previous_step();

    /**
     * @brief Render the tutorial UI window
     */
    void render_tutorial_window();

    /**
     * @brief Check if tutorial should be shown on startup
     * @return True if tutorial should be shown on first run
     */
    bool should_show_on_startup() const;

    /**
     * @brief Set whether tutorial should be shown on startup
     * @param show True to show tutorial on startup, false otherwise
     */
    void set_show_on_startup(bool show);

    /**
     * @brief Update tutorial state (called each frame)
     * @param dt Delta time since last frame
     */
    void update(float dt);

    /**
     * @brief Mark tutorial as shown/not shown
     * @param shown True if tutorial has been shown
     */
    void set_has_been_shown(bool shown);

    /**
     * @brief Check if tutorial has been shown
     * @return True if tutorial has been shown
     */
    bool has_been_shown() const;

private:
    /**
     * @brief Initialize the default tutorial steps
     */
    void initialize_tutorial_steps();

    /**
     * @brief Draw the highlight overlay for the current step
     */
    void draw_highlight_overlay();

    std::vector<TutorialStep> steps_;    ///< Collection of tutorial steps
    bool is_active_;                     ///< Whether the tutorial is currently active
    size_t current_step_;                ///< Current step index
    bool show_tutorial_on_startup_;      ///< Whether to show tutorial on first run
    bool has_been_shown_;                ///< Whether the tutorial has been shown
    float window_alpha_;                 ///< Alpha value for tutorial window transparency
};

/**
 * @brief Get the global tutorial manager instance
 * @return Reference to the global TutorialManager
 */
TutorialManager& get_global_tutorial_manager();

/**
 * @brief Show the tutorial if it's the first run
 */
void show_tutorial_if_first_run();

/**
 * @brief Start the tutorial
 */
void start_tutorial();

/**
 * @brief Stop the tutorial
 */
void stop_tutorial();

/**
 * @brief Render the tutorial UI
 */
void render_tutorial();

} // namespace UI
} // namespace BTQuant