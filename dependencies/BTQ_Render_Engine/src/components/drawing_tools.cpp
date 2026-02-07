#include "../include/components/drawing_tools.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace BTQuant {

DrawingToolsManager::DrawingToolsManager() {
    // Initialize the drawing tools manager
}

DrawingToolsManager::~DrawingToolsManager() {
    // Clean up resources
    clear_all_tools();
}

void DrawingToolsManager::add_tool(std::unique_ptr<DrawingTool> tool) {
    if (tool) {
        tools_.push_back(std::move(tool));
    }
}

void DrawingToolsManager::remove_tool(const std::string& id) {
    tools_.erase(
        std::remove_if(tools_.begin(), tools_.end(),
            [&id](const std::unique_ptr<DrawingTool>& tool) {
                return tool->id == id;
            }),
        tools_.end()
    );
}

void DrawingToolsManager::clear_all_tools() {
    tools_.clear();
}

void DrawingToolsManager::render_all() {
    for (auto& tool : tools_) {
        if (tool->visible) {
            tool->render();
        }
    }

    // Also render the tool currently being created
    if (creating_tool_) {
        creating_tool_->render();
    }
}

void DrawingToolsManager::toggle_visibility(const std::string& id) {
    for (auto& tool : tools_) {
        if (tool->id == id) {
            tool->visible = !tool->visible;
            break;
        }
    }
}

void DrawingToolsManager::update_tool_color(const std::string& id, ImVec4 new_color) {
    for (auto& tool : tools_) {
        if (tool->id == id) {
            tool->color = new_color;
            break;
        }
    }
}

void DrawingToolsManager::update_tool_thickness(const std::string& id, float new_thickness) {
    for (auto& tool : tools_) {
        if (tool->id == id) {
            tool->thickness = new_thickness;
            break;
        }
    }
}

void DrawingToolsManager::render_ui_controls() {
    if (!show_ui_controls_) return;

    // Check if we're in a valid ImGui frame scope to prevent assertion errors
    ImGuiContext& g = *GImGui;
    if (!g.WithinFrameScope) {
        // If we're not within a frame scope, skip rendering this frame to avoid the assertion
        // The drawing tools UI will be rendered in the next frame when the scope is valid
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Drawing Tools", &show_ui_controls_)) {
        // Tool selection and creation
        static int selected_tool_type = 0;
        const char* tool_types[] = { "Trend Line", "Horizontal Line", "Fibonacci", "Rectangle", "Text Annotation" };

        if (ImGui::Combo("Tool Type", &selected_tool_type, tool_types, IM_ARRAYSIZE(tool_types))) {
            // Update the current tool type when selection changes
            current_tool_type_ = selected_tool_type;
        }

        // Tool properties
        static ImVec4 current_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        static float current_thickness = 1.0f;
        static std::string current_text = "Note";
        static float current_font_size = 14.0f;

        ImGui::ColorEdit4("Color", (float*)&current_color);
        ImGui::SliderFloat("Thickness", &current_thickness, 0.5f, 5.0f, "%.1f");

        if (selected_tool_type == 4) { // Text Annotation
            char text_buffer[256];
            strncpy(text_buffer, current_text.c_str(), sizeof(text_buffer) - 1);
            text_buffer[sizeof(text_buffer) - 1] = '\0';
            if (ImGui::InputText("Text", text_buffer, sizeof(text_buffer))) {
                current_text = std::string(text_buffer);
            }
            ImGui::SliderFloat("Font Size", &current_font_size, 8.0f, 24.0f, "%.0f");
        }

        // Add button to create new tool
        if (ImGui::Button("Add New Tool")) {
            // In a real implementation, we would capture mouse clicks on the chart
            // to determine where to place the tool. For now, we'll just add a sample tool.

            // Generate a unique ID for the new tool
            static int tool_counter = 0;
            std::string new_id = "tool_" + std::to_string(++tool_counter);

            switch (selected_tool_type) {
                case 0: { // Trend Line
                    // In real implementation, we would capture two points from user interaction
                    // For now, using sample points
                    ImVec2 p1(1.0f, 100.0f);
                    ImVec2 p2(10.0f, 150.0f);
                    auto tool = std::make_unique<TrendLine>(new_id, p1, p2);
                    tool->color = current_color;
                    tool->thickness = current_thickness;
                    add_tool(std::move(tool));
                    break;
                }
                case 1: { // Horizontal Line
                    // In real implementation, we would capture the y-value from user interaction
                    // For now, using a sample value
                    auto tool = std::make_unique<HorizontalLine>(new_id, 125.0);
                    tool->color = current_color;
                    tool->thickness = current_thickness;
                    add_tool(std::move(tool));
                    break;
                }
                case 2: { // Fibonacci
                    // In real implementation, we would capture two points from user interaction
                    // For now, using sample points
                    ImVec2 p1(1.0f, 100.0f);
                    ImVec2 p2(10.0f, 150.0f);
                    auto tool = std::make_unique<FibonacciRetracement>(new_id, p1, p2);
                    tool->color = current_color;
                    tool->thickness = current_thickness;
                    add_tool(std::move(tool));
                    break;
                }
                case 3: { // Rectangle
                    // In real implementation, we would capture two diagonal points from user interaction
                    // For now, using sample points
                    ImVec2 p1(2.0f, 110.0f);
                    ImVec2 p2(8.0f, 140.0f);
                    auto tool = std::make_unique<Rectangle>(new_id, p1, p2);
                    tool->color = current_color;
                    tool->thickness = current_thickness;
                    add_tool(std::move(tool));
                    break;
                }
                case 4: { // Text Annotation
                    // In real implementation, we would capture position and text from user interaction
                    // For now, using sample position and text
                    auto tool = std::make_unique<TextAnnotation>(new_id, ImVec2(5.0f, 130.0f), current_text);
                    static_cast<TextAnnotation*>(tool.get())->font_size = current_font_size;
                    tool->color = current_color;
                    tool->thickness = current_thickness;
                    add_tool(std::move(tool));
                    break;
                }
            }
        }

        // Show status if a tool is being created
        if (creating_tool_) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Creating tool... Click and drag on chart");
        }

        // List existing tools with controls
        if (!tools_.empty()) {
            ImGui::Separator();
            ImGui::Text("Existing Tools:");

            for (size_t i = 0; i < tools_.size(); ++i) {
                auto& tool = tools_[i];

                ImGui::PushID(static_cast<int>(i));

                // Show tool type and ID
                std::string tool_label = tool->id + "##" + std::to_string(i);
                ImGui::Text("%s", tool_label.c_str());

                // Visibility toggle
                bool is_visible = tool->visible;
                if (ImGui::Checkbox(("Visible##" + std::to_string(i)).c_str(), &is_visible)) {
                    tool->visible = is_visible;
                }

                // Lock toggle
                bool is_locked = tool->locked;
                if (ImGui::Checkbox(("Locked##" + std::to_string(i)).c_str(), &is_locked)) {
                    tool->locked = is_locked;
                }

                // Color picker
                ImVec4 tool_color = tool->color;
                if (ImGui::ColorEdit4(("Color##" + std::to_string(i)).c_str(), (float*)&tool_color)) {
                    tool->color = tool_color;
                }

                // Thickness slider
                float tool_thickness = tool->thickness;
                if (ImGui::SliderFloat(("Thickness##" + std::to_string(i)).c_str(), &tool_thickness, 0.5f, 5.0f, "%.1f")) {
                    tool->thickness = tool_thickness;
                }

                // Additional controls for text annotations
                if (auto* text_tool = dynamic_cast<TextAnnotation*>(tool.get())) {
                    char text_buffer[256];
                    strncpy(text_buffer, text_tool->text.c_str(), sizeof(text_buffer) - 1);
                    text_buffer[sizeof(text_buffer) - 1] = '\0';
                    if (ImGui::InputText(("Text##" + std::to_string(i)).c_str(), text_buffer, sizeof(text_buffer))) {
                        text_tool->text = std::string(text_buffer);
                    }

                    float font_size = text_tool->font_size;
                    if (ImGui::SliderFloat(("Font Size##" + std::to_string(i)).c_str(), &font_size, 8.0f, 24.0f, "%.0f")) {
                        text_tool->font_size = font_size;
                    }
                }

                // Delete button
                if (ImGui::Button(("Delete##" + std::to_string(i)).c_str())) {
                    tools_.erase(tools_.begin() + i);
                    --i; // Adjust index after removal
                }

                ImGui::PopID();
            }
        }

        // Add save/load buttons
        if (ImGui::Button("Save Tools")) {
            save_to_file("drawing_tools.json");
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Tools")) {
            load_from_file("drawing_tools.json");
        }
    }
    ImGui::End();
}

void DrawingToolsManager::save_to_file(const std::string& filename) {
    nlohmann::json root;
    nlohmann::json tools_array = nlohmann::json::array();

    for (const auto& tool : tools_) {
        nlohmann::json tool_obj;
        tool_obj["id"] = tool->id;
        tool_obj["color"] = {tool->color.x, tool->color.y, tool->color.z, tool->color.w};
        tool_obj["thickness"] = tool->thickness;
        tool_obj["visible"] = tool->visible;
        tool_obj["locked"] = tool->locked;

        // Add type-specific properties
        if (dynamic_cast<TrendLine*>(tool.get())) {
            tool_obj["type"] = "TrendLine";
            auto* trend_line = static_cast<TrendLine*>(tool.get());
            tool_obj["point1"] = {trend_line->point1.x, trend_line->point1.y};
            tool_obj["point2"] = {trend_line->point2.x, trend_line->point2.y};
        } else if (dynamic_cast<HorizontalLine*>(tool.get())) {
            tool_obj["type"] = "HorizontalLine";
            auto* h_line = static_cast<HorizontalLine*>(tool.get());
            tool_obj["y_value"] = h_line->y_value;
        } else if (dynamic_cast<FibonacciRetracement*>(tool.get())) {
            tool_obj["type"] = "FibonacciRetracement";
            auto* fib = static_cast<FibonacciRetracement*>(tool.get());
            tool_obj["point1"] = {fib->point1.x, fib->point1.y};
            tool_obj["point2"] = {fib->point2.x, fib->point2.y};
        } else if (dynamic_cast<Rectangle*>(tool.get())) {
            tool_obj["type"] = "Rectangle";
            auto* rect = static_cast<Rectangle*>(tool.get());
            tool_obj["point1"] = {rect->point1.x, rect->point1.y};
            tool_obj["point2"] = {rect->point2.x, rect->point2.y};
        } else if (dynamic_cast<TextAnnotation*>(tool.get())) {
            tool_obj["type"] = "TextAnnotation";
            auto* text = static_cast<TextAnnotation*>(tool.get());
            tool_obj["position"] = {text->position.x, text->position.y};
            tool_obj["text"] = text->text;
            tool_obj["font_size"] = text->font_size;
        }

        tools_array.push_back(tool_obj);
    }

    root["tools"] = tools_array;

    std::ofstream file(filename);
    if (file.is_open()) {
        file << root.dump(4);  // Pretty print with 4-space indentation
        file.close();
    }
}

void DrawingToolsManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (...) {
        file.close();
        return;
    }
    file.close();

    clear_all_tools();

    const auto& tools_array = root["tools"];
    if (!tools_array.is_array()) {
        return;
    }

    for (const auto& tool_obj : tools_array) {
        std::string type = tool_obj["type"].get<std::string>();
        std::string id = tool_obj["id"].get<std::string>();

        auto color_array = tool_obj["color"];
        ImVec4 color = ImVec4(
            color_array[0].get<float>(),
            color_array[1].get<float>(),
            color_array[2].get<float>(),
            color_array[3].get<float>()
        );

        float thickness = tool_obj["thickness"].get<float>();
        bool visible = tool_obj["visible"].get<bool>();
        bool locked = tool_obj["locked"].get<bool>();

        std::unique_ptr<DrawingTool> tool = nullptr;

        if (type == "TrendLine") {
            auto p1_array = tool_obj["point1"];
            auto p2_array = tool_obj["point2"];
            ImVec2 p1 = ImVec2(p1_array[0].get<float>(), p1_array[1].get<float>());
            ImVec2 p2 = ImVec2(p2_array[0].get<float>(), p2_array[1].get<float>());

            tool = std::make_unique<TrendLine>(id, p1, p2);
        } else if (type == "HorizontalLine") {
            double y_value = tool_obj["y_value"].get<double>();
            tool = std::make_unique<HorizontalLine>(id, y_value);
        } else if (type == "FibonacciRetracement") {
            auto p1_array = tool_obj["point1"];
            auto p2_array = tool_obj["point2"];
            ImVec2 p1 = ImVec2(p1_array[0].get<float>(), p1_array[1].get<float>());
            ImVec2 p2 = ImVec2(p2_array[0].get<float>(), p2_array[1].get<float>());

            tool = std::make_unique<FibonacciRetracement>(id, p1, p2);
        } else if (type == "Rectangle") {
            auto p1_array = tool_obj["point1"];
            auto p2_array = tool_obj["point2"];
            ImVec2 p1 = ImVec2(p1_array[0].get<float>(), p1_array[1].get<float>());
            ImVec2 p2 = ImVec2(p2_array[0].get<float>(), p2_array[1].get<float>());

            tool = std::make_unique<Rectangle>(id, p1, p2);
        } else if (type == "TextAnnotation") {
            auto pos_array = tool_obj["position"];
            ImVec2 pos = ImVec2(pos_array[0].get<float>(), pos_array[1].get<float>());
            std::string text = tool_obj["text"].get<std::string>();
            float font_size = tool_obj["font_size"].get<float>();

            tool = std::make_unique<TextAnnotation>(id, pos, text);
            static_cast<TextAnnotation*>(tool.get())->font_size = font_size;
        }

        if (tool) {
            tool->color = color;
            tool->thickness = thickness;
            tool->visible = visible;
            tool->locked = locked;

            tools_.push_back(std::move(tool));
        }
    }
}

void DrawingToolsManager::handle_mouse_events() {
    // Handle mouse events for creating and manipulating drawing tools
    if (ImPlot::IsPlotHovered()) {
        // Check for left mouse button press to start creating a new tool
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            ImPlotPoint plot_pos = ImPlot::GetPlotMousePos();
            ImVec2 mouse_pos = ImVec2((float)plot_pos.x, (float)plot_pos.y);

            // If we're not currently creating a tool, start creating one
            if (!creating_tool_ && current_tool_type_ >= 0) {
                start_new_tool(current_tool_type_, mouse_pos);
            }
        }

        // If we're currently creating a tool, update it as the mouse moves
        if (creating_tool_ && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImPlotPoint plot_pos = ImPlot::GetPlotMousePos();
            ImVec2 mouse_pos = ImVec2((float)plot_pos.x, (float)plot_pos.y);
            update_current_tool(mouse_pos);
        }

        // Check for mouse release to finalize the tool
        if (creating_tool_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            ImPlotPoint plot_pos = ImPlot::GetPlotMousePos();
            ImVec2 mouse_pos = ImVec2((float)plot_pos.x, (float)plot_pos.y);
            update_current_tool(mouse_pos);  // Update one final time
            finalize_current_tool();
        }
    }
}

void DrawingToolsManager::start_new_tool(int tool_type, ImVec2 start_pos) {
    // Generate a unique ID for the new tool
    static int tool_counter = 0;
    std::string new_id = "tool_" + std::to_string(++tool_counter);

    // Store the start position
    tool_start_pos_ = start_pos;

    switch (tool_type) {
        case 0: { // Trend Line
            creating_tool_ = std::make_unique<TrendLine>(new_id, start_pos, start_pos);
            break;
        }
        case 1: { // Horizontal Line
            creating_tool_ = std::make_unique<HorizontalLine>(new_id, start_pos.y);
            break;
        }
        case 2: { // Fibonacci
            creating_tool_ = std::make_unique<FibonacciRetracement>(new_id, start_pos, start_pos);
            break;
        }
        case 3: { // Rectangle
            creating_tool_ = std::make_unique<Rectangle>(new_id, start_pos, start_pos);
            break;
        }
        case 4: { // Text Annotation
            creating_tool_ = std::make_unique<TextAnnotation>(new_id, start_pos, "Note");
            break;
        }
    }

    if (creating_tool_) {
        // Apply default properties
        creating_tool_->color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);  // Yellow for visibility during creation
        creating_tool_->thickness = 2.0f;
    }
}

void DrawingToolsManager::update_current_tool(ImVec2 current_pos) {
    if (!creating_tool_) return;

    // Update the tool based on its type
    if (auto* trend_line = dynamic_cast<TrendLine*>(creating_tool_.get())) {
        trend_line->point2 = current_pos;
    } else if (auto* fib = dynamic_cast<FibonacciRetracement*>(creating_tool_.get())) {
        fib->point2 = current_pos;
    } else if (auto* rect = dynamic_cast<Rectangle*>(creating_tool_.get())) {
        rect->point2 = current_pos;
    } else if (auto* text = dynamic_cast<TextAnnotation*>(creating_tool_.get())) {
        text->position = current_pos;
    }
    // Horizontal line doesn't need updating since it only depends on Y value
}

void DrawingToolsManager::finalize_current_tool() {
    if (creating_tool_) {
        // Reset the color to white for the final tool
        creating_tool_->color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

        // Add the completed tool to the tools list
        add_tool(std::move(creating_tool_));

        // Reset tool creation state
        current_tool_type_ = -1;
    }
}

} // namespace BTQuant