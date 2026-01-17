#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace BTQuant {

struct ViewPort {
  double minTime;
  double maxTime;
  double minPrice;
  double maxPrice;

  double width() const { return maxTime - minTime; }
  double height() const { return maxPrice - minPrice; }
};

class ChartMath {
public:
  static glm::mat4 CreateProjectionMatrix(float windowWidth,
                                          float windowHeight) {
    // Map 0..width, 0..height to NDC -1..1
    // Using orthographic projection: left=0, right=width, bottom=height, top=0
    // (Y-down for Vulkan/ImGui compatibility)
    return glm::ortho(0.0f, windowWidth, windowHeight, 0.0f, -1.0f, 1.0f);
  }

  static glm::vec2 MapToScreen(const glm::vec2 &worldPos, const ViewPort &view,
                               float windowWidth, float windowHeight) {
    float normX = (worldPos.x - (float)view.minTime) / (float)view.width();
    float normY = (worldPos.y - (float)view.minPrice) / (float)view.height();

    return glm::vec2(normX * windowWidth, (1.0f - normY) * windowHeight);
  }

  // Adjust ViewPort to maintain aspect ratio if needed, or handle it in
  // projection
  static void ApplyAspectRatio(ViewPort &view, float windowWidth,
                               float windowHeight) {
    if (windowHeight == 0)
      return;
    // Removed unused aspect ratio calculations

    // In trading charts, we usually don't force a fixed 1:1 aspect ratio
    // between price and time because they are different units. Instead, we just
    // ensure the drawing doesn't stretch. The current shader implementation
    // maps 0..1 range to 0..1280/0..720.
  }
};

} // namespace BTQuant
