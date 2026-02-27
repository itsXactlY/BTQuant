#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace BTQuant {

struct ViewPort {
  double minTime = 0.0;
  double maxTime = 1.0;
  double minPrice = 0.0;
  double maxPrice = 1.0;

  bool isValid() const { return (maxTime > minTime) && (maxPrice > minPrice); }

  double width() const { return maxTime - minTime; }
  double height() const { return maxPrice - minPrice; }

  void clamp() {
    if (minTime > maxTime) std::swap(minTime, maxTime);
    if (minPrice > maxPrice) std::swap(minPrice, maxPrice);
    if (width() < 1e-9) maxTime = minTime + 1e-9;
    if (height() < 1e-9) maxPrice = minPrice + 1e-9;
  }
};

class ChartMath {
 public:
  static glm::mat4 CreateProjectionMatrix(float windowWidth, float windowHeight) {
    // Map 0..width, 0..height to NDC -1..1
    // (Y-down for Vulkan/ImGui compatibility)
    return glm::ortho(0.0f, windowWidth, windowHeight, 0.0f, -1.0f, 1.0f);
  }

  // Market (Time, Price) -> Screen (Pixels)
  static glm::vec2 MapToScreen(const glm::vec2& worldPos, const ViewPort& view, float windowWidth,
                               float windowHeight) {
    if (!view.isValid()) return glm::vec2(0.0f);
    float normX = (worldPos.x - (float)view.minTime) / (float)view.width();
    float normY = (worldPos.y - (float)view.minPrice) / (float)view.height();

    return glm::vec2(normX * windowWidth, (1.0f - normY) * windowHeight);
  }

  // Screen (Pixels) -> Market (Time, Price)
  static glm::vec2 ScreenToMarket(const glm::vec2& screenPos, const ViewPort& view,
                                  float windowWidth, float windowHeight) {
    if (!view.isValid() || windowWidth <= 0 || windowHeight <= 0) return glm::vec2(0.0f);
    float normX = screenPos.x / windowWidth;
    float normY = 1.0f - (screenPos.y / windowHeight);

    return glm::vec2((float)view.minTime + normX * (float)view.width(),
                     (float)view.minPrice + normY * (float)view.height());
  }

  // Market (Time, Price) -> Vulkan NDC (-1 to 1)
  static glm::vec2 MapToNDC(const glm::vec2& worldPos, const ViewPort& view) {
    if (!view.isValid()) return glm::vec2(0.0f);
    float normX = (worldPos.x - (float)view.minTime) / (float)view.width();
    float normY = (worldPos.y - (float)view.minPrice) / (float)view.height();

    return glm::vec2(normX * 2.0f - 1.0f, normY * 2.0f - 1.0f);
  }

  // Vulkan NDC (-1 to 1) -> Market (Time, Price)
  static glm::vec2 NDCToMarket(const glm::vec2& ndcPos, const ViewPort& view) {
    if (!view.isValid()) return glm::vec2(0.0f);
    float normX = (ndcPos.x + 1.0f) * 0.5f;
    float normY = (ndcPos.y + 1.0f) * 0.5f;

    return glm::vec2((float)view.minTime + normX * (float)view.width(),
                     (float)view.minPrice + normY * (float)view.height());
  }

  static void ApplyAspectRatio(ViewPort& view, float windowWidth, float windowHeight) {
    (void)view; (void)windowWidth; (void)windowHeight;  // Suppress unused parameter warnings
    // For financial charts, we don't usually force fixed aspect ratio
    // but we can ensure minimum visibility ranges here if needed.
  }
};

}  // namespace BTQuant
