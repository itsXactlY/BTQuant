#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <vector>

// DEAD SIMPLE CANDLESTICK TEST
int main() {
  // Init GLFW
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Candlestick Test", nullptr, nullptr);

  // Init ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  // Hardcoded test data - 10 candles
  std::vector<double> dates = {1.0, 2.0, 3.0, 4.0, 5.0,
                               6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> opens = {100.0, 102.0, 101.5, 103.0, 102.5,
                               104.0, 103.5, 105.0, 104.5, 106.0};
  std::vector<double> highs = {102.0, 103.5, 103.0, 104.5, 104.0,
                               105.5, 105.0, 106.5, 106.0, 107.5};
  std::vector<double> lows = {99.5,  101.0, 101.0, 102.0, 102.0,
                              103.0, 103.0, 104.0, 104.0, 105.0};
  std::vector<double> closes = {102.0, 101.0, 103.0, 102.0, 104.0,
                                103.0, 105.0, 104.0, 106.0, 105.0};

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    ImGui::NewFrame();

    ImGui::Begin("Candles");
    if (ImPlot::BeginPlot("Test", ImVec2(-1, -1))) {
      ImPlot::SetupAxis(ImAxis_X1, "Time");
      ImPlot::SetupAxis(ImAxis_Y1, "Price");

      // Plot 4 lines for OHLC
      ImPlot::PlotLine("Open", dates.data(), opens.data(), 10);
      ImPlot::PlotLine("High", highs.data(), highs.data(), 10);
      ImPlot::PlotLine("Low", lows.data(), lows.data(), 10);
      ImPlot::PlotLine("Close", dates.data(), closes.data(), 10);

      ImPlot::EndPlot();
    }
    ImGui::End();

    ImGui::Render();
    // Vulkan render here
    glfwSwapBuffers(window);
  }

  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
