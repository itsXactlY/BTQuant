#pragma once

namespace pubbtquant::rendering {

/**
 * @brief Pre-allocates ImGui vertex and index buffers for improved performance.
 * 
 * Call this once after ImGui::CreateContext() and before the first frame.
 */
void InitializeImGuiBuffers();

} // namespace pubbtquant::rendering
