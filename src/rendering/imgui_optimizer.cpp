#include <imgui.h>

namespace pubbtquant::rendering {

/**
 * @brief Pre-allocates ImGui vertex and index buffers for improved performance.
 * 
 * Call this once after ImGui::CreateContext() and before the first frame.
 */
void InitializeImGuiBuffers() {
    auto& io = ImGui::GetIO();
    
    // Reserve vertex buffer space for 250,000 vertices
    io.VtxBuffer.reserve(250000);
    
    // Reserve index buffer space for 500,000 indices
    io.IdxBuffer.reserve(500000);
}

} // namespace pubbtquant::rendering
