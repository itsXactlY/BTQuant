#include "ui_context.hpp"

#include <cstdio>
#include <vector>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <implot.h>

#include "../core/vulkan_context.hpp"

#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

namespace btquant::ui {

UIContext::UIContext() = default;
UIContext::~UIContext() = default;

bool UIContext::initialize(void* window, VkInstance instance,
                            VkPhysicalDevice physicalDevice, VkDevice device,
                            uint32_t graphicsQueueFamily, VkQueue graphicsQueue,
                            VkRenderPass renderPass, const char* iniFilename) {
    if (!window || !instance || !device || !renderPass) {
        std::fprintf(stderr, "[UIContext] initialize: invalid handles (window=%p, inst=%p, dev=%p, rp=%p)\n",
                     window, (void*)instance, (void*)device, (void*)renderPass);
        return false;
    }

    // 1. Create ImGui + ImPlot contexts.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // ImGui ini file (window positions, dock layouts). If path is given AND the
    // file exists, ImGui auto-loads it on first BeginFrame. If we set IniFilename
    // to a non-NULL path we also need to remember it for saveIniSettings() at
    // shutdown — ImGui will auto-save on every frame too, but we call it once
    // explicitly on shutdown so we get the latest state.
    if (iniFilename) {
        io.IniFilename = iniFilename;
        m_iniFilename = iniFilename;
    } else {
        io.IniFilename = nullptr;  // disable ImGui ini save
        m_iniFilename.clear();
    }

    // 2. Apply BTQuant theme (Linear Dark + Kraken Purple per btquant-ui-design.md).
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 8.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg]        = ImVec4(0.031f, 0.035f, 0.039f, 1.0f);  // #08090a
    style.Colors[ImGuiCol_Text]           = ImVec4(0.969f, 0.973f, 0.973f, 1.0f);  // #f7f8f8
    style.Colors[ImGuiCol_Border]         = ImVec4(1.000f, 1.000f, 1.000f, 0.08f);
    style.Colors[ImGuiCol_Button]         = ImVec4(0.443f, 0.196f, 0.961f, 1.0f);  // #7132f5
    style.Colors[ImGuiCol_ButtonHovered]  = ImVec4(0.523f, 0.286f, 1.000f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive]   = ImVec4(0.392f, 0.157f, 0.886f, 1.0f);
    style.Colors[ImGuiCol_FrameBg]        = ImVec4(1.000f, 1.000f, 1.000f, 0.02f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.05f);

    // 3. Init GLFW backend.
    if (!ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(window), true)) {
        std::fprintf(stderr, "[UIContext] ImGui_ImplGlfw_InitForVulkan failed\n");
        return false;
    }

    // 4. Create descriptor pool for ImGui (ImGui recommends large pool).
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000 },
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
    pool_info.pPoolSizes = pool_sizes;
    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &m_imguiDescriptorPool) != VK_SUCCESS) {
        std::fprintf(stderr, "[UIContext] failed to create ImGui descriptor pool\n");
        ImGui_ImplGlfw_Shutdown();
        return false;
    }
    m_imguiDevice = device;

    // 5. Init Vulkan backend (NEW API as of 2025-09-26 docking branch: PipelineInfoMain).
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion = VK_API_VERSION_1_3;
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.QueueFamily = graphicsQueueFamily;
    init_info.Queue = graphicsQueue;
    init_info.DescriptorPool = m_imguiDescriptorPool;
    init_info.DescriptorPoolSize = 0;  // we provide our own pool
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.PipelineInfoMain.RenderPass = renderPass;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = false;
    init_info.CheckVkResultFn = nullptr;

    if (!ImGui_ImplVulkan_Init(&init_info)) {
        std::fprintf(stderr, "[UIContext] ImGui_ImplVulkan_Init failed\n");
        vkDestroyDescriptorPool(device, m_imguiDescriptorPool, nullptr);
        ImGui_ImplGlfw_Shutdown();
        return false;
    }

    m_initialized = true;
    return true;
}

void UIContext::shutdown() {
    if (!m_initialized) return;

    // Save ImGui state before destroying the context — ImGui::SaveIniSettingsToDisk
    // is the only safe way to flush pending changes (auto-save happens on
    // platform events but not on plain shutdown).
    if (!m_iniFilename.empty()) {
        ImGui::SaveIniSettingsToDisk(m_iniFilename.c_str());
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    if (m_imguiDevice != VK_NULL_HANDLE && m_imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_imguiDevice, m_imguiDescriptorPool, nullptr);
    }
    m_imguiDescriptorPool = VK_NULL_HANDLE;
    m_imguiDevice = VK_NULL_HANDLE;
    m_initialized = false;
}

void UIContext::saveIniSettings() const {
    if (!m_iniFilename.empty()) {
        ImGui::SaveIniSettingsToDisk(m_iniFilename.c_str());
    }
}

void UIContext::newFrame() {
    if (!m_initialized) return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UIContext::render(VkCommandBuffer commandBuffer) {
    if (!m_initialized) return;
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data && draw_data->CmdListsCount > 0) {
        ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
    }
}

} // namespace btquant::ui
