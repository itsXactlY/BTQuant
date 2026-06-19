#ifndef BTQUANT_VULKAN_UTILS_HPP
#define BTQUANT_VULKAN_UTILS_HPP

#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace btquant::vulkan {

std::vector<char> readShaderFile(const std::string& filename);
VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
VkShaderModule createShaderModuleFromFile(VkDevice device, const std::string& filename);

} // namespace btquant::vulkan

#endif
