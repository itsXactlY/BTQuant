#include "ssbo_snapshot_updater.h"
#include <cstring>
#include <iostream>

namespace BTQuant {

SSBOSnapshotUpdater::SSBOSnapshotUpdater(VulkanCore* vulkan_core)
    : vulkan_core_(vulkan_core), ssbo_buffer_{VK_NULL_HANDLE}, ssbo_memory_{VK_NULL_HANDLE}, 
      descriptor_set_{VK_NULL_HANDLE} {
    if (!vulkan_core_) {
        throw std::runtime_error("VulkanCore pointer is null");
    }
}

SSBOSnapshotUpdater::~SSBOSnapshotUpdater() {
    cleanup();
}

void SSBOSnapshotUpdater::initialize(size_t max_snapshots) {
    max_snapshots_ = max_snapshots;
    
    // Calculate the size needed for the SSBO
    VkDeviceSize buffer_size = sizeof(OrderBookSnapshotSSBO) + 
                              (max_snapshots - 1) * sizeof(OrderBookSnapshotGPU); // Account for array size
    
    GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();
    
    // Allocate storage buffer
    storage_buffer_allocation_ = memory_manager.allocate_storage_buffer(buffer_size);
    
    if (storage_buffer_allocation_.buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to allocate storage buffer for snapshots");
    }
    
    ssbo_buffer_ = storage_buffer_allocation_.buffer;
    ssbo_memory_ = storage_buffer_allocation_.memory;
    
    // Create descriptor set for the SSBO
    create_descriptor_set();
    
    std::cout << "[SSBOSnapshotUpdater] Initialized SSBO with capacity for " 
              << max_snapshots << " snapshots" << std::endl;
}

void SSBOSnapshotUpdater::create_descriptor_set() {
    if (!vulkan_core_) return;
    
    VkDevice device = vulkan_core_->get_device();
    VkDescriptorPool descriptor_pool = vulkan_core_->get_descriptor_pool();
    
    // Create descriptor set layout binding for SSBO
    VkDescriptorSetLayoutBinding ssbo_layout_binding{};
    ssbo_layout_binding.binding = 0; // Binding 0 for OrderBookSnapshots SSBO
    ssbo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssbo_layout_binding.descriptorCount = 1;
    ssbo_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssbo_layout_binding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &ssbo_layout_binding;
    
    VkDescriptorSetLayout descriptor_set_layout;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout for SSBO");
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout;
    
    if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set_) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        throw std::runtime_error("Failed to allocate descriptor set for SSBO");
    }
    
    // Update descriptor set
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = ssbo_buffer_;
    buffer_info.offset = 0;
    buffer_info.range = sizeof(OrderBookSnapshotSSBO) + 
                       (max_snapshots_ - 1) * sizeof(OrderBookSnapshotGPU);
    
    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = descriptor_set_;
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorCount = 1;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_write.pImageInfo = nullptr;
    descriptor_write.pBufferInfo = &buffer_info;
    descriptor_write.pTexelBufferView = nullptr;
    
    vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);
    
    // Clean up temporary layout
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
}

void SSBOSnapshotUpdater::updateSSBO(const std::vector<RenderEngine::OrderBookSnapshot>& snapshots) {
    if (!vulkan_core_ || ssbo_buffer_ == VK_NULL_HANDLE) {
        std::cerr << "[SSBOSnapshotUpdater] Vulkan resources not initialized" << std::endl;
        return;
    }
    
    if (snapshots.empty()) {
        std::cout << "[SSBOSnapshotUpdater] No snapshots to update" << std::endl;
        return;
    }
    
    size_t num_snapshots = std::min(snapshots.size(), max_snapshots_);
    
    // Prepare the SSBO data structure
    OrderBookSnapshotSSBO ssbo_data;
    ssbo_data.numSnapshots = static_cast<uint32_t>(num_snapshots);
    
    // Convert snapshots to GPU-compatible format
    for (size_t i = 0; i < num_snapshots; ++i) {
        const auto& cpu_snapshot = snapshots[i];
        
        // Convert to GPU format
        ssbo_data.snapshots[i].timestamp = cpu_snapshot.timestamp;
        ssbo_data.snapshots[i].symbol_id = cpu_snapshot.symbol_id;
        ssbo_data.snapshots[i].best_bid = cpu_snapshot.best_bid;
        ssbo_data.snapshots[i].best_ask = cpu_snapshot.best_ask;
        ssbo_data.snapshots[i].best_bid_size = cpu_snapshot.best_bid_size;
        ssbo_data.snapshots[i].best_ask_size = cpu_snapshot.best_ask_size;
        ssbo_data.snapshots[i].spread = cpu_snapshot.spread;
        ssbo_data.snapshots[i].total_bid_volume = cpu_snapshot.total_bid_volume;
        ssbo_data.snapshots[i].total_ask_volume = cpu_snapshot.total_ask_volume;
        ssbo_data.snapshots[i].bid_levels_count = cpu_snapshot.bid_levels_count;
        ssbo_data.snapshots[i].ask_levels_count = cpu_snapshot.ask_levels_count;
        
        // Copy bid levels
        for (size_t j = 0; j < OrderBookSnapshotGPU::MAX_LEVELS && j < cpu_snapshot.bid_levels_count; ++j) {
            ssbo_data.snapshots[i].bids[j].price = cpu_snapshot.bids[j].price;
            ssbo_data.snapshots[i].bids[j].size = cpu_snapshot.bids[j].size;
        }
        
        // Copy ask levels
        for (size_t j = 0; j < OrderBookSnapshotGPU::MAX_LEVELS && j < cpu_snapshot.ask_levels_count; ++j) {
            ssbo_data.snapshots[i].asks[j].price = cpu_snapshot.asks[j].price;
            ssbo_data.snapshots[i].asks[j].size = cpu_snapshot.asks[j].size;
        }
    }
    
    // Copy data to the SSBO buffer
    VkDeviceSize buffer_size = sizeof(OrderBookSnapshotSSBO) + 
                              (max_snapshots_ - 1) * sizeof(OrderBookSnapshotGPU);
    
    // Use the mapped pointer if available, otherwise use staging buffer approach
    if (storage_buffer_allocation_.mapped_ptr) {
        // Direct copy to mapped memory
        std::memcpy(storage_buffer_allocation_.mapped_ptr, &ssbo_data, 
                   std::min(buffer_size, 
                           sizeof(uint32_t) + num_snapshots * sizeof(OrderBookSnapshotGPU)));
    } else {
        // Use staging buffer approach
        copy_via_staging_buffer(&ssbo_data, buffer_size);
    }
    
    std::cout << "[SSBOSnapshotUpdater] Updated SSBO with " << num_snapshots << " snapshots" << std::endl;
}

void SSBOSnapshotUpdater::copy_via_staging_buffer(const void* data, VkDeviceSize size) {
    if (!vulkan_core_) return;
    
    VkCommandBuffer command_buffer = vulkan_core_->begin_single_time_commands();
    
    GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();
    
    // Allocate staging buffer
    BufferAllocation staging_buffer = memory_manager.allocate_staging_buffer(size);
    
    if (staging_buffer.buffer == VK_NULL_HANDLE) {
        std::cerr << "[SSBOSnapshotUpdater] Failed to allocate staging buffer" << std::endl;
        vulkan_core_->end_single_time_commands(command_buffer);
        return;
    }
    
    // Copy data to staging buffer
    std::memcpy(staging_buffer.mapped_ptr, data, size);
    
    // Copy from staging buffer to SSBO
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    
    vkCmdCopyBuffer(command_buffer, staging_buffer.buffer, ssbo_buffer_, 1, &copy_region);
    
    vulkan_core_->end_single_time_commands(command_buffer);
    
    // Deallocate staging buffer
    memory_manager.deallocate_buffer(staging_buffer);
}

void SSBOSnapshotUpdater::cleanup() {
    if (vulkan_core_ && ssbo_buffer_ != VK_NULL_HANDLE) {
        GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();
        memory_manager.deallocate_buffer(storage_buffer_allocation_);
        ssbo_buffer_ = VK_NULL_HANDLE;
        ssbo_memory_ = VK_NULL_HANDLE;
    }
}

VkDescriptorSet SSBOSnapshotUpdater::get_descriptor_set() const {
    return descriptor_set_;
}

} // namespace BTQuant