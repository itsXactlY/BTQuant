/**
 * @file ssbo_snapshot_updater.hpp
 * @brief Vulkan SSBO Buffer Updater for OrderBookSnapshot with std430 alignment
 * 
 * Header file for CPU-side structures that mirror the GPU SSBO layout
 * defined in lob_heatmap.comp. All structures enforce std430 alignment rules
 * to ensure perfect mapping to vec4 arrays in the compute shader.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <vulkan/vulkan.h>

/**
 * @brief OrderBookLevel structure - matches the GLSL struct exactly
 * 
 * GLSL Definition:
 *   struct OrderBookLevel {
 *       float price;           // 4 bytes
 *       uint askQuantity;      // 4 bytes
 *       uint bidQuantity;      // 4 bytes
 *       uint numOrders;        // 4 bytes
 *   };                         // Total: 16 bytes = 1x vec4
 * 
 * std430 Alignment:
 * - All members are 4-byte types, naturally aligned
 * - Total size is 16 bytes, which is vec4-aligned
 * - Can be indexed as an array of vec4 in the shader
 */
struct alignas(16) OrderBookLevel {
    float price;           ///< Price level (matches GLSL float)
    uint32_t askQuantity;  ///< Ask volume at this price (matches GLSL uint)
    uint32_t bidQuantity;  ///< Bid volume at this price (matches GLSL uint)
    uint32_t numOrders;    ///< Number of orders at this price (matches GLSL uint)

    OrderBookLevel() : price(0.0f), askQuantity(0), bidQuantity(0), numOrders(0) {}
    
    OrderBookLevel(float p, uint32_t askQty, uint32_t bidQty, uint32_t nOrders)
        : price(p), askQuantity(askQty), bidQuantity(bidQty), numOrders(nOrders) {}
};

// Static assertions to verify std430 alignment
static_assert(sizeof(OrderBookLevel) == 16, 
              "OrderBookLevel must be exactly 16 bytes (1x vec4) for std430 alignment");
static_assert(alignof(OrderBookLevel) == 16, 
              "OrderBookLevel must be 16-byte aligned for std430");
static_assert(offsetof(OrderBookLevel, price) == 0, "price offset must be 0");
static_assert(offsetof(OrderBookLevel, askQuantity) == 4, "askQuantity offset must be 4");
static_assert(offsetof(OrderBookLevel, bidQuantity) == 8, "bidQuantity offset must be 8");
static_assert(offsetof(OrderBookLevel, numOrders) == 12, "numOrders offset must be 12");

/**
 * @brief OrderBookSnapshot structure - SSBO header for GPU buffer
 * 
 * GLSL Definition:
 *   layout(std430, set = 0, binding = 0) readonly buffer OrderBookSnapshot {
 *       uint currentTimeIndex;     // 4 bytes
 *       uint priceLevelsCount;     // 4 bytes
 *       float basePrice;           // 4 bytes
 *       float priceRange;          // 4 bytes
 *       OrderBookLevel levels[];   // Variable-length array
 *   };
 * 
 * std430 Alignment:
 * - Header: 4x 4-byte scalars = 16 bytes = 1x vec4
 * - Each OrderBookLevel: 16 bytes = 1x vec4
 * - Entire buffer maps to array of vec4 in shader
 */
struct alignas(16) OrderBookSnapshot {
    uint32_t currentTimeIndex;   ///< Current cyclic time index (matches GLSL uint)
    uint32_t priceLevelsCount;   ///< Number of price levels (matches GLSL uint)
    float basePrice;             ///< Base price for rendering (matches GLSL float)
    float priceRange;            ///< Price range to display (matches GLSL float)
    // OrderBookLevel levels[];  ///< Variable-length array follows in buffer

    OrderBookSnapshot() 
        : currentTimeIndex(0), priceLevelsCount(0), basePrice(0.0f), priceRange(0.0f) {}

    /**
     * @brief Calculate total buffer size for given number of levels
     * @param numLevels Number of OrderBookLevel entries
     * @return Total size in bytes (header + levels array)
     */
    static constexpr size_t calculateBufferSize(size_t numLevels) {
        return sizeof(OrderBookSnapshot) + (numLevels * sizeof(OrderBookLevel));
    }

    /**
     * @brief Get pointer to the levels array
     * @return Pointer to first OrderBookLevel after header
     */
    OrderBookLevel* getLevels() {
        return reinterpret_cast<OrderBookLevel*>(this + 1);
    }

    const OrderBookLevel* getLevels() const {
        return reinterpret_cast<const OrderBookLevel*>(this + 1);
    }

    /**
     * @brief Get level at specific index
     * @param index Level index (0-based)
     * @return Pointer to the level, or nullptr if out of bounds
     */
    OrderBookLevel* getLevel(size_t index) {
        if (index >= priceLevelsCount) return nullptr;
        return getLevels() + index;
    }

    const OrderBookLevel* getLevel(size_t index) const {
        if (index >= priceLevelsCount) return nullptr;
        return getLevels() + index;
    }
};

// Static assertions to verify std430 alignment for OrderBookSnapshot
static_assert(sizeof(OrderBookSnapshot) == 16, 
              "OrderBookSnapshot header must be exactly 16 bytes (1x vec4) for std430");
static_assert(alignof(OrderBookSnapshot) == 16, 
              "OrderBookSnapshot must be 16-byte aligned for std430");
static_assert(offsetof(OrderBookSnapshot, currentTimeIndex) == 0, "currentTimeIndex offset must be 0");
static_assert(offsetof(OrderBookSnapshot, priceLevelsCount) == 4, "priceLevelsCount offset must be 4");
static_assert(offsetof(OrderBookSnapshot, basePrice) == 8, "basePrice offset must be 8");
static_assert(offsetof(OrderBookSnapshot, priceRange) == 12, "priceRange offset must be 12");

/**
 * @brief SSBO Buffer wrapper for Vulkan
 * 
 * Manages a Vulkan buffer with proper memory flags for GPU access.
 * Supports both device-local and host-visible memory configurations.
 */
class SSBOBuffer {
public:
    SSBOBuffer() = default;
    
    ~SSBOBuffer();

    /**
     * @brief Create the SSBO buffer
     * @param device Vulkan device
     * @param numLevels Number of OrderBookLevel entries to store
     * @param usage Buffer usage flags (default: STORAGE_BUFFER_BIT)
     * @return true on success
     */
    bool create(VkDevice device, size_t numLevels, 
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    /**
     * @brief Bind memory to the buffer
     * @param deviceMemory Vulkan device memory (must be allocated with proper flags)
     * @param memoryOffset Offset into the device memory
     * @return true on success
     */
    bool bindMemory(VkDeviceMemory deviceMemory, VkDeviceSize memoryOffset = 0);

    /**
     * @brief Map the buffer and write snapshot data
     * @param snapshot The snapshot data to write
     * @param levels Pointer to array of OrderBookLevel entries
     * @return true on success
     */
    bool writeData(const OrderBookSnapshot& snapshot, const OrderBookLevel* levels);

    /**
     * @brief Get the Vulkan buffer handle
     */
    VkBuffer getBuffer() const { return buffer_; }

    /**
     * @brief Get the Vulkan device memory handle
     */
    VkDeviceMemory getMemory() const { return memory_; }

    /**
     * @brief Get the buffer size in bytes
     */
    VkDeviceSize getBufferSize() const { return bufferSize_; }

    /**
     * @brief Get the number of levels the buffer can hold
     */
    size_t getNumLevels() const { return numLevels_; }

    /**
     * @brief Create a descriptor info structure for binding
     */
    VkDescriptorBufferInfo getDescriptorInfo() const;

private:
    void cleanup();

    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize bufferSize_ = 0;
    size_t numLevels_ = 0;
};

/**
 * @brief Helper function to create a properly aligned SSBO buffer
 * 
 * @param device Vulkan device
 * @param physicalDevice Vulkan physical device (for memory properties)
 * @param numLevels Number of price levels to store
 * @param memoryFlags Memory property flags (e.g., HOST_VISIBLE | HOST_COHERENT)
 * @param outBuffer Output buffer wrapper
 * @param outMemory Output memory handle
 * @return true on success
 */
bool createSSBOBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                      size_t numLevels, VkMemoryPropertyFlags memoryFlags,
                      SSBOBuffer& outBuffer, VkDeviceMemory& outMemory);
