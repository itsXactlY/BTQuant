#ifndef BTQUANT_DATA_SPINE_HPP
#define BTQUANT_DATA_SPINE_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <functional>

namespace btquant::data {

struct HotSpineEntry {
    double bid_price;
    double ask_price;
    double bid_size;
    double ask_size;
    uint64_t timestamp;
    uint32_t seq;
    uint32_t flags;
    char padding[128 - 40];  // Pad to 128 bytes
};
static_assert(alignof(HotSpineEntry) == 8, "HotSpineEntry alignment must be 8");

struct Symbol {
    int id;
    std::string exchange;
    std::string symbol;
};

class DataSpine {
public:
    DataSpine();
    ~DataSpine();

    bool open(const std::string& path);
    void close();

    [[nodiscard]] bool isOpen() const noexcept { return m_fd >= 0; }

    [[nodiscard]] uint32_t version() const noexcept { return m_version; }
    [[nodiscard]] uint32_t symbolCount() const noexcept { return m_symbolCount; }
    [[nodiscard]] const std::vector<Symbol>& symbols() const noexcept { return m_symbolSymbols; }

    std::optional<HotSpineEntry> readEntry(uint32_t symbolIndex);
    std::optional<HotSpineEntry> readLatest();
    
    // Additional methods for data pipeline
    std::vector<HotSpineEntry> readAllEntries();
    void subscribeToSymbol(uint32_t symbolIndex, std::function<void(const HotSpineEntry&)> callback);

private:
    int m_fd = -1;
    void* m_mapped = nullptr;
    size_t m_size = 0;
    uint32_t m_version = 0;
    uint32_t m_symbolCount = 0;
    std::vector<Symbol> m_symbolSymbols;
    size_t m_dataOffset = 0x1000;
};

} // namespace btquant::data

#endif