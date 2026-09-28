#include "data_spine.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#ifdef BTQUANT_USE_NLOHMANN
#include <nlohmann/json.hpp>
#endif

namespace btquant::data {

DataSpine::DataSpine() = default;

DataSpine::~DataSpine() {
    close();
}

bool DataSpine::open(const std::string& path) {
    m_fd = ::open(path.c_str(), O_RDONLY);
    if (m_fd < 0) {
        std::cerr << "Failed to open: " << path << "\n";
        return false;
    }

    struct stat st;
    if (fstat(m_fd, &st) < 0) {
        close();
        return false;
    }
    m_size = st.st_size;

    m_mapped = mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    if (m_mapped == MAP_FAILED) {
        close();
        return false;
    }

    // Read header
    char magic[5] = {};
    std::memcpy(magic, m_mapped, 4);
    if (std::strncmp(magic, "UQTB", 4) != 0) {
        std::cerr << "Invalid magic: " << magic << "\n";
        close();
        return false;
    }

    std::memcpy(&m_version, static_cast<char*>(m_mapped) + 4, 2);
    std::memcpy(&m_symbolCount, static_cast<char*>(m_mapped) + 6, 2);

    // Load symbols from JSON if available
    std::ifstream symfile("/dev/shm/btquant_symbols.json");
    if (symfile.is_open()) {
#ifdef BTQUANT_USE_NLOHMANN
        try {
            auto j = nlohmann::json::parse(symfile);
            for (const auto& s : j["symbols"]) {
                m_symbolSymbols.push_back({static_cast<int>(m_symbolSymbols.size()), s["exchange"], s["symbol"]});
            }
        } catch (...) {
            // If JSON parsing fails, create basic symbols
            for (uint32_t i = 0; i < m_symbolCount; ++i) {
                m_symbolSymbols.push_back({static_cast<int>(i), "UNKNOWN", "SYM" + std::to_string(i)});
            }
        }
#else
        // If nlohmann json is not available, create basic symbols
        for (uint32_t i = 0; i < m_symbolCount; ++i) {
            m_symbolSymbols.push_back({static_cast<int>(i), "UNKNOWN", "SYM" + std::to_string(i)});
        }
#endif
    } else {
        // If no symbols file, create basic symbols
        for (uint32_t i = 0; i < m_symbolCount; ++i) {
            m_symbolSymbols.push_back({static_cast<int>(i), "UNKNOWN", "SYM" + std::to_string(i)});
        }
    }

    return true;
}

void DataSpine::close() {
    if (m_mapped && m_mapped != MAP_FAILED) {
        munmap(m_mapped, m_size);
        m_mapped = nullptr;
    }
    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }
}

std::optional<HotSpineEntry> DataSpine::readEntry(uint32_t symbolIndex) {
    if (!isOpen() || symbolIndex >= m_symbolCount) {
        return std::nullopt;
    }

    HotSpineEntry entry;
    size_t offset = m_dataOffset + symbolIndex * sizeof(HotSpineEntry);
    
    // Check bounds
    if (offset + sizeof(HotSpineEntry) > m_size) {
        return std::nullopt;
    }
    
    std::memcpy(&entry, static_cast<char*>(m_mapped) + offset, sizeof(HotSpineEntry));
    return entry;
}

std::optional<HotSpineEntry> DataSpine::readLatest() {
    if (!isOpen() || m_symbolCount == 0) {
        return std::nullopt;
    }
    return readEntry(0);
}

// Method to read all entries for all symbols
std::vector<HotSpineEntry> DataSpine::readAllEntries() {
    std::vector<HotSpineEntry> entries;
    if (!isOpen()) {
        return entries;
    }
    
    for (uint32_t i = 0; i < m_symbolCount; ++i) {
        auto entry = readEntry(i);
        if (entry.has_value()) {
            entries.push_back(entry.value());
        }
    }
    
    return entries;
}

// Resolve a symbol name (e.g. "BTC/USDT") to its index in the spine.
// Returns std::nullopt if the spine doesn't carry that symbol — callers
// can then decide to fall back to a synthetic generator for that symbol.
std::optional<uint32_t> DataSpine::findSymbolIndex(const std::string& symbol) const {
    for (size_t i = 0; i < m_symbolSymbols.size(); ++i) {
        if (m_symbolSymbols[i].symbol == symbol) {
            return static_cast<uint32_t>(i);
        }
    }
    return std::nullopt;
}

// Method to subscribe to updates for a specific symbol
void DataSpine::subscribeToSymbol(uint32_t symbolIndex, std::function<void(const HotSpineEntry&)> callback) {
    // This would typically run in a separate thread
    std::thread([this, symbolIndex, callback]() {
        HotSpineEntry lastEntry;
        bool firstRead = true;
        
        while (true) {  // In a real implementation, this would have a way to stop
            auto entry = readEntry(symbolIndex);
            if (entry.has_value()) {
                // Only trigger callback if the entry changed
                if (firstRead || entry->timestamp != lastEntry.timestamp || 
                    entry->bid_price != lastEntry.bid_price || 
                    entry->ask_price != lastEntry.ask_price) {
                    
                    callback(entry.value());
                    lastEntry = entry.value();
                    firstRead = false;
                }
            }
            
            // Sleep briefly to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }).detach();
}

} // namespace btquant::data