#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>

struct ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t capacity;
    uint64_t used;
    uint64_t trade_write_pos;
    uint64_t trade_read_pos;
    uint64_t orderbook_write_pos;
    uint64_t orderbook_read_pos;
    uint64_t last_update_us;
    uint32_t checksum;
};

int main() {
    const char* shm_name = "/btquant_hotspine";

    int fd = shm_open(shm_name, O_RDONLY, 0666);
    if (fd < 0) {
        std::cerr << "Failed to open shared memory: " << strerror(errno) << std::endl;
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "Failed to stat: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }

    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "Failed to mmap: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }

    ShmHeader* header = static_cast<ShmHeader*>(mapped);

    std::cout << "Reader's view of header:" << std::endl;
    std::cout << "  magic: 0x" << std::hex << header->magic << std::dec << std::endl;
    std::cout << "  version: " << header->version << std::endl;
    std::cout << "  capacity: " << header->capacity << std::endl;
    std::cout << "  used: " << header->used << std::endl;
    std::cout << "  trade_write_pos: " << header->trade_write_pos << std::endl;
    std::cout << "  trade_read_pos: " << header->trade_read_pos << std::endl;
    std::cout << "  orderbook_write_pos: " << header->orderbook_write_pos << std::endl;
    std::cout << "  orderbook_read_pos: " << header->orderbook_read_pos << std::endl;
    std::cout << "  last_update_us: " << header->last_update_us << std::endl;
    std::cout << "  checksum: " << header->checksum << std::endl;

    // Also dump raw bytes
    std::cout << "\nRaw header bytes:" << std::endl;
    unsigned char* bytes = static_cast<unsigned char*>(mapped);
    for (size_t i = 0; i < sizeof(ShmHeader); ++i) {
        if (i % 16 == 0) std::cout << std::hex << std::setw(8) << std::setfill('0') << i << ": ";
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
        if (i % 16 == 15) std::cout << std::endl;
    }
    std::cout << std::dec << std::endl;

    munmap(mapped, st.st_size);
    close(fd);

    return 0;
}