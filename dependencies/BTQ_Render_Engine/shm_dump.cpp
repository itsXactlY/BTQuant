#include <cstdint>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct HotPriceLevel {
  double price;
  double size;
};

struct HotOrderbookSnapshot {
  uint64_t ts_exchange;
  uint64_t ts_local;
  uint32_t symbol_id;
  uint8_t bids_count;
  uint8_t asks_count;
  uint8_t padding[2];
  HotPriceLevel bids[20];
  HotPriceLevel asks[20];
};

struct SharedMemoryHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t capacity;
  uint64_t write_index;
  uint64_t read_index;
  uint64_t lost_count;
  uint64_t orderbook_write_index;
  uint64_t orderbook_read_index;
  uint64_t orderbook_lost_count;
  uint64_t orderbook_capacity;
  uint8_t padding[8];
};

int main() {
  int fd = shm_open("/btquant_hotspine", O_RDONLY, 0666);
  if (fd == -1) {
    perror("shm_open");
    return 1;
  }

  struct stat sb;
  fstat(fd, &sb);
  void *ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
  SharedMemoryHeader *header = (SharedMemoryHeader *)ptr;

  std::cout
      << "SHM Header Verified. Symbol ID for ETH is usually 10008 or 10004."
      << std::endl;

  unsigned char *base = (unsigned char *)ptr + 4096;
  size_t trade_block_size =
      header->capacity * 40; // HotTrade is exactly 40 bytes.
  unsigned char *book_base = base + trade_block_size;

  for (int idx = 0; idx < 1000; ++idx) {
    HotOrderbookSnapshot *snap =
        (HotOrderbookSnapshot *)(book_base +
                                 (idx % header->orderbook_capacity) * 664);
    if (snap->symbol_id == 10008 || snap->symbol_id == 10004 ||
        snap->symbol_id == 10009) {
      std::cout << "--- Snapshot for ETH (SymID=" << snap->symbol_id
                << ") at index " << idx << " ---" << std::endl;
      std::cout << "Counts: Bids=" << (int)snap->bids_count
                << " Asks=" << (int)snap->asks_count << std::endl;

      std::cout << "Bids Levels (first 5):" << std::endl;
      for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] Price: " << snap->bids[i].price
                  << " Size: " << snap->bids[i].size << std::endl;
      }
      std::cout << "Asks Levels (first 5):" << std::endl;
      for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] Price: " << snap->asks[i].price
                  << " Size: " << snap->asks[i].size << std::endl;
      }
      break;
    }
  }

  munmap(ptr, sb.st_size);
  close(fd);
  return 0;
}
