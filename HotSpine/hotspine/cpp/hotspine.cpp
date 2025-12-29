#include "hotspine.h"
#include <cstring>
#include <sched.h>
#include <unistd.h>

static Ring ring;

Ring* hotspine_init() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    ring.write_idx.store(0);
    ring.read_idx.store(0);
    return &ring;
}

void hotspine_push(Ring* r, Trade t) {
    uint64_t w = r->write_idx.load(std::memory_order_relaxed);
    r->buffer[w % RING_SIZE] = t;
    r->write_idx.store(w + 1, std::memory_order_release);
}

int hotspine_poll(Ring* r, Trade* out) {
    uint64_t r_idx = r->read_idx.load(std::memory_order_relaxed);
    uint64_t w_idx = r->write_idx.load(std::memory_order_acquire);

    if (r_idx == w_idx) return 0;

    *out = r->buffer[r_idx % RING_SIZE];
    r->read_idx.store(r_idx + 1, std::memory_order_release);
    return 1;
}
