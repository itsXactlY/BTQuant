#include "../../include/data/orderbook_snapshot_manager.hpp"

// The implementation is header-only since it's mostly template/inline functions
// and atomic operations that should be inlined for performance.
// The class is designed to be lock-free with atomic operations only.