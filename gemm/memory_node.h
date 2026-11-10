#pragma once

#include <cstdlib>

// run/train.sh selects one physical memory node for each MPI rank.
// Standalone benchmarks retain their own NUMA placement when it is unset.
inline int harp_memory_node(int default_node)
{
    const char* value = std::getenv("HARP_MEMORY_NODE");
    if (!value || !*value) return default_node;
    char* end = nullptr;
    long node = std::strtol(value, &end, 10);
    return *end == '\0' && node >= 0 && node < 64 ? static_cast<int>(node) : default_node;
}
