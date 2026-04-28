#pragma once

#include <errno.h>
#include <limits.h>
#include <numaif.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

namespace bgemm_shared_workspace {

constexpr size_t kSharedABytes = 100ull * 1024ull * 1024ull;
constexpr size_t kSharedBBytes = 100ull * 1024ull * 1024ull;
constexpr size_t kSharedCBytes = 200ull * 1024ull * 1024ull;
constexpr int kCoresPerNuma = 38;
constexpr int kNumaOffset = 16;
constexpr unsigned long kMbindMaxNodes = 64;
constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;

struct SharedWorkspace {
    void* a = NULL;
    size_t a_bytes = 0;
    void* b = NULL;
    size_t b_bytes = 0;
    void* c = NULL;
    size_t c_bytes = 0;
    int numa_id = INT_MIN;
    size_t bytes_in_use = 0;
    size_t peak_bytes = 0;
};

inline SharedWorkspace g_shared_workspace;

inline int read_env_rank(const char* name, int* rank);
inline int detect_mpi_rank(int* rank);

inline int trace_enabled()
{
    static int cached = -1;
    if (cached == -1) {
        const char* value = getenv("FINAL_DEMO_HBM_TRACE");
        if (!value || value[0] == '\0' || strcmp(value, "0") == 0) {
            cached = 0;
        } else {
            int rank = 0;
            cached = (!detect_mpi_rank(&rank) || rank == 0) ? 1 : 0;
        }
    }
    return cached;
}

inline double bytes_to_mib(size_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0);
}

inline size_t align_to_hugepage(size_t size)
{
    return (size + kHugePageSize - 1) & ~(kHugePageSize - 1);
}

inline int read_env_rank(const char* name, int* rank)
{
    const char* value = getenv(name);
    if (!value || !*value) {
        return 0;
    }
    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    *rank = (int)parsed;
    return 1;
}

inline int detect_mpi_rank(int* rank)
{
    static const char* const kRankEnvNames[] = {
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
        "RANK",
    };
    for (size_t i = 0; i < sizeof(kRankEnvNames) / sizeof(kRankEnvNames[0]); ++i) {
        if (read_env_rank(kRankEnvNames[i], rank)) {
            return 1;
        }
    }
    return 0;
}

inline int detect_current_numa()
{
    int rank = 0;
    if (detect_mpi_rank(&rank)) {
        return (rank % 16) + kNumaOffset;
    }
    const int current_cpu = sched_getcpu();
    return current_cpu >= 0 ? (current_cpu / kCoresPerNuma) + kNumaOffset : kNumaOffset;
}

inline int resolve_numa_id(int numa_id)
{
    if (numa_id < 0) {
        return detect_current_numa();
    }
    return numa_id >= kNumaOffset ? numa_id : (numa_id + kNumaOffset);
}

inline void log_event(const char* module, const char* action, const char* tag,
                      size_t size, int numa_id, const void* ptr)
{
    if (!trace_enabled()) {
        return;
    }
    fprintf(stderr,
            "[shared-workspace][%s][%s] tag=%s ptr=%p numa=%d size=%zu(%.3f MiB) current=%zu(%.3f MiB) peak=%zu(%.3f MiB)\n",
            module ? module : "unknown", action, tag ? tag : "unknown", ptr, numa_id,
            size, bytes_to_mib(size),
            g_shared_workspace.bytes_in_use, bytes_to_mib(g_shared_workspace.bytes_in_use),
            g_shared_workspace.peak_bytes, bytes_to_mib(g_shared_workspace.peak_bytes));
}

inline void* alloc_bytes(size_t size, int numa_id, const char* module, const char* tag)
{
    const int resolved_numa = resolve_numa_id(numa_id);
    const size_t mapped_size = align_to_hugepage(size);

    void* ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (ptr == MAP_FAILED) {
        ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            log_event(module, "alloc_fail", tag, mapped_size, resolved_numa, NULL);
            fprintf(stderr, "[shared-workspace] mmap failed: errno=%d (%s)\n",
                    errno, strerror(errno));
            return NULL;
        }
    }

    const unsigned long mask = 1ul << (unsigned long)resolved_numa;
    long rc = mbind(ptr, mapped_size, MPOL_BIND, &mask, kMbindMaxNodes, 0);
    if (rc != 0) {
        munmap(ptr, mapped_size);
        ptr = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            log_event(module, "alloc_fail", tag, mapped_size, resolved_numa, NULL);
            fprintf(stderr, "[shared-workspace] mmap failed: errno=%d (%s)\n",
                    errno, strerror(errno));
            return NULL;
        }
        rc = mbind(ptr, mapped_size, MPOL_BIND, &mask, kMbindMaxNodes, 0);
        if (rc != 0) {
            fprintf(stderr,
                    "[shared-workspace] mbind failed (numa=%d): errno=%d (%s)\n",
                    resolved_numa, errno, strerror(errno));
        }
    }

    memset(ptr, 0, mapped_size);
    g_shared_workspace.bytes_in_use += mapped_size;
    if (g_shared_workspace.bytes_in_use > g_shared_workspace.peak_bytes) {
        g_shared_workspace.peak_bytes = g_shared_workspace.bytes_in_use;
    }
    log_event(module, "alloc", tag, mapped_size, resolved_numa, ptr);
    return ptr;
}

inline void free_bytes(void* ptr, size_t size, const char* module, const char* tag)
{
    if (!ptr) {
        return;
    }
    const size_t mapped_size = align_to_hugepage(size);
    if (mapped_size <= g_shared_workspace.bytes_in_use) {
        g_shared_workspace.bytes_in_use -= mapped_size;
    } else {
        g_shared_workspace.bytes_in_use = 0;
    }
    log_event(module, "free", tag, mapped_size, -1, ptr);
    if (munmap(ptr, mapped_size) != 0) {
        fprintf(stderr, "[shared-workspace] munmap failed: errno=%d (%s)\n",
                errno, strerror(errno));
    }
}

inline void* alloc_external(size_t size, int numa_id, const char* module, const char* tag)
{
    return alloc_bytes(size, numa_id, module, tag);
}

inline void free_external(void* ptr, size_t size, const char* module, const char* tag)
{
    free_bytes(ptr, size, module, tag);
}

inline void reset_for_numa_change(int numa_id, const char* module)
{
    const int resolved_numa = resolve_numa_id(numa_id);
    if (g_shared_workspace.numa_id == INT_MIN) {
        g_shared_workspace.numa_id = resolved_numa;
        return;
    }
    if (g_shared_workspace.numa_id == resolved_numa) {
        return;
    }
    free_bytes(g_shared_workspace.a, g_shared_workspace.a_bytes, module, "shared.A");
    free_bytes(g_shared_workspace.b, g_shared_workspace.b_bytes, module, "shared.B");
    free_bytes(g_shared_workspace.c, g_shared_workspace.c_bytes, module, "shared.C");
    g_shared_workspace.a = NULL;
    g_shared_workspace.a_bytes = 0;
    g_shared_workspace.b = NULL;
    g_shared_workspace.b_bytes = 0;
    g_shared_workspace.c = NULL;
    g_shared_workspace.c_bytes = 0;
    g_shared_workspace.numa_id = resolved_numa;
}

inline int ensure_buffers(size_t a_bytes, size_t b_bytes, size_t c_bytes, int numa_id,
                          const char* module, void** a_ptr, void** b_ptr, void** c_ptr)
{
    if (a_bytes > kSharedABytes || b_bytes > kSharedBBytes || c_bytes > kSharedCBytes) {
        return -1;
    }

    reset_for_numa_change(numa_id, module);

    if (a_bytes != 0 && g_shared_workspace.a == NULL) {
        g_shared_workspace.a = alloc_bytes(kSharedABytes, numa_id, module, "shared.A");
        if (!g_shared_workspace.a) {
            return -1;
        }
        g_shared_workspace.a_bytes = kSharedABytes;
    }
    if (b_bytes != 0 && g_shared_workspace.b == NULL) {
        g_shared_workspace.b = alloc_bytes(kSharedBBytes, numa_id, module, "shared.B");
        if (!g_shared_workspace.b) {
            return -1;
        }
        g_shared_workspace.b_bytes = kSharedBBytes;
    }
    if (c_bytes != 0 && g_shared_workspace.c == NULL) {
        g_shared_workspace.c = alloc_bytes(kSharedCBytes, numa_id, module, "shared.C");
        if (!g_shared_workspace.c) {
            return -1;
        }
        g_shared_workspace.c_bytes = kSharedCBytes;
    }

    if (a_ptr) {
        *a_ptr = g_shared_workspace.a;
    }
    if (b_ptr) {
        *b_ptr = g_shared_workspace.b;
    }
    if (c_ptr) {
        *c_ptr = g_shared_workspace.c;
    }
    return 0;
}

}  // namespace bgemm_shared_workspace
