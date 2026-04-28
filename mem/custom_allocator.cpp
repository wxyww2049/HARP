#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h> 
#include <c10/core/DeviceType.h>
#include <iostream>
#include <cstdlib>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <unordered_map>
#include <array>
#include <functional>
#include <stdexcept>
#include <numa.h>
#include <pybind11/pybind11.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cerrno>
#include <memkind.h>
#include <string>

namespace py = pybind11;
static c10::Allocator* g_original_allocator = nullptr;
static std::once_flag g_allocator_once;
static std::atomic<bool> g_allocator_initialized{false};
static bool g_use_custom_pool = false;
static int g_hbm_numa_node = -1;
static bool g_enable_huge = false;
static thread_local bool g_allocate_on_hbm = false;
static std::atomic<uint64_t> g_alloc_id = 0;

static const int HBM_MIN_SIZE = 512 * 1024;
static const int HUGE_PAGE_SIZE = 2 * 1024 * 1024;
static const int HUGEPAGE_MIN_SIZE = 2 * HUGE_PAGE_SIZE;

static bool check(const char* env_name) {
    const char* value = std::getenv(env_name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static size_t get_env_size(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (value) {
        try {
            return std::stoull(value);
        } catch (...) {
            return SIZE_MAX;
        }
    }
    return SIZE_MAX;
}

static size_t g_max_memory[2] = { get_env_size("MAX_DDR_MEMORY"), get_env_size("MAX_HBM_MEMORY") };
static std::atomic<size_t> g_current_os_memory[2]{0, 0};


class HugeSharedMemory {
private:
    void* ptr_;
    size_t size_;
    int shmid_;

public:
    HugeSharedMemory(void* ptr, size_t size, int shmid) 
        : ptr_(ptr), size_(size), shmid_(shmid) {}

    static HugeSharedMemory create(int key, size_t size) {
        int shmid = shmget(key, size, IPC_CREAT | 0666 | SHM_HUGETLB);
        if (shmid < 0) {
            throw std::runtime_error(std::string("shmget failed (create): ") + std::strerror(errno));
        }
        
        void* ptr = shmat(shmid, nullptr, 0);
        if (ptr == (void*)-1) {
            throw std::runtime_error(std::string("shmat failed: ") + std::strerror(errno));
        }
        
        return HugeSharedMemory(ptr, size, shmid);
    }

    static HugeSharedMemory attach(int key, size_t size) {
        int shmid = shmget(key, 0, 0666);
        if (shmid < 0) {
            throw std::runtime_error(std::string("shmget failed (attach): ") + std::strerror(errno));
        }
        
        void* ptr = shmat(shmid, nullptr, 0);
        if (ptr == (void*)-1) {
            throw std::runtime_error(std::string("shmat failed: ") + std::strerror(errno));
        }
        
        return HugeSharedMemory(ptr, size, shmid);
    }

    py::memoryview get_buffer() const {
        if (!ptr_) {
            throw std::runtime_error("Memory pointer is null");
        }
        return py::memoryview::from_memory(ptr_, size_);
    }

    void close() {
        if (ptr_) {
            shmdt(ptr_);
            ptr_ = nullptr;
        }
    }

    void unlink() {
        if (shmid_ != -1) {
            shmctl(shmid_, IPC_RMID, nullptr);
            shmid_ = -1;
        }
    }

    int get_shmid() const { return shmid_; }
};

class MyMemoryPool {
public:
    struct MemoryBlock {
        void* ptr = nullptr;
        bool is_hbm_allocation = false;
        bool is_hugepage = false;
    };

private:
    struct PoolShard {
        std::mutex mtx[2];
        std::condition_variable cv[2];
        std::unordered_map<size_t, std::vector<MemoryBlock>> free_blocks[2];
    };

    static constexpr size_t NUM_SHARDS = 32;
    static std::array<PoolShard, NUM_SHARDS> shards;

    static inline size_t get_shard_idx(size_t size) {
        return (size >> 6) & (NUM_SHARDS - 1);
    }

public:
    static MemoryBlock allocate(size_t size, bool allocate_on_hbm, bool use_huge) {
        if ((allocate_on_hbm && check("ALLOC_HBM_DEBUG")) ||
            (!allocate_on_hbm && check("ALLOC_DDR_DEBUG"))) {
            std::cout << "alloc," << size << "," << (allocate_on_hbm ? "HBM" : "DDR")<<","<<g_hbm_numa_node << std::endl;
        }
        MemoryBlock block;
        
        size_t shard_idx = get_shard_idx(size);
        PoolShard& shard = shards[shard_idx];
        int mem_type = allocate_on_hbm ? 1 : 0;

        std::unique_lock<std::mutex> lock(shard.mtx[mem_type]);

        while (true) {
            auto it = shard.free_blocks[mem_type].find(size);
            if (it != shard.free_blocks[mem_type].end() && !it->second.empty()) {
                block = it->second.back();
                it->second.pop_back(); 
                return block;
            }

            size_t current = g_current_os_memory[mem_type].load(std::memory_order_relaxed);
            bool can_allocate_os = false;
            while (current + size <= g_max_memory[mem_type]) {
                if (g_current_os_memory[mem_type].compare_exchange_weak(
                        current, current + size, 
                        std::memory_order_acquire, std::memory_order_relaxed)) {
                    can_allocate_os = true;
                    break;
                }
            }

            if (can_allocate_os) {
                lock.unlock();

                if(check("ALLOC_NEW_DEBUG"))
                    std::cout<<"[MyMemoryPool] Cache miss for size " << size << " on " << (allocate_on_hbm ? "HBM" : "DDR") << " total OS allocated size:" << (current + size) <<". Allocating from OS." << std::endl;
                
                try {
                    if (use_huge) {
                        if (allocate_on_hbm) {
                            if (g_hbm_numa_node < 0) throw std::runtime_error("HBM NUMA node is not initialized");
                            if (memkind_posix_memalign(MEMKIND_HBW_HUGETLB, &block.ptr, HUGE_PAGE_SIZE, size) != 0) throw std::bad_alloc();
                        } else {
                            if(memkind_posix_memalign(MEMKIND_HUGETLB, &block.ptr, HUGE_PAGE_SIZE, size) != 0) throw std::bad_alloc();
                        }
                    } else {
                        if (allocate_on_hbm) {
                            if (g_hbm_numa_node < 0) throw std::runtime_error("HBM NUMA node is not initialized");
                            if (memkind_posix_memalign(MEMKIND_HBW, &block.ptr, 4 * 1024, size) != 0) throw std::bad_alloc();
                        } else {
                            if(memkind_posix_memalign(MEMKIND_DEFAULT, &block.ptr, 4 * 1024, size) != 0) throw std::bad_alloc();
                        }
                    }
                    block.is_hugepage = use_huge;
                    block.is_hbm_allocation = allocate_on_hbm;
                    return block;
                } catch (...) {
                    g_current_os_memory[mem_type].fetch_sub(size, std::memory_order_release);
                    throw; 
                }
            }
            if (Py_IsInitialized() && PyGILState_Check()) {
                py::gil_scoped_release release;
                std::cout<<"memory limited"<<std::endl;
                shard.cv[mem_type].wait(lock);
            } else {
                std::cout<<"memory limited"<<std::endl;
                shard.cv[mem_type].wait(lock);
            }
        }
    }

    static void free_with_size(const MemoryBlock& block, size_t size, bool use_pool) {
        if (!block.ptr) return;
        
        int mem_type = block.is_hbm_allocation ? 1 : 0;

        if (use_pool) {
            size_t shard_idx = get_shard_idx(size);
            PoolShard& shard = shards[shard_idx];
            if ((block.is_hbm_allocation && check("ALLOC_HBM_DEBUG")) ||
                (!block.is_hbm_allocation && check("ALLOC_DDR_DEBUG"))) {
                std::cout << "free," << size << "," << (block.is_hbm_allocation ? "HBM" : "DDR")<<","<<g_hbm_numa_node << std::endl;
            }
            
            {
                std::lock_guard<std::mutex> lock(shard.mtx[mem_type]);
                shard.free_blocks[mem_type][size].push_back(block);
            }
            shard.cv[mem_type].notify_all();
            
        } else {
            bool use_huge = block.is_hugepage;
            bool allocate_on_hbm = block.is_hbm_allocation;
            if (use_huge) {
                if (allocate_on_hbm) {
                    memkind_free(MEMKIND_HBW_HUGETLB, block.ptr);
                } else {
                    memkind_free(MEMKIND_HUGETLB, block.ptr);
                }
            }
            else {
                if (allocate_on_hbm) {
                    memkind_free(MEMKIND_HBW, block.ptr);
                } else {
                    memkind_free(MEMKIND_DEFAULT, block.ptr);
                }
            }
            
            g_current_os_memory[mem_type].fetch_sub(size, std::memory_order_release);
            
            for (size_t i = 0; i < NUM_SHARDS; ++i) {
                shards[i].cv[mem_type].notify_all();
            }
        }
    }

    static void clear_pool() {
        size_t total_cleared = 0;
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            PoolShard& shard = shards[i];
            
            for (int mem_type = 0; mem_type < 2; ++mem_type) {
                std::lock_guard<std::mutex> lock(shard.mtx[mem_type]);
                for (auto& pair : shard.free_blocks[mem_type]) {
                    size_t block_size = pair.first;
                    for (const MemoryBlock& block : pair.second) {
                        if (block.ptr) {
                            memkind_free(NULL, block.ptr);
                            g_current_os_memory[mem_type].fetch_sub(block_size, std::memory_order_relaxed);
                            total_cleared++;
                        }
                    }
                    pair.second.clear();
                }
                shard.free_blocks[mem_type].clear();
            }
        }
        
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            shards[i].cv[0].notify_all();
            shards[i].cv[1].notify_all();
        }
        
        std::cout << "[C++] Cleared " << total_cleared << " cached memory blocks from sharded pool." << std::endl;
    }
};

std::array<MyMemoryPool::PoolShard, MyMemoryPool::NUM_SHARDS> MyMemoryPool::shards;

struct AllocationHeader {
    uint32_t magic_number; 
    size_t total_size;
    uint64_t alloc_id;
    void* raw_ptr;
    bool is_hbm_allocation;
    bool is_hugepage;
    bool use_pool;
};
static_assert(sizeof(AllocationHeader) <= 64,
              "AllocationHeader must fit within 64 bytes");

class ProxyAllocator : public c10::Allocator {
public:
    c10::DataPtr allocate(size_t n) override {
        if (n == 0) {
            return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::CPU));
        }

        size_t total_size = n + 64;
        const bool allocate_on_hbm = g_allocate_on_hbm && total_size > HBM_MIN_SIZE;
        
        bool use_huge = false;
        if (g_enable_huge && total_size > HUGEPAGE_MIN_SIZE) {
            total_size = get_huge_size(total_size);
            use_huge = true;
        }

        MyMemoryPool::MemoryBlock block =
            MyMemoryPool::allocate(total_size, allocate_on_hbm, use_huge);
        
        void* raw_ptr = block.ptr;

        auto* header = static_cast<AllocationHeader*>(raw_ptr);
        header->magic_number = 0xDEADBEEF;
        header->total_size = total_size;
        header->raw_ptr = raw_ptr;
        header->is_hbm_allocation = allocate_on_hbm;
        header->alloc_id = ++g_alloc_id;
        header->is_hugepage = use_huge;
        header->use_pool = g_use_custom_pool;
        void* data_ptr = static_cast<char*>(raw_ptr) + 64;
        
        return c10::DataPtr(
            data_ptr, 
            raw_ptr, 
            &ProxyAllocator::proxy_pool_deleter, 
            c10::Device(c10::DeviceType::CPU)
        );

    }


    void copy_data(void* dest, const void* src, std::size_t count) const override {
        std::memcpy(dest, src, count);
    }

private:
    static inline size_t get_huge_size(size_t size) {
        return (size + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE;
    }
    static void proxy_pool_deleter(void* ctx) {
        if (ctx != nullptr) {
            const auto* header = static_cast<const AllocationHeader*>(ctx);
            size_t total_size = header->total_size;
            
            MyMemoryPool::free_with_size(
                MyMemoryPool::MemoryBlock{
                    header->raw_ptr,
                    header->is_hbm_allocation,
                    header->is_hugepage
                },
                total_size,
                header->use_pool
            ); 
        }
    }
};

static ProxyAllocator g_proxy_allocator;

void init_allocator(int hbm_numa_node = -1, bool allocate_on_hbm = false) {
    std::call_once(g_allocator_once, []() {
        g_original_allocator = c10::GetAllocator(c10::DeviceType::CPU);
        c10::SetAllocator(c10::DeviceType::CPU, &g_proxy_allocator);
        g_allocator_initialized.store(true);
    });

    const int current_hbm_numa_node =
        hbm_numa_node >= 0 ? hbm_numa_node : g_hbm_numa_node;

    if (allocate_on_hbm && current_hbm_numa_node < 0) {
        throw std::runtime_error("HBM allocation requires a valid HBM NUMA node");
    }

    if (hbm_numa_node >= 0) {
        g_hbm_numa_node = hbm_numa_node;
    }

    g_allocate_on_hbm = allocate_on_hbm;
}

void set_allocate_on_hbm(bool allocate_on_hbm) {
    if (allocate_on_hbm && g_hbm_numa_node < 0) {
        throw std::runtime_error("Set HBM NUMA node with init_allocator before enabling HBM allocation");
    }
    g_allocate_on_hbm = allocate_on_hbm;
}

void set_huge_state(bool is_huge_enable) {
    g_enable_huge = is_huge_enable;    
}

bool get_allocate_state() {
    return g_allocate_on_hbm;
}

void enable_pool() {
    if (!g_allocator_initialized.load()) throw std::runtime_error("Call init_allocator first!");
    g_use_custom_pool = true;
}

void disable_pool() {
    g_use_custom_pool = false;
}

void clear_pool() {
    MyMemoryPool::clear_pool();
}


int get_alloc_id_on_hbm(intptr_t data_ptr_int) {
    if (!data_ptr_int) return -1;
    void* data_ptr = reinterpret_cast<void*>(data_ptr_int);
    auto* header = reinterpret_cast<AllocationHeader*>(
        static_cast<char*>(data_ptr) - 64
    );
    if (header->magic_number != 0xDEADBEEF || !header->is_hbm_allocation) {
        return -1;
    }
    return header->alloc_id;
}

void copy_from_hbm(at::Storage src, at::Storage dst) {

    const size_t size_bytes = src.nbytes();
    if (dst.nbytes() < size_bytes) {
        throw std::runtime_error("Destination storage is too small for the copy operation");
    }
    if (size_bytes == 0) {
        return;
    }
    
    {
        py::gil_scoped_release release;
        const void* src_ptr = src.data();
        void* dst_ptr = dst.mutable_data();
        if (src_ptr == nullptr || dst_ptr == nullptr) {
            throw std::runtime_error("Source or destination storage has a null data pointer");
        }
        if (src_ptr == dst_ptr) {
            return;
        }
        std::memcpy(dst_ptr, src_ptr, size_bytes);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_allocator", &init_allocator, "Initialize the proxy allocator");
    m.def("set_allocate_on_hbm", &set_allocate_on_hbm, "Switch between DDR and HBM allocation");
    m.def("set_huge_state", &set_huge_state, "Switch huge state");
    m.def("get_allocate_state", &get_allocate_state, "Get current allocation mode (HBM or DDR)");
    m.def("enable", &enable_pool, "Enable custom memory pool");
    m.def("disable", &disable_pool, "Disable custom memory pool");
    m.def("clear_pool", &clear_pool, "Free all cached memory in the pool to the OS");
    m.def("get_alloc_id_on_hbm", &get_alloc_id_on_hbm, "Check if tensor's underlying memory is on HBM");
    m.def("copy_from_hbm", &copy_from_hbm, "Copy all data from src Storage to dst Storage");
    py::class_<HugeSharedMemory>(m, "HugeSharedMemory")
        .def_static("create", &HugeSharedMemory::create, "Create and attach huge shared memory")
        .def_static("attach", &HugeSharedMemory::attach, "Attach to existing huge shared memory")
        .def("get_buffer", &HugeSharedMemory::get_buffer, "Get python memoryview of the shared memory")
        .def("close", &HugeSharedMemory::close, "Detach the shared memory")
        .def("unlink", &HugeSharedMemory::unlink, "Destroy the shared memory")
        .def_property_readonly("shmid", &HugeSharedMemory::get_shmid);
}