#include <torch/extension.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>
#include <iostream>
#include <string>
#include <atomic>
#include <vector>
#include <arm_sve.h>
#include <sched.h>
#include <cstdlib>
#include <cstring>

#ifndef ACCUMULATE_ENABLE_PROFILER
#define ACCUMULATE_ENABLE_PROFILER 0
#endif

#if ACCUMULATE_ENABLE_PROFILER
#define ACCUMULATE_PROFILER_ENABLED 1
#else
#define ACCUMULATE_PROFILER_ENABLED 0
#endif

#if ACCUMULATE_PROFILER_ENABLED
#define ACCUMULATE_RECORD_FUNCTION(name_literal) \
    RECORD_FUNCTION((name_literal), std::vector<c10::IValue>())
#else
#define ACCUMULATE_RECORD_FUNCTION(name_literal) \
    do {                                         \
    } while (0)
#endif
int get_dp_comm_mode() {
    const char* value = std::getenv("COMM_MODE");
    if (value == nullptr) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "2") == 0) return 2;
    return 0;
}
static inline bool env_is_true(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (value == nullptr) {
        return false;
    }

    return std::strcmp(value, "1") == 0 ||
           std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "True") == 0 ||
           std::strcmp(value, "TRUE") == 0;
}
static int comm_mode;
struct SharedSyncBlock {
    pthread_mutex_t add_lock;
    pthread_mutex_t sync_mtx;
    pthread_cond_t  sync_cv;
    int count; 
    int expected_count;
    volatile int initialized;
};

struct Task {
    std::string tensor_id;
    at::Tensor private_tensor;
    at::Tensor shared_tensor;
    SharedSyncBlock* sync_block;
#if ACCUMULATE_PROFILER_ENABLED
    at::ThreadLocalState tls;
#endif
};

struct CommTask {
    std::string tensor_id;
    torch::Tensor chunk_tensor;
    torch::Tensor tensor;
#if ACCUMULATE_PROFILER_ENABLED
    at::ThreadLocalState tls;
#endif
};

static std::deque<Task> task_queue;
static std::mutex queue_mutex;
static std::condition_variable queue_cond;
static int pending_tasks = 0;
static std::condition_variable sync_cond;

static std::queue<CommTask> comm_queue;
static std::mutex comm_mutex;
static std::condition_variable comm_cv;

static int pending_comm_tasks = 0;
static std::condition_variable comm_sync_cond;

static std::vector<std::thread> worker_threads;
static std::thread comm_thread;
static bool stop_flag = false;

static std::unordered_map<std::string, SharedSyncBlock*> lock_cache;
static std::mutex cache_mutex;

SharedSyncBlock* get_shared_sync_block(const std::string& tensor_id, int expected_count = 4) {
    std::lock_guard<std::mutex> lk(cache_mutex);
    
    if (lock_cache.find(tensor_id) != lock_cache.end()) {
        return lock_cache[tensor_id];
    }

    std::string shm_name = "/" + tensor_id + "_sync_block";
    int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    bool is_creator = (fd >= 0);
    
    if (!is_creator) {
        fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (fd < 0) throw std::runtime_error("Failed to open shared memory lock.");
        struct stat statbuf;
        while (true) {
            fstat(fd, &statbuf);
            if (statbuf.st_size == sizeof(SharedSyncBlock)) break;
            std::this_thread::yield();
        }
    } else {
        ftruncate(fd, sizeof(SharedSyncBlock));
    }

    void* ptr = mmap(NULL, sizeof(SharedSyncBlock), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed.");

    SharedSyncBlock* block = static_cast<SharedSyncBlock*>(ptr);
    if (is_creator) {
        block->initialized = 0;
        pthread_mutexattr_t mattr;
        pthread_mutexattr_init(&mattr);
        pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
        
        pthread_mutex_init(&block->add_lock, &mattr);
        pthread_mutex_init(&block->sync_mtx, &mattr);
        pthread_mutexattr_destroy(&mattr);

        pthread_condattr_t cattr;
        pthread_condattr_init(&cattr);
        pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
        pthread_cond_init(&block->sync_cv, &cattr);
        pthread_condattr_destroy(&cattr);

        block->count = expected_count;
        block->expected_count = expected_count;
        block->initialized = 1;
    } else {
        while (block->initialized == 0) {
            std::this_thread::yield();
        }
    }
    
    lock_cache[tensor_id] = block;
    return block;
}

static inline void sve_add(bfloat16_t * A,const bfloat16_t *B, int len){
    int i = 0;
    int step = svcnth();
    
    svbool_t pg32 = svptrue_b32();
    svbool_t pg16 = svptrue_b16();
    int t_new = len / step * step;
    int t_tail = len - t_new;
    #pragma unroll 4
    for(; i < t_new; i += step){
        svuint16_t vb1 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&A[i]));
        svuint16_t vb2 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&B[i]));

        svuint16_t vb_sign1 = svuzp1_u16(vb1, vb1);
        svuint16_t vb_sign2 = svuzp1_u16(vb2, vb2);

        svuint32_t lo1 = svunpklo_u32(vb_sign1);
        svuint32_t lo2 = svunpklo_u32(vb_sign2);
        lo1 = svlsl_n_u32_x(pg32, lo1, 16);
        lo2 = svlsl_n_u32_x(pg32, lo2, 16);

        svfloat32_t v1 = svreinterpret_f32_u32(lo1);
        svfloat32_t v2 = svreinterpret_f32_u32(lo2);

        v1 = svadd_f32_m(pg32, v1, v2);
        svbfloat16_t out = svcvt_bf16_x(pg32, v1);

        vb_sign1 = svuzp2_u16(vb1, vb1);
        vb_sign2 = svuzp2_u16(vb2, vb2);

        lo1 = svunpkhi_u32(vb_sign1);
        lo2 = svunpkhi_u32(vb_sign2);
        lo1 = svlsl_n_u32_x(pg32, lo1, 16);
        lo2 = svlsl_n_u32_x(pg32, lo2, 16);

        v1 = svreinterpret_f32_u32(lo1);
        v2 = svreinterpret_f32_u32(lo2);

        v1 = svadd_f32_m(pg32, v1, v2);

        out = svcvtnt_bf16_x(out, pg32, v1);
        svst1_bf16(pg16, &A[i], out);
    }

    if(t_tail!=0){
        pg32 = svwhilelt_b32((uint64_t)0, (uint64_t)t_tail/2);
        pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)t_tail);
        svuint16_t vb1 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&A[i]));
        svuint16_t vb2 = svld1_u16(pg16, reinterpret_cast<const uint16_t*>(&B[i]));

        svuint16_t vb_sign1 = svuzp1_u16(vb1, vb1);
        svuint16_t vb_sign2 = svuzp1_u16(vb2, vb2);

        svuint32_t lo1 = svunpklo_u32(vb_sign1);
        svuint32_t lo2 = svunpklo_u32(vb_sign2);
        lo1 = svlsl_n_u32_x(pg32, lo1, 16);
        lo2 = svlsl_n_u32_x(pg32, lo2, 16);

        svfloat32_t v1 = svreinterpret_f32_u32(lo1);
        svfloat32_t v2 = svreinterpret_f32_u32(lo2);

        v1 = svadd_f32_m(pg32, v1, v2);
        svbfloat16_t out = svcvt_bf16_x(pg32, v1);

        vb_sign1 = svuzp2_u16(vb1, vb1);
        vb_sign2 = svuzp2_u16(vb2, vb2);

        lo1 = svunpkhi_u32(vb_sign1);
        lo2 = svunpkhi_u32(vb_sign2);
        lo1 = svlsl_n_u32_x(pg32, lo1, 16);
        lo2 = svlsl_n_u32_x(pg32, lo2, 16);

        v1 = svreinterpret_f32_u32(lo1);
        v2 = svreinterpret_f32_u32(lo2);

        v1 = svadd_f32_m(pg32, v1, v2);

        out = svcvtnt_bf16_x(out, pg32, v1);
        svst1_bf16(pg16, &A[i], out);
    }
}

void bind_current_thread_to_core(int core_id) {
    if (core_id < 0) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        throw std::runtime_error(
            std::string("pthread_setaffinity_np failed: ") + std::strerror(rc)
        );
    }
}

void set_current_thread_name(const char* name) {
#ifdef __linux__
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}

void copy_or_add_tensor(const at::Tensor& private_tensor, at::Tensor& shared_tensor, bool is_add) {
    TORCH_CHECK(private_tensor.defined(), "private_tensor is undefined");
    TORCH_CHECK(shared_tensor.defined(), "shared_tensor is undefined");

    TORCH_CHECK(
        private_tensor.sizes() == shared_tensor.sizes(),
        "Shape mismatch: private_tensor shape is ", private_tensor.sizes(),
        ", shared_tensor shape is ", shared_tensor.sizes()
    );

    TORCH_CHECK(
        private_tensor.is_contiguous(),
        "private tensor must be contiguous"
    );
    TORCH_CHECK(
        shared_tensor.is_contiguous(),
        "shared tensor must be contiguous"
    )

    TORCH_CHECK(
        private_tensor.scalar_type() == shared_tensor.scalar_type(),
        "Dtype mismatch: private_tensor dtype is ", private_tensor.scalar_type(),
        ", shared_tensor dtype is ", shared_tensor.scalar_type()
    );
    TORCH_CHECK(
        private_tensor.scalar_type() == at::kBFloat16,
        "Dtype mismatch: private_tensor dtype is ", private_tensor.scalar_type(),
        ", shared_tensor dtype is ", shared_tensor.scalar_type()
    );

    TORCH_CHECK(
        private_tensor.layout() == at::kStrided &&
        shared_tensor.layout() == at::kStrided,
        "Only strided tensors are supported"
    );
    const bfloat16_t* private_ptr = reinterpret_cast<const bfloat16_t*>(private_tensor.data_ptr<at::BFloat16>());
    bfloat16_t* shared_ptr = reinterpret_cast<bfloat16_t*>(shared_tensor.data_ptr<at::BFloat16>());
    if (is_add) {
        sve_add(shared_ptr, private_ptr, private_tensor.numel());
    } else {
        std::memcpy(shared_ptr, private_ptr, private_tensor.numel() * sizeof(bfloat16_t));
    }
}
void worker_loop(int core_id) {
    set_current_thread_name("worker_loop");
    bind_current_thread_to_core(core_id);

    while (true) {
        Task task;
        bool got_task = false;

        {
            std::unique_lock<std::mutex> lk(queue_mutex);
            queue_cond.wait(lk, [&]{ return !task_queue.empty() || stop_flag; });
            if (stop_flag && task_queue.empty()) break;
            
            if (!task_queue.empty()) {
                task = task_queue.front();
                task_queue.pop_front();
                got_task = true;
            }
        }

        if (got_task) {
            if (pthread_mutex_trylock(&task.sync_block->add_lock) == 0) {
                try {
#if ACCUMULATE_PROFILER_ENABLED
                    at::ThreadLocalStateGuard tls_guard(task.tls);
#endif
                    torch::NoGradGuard no_grad;
                    ACCUMULATE_RECORD_FUNCTION("accumulate::worker_loop");
                    bool is_add = false;

                    pthread_mutex_lock(&task.sync_block->sync_mtx);
                    is_add = (task.sync_block->count < task.sync_block->expected_count);
                    pthread_mutex_unlock(&task.sync_block->sync_mtx);

                    copy_or_add_tensor(task.private_tensor, task.shared_tensor, is_add);
                } catch (const std::exception& e) {
                    std::cerr << "[Error] Add failed: " << e.what() << std::endl;
                }

                pthread_mutex_lock(&task.sync_block->sync_mtx);
                task.sync_block->count--;
                if (task.sync_block->count == 0) {
                    pthread_cond_broadcast(&task.sync_block->sync_cv);
                }
                pthread_mutex_unlock(&task.sync_block->sync_mtx);

                pthread_mutex_unlock(&task.sync_block->add_lock);


                bool notify_done = false;
                {
                    std::lock_guard<std::mutex> lk(queue_mutex);
                    --pending_tasks;
                    notify_done = (pending_tasks == 0);
                }
                if (notify_done) sync_cond.notify_all();
                
            } else {
                {
                    std::lock_guard<std::mutex> lk(queue_mutex);
                    size_t delay_offset = 4;
                    size_t insert_pos = std::min(task_queue.size(), delay_offset);
                    task_queue.insert(task_queue.begin() + insert_pos, task);
                }
                queue_cond.notify_one();
                std::this_thread::yield(); 
            }
        }
    }
}

void comm_loop(int comm_core_id, c10::intrusive_ptr<c10d::ProcessGroup> dp_group,c10::intrusive_ptr<c10d::ProcessGroup> rs_group) {
    set_current_thread_name("comm_loop");
    bind_current_thread_to_core(comm_core_id); 

    while (true) {
        CommTask comm_task;
        {
            std::unique_lock<std::mutex> lk(comm_mutex);
            comm_cv.wait(lk, []{ return !comm_queue.empty() || stop_flag; });
            if (stop_flag && comm_queue.empty()) break;
            
            comm_task = comm_queue.front();
            comm_queue.pop();
        }

        SharedSyncBlock* sync_block = get_shared_sync_block(comm_task.tensor_id);

        try {
#if ACCUMULATE_PROFILER_ENABLED
            at::ThreadLocalStateGuard tls_guard(comm_task.tls);
#endif
            torch::NoGradGuard no_grad;
            ACCUMULATE_RECORD_FUNCTION("accumulate::comm_loop");

            pthread_mutex_lock(&sync_block->sync_mtx);
            while (sync_block->count > 0) {
                pthread_cond_wait(&sync_block->sync_cv, &sync_block->sync_mtx);
            }
            pthread_mutex_unlock(&sync_block->sync_mtx);

            if (!env_is_true("DISABLE_DP")) {
                if (comm_mode == 1) {
                    c10d::ReduceScatterOptions rs_opts;
                    rs_opts.reduceOp = c10d::ReduceOp::SUM;
                    rs_opts.asyncOp = false;
                    
                    c10d::BarrierOptions barrier_opts; 
                    auto barrier_work = rs_group->barrier(barrier_opts);
                    barrier_work->wait();

                    auto rs_work = rs_group->_reduce_scatter_base(
                        comm_task.chunk_tensor,
                        comm_task.tensor,
                        rs_opts);
                    rs_work->wait();
                } else if (comm_mode == 2){
                    std::vector<torch::Tensor> tensors = { comm_task.tensor };
                    c10d::AllreduceOptions allreduce_opts;
                    allreduce_opts.reduceOp = c10d::ReduceOp::SUM;
                    c10d::BarrierOptions barrier_opts; 
                    auto barrier_work = rs_group->barrier(barrier_opts);
                    barrier_work->wait();
                    auto rs_work = rs_group->allreduce(tensors, allreduce_opts);
                    rs_work->wait();
                } else {
                    std::vector<torch::Tensor> tensors = { comm_task.tensor };
                    c10d::AllreduceOptions allreduce_opts;
                    allreduce_opts.reduceOp = c10d::ReduceOp::SUM;
                    c10d::BarrierOptions barrier_opts; 
                    auto barrier_work = dp_group->barrier(barrier_opts);
                    barrier_work->wait();
                    auto dp_work = dp_group->allreduce(tensors, allreduce_opts);
                    dp_work->wait();
                }
                
                if (comm_mode > 0) {
                    std::vector<torch::Tensor> chunk_tensors = { comm_task.chunk_tensor };
                    c10d::AllreduceOptions dp_opts;
                    dp_opts.reduceOp = c10d::ReduceOp::SUM;
                    c10d::BarrierOptions barrier_opts; 
                    auto barrier_work = dp_group->barrier(barrier_opts);
                    barrier_work->wait();
                    auto dp_work = dp_group->allreduce(chunk_tensors, dp_opts);
                    dp_work->wait();
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "[Error] Comm failed: " << e.what() << std::endl;
        }
        
        pthread_mutex_lock(&sync_block->sync_mtx);
        sync_block->count = sync_block->expected_count;
        pthread_mutex_unlock(&sync_block->sync_mtx);
        
        bool notify_comm_done = false;
        {
            std::lock_guard<std::mutex> lk(comm_mutex);
            --pending_comm_tasks;
            notify_comm_done = (pending_comm_tasks == 0);
        }
        if (notify_comm_done) {
            comm_sync_cond.notify_all();
        }
    }
}

void init_system(const std::vector<int>& core_ids, int comm_core_id, const c10::intrusive_ptr<c10d::ProcessGroup>& dp_group,const c10::intrusive_ptr<c10d::ProcessGroup>& rs_group) {
    if (core_ids.empty()) throw std::runtime_error("core_ids must not be empty");
    if (!worker_threads.empty() || comm_thread.joinable()) {
        throw std::runtime_error("shutdown_system must be called before reinitializing");
    }

    stop_flag = false;
    pending_tasks = 0;
    comm_mode = get_dp_comm_mode();
    for (int core_id : core_ids) {
        worker_threads.emplace_back(worker_loop, core_id);
    }
    comm_thread = std::thread(comm_loop, comm_core_id, dp_group,rs_group);
}

void push_task(at::Tensor private_tensor, at::Tensor shared_tensor, std::string tensor_id, int expected_count) {
    Task task;
    task.tensor_id = tensor_id;
    task.private_tensor = private_tensor;
    task.shared_tensor = shared_tensor;
#if ACCUMULATE_PROFILER_ENABLED
    task.tls = at::ThreadLocalState();
#endif
    
    {
        pybind11::gil_scoped_release release;
        task.sync_block = get_shared_sync_block(tensor_id, expected_count); 
        
        {
            std::lock_guard<std::mutex> lk(queue_mutex);
            task_queue.push_back(task);
            pending_tasks++;
        }
        queue_cond.notify_one();
    } 
}

void push_comm_task(torch::Tensor tensor, torch::Tensor chunk_tensor, std::string tensor_id) {
    CommTask task;
    task.tensor_id = std::move(tensor_id);
    task.tensor = std::move(tensor);
    task.chunk_tensor = std::move(chunk_tensor);
#if ACCUMULATE_PROFILER_ENABLED
    task.tls = at::ThreadLocalState();
#endif

    std::lock_guard<std::mutex> lk(comm_mutex);
    comm_queue.push(std::move(task));
    pending_comm_tasks++;
    comm_cv.notify_one();
}

void wait_all() {
    pybind11::gil_scoped_release release; 
    std::unique_lock<std::mutex> lk(queue_mutex);
    sync_cond.wait(lk, []{ return pending_tasks == 0; });
}
void wait_comm_all() {
    pybind11::gil_scoped_release release; 
    std::unique_lock<std::mutex> lk(comm_mutex);
    comm_sync_cond.wait(lk, []{ return pending_comm_tasks == 0; });
}
void shutdown_system() {
    {
        std::lock_guard<std::mutex> lk1(queue_mutex);
        std::lock_guard<std::mutex> lk2(comm_mutex);
        stop_flag = true;
    }
    queue_cond.notify_all();
    comm_cv.notify_all();

    for (auto& worker_thread : worker_threads) {
        if (worker_thread.joinable()) worker_thread.join();
    }
    worker_threads.clear();
    
    if (comm_thread.joinable()) comm_thread.join();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init_system);
    m.def("push_task", &push_task);
    m.def("push_comm_task", &push_comm_task);
    m.def("wait_all", &wait_all, "Block until all tasks in the queue are completed");
    m.def("wait_comm_all", &wait_comm_all, "Block until all communication tasks are completed");
    m.def("shutdown", &shutdown_system);
}
