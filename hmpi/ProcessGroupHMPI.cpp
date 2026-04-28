#include "ProcessGroupHMPI.hpp"

#include <iostream>
#include <map>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>

#include <pybind11/chrono.h>

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

#include <dirent.h>
#include <sys/types.h>
#include <set>
#include <string>

namespace {
std::set<pid_t> get_current_tids() {
    std::set<pid_t> tids;
    std::string task_path = "/proc/self/task";
    DIR* dir = opendir(task_path.c_str());
    if (!dir) return tids;

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        tids.insert(std::stoi(ent->d_name));
    }
    closedir(dir);
    return tids;
}

void evict_tids_to_core(const std::set<pid_t>& tids, int target_cpuid) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpuid, &cpuset);

    for (pid_t tid : tids) {
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("[HMPI Backend] sched_setaffinity failed for background thread");
        }
    }
}

} // namespace

PYBIND11_MODULE(ProcessGroupHMPI, m) {
    py::class_<c10d::ProcessGroupHMPI, 
               c10d::Backend, 
               c10::intrusive_ptr<c10d::ProcessGroupHMPI>>(
        m, "ProcessGroupHMPI"
    );
    py::class_<c10d::ProcessGroupHMPI::Options, 
              //  c10d::Backend::Options, 
               c10::intrusive_ptr<c10d::ProcessGroupHMPI::Options>>(
        m, "Options", "HMPI-specific process group options")
        
        .def(py::init<int>(),
             py::arg("thread_id_offset") = 0)
             
        .def_readwrite("thread_id_offset", &c10d::ProcessGroupHMPI::Options::thread_id_offset);

    m.def("createProcessGroupHMPI", &c10d::ProcessGroupHMPI::createProcessGroupHMPI);
    py::object dist_module = py::module::import("torch.distributed");
    py::object register_backend = dist_module.attr("Backend").attr("register_backend");
 
    register_backend(
        "hmpi",                           
        m.attr("createProcessGroupHMPI"), 
        py::bool_(true),                  
        py::str("cpu")                    
    );
}

namespace c10d {

#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)

namespace bf16_mpi {
    static MPI_Datatype g_mpi_bf16_type = MPI_DATATYPE_NULL;
    static MPI_Op       g_mpi_bf16_sum_op = MPI_OP_NULL;
    static std::once_flag g_bf16_init_flag;
    static bool g_bf16_initialized = false;


    void initBF16MpiSupport() {
      std::call_once(g_bf16_init_flag, []() {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized) {
          throw std::runtime_error(
              "initBF16MpiSupport: MPI must be initialized before "
              "registering BF16 custom datatype and op");
        }
        int ret = MPI_Type_contiguous(2, MPI_BYTE, &g_mpi_bf16_type);
        if (ret != MPI_SUCCESS) {
            throw std::runtime_error(
                "initBF16MpiSupport: MPI_Type_contiguous failed for BF16");
        }

        ret = MPI_Type_commit(&g_mpi_bf16_type);
        if (ret != MPI_SUCCESS) {
            throw std::runtime_error(
                "initBF16MpiSupport: MPI_Type_commit failed for BF16");
        }

        ret = MPI_Op_create(
            reinterpret_cast<MPI_User_function*>(mpi_bf16_sum_op),
            /*commute=*/1,
            &g_mpi_bf16_sum_op);
        if (ret != MPI_SUCCESS) {
            throw std::runtime_error(
                "initBF16MpiSupport: MPI_Op_create failed for BF16 SUM");
        }

        g_bf16_initialized = true;

      });
    }
    

    void finalizeBF16MpiSupport() {
      if (!g_bf16_initialized) return;
    
      if (g_mpi_bf16_sum_op != MPI_OP_NULL) {
          MPI_Op_free(&g_mpi_bf16_sum_op);
          g_mpi_bf16_sum_op = MPI_OP_NULL;
      }
      if (g_mpi_bf16_type != MPI_DATATYPE_NULL) {
          MPI_Type_free(&g_mpi_bf16_type);
          g_mpi_bf16_type = MPI_DATATYPE_NULL;
      }
      g_bf16_initialized = false;
    }

    MPI_Datatype getBF16MpiDatatype() {
      if (!g_bf16_initialized) {
          throw std::runtime_error(
              "getBF16MpiDatatype: BF16 MPI support not initialized. "
              "Call initBF16MpiSupport() first.");
      }
      return g_mpi_bf16_type;
    }

    MPI_Op getBF16MpiSumOp() {
      if (!g_bf16_initialized) {
          throw std::runtime_error(
              "getBF16MpiSumOp: BF16 MPI support not initialized. "
              "Call initBF16MpiSupport() first.");
      }
      return g_mpi_bf16_sum_op;
    }

    
}


namespace {

std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};



MPI_Datatype getMpiDatatype(at::ScalarType scalar_type) {
  if (scalar_type == at::kBFloat16) {
      return c10d::bf16_mpi::getBF16MpiDatatype();
  }
  auto it = mpiDatatype.find(scalar_type);
  TORCH_CHECK(it != mpiDatatype.end(),
      "HMPI Backend: unsupported scalar type: ", toString(scalar_type));
  return it->second;
}

MPI_Op getMpiOp(ReduceOp::RedOpType op, at::ScalarType scalar_type) {
  if (scalar_type == at::kBFloat16) {
      TORCH_CHECK(op == ReduceOp::SUM,
          "HMPI Backend: BF16 custom reduction currently only supports SUM. "
          "Received op=", static_cast<int>(op),
          ". To add MIN/MAX/PROD, implement additional MPI_Op_create in "
          "bf16_mpi_support and extend this function.");
      return c10d::bf16_mpi::getBF16MpiSumOp();
  }
  auto it = mpiOp.find(op);
  TORCH_CHECK(it != mpiOp.end(),
      "HMPI Backend: unsupported reduce op: ", static_cast<int>(op));
  return it->second;
}


void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }

}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(
        false, "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      TORCH_CHECK(false, "Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

}

std::vector<at::Tensor> ProcessGroupHMPI::WorkHMPI::result() {
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupHMPI::WorkHMPI::getFuture() {
  return future_;
}

void ProcessGroupHMPI::WorkHMPI::finishWorkHMPIError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupHMPI::WorkHMPI::finishWorkHMPI() {
  future_->markCompleted(at::IValue(outputTensors_));
  finish();
}

ProcessGroupHMPI::AsyncWork::AsyncWork(
    MPI_Request request,
    std::vector<at::Tensor> outputTensors,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupHMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << '\n';
    std::terminate();
  }
}

bool ProcessGroupHMPI::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupHMPI::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    TORCH_CHECK(
        false,
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

int ProcessGroupHMPI::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupHMPI::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    if (Work::recordFunctionEndCallback_) {
      Work::recordFunctionEndCallback_();
      Work::recordFunctionEndCallback_ = nullptr;
    }
    return true;
  }

  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }

  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }

  return true;
}

void ProcessGroupHMPI::AsyncWork::abort(){
    TORCH_CHECK(false, "ProcessGroupHMPI::AsyncWork::abort not implemented.")}

std::vector<at::Tensor> ProcessGroupHMPI::AsyncWork::result() {
  return outputTensors_;
}

void ProcessGroupHMPI::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

int ProcessGroupHMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupHMPI::pgGlobalMutex_;
c10::once_flag ProcessGroupHMPI::onceFlagInitMPI;

void ProcessGroupHMPI::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  c10d::bf16_mpi::finalizeBF16MpiSupport();
  MPI_CHECK(MPI_Finalize());
}


void ProcessGroupHMPI::initMPIOnce() {
  c10::call_once(onceFlagInitMPI, []() {
    int mpi_was_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    if (mpi_was_initialized == 0) {

      std::set<pid_t> tids_before = get_current_tids();

      MPI_CHECK(MPI_Init_thread(
          nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpiThreadSupport_));

      if (mpiThreadSupport_ < MPI_THREAD_MULTIPLE) {
        TORCH_CHECK(
            false,
            "Used MPI implementation doesn't have the "
            "minimum level of threading support: "
            "MPI_THREAD_SERIALIZED. This is required by "
            "c10d package");
      }
      if (std::atexit(ProcessGroupHMPI::mpiExit)) {
        TORCH_CHECK(false, "Fail to register the MPI exit handler");
      }
      c10d::bf16_mpi::initBF16MpiSupport();
      std::set<pid_t> tids_after = get_current_tids();
      std::set<pid_t> mpi_background_tids;
      
      for (pid_t tid : tids_after) {
          if (tids_before.find(tid) == tids_before.end()) {
              mpi_background_tids.insert(tid);
          }
      }

      if (!mpi_background_tids.empty()) {
          int global_rank = -1;
          MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
          
          const int cores_per_rank = std::getenv("CORES_PER_PROCESS") ? std::atoi(std::getenv("CORES_PER_PROCESS"))  : 38 ;
          
          int ranks_per_node = (int)(608 / cores_per_rank);
          
          int target_id = (global_rank % ranks_per_node + 1) * cores_per_rank - 2;
          
          evict_tids_to_core(mpi_background_tids, target_id);
      }


    } else {
      TORCH_WARN_ONCE("MPI was previously initialized.");
      c10d::bf16_mpi::initBF16MpiSupport();
    }
  });
}
c10::intrusive_ptr<ProcessGroupHMPI> ProcessGroupHMPI::createProcessGroupHMPI(
   const c10d::DistributedBackendOptions & opts,
   const c10::intrusive_ptr<Options>& pg_options) {
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;
  int thread_id_offset = 0;
  if(pg_options) {
    thread_id_offset = pg_options -> thread_id_offset;
  }
  std::vector<int> ranks(opts.global_ranks_in_group.begin(), 
                         opts.global_ranks_in_group.end());
  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    if (!ranks.empty()) {
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      constexpr int kMaxNumRetries = 3;
      bool groupComm_updated = false;
      MPI_Barrier(MPI_COMM_WORLD);
      for (const auto i : c10::irange(kMaxNumRetries)) {
        (void)i;
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          groupComm_updated = true;
          break;
        }
      }
      MPI_CHECK(groupComm_updated);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      if (rank < 0 || size < 0) {
        TORCH_CHECK(false, "Failed to get the world_size / rank");
      }
    }
  }

  if (groupComm == MPI_COMM_NULL) {
    return c10::intrusive_ptr<ProcessGroupHMPI>();
  }

  return c10::make_intrusive<ProcessGroupHMPI>(rank, size, groupComm, thread_id_offset);
}

void set_thread_to_cpu(int cpu_id, std::thread &current_thread){
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  
  pthread_t thread_id = current_thread.native_handle();
  if(pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset) != 0){
    perror("pthread_setaffinity_np error from custom comm backend hmpi, target cpuid");
  }
  std::string thread_name = "HMPI_" + std::to_string(cpu_id);
  pthread_setname_np(thread_id, thread_name.substr(0, 15).c_str());

}
const int cores_per_rank = std::getenv("CORES_PER_PROCESS") ? std::atoi(std::getenv("CORES_PER_PROCESS"))  : 38 ;
const int rank_core_offset = std::getenv("MPI_CORE_OFFSET") ? std::atoi(std::getenv("MPI_CORE_OFFSET"))    : 0 ;
ProcessGroupHMPI::ProcessGroupHMPI(int rank, int size, void* pgComm, int thread_id_offset)
    : Backend(rank, size), stop_(false), pgComm_((MPI_Comm)pgComm) {
  if (pgComm_ == MPI_COMM_NULL) {
    TORCH_CHECK(false, "pgComm_ must not be MPI_COMM_NULL");
  }

  workerThread_ = std::thread(&ProcessGroupHMPI::runLoop, this);
  int global_rank = -1;
  int local_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  MPI_Comm_rank(pgComm_, &local_rank);
  int ranks_per_node = (int)(608/cores_per_rank);
  
  int target_cpuid = (int)(cores_per_rank * (global_rank % ranks_per_node) + rank_core_offset + thread_id_offset);
  if(target_cpuid >= cores_per_rank * ((global_rank % ranks_per_node) + 1)) {
    TORCH_WARN("[WARNNNNNNNNNNNNNNNNNNNNING]: HMPI run loop threads exceed process range!!!!")
  }

  set_thread_to_cpu(target_cpuid, workerThread_);

  init();
}

ProcessGroupHMPI::~ProcessGroupHMPI() {
  destroy();
}

void ProcessGroupHMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  stop_ = true;

  lock.unlock();
  queueProduceCV_.notify_all();

  workerThread_.join();
}

void ProcessGroupHMPI::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupHMPI::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto workTuple = std::move(queue_.front());

    queue_.pop_front();

    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    lock.unlock();
    queueConsumeCV_.notify_one();

    try {
      workEntry->run(workEntry);
      work->finishWorkHMPI();
    } catch (...) {
      work->finishWorkHMPIError(std::current_exception());
    }

    lock.lock();
  }
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::enqueue(
    std::unique_ptr<WorkEntry> entry,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto work =
      c10::make_intrusive<WorkHMPI>(entry->dst, profilingTitle, inputTensors);
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.emplace_back(std::move(entry), work);
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:broadcast",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  checkSingleTensor(tensors);


  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            getMpiOp(opts.reduceOp, data.scalar_type()),
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_reduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(false, "allreduce_coalesced is currently not supported with MPI");
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            getMpiOp(opts.reduceOp, data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:reduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}


c10::intrusive_ptr<Work> ProcessGroupHMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    TORCH_CHECK(
        false,
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    TORCH_CHECK(
        false,
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor> outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            pgComm_));

        for (const auto i : c10::irange(outputDataVec.size())) {
          outputDataVec[i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors[0], std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_gather",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}


c10::intrusive_ptr<Work> ProcessGroupHMPI::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupHMPI does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (!outputTensors.empty()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      TORCH_CHECK(false, "Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        std::vector<at::Tensor> dstdata = entry->dst;
        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        c10::DeviceGuard guard(data.device());
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            recvbuf,
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            opts.rootRank,
            pgComm_));

        if (rank_ == opts.rootRank) {
          const std::vector<at::Tensor>& outputDataVec = entry->dst;
          for (const auto i : c10::irange(outputDataVec.size())) {
            outputDataVec.at(i).copy_(
                flatOutputTensor[static_cast<int64_t>(i)]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors[0], std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    auto entry =
        std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
 	std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (!inputTensors.empty()) {
      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should be 0 "
          "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      TORCH_CHECK(false, "Scatter: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          for (const auto i : c10::irange(inputDataVec.size())) {
            flatInputTensor[static_cast<int64_t>(i)].copy_(inputDataVec.at(i));
          }
        }

        c10::DeviceGuard guard(data.device());
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors[0], &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : std::nullopt);
  } else {
    auto entry = std::make_unique<WorkEntry>(
        nullptr, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : std::nullopt);
  }
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::reduce_scatter(
  std::vector<at::Tensor>& outputTensors,
  std::vector<std::vector<at::Tensor>>& inputTensors,
  const ReduceScatterOptions& opts) {
  checkSingleTensor(outputTensors);
  if (inputTensors.size() != 1) {
    TORCH_CHECK(
      false,
      "MPI process group only supports a single "
      "tensor op");
  }
  if (static_cast<size_t>(size_) != inputTensors[0].size()) {
    TORCH_CHECK(
      false,
      "Reduce scatter: number of input tensors should equal "
      "to the world size");
  }
  checkSameSizeAndType(outputTensors[0], inputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
    [opts, this](std::unique_ptr<WorkEntry>& entry) {
      auto data = (entry->dst)[0];
      auto flatInputTensor = newLikeFlat(entry->src);
      for (const auto i : c10::irange(entry->src.size())) {
        flatInputTensor[static_cast<int64_t>(i)].copy_(entry->src[i]);
      }
      int recvcount = flatInputTensor.numel() / size_;

      c10::DeviceGuard guard(data.device());
      MPI_CHECK(MPI_Reduce_scatter_block(
          flatInputTensor.data_ptr(),
          data.data_ptr(),
          recvcount,
          getMpiDatatype(data.scalar_type()),
          getMpiOp(opts.reduceOp, data.scalar_type()),
          pgComm_));
    };

  auto entry = std::make_unique<WorkEntry>(
      &inputTensors[0], &outputTensors, std::move(runFunc));
  return enqueue(
    std::move(entry),
    "mpi:reduce_scatter",
    std::optional<std::vector<at::Tensor>>(inputTensors[0]));
}



c10::intrusive_ptr<Work> ProcessGroupHMPI::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    // We can use alltoall
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this](std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          c10::DeviceGuard guard(srcdata.device());
          MPI_CHECK(MPI_Alltoall(
              srcdata.data_ptr(),
              srcdata.numel() / size_,
              getMpiDatatype(srcdata.scalar_type()),
              dstdata.data_ptr(),
              dstdata.numel() / size_,
              getMpiDatatype(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // Need alltoallv
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this, inputSplitSizes, outputSplitSizes](
            std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          std::vector<int> send_lengths(size_);
          std::vector<int> recv_lengths(size_);
          std::vector<int> send_offsets(size_);
          std::vector<int> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, srcdata, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, dstdata, &recv_lengths, &recv_offsets);
          c10::DeviceGuard guard(srcdata.device());
          MPI_CHECK(MPI_Alltoallv(
              srcdata.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              getMpiDatatype(srcdata.scalar_type()),
              dstdata.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              getMpiDatatype(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  TORCH_CHECK(
      inputTensors.size() == static_cast<size_t>(size_),
      "Number of input tensors are not equal to group size");
  TORCH_CHECK(
      outputTensors.size() == static_cast<size_t>(size_),
      "Number of output tensors are not equal to group size");
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        auto srcdata = entry->src;
        auto dstdata = entry->dst;
        auto src_len = c10d::computeLengthsAndOffsets(
            srcdata, &send_lengths, &send_offsets);
        auto dst_len = c10d::computeLengthsAndOffsets(
            dstdata, &recv_lengths, &recv_offsets);
        std::vector<int64_t> send_lengthsL(
            send_lengths.begin(), send_lengths.end());
        std::vector<int64_t> recv_lengthsL(
            recv_lengths.begin(), recv_lengths.end());
        at::Tensor srcFlatData =
            at::empty({static_cast<int64_t>(src_len)}, srcdata[0].options());
        at::Tensor dstFlatData =
            at::empty({static_cast<int64_t>(dst_len)}, dstdata[0].options());
        auto srcFlatDataSplits =
            srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
        }
        c10::DeviceGuard guard1(srcdata[0].device());
        MPI_CHECK(MPI_Alltoallv(
            srcFlatData.data_ptr(),
            send_lengths.data(),
            send_offsets.data(),
            getMpiDatatype(srcdata[0].scalar_type()),
            dstFlatData.data_ptr(),
            recv_lengths.data(),
            recv_offsets.data(),
            getMpiDatatype(dstdata[0].scalar_type()),
            pgComm_));

        auto dstFlatDataSplits =
            dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_to_all",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        tensor.numel(),
        getMpiDatatype(tensor.scalar_type()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      std::vector<at::Tensor>(),
      "mpi:send",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        getMpiDatatype(tensor.scalar_type()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      tensors,
      "mpi:recv",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        getMpiDatatype(tensor.scalar_type()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      tensors,
      "mpi:recvAnySource",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::barrier(const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:barrier", std::nullopt);
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts) {

  TORCH_CHECK(
      inputTensor.is_contiguous(),
      "mpi:all_gather_into_tensor requires input tensor to be contiguous for optimal performance");
  TORCH_CHECK(
      outputTensor.is_contiguous(),
      "mpi:all_gather_into_tensor requires output tensor to be contiguous for optimal performance");

  TORCH_CHECK(
      inputTensor.scalar_type() == outputTensor.scalar_type(),
      "mpi:all_gather_into_tensor: input and output tensors must have the same type");
  TORCH_CHECK(
      outputTensor.numel() == inputTensor.numel() * size_,
      "mpi:all_gather_into_tensor: output tensor total elements must equal input tensor total elements times comm_size");

  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto out_data = (entry->dst)[0];

        c10::DeviceGuard guard(data.device());

        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            getMpiDatatype(data.scalar_type()),
            out_data.data_ptr(),
            data.numel(),
            getMpiDatatype(out_data.scalar_type()),
            pgComm_));
      };

  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));

  return enqueue(
      std::move(entry),
      "mpi:all_gather_into_tensor",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupHMPI::_reduce_scatter_base(
  at::Tensor& outputTensor,
  at::Tensor& inputTensor,
  const ReduceScatterOptions& opts) {
  TORCH_CHECK(
    outputTensor.numel() * size_ == inputTensor.numel(),
    "Reduce scatter: input tensor size must be equal to output tensor size times the world size");

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
    [opts, this](std::unique_ptr<WorkEntry>& entry) {
      auto dstdata = (entry->dst)[0];
      auto srcdata = (entry->src)[0];
      c10::DeviceGuard guard(srcdata.device());
      MPI_CHECK(MPI_Reduce_scatter_block(
        srcdata.data_ptr(),
        dstdata.data_ptr(),
        dstdata.numel(),
        getMpiDatatype(srcdata.scalar_type()),
        getMpiOp(opts.reduceOp, srcdata.scalar_type()),
        pgComm_));
    };

  auto inputTensors = std::vector<at::Tensor>({inputTensor});
  auto outputTensors = std::vector<at::Tensor>({outputTensor});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(
    std::move(entry),
    "mpi:_reduce_scatter_base",
    std::optional<std::vector<at::Tensor>>(inputTensors));
}


}