#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>


#include <pybind11/chrono.h>
#include <torch/python.h>


#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <c10/util/CallOnce.h>

#include <mpi.h>
#include <arm_sve.h>
namespace py = pybind11;
namespace c10d {

constexpr const char* MPI_BACKEND_NAME = "hmpi";




namespace bf16_mpi {
    
    static inline void sve_add(bfloat16_t * A, bfloat16_t *B, int len){
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

    inline void mpi_bf16_sum_op(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype){
        bfloat16_t* input = reinterpret_cast<bfloat16_t*>(invec);
        bfloat16_t* output = reinterpret_cast<bfloat16_t*>(inoutvec);
        sve_add(output, input, *len);
    }
    void initBF16MpiSupport();
    void finalizeBF16MpiSupport();
    MPI_Datatype getBF16MpiDatatype();
    MPI_Op getBF16MpiSumOp();
}


struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
};


class TORCH_API ProcessGroupHMPI : public Backend {
 public:
    struct Options : public Backend::Options {
      explicit Options(int thread_id_offset = 0)
	  : Backend::Options(MPI_BACKEND_NAME, std::chrono::milliseconds(1800000)), thread_id_offset(thread_id_offset) {}
      int thread_id_offset;
    };
  class WorkHMPI : public Work {
   public:



    explicit WorkHMPI(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;


   protected:
    friend class ProcessGroupHMPI;

   private:
    void finishWorkHMPI();
    void finishWorkHMPIError(const std::exception_ptr& eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class AsyncWork : public Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    std::vector<at::Tensor> result() override;

   protected:
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };
  


  // Constructor will spawn up the worker thread loop
  //explicit ProcessGroupHMPI(int rank, int size, MPI_Comm pgComm);
  explicit ProcessGroupHMPI(int rank, int size, void* pgComm,int thread_id_offset);

  ~ProcessGroupHMPI() override;

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  const std::string getBackendName() const override {
    return std::string(MPI_BACKEND_NAME);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  static c10::intrusive_ptr<ProcessGroupHMPI> createProcessGroupHMPI(
      const c10d::DistributedBackendOptions& opts, const c10::intrusive_ptr<Options>& pg_options );

  MPI_Comm getMPIComm() const{
    return pgComm_;
  }




 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkHMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<Work> enqueue(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();
  static c10::once_flag onceFlagInitMPI;

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
};

}