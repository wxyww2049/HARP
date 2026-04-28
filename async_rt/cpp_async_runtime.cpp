#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <string>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace py = pybind11;

#ifndef ASYNC_RUNTIME_ENABLE_PROFILER
#define ASYNC_RUNTIME_ENABLE_PROFILER 0
#endif

#if ASYNC_RUNTIME_ENABLE_PROFILER
#define ASYNC_RT_PROFILER_ENABLED 1
#else
#define ASYNC_RT_PROFILER_ENABLED 0
#endif

#if ASYNC_RT_PROFILER_ENABLED
#define ASYNC_RT_RECORD_FUNCTION(name_literal) \
    RECORD_FUNCTION((name_literal), std::vector<c10::IValue>())
#else
#define ASYNC_RT_RECORD_FUNCTION(name_literal) \
    do {                                       \
    } while (0)
#endif

static inline int64_t get_rank_from_env() {
    const char* p = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!p) return 0;
    return std::strtoll(p, nullptr, 10);
}

static inline void set_thread_affinity_if_needed(int core_id) {
#ifdef __linux__
    if (core_id < 0) return;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)core_id;
#endif
}

static inline std::vector<int64_t> infer_sum_shape(
    at::IntArrayRef shape,
    const std::vector<int64_t>& dim,
    bool keepdim) {

    std::unordered_set<int64_t> dims;
    const int64_t nd = static_cast<int64_t>(shape.size());
    for (auto d : dim) {
        if (d < 0) d += nd;
        dims.insert(d);
    }

    std::vector<int64_t> out;
    out.reserve(shape.size());
    for (int64_t i = 0; i < nd; ++i) {
        if (dims.count(i)) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

struct ReadyHandle {
    std::mutex mu;
    std::condition_variable cv;
    bool done{false};

    void mark_done() {
        {
            std::lock_guard<std::mutex> lk(mu);
            done = true;
        }
        cv.notify_all();
    }

    void wait() {
        // if (done.load(std::memory_order_acquire)) return;
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this] { return done; });
    }

    bool is_ready() const {
        return done;
    }
};

struct OpCaller {
    static void call_weight_wait(const at::Tensor& x) {
        TORCH_CHECK(x.dim() == 0, "x must be a 0-D tensor");
        auto* p = x.data_ptr<uint8_t>();
        while (__atomic_load_n(p, __ATOMIC_ACQUIRE) != 1) {
            std::this_thread::yield();
        }
    }
    static void call_async_op_tanh_out(const at::Tensor& x, at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::tanh_out", "")
                      .typed<void(const at::Tensor&, at::Tensor&)>();
        op.call(x, out);
    }

    static void call_async_op_tanh_backward_out(const at::Tensor& grad_out,
                                                const at::Tensor& out,
                                                at::Tensor& grad_in) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::tanh_backward_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(grad_out, out, grad_in);
    }

    static void call_async_op_silu_out(const at::Tensor& x, at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::silu_out", "")
                      .typed<void(const at::Tensor&, at::Tensor&)>();
        op.call(x, out);
    }

    static void call_async_op_silu_backward_out(const at::Tensor& grad_out,
                                                const at::Tensor& x,
                                                at::Tensor& grad_in) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::silu_backward_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(grad_out, x, grad_in);
    }

    static void call_async_op_mul_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::mul_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(a, b, out);
    }

    static void call_async_op_add_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::add_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(a, b, out);
    }

    static void call_async_op_add_scalar(const at::Tensor& input, at::Tensor& out, double scalar) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::add_scalar_out", "")
                      .typed<void(const at::Tensor&, at::Tensor&, double)>();
        op.call(input, out, scalar);
    }

    static void call_async_op_rmsnorm_forward_out(const at::Tensor& x,
                                                  const at::Tensor& w,
                                                  at::Tensor& out,
                                                  at::Tensor& rms,
                                                  double eps) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::rmsnorm_forward_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, double)>();
        op.call(x, w, out, rms, eps);
    }

    static void call_async_op_rmsnorm_backward_out(const at::Tensor& x,
                                                   const at::Tensor& grad_out,
                                                   const at::Tensor& w,
                                                   const at::Tensor& rms,
                                                   at::Tensor& grad_in,
                                                   at::Tensor& grad_w) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::rmsnorm_backward_out", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&)>();
        op.call(x, grad_out, w, rms, grad_in, grad_w);
    }

    static void call_async_op_rope_forward_impl(const at::Tensor& x,
                                                const at::Tensor& freqs_cis,
                                                at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::rope_forward_impl", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(x, freqs_cis, out);
    }

    static void call_async_op_rope_backward_impl(const at::Tensor& grad_out,
                                                 const at::Tensor& freqs_cis,
                                                 at::Tensor& grad_in) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("async_op::rope_backward_impl", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, at::Tensor&)>();
        op.call(grad_out, freqs_cis, grad_in);
    }

    static void call_xhpops_kp_linear_forward(const at::Tensor& input,
                                              const at::Tensor& weight,
                                              c10::optional<at::Tensor> bias,
                                              at::Tensor& out) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("xhpops::kp_linear_forward", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, c10::optional<at::Tensor>, at::Tensor&)>();
        op.call(input, weight, bias, out);
    }

    static void call_xhpops_kp_linear_backward(const at::Tensor& grad_out,
                                               const at::Tensor& input,
                                               const at::Tensor& weight,
                                               at::Tensor& grad_input,
                                               at::Tensor& grad_weight,
                                               c10::optional<at::Tensor> grad_bias) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("xhpops::kp_linear_backward", "")
                      .typed<void(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, c10::optional<at::Tensor>)>();
        op.call(grad_out, input, weight, grad_input, grad_weight, grad_bias);
    }

    static void call_ring_attention_forward(const at::Tensor& q,
                                            const at::Tensor& k,
                                            const at::Tensor& v,
                                            at::Tensor out,
                                            at::Tensor m_i,
                                            at::Tensor l_i,
                                            int64_t sp_size,
                                            double scale) {
        auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow("ring_attention_ops::ring_attention_forward", "")
                  .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                      const at::Tensor&,
                      const at::Tensor&,
                      const at::Tensor&,
                      at::Tensor,
                      at::Tensor,
                      at::Tensor,
                      int64_t,
                      double)>();
        op.call(q, k, v, out, m_i, l_i, sp_size, scale);
    }

    static void call_ring_attention_backward(const at::Tensor& q,
                                             at::Tensor grad_q,
                                             const at::Tensor& k,
                                             at::Tensor grad_k,
                                             const at::Tensor& v,
                                             at::Tensor grad_v,
                                             const at::Tensor& grad_out,
                                             const at::Tensor& m_i,
                                             const at::Tensor& l_i,
                                             const at::Tensor& D,
                                             int64_t sp_size,
                                             double scale) {
        auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("ring_attention_ops::ring_attention_backward", "")
                      .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                        const at::Tensor&, at::Tensor,
                        const at::Tensor&, at::Tensor,
                        const at::Tensor&, at::Tensor,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        int64_t, double)>();
        op.call(q, grad_q, k, grad_k, v, grad_v, grad_out, m_i, l_i, D, sp_size, scale);
    }
};

class AsyncRuntime {
public:
    explicit AsyncRuntime()
        : stop_(false),
          pending_(0),
          rank_(get_rank_from_env()),
          offloader_enable_(false),
          worker_core_id_(-1) {
        start_worker();
    }

    ~AsyncRuntime() {
        shutdown();
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (stop_) return;
            stop_ = true;
        }
        cv_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

    void sync() {
        py::gil_scoped_release release;

        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return pending_.load() == 0; });

        std::lock_guard<std::mutex> elk(err_mu_);
        if (first_exception_) std::rethrow_exception(first_exception_);
    }

    void bind_worker(int64_t core_id) {
#ifdef __linux__
        std::lock_guard<std::mutex> lk(aff_mu_);
        worker_core_id_ = static_cast<int>(core_id);
#else
        (void)core_id;
#endif
    }

    void set_offloader_enable(bool flag) {
        offloader_enable_ = flag;
    }

    int64_t rank() const {
        return rank_;
    }

    void submit_wait_event(const at::Tensor& x) {
        enqueue("weight_algather_event", [=]() mutable {
            OpCaller::call_weight_wait(x);
        });
    }

    // ---------------- submit apis ----------------
    at::Tensor submit_kp_linear_forward(at::Tensor& output,
                                        const at::Tensor& input,
                                        const at::Tensor& weight,
                                        c10::optional<at::Tensor> bias) {
        enqueue("AsyncRuntime::kp_linear_forward", [=]() mutable {
            OpCaller::call_xhpops_kp_linear_forward(input, weight, bias, output);
        });
        return output;
    }

    at::Tensor submit_kp_linear_backward(at::Tensor& grad_input,
                                         const at::Tensor& grad_output_flat,
                                         const at::Tensor& inputx,
                                         const at::Tensor& weight,
                                         at::Tensor& grad_weight,
                                         c10::optional<at::Tensor> grad_bias) {
        enqueue("AsyncRuntime::kp_linear_backward", [=]() mutable {
            OpCaller::call_xhpops_kp_linear_backward(
                grad_output_flat, inputx, weight, grad_input, grad_weight, grad_bias);
        });
        return grad_input;
    }

    py::tuple submit_ring_attention_forward(at::Tensor out,
                                            const at::Tensor& local_q,
                                            const at::Tensor& local_k,
                                            const at::Tensor& local_v,
                                            at::Tensor m_i,
                                            at::Tensor l_i,
                                            int64_t sp_size,
                                            double scale) {
        enqueue("AsyncRuntime::ring_attention_forward", [=]() mutable {
            OpCaller::call_ring_attention_forward(local_q, local_k, local_v, out, m_i, l_i, sp_size, scale);
        });
        return py::make_tuple(out, m_i, l_i);
    }

    py::tuple submit_ring_attention_backward(at::Tensor& grad_q,
                                             const at::Tensor& local_q,
                                             const at::Tensor& local_k,
                                             at::Tensor& grad_k,
                                             const at::Tensor& local_v,
                                             at::Tensor& grad_v,
                                             const at::Tensor& grad_out,
                                             const at::Tensor& m_i,
                                             const at::Tensor& l_i,
                                             const at::Tensor& D,
                                             int64_t sp_size,
                                             double scale) {
        enqueue("AsyncRuntime::ring_attention_backward", [=]() mutable {
            OpCaller::call_ring_attention_backward(
                local_q, grad_q, local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale);
        });
        return py::make_tuple(grad_q, grad_k, grad_v);
    }

    at::Tensor submit_rmsnorm_forward(at::Tensor& output,
                                      const at::Tensor& inputx,
                                      const at::Tensor& weight,
                                      at::Tensor& rms,
                                      double eps) {
        enqueue("AsyncRuntime::rmsnorm_forward", [=]() mutable {
            OpCaller::call_async_op_rmsnorm_forward_out(inputx, weight, output, rms, eps);
        });
        return output;
    }

    at::Tensor submit_rmsnorm_backward(at::Tensor& grad_input,
                                       const at::Tensor& input_bf16,
                                       const at::Tensor& grad_output,
                                       const at::Tensor& weight_bf16,
                                       const at::Tensor& sve_rms,
                                       at::Tensor& grad_weight) {
        enqueue("AsyncRuntime::rmsnorm_backward", [=]() mutable {
            OpCaller::call_async_op_rmsnorm_backward_out(
                input_bf16, grad_output, weight_bf16, sve_rms, grad_input, grad_weight);
        });
        return grad_input;
    }

    at::Tensor submit_rope_forward(at::Tensor& output,
                                   const at::Tensor& inputx,
                                   const at::Tensor& freqs_cis) {
        enqueue("AsyncRuntime::rope_forward", [=]() mutable {
            OpCaller::call_async_op_rope_forward_impl(inputx, freqs_cis, output);
        });
        return output;
    }

    at::Tensor submit_rope_backward(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& freqs_cis) {
        enqueue("AsyncRuntime::rope_backward", [=]() mutable {
            OpCaller::call_async_op_rope_backward_impl(grad_output, freqs_cis, grad_input);
        });
        return grad_input;
    }

    at::Tensor submit_silu(at::Tensor& output, const at::Tensor& inputx) {
        enqueue("AsyncRuntime::silu", [=]() mutable {
            OpCaller::call_async_op_silu_out(inputx, output);
        });
        return output;
    }

    at::Tensor submit_silu_backward(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& inputx) {
        enqueue("AsyncRuntime::silu_backward", [=]() mutable {
            OpCaller::call_async_op_silu_backward_out(grad_output, inputx, grad_input);
        });
        return grad_input;
    }

    at::Tensor submit_tanh(at::Tensor& output, const at::Tensor& inputx) {
        enqueue("AsyncRuntime::tanh", [=]() mutable {
            OpCaller::call_async_op_tanh_out(inputx, output);
        });
        return output;
    }

    at::Tensor submit_tanh_backward(at::Tensor& grad_input,
                                    const at::Tensor& grad_output,
                                    const at::Tensor& output) {
        enqueue("AsyncRuntime::tanh_backward", [=]() mutable {
            OpCaller::call_async_op_tanh_backward_out(grad_output, output, grad_input);
        });
        return grad_input;
    }

    at::Tensor submit_mul_fast(at::Tensor& output,
                               const at::Tensor& input_a,
                               const at::Tensor& input_b) {
        enqueue("AsyncRuntime::mul_fast", [=]() mutable {
            OpCaller::call_async_op_mul_out(input_a, input_b, output);
        });
        return output;
    }

    at::Tensor submit_add_fast(at::Tensor& output,
                               const at::Tensor& input_a,
                               const at::Tensor& input_b) {
        enqueue("AsyncRuntime::add_fast", [=]() mutable {
            OpCaller::call_async_op_add_out(input_a, input_b, output);
        });
        return output;
    }

    at::Tensor submit_add_scalar(at::Tensor& output, const at::Tensor& input, double scalar) {
        enqueue("AsyncRuntime::add_scalar", [=]() mutable {
            OpCaller::call_async_op_add_scalar(input, output, scalar);
        });
        return output;
    }

    at::Tensor submit_mul(at::Tensor& output,
                          const at::Tensor& a,
                          const at::Tensor& b) {
        enqueue("AsyncRuntime::mul", [a, b, output]() mutable {
            at::mul_out(output, a, b);
        });
        return output;
    }

    at::Tensor submit_add(at::Tensor& output,
                          const at::Tensor& a,
                          const at::Tensor& b) {
        enqueue("AsyncRuntime::add", [=]() mutable {
            at::add_out(output, a, b);
        });
        return output;
    }

    at::Tensor submit_add_inplace(at::Tensor self, const at::Tensor& other) {
        enqueue("AsyncRuntime::add_inplace", [self, other]() mutable {
            self.add_(other);
        });
        return self;
    }

    std::shared_ptr<ReadyHandle> submit_copy_ready_only() {
        auto h = std::make_shared<ReadyHandle>();
        enqueue("AsyncRuntime::copy_ready_only", [h] {
            h->mark_done();
        });
        return h;
    }

    at::Tensor submit_sum_dim(const at::Tensor& input,
                              const std::vector<int64_t>& dim,
                              bool keepdim) {
        auto out_shape = infer_sum_shape(input.sizes(), dim, keepdim);
        auto output = at::empty(out_shape, input.options());

        enqueue("AsyncRuntime::sum_dim", [input, dim, keepdim, output]() mutable {
            at::sum_out(output, input, dim, keepdim);
        });
        return output;
    }

    at::Tensor submit_cat(const std::vector<at::Tensor>& tensors, int64_t dim) {
        TORCH_CHECK(!tensors.empty(), "cat: tensors must not be empty");

        auto shape = tensors[0].sizes().vec();
        int64_t cat_dim = 0;
        for (const auto& t : tensors) {
            cat_dim += t.size(dim);
        }
        shape[dim] = cat_dim;

        auto output = at::empty(shape, tensors[0].options());
        enqueue("AsyncRuntime::cat", [tensors, dim, output]() mutable {
            at::cat_out(output, tensors, dim);
        });
        return output;
    }

    at::Tensor submit_full(at::IntArrayRef size,
                           double fill_value,
                           at::ScalarType dtype,
                           c10::Device device) {
        auto size_vec = size.vec();
        auto output = at::empty(size_vec, at::TensorOptions().dtype(dtype).device(device));

        enqueue("AsyncRuntime::full", [size_vec, fill_value, output]() mutable {
            at::full_out(output, size_vec, fill_value);
        });

        return output;
    }

    at::Tensor submit_to_copy(const at::Tensor& input, c10::ScalarType dtype) {
        at::Tensor output = at::empty_like(input, input.options().dtype(dtype));
        enqueue("AsyncRuntime::to_copy", [=]() mutable {
            at::_to_copy_out(output, input);
        });
        return output;
    }

private:
    struct Task {
        std::function<void()> fn;
        std::string name;
#if ASYNC_RT_PROFILER_ENABLED
        at::ThreadLocalState tls;
#endif

        Task() = default;

#if ASYNC_RT_PROFILER_ENABLED
        Task(std::function<void()> fn_, std::string name_, at::ThreadLocalState tls_)
            : fn(std::move(fn_)), name(std::move(name_)), tls(std::move(tls_)) {}
#else
        Task(std::function<void()> fn_, std::string name_)
            : fn(std::move(fn_)), name(std::move(name_)) {}
#endif
    };

    void start_worker() {
        worker_ = std::thread([this] {
            int core = static_cast<int>((rank_ % 16) * 38 + 0);

#ifdef __linux__
            {
                std::lock_guard<std::mutex> lk(aff_mu_);
                if (worker_core_id_ >= 0) {
                    core = worker_core_id_;
                }
            }
#endif

            set_thread_affinity_if_needed(core);

            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                    if (stop_ && q_.empty()) return;
                    task = std::move(q_.front());
                    q_.pop_front();
                }

                try {
                    c10::InferenceMode infer_mode_guard;
#if ASYNC_RT_PROFILER_ENABLED
                    at::ThreadLocalStateGuard tls_guard(task.tls);
#endif
                    ASYNC_RT_RECORD_FUNCTION(task.name.c_str());
                    task.fn();
                } catch (...) {
                    std::lock_guard<std::mutex> lk(err_mu_);
                    if (!first_exception_) {
                        first_exception_ = std::current_exception();
                    }
                }

                if (--pending_ == 0) {
                    done_cv_.notify_all();
                }
            }
        });
    }

    void enqueue(const char* name, std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lk(err_mu_);
            if (first_exception_) std::rethrow_exception(first_exception_);
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            ++pending_;
#if ASYNC_RT_PROFILER_ENABLED
            q_.emplace_back(std::move(fn), std::string(name), at::ThreadLocalState());
#else
            q_.emplace_back(std::move(fn), std::string(name));
#endif
        }
        cv_.notify_one();
    }

private:
    std::thread worker_;
    std::deque<Task> q_;

    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_;

    std::atomic<int64_t> pending_;
    std::mutex done_mu_;
    std::condition_variable done_cv_;

    int64_t rank_;
    bool offloader_enable_;

    std::mutex aff_mu_;
    int worker_core_id_;

    std::mutex err_mu_;
    std::exception_ptr first_exception_;
};

static std::shared_ptr<AsyncRuntime> g_runtime;

static std::shared_ptr<AsyncRuntime> get_runtime() {
    if (!g_runtime) {
        g_runtime = std::make_shared<AsyncRuntime>();
    }
    return g_runtime;
}

PYBIND11_MODULE(cpp_async_runtime, m) {
    py::class_<AsyncRuntime, std::shared_ptr<AsyncRuntime>>(m, "AsyncRuntime")
        .def(py::init<>())
        .def("sync", &AsyncRuntime::sync)
        .def("shutdown", &AsyncRuntime::shutdown)
        .def("bind_worker", &AsyncRuntime::bind_worker, py::arg("core_id"))
        .def("set_offloader_enable", &AsyncRuntime::set_offloader_enable)
        .def_property_readonly("rank", &AsyncRuntime::rank)

        .def("submit_kp_linear_forward", &AsyncRuntime::submit_kp_linear_forward)
        .def("submit_kp_linear_backward", &AsyncRuntime::submit_kp_linear_backward)
        .def("submit_ring_attention_forward", &AsyncRuntime::submit_ring_attention_forward)
        .def("submit_ring_attention_backward", &AsyncRuntime::submit_ring_attention_backward)
        .def("submit_rmsnorm_forward", &AsyncRuntime::submit_rmsnorm_forward)
        .def("submit_rmsnorm_backward", &AsyncRuntime::submit_rmsnorm_backward)
        .def("submit_rope_forward", &AsyncRuntime::submit_rope_forward)
        .def("submit_rope_backward", &AsyncRuntime::submit_rope_backward)
        .def("submit_silu", &AsyncRuntime::submit_silu)
        .def("submit_silu_backward", &AsyncRuntime::submit_silu_backward)
        .def("submit_tanh", &AsyncRuntime::submit_tanh)
        .def("submit_tanh_backward", &AsyncRuntime::submit_tanh_backward)
        .def("submit_mul_fast", &AsyncRuntime::submit_mul_fast)
        .def("submit_add_fast", &AsyncRuntime::submit_add_fast)
        .def("submit_add_scalar", &AsyncRuntime::submit_add_scalar)
        .def("submit_mul", &AsyncRuntime::submit_mul)
        .def("submit_add", &AsyncRuntime::submit_add)
        .def("submit_add_inplace", &AsyncRuntime::submit_add_inplace)
        .def("submit_copy_ready_only", &AsyncRuntime::submit_copy_ready_only)
        .def("submit_sum_dim", &AsyncRuntime::submit_sum_dim)
        .def("submit_cat", &AsyncRuntime::submit_cat)
        .def("submit_full", &AsyncRuntime::submit_full)
        .def("submit_wait_event", &AsyncRuntime::submit_wait_event)
        .def("submit_to_copy", &AsyncRuntime::submit_to_copy);

    py::class_<ReadyHandle, std::shared_ptr<ReadyHandle>>(m, "ReadyHandle")
        .def(py::init<>())
        .def("mark_done", &ReadyHandle::mark_done)
        .def("wait", &ReadyHandle::wait)
        .def("is_ready", &ReadyHandle::is_ready);

    m.def("get_runtime", &get_runtime);
}