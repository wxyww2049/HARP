#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace py = pybind11;

namespace {

void set_thread_affinity_if_needed(const std::optional<int>& core_id) {
#ifdef __linux__
    if (!core_id.has_value()) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(*core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)core_id;
#endif
}

void check_state_tensor(const at::Tensor& state) {
    TORCH_CHECK(state.device().is_cpu(), "state tensor must live on CPU");
    TORCH_CHECK(state.scalar_type() == at::kByte, "state tensor must be torch.uint8");
    TORCH_CHECK(state.numel() == 1, "state tensor must be 0-D or 1-element");
}

void wait_state_ready(const at::Tensor& state, int64_t poll_interval_us) {
    check_state_tensor(state);
    auto* ptr = reinterpret_cast<volatile uint8_t*>(state.data_ptr<uint8_t>());

    py::gil_scoped_release release;
    while (*ptr != 1) {
        if (poll_interval_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(poll_interval_us));
        } else {
            std::this_thread::yield();
        }
    }
}

struct ParamTask {
    at::Tensor param;
    at::Tensor chunk_part;
    at::Tensor state;
};

}  // namespace

class AsyncParamAllGatherManager {
public:
    AsyncParamAllGatherManager(py::object rs_group, int64_t rs_size, std::optional<int> core_id = std::nullopt)
        : rs_group_(std::move(rs_group)),
          rs_size_(rs_size),
          core_id_(core_id),
          worker_(&AsyncParamAllGatherManager::worker_loop, this) {}

    ~AsyncParamAllGatherManager() {
        if (worker_.joinable()) {
            try {
                shutdown();
            } catch (...) {
            }
        }
    }

    void push(py::object p_obj) {
        ParamTask task = extract_task(std::move(p_obj));
        {
            std::lock_guard<std::mutex> lock(mu_);
            throw_if_failed_locked();
            pending_ += 1;
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    void wait_all() {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [&] { return pending_ == 0 || failed_; });
        throw_if_failed_locked();
    }

    void shutdown() {
        {
            py::gil_scoped_release release;
            {
                std::lock_guard<std::mutex> lock(mu_);
                shutdown_requested_ = true;
            }
            cv_.notify_all();
            if (worker_.joinable()) {
                worker_.join();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            throw_if_failed_locked();
        }
    }

private:
    ParamTask extract_task(py::object p_obj) {
        if (!py::hasattr(p_obj, "chunk_part")) {
            throw std::runtime_error("queued param does not have chunk_part");
        }
        if (!py::hasattr(p_obj, "state")) {
            throw std::runtime_error("queued param does not have state");
        }

        return ParamTask{
            p_obj.cast<at::Tensor>(),
            p_obj.attr("chunk_part").cast<at::Tensor>(),
            p_obj.attr("state").cast<at::Tensor>(),
        };
    }

    void worker_loop() {
        set_thread_affinity_if_needed(core_id_);

        while (true) {
            ParamTask task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return shutdown_requested_ || !tasks_.empty(); });
                if (tasks_.empty()) {
                    if (shutdown_requested_) {
                        break;
                    }
                    continue;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            try {
                all_gather_param(task);
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    pending_ -= 1;
                    if (pending_ == 0) {
                        done_cv_.notify_all();
                    }
                }
            } catch (const py::error_already_set& e) {
                record_failure(e.what());
                break;
            } catch (const std::exception& e) {
                record_failure(e.what());
                break;
            }
        }

        std::lock_guard<std::mutex> lock(mu_);
        done_cv_.notify_all();
    }

    void all_gather_param(const ParamTask& task) {
        const at::Tensor& p = task.param;
        const at::Tensor& chunk_part = task.chunk_part;
        const at::Tensor& state = task.state;

        TORCH_CHECK(p.dim() > 0, "param shape ", p.sizes(), " cannot be evenly split by rs_size=", rs_size_);
        TORCH_CHECK(p.size(0) % rs_size_ == 0, "param shape ", p.sizes(), " cannot be evenly split by rs_size=", rs_size_);

        const int64_t expected_rows = p.size(0) / rs_size_;
        TORCH_CHECK(
            chunk_part.size(0) == expected_rows,
            "chunk_part.shape[0]=", chunk_part.size(0),
            ", expected ", expected_rows,
            ", param shape=", p.sizes()
        );

        at::Tensor full_param = p.detach();
        at::Tensor local_chunk = chunk_part.detach();

        py::gil_scoped_acquire acquire;
        py::module_ dist = py::module_::import("torch.distributed");

        py::list gathered;
        for (int64_t i = 0; i < rs_size_; ++i) {
            gathered.append(full_param.narrow(0, i * expected_rows, expected_rows));
        }

        dist.attr("barrier")(py::arg("group") = rs_group_);
        dist.attr("all_gather")(gathered, local_chunk, py::arg("group") = rs_group_);
        state.fill_(1);
    }

    void record_failure(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mu_);
        failed_ = true;
        error_message_ = msg;
        done_cv_.notify_all();
    }

    void throw_if_failed_locked() const {
        if (failed_) {
            throw std::runtime_error(error_message_);
        }
    }

    py::object rs_group_;
    int64_t rs_size_;
    std::optional<int> core_id_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::queue<ParamTask> tasks_;
    size_t pending_ = 0;
    bool shutdown_requested_ = false;
    bool failed_ = false;
    std::string error_message_;
    std::thread worker_;
};

class ParamReadyHook {
public:
    ParamReadyHook(std::string module_name, int64_t poll_interval_us)
        : module_name_(std::move(module_name)), poll_interval_us_(poll_interval_us) {}

    py::object operator()(py::object module, py::object inputs) const {
        std::vector<at::Tensor> states;

        for (auto item : module.attr("named_parameters")(py::arg("recurse") = false)) {
            py::tuple named_param = item.cast<py::tuple>();
            py::object param_obj = named_param[1];
            if (py::hasattr(param_obj, "state") && py::hasattr(param_obj, "chunk_part")) {
                states.push_back(param_obj.attr("state").cast<at::Tensor>());
            }
        }

        for (const auto& state : states) {
            wait_state_ready(state, poll_interval_us_);
        }

        return py::none();
    }

private:
    std::string module_name_;
    int64_t poll_interval_us_;
};

void register_param_wait_hooks(py::object model, int64_t poll_interval_us = 10) {
    for (auto item : model.attr("named_modules")()) {
        py::tuple named_module = item.cast<py::tuple>();
        std::string module_name = named_module[0].cast<std::string>();
        py::object module = named_module[1];

        py::list params_list = py::list(module.attr("parameters")(py::arg("recurse") = false));
        if (py::len(params_list) == 0) {
            continue;
        }

        module.attr("register_forward_pre_hook")(ParamReadyHook(module_name, poll_interval_us));
    }
}

PYBIND11_MODULE(weight_sync_cpp, m) {
    m.doc() = "C++ helpers for async parameter all-gather and param-ready hooks";

    py::class_<AsyncParamAllGatherManager>(m, "AsyncParamAllGatherManager")
        .def(py::init<py::object, int64_t, std::optional<int>>(),
             py::arg("rs_group"),
             py::arg("rs_size"),
             py::arg("core_id") = std::nullopt)
        .def("push", &AsyncParamAllGatherManager::push)
        .def("wait_all", &AsyncParamAllGatherManager::wait_all)
        .def("shutdown", &AsyncParamAllGatherManager::shutdown);

    py::class_<ParamReadyHook>(m, "ParamReadyHook")
        .def(py::init<std::string, int64_t>(),
             py::arg("module_name"),
             py::arg("poll_interval_us") = 10)
        .def("__call__", &ParamReadyHook::operator());

    m.def("wait_state_ready", &wait_state_ready, py::arg("state"), py::arg("poll_interval_us") = 10);
    m.def("register_param_wait_hooks", &register_param_wait_hooks, py::arg("model"), py::arg("poll_interval_us") = 10);
}
