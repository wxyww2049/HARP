# HARP

Hierarchical Arm Runtime and Parallelization Framework 是面向深算 AArch64 计算节点的 BF16 分布式训练框架。训练入口是 `train.py`。仓库包含原生算子源码、模型与调度器配置，以及 8 个 NPZ 样本。默认训练从配置初始化权重，拿到源码后可以离线编译并训练。

## 在深算上编译

将仓库放在登录节点与计算节点都能访问的共享目录中。从仓库根目录提交一次编译作业：

```bash
dsub -nn 1 -rpn 1 -R "cpu=16" -n HARP-build -q q_grapes \
    -oo "$PWD/build-%J.out" -eo "$PWD/build-%J.err" "bash build_all.sh"
```

作业完成后，标准输出末尾会出现 `Build completed successfully.`。已在分配到的计算节点上时，也可以直接执行 `bash build_all.sh`。构建产物位于 `kernels/`。

| 依赖 | 深算上的默认位置或来源 |
| --- | --- |
| BiSheng Clang、HMPI 和 SME KBLAS | `/work_ssd/software/HPCKit/25.2.1` |
| Python、PyTorch 2.5.1、NumPy、setuptools 和 Ninja | `$HOME/miniconda3/envs/DiT_DNN` |
| PyTorch 链接的旧版 KBLAS | HPCKit 25.1.0.SPC001 的 SME 多线程库，构建时在 `kernels/compat/` 创建兼容库名 |
| OpenMP、libnuma、memkind、make、numactl 和 taskset | 工具链及计算节点的系统安装 |

路径集中在 `run/env.sh`。安装位置有差异时，可设置 `HARP_PYTHON_ENV`、`HPCKIT_ROOT`、`HMPI_ROOT`、`KBLAS_INCLUDE`、`KBLAS_LIBDIR` 和 `TORCH_KBLAS_LIB`。构建还接受 `PYTHON_BIN`、`MAKE_BIN` 和 `MAX_JOBS`。更换 Python 或工具链后，请在新的源码目录中构建。

## 运行训练

在同一个仓库根目录提交单节点验证作业：

```bash
bash run/run.sh 1 --max_train_steps 2
```

第一个参数是节点数，后续参数传给 `train.py`。默认队列是 `q_grapes`，可通过 `HARP_QUEUE` 覆盖。每个节点按 608 个核、16 个 MPI 进程布局，每个序列并行组 4 个进程，每进程批大小为 1。启动时优先使用该进程可访问的 HBM，若节点仅开放 DDR，则使用本地 DDR。也可设置 `HARP_USE_HBM=0` 选择 DDR。默认使用 `z_image/` 的模型配置和 `data/index.json`，共训练 1000 步。

```bash
djob JOB_ID
bash run/watch.sh JOB_ID
```

用调度器返回的作业号替换 `JOB_ID`。标准输出和错误分别写入 `e2e-logs/out-JOB_ID` 与 `e2e-logs/err-JOB_ID`。训练参数、loss 和学习率写入 `results/` 下对应运行的目录。若已有 Donau 资源分配和 `CCS_ALLOC_FILE`，可在作业内直接执行 `bash run/sub.sh 1 --max_train_steps 2`。

自定义输入可通过 `--pretrained_model_name_or_path` 指定包含 `transformer/config.json` 和 `scheduler/scheduler_config.json` 的目录，通过 `--data_root` 指定 NPZ 路径列表的 JSON 文件。相对路径以该 JSON 所在目录为准。样本包含 `norm_metadata`、`prompt_embeds`、`latent_image`、`cond` 四个数组。这里的模型路径用于加载配置，默认流程不读取预训练权重。

## 源码入口

- `gemm/bgemm/` 实现按矩阵形状分派的 SME GEMM。`async_rt/csrc/linear.cc` 将它接入 PyTorch，其他形状使用 PyTorch 矩阵乘。
- `gemm/ring_attention/` 实现序列并行的环形注意力。
- `async_rt/`、`accumulate_sp/`、`hmpi/` 和 `mem/` 分别提供异步算子与运行时、梯度累加、通信后端和内存分配器。
- `src/` 和 `managers/` 提供模型、数据读取与并行训练逻辑。
