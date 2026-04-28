# HARP

Hierarchical Arm Runtime and Parallelization Framework

This repository contains the custom training stack and native kernels used by `train.py`. The project is designed for ARMv9 platforms and mixes Python code with several C++/PyTorch extensions, GEMM backends, ring-attention operators, and async runtime components. Most compiled artifacts are collected into the top-level `kernels/` directory so the training entrypoint can load them from one place.

## Build environment

The build chain is intended for a Linux AArch64 environment targeting ARMv9, not for Windows directly. Several build scripts and Makefiles currently depend on:

- `bash`
- `python`
- `make`
- `clang` or `clang++`
- `mpicxx`
- PyTorch `2.5.1` or newer with C++ extension support
- OpenMP, `numa`, and `memkind`
- HMPI and KBLAS headers and libraries available at the paths referenced in the existing scripts

The compile flags in this repository target ARMv9 features such as `sve`, `sve2`, `sme`, and `bf16`, so the target machine needs to match that toolchain and hardware setup.

## Checkpoint Preparation

Before building or launching the training stack, download the required checkpoints and datasets from the repository root:

```bash
modelscope download --dataset wxyww2049/ckpts --local_dir ./
modelscope download --dataset wxyww2049/datasets --local_dir ./
```

## One-command build

From the repository root, run:

```bash
bash build_all.sh
```

This script builds components in the following order:

1. `accumulate_sp/compile.sh`
2. `hmpi/compile.sh`
3. `mem/compile.sh`
4. `gemm/bgemm` via `make`
5. `gemm/ring_attention` via `make`
6. `build.py`

After a successful build, the required shared libraries are copied or generated under `kernels/`.

If you need to override the tool binaries, you can do:

```bash
PYTHON_BIN=python3 MAKE_BIN=make bash build_all.sh
```

## Main outputs

The top-level `kernels/` directory is the runtime-facing artifact directory. It is expected to contain libraries such as:

- `accumulate_sp*.so`
- `ProcessGroupHMPI*.so`
- `my_allocator*.so`
- `libsmeblas.so`
- `libring_attention_ops.so`
- `async_op*.so`
- `cpp_async_runtime*.so`

Some components also keep intermediate build outputs in their own local build directories.

## Project layout

- `train.py`: main training entrypoint
- `build.py`: builds the CPU async computation management components and small operator extensions
- `accumulate_sp/`: memory offloading and communication framework component
- `hmpi/`: custom MPI communication backend
- `mem/`: custom memory pool and memory management component
- `gemm/`: matrix multiplication operators and related kernels
- `async_rt/`: CPU async computation management components, including small operators such as RMSNorm and SiLU
- `managers/`: Python-side framework code
- `run/`: scheduler and MPI launch scripts
- `src/`: model, dataset, and utility Python code
- `kernels/`: collected shared libraries used at runtime

## Running training

Before launching training, make sure the native libraries have been built and copied into `kernels/`. The training stack relies on those shared libraries at runtime.

The launch scripts are under `run/`:

- `run/run.sh N`: launch training through the Donau scheduling system
- `run/sub.sh N`: launch training directly through MPI

Here, `N` is the number of nodes to start.

Typical Donau-based flow:

```bash
bash build_all.sh
bash run/run.sh N
```

If you want to bypass the scheduler and launch directly with MPI:

```bash
bash build_all.sh
bash run/sub.sh N
```

In the current setup, `run/run.sh` submits a Donau job and then calls `run/watch.sh`, while `run/sub.sh` generates the rankfile and launches `run/train.sh` through `mpirun`.
