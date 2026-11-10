from pathlib import Path
from shutil import copy2

from torch.utils.cpp_extension import load


def build_async_op(here: Path):
    csrc_dir = here / "async_rt" / "csrc"
    if not csrc_dir.exists():
        raise FileNotFoundError(f"csrc directory not found: {csrc_dir}")

    cc_files = sorted(str(p) for p in csrc_dir.glob("*.cc"))
    if not cc_files:
        raise FileNotFoundError(f"No .cc files found in: {csrc_dir}")

    build_dir = here / "kernels" / "async_op"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/2: compiling async_op ...")

    load(
        name="async_op",
        sources=cc_files,
        extra_include_paths=[str(csrc_dir), str(here / "gemm" / "bgemm")],
        extra_ldflags=[f"-L{here / 'kernels'}", "-lsmeblas", "-fopenmp"],
        extra_cflags=[
            "-O3",
            "-g",
            "-std=c++17",
            "-fopenmp",
            "-march=armv9+sve+sve2+sme+bf16",
            "-DBGEMM",
        ],
        verbose=True,
        is_python_module=False,
        is_standalone=False,
        build_directory=str(build_dir),
    )

    print("Step 1/2 finished: async_op compiled.")


def build_cpp_async_runtime(here: Path):
    runtime_src = here / "async_rt" / "cpp_async_runtime.cpp"
    if not runtime_src.exists():
        raise FileNotFoundError(f"source file not found: {runtime_src}")

    build_dir = here / "kernels" / "cpp_async_runtime"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2/2: compiling cpp_async_runtime ...")

    module = load(
        name="cpp_async_runtime",
        sources=[str(runtime_src)],
        extra_cflags=[
            "-O1",
            # "-g",
            # "-fsanitize=address,undefined",
            # "-fno-omit-frame-pointer",
            "-std=c++17",
            "-fopenmp",
            "-DASYNC_RUNTIME_ENABLE_PROFILER=1",
        ],
        verbose=True,
        is_python_module=True,
        is_standalone=False,
        build_directory=str(build_dir),
    )

    print("Step 2/2 finished: cpp_async_runtime compiled.")
    return module


def main():
    here = Path(__file__).resolve().parent
    build_async_op(here)
    build_cpp_async_runtime(here)
    for name in ("async_op", "cpp_async_runtime"):
        copy2(here / "kernels" / name / f"{name}.so", here / "kernels")


if __name__ == "__main__":
    main()
