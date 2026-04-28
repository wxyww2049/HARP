from pathlib import Path

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
        extra_include_paths=[str(csrc_dir)],
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


def find_latest_so(build_dir: Path):
    so_candidates = sorted(
        build_dir.glob("*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return so_candidates[0] if so_candidates else None


def main():
    here = Path(__file__).resolve().parent

    build_async_op(here)
    build_cpp_async_runtime(here)

    async_op_so = find_latest_so(here / "kernels" / "async_op")
    runtime_so = find_latest_so(here / "kernels" / "cpp_async_runtime")

    print("\nBuild summary:")

    # copy async_op_so to here / "kernels"
    if async_op_so is not None:
        target_path = here / "kernels" / async_op_so.name
        target_path.write_bytes(async_op_so.read_bytes())
        print(f"  async_op .so copied to: {target_path}")
    else:
        print("  async_op .so not found, skipping copy.")

    # copy cpp_async_runtime to here / "kernels"
    if runtime_so is not None:
        target_path = here / "kernels" / runtime_so.name
        target_path.write_bytes(runtime_so.read_bytes())
        print(f"  cpp_async_runtime .so copied to: {target_path}")
    else:
        print("  cpp_async_runtime .so not found, skipping copy.")

if __name__ == "__main__":
    main()