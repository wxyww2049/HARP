from pathlib import Path
from shutil import copy2
from torch.utils.cpp_extension import load

def build_my_allocator(here: Path):
    src_file = here / "custom_allocator.cpp"
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    build_dir = here / "kernels" / "my_allocator"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling my_allocator ...")

    module = load(
        name="my_allocator",
        sources=[str(src_file)],
        extra_cflags=[
            "-O3",
            "-g",
            "-std=c++17",
            "-fopenmp",
            "-march=armv9+sve+sve2+sme+bf16",
        ],
        extra_ldflags=[
            "-lmemkind",
        ],
        verbose=True,
        is_python_module=True,
        is_standalone=False,
        build_directory=str(build_dir),
    )

    print("Compilation finished.")
    return module


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build_my_allocator(here)
    target = here.parent / "kernels"
    target.mkdir(exist_ok=True)
    copy2(here / "kernels" / "my_allocator" / "my_allocator.so", target)
