from pathlib import Path
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

def find_latest_so(build_dir: Path):
    so_candidates = sorted(
        build_dir.glob("*.so"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return so_candidates[0] if so_candidates else None

def main():
    here = Path(__file__).resolve().parent

    module = build_my_allocator(here)

    print("\nBuild summary:")
    allocator_so = find_latest_so(here / "kernels" / "my_allocator")

    if allocator_so is not None:
        target_path = here / "kernels" / allocator_so.name
        target_path.write_bytes(allocator_so.read_bytes())
        print(f"  my_allocator .so copied to: {target_path}")
    else:
        print("  my_allocator .so not found, skipping copy.")

if __name__ == "__main__":
    main()