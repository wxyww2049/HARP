from pathlib import Path
from torch.utils.cpp_extension import load

def build_process_group_hmpi(here: Path):
    src_file = here / "ProcessGroupHMPI.cpp"
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    build_dir = here / "kernels" / "ProcessGroupHMPI"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling ProcessGroupHMPI ...")

    hmpi_include = "/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/include"
    hmpi_lib = "/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/lib"

    module = load(
        name="ProcessGroupHMPI",
        sources=[str(src_file)],
        
        extra_include_paths=[hmpi_include], 
        
        extra_cflags=[
            "-O3",
            "-g",
            "-std=c++17",
            "-fopenmp",
            "-march=armv9+sve+sve2+sme+bf16",
        ],
        
        extra_ldflags=[
            f"-L{hmpi_lib}",
            "-lmpi",
            f"-Wl,-rpath,{hmpi_lib}"
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

    module = build_process_group_hmpi(here)

    print("\nBuild summary:")
    process_group_so = find_latest_so(here / "kernels" / "ProcessGroupHMPI")

    if process_group_so is not None:
        target_path = here / "kernels" / process_group_so.name
        target_path.write_bytes(process_group_so.read_bytes())
        print(f"  ProcessGroupHMPI .so copied to: {target_path}")
    else:
        print("  ProcessGroupHMPI .so not found, skipping copy.")

if __name__ == "__main__":
    main()