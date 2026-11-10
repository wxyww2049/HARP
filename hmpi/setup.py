import os
from pathlib import Path
from shutil import copy2
from torch.utils.cpp_extension import load

def build_process_group_hmpi(here: Path):
    src_file = here / "ProcessGroupHMPI.cpp"
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    build_dir = here / "kernels" / "ProcessGroupHMPI"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling ProcessGroupHMPI ...")

    hmpi_include = str(Path(os.environ["HMPI_ROOT"]) / "include")
    hmpi_lib = str(Path(os.environ["HMPI_ROOT"]) / "lib")

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


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build_process_group_hmpi(here)
    target = here.parent / "kernels"
    target.mkdir(exist_ok=True)
    copy2(here / "kernels" / "ProcessGroupHMPI" / "ProcessGroupHMPI.so", target)
