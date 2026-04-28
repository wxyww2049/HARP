import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

GCC_ROOT = "/work_ssd/software/HPCKit/25.1.0.SPC001/compiler/gcc"

extra_compile_args = [
    '-O3',
    # '-g',
    '-march=armv9-a+sme',
    '--rtlib=compiler-rt',
    f'--gcc-toolchain={GCC_ROOT}',
    '-Wall',
    f'-isystem{GCC_ROOT}/include/c++/12.3.1',
    f'-isystem{GCC_ROOT}/include/c++/12.3.1/aarch64-linux-gnu/',
]

extra_link_args = [
    '--rtlib=compiler-rt',
    f'--gcc-toolchain={GCC_ROOT}',
]

setup(
    name='accumulate_sp',
    ext_modules=[
        CppExtension(
            name='accumulate_sp',
            sources=['accumulate.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=['numa'], 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(parallel=True) 
    }
)