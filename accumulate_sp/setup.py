from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = [
    '-O3',
    # '-g',
    '-march=armv9-a+sme',
    '--rtlib=compiler-rt',
    '-Wall',
]

extra_link_args = [
    '--rtlib=compiler-rt',
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