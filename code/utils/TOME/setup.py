from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, check_compiler_abi_compatibility

check_compiler_abi_compatibility("gcc")
check_compiler_abi_compatibility("g++")

setup(
    name='TOME',
    packages=['TOME'],
    ext_modules=[
        CUDAExtension(
            '_implementation', 
            [
                'TOME_cuda.cpp',
                'TOME_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-g', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [],#['-O2']
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)