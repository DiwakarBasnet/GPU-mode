from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="scharr_cuda", #importable python name
    ext_modules=[
        CUDAExtension("scharr_cuda", ["scharr.cpp", "scharr_kernel.cu"]),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)