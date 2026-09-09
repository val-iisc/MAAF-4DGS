from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="parzen_filter",
    ext_modules=[
        CUDAExtension(
            name="parzen_filter._C",
            sources=["parzen_filter_kernel.cu", "ext.cpp"],
            extra_compile_args={
                "nvcc": ["-O3", "--use_fast_math"],
                "cxx":  ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
