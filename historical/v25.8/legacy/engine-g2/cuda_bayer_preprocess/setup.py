# Optional build script. The main project normally uses torch.utils.cpp_extension.load
# from cuda_bayer_preprocess/preprocess.py and builds on first import.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_bayer_preprocess',
    ext_modules=[
        CUDAExtension(
            'bayer_preprocess_cuda_ext',
            ['bayer_preprocess.cpp', 'bayer_preprocess_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
