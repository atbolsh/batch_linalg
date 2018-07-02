from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='batch_linalg_cuda',
    version='0.4',
    ext_modules=[
        CUDAExtension('batch_linalg_cuda', [
            'connective.cpp',
#            'backend.cpp',
            'batchSolverDouble.cpp',
            'batchSolverSingle.cpp'
#            'solver_demo.cu'
        ],
        extra_link_args=[
#            "-l", "cufft",
            "-l", "cusolver",
            "-l", "cublas",
            "-l", "cusparse"
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

