import numpy as np
from distutils.core import setup
from Cython.Distutils import Extension, build_ext

from numpy.distutils import system_info

# this is probably not very portable!
try:
    # first look for OpenBLAS libs/include
    info = system_info.openblas_info()
except AttributeError:
    # old versions of numpy don't have this config field - try and look for
    # standard LAPACK info
    info = system_info.lapack_info()

lapack_libs = ['-l' + name for name in info.get_libraries()]
include_dirs = info.get_include_dirs() + [np.get_include()]
library_dirs = info.get_lib_dirs()

# NB: use dotted relative module names here!
# -----------------------------------------------------------------------------

# extra compiler flags
CF = ['-O3']

gte = Extension(
    name="cymodules._fast_gte",
    sources=["cymodules/_fast_gte.pyx"],
    extra_compile_args=CF + ['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

kern = Extension(
    name="cymodules._fast_kernels",
    sources=["cymodules/_fast_kernels.pyx"],
    extra_compile_args=CF + ['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

spike = Extension(
    name="cymodules._spiketrains",
    sources=["cymodules/_spiketrains.pyx"],
    extra_compile_args=CF + ['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

solve = Extension(
    name="cymodules._fnndeconv",
    sources=["cymodules/_fnndeconv.pyx"],
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    extra_compile_args=CF + ['-fopenmp'],
    extra_link_args=['-fopenmp'] + lapack_libs,
    runtime_library_dirs=library_dirs
)

ce = Extension(
    name="cymodules._causal_entropy",
    sources=["cymodules/_causal_entropy.pyx"],
    extra_compile_args=CF + ['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

test_flann = Extension(
    name="cymodules._test_flann",
    sources=["cymodules/_test_flann.pyx"],
    extra_compile_args=CF + ['-fopenmp'],
    include_dirs=['/opt/FLANN/include/flann'],
    library_dirs=['/opt/FLANN/lib'],
    extra_link_args=['-fopenmp'],
)

# -----------------------------------------------------------------------------

# NB: nonstandard position of distribution root (should prob change this)
setup(
    name='konnectomics',
    package_dir={'konnectomics': ''},
    cmdclass={'build_ext': build_ext},
    ext_modules=[gte, kern, spike, solve, ce, test_flann],
)
