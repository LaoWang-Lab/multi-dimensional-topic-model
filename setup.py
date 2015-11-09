from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
        name='_cymlda',
        sources=['_cymlda.pyx'],
        include_dirs=[np.get_include()]
        )

setup(
    ext_modules=cythonize(ext),
    include_dirs = [np.get_include()]
)
