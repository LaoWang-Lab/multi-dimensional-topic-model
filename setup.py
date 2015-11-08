from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension(name='_cymlda', sources=['_cymlda.pyx'])

setup(ext_modules=cythonize(ext))
