import cython
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("imdb.ParagraphVector", ["imdb/ParagraphVector.pyx"]),
    Extension("nqe.findmax", ["nqe/findmax.pyx"]),
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[np.get_include()]
)
