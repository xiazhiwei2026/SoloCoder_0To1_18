import os
import sys

try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext

import pybind11


ext_modules = [
    Extension(
        'morphology_cpp',
        sources=[
            'morphology/morphology.cpp',
            'morphology/bindings.cpp'
        ],
        include_dirs=[
            pybind11.get_include(),
            'morphology'
        ],
        language='c++',
        extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
    ),
]


setup(
    name='morphology_cpp',
    version='1.0.0',
    ext_modules=ext_modules,
    zip_safe=False,
)
