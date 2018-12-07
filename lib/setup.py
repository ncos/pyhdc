#!/usr/bin/python3

from distutils.core import setup, Extension
import numpy

# define the extension module
libpyhdc = Extension('pyhdc', sources=['pyhdc.cpp'],
                     extra_compile_args=['-std=c++11'],
                     include_dirs=[numpy.get_include()])

# run the setup
setup(name='pyhdc',
      version='1.0',
      description='Python toolkit to work with long binary vectors',
      author='Anton Mitrokhin',
      author_email='amitrokh@umd.edu',
      url='https://github.com/ncos/pyhdc',
      ext_modules=[libpyhdc]
     )
