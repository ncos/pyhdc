#!/usr/bin/python3

from distutils.core import setup, Extension
import numpy

def install(width):
    name = 'pyhdc'
    hname = 'permutations_' + str(width) + '.h'

    # define the extension module
    libpyhdc = Extension(name, sources=['pyhdc.cpp'],
                         define_macros=[('HNAME', hname)],
                         extra_compile_args=['-std=c++11'],
                         include_dirs=[numpy.get_include(), '.'])

    # run the setup
    setup(name=name,
      version='1.0',
      description='Python toolkit to work with long binary vectors',
      author='Anton Mitrokhin',
      author_email='amitrokh@umd.edu',
      url='https://github.com/ncos/pyhdc',
      ext_modules=[libpyhdc]
     )


#install(128)
#install(3200)
install(8160)
