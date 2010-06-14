#!/usr/bin/env python
from numpy.distutils.core import Extension, setup
from numpy import get_include
from os.path import join
import sys
sys.path.extend('config_fc --fcompiler=gnu95 --f90flags=-fopenmp --f90exec=/usr/bin/gfortran '.split())
setup(name='fht',
      version='1.0',
      description='Fast Hadamard Transform',
      author='Nicolas Barbey',
      author_email='nicolas.a.barbey@gmail.com',
      install_requires = ['numpy>=1.3.0'],
      packages=['fht'],
      ext_modules=[Extension('fht._C_fht', 
                             [join('fht', 'C_fht.c')],
                             include_dirs=[join(get_include(), 'numpy')],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],),
                   ]
      )
