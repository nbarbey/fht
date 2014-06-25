#!/usr/bin/env python
from numpy.distutils.core import Extension, setup
from numpy import get_include
from os.path import join
import os
import sys

# possible types
types = ("int", "long", "float", "double")

# generate sources
pth = os.getcwd()
template_file = join(os.getcwd(), "fht", "C_fht.template.c")
f = open(template_file, "r")
txt = f.read()
f.close()
for t in types:
    d = {"ctype":t}
    filled_txt = txt % d
    source = join(os.getcwd(), "fht", "C_fht_%(ctype)s.c" % d)
    f = open(source, "w")
    f.write(filled_txt)
    f.close()

# distutils

sys.path.extend('config_fc --fcompiler=gnu95 --f90flags=-fopenmp --f90exec=/usr/bin/gfortran '.split())
setup(name='fht',
      version='1.0.1',
      description='Fast Hadamard Transform',
      author='Nicolas Barbey',
      author_email='nicolas.a.barbey@gmail.com',
      requires=['numpy (>1.3.0)'],
      packages=['fht'],
      ext_modules=[Extension('fht._C_fht_%(ctype)s' % {"ctype":t},
                             [join('fht', 'C_fht_%(ctype)s.c') % {"ctype":t}],
                             include_dirs=[join(get_include(), 'numpy')],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],)
                   for t in types],
      data_files=[('tests', ['test_fht'])]
      )
