from __future__ import absolute_import

from fht._fht import *

def test(level=1, verbosity=1):
    from numpy.testing import Tester
    return Tester().test(level, verbosity)
