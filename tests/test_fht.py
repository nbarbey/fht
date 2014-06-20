from numpy.testing import *
import numpy as np
import fht

class FhtTestCase:
    def test_size_2_1(self):
        a = [np.sqrt(2), 0]
        b = [1., 1.]
        self.do(a, b)

    def test_size_2_2(self):
        a = [0, np.sqrt(2)]
        b = [1., -1.]
        self.do(a, b)

    def test_size_4_1(self):
        a = [2, 0, 0, 0]
        b = [1, 1, 1, 1]
        self.do(a, b)

    def test_size_4_2(self):
        a = [0, 2, 0, 0]
        b = [1, -1, 1, -1]
        self.do(a, b)

    def test_size_4_3(self):
        a = [0, 0, 2, 0]
        b = [1, 1, -1, -1]
        self.do(a, b)

    def test_size_4_4(self):
        a = [0, 0, 0, 2]
        b = [1, -1, -1, 1]
        self.do(a, b)

    def test_2d_4_1(self):
        a = [[2, 0], [0, 0]]
        b = [[1, 1], [1, 1]]
        self.do(a, b)

    def test_2d_4_2(self):
        a = [[0, 2], [0, 0]]
        b = [[1, -1], [1, -1]]
        self.do(a, b)

    def test_2d_4_3(self):
        a = [[0, 0], [2, 0]]
        b = [[1, 1], [-1, -1]]
        self.do(a, b)

    def test_2d_4_4(self):
        a = [[0, 0], [0, 2]]
        b = [[1, -1], [-1, 1]]
        self.do(a, b)

class TestFht(FhtTestCase):
    def do(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        assert_array_equal(fht.fht(a), b)

def check_random_inverse(shape):
    a = np.random.rand(*shape)
    assert_array_almost_equal(fht.fht(fht.fht(a)), a)

def test_random_inverse():
    for shape in (2, 2), (4, 4), (8, 4), (128, 128), (2, 2, 2):
        yield check_random_inverse, shape

@raises(ValueError)
def check_power_of_two(i):
    fht.fht(np.ones(i))

def test_not_power_of_two():
    for i in [1, 3, 5, 6, 7, 9, 11, 127, 100000]:
        yield check_power_of_two, i

def test_is_power_of_two():
    assert_equal(fht.is_power_of_two(8L), True)

if __name__ == "__main__":
    run_module_suite()
