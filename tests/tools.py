import os
import sys
import unittest

import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_center_avalanche(self):

        S = np.array([1, 1, 0, 0, 0])
        T = np.array([0, 0, 1, 1, 0])

        R = my.tools.center_avalanche(S)
        C = np.roll(S, R)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_a(self):

        S = np.array([[1, 1, 0, 0, 0], [3, 3, 0, 0, 0], [0, 0, 0, 4, 4], [0, 0, 7, 9, 0]])

        T = np.array([[0, 0, 1, 1, 0], [0, 0, 3, 3, 0], [0, 0, 4, 4, 0], [0, 0, 7, 9, 0]])

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_aa(self):

        S = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [3, 3, 0, 0, 0, 0],
                [0, 0, 0, 4, 4, 0],
                [0, 0, 7, 9, 0, 0],
            ]
        )

        T = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 7, 9, 0, 0],
            ]
        )

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_aaa(self):

        S = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [3, 3, 0, 0, 0, 0],
                [0, 0, 0, 4, 4, 0],
                [0, 0, 7, 9, 0, 0],
            ]
        )

        T = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 7, 9, 0, 0],
            ]
        )

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_aaaa(self):

        S = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [3, 3, 0, 0, 0, 0],
                [0, 0, 0, 4, 4, 0],
                [0, 0, 7, 9, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        T = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 7, 9, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_missing(self):

        S = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [3, 3, 0, 0, 0, 0],
                [0, 0, 0, 4, 4, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        T = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 4, 4, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_center_avalanche_per_row_b(self):

        S = np.array([[1, 1, 0, 0, 0], [3, 3, 0, 0, 0], [4, 4, 0, 0, 4], [7, 8, 9, 0, 8]])

        T = np.array([[0, 0, 1, 1, 0], [0, 0, 3, 3, 0], [0, 4, 4, 4, 0], [0, 8, 7, 8, 9]])

        R = my.tools.center_avalanche_per_row(S)
        C = my.tools.indep_roll(S, R, axis=1)

        self.assertTrue(np.all(C == T))

    def test_fill_avalanche(self):

        a = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        b = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

        for i in range(a.size + 1):
            a = np.roll(a, 1)
            b = np.roll(b, 1)
            self.assertTrue(np.all(my.tools.fill_avalanche(a) == b))

        a = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0])
        b = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

        for i in range(a.size + 1):
            a = np.roll(a, 1)
            b = np.roll(b, 1)
            self.assertTrue(np.all(my.tools.fill_avalanche(a) == b))

        a = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0])
        b = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0])

        for i in range(a.size + 1):
            a = np.roll(a, 1)
            b = np.roll(b, 1)
            self.assertTrue(np.all(my.tools.fill_avalanche(a) == b))

        a = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0])
        b = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0])

        for i in range(a.size + 1):
            a = np.roll(a, 1)
            b = np.roll(b, 1)
            self.assertTrue(np.all(my.tools.fill_avalanche(a) == b))

        a = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0])
        b = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1])

        for i in range(a.size + 1):
            a = np.roll(a, 1)
            b = np.roll(b, 1)
            self.assertTrue(np.all(my.tools.fill_avalanche(a) == b))


if __name__ == "__main__":

    unittest.main()
