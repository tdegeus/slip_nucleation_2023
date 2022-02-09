import os
import shutil
import sys
import unittest

import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    """
    namespace tools
    """

    def test_h5py_save_unique(self):

        dirname = "mytest"
        filepath = os.path.join(dirname, "foo.h5")

        for file in [filepath]:
            if os.path.isfile(file):
                os.remove(file)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        a = ["a", "foo", "a", "a", "bar", "foo"]
        b = (np.random.random((3, 4, 5)) * 10).astype(int)
        c = [
            ["a", "some"],
            ["foo", "bar"],
            ["a", "other"],
            ["a", "some"],
            ["bar", "foo"],
            ["foo", "bar"],
        ]

        with h5py.File(filepath, "w") as file:
            my.tools.h5py_save_unique(a, file, "a", asstr=True)
            a_r = my.tools.h5py_read_unique(file, "a", asstr=True)

            my.tools.h5py_save_unique(b, file, "b")
            b_r = my.tools.h5py_read_unique(file, "b")

            my.tools.h5py_save_unique([";".join(i) for i in c], file, "c", split=";")
            c_r = my.tools.h5py_read_unique(file, "c", asstr=True)

        self.assertEqual(a, a_r)
        self.assertTrue(np.all(np.equal(b, b_r)))
        self.assertEqual(c, c_r)

        shutil.rmtree(dirname)

    def test_check_docstring(self):

        docstring = """\
        Foo bar.

        :param a: ...
        :param b: ...
        :return:
            A dictionary as follows,
            with some comment::

                a: My test
                b: Other text
        """

        my.tools.check_docstring(docstring, dict(a=None, b=None))

        docstring = """\
        Foo bar.

        :param a: ...
        :param b: ...
        :return:
            A dictionary as follows::

                a: My test
                b: Other text

            Some notes.
        """

        my.tools.check_docstring(docstring, dict(a=None, b=None))

        docstring = """\
        Foo bar.

        :param a: ...
        :param b: ...
        :return:
            A dictionary as follows:

            .. code-block:: yaml

                a: My test
                b: Other text

            Some notes.
        """

        my.tools.check_docstring(docstring, dict(a=None, b=None))

    def test_read_parameters(self):

        a = "/this/is/my/a=10_b=20/c=30.2"
        b = f"{a}.txt"

        convert = dict(
            a=int,
            b=int,
            c=float,
        )

        value = dict(
            a=10,
            b=20,
            c=30.2,
        )

        string = dict(
            a="10",
            b="20",
            c="30.2",
        )

        self.assertEqual(my.tools.read_parameters(os.path.splitext(b)[0]), string)
        self.assertEqual(my.tools.read_parameters(a), string)
        self.assertEqual(my.tools.read_parameters(a, convert=convert), value)

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

    def test_distance(self):

        a = np.random.random((10, 3))
        b = np.random.random((15, 3))

        D = np.zeros((a.shape[0], b.shape[0]))

        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                for d in range(a.shape[1]):
                    D[i, j] += (b[j, d] - a[i, d]) ** 2

        D = np.sqrt(D)

        self.assertTrue(np.allclose(D, my.tools.distance(a, b)))

    def test_minimal_distance(self):

        a = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
            ]
        )

        b = np.array(
            [
                [3, 1],
                [0, 1],
            ]
        )

        closest = np.array([1, 1, 0, 0])

        self.assertTrue(np.all(np.equal(closest, np.argmin(my.tools.distance(a, b), axis=1))))

    def test_distance1d(self):

        a = np.random.random(10)
        b = np.random.random(15)

        D = np.zeros((a.shape[0], b.shape[0]))

        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                D[i, j] += (b[j] - a[i]) ** 2

        D = np.sqrt(D)

        self.assertTrue(np.allclose(D, my.tools.distance1d(a, b)))

    def test_minimal_distance1d(self):

        a = np.array([0, 1, 2, 3])
        b = np.array([3, 0])
        closest = np.array([1, 1, 0, 0])

        self.assertTrue(np.all(np.equal(closest, np.argmin(my.tools.distance1d(a, b), axis=1))))

    def test_minimal_distance1d_negative(self):

        a = np.array([0, 1, 2, 3])
        b = np.array([-2, -1, 0, 1])
        closest = np.array([0, 0, 0, 1])

        self.assertTrue(np.all(np.equal(closest, np.argmin(my.tools.distance1d(b, a), axis=1))))


if __name__ == "__main__":

    unittest.main()
