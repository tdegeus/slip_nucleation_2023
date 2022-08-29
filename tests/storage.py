import os
import shutil
import sys
import unittest

import GMatTensor.Cartesian2d as tensor
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

from mycode_front import storage  # noqa: E402


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_symtens2(self):

        dirname = "mytest"
        filename = "foo.h5"
        filepath = os.path.join(dirname, filename)
        key = "foo"

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        data = np.random.random([50, 2, 2])
        data = tensor.A4_ddot_B2(tensor.Array1d([50]).I4s, data)

        with h5py.File(filepath, "w") as file:

            storage.symtens2_create(file, key, np.float64)

            for i in range(data.shape[0]):
                storage.symtens2_extend(file, key, i, data[i, ...])

            self.assertTrue(np.allclose(data, storage.symtens2_read(file, key)))

        shutil.rmtree(dirname)

    def test_extend1d(self):

        dirname = "mytest"
        filename = "foo.h5"
        filepath = os.path.join(dirname, filename)
        key = "foo"

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        data = np.random.random(50)

        with h5py.File(filepath, "w") as file:

            storage.create_extendible(file, key, np.float64)

            for i, d in enumerate(data):
                storage.dset_extend1d(file, key, i, d)

            self.assertTrue(np.allclose(data, file[key][...]))

        shutil.rmtree(dirname)

    def test_dump_overwrite(self):

        dirname = "mytest"
        filename = "foo.h5"
        filepath = os.path.join(dirname, filename)
        key = "foo"

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        data = np.random.random(50)

        with h5py.File(filepath, "w") as file:

            for i in range(3):
                storage.dump_overwrite(file, key, data)
                self.assertTrue(np.allclose(data, file[key][...]))

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
