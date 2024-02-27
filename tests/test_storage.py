import os

import GMatTensor.Cartesian2d as tensor
import h5py
import numpy as np

from slip_nucleation_2023 import storage


def test_symtens2(tmp_path):
    filename = "foo.h5"
    filepath = os.path.join(tmp_path, filename)
    key = "foo"
    data = np.random.random([50, 2, 2])
    data = tensor.A4_ddot_B2(tensor.Array1d([50]).I4s, data)

    with h5py.File(filepath, "w") as file:
        storage.symtens2_create(file, key, np.float64)

        for i in range(data.shape[0]):
            storage.symtens2_extend(file, key, i, data[i, ...])

        assert np.allclose(data, storage.symtens2_read(file, key))


def test_extend1d(tmp_path):
    filename = "foo.h5"
    filepath = os.path.join(tmp_path, filename)
    key = "foo"
    data = np.random.random(50)

    with h5py.File(filepath, "w") as file:
        storage.create_extendible(file, key, np.float64)

        for i, d in enumerate(data):
            storage.dset_extend1d(file, key, i, d)

        assert np.allclose(data, file[key][...])


def test_dump_overwrite(tmp_path):
    filename = "foo.h5"
    filepath = os.path.join(tmp_path, filename)
    key = "foo"
    data = np.random.random(50)

    with h5py.File(filepath, "w") as file:
        for i in range(3):
            storage.dump_overwrite(file, key, data)
            assert np.allclose(data, file[key][...])
