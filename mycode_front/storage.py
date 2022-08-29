"""
h5py 'extensions'.
"""
from typing import TypeVar

import h5py
import numpy as np
from numpy.typing import ArrayLike


def create_extendible(file: h5py.File, key: str, dtype, ndim: int = 1, **kwargs) -> h5py.Dataset:
    """
    Create extendible dataset.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param dtype: Data-type to use.
    :param ndim: Number of dimensions.
    :param kwargs: An optional dictionary with attributes.
    """

    if key in file:
        return file[key]

    shape = tuple(0 for i in range(ndim))
    maxshape = tuple(None for i in range(ndim))
    dset = file.create_dataset(key, shape, maxshape=maxshape, dtype=dtype)

    for attr in kwargs:
        dset.attrs[attr] = kwargs[attr]

    return dset


def symtens2_create(file: h5py.File, key: str, dtype, **kwargs) -> h5py.Dataset:
    """
    Create extendible dataset to store a **symmetric** 2nd-order tensor.
    This write the attribute ``"components" = ["xx", "xy", "yy"]``.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param dtype: Data-type to use.
    :param kwargs: An optional dictionary with attributes.
    """
    assert key not in file, "Dataset already exists"
    shape = (3, 0)
    maxshape = (3, None)
    dset = file.create_dataset(key, shape, maxshape=maxshape, dtype=dtype)
    dset.attrs["components"] = ["xx", "xy", "yy"]

    for attr in kwargs:
        dset.attrs[attr] = kwargs[attr]

    return dset


def symtens2_extend(file: h5py.File, key: str, index: int, data: ArrayLike):
    """
    Extend and write a **symmetric** 2nd-order tensor to an extendible dataset,
    see :py:func:`create_symtens2`.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param index: Index to write.
    :param data: Data to write [2, 2].
    """
    assert data.shape == (2, 2)
    dset = file[key]

    if dset.shape[1] <= index:
        dset.resize((3, index + 1))

    dset[0, index] = data[0, 0]
    dset[1, index] = data[0, 1]
    dset[2, index] = data[1, 1]

    return dset


def symtens2_read(file: h5py.File, key: str) -> np.ndarray:
    """
    Read a **symmetric** 2nd-order tensor from a dataset.
    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :return: Data [n, 2, 2].
    """
    dset = file[key]
    assert dset.ndim == 2
    assert dset.shape[0] == 3
    assert list(dset.attrs["components"]) == ["xx", "xy", "yy"]
    ret = np.zeros((dset.shape[1], 2, 2))
    ret[:, 0, 0] = dset[0, :]
    ret[:, 0, 1] = dset[1, :]
    ret[:, 1, 1] = dset[2, :]
    return ret


def dset_extendible1d(
    file: h5py.File, key: str, dtype, value: TypeVar("T"), **kwargs
) -> h5py.Dataset:
    """
    Create extendible 1d dataset and store the first value.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param dtype: Data-type to use.
    :param value: Value to write at index 0.
    :param kwargs: An optional dictionary with attributes.
    """

    dset = file.create_dataset(key, (1,), maxshape=(None,), dtype=dtype)
    dset[0] = value

    for attr in kwargs:
        dset.attrs[attr] = kwargs[attr]

    return dset


def dset_extend1d(file: h5py.File, key: str, i: int, value: TypeVar("T")):
    """
    Dump and auto-extend a 1d extendible dataset.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param i: Index to which to write.
    :param value: Value to write at index ``i``.
    """

    dset = file[key]
    if dset.size <= i:
        dset.resize((i + 1,))
    dset[i] = value


def dump_with_atttrs(file: h5py.File, key: str, data: TypeVar("T"), **kwargs):
    """
    Write dataset and an optional number of attributes.
    The attributes are stored based on the name that is used for the option.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param data: Data to write.
    """

    file[key] = data
    for attr in kwargs:
        file[key].attrs[attr] = kwargs[attr]


def dump_overwrite(file: h5py.File, key: str, data: TypeVar("T")):
    """
    Dump or overwrite data.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param data: Data to write.
    """

    if key in file:
        file[key][...] = data
        return

    file[key] = data
