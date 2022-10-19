from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import click
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike


class PartialDisplacement:
    """
    Helper class to store/read only a partial displacement field.
    Note that options makers (-) are mutually exclusive.

    :param conn: Connectivity.
    :param dofs: DOFs per node.
    :param dof_is_stored: Per DOF: ``True`` if it is stored (-).
    :param node_is_stored: Per node: ``True`` if it is stored (-).
    :param element_is_stored: Per element: ``True`` if it is stored (-).
    :param dof_list: List of DOFs stored (-).
    :param node_list: List of nodes stored (-).
    :param element_list: List of elements stored (-).
    """

    def __init__(
        self,
        conn: ArrayLike,
        dofs: ArrayLike,
        dof_is_stored: ArrayLike = None,
        node_is_stored: ArrayLike = None,
        element_is_stored: ArrayLike = None,
        dof_list: ArrayLike = None,
        node_list: ArrayLike = None,
        element_list: ArrayLike = None,
    ):

        kwargs = locals()
        del kwargs["conn"]
        del kwargs["dofs"]
        del kwargs["self"]
        assert sum(kwargs[key] is not None for key in kwargs) == 1

        if dof_list is not None:
            dof_is_stored = np.zeros(int(np.max(dofs) + 1), dtype=bool)
            dof_is_stored[dof_list] = True

        if node_list is not None:
            node_is_stored = np.zeros(dofs.shape[0], dtype=bool)
            node_is_stored[node_list] = True

        if element_list is not None:
            element_is_stored = np.zeros(conn.shape[0], dtype=bool)
            element_is_stored[element_list] = True

        self.m_conn = np.copy(conn)
        self.m_dofs = np.copy(dofs)
        self.m_s_dofs = None
        self.m_s_node = None
        self.m_s_elem = None
        self.m_s_assembly = None
        self.m_i_dofs = None
        self.m_i_node = None
        self.m_i_elem = None
        self.m_i_assembly = None

        if dof_is_stored is not None:
            dof_is_stored = np.array(dof_is_stored)
            assert dof_is_stored.size == np.max(dofs) + 1
            assert dof_is_stored.ndim == 1
            self.m_s_dofs = np.copy(dof_is_stored).astype(bool)

        if node_is_stored is not None:
            node_is_stored = np.array(node_is_stored)
            assert node_is_stored.size == dofs.shape[0]
            assert node_is_stored.ndim == 1
            self.m_s_node = np.copy(node_is_stored).astype(bool)
            self.m_i_dofs = np.unique(dofs[self.m_s_node])

        if element_is_stored is not None:
            element_is_stored = np.array(element_is_stored)
            assert element_is_stored.size == conn.shape[0]
            assert element_is_stored.ndim == 1
            self.m_s_elem = np.copy(element_is_stored).astype(bool)
            self.m_i_node = np.unique(conn[self.m_s_elem])
            self.m_i_dofs = np.unique(dofs[self.m_i_node, :].ravel())

        if self.m_s_dofs is None:
            self.m_s_dofs = np.zeros(int(np.max(dofs) + 1), dtype=bool)
            self.m_s_dofs[self.m_i_dofs] = True

        if self.m_i_dofs is None:
            self.m_i_dofs = np.argwhere(self.m_s_node).ravel()

        if self.m_s_node is None:
            self.m_s_node = np.min(self.m_s_dofs[dofs], axis=1)

        if self.m_i_node is None:
            self.m_i_node = np.argwhere(self.m_s_node).ravel()

        if self.m_s_elem is None:
            self.m_s_elem = np.min(self.m_s_node[conn], axis=1)

        if self.m_i_elem is None:
            self.m_i_elem = np.argwhere(self.m_s_elem).ravel()

    def dof_is_stored(self):
        """
        Per DOF: ``True`` if it can be reconstructed based on the storage.
        """
        return self.m_s_dofs

    def node_is_stored(self):
        """
        Per node: ``True`` if it can be reconstructed based on the storage.
        """
        return self.m_s_node

    def element_is_stored(self):
        """
        Per element: ``True`` if it can be reconstructed based on the storage.
        """
        return self.m_s_elem

    def nodeassembly_is_stored(self):
        """
        Per node: ``True`` if it can be reconstructed based on the storage,
        in the case that the nodal quantity follows from an assembly (e.g. a force).
        """
        if self.m_s_assembly is None:
            self.m_s_assembly = np.zeros((self.m_dofs.shape[0]), dtype=bool)
            nodemap = GooseFEM.Mesh.elem2node(conn=self.m_conn, dofs=self.m_dofs)
            for i in np.argwhere(self.m_s_node).ravel():
                self.m_s_assembly[i] = all(self.m_s_elem[nodemap[i]])
            self.m_i_assembly = np.argwhere(self.m_s_assembly).ravel()
        return self.m_s_assembly

    def dof_list(self):
        """
        List of DOFs that can be reconstructed based on the storage.
        """
        return self.m_i_dofs

    def node_list(self):
        """
        List of nodes that can be reconstructed based on the storage.
        """
        return self.m_i_node

    def element_list(self):
        """
        List of elements that can be reconstructed based on the storage.
        """
        return self.m_i_elem

    def nodeassembly_list(self):
        """
        List of nodes that can be reconstructed based on the storage,
        in the case that the nodal quantity follows from an assembly (e.g. a force).
        """
        self.nodeassembly_is_stored()
        return self.m_i_assembly


def h5py_read_unique(
    file: h5py.File,
    path: str,
    asstr: bool = False,
) -> np.ndarray:
    """
    Return original array stored by :py:func:`h5py_save_unique`.

    :param file: HDF5 archive.
    :param path: Group containing ``index`` and ``value``.
    :param asstr: Return as list of strings.
    :returns: Data.
    """

    index = file[g5.join(path, "index", root=True)][...]

    if asstr:
        value = file[g5.join(path, "value", root=True)].asstr()[...]
    else:
        value = file[g5.join(path, "value", root=True)][...]

    ret = value[index]

    if asstr:
        return ret.tolist()

    return ret


def h5py_save_unique(
    data: ArrayLike,
    file: h5py.File,
    path: str,
    asstr: bool = False,
    split: str = None,
):
    """
    Save a list of strings (or other data, but mostly relevant for strings)
    with many duplicates as two datasets:

    -   ``path/value``: list of unique strings.
    -   ``path/index``: per item which index from ``path/value`` to take.

    Use :py:func:`h5py_read_unique` to read data.

    :param data: Data to store.
    :param file: HDF5 archive.
    :param path: Group containing ``index`` and ``value``.
    :param asstr: Convert to list of strings before storing.
    :param split: Split every item for a list of strings before storing.
    """

    value, index = np.unique(data, return_inverse=True)

    if split is not None:
        # catch bug: can only store square arrays
        n = [len(i) for i in list(map(lambda i: str(i).split(split), value))]
        if np.all(np.equal(n, n[0])):
            value = list(map(lambda i: str(i).split(split), value))
        else:
            value = list(map(str, value))
    elif asstr:
        value = list(map(str, value))

    if isinstance(data, np.ndarray):
        index = index.reshape(data.shape)

    file[g5.join(path, "index", root=True)] = index
    file[g5.join(path, "value", root=True)] = value


def inboth(a: dict | list, b: dict | list, name_a: str = "a", name_b: str = "b"):
    """
    Check if a dictionary/list ``a`` has all fields as ``b`` and vice-versa.

    :param a: List or dictionary.
    :param b: List or dictionary.
    """

    for key in a:
        if key not in b:
            raise OSError(f"{key} not in {name_b}")

    for key in b:
        if key not in a:
            raise OSError(f"{key} not in {name_a}")


def check_docstring(string: str, variable: dict, key: str = ":return:"):
    """
    Make sure that all variables in a dictionary are documented in a docstring.
    The function assumes a docstring as follows::

        :param a: ...
        :param b: ...
        :return: ...::
            name: description

    Thereby the docstring is split:
    1.  At a parameter (e.g. `":return:"`)
    2.  At `.. code-block::` or `::`

    The indented code after is assumed to be formatted as YAML and is the code we search.
    """

    d = string.split(":return:")[1]

    if len(d.split(".. code-block::")) > 1:
        d = d.split(".. code-block::")[1].split("\n", 1)[1]
    elif len(d.split("::")) > 1:
        d = d.split("::")[1]

    d = d.split("\n")
    d = list(filter(None, d))
    d = list(filter(lambda name: name.strip(), d))
    indent = len(d[0]) - len(d[0].lstrip())
    d = list(filter(lambda name: len(name) - len(name.lstrip()) == indent, d))
    d = "\n".join(d)

    inboth(yaml.safe_load(d), variable, "docstring", "variable")


def read_parameters(string: str, convert: dict = None) -> dict:
    """
    Read parameters from a string: it is assumed that parameters are split by ``_`` or ``/``
    and that parameters are stored as ``name=value``.

    :param string: ``key=value`` separated by ``/`` or ``_``.
    :param convert: Type conversion for a selection of keys. E.g. ``{"id": int}``.
    :return: Parameters as dictionary.
    """

    part = re.split("_|/", string)

    ret = {}

    for i in part:
        if len(i.split("=")) > 1:
            key, value = i.split("=")
            ret[key] = value

    if convert:
        for key in convert:
            ret[key] = convert[key](ret[key])

    return ret


def signal_filter(xn):
    """
    Filter a signal.

    :param xn: The signal.
    :return: The filtered signal
    """

    from scipy import signal

    N = xn.size
    xn = np.tile(xn, (3))

    # Create an order 3 lowpass butterworth filter:
    b, a = signal.butter(3, 0.1)

    # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi * xn[0])

    # Apply the filter again, to have a result filtered at an order the same as filtfilt:
    z2, _ = signal.lfilter(b, a, z, zi=zi * z[0])

    # Use filtfilt to apply the filter:
    y = signal.filtfilt(b, a, xn)

    i = N
    j = N * 2
    return y[i:j]


def sigd(xx, xy, yy):

    Sig = np.empty(list(xx.shape) + [2, 2])
    Sig[..., 0, 0] = xx
    Sig[..., 0, 1] = xy
    Sig[..., 1, 0] = xy
    Sig[..., 1, 1] = yy
    return GMat.Sigd(Sig)


def epsd(xx, xy, yy):

    Eps = np.empty(list(xx.shape) + [2, 2])
    Eps[..., 0, 0] = xx
    Eps[..., 0, 1] = xy
    Eps[..., 1, 0] = xy
    Eps[..., 1, 1] = yy
    return GMat.Epsd(Eps)


def _center_of_mass(x, L):
    """
    Compute the center of mass of a periodic system.
    Assume: all equal masses.

    :param x: List of coordinates.
    :param L: Length of the system.
    :return: Coordinate of the center of mass.
    """

    # todo: vectorise implementation
    # todo: implementation without allocation of coordinates

    if np.allclose(x, 0):
        return 0

    theta = 2.0 * np.pi * x / L
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi_bar = np.mean(xi)
    zeta_bar = np.mean(zeta)
    theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
    return L * theta_bar / (2.0 * np.pi)


def _center_of_mass_per_row(arr):
    """
    Compute the center of mass per row.
    The function assumes that masses can be either 0 or 1:
    -   1: any positive value
    -   0: any zero or negative value

    :param: Input array [M, N].
    :return: x-position of the center of mass per row [M].
    """

    assert arr.ndim == 2
    m, n = arr.shape

    ret = np.empty(m)

    for i in range(m):
        ret[i] = _center_of_mass(np.argwhere(arr[i, :] > 0).ravel(), n)

    return ret


def indep_roll(arr, shifts, axis=1):
    """
    Apply an independent roll for each dimensions of a single axis.
    See: https://stackoverflow.com/a/56175538/2646505

    :param arr: Array of any shape.
    :param shifts: Shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
    :param axis: Axis along which elements are shifted.
    :return: Rolled array.
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def center_avalanche_per_row(arr):
    """
    Shift to center avalanche, per row. Example usage::

        R = center_avalanche_per_row(S)
        C = indep_roll(S, R, axis=1)

    Note that the input array is interpreted as follows:
    -   any positive value == 1
    -   any zero or negative value == 0

    :param arr: Per row: if the block yielded.
    :return: Shift per row.
    """

    assert arr.ndim == 2
    n = arr.shape[1]
    shift = np.floor(n / 2 - _center_of_mass_per_row(arr)).astype(int)
    return np.where(shift < 0, n + shift, shift)


def center_avalanche(arr):
    """
    Shift to center avalanche. Example usage::
        R = center_avalanche(S)
        C = np.roll(S, R)

    :param arr: If the block yielded (or the number of times it yielded).
    :return: Shift.
    """

    return center_avalanche_per_row(arr.reshape(1, -1))[0]


def fill_avalanche(broken):
    """
    Fill avalanche such that the largest spatial extension can be selected.

    :param broken: Per block if it is broken.
    :return: ``broken`` for filled avalanche.
    """

    assert broken.ndim == 1

    if np.sum(broken) <= 1:
        return broken

    N = broken.size
    broken = np.tile(broken, 3)
    ret = np.ones_like(broken)
    zero = np.zeros_like(broken)[0]

    x = np.argwhere(broken).ravel()
    dx = np.diff(x)
    maxdx = np.max(dx)
    j = np.argwhere(dx == maxdx).ravel()

    x0 = x[j[0]]
    x1 = x[j[0] + 1]
    ret[x0:x1] = zero

    x0 = x[j[1]] + 1
    x1 = x[j[1] + 1]
    ret[x0:x1] = zero

    i = N
    j = 2 * N
    return ret[i:j]


def distance(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """
    Compute the distance of each point in ``a`` to each point in ``b``,
    whereby each row corresponds to the distance between the corresponding entry in ``a``
    to all points in ``b``.

    :param a: List with coordinates, shape: ``[n, d]``.
    :param b: List with coordinates, shape: ``[m, d]``.
    :return: Matrix with distances, shape: ``[n, m]``.
    """

    # https://mlxai.github.io/2017/01/03/finding-distances-between-data-points-with-numpy.html

    M = np.dot(a, b.T)
    da = np.square(a).sum(axis=1).reshape(-1, 1)
    db = np.square(b).sum(axis=1).reshape(-1, 1)
    return np.sqrt(-2 * M + db.T + da)


def distance1d(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """
    Compute the distance of each point in ``a`` to each point in ``b``,
    whereby each row corresponds to the distance between the corresponding entry in ``a``
    to all points in ``b``.

    :param a: List with coordinates, shape: ``[n]``.
    :param b: List with coordinates, shape: ``[m]``.
    :return: Matrix with distances, shape: ``[n, m]``.
    """

    # https://mlxai.github.io/2017/01/03/finding-distances-between-data-points-with-numpy.html

    assert a.ndim == 1
    assert b.ndim == 1

    M = np.dot(a.reshape(-1, 1), b.reshape(1, -1))
    da = np.square(a).reshape(-1, 1)
    db = np.square(b).reshape(-1, 1)
    return np.sqrt(-2 * M + db.T + da)


def _parse(parser: argparse.ArgumentParser, cli_args: list[str]) -> argparse.ArgumentParser:

    if cli_args is None:
        return parser.parse_args(sys.argv[1:])

    return parser.parse_args([str(arg) for arg in cli_args])


def _check_overwrite_file(filepath: str, force: bool):

    if force or not os.path.isfile(filepath):
        return

    if not click.confirm(f'Overwrite "{filepath}"?'):
        raise OSError("Cancelled")


def _create_or_clear_directory(dirpath: str, force: bool):

    if os.path.isdir(dirpath):

        if not force:
            if not click.confirm(f'Clear "{dirpath}"?'):
                raise OSError("Cancelled")

        shutil.rmtree(dirpath)

    os.makedirs(dirpath)


if __name__ == "__main__":
    pass
