from __future__ import annotations

import re

import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
from numpy.typing import ArrayLike


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


def filter(xn):
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
    m, n = arr.shape
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


if __name__ == "__main__":
    pass
