import numpy as np

# todo: vectorise implementation
# todo: implementation without allocation of coordinates
def _center_of_mass(x, L):
    """
    Compute the center of mass of a periodic system.
    Assume: all equal masses.

    :param x: List of coordinates.
    :param L: Length of the system.
    :return: Coordinate of the center of mass.
    """

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

    ret = np.empty((m))

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
    arr = np.swapaxes(arr,axis,-1)
    all_idcs = np.ogrid[[slice(0,n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result,-1,axis)
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

    N = broken.size
    broken = np.tile(broken, 3)
    ret = np.ones_like(broken)
    zero = np.zeros_like(broken)[0]

    i = np.argwhere(broken).ravel()
    di = np.diff(i)
    mi = np.max(di)
    j = np.argwhere(di == mi).ravel()
    ret[i[j[0]]: i[j[0] + 1]] = zero
    ret[i[j[1]] + 1: i[j[1] + 1]] = zero

    return ret[N: 2 * N]



if __name__ == "__main__":
    pass
