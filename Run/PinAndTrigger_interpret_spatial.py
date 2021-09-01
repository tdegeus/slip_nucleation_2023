import argparse
import enstat.mean
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import itertools
import numpy as np
import os
import sys
import tqdm
from numpy.typing import ArrayLike

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import PinAndTrigger


def center_of_mass(x: ArrayLike, L: float) -> float:
    """
    Compute the center of mass of a list if coordinates ``x``,
    accounting for periodicity on a length ``L``.
    Ref: https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

    :param x: List of coordinates.
    :param L: Linear length of the system.
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


def renumber(x: ArrayLike, L: float, N: int = None) -> np.ndarray:
    """
    Return indices to roll a system such that the center of mass of a list of points end up in
    the center of the system.

    :param x: List of coordinates.
    :param L: Linear length of the system.
    :param N: Number of blocks. By default it is assumed that ``N = L``.
    :return: List of indices to renumber the system.
    """

    if N is None:
        N = int(L)

    center = center_of_mass(x, L)
    M = int((N - N % 2) / 2)
    C = int(center)
    return np.roll(np.arange(N), M - C)


def fill_avalanche(broken: ArrayLike):
    """
    Fill-in avalanche.

    :param np.array broken:
        Per block if it was 'broken' or not.
        The size of the system ``N = broken.size``.

    :returns: Per block if it was broken, with all blocks 'inside' the avalanche filled-in.
    """

    N = broken.size
    broken = np.tile(broken, 3)
    ret = np.ones_like(broken)
    zero = np.zeros_like(broken)[0]

    i = np.argwhere(broken).ravel()
    di = np.diff(i)
    mi = np.max(di)
    j = np.argwhere(di == mi).ravel()
    ret[i[j[0]] : i[j[0] + 1]] = zero
    ret[i[j[1]] + 1 : i[j[1] + 1]] = zero

    return ret[N : 2 * N]


def average(data: h5py.File, paths: list[str], sig0: float) -> dict:
    """
    Compute the average spatial distributions.
    At the end of averaging the fields are interpolated to a regular grid and output at matrix.
    This function only makes senses for data that is 'aligned'.

    :param data: The opened file from which to read.
    :param paths: List of the datasets to average.
    :param sig0: Stress normalisation to apply.
    :return: Averaged fields, as 'matrix'.
    """

    system = None

    sig_xx = enstat.mean.StaticNd()
    sig_xy = enstat.mean.StaticNd()
    sig_yy = enstat.mean.StaticNd()

    for path in tqdm.tqdm(paths):

        if "disp" not in data[path]:
            continue

        # restore system

        file = str(data[path]["file"].asstr()[...])
        e = int(path.split("element=")[1].split("/")[0])
        a = int(path.split("A=")[1].split("/")[0])

        with h5py.File(file, "r") as mysim:
            if system is None:
                system = PinAndTrigger.initsystem(mysim)
                dV = system.quad().AsTensor(2, system.quad().dV())
                plastic = system.plastic()
                N = plastic.size
                mid = int((N - N % 2) / 2)
            else:
                PinAndTrigger.reset_epsy(system, mysim)

        # interpret event

        system.setU(data[path]["disp"]["0"][...])
        pinned = PinAndTrigger.pinsystem(system, e, a)

        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
        system.setU(data[path]["disp"]["1"][...])
        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

        # store stress

        Sig = system.Sig() / sig0
        S = np.average(Sig, weights=dV, axis=(1,))
        sig_xx.add_sample(S[:, 0, 0])
        sig_xy.add_sample(S[:, 0, 1])
        sig_yy.add_sample(S[:, 1, 1])

    pinned = PinAndTrigger.pinning(system, e, a)
    mesh = GooseFEM.Mesh.Quad4.FineLayer(system.coor(), system.conn())
    mapping = GooseFEM.Mesh.Quad4.Map.FineLayer2Regular(mesh)
    regular = mapping.getRegularMesh()
    elmat = regular.elementgrid()
    renum = renumber(np.argwhere(np.logical_not(pinned)).ravel(), pinned.size)

    is_plastic = np.zeros((system.conn().shape[0]), dtype=bool)
    is_plastic[plastic] = True
    is_plastic = mapping.mapToRegular(is_plastic)[elmat[:, renum].ravel()].reshape(
        elmat.shape
    )

    sig_xx = mapping.mapToRegular(sig_xx.mean())[elmat[:, renum].ravel()].reshape(
        elmat.shape
    )
    sig_xy = mapping.mapToRegular(sig_xy.mean())[elmat[:, renum].ravel()].reshape(
        elmat.shape
    )
    sig_yy = mapping.mapToRegular(sig_yy.mean())[elmat[:, renum].ravel()].reshape(
        elmat.shape
    )

    return dict(
        sig_xx=sig_xx,
        sig_xy=sig_xy,
        sig_yy=sig_yy,
        is_plastic=is_plastic,
    )


if __name__ == "__main__":

    basename = os.path.splitext(os.path.basename(__file__))[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input file ('r')")
    parser.add_argument(
        "-o", "--output", type=str, default=f"{basename}.h5", help="Output file ('w')"
    )
    parser.add_argument(
        "-i", "--info", type=str, default="EnsembleInfo.h5", help="Read normalisation"
    )
    args = parser.parse_args()
    assert os.path.isfile(os.path.realpath(args.file))
    assert os.path.isfile(os.path.realpath(args.info))
    assert os.path.realpath(args.file) != os.path.realpath(args.output)

    with h5py.File(args.info, "r") as data:
        sig0 = data["/normalisation/sig0"][...]

    with h5py.File(args.output, "w") as output:
        pass

    with h5py.File(args.file, "r") as data:

        Stress = list(g5.getpaths(data, root="data", max_depth=1))
        paths = list(g5.getpaths(data, root="data", max_depth=5))

        Element = np.unique(
            ["element=" + path.split("element=")[1].split("/...")[0] for path in paths]
        )
        Stress = np.array([path.split("data/")[1].split("/...")[0] for path in Stress])
        paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

        stress = np.array(
            ["stress=" + path.split("stress=")[1].split("/")[0] for path in paths]
        )
        element = np.array(
            ["element=" + path.split("element=")[1].split("/")[0] for path in paths]
        )
        a_target = np.array([int(path.split("A=")[1].split("/")[0]) for path in paths])
        a_real = np.array(
            [int(data[g5.join("/data", path, "A")][...]) for path in paths]
        )

        A_target = np.unique(a_target)

        for a, s in itertools.product(A_target, Stress):

            sig_xx = enstat.mean.StaticNd()
            sig_xy = enstat.mean.StaticNd()
            sig_yy = enstat.mean.StaticNd()

            for e in Element:

                subset = paths[
                    (a_real > a - 10)
                    * (a_real < a + 10)
                    * (element == e)
                    * (stress == s)
                ]
                ret = average(data, [g5.join("/data", path) for path in subset], sig0)
                sig_xx.add_sample(ret["sig_xx"])
                sig_xy.add_sample(ret["sig_xy"])
                sig_yy.add_sample(ret["sig_yy"])

            sig_xx = sig_xx.mean()
            sig_xy = sig_xy.mean()
            sig_yy = sig_yy.mean()

            with h5py.File(args.output, "a") as output:

                output[g5.join("/data", str(s), str(a), "sig_xx")] = sig_xx
                output[g5.join("/data", str(s), str(a), "sig_xy")] = sig_xy
                output[g5.join("/data", str(s), str(a), "sig_yy")] = sig_yy

                if "is_plastic" not in output:
                    output["is_plastic"] = ret["is_plastic"]
