import argparse
import os

import FrictionQPotFEM.UniformSingleLayer2d as model
import git
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM  # noqa: F401
import h5py
import numpy as np
import QPot  # noqa: F401
import setuptools_scm


def initsystem(data: h5py.File) -> model.System:
    r"""
    Read system from file.

    :param data: Open simulation HDF5 archive (read-only).
    :return: The initialised system.
    """

    system = model.System(
        data["/coor"][...],
        data["/conn"][...],
        data["/dofs"][...],
        data["/dofsP"][...],
        data["/elastic/elem"][...],
        data["/cusp/elem"][...],
    )

    system.setMassMatrix(data["/rho"][...])
    system.setDampingMatrix(data["/damping/alpha"][...])
    system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
    system.setPlastic(
        data["/cusp/K"][...], data["/cusp/G"][...], data["/cusp/epsy"][...]
    )
    system.setDt(data["/run/dt"][...])

    return system


def reset_epsy(system: model.System, data: h5py.File):
    r"""
    Reset yield strain history from file.

    :param system: The system (modified: yield strains changed).
    :param data: Open simulation HDF5 archive (read-only).
    """

    e = data["/cusp/epsy"][...]
    epsy = np.empty((e.shape[0], e.shape[1] + 1), dtype=e.dtype)
    epsy[:, 0] = -e[:, 0]
    epsy[:, 1:] = e

    plastic = system.plastic()
    N = plastic.size
    nip = system.quad().nip()
    material = system.material()
    material_plastic = system.material_plastic()

    assert epsy.shape[0] == N

    for i, e in enumerate(plastic):
        for q in range(nip):
            for cusp in [
                material.refCusp([e, q]),
                material_plastic.refCusp([i, q]),
            ]:
                chunk = cusp.refQPotChunked()
                chunk.set_y(epsy[i, :])


def pushincrements(
    system: model.System, data: h5py.File, target_stress: float
) -> (np.ndarray, np.ndarray):
    r"""
    Get a list of increment from which the stress can be reached by elastic loading only.

    :param system: The system (modified: all increments visited).
    :param data: Open simulation HDF5 archive (read-only).
    :param target_stress: The stress at which to push (in real units).
    :return:
        ``inc_system`` List of system spanning avalanches.
        ``inc_push`` List of increment from which the stress can be reached by elastic loading only.
    """

    dV = system.quad().AsTensor(2, system.quad().dV())
    kick = data["/kick"][...].astype(bool)
    incs = data["/stored"][...].astype(int)
    assert np.all(incs == np.arange(incs.size))
    assert kick.shape == incs.shape
    assert np.all(not kick[::2])
    assert np.all(kick[1::2])

    A = np.zeros(incs.shape, dtype=int)
    Strain = np.zeros(incs.shape, dtype=float)
    Stress = np.zeros(incs.shape, dtype=float)

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

    for inc in incs:

        system.setU(data[f"/disp/{inc:d}"])

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        Sig = system.Sig()
        Eps = system.Eps()

        A[inc] = np.sum(idx != idx_n)
        Strain[inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        Stress[inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.array(idx, copy=True)

    # estimate steady-state using secant modulus:
    # - always skip two increments
    # - start with elastic loading
    K = np.empty_like(Stress)
    K[0] = np.inf
    K[1:] = (Stress[1:] - Stress[0]) / (Strain[1:] - Strain[0])
    steadystate = max(2, np.argmax(K <= 0.95 * K[1]))
    if kick[steadystate]:
        steadystate += 1

    A[:steadystate] = 0

    inc_system = np.argwhere(A == N).ravel()
    inc_push = []
    inc_system_ret = []

    for i in range(inc_system.size - 1):

        # state after elastc loading
        ii = inc_system[i] + 1
        jj = inc_system[i + 1]
        s = Stress[ii:jj:2]
        n = incs[ii:jj:2]

        if not np.any(s > target_stress):
            continue

        j = np.argmax(s > target_stress)
        ipush = n[j] - 1

        assert Stress[ipush] <= target_stress
        assert not kick[ipush + 1]

        inc_push += [ipush]
        inc_system_ret += [n[0] - 1]

    inc_push = np.array(inc_push)
    inc_system_ret = np.array(inc_system_ret)

    return inc_system_ret, inc_push


def pinning(system: model.System, target_element: int, target_A: int) -> np.ndarray:
    r"""
    Return pinning used in ``pinsystem``.

    :param system:
        The system (modified: yield strains changed).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

    :return: Per element: pinned (``True``) or not (``False``)
    """

    plastic = system.plastic()
    N = plastic.size

    assert target_A <= N
    assert target_element <= N

    i = int(N - target_A / 2)
    pinned = np.ones((3 * N), dtype=bool)

    ii = i
    jj = i + target_A
    pinned[ii:jj] = False

    ii = N + i
    jj = N + i + target_A
    pinned[ii:jj] = False

    ii = N
    jj = 2 * N
    pinned = pinned[ii:jj]

    pinned = np.roll(pinned, target_element)

    return pinned


def pinsystem(system: model.System, target_element: int, target_A: int) -> np.ndarray:
    r"""
    Pin down part of the system by converting blocks to being elastic:
    having a single parabolic potential with the minimum equal to the current minimum.

    :param system:
        The system (modified: yield strains changed).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

    :return: Per element: pinned (``True``) or not (``False``)
    """

    plastic = system.plastic()
    nip = system.quad().nip()
    pinned = pinning(system, target_element, target_A)
    idx = system.plastic_CurrentIndex()
    material = system.material()
    material_plastic = system.material_plastic()

    for i, e in enumerate(plastic):
        if pinned[i]:
            for q in range(nip):
                for cusp in [
                    material.refCusp([e, q]),
                    material_plastic.refCusp([i, q]),
                ]:
                    chunk = cusp.refQPotChunked()
                    y = chunk.y()
                    ymax = y[-1]  # get some scale
                    ii = int(idx[i, q])
                    jj = int(idx[i, q] + 2)  # slicing is up to not including
                    y = y[ii:jj]
                    ymin = 0.5 * sum(y)  # current minimum
                    chunk.set_y([ymin - 2 * ymax, ymin + 2 * ymax])

    return pinned


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f", "--file", type=str, help="Filename of simulation file (read-only)"
    )

    parser.add_argument(
        "-o", "--output", type=str, help="Filename of output file (overwritten)"
    )

    parser.add_argument(
        "-s", "--stress", type=float, help="Stress as which to trigger (in real units)"
    )

    parser.add_argument(
        "-i", "--incc", type=int, help="Increment number of last system-spanning event"
    )

    parser.add_argument(
        "-e", "--element", type=int, help="Element to push (index along the weak layer)"
    )

    parser.add_argument(
        "-a", "--size", type=int, help="Number of elements to keep unpinned"
    )

    args = parser.parse_args()
    assert os.path.isfile(os.path.realpath(args.file))
    assert os.path.realpath(args.file) != os.path.realpath(args.output)

    print("starting:", args.output)

    root = git.Repo(
        os.path.dirname(__file__), search_parent_directories=True
    ).working_tree_dir
    myversion = setuptools_scm.get_version(root=root)

    target_stress = args.stress
    target_inc_system = args.incc
    target_A = args.size  # number of blocks to keep unpinned
    target_element = args.element  # element to trigger

    with h5py.File(args.file, "r") as data:

        system = initsystem(data)
        eps_kick = data["/run/epsd/kick"][...]
        N = system.plastic().size

        # (*) Determine at which increment a push could be applied

        inc_system, inc_push = pushincrements(system, data, target_stress)

        # (*) Reload specific increment based on target stress and system-spanning increment

        assert target_inc_system in inc_system
        i = np.argmax(
            (target_inc_system == inc_system) * (target_inc_system <= inc_push)
        )
        inc = inc_push[i]
        assert target_inc_system == inc_system[i]

        system.setU(data[f"/disp/{inc:d}"])
        idx_n = system.plastic_CurrentIndex()
        system.addSimpleShearToFixedStress(target_stress)
        idx = system.plastic_CurrentIndex()
        assert np.all(idx == idx_n)

        # (*) Pin down a fraction of the system

        pinsystem(system, target_element, target_A)

    # (*) Apply push and minimise energy

    with h5py.File(args.output, "w") as output:

        output["/disp/0"] = system.u()
        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

        system.triggerElementWithLocalSimpleShear(eps_kick, target_element)
        niter = system.minimise()

        output["/disp/1"] = system.u()
        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

        print("done:", args.output, ", niter = ", niter)

        root = "/meta/PinAndTrigger"
        output[f"{root}/file"] = args.file
        output[f"{root}/version"] = myversion
        output[f"{root}/version_dependencies"] = model.version_dependencies()
        output[f"{root}/target_stress"] = target_stress
        output[f"{root}/target_inc_system"] = target_inc_system
        output[f"{root}/target_A"] = target_A
        output[f"{root}/target_element"] = target_element
        output[f"{root}/S"] = np.sum(idx - idx_n)
        output[f"{root}/A"] = np.sum(idx != idx_n)
