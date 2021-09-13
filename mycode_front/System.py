"""
-   Initialise system.
-   Write IO file.
-   Run simulation.
-   Get basic output.
"""
import os
import sys
import uuid
from typing import TypeVar

import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import prrng
import tqdm

from . import tag
from ._version import version


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


def read_epsy(file: h5py.File) -> np.ndarray:
    """
    Regenerate yield strain sequence per plastic element.
    Note that two ways of storage are supported:
    - "classical": the yield strains are stored.
    - "prrng": only the seeds per block are stored, that can then be unique restored.

    :param file: Opened simulation archive.
    """

    if isinstance(file["/cusp/epsy"], h5py.Dataset):
        return file["/cusp/epsy"][...]

    initstate = file["/cusp/epsy/initstate"][...]
    initseq = file["/cusp/epsy/initseq"][...]
    eps_offset = file["/cusp/epsy/eps_offset"][...]
    eps0 = file["/cusp/epsy/eps0"][...]
    k = file["/cusp/epsy/k"][...]
    nchunk = file["/cusp/epsy/nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= 2.0 * eps0
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


def init(file: h5py.File) -> model.System:
    r"""
    Read system from file.

    :param file: Open simulation HDF5 archive (read-only).
    :return: The initialised system.
    """

    system = model.System(
        file["coor"][...],
        file["conn"][...],
        file["dofs"][...],
        file["dofsP"][...] if "dofsP" in file else file["iip"][...],
        file["/elastic/elem"][...],
        file["/cusp/elem"][...],
    )

    system.setMassMatrix(file["rho"][...])
    system.setDampingMatrix(
        file["alpha"][...] if "alpha" in file else file["damping/alpha"][...]
    )
    system.setElastic(file["/elastic/K"][...], file["/elastic/G"][...])
    system.setPlastic(file["/cusp/K"][...], file["/cusp/G"][...], read_epsy(file))
    system.setDt(file["/run/dt"][...])

    return system


def reset_epsy(system: model.System, file: h5py.File):
    r"""
    Reset yield strain history from file.
    This can for example be used to speed-up things by avoiding re-initialising the system.

    :param system: The system (modified: yield strains changed).
    :param file: Open simulation HDF5 archive (read-only).
    """

    e = read_epsy(file)
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


def generate(
    filename: str, N: int, seed: int = 0, classic: bool = False, test_mode: bool = False
):
    """
    Generate input file.

    :param filename: The filename of the input file (overwritten).
    :param N: The number of blocks.
    :param seed: Base seed to use to generate the disorder.
    :param classic: The yield strain are hard-coded in the file, otherwise prrng is used.
    :param test_mode: Run in test mode (smaller chunk).
    """

    # parameters
    h = np.pi
    L = h * float(N)

    # define mesh and element sets
    mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)
    nelem = mesh.nelem()
    plastic = mesh.elementsMiddleLayer()
    elastic = np.setdiff1d(np.arange(nelem), plastic)

    # extract node sets to set the boundary conditions
    mesh.ndim()
    top = mesh.nodesTopEdge()
    bottom = mesh.nodesBottomEdge()
    left = mesh.nodesLeftOpenEdge()
    right = mesh.nodesRightOpenEdge()

    # periodicity in horizontal direction
    dofs = mesh.dofs()
    dofs[right, :] = dofs[left, :]
    dofs = GooseFEM.Mesh.renumber(dofs)

    # fixed top and bottom
    iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, :].ravel()))

    # yield strains
    k = 2.0
    eps0 = 1.0e-3 / 2.0
    eps_offset = 1.0e-5
    nchunk = 1000

    if classic:
        assert seed == 0  # at the moment seeding is not controlled locally
        realization = str(uuid.uuid4())
        epsy = eps_offset + 2.0 * eps0 * np.random.weibull(k, size=nchunk * N).reshape(
            N, -1
        )
        epsy[:, 0] = eps_offset + 2.0 * eps0 * np.random.random(N)
        epsy = np.cumsum(epsy, axis=1)
        i = np.min(np.where(np.min(epsy, axis=0) > 0.55)[0])
        epsy = epsy[:, :i]
    else:
        eps0 /= 10.0
        nchunk *= 6
        initstate = seed + np.arange(N).astype(np.int64)
        initseq = np.zeros_like(initstate)
        if test_mode:
            nchunk = 200

    # elasticity & damping
    c = 1.0
    G = 1.0
    K = 10.0 * G
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho
    dt = (1.0 / (c * qh)) / 10.0

    with h5py.File(filename, "w") as file:

        dump_with_atttrs(
            file,
            "/coor",
            mesh.coor(),
            desc="Nodal coordinates [nnode, ndim]",
        )

        dump_with_atttrs(
            file,
            "/conn",
            mesh.conn(),
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        dump_with_atttrs(
            file,
            "/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        dump_with_atttrs(
            file,
            "/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        dump_with_atttrs(
            file,
            "/run/epsd/kick",
            eps0 * 2e-4,
            desc="Strain kick to apply",
        )

        dump_with_atttrs(
            file,
            "/run/dt",
            dt,
            desc="Time step",
        )

        dump_with_atttrs(
            file,
            "/rho",
            rho * np.ones(nelem),
            desc="Mass density [nelem]",
        )

        dump_with_atttrs(
            file,
            "/alpha",
            alpha * np.ones(nelem),
            desc="Damping coefficient (density) [nelem]",
        )

        dump_with_atttrs(
            file,
            "/elastic/elem",
            elastic,
            desc="Elastic elements [nelem - N]",
        )

        dump_with_atttrs(
            file,
            "/elastic/K",
            K * np.ones(len(elastic)),
            desc="Bulk modulus for elements in '/elastic/elem' [nelem - N]",
        )

        dump_with_atttrs(
            file,
            "/elastic/G",
            G * np.ones(len(elastic)),
            desc="Shear modulus for elements in '/elastic/elem' [nelem - N]",
        )

        dump_with_atttrs(
            file,
            "/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        dump_with_atttrs(
            file,
            "/cusp/K",
            K * np.ones(len(plastic)),
            desc="Bulk modulus for elements in '/cusp/elem' [nplastic]",
        )

        dump_with_atttrs(
            file,
            "/cusp/G",
            G * np.ones(len(plastic)),
            desc="Shear modulus for elements in '/cusp/elem' [nplastic]",
        )

        if classic:

            file["/cusp/epsy"] = epsy
            file["/uuid"] = realization

        else:
            dump_with_atttrs(
                file,
                "/cusp/epsy/initstate",
                initstate,
                desc="State to initialise prrng.pcg32_array",
            )

            dump_with_atttrs(
                file,
                "/cusp/epsy/initseq",
                initseq,
                desc="Sequence to initialise prrng.pcg32_array",
            )

            dump_with_atttrs(
                file,
                "/cusp/epsy/k",
                k,
                desc="Shape factor of Weibull distribution",
            )

            dump_with_atttrs(
                file,
                "/cusp/epsy/eps0",
                eps0,
                desc="Normalisation: epsy(i + 1) - epsy(i) = 2.0 * eps0 * random + eps_offset",
            )

            dump_with_atttrs(
                file,
                "/cusp/epsy/eps_offset",
                eps_offset,
                desc="Offset, see eps0",
            )

            dump_with_atttrs(
                file,
                "/cusp/epsy/nchunk",
                nchunk,
                desc="Chunk size",
            )

        dump_with_atttrs(
            file,
            "/meta/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/l",
            h,
            desc="Elementary block size",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        dump_with_atttrs(
            file,
            "/meta/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        dump_with_atttrs(
            file,
            "/meta/seed_base",
            seed,
            desc="Basic seed == 'unique' identifier",
        )


def run(filename: str, dev: bool):
    """
    Run the simulation.

    :param filename: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    """

    basename = os.path.basename(filename)

    with h5py.File(filename, "a") as file:

        system = init(file)

        # check version compatibility

        assert dev or not tag.has_uncommited(version)
        assert dev or not tag.any_has_uncommited(model.version_dependencies())

        path = "/meta/Run/version"
        if version != "None":
            if path in file:
                assert tag.greater_equal(version, str(file[path].asstr()[...]))
            else:
                file[path] = version

        path = "/meta/Run/version_dependencies"
        if path in file:
            assert tag.all_greater_equal(
                model.version_dependencies(), file[path].asstr()[...]
            )
        else:
            file[path] = model.version_dependencies()

        if "/meta/Run/completed" in file or "/completed" in file:
            print("Marked completed, skipping")
            return 1

        # restore or initialise the this / output

        if "/stored" in file:

            inc = int(file["/stored"][-1])
            kick = file["/kick"][inc]
            system.setT(file["/t"][inc])
            system.setU(file[f"/disp/{inc:d}"][...])
            print(f'"{basename}": Loading, inc = {inc:d}')
            kick = not kick

        else:

            inc = int(0)
            kick = True

            dset = file.create_dataset(
                "/stored", (1,), maxshape=(None,), dtype=np.uint64
            )
            dset[0] = inc
            dset.attrs[
                "desc"
            ] = 'List of increments in "/disp/{:d}" and "/drive/ubar/{0:d}"'

            dset = file.create_dataset("/t", (1,), maxshape=(None,), dtype=np.float64)
            dset[0] = system.t()
            dset.attrs["desc"] = "Per increment: time at the end of the increment"

            dset = file.create_dataset(
                "/kick", (1,), maxshape=(None,), dtype=np.dtype(bool)
            )
            dset[0] = kick
            dset.attrs["desc"] = "Per increment: True is a kick was applied"

            file[f"/disp/{inc}"] = system.u()
            file[f"/disp/{inc}"].attrs[
                "desc"
            ] = "Displacement (at the end of the increment)."

        # run

        inc += 1
        deps_kick = file["/run/epsd/kick"][...]

        for inc in range(inc, sys.maxsize):

            system.addSimpleShearEventDriven(deps_kick, kick)

            if kick:
                niter = system.minimise_boundcheck()
                if niter == 0:
                    break
                print(f'"{basename}": inc = {inc:8d}, niter = {niter:8d}')

            dset_extend1d(file, "/stored", inc, inc)
            dset_extend1d(file, "/t", inc, system.t())
            dset_extend1d(file, "/kick", inc, kick)
            file[f"/disp/{inc:d}"] = system.u()

            inc += 1
            kick = not kick

        print(f'"{basename}": completed')
        file["/meta/Run/completed"] = 1


def basic_output(system: model.System, file: h5py.File) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print summary of every increment.
    """

    if "/meta/normalisation/N" in file:
        N = file["/meta/normalisation/N"][...]
        eps0 = file["/meta/normalisation/eps"][...]
        sig0 = file["/meta/normalisation/sig"][...]
    else:
        N = system.plastic().size
        G = 1.0
        eps0 = 1.0e-3 / 2.0
        sig0 = 2.0 * G * eps0

    dV = system.quad().AsTensor(2, system.quad().dV())
    incs = file["/stored"][...]
    ninc = incs.size
    assert np.all(incs == np.arange(ninc))
    idx_n = None

    ret = dict(
        Eps=np.empty((ninc), dtype=float),
        Sig=np.empty((ninc), dtype=float),
        S=np.zeros((ninc), dtype=int),
        A=np.zeros((ninc), dtype=int),
    )

    for inc in tqdm.tqdm(incs):

        system.setU(file[f"/disp/{inc:d}"][...])

        if idx_n is None:
            idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

        Sig = system.Sig() / sig0
        Eps = system.Eps() / eps0
        idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

        ret["S"][inc] = np.sum(idx - idx_n, axis=1)
        ret["A"][inc] = np.sum(idx != idx_n, axis=1)
        ret["Eps"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["Sig"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.array(idx, copy=True)

    return ret


def pushincrements(
    system: model.System, file: h5py.File, target_stress: float
) -> (np.ndarray, np.ndarray):
    r"""
    Get a list of increment from which the stress can be reached by elastic loading only.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param target_stress: The stress at which to push (in real units).
    :return:
        ``inc_system`` List of system spanning avalanches.
        ``inc_push`` List of increment from which the stress can be reached by elastic loading only.
    """

    plastic = system.plastic()
    N = plastic.size
    dV = system.quad().AsTensor(2, system.quad().dV())
    kick = file["/kick"][...].astype(bool)
    incs = file["/stored"][...].astype(int)
    assert np.all(incs == np.arange(incs.size))
    assert kick.shape == incs.shape
    assert np.all(not kick[::2])
    assert np.all(kick[1::2])

    A = np.zeros(incs.shape, dtype=int)
    Strain = np.zeros(incs.shape, dtype=float)
    Stress = np.zeros(incs.shape, dtype=float)

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

    for inc in incs:

        system.setU(file[f"/disp/{inc:d}"])

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
