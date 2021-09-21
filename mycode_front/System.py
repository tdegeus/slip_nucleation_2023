"""
-   Initialise system.
-   Write IO file.
-   Run simulation.
-   Get basic output.
"""
import argparse
import os
import sys
import textwrap
import uuid

import click
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import prrng
import tqdm
from numpy.typing import ArrayLike

from . import slurm
from . import storage
from . import tag
from ._version import version


def read_epsy(file: h5py.File) -> np.ndarray:
    """
    Regenerate yield strain sequence per plastic element.
    Note that two ways of storage are supported:
    -   "classical": the yield strains are stored.
    -   "prrng": only the seeds per block are stored, that can then be uniquely restored.
        Note that in this case a larger strain history is used.

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
    Initialise system from file.

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


def generate(
    filename: str, N: int, seed: int = 0, classic: bool = False, test_mode: bool = False
):
    """
    Generate input file.
    Note that two ways of storage of yield strains are supported:
    -   "classical": the yield strains are stored.
    -   "prrng": only the seeds per block are stored, that can then be uniquely restored.
        Note that in this case a larger strain history is used.

    :param filename: The filename of the input file.
    :param N: The number of blocks.
    :param seed: Base seed to use to generate the disorder.
    :param classic: The yield strain are hard-coded in the file, otherwise prrng is used.
    :param test_mode: Run in test mode (smaller chunk).
    """

    assert not os.path.isfile(os.path.realpath(filename))

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

        storage.dump_with_atttrs(
            file,
            "/coor",
            mesh.coor(),
            desc="Nodal coordinates [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/conn",
            mesh.conn(),
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        storage.dump_with_atttrs(
            file,
            "/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        storage.dump_with_atttrs(
            file,
            "/run/epsd/kick",
            eps0 * 2e-4,
            desc="Strain kick to apply",
        )

        storage.dump_with_atttrs(
            file,
            "/run/dt",
            dt,
            desc="Time step",
        )

        storage.dump_with_atttrs(
            file,
            "/rho",
            rho * np.ones(nelem),
            desc="Mass density [nelem]",
        )

        storage.dump_with_atttrs(
            file,
            "/alpha",
            alpha * np.ones(nelem),
            desc="Damping coefficient (density) [nelem]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/elem",
            elastic,
            desc="Elastic elements [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/K",
            K * np.ones(len(elastic)),
            desc="Bulk modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/G",
            G * np.ones(len(elastic)),
            desc="Shear modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/K",
            K * np.ones(len(plastic)),
            desc="Bulk modulus for elements in '/cusp/elem' [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/G",
            G * np.ones(len(plastic)),
            desc="Shear modulus for elements in '/cusp/elem' [nplastic]",
        )

        if classic:

            file["/cusp/epsy"] = epsy
            file["/uuid"] = realization

        else:
            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/initstate",
                initstate,
                desc="State to initialise prrng.pcg32_array",
            )

            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/initseq",
                initseq,
                desc="Sequence to initialise prrng.pcg32_array",
            )

            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/k",
                k,
                desc="Shape factor of Weibull distribution",
            )

            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/eps0",
                eps0,
                desc="Normalisation: epsy(i + 1) - epsy(i) = 2.0 * eps0 * random + eps_offset",
            )

            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/eps_offset",
                eps_offset,
                desc="Offset, see eps0",
            )

            storage.dump_with_atttrs(
                file,
                "/cusp/epsy/nchunk",
                nchunk,
                desc="Chunk size",
            )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/l",
            h,
            desc="Elementary block size",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/seed_base",
            seed,
            desc="Basic seed == 'unique' identifier",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/Run/version",
            version,
            desc="Version when generating",
        )


def cli_generate(cli_args=None):
    """
    Generate IO files, including job-scripts to run simulations.
    """

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(cli_generate.__doc__),
    )

    parser.add_argument(
        "outdir",
        type=str,
        help="Output directory",
    )

    parser.add_argument(
        "-N",
        "--size",
        type=int,
        default=2 * (3 ** 6),
        help="Number of plastic blocks",
    )

    parser.add_argument(
        "-n",
        "--nsim",
        type=int,
        default=1,
        help="Number of simulations to generate",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Starting simulation (sets the seed)",
    )

    parser.add_argument(
        "-w",
        "--time",
        type=str,
        default="72h",
        help="Walltime to allocate for the job",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )

    args = parser.parse_args(cli_args)

    assert os.path.isdir(os.path.realpath(args.outdir))

    files = []

    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:03d}.h5"]
        generate(
            filename=os.path.join(args.outdir, f"id={i:03d}.h5"),
            N=args.size,
            seed=i * args.size,
        )

    commands = [f"Run {file}" for file in files]
    slurm.serial_group(
        commands,
        basename="Run",
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
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

        # restore or initialise the system / output

        if "/stored" in file:

            inc = int(file["/stored"][-1])
            kick = file["/kick"][inc]
            system.setT(file["/t"][inc])
            system.setU(file[f"/disp/{inc:d}"][...])
            print(f'"{basename}": Loading, inc = {inc:d}')

        else:

            inc = int(0)
            kick = False

            storage.dset_extendible1d(
                file=file,
                key="/stored",
                dtype=np.uint64,
                value=inc,
                desc="List of stored increments",
            )

            storage.dset_extendible1d(
                file=file,
                key="/t",
                dtype=np.float64,
                value=system.t(),
                desc="Per increment: time at the end of the increment",
            )

            storage.dset_extendible1d(
                file=file,
                key="/kick",
                dtype=np.dtype(bool),
                value=kick,
                desc="Per increment: True is a kick was applied",
            )

            storage.dump_with_atttrs(
                file=file,
                key=f"/disp/{inc}",
                data=system.u(),
                desc="Displacement (end of increment).",
            )

        # run

        inc += 1
        kick = not kick
        deps_kick = file["/run/epsd/kick"][...]

        for inc in range(inc, sys.maxsize):

            system.addSimpleShearEventDriven(deps_kick, kick)

            if kick:
                niter = system.minimise_boundcheck()
                if niter == 0:
                    break
                print(f'"{basename}": inc = {inc:8d}, niter = {niter:8d}')

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            storage.dset_extend1d(file, "/kick", inc, kick)
            file[f"/disp/{inc:d}"] = system.u()

            kick = not kick

        print(f'"{basename}": completed')
        file["/meta/Run/completed"] = 1


def cli_run(cli_args=None):
    """
    Run simulation.
    """

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(cli_run.__doc__),
    )

    parser.add_argument(
        "file",
        type=str,
        help="Simulation file",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Allow uncommitted changed",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))
    run(args.file, dev=args.force)


def steadystate(epsd: ArrayLike, sigd: ArrayLike, kick: ArrayLike, **kwargs):
    """
    Estimate the first increment of the steady-state, with additional constraints:
    -   Skip at least two increments.
    -   Start with elastic loading.

    :param epsd: Strain history [ninc].
    :param sigd: Stress history [ninc].
    :param kick: Per increment, skip or not [ninc].
    :return: Increment number.
    """

    K = np.empty_like(sigd)
    K[0] = np.inf
    K[1:] = (sigd[1:] - sigd[0]) / (epsd[1:] - epsd[0])

    steadystate = max(2, np.argmax(K <= 0.95 * K[1]))

    if kick[steadystate]:
        steadystate += 1

    return steadystate


def basic_output(system: model.System, file: h5py.File, verbose: bool = True) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.
    """

    incs = file["/stored"][...]
    ninc = incs.size
    assert np.all(incs == np.arange(ninc))

    ret = dict(
        epsd=np.empty((ninc), dtype=float),
        sigd=np.empty((ninc), dtype=float),
        S=np.zeros((ninc), dtype=int),
        A=np.zeros((ninc), dtype=int),
        kick=file["/kick"][...].astype(bool),
        inc=incs,
    )

    # read normalisation
    if "/meta/normalisation/N" in file:
        N = file["/meta/normalisation/N"][...]
        eps0 = file["/meta/normalisation/eps"][...]
        sig0 = file["/meta/normalisation/sig"][...]
        ret["l0"] = file["/meta/normalisation/l"][...]
        ret["G"] = file["/meta/normalisation/G"][...]
        ret["K"] = file["/meta/normalisation/K"][...]
        ret["rho"] = file["/meta/normalisation/rho"][...]
        ret["seed"] = file["/meta/seed_base"][...]
    else:
        N = system.plastic().size
        G = 1.0
        eps0 = 1.0e-3 / 2.0
        sig0 = 2.0 * G * eps0
        ret["l0"] = np.pi
        ret["G"] = G
        ret["K"] = 10.0 * G
        ret["rho"] = G / 1.0 ** 2.0
        ret["seed"] = str(file["/uuid"].asstr()[...])

    # interpret / store additional normalisation
    kappa = ret["K"] / 3.0
    mu = ret["G"] / 2.0
    ret["cs"] = np.sqrt(mu / ret["rho"])
    ret["cd"] = np.sqrt((kappa + 4.0 / 3.0 * mu) / ret["rho"])
    ret["sig0"] = sig0
    ret["eps0"] = eps0
    ret["N"] = N
    ret["t0"] = ret["l0"] / ret["cs"]
    ret["dt"] = file["/run/dt"][...]

    dV = system.quad().AsTensor(2, system.quad().dV())
    idx_n = None

    for inc in tqdm.tqdm(incs, disable=not verbose):

        system.setU(file[f"/disp/{inc:d}"][...])

        if idx_n is None:
            idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

        Sig = system.Sig() / sig0
        Eps = system.Eps() / eps0
        idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

        ret["S"][inc] = np.sum(idx - idx_n, axis=1)
        ret["A"][inc] = np.sum(idx != idx_n, axis=1)
        ret["epsd"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sigd"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.array(idx, copy=True)

    # estimate steady-state
    ret["steadystate"] = steadystate(**ret)

    return ret


def cli_ensembleinfo(cli_args=None):
    """
    Read information (avalanche size, stress, strain, ...) of an ensemble, and combine into
    a single output file.
    """

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(cli_ensembleinfo.__doc__),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="EnsembleInfo.h5",
        help="Output file",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of output file if it exists",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )

    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    files = [
        os.path.normpath(os.path.relpath(file, os.path.dirname(args.output)))
        for file in args.files
    ]
    seeds = []

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    fields_norm = ["l0", "G", "K", "rho", "cs", "cd", "sig0", "eps0", "N", "t0", "dt"]
    fields_full = ["epsd", "sigd", "S", "A", "kick", "inc", "steadystate"]
    combine_load = {key: [] for key in ["epsd", "sigd", "S", "A", "kick", "inc"]}
    combine_kick = {key: [] for key in ["epsd", "sigd", "S", "A", "kick", "inc"]}
    file_load = []
    file_kick = []

    with h5py.File(args.output, "w") as output:

        for i, (filename, filepath) in enumerate(tqdm.tqdm(zip(files, args.files))):

            with h5py.File(filepath, "r") as file:

                if i == 0:
                    system = init(file)
                else:
                    system.reset_epsy(read_epsy(file))

                out = basic_output(system, file, verbose=False)
                assert np.all(out["kick"][1::2])
                assert not np.any(out["kick"][::2])
                kick = np.array(out["kick"], copy=True)
                load = np.logical_not(out["kick"])
                kick[: out["steadystate"]] = False
                load[: out["steadystate"]] = False

                if np.sum(load) > np.sum(kick):
                    load[-1] = False

                assert np.sum(load) == np.sum(kick)

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]

                for key in combine_load:
                    combine_load[key] += list(out[key][load])
                    combine_kick[key] += list(out[key][kick])

                file_load += list(i * np.ones(np.sum(load), dtype=int))
                file_kick += list(i * np.ones(np.sum(kick), dtype=int))
                seeds += [out["seed"]]

                if i == 0:
                    norm = {key: out[key] for key in fields_norm}
                else:
                    for key in fields_norm:
                        assert np.isclose(norm[key], out[key])

        combine_load["file"] = np.array(file_load, dtype=np.uint64)
        combine_kick["file"] = np.array(file_kick, dtype=np.uint64)

        for key in ["A", "inc"]:
            combine_load[key] = np.array(combine_load[key], dtype=np.uint64)
            combine_kick[key] = np.array(combine_kick[key], dtype=np.uint64)

        for key in ["epsd", "sigd"]:
            combine_load[key] = np.array(combine_load[key])
            combine_kick[key] = np.array(combine_kick[key])

        for key in combine_load:
            output[f"/loading/{key}"] = combine_load[key]
            output[f"/avalanche/{key}"] = combine_kick[key]

        for key, value in norm.items():
            output[f"/normalisation/{key}"] = value

        ss = np.equal(combine_kick["A"], norm["N"])
        assert np.all(np.equal(combine_kick["inc"][ss], combine_load["inc"][ss] + 1))
        output["/averages/sigd_top"] = np.mean(combine_load["sigd"][ss])
        output["/averages/sigd_bottom"] = np.mean(combine_kick["sigd"][ss])

        output["files"] = files
        output["seeds"] = seeds


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
    kick = file["/kick"][...].astype(bool)
    incs = file["/stored"][...].astype(int)
    assert np.all(incs == np.arange(incs.size))
    assert kick.shape == incs.shape
    assert np.all(np.logical_not(kick[::2]))
    assert np.all(kick[1::2])

    output = basic_output(system, file)
    Stress = output["sigd"] * output["sig0"]
    A = output["A"]
    A[: output["steadystate"]] = 0

    inc_system = np.argwhere(A == N).ravel()
    inc_push = []
    inc_system_ret = []

    for ii, jj in zip(inc_system[:-1], inc_system[1:]):

        # state after elastic loading (before kick)
        i = ii + 1
        s = Stress[i:jj:2]
        n = incs[i:jj:2]

        if not np.any(s > target_stress) or Stress[ii] > target_stress:
            continue

        ipush = n[np.argmax(s > target_stress)] - 1

        assert Stress[ipush] <= target_stress
        assert not kick[ipush + 1]

        inc_push += [ipush]
        inc_system_ret += [n[0] - 1]

    inc_push = np.array(inc_push)
    inc_system_ret = np.array(inc_system_ret)

    return inc_system_ret, inc_push
