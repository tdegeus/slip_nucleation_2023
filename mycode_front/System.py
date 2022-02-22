"""
-   Initialise system.
-   Write IO file.
-   Run simulation.
-   Get basic output.
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import textwrap
import uuid

import FrictionQPotFEM  # noqa: F401
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot  # noqa: F401
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import matplotlib.pyplot as plt
import numpy as np
import prrng
import tqdm
from numpy.typing import ArrayLike

from . import MeasureDynamics
from . import slurm
from . import storage
from . import tag
from . import tools
from ._version import version

plt.style.use(["goose", "goose-latex"])


entry_points = dict(
    cli_ensembleinfo="EnsembleInfo",
    cli_generate="Run_generate",
    cli_run="Run",
    cli_plot="Run_plot",
    cli_rerun_event="RunEventMap",
    cli_rerun_event_job_systemspanning="RunEventMap_JobAllSystemSpanning",
    cli_rerun_event_collect="EventMapInfo",
    cli_rerun_dynamics_job_systemspanning="RunDynamics_JobAllSystemSpanning",
)


file_defaults = dict(
    cli_ensembleinfo="EnsembleInfo.h5",
    cli_rerun_event="EventMap.h5",
    cli_rerun_event_job_systemspanning="EventMap_SystemSpanning",
    cli_rerun_event_collect="EventMapInfo.h5",
    cli_rerun_dynamics_job_systemspanning="RunDynamics_SystemSpanning",
)


def dependencies(system: model.System) -> list[str]:
    """
    Return list with version strings.
    Compared to model.System.version_dependencies() this adds the version of prrng.
    """
    return sorted(list(model.version_dependencies()) + ["prrng=" + prrng.version()])


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def interpret_filename(filename: str) -> dict:
    """
    Split filename in useful information.
    """

    part = re.split("_|/", os.path.splitext(filename)[0])
    info = {}

    for i in part:
        key, value = i.split("=")
        info[key] = value

    for key in info:
        info[key] = int(info[key])

    return info


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


def epsy_nchunk(file: h5py.File) -> int:
    """
    Return the size of the chunk of yield strains stored.
    :param file: Opened simulation archive.
    :return: Size.
    """

    if isinstance(file["/cusp/epsy"], h5py.Dataset):
        return file["/cusp/epsy"].shape[1]

    return file["/cusp/epsy/nchunk"][...]


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
    system.setDampingMatrix(file["alpha"][...] if "alpha" in file else file["damping/alpha"][...])
    system.setElastic(file["/elastic/K"][...], file["/elastic/G"][...])
    system.setPlastic(file["/cusp/K"][...], file["/cusp/G"][...], read_epsy(file))

    system.setDt(file["/run/dt"][...])

    return system


def clone(source: h5py.File, dest: h5py.File, skip: list[str] = None) -> list[str]:
    """
    Clone a configuration.
    This clone does not include::

        /stored
        /t
        /kick
        /disp/...

    :param source: Source file.
    :param dest: Destination file.
    :parma skip: List with additional dataset to skip.
    :return: List of copied datasets.
    """

    datasets = list(g5.getdatasets(source, fold="/disp"))
    groups = list(g5.getgroups(source, has_attrs=True))

    for key in ["/stored", "/t", "/kick", "/disp/..."]:
        datasets.remove(key)

    for key in ["/disp"]:
        groups.remove("/disp")

    ret = datasets + groups

    if skip:
        for key in skip:
            ret.remove(key)

    g5.copy(source, dest, ret)

    return ret


def _init_run_state(file: h5py.File):
    """
    Initialise as extendible datasets:
    *   ``"/stored"``: stored increments.
    *   ``"/t"``: time at the end of the increment.
    *   ``"/kick"``: kick setting of the increment.

    Furthermore, add a description to ``"/disp"``.
    """

    desc = 'One entry per item in "/stored".'
    storage.create_extendible(file, "/stored", np.uint64, desc="List of stored increments")
    storage.create_extendible(file, "/t", np.float64, desc=f"Time (end of increment). {desc}")
    storage.create_extendible(file, "/kick", bool, desc=f"Kick used. {desc}")

    storage.dset_extend1d(file, "/stored", 0, 0)
    storage.dset_extend1d(file, "/t", 0, 0.0)
    storage.dset_extend1d(file, "/kick", 0, False)

    if "disp" not in file:
        file.create_group("/disp")

    file["/disp"].attrs["desc"] = f"Displacement {desc}"


def branch_fixed_stress(
    source: h5py.File,
    dest: h5py.File,
    inc: int = None,
    incc: int = None,
    stress: float = None,
    normalised: bool = False,
    system: model.System = None,
    init_system: bool = False,
    init_dest: bool = True,
    output: ArrayLike = None,
    dev: bool = False,
):
    """
    Branch a configuration at a given:
    *   Increment (``inc``).
        ``incc`` and ``stress`` are ignored if supplied, but are stored as meta-data in that case.
        To ensure meta-data integrity,
        they are only checked to be consistent with the state at ``inc``.
    *   Fixed stress after a system spanning event (``incc`` and ``stress``).
        Note that ``stress`` is approximated as best as possible,
        and its actual value is stored in the meta-data.

    :param source: Source file.
    :param dest: Destination file.
    :param inc: Branch at specific increment.
    :param incc: Branch at fixed stress after this system-spanning event.
    :param stress: Branch at fixed stress.
    :param normalised: Assume ``stress`` to be normalised (see ``sig0`` in :py:func:`basic_output`).
    :param system: The system (optional, specify to avoid reallocation).
    :param init_system: Read yield strains (otherwise ``system`` is assumed fully initialised).
    :param init_dest: Initialise ``dest`` from ``source``.
    :param output: Output of :py:func:`basic_output` (read if not specified).
    :param dev: Allow uncommitted changes.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function

    if system is None:
        system = init(source)
    elif init_system:
        system.reset_epsy(read_epsy(source))

    if output is None:
        output = basic_output(system, source, verbose=False)

    if init_dest:
        clone(source, dest)
        _init_run_state(dest)

    if not normalised:
        stress /= output["sig0"]

    load_inc = None

    # determine at which increment a push could be applied
    if inc is None:

        assert incc is not None and stress is not None

        jncc = int(incc) + np.argwhere(output["A"][incc:] == output["N"]).ravel()[1]
        assert (jncc - incc) % 2 == 0
        i = output["inc"][incc:jncc].reshape(-1, 2)
        s = output["sigd"][incc:jncc].reshape(-1, 2)
        k = output["kick"][incc:jncc].reshape(-1, 2)
        t = np.sum(s < stress, axis=1)
        assert np.all(k[:, 0])
        assert np.sum(t) > 0

        if np.sum(t == 1) >= 1:
            j = np.argmax(t == 1)
            if np.abs(s[j, 0] - stress) / s[j, 0] < 1e-4:
                inc = i[j, 0]
                stress = s[j, 0]
            elif np.abs(s[j, 1] - stress) / s[j, 1] < 1e-4:
                inc = i[j, 1]
                stress = s[j, 1]
            else:
                load_inc = int(i[j, 0])
        else:
            j = np.argmax(t == 0)
            inc = i[j, 0]
            stress = s[j, 0]

    # restore specific increment
    if inc is not None:

        dest["/disp/0"] = source[f"/disp/{inc:d}"][...]

        if stress is not None:
            assert np.isclose(stress, output["sigd"][inc])
        if incc is not None:
            i = output["inc"][output["A"] == output["N"]]
            assert i[i >= incc][1] > inc

    # apply elastic loading to reach a specific stress
    if load_inc is not None:

        system.setU(source[f"/disp/{load_inc:d}"])
        idx_n = system.plastic_CurrentIndex()
        d = system.addSimpleShearToFixedStress(stress * output["sig0"])
        idx = system.plastic_CurrentIndex()
        assert np.all(idx == idx_n)
        assert d >= 0.0

        dest["/disp/0"] = system.u()

    assert f"/meta/{funcname}" not in dest
    meta = create_check_meta(dest, f"/meta/{funcname}", dev=dev)
    meta.attrs["file"] = os.path.basename(source.filename)
    if stress is not None:
        meta.attrs["stress"] = stress
    if incc is not None:
        meta.attrs["incc"] = int(incc)
    if inc is not None:
        meta.attrs["inc"] = int(inc)


def generate(
    filepath: str,
    N: int,
    seed: int = 0,
    init_run: bool = True,
    classic: bool = False,
    test_mode: bool = False,
    dev: bool = False,
):
    """
    Generate input file. See :py:func:`read_epsy` for different strategies to store yield strains.

    :param filepath: The filepath of the input file.
    :param N: The number of blocks.
    :param seed: Base seed to use to generate the disorder.
    :param init_run: Initialise for use with :py:func:`run`.
    :param classic: The yield strain are hard-coded in the file, otherwise prrng is used.
    :param test_mode: Run in test mode (smaller chunk).
    :param dev: Allow uncommitted changes.
    """

    assert test_mode or not tag.has_uncommitted(version)

    assert not os.path.isfile(filepath)
    progname = entry_points["cli_generate"]

    # parameters
    h = np.pi
    L = h * float(N)

    # define mesh and element sets
    mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)
    coor = mesh.coor()
    conn = mesh.conn()
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
        epsy = eps_offset + 2.0 * eps0 * np.random.weibull(k, size=nchunk * N).reshape(N, -1)
        epsy[:, 0] = eps_offset + 2.0 * eps0 * np.random.random(N)
        epsy = np.cumsum(epsy, axis=1)
        nchunk = np.min(np.where(np.min(epsy, axis=0) > 0.55)[0])
        if test_mode:
            nchunk = 200
        epsy = epsy[:, :nchunk]
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
    rho = G / c**2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho
    dt = (1.0 / (c * qh)) / 10.0

    with h5py.File(filepath, "w") as file:

        storage.dump_with_atttrs(
            file,
            "/coor",
            coor,
            desc="Nodal coordinates [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/conn",
            conn,
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

        create_check_meta(file, f"/meta/{progname}", dev=dev)

        if init_run:

            storage.dump_with_atttrs(
                file,
                "/run/epsd/kick",
                eps0 * 2e-4,
                desc="Strain kick to apply",
            )

            assert np.min(np.diff(read_epsy(file), axis=1)) > file["/run/epsd/kick"][...]

            file["/disp/0"] = np.zeros_like(coor)
            _init_run_state(file)


def cli_generate(cli_args=None):
    """
    Generate IO files (including job-scripts) to run simulations.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=2 * (3**6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="72h", help="Walltime")
    parser.add_argument("outdir", type=str, help="Output directory")

    args = tools._parse(parser, cli_args)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    files = []

    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:03d}.h5"]
        generate(
            filepath=os.path.join(args.outdir, f"id={i:03d}.h5"),
            N=args.size,
            seed=i * args.size,
            test_mode=args.develop,
            dev=args.develop,
        )

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in files]
    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def create_check_meta(
    file: h5py.File,
    path: str,
    ver: str = version,
    deps: str = dependencies(model),
    dev: bool = False,
) -> h5py.Group:
    """
    Create or read/check metadata. This function asserts that:

    -   There are no uncommitted changes.
    -   There are no version changes.

    It create metadata as attributes to a group ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations if needed.
        "version": The current version of this code (see below).
        "dependencies": The current version of all relevant dependencies (see below).

    :param file: HDF5 archive.
    :param path: Path in ``file`` to store/read metadata.
    :param ver: Version string.
    :param deps: List of dependencies.
    :param dev: Allow uncommitted changes.
    :return: Group to metadata.
    """

    assert dev or not tag.has_uncommitted(ver)
    assert dev or not tag.any_has_uncommitted(deps)

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["version"] = ver
        meta.attrs["dependencies"] = deps
        return meta

    meta = file[path]
    assert dev or tag.equal(ver, meta.attrs["version"])
    assert dev or tag.all_equal(deps, meta.attrs["dependencies"])
    return meta


def _restore_inc(file: h5py.File, system: model.System, inc: int):
    """
    Restore an increment.

    :param file: Open simulation HDF5 archive (read-only).
    :param system: The system,
    :param inc: Increment number.
    """

    system.quench()
    system.setT(file["/t"][inc])
    system.setU(file[f"/disp/{inc:d}"][...])


def run(filepath: str, dev: bool = False, progress: bool = True):
    """
    Run the simulation.

    :param filepath: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    :param progress: Show progress bar.
    """

    assert os.path.isfile(filepath)
    basename = os.path.basename(filepath)
    progname = entry_points["cli_run"]

    with h5py.File(filepath, "a") as file:

        system = init(file)
        meta = create_check_meta(file, f"/meta/{progname}", dev=dev)

        if "completed" in meta:
            print(f"{basename}: marked completed, skipping")
            return 1

        inc = int(file["/stored"][-1])
        kick = file["/kick"][inc]
        deps = file["/run/epsd/kick"][...]
        _restore_inc(file, system, inc)

        system.initEventDrivenSimpleShear()

        nchunk = epsy_nchunk(file) - 5
        pbar = tqdm.tqdm(total=nchunk, disable=not progress)
        pbar.set_description(f"{basename}: inc = {inc:8d}, niter = {'-':8s}")

        for inc in range(inc + 1, sys.maxsize):

            kick = not kick
            system.eventDrivenStep(deps, kick)

            if kick:

                niter = system.minimise_boundcheck(5)

                if niter == 0:
                    break

                if progress:
                    pbar.n = np.max(system.plastic_CurrentIndex())
                    pbar.set_description(f"{basename}: inc = {inc:8d}, niter = {niter:8d}")
                    pbar.refresh()

            if not kick:
                if not system.boundcheck_right(5):
                    break

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            storage.dset_extend1d(file, "/kick", inc, kick)
            file[f"/disp/{inc:d}"] = system.u()
            file.flush()

        pbar.set_description(f"{basename}: inc = {inc:8d}, {'completed':16s}")
        meta.attrs["completed"] = 1


def cli_run(cli_args=None):
    """
    Run simulation.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")
    args = tools._parse(parser, cli_args)
    run(args.file, dev=args.develop)


def steadystate(
    epsd: ArrayLike, sigd: ArrayLike, kick: ArrayLike, A: ArrayLike, N: int, **kwargs
) -> int:
    """
    Estimate the first increment of the steady-state. Constraints:
    -   Start with elastic loading.
    -   Sufficiently low tangent modulus.
    -   All blocks yielded at least once.

    .. note::

        Keywords arguments that are not explicitly listed are ignored.

    :param epsd: Macroscopic strain [ninc].
    :param sigd: Macroscopic stress [ninc].
    :param kick: Whether a kick was applied [ninc].
    :param A: Number of blocks that yielded at least once [ninc].
    :param N: Number of blocks.
    :return: Increment number.
    """

    if sigd.size <= 2:
        return None

    tangent = np.empty_like(sigd)
    tangent[0] = np.inf
    tangent[1:] = (sigd[1:] - sigd[0]) / (epsd[1:] - epsd[0])

    i_yield = np.argmax(A == N)
    i_tangent = np.argmax(tangent <= 0.95 * tangent[1])
    steadystate = max(i_yield + 1, i_tangent)

    if i_yield == 0 or i_tangent == 0:
        return None

    if steadystate >= kick.size - 1:
        return None

    if kick[steadystate]:
        steadystate += 1

    return steadystate


def interface_state(filepaths: dict[int], read_disp: dict[str] = None) -> dict[np.ndarray]:
    """
    State of the interface at one or several increments per realisation.

    :oaram filepaths:
        Dictionary with a list of increments to consider per file, e.g.
        ``{"id=0.h5": [10, 20, 30], "id=1.h5": [20, 30, 40]}``

    :oaram read_disp:
        Dictionary with a file-paths to read the displacement field from, e.g.
        ``{"id=0.h5": "mydisplacement.h5"}``, whereby the increment numbers should correspond
        to those in ``mydisplacement.h5``.
        By default the displacement is read from the same files as the realisation.

    :return:
        A dictionary with per field a matrix of shape ``[n, N]``,
        with each row corresponding to an increment of a file,
        and the columns corresponding to the spatial distribution.
        The output consists of the following::
            sig_xx: xx-component of the average stress tensor of a block.
            sig_xy: xy-component of the average stress tensor of a block.
            sig_yy: yy-component of the average stress tensor of a block.
            eps_xx: xx-component of the average strain tensor of a block.
            eps_xy: xy-component of the average strain tensor of a block.
            eps_yy: yy-component of the average strain tensor of a block.
            epsp: Average plastic strain of a block.
            S: The number of times the first integration point of a block yielded.
    """

    for filepath in filepaths:
        if isinstance(filepaths[filepath], int):
            filepaths[filepath] = [filepaths[filepath]]

    if read_disp:
        for filepath in read_disp:
            if isinstance(read_disp[filepath], str):
                read_disp[filepath] = [read_disp[filepath]]
            assert len(read_disp[filepath]) == len(filepaths[filepath])

    n = sum(len(filepaths[filepath]) for filepath in filepaths)
    i = 0

    for filepath in tqdm.tqdm(filepaths):

        with h5py.File(filepath, "r") as file:

            if i == 0:
                system = init(file)
                plastic = system.plastic()
                N = plastic.size
                dV = system.quad().dV()[plastic, :]
                dV2 = system.quad().AsTensor(2, dV)
                ret = {
                    "sig_xx": np.empty((n, N), dtype=float),
                    "sig_xy": np.empty((n, N), dtype=float),
                    "sig_yy": np.empty((n, N), dtype=float),
                    "eps_xx": np.empty((n, N), dtype=float),
                    "eps_xy": np.empty((n, N), dtype=float),
                    "eps_yy": np.empty((n, N), dtype=float),
                    "epsp": np.empty((n, N), dtype=float),
                    "S": np.empty((n, N), dtype=int),
                }
            else:
                system.reset_epsy(read_epsy(file))

            for j, inc in enumerate(filepaths[filepath]):

                if read_disp:
                    with h5py.File(read_disp[filepath][j], "r") as disp:
                        system.setU(disp[f"/disp/{inc:d}"][...])
                else:
                    system.setU(file[f"/disp/{inc:d}"][...])

                Sig = np.average(system.plastic_Sig(), weights=dV2, axis=1)
                Eps = np.average(system.plastic_Eps(), weights=dV2, axis=1)
                ret["sig_xx"][i, :] = Sig[:, 0, 0]
                ret["sig_xy"][i, :] = Sig[:, 0, 1]
                ret["sig_yy"][i, :] = Sig[:, 1, 1]
                ret["eps_xx"][i, :] = Eps[:, 0, 0]
                ret["eps_xy"][i, :] = Eps[:, 0, 1]
                ret["eps_yy"][i, :] = Eps[:, 1, 1]
                ret["epsp"][i, :] = np.average(system.plastic_Epsp(), weights=dV, axis=1)
                ret["S"][i, :] = system.plastic_CurrentIndex()[:, 0]
                i += 1

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def normalisation(file: h5py.File):
    """
    Read normalisation from file (or use default value in "classic" mode).

    :param file: Open simulation HDF5 archive (read-only).
    :return: Basic information as follows::
        l0: Block size (float).
        G: Shear modulus (float).
        K: Bulk modulus (float).
        rho: Mass density (float).
        seed: Base seed (uint64) or uuid (str).
        cs: Shear wave speed (float)
        cd: Longitudinal wave speed (float).
        eps0: Typical yield strain (float).
        sig0: Typical yield stress (float).
        N: Number of blocks (int).
        t0: Unit of time == l0 / cs (float).
        dt: Time step of time discretisation.
    """

    ret = {}

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
        N = file["/cusp/epsy"].shape[0]
        G = 1.0
        eps0 = 1.0e-3 / 2.0
        sig0 = 2.0 * G * eps0
        ret["l0"] = np.pi
        ret["G"] = G
        ret["K"] = 10.0 * G
        ret["rho"] = G / 1.0**2.0
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

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def basic_output(
    system: model.System, file: h5py.File, norm: dict = None, verbose: bool = True
) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param norm: Normalisation, see :py:func:`normalisation` (read if not specified).
    :param verbose: Print progress.

    :return: Basic information as follows::
        epsd: Macroscopic strain [ninc].
        sigd: Macroscopic stress [ninc].
        S: Number of times a block yielded [ninc].
        A: Number of blocks that yielded at least once [ninc].
        xi: Largest extension corresponding to A [ninc].
        duration: Duration of the event [ninc].
        kick: Increment started with a kick (True), or contains only elastic loading (False) [ninc].
        inc: Increment numbers == np.arange(ninc).
        steadystate: Increment number where the steady state starts (int).
        l0: Block size (float).
        G: Shear modulus (float).
        K: Bulk modulus (float).
        rho: Mass density (float).
        seed: Base seed (uint64) or uuid (str).
        cs: Shear wave speed (float)
        cd: Longitudinal wave speed (float).
        eps0: Typical yield strain (float).
        sig0: Typical yield stress (float).
        N: Number of blocks (int).
        t0: Unit of time == l0 / cs (float).
        dt: Time step of time discretisation.
    """

    incs = file["/stored"][...]
    ninc = incs.size
    assert all(incs == np.arange(ninc))

    if norm is None:
        ret = normalisation(file)
    else:
        ret = dict(norm)

    ret["epsd"] = np.empty((ninc), dtype=float)
    ret["sigd"] = np.empty((ninc), dtype=float)
    ret["S"] = np.zeros((ninc), dtype=int)
    ret["A"] = np.zeros((ninc), dtype=int)
    ret["xi"] = np.zeros((ninc), dtype=int)
    ret["kick"] = file["/kick"][...].astype(bool)
    ret["inc"] = incs

    dV = system.quad().AsTensor(2, system.quad().dV())
    idx_n = None

    for inc in tqdm.tqdm(incs, disable=not verbose):

        system.setU(file[f"/disp/{inc:d}"][...])

        if idx_n is None:
            idx_n = system.plastic_CurrentIndex().astype(int)[:, 0]

        Sig = system.Sig() / ret["sig0"]
        Eps = system.Eps() / ret["eps0"]
        idx = system.plastic_CurrentIndex().astype(int)[:, 0]

        ret["S"][inc] = np.sum(idx - idx_n)
        ret["A"][inc] = np.sum(idx != idx_n)
        ret["xi"][inc] = np.sum(tools.fill_avalanche(idx != idx_n))
        ret["epsd"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sigd"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.copy(idx)

    ret["duration"] = np.diff(file["/t"][...], prepend=0) / ret["t0"]
    ret["steadystate"] = steadystate(**ret)

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def cli_ensembleinfo(cli_args=None):
    """
    Read information (avalanche size, stress, strain, ...) of an ensemble,
    see :py:func:`basic_output`.
    Store into a single output file.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)
    info = dict(
        filepath=[os.path.relpath(i, os.path.dirname(args.output)) for i in args.files],
        seed=[],
        uuid=[],
        version=[],
        dependencies=[],
    )

    fields_full = ["epsd", "sigd", "S", "A", "kick", "inc", "steadystate"]
    combine_load = {key: [] for key in ["epsd", "sigd", "S", "A", "kick", "inc"]}
    combine_kick = {key: [] for key in ["epsd", "sigd", "S", "A", "kick", "inc"]}
    file_load = []
    file_kick = []

    fmt = "{:" + str(max(len(i) for i in info["filepath"])) + "s}"
    pbar = tqdm.tqdm(info["filepath"])
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:

        for i, (filename, filepath) in enumerate(zip(pbar, args.files)):

            pbar.set_description(fmt.format(filename), refresh=True)

            with h5py.File(filepath, "r") as file:

                if i == 0:
                    system = init(file)
                    norm = normalisation(file)
                else:
                    system.reset_epsy(read_epsy(file))

                out = basic_output(system, file, norm=norm, verbose=False)

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]
                output.flush()

                info["seed"].append(out["seed"])

                if f"/meta/{entry_points['cli_run']}" in file:
                    meta = file[f"/meta/{entry_points['cli_run']}"]
                    for key in ["uuid", "version", "dependencies"]:
                        if key in meta.attrs:
                            info[key].append(meta.attrs[key])
                        else:
                            info[key].append("?")
                else:
                    info["uuid"].append("?")
                    info["version"].append("?")
                    info["dependencies"].append("?")

                if out["steadystate"] is None:
                    continue

                assert all(out["kick"][1::2])
                assert not any(out["kick"][::2])
                kick = np.copy(out["kick"])
                load = np.logical_not(out["kick"])
                kick[: out["steadystate"]] = False
                load[: out["steadystate"]] = False

                if np.sum(load) > np.sum(kick):
                    load[-1] = False

                assert np.sum(load) == np.sum(kick)

                for key in combine_load:
                    combine_load[key] += list(out[key][load])
                    combine_kick[key] += list(out[key][kick])

                file_load += list(i * np.ones(np.sum(load), dtype=int))
                file_kick += list(i * np.ones(np.sum(kick), dtype=int))

        # store steady-state of full ensemble together

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

        # extract ensemble averages

        ss = np.equal(combine_kick["A"], norm["N"])
        assert all(np.equal(combine_kick["inc"][ss], combine_load["inc"][ss] + 1))
        output["/averages/sigd_top"] = np.mean(combine_load["sigd"][ss])
        output["/averages/sigd_bottom"] = np.mean(combine_kick["sigd"][ss])

        # store metadata at runtime for each input file

        for key in info:
            assert len(info[key]) == len(info["filepath"])

        output["/lookup/filepath"] = info["filepath"]
        output["/lookup/seed"] = info["seed"]
        output["/lookup/uuid"] = info["uuid"]
        tools.h5py_save_unique(info["version"], output, "/lookup/version", asstr=True)
        tools.h5py_save_unique(
            [";".join(i) for i in info["dependencies"]], output, "/lookup/dependencies", split=";"
        )
        output["files"] = output["/lookup/filepath"]

        # metadata for this program

        meta = create_check_meta(output, f"/meta/{progname}", dev=args.develop)


def cli_plot(cli_args=None):
    """
    Plot overview of simulation.
    Plots the stress-strain response and the identified steady-state.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-m", "--marker", type=str, help="Use marker")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file, "r") as file:
        system = init(file)
        data = basic_output(system, file)

    fig, ax = plt.subplots()

    opts = {}
    if args.marker:
        opts["marker"] = args.marker

    ax.plot(data["epsd"], data["sigd"], **opts)

    lim = ax.get_ylim()
    lim = [0, lim[-1]]
    ax.set_ylim(lim)

    e = data["epsd"][data["steadystate"]]
    ax.plot([e, e], lim, c="r", lw=1)

    plt.show()


def runinc_event_basic(system: model.System, file: h5py.File, inc: int, Smax=None) -> dict:
    """
    Rerun increment and get basic event information.

    :param system: The system (modified: increment loaded/rerun).
    :param file: Open simulation HDF5 archive (read-only).
    :param inc: The increment to rerun.
    :param Smax: Stop at given S (to avoid spending time on final energy minimisation).
    :return: A dictionary as follows::

        r: Position of yielding event (block index).
        t: Time of each yielding event.
        S: Size (signed) of the yielding event.
    """

    stored = file["/stored"][...]

    if Smax is None:
        Smax = sys.maxsize

    assert inc > 0
    assert inc in stored
    assert inc - 1 in stored

    _restore_inc(file, system, inc - 1)
    system.initEventDrivenSimpleShear()

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
    idx_t = system.plastic_CurrentIndex()[:, 0].astype(int)

    system.eventDrivenStep(file["/run/epsd/kick"][...], file["/kick"][inc])

    R = []
    T = []
    S = []

    while True:

        niter = system.timeStepsUntilEvent()

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        t = system.t()

        for r in np.argwhere(idx != idx_t):
            R += [r]
            T += [t * np.ones(r.shape)]
            S += [(idx - idx_t)[r]]

        idx_t = np.copy(idx)

        if np.sum(idx - idx_n) >= Smax:
            break

        if niter == 0:
            break

    ret = dict(r=np.array(R).ravel(), t=np.array(T).ravel(), S=np.array(S).ravel())

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def cli_rerun_event(cli_args=None):
    """
    Rerun increment and store basic event info (position and time).
    Tip: truncate when (known) S is reached to not waste time on final stage of energy minimisation.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output file")
    parser.add_argument("-i", "--inc", required=True, type=int, help="Increment number")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-s", "--smax", type=int, help="Truncate at a maximum total S")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.file, "r") as file:
        system = init(file)
        ret = runinc_event_basic(system, file, args.inc, args.smax)

    with h5py.File(args.output, "w") as file:
        file["r"] = ret["r"]
        file["t"] = ret["t"]
        file["S"] = ret["S"]

        meta = create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = args.file
        meta.attrs["inc"] = args.inc
        meta.attrs["Smax"] = args.smax if args.smax else sys.maxsize

    if cli_args is not None:
        return ret


def cli_rerun_event_job_systemspanning(cli_args=None):
    """
    Generate a job to rerun all system-spanning events and generate an event-map.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    output = file_defaults[funcname]

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("-f", "--force", action="store_true", help="Force clean output directory")
    parser.add_argument("-n", "--group", type=int, default=20, help="#increments to group")
    parser.add_argument("-o", "--outdir", type=str, default=output, help="Output directory")
    parser.add_argument("-t", "--truncate", action="store_true", help="Truncate at known Smax")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._create_or_clear_directory(args.outdir, args.force)

    with h5py.File(args.EnsembleInfo, "r") as file:
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        S = file["/avalanche/S"][...]
        inc = file["/avalanche/inc"][...]
        ifile = file["/avalanche/file"][...]
        files = file["/files"].asstr()[...]

    keep = A == N
    S = S[keep]
    inc = inc[keep]
    ifile = ifile[keep]

    commands = []
    executable = entry_points["cli_rerun_event"]
    basedir = os.path.dirname(args.EnsembleInfo)
    basedir = basedir if basedir else "."
    relpath = os.path.relpath(basedir, args.outdir)

    for s, i, f in zip(S, inc, ifile):
        fname = files[f]
        basename = os.path.splitext(os.path.basename(fname))[0]
        cmd = [executable, "-o", f"{basename}_inc={i:d}.h5", "-i", f"{i:d}"]
        if args.truncate:
            cmd += ["-s", f"{s:d}"]
        cmd += [os.path.join(relpath, fname)]
        commands.append(" ".join(cmd))

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return commands


def cli_rerun_dynamics_job_systemspanning(cli_args=None):
    """
    Generate a job to rerun all system-spanning events and measure the dynamics.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    output = file_defaults[funcname]

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("-f", "--force", action="store_true", help="Force clean output directory")
    parser.add_argument("-n", "--group", type=int, default=20, help="#increments to group")
    parser.add_argument("-o", "--outdir", type=str, default=output, help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._create_or_clear_directory(args.outdir, args.force)

    with h5py.File(args.EnsembleInfo, "r") as file:
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        inc = file["/avalanche/inc"][...]
        ifile = file["/avalanche/file"][...]
        files = file["/files"].asstr()[...]

    keep = A == N
    inc = inc[keep]
    ifile = ifile[keep]

    commands = []
    executable = MeasureDynamics.entry_points["cli_run"]
    basedir = os.path.dirname(args.EnsembleInfo)
    basedir = basedir if basedir else "."
    relpath = os.path.relpath(basedir, args.outdir)

    for i, f in zip(inc, ifile):
        fname = files[f]
        basename = os.path.splitext(os.path.basename(fname))[0]
        cmd = [executable, "-o", f"{basename}_inc={i:d}.h5", "-i", f"{i:d}"]
        cmd += [os.path.join(relpath, fname)]
        commands.append(" ".join(cmd))

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return commands


def cli_rerun_event_collect(cli_args=None):
    """
    Collect basis information from :py:func:`cli_rerun_event` and combine in a single output file.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    # collecting data

    data = dict(
        t=[],
        A=[],
        S=[],
        file=[],
        inc=[],
        Smax=[],
        version=[],
        dependencies=[],
    )

    executable = entry_points["cli_rerun_event"]

    for filepath in tqdm.tqdm(args.files):
        with h5py.File(filepath, "r") as file:
            meta = file[f"/meta/{executable}"]
            data["t"].append(file["t"][...][-1] - file["t"][...][0])
            data["S"].append(np.sum(file["S"][...]))
            data["A"].append(np.unique(file["r"][...]).size)
            data["file"].append(meta.attrs["file"])
            data["inc"].append(meta.attrs["inc"])
            data["Smax"].append(meta.attrs["Smax"])
            data["version"].append(meta.attrs["version"])
            data["dependencies"].append(meta.attrs["dependencies"])

    # sorting simulation-id and then increment

    _, index = np.unique(data["file"], return_inverse=True)
    index = index * int(10 ** (np.ceil(np.log10(np.max(data["inc"]))) + 1)) + np.array(data["inc"])
    index = np.argsort(index)

    for key in data:
        data[key] = [data[key][i] for i in index]

    # store (compress where possible)

    with h5py.File(args.output, "w") as file:

        for key in ["t", "A", "S", "inc"]:
            file[key] = data[key]

        prefix = os.path.dirname(os.path.commonprefix(data["file"]))
        if data["file"][0].removeprefix(prefix)[0] == "/":
            prefix += "/"
        data["file"] = [i.removeprefix(prefix) for i in data["file"]]
        file["/file/prefix"] = prefix

        for key in ["file", "version"]:
            value, index = np.unique(data[key], return_inverse=True)
            file[f"/{key}/index"] = index
            file[f"/{key}/value"] = list(str(i) for i in value)

        dep = [";".join(i) for i in data["dependencies"]]
        value, index = np.unique(dep, return_inverse=True)
        file["/dependencies/index"] = index
        file["/dependencies/value"] = [data["dependencies"][i] for i in np.unique(index)]

        create_check_meta(file, f"/meta/{progname}", dev=args.develop)
