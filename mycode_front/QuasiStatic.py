"""
-   Initialise system.
-   Write IO file.
-   Run quasi-static simulation and get basic output.
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
import shelephant
import tqdm
from numpy.typing import ArrayLike

from . import EventMap
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
    cli_plot="Run_plot",
    cli_rerun_dynamics_job_systemspanning="RunDynamics_JobAllSystemSpanning",
    cli_rerun_event_job_systemspanning="RunEventMap_JobAllSystemSpanning",
    cli_run="Run",
    cli_state_after_systemspanning="StateAfterSystemSpanning",
    cli_status="SimulationStatus",
)


file_defaults = dict(
    cli_ensembleinfo="EnsembleInfo.h5",
    cli_rerun_dynamics_job_systemspanning="RunDynamics_SystemSpanning",
    cli_rerun_event_job_systemspanning="EventMap_SystemSpanning",
    cli_state_after_systemspanning="StateAfterSystemSpanning.h5",
)


class System(model.System):
    """
    The system.
    Other than the underlying system this class takes an open HDF5-file to construct
    (to which a certain structure is enforced in different parts of the module).
    """

    def __init__(self, file: h5py.File):
        """
        Construct system from file.

        :param file: HDF5-file opened for reading.
        """

        super().__init__(
            file["coor"][...],
            file["conn"][...],
            file["dofs"][...],
            file["dofsP"][...] if "dofsP" in file else file["iip"][...],
            file["elastic"]["elem"][...],
            file["cusp"]["elem"][...],
        )

        self.setMassMatrix(file["rho"][...])

        if "alpha" in file:
            self.setDampingMatrix(file["alpha"][...])
        elif "/damping/alpha" in file:
            self.setDampingMatrix(file["damping/alpha"][...])
        else:
            raise OSError('No damping parameter "alpha" found')

        if "eta" in file:
            self.setEta(file["eta"][...])

        self.setElastic(file["elastic"]["K"][...], file["elastic"]["G"][...])

        y = read_epsy(file)
        self.N = y.shape[0]
        self.nchunk = y.shape[1]
        self.setPlastic(file["cusp"]["K"][...], file["cusp"]["G"][...], y)

        self.setDt(file["run"]["dt"][...])

        self.normalisation = normalisation(file)

    def reset(self, file: h5py.File):
        """
        Reinitialise system:

        *   Read yield strains from file.
        *   Set all displacements, velocities, and accelerations to zero.

        .. tip::

            For different realisations (yield strains) for the same ensemble: favour this function
            over the constructor to avoid reallocation and file-IO.
        """
        self.quench()
        self.setU(np.zeros_like(self.u()))
        self.reset_epsy(read_epsy(file))

    def restore_inc(self, file: h5py.File, inc: int):
        """
        Quench, and read time and displacement from a file.
        """
        self.quench()
        self.setT(file["t"][inc])
        self.setU(file["disp"][str(inc)][...])

    def dV(self, rank: int = 0):
        """
        Return integration point 'volume' for all quadrature points of all elements.
        Broadcast to an integration point tensor if needed.
        """
        ret = model.System.quad(self).dV()
        if rank == 0:
            return ret
        return model.System.quad(self).AsTensor(rank, ret)

    def plastic_dV(self, rank: int = 0):
        """
        Return integration point 'volume' for all quadrature points of plastic elements.
        Broadcast to an integration point tensor if needed.
        """
        ret = model.System.quad(self).dV()[self.plastic(), :]
        if rank == 0:
            return ret
        return model.System.quad(self).AsTensor(rank, ret)


class DimensionlessSystem(System):
    """
    The system with all (plastic) stain, stress, and time output in normalised unit.
    """

    def __init__(self, file: h5py.File):
        super().__init__(file)
        self.eps0 = self.normalisation["eps0"]
        self.sig0 = self.normalisation["sig0"]
        self.t0 = self.normalisation["t0"]

    def plastic_Epsp(self):
        """
        Return the plastic strain for each integration point of plastic elements. **Normalised**
        """
        return model.System.plastic_Epsp(self) / self.eps0

    def plastic_Eps(self):
        """
        Return the strain tensor for each integration point of plastic elements. **Normalised**
        """
        return model.System.plastic_Eps(self) / self.eps0

    def plastic_Sig(self):
        """
        Return the stress tensor for each integration point of plastic elements. **Normalised**
        """
        return model.System.plastic_Sig(self) / self.sig0

    def Eps(self):
        """
        Return the strain tensor for each integration point of all elements. **Normalised**
        """
        return model.System.Eps(self) / self.eps0

    def Epsdot(self):
        """
        Return the strain-rate tensor for each integration point of all elements. **Normalised**
        """
        return model.System.Epsdot(self) / self.eps0 * self.t0

    def Epsddot(self):
        """
        Return the symmetric gradient of accelerations for each integration point of all elements.
        **Normalised**
        """
        return model.System.Epsddot(self) / self.eps0 * self.t0**2

    def Sig(self):
        """
        Return the stress tensor for each integration point of all elements. **Normalised**
        """
        return model.System.Sig(self) / self.sig0

    def t(self):
        """
        Return time. **Normalised**
        """
        return model.System.t(self) / self.t0


def dependencies(model) -> list[str]:
    """
    Return list with version strings.
    Compared to model.version_dependencies() this adds the version of prrng.
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

    alias = file["cusp"]["epsy"]

    if isinstance(alias, h5py.Dataset):
        return alias[...]

    initstate = alias["initstate"][...]
    initseq = alias["initseq"][...]
    eps_offset = alias["eps_offset"][...]
    eps0 = alias["eps0"][...]
    k = alias["k"][...]
    nchunk = alias["nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= 2.0 * eps0
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


def clone(
    source: h5py.File,
    dest: h5py.File,
    skip: list[str] = None,
    root: str = None,
    dry_run: bool = False,
) -> list[str]:
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
    :parma root: Root in ``dest``.
    :parma dry_run: Do not perform the copy.
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
            if key in ret:
                ret.remove(key)

    if not dry_run:
        g5.copy(source, dest, ret, root=root)

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
    system: System = None,
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

    assert type(system) == System

    funcname = inspect.getframeinfo(inspect.currentframe()).function

    if system is None:
        system = System(source)
    elif init_system:
        system.reset(source)

    if output is None:
        output = basic_output(system, source, verbose=False)

    if init_dest:
        clone(source, dest)
        _init_run_state(dest)

    if not normalised:
        stress /= output["sig0"]

    inci = None

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
                inci = int(i[j, 0])
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
    if inci is not None:

        system.setU(source[f"/disp/{inci:d}"])
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
    if inci is not None:
        meta.attrs["inci"] = inci


def generate(
    filepath: str,
    N: int,
    seed: int = 0,
    scale_alpha: float = 1.0,
    eta: float = None,
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
    :param scale_alpha: Scale default general damping ``alpha`` by factor.
    :param eta: Set damping coefficient at the interface.
    :param init_run: Initialise for use with :py:func:`run`.
    :param classic: The yield strain are hard-coded in the file, otherwise prrng is used.
    :param test_mode: Run in test mode (smaller chunk).
    :param dev: Allow uncommitted changes.
    """

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
    alpha = np.sqrt(2.0) * qL * c * rho * scale_alpha
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

        if eta is not None:
            storage.dump_with_atttrs(
                file,
                "/eta",
                eta,
                desc="Damping coefficient at the interface",
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

        system = System(file)
        meta = create_check_meta(file, f"/meta/{progname}", dev=dev)

        if "completed" in meta.attrs:
            print(f"{basename}: marked completed, skipping")
            return 1

        inc = int(file["/stored"][-1])
        kick = file["/kick"][inc]
        deps = file["/run/epsd/kick"][...]
        system.restore_inc(file, inc)
        system.initEventDrivenSimpleShear()

        nchunk = system.nchunk - 5
        pbar = tqdm.tqdm(
            total=nchunk, disable=not progress, desc=f"{basename}: inc = {inc:8d}, niter = {'-':8s}"
        )

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


def cli_status(cli_args=None):
    """
    Find status for files.

    For an output YAML-file the structure is as follows::

        completed:
        - ...
        - ...
        new:
        - ...
        - ...
        error:
        - ...
        - ...
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

    parser.add_argument("-c", "--completed", action="store_true", help="List completed simulations")
    parser.add_argument("-e", "--partial", action="store_true", help="List partial simulations")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-k", "--key", type=str, required=True, help="Key to read from file")
    parser.add_argument("-n", "--new", action="store_true", help="List 'new' simulations")
    parser.add_argument("-o", "--output", type=str, help="YAML-file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation files")
    args = tools._parse(parser, cli_args)
    assert np.all([os.path.isfile(file) for file in args.files])

    ret = {
        "completed": [],
        "new": [],
        "partial": [],
    }

    for filepath in tqdm.tqdm(args.files):
        with h5py.File(filepath, "r") as file:
            if args.key not in file:
                ret["new"].append(filepath)
            elif "completed" in file[args.key].attrs:
                ret["completed"].append(filepath)
            else:
                ret["partial"].append(filepath)

    if args.output is not None:
        shelephant.yaml.dump(args.output, ret, args.force)
    elif args.completed:
        print(" ".join(ret["completed"]))
    elif args.new:
        print(" ".join(ret["new"]))
    elif args.partial:
        print(" ".join(ret["partial"]))

    if cli_args is not None:
        return ret


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
        The output is dimensionless (see :py:class:`DimensionlessSystem`), and consists of::
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
                system = DimensionlessSystem(file)
                dV = system.plastic_dV()
                dV2 = system.plastic_dV(rank=2)
                ret = {
                    "sig_xx": np.empty((n, system.N), dtype=float),
                    "sig_xy": np.empty((n, system.N), dtype=float),
                    "sig_yy": np.empty((n, system.N), dtype=float),
                    "eps_xx": np.empty((n, system.N), dtype=float),
                    "eps_xy": np.empty((n, system.N), dtype=float),
                    "eps_yy": np.empty((n, system.N), dtype=float),
                    "epsp": np.empty((n, system.N), dtype=float),
                    "S": np.empty((n, system.N), dtype=int),
                }
            else:
                system.reset(file)

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

    if "meta" in file:
        if "normalisation" in file["meta"]:
            alias = file["meta"]["normalisation"]
            N = alias["N"][...]
            eps0 = alias["eps"][...]
            sig0 = alias["sig"][...]
            ret["l0"] = alias["l"][...]
            ret["G"] = alias["G"][...]
            ret["K"] = alias["K"][...]
            ret["rho"] = alias["rho"][...]
            ret["seed"] = file["meta"]["seed_base"][...]

    if len(ret) == 0:
        N = file["cusp"]["epsy"].shape[0]
        G = 1.0
        eps0 = 1.0e-3 / 2.0
        sig0 = 2.0 * G * eps0
        ret["l0"] = np.pi
        ret["G"] = G
        ret["K"] = 10.0 * G
        ret["rho"] = G / 1.0**2.0
        ret["seed"] = str(file["uuid"].asstr()[...])

    # interpret / store additional normalisation
    kappa = ret["K"] / 3.0
    mu = ret["G"] / 2.0
    ret["cs"] = np.sqrt(mu / ret["rho"])
    ret["cd"] = np.sqrt((kappa + 4.0 / 3.0 * mu) / ret["rho"])
    ret["sig0"] = sig0
    ret["eps0"] = eps0
    ret["N"] = N
    ret["t0"] = ret["l0"] / ret["cs"]
    ret["dt"] = file["run"]["dt"][...]

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def basic_output(system: DimensionlessSystem, file: h5py.File, verbose: bool = True) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.

    :return: Basic output as follows::
        epsd: Macroscopic strain (dimensionless units) [ninc].
        sigd: Macroscopic stress (dimensionless units) [ninc].
        S: Number of times a block yielded [ninc].
        A: Number of blocks that yielded at least once [ninc].
        xi: Largest extension corresponding to A [ninc].
        duration: Duration of the event (dimensionless units) [ninc].
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

    incs = file["stored"][...]
    ninc = incs.size
    assert all(incs == np.arange(ninc))

    ret = dict(system.normalisation)
    ret["epsd"] = np.empty((ninc), dtype=float)
    ret["sigd"] = np.empty((ninc), dtype=float)
    ret["S"] = np.zeros((ninc), dtype=int)
    ret["A"] = np.zeros((ninc), dtype=int)
    ret["xi"] = np.zeros((ninc), dtype=int)
    ret["kick"] = file["kick"][...].astype(bool)
    ret["inc"] = incs

    dV = system.dV(rank=2)
    idx_n = None

    for inc in tqdm.tqdm(incs, disable=not verbose):

        system.setU(file["disp"][str(inc)][...])

        if idx_n is None:
            idx_n = system.plastic_CurrentIndex().astype(int)[:, 0]

        Sig = system.Sig()
        Eps = system.Eps()
        idx = system.plastic_CurrentIndex().astype(int)[:, 0]

        ret["S"][inc] = np.sum(idx - idx_n)
        ret["A"][inc] = np.sum(idx != idx_n)
        ret["xi"][inc] = np.sum(tools.fill_avalanche(idx != idx_n))
        ret["epsd"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sigd"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.copy(idx)

    ret["duration"] = np.diff(file["t"][...], prepend=0) / ret["t0"]
    ret["steadystate"] = steadystate(**ret)

    if type(system) == System:
        ret["epsd"] /= system.normalisation["eps0"]
        ret["sigd"] /= system.normalisation["sig0"]

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

    fields_full = ["epsd", "sigd", "S", "A", "kick", "inc"]
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
                    system = DimensionlessSystem(file)
                else:
                    system.reset(file)

                out = basic_output(system, file, verbose=False)

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]
                if out["steadystate"] is not None:
                    output[f"/full/{filename}/steadystate"] = out["steadystate"]
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

        for key, value in system.normalisation.items():
            output[f"/normalisation/{key}"] = value

        # extract ensemble averages

        ss = np.equal(combine_kick["A"], system.N)
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
        data = basic_output(DimensionlessSystem(file), file)

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
    executable = EventMap.entry_points["cli_run"]
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


def cli_state_after_systemspanning(cli_args=None):
    """
    Extract state after system-spanning avalanches.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    output = file_defaults[funcname]

    parser.add_argument("--all", action="store_true", help="Store all output")
    parser.add_argument("--sig", action="store_true", help="Include sig in output")
    parser.add_argument("--eps", action="store_true", help="Include eps in output")
    parser.add_argument("--epsp", action="store_true", help="Include epsp in output")
    parser.add_argument("--size", action="store_true", help="Include S in output")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.EnsembleInfo) as file:
        files = file["/files"].asstr()[...]
        inc = file["/avalanche/inc"][...]
        fid = file["/loading/file"][...]
        A = file["/avalanche/A"][...]
        N = int(file["/normalisation/N"][...])

    keep = A == N
    inc = inc[keep]
    fid = fid[keep]
    A = A[keep]

    dirname = os.path.dirname(args.EnsembleInfo)
    select = {}

    for f in np.unique(fid):
        filepath = os.path.join(dirname, files[f])
        select[filepath] = inc[fid == f]

    data = interface_state(select)

    if args.all:
        keep = [i for i in data]
    else:
        keep = []
        if args.sig:
            keep += ["sig_xx", "sig_xy", "sig_yy"]
        if args.eps:
            keep += ["eps_xx", "eps_xy", "eps_yy"]
        if args.epsp:
            keep += ["epsp"]
        if args.size:
            keep += ["S"]

    with h5py.File(args.output, "w") as file:
        for key in data:
            if key in keep:
                file[key] = data[key]
