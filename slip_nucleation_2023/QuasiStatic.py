"""
-   Initialise system.
-   Write IO file.
-   Run quasi-static simulation and get basic output.
"""

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
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

from . import storage
from . import tag
from . import tools
from ._version import version

plt.style.use(["goose", "goose-latex"])


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

        root = file["param"]
        y = read_epsy(file)
        self.nchunk = y.shape[1] + 1  # not yet allocated elastic

        assert "alpha" in root or "eta" in root

        def to_field(value, n):
            if value.size == n:
                return value
            if value.size == 1:
                return value * np.ones(n)
            raise ValueError(f"{value.size} != {n}")

        plastic = root["cusp"]["elem"][...]
        elastic = np.setdiff1d(np.arange(root["conn"].shape[0]), plastic)
        nel = elastic.size
        npl = plastic.size

        super().__init__(
            coor=root["coor"][...],
            conn=root["conn"][...],
            dofs=root["dofs"][...],
            iip=root["iip"][...],
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(to_field(root["elastic"]["K"][...], nel)),
            elastic_G=FrictionQPotFEM.moduli_toquad(to_field(root["elastic"]["G"][...], nel)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(to_field(root["cusp"]["K"][...], npl)),
            plastic_G=FrictionQPotFEM.moduli_toquad(to_field(root["cusp"]["G"][...], npl)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(y),
            dt=root["dt"][...],
            rho=root["rho"][...],
            alpha=root["alpha"][...] if "alpha" in root else 0,
            eta=root["eta"][...] if "eta" in root else 0,
        )

        self.normalisation = normalisation(file)
        self.eps0 = self.normalisation["eps0"]
        self.sig0 = self.normalisation["sig0"]
        self.t0 = self.normalisation["t0"]

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
        self.u = np.zeros_like(self.u)
        self.plastic.epsy = FrictionQPotFEM.epsy_initelastic_toquad(read_epsy(file))

    def restore_quasistatic_step(self, root: h5py.Group, step: int):
        """
        Quench and restore an a quasi-static step for the relevant root.
        The ``root`` group should contain::

            root["u"][str(step)]   # Displacements
            root["inc"][step]      # Increment (-> time)

        :param root: HDF5 archive opened in the right root (read-only).
        :param step: Step number.
        """
        self.quench()
        self.t = root["inc"][step]
        self.u = root["u"][str(step)][...]

    def dV(self, rank: int = 0):
        """
        Return integration point 'volume' for all quadrature points of all elements.
        Broadcast to an integration point tensor if needed.
        """
        ret = self.quad.dV
        if rank == 0:
            return ret
        return self.quad.AsTensor(rank, ret)

    def plastic_dV(self, rank: int = 0):
        """
        Return integration point 'volume' for all quadrature points of plastic elements.
        Broadcast to an integration point tensor if needed.
        """
        ret = self.quad.dV[self.plastic_elem, ...]
        if rank == 0:
            return ret
        return self.quad.AsTensor(rank, ret)


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


def read_epsy(file: h5py.Group) -> np.ndarray:
    """
    (Re)Generate yield strain sequence per plastic element.
    :param root: HDF5 archive
    """

    root = file["param"]["cusp"]["epsy"]

    if isinstance(root, h5py.Dataset):
        return root[...]

    seed = file["realisation"]["seed"][...]

    generators = prrng.pcg32_array(
        initstate=seed + root["initstate"][...],
        initseq=root["initseq"][...],
    )

    epsy = generators.weibull([root["nchunk"][...]], root["weibull"]["k"][...])
    epsy *= 2 * root["weibull"]["typical"][...]
    epsy += root["weibull"]["offset"][...]

    return np.cumsum(epsy, 1)


def _init_run_state(root: h5py.File, u: ArrayLike):
    """
    Initialise as extendible datasets:
    *   ``"/root/inc"``: increment number (-> time) at the end of each step.
    *   ``"/root/kick"``: kick setting of each step.

    Initialise datasets:
    *   ``"/root/u/0"``: displacements at the end of the step.

    :param root: HDF5 archive opened in the right root (read-write).
    :param u: Displacement field.
    """

    storage.create_extendible(root, "inc", np.uint64, desc="Increment number (end of step)")
    storage.create_extendible(root, "kick", bool, desc="Kick used.")
    group = root.create_group("u")
    group.attrs["desc"] = 'Displacements at end of step. Index corresponds to "inc" and "kick".'
    group["0"] = u

    storage.dset_extend1d(root, "inc", 0, 0)
    storage.dset_extend1d(root, "kick", 0, False)


def branch_fixed_stress(
    source: h5py.File,
    dest: h5py.File,
    root: str,
    step: int = None,
    step_c: int = None,
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

    *   Quasistatic step (``step``).
        ``step_c`` and ``stress`` are ignored if supplied, but are stored as meta-data in that case.
        To ensure meta-data integrity,
        they are only checked to be consistent with the state at ``step``.

    *   Fixed stress after a system spanning event (``step_c`` and ``stress``).
        Note that ``stress`` is approximated as best as possible,
        and its actual value is stored in the meta-data.

    :param source: Source file.
    :param dest: Destination file.
    :param root: Root to which to write the branched configuration (e.g. `QuastiStatic`).
    :param step: Branch at specific quasistatic step.
    :param step_c: Branch at fixed stress after this system-spanning event.
    :param stress: Branch at fixed stress.
    :param normalised: Assume ``stress`` to be normalised (see ``sig0`` in :py:func:`basic_output`).
    :param system: The system (optional, specify to avoid reallocation).
    :param init_system: Read yield strains (otherwise ``system`` is assumed fully initialised).
    :param init_dest: Initialise ``dest`` from ``source``.
    :param output: Output of :py:func:`basic_output` (read if not specified).
    :param dev: Allow uncommitted changes.
    """
    if system is None:
        system = System(source)
    elif init_system:
        system.reset(source)

    if output is None:
        output = basic_output(system, source, verbose=False)

    if init_dest:
        g5.copy(source, dest, ["/param", "/realisation", "/meta"])
        _init_run_state(root=dest.create_group(root), u=system.u)

    if not normalised and stress is not None:
        stress /= output["sig0"]

    step_i = None

    # determine at which step to branch
    if step is None:
        assert step_c is not None and stress is not None

        next_step = int(step_c) + np.argwhere(output["A"][step_c:] == output["N"]).ravel()[1]
        assert (next_step - step_c) % 2 == 0
        i = output["step"][step_c:next_step].reshape(-1, 2)
        s = output["sig"][step_c:next_step].reshape(-1, 2)
        k = output["kick"][step_c:next_step].reshape(-1, 2)
        t = np.sum(s < stress, axis=1)
        assert np.all(k[:, 0])
        assert np.sum(t) > 0

        if np.sum(t == 1) >= 1:
            j = np.argmax(t == 1)
            if np.abs(s[j, 0] - stress) / s[j, 0] < 1e-4:
                step = i[j, 0]
                stress = s[j, 0]
            elif np.abs(s[j, 1] - stress) / s[j, 1] < 1e-4:
                step = i[j, 1]
                stress = s[j, 1]
            else:
                step_i = int(i[j, 0])
        else:
            j = np.argmax(t == 0)
            step = i[j, 0]
            stress = s[j, 0]

    # restore specific step
    if step is not None:
        system.u = source[f"/QuasiStatic/u/{step:d}"][...]
        dest[root]["kick"][0] = source["/QuasiStatic/kick"][step]
        if stress is not None:
            assert np.isclose(stress, output["sig"][step])
        if step_c is not None:
            i = output["step"][output["A"] == output["N"]]
            assert i[i >= step_c][1] > step

    # apply elastic loading to reach a specific stress
    elif step_i is not None:
        system.u = source[f"/QuasiStatic/u/{step_i:d}"]
        i_n = np.copy(system.plastic.i)
        d = system.addSimpleShearToFixedStress(stress * output["sig0"])
        assert np.all(system.plastic.i == i_n)
        assert d >= 0.0

    else:
        raise ValueError("Invalid state definition.")

    # store branch
    dest[root]["u"]["0"][...] = system.u

    metaname = "branch_fixed_stress"
    i = 1
    while f"/meta/{metaname}" in dest:
        metaname = f"branch_fixed_stress_{i:d}"
        i += 1

    meta = create_check_meta(dest, "/meta/branch_fixed_stress", dev=dev)
    meta.attrs["file"] = os.path.basename(source.filename)
    if stress is not None:
        meta.attrs["stress"] = stress
    if step_c is not None:
        meta.attrs["step_c"] = int(step_c)
    if step is not None:
        meta.attrs["step"] = int(step)
    if step_i is not None:
        meta.attrs["step_i"] = step_i


def generate(
    filepath: str,
    N: int,
    seed: int = 0,
    scale_alpha: float = 1.0,
    eta: float = None,
    init_run: bool = True,
    dev: bool = False,
):
    r"""
    Generate input file. See :py:func:`read_epsy` for different strategies to store yield strains.

    :param filepath: The filepath of the input file.
    :param N: The number of blocks.
    :param seed: Base seed to use to generate the disorder.
    :param scale_alpha: Scale default general damping ``alpha`` by factor.
    :param eta:
        Damping coefficient at the interface.
        Note: :math:`\eta / \eta_\mathrm{rd} = \eta / (G t_0)

    :param init_run: Initialise for use with :py:func:`run`.
    :param dev: Allow uncommitted changes.
    """

    assert not os.path.isfile(filepath)

    # parameters
    h = np.pi
    L = h * float(N)

    # define mesh and element sets
    mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)
    coor = mesh.coor
    conn = mesh.conn
    plastic = mesh.elementsMiddleLayer

    # extract node sets to set the boundary conditions
    top = mesh.nodesTopEdge
    bottom = mesh.nodesBottomEdge
    left = mesh.nodesLeftOpenEdge
    right = mesh.nodesRightOpenEdge

    # periodicity in horizontal direction
    dofs = mesh.dofs
    dofs[right, :] = dofs[left, :]
    dofs = GooseFEM.Mesh.renumber(dofs)

    # fixed top and bottom
    iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, :].ravel()))

    # yield strains
    k = 2.0
    eps0 = 5e-5
    eps_offset = 1e-5
    nchunk = int(6e3)
    initstate = np.arange(N).astype(np.int64)
    initseq = np.zeros_like(initstate)

    # elasticity & damping
    c = 1
    G = 1
    K = 10 * G
    rho = G / c**2
    qL = 2 * np.pi / L
    qh = 2 * np.pi / h
    alpha = np.sqrt(2) * qL * c * rho
    dt = (1 / (c * qh)) / 10

    with h5py.File(filepath, "w") as file:
        storage.dump_with_atttrs(
            file,
            "/realisation/seed",
            seed,
            desc="Basic seed == 'unique' identifier",
        )

        storage.dump_with_atttrs(
            file,
            "/param/coor",
            coor,
            desc="Nodal coordinates [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/param/conn",
            conn,
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        storage.dump_with_atttrs(
            file,
            "/param/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/param/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        storage.dump_with_atttrs(
            file,
            "/param/dt",
            dt,
            desc="Time step",
        )

        storage.dump_with_atttrs(
            file,
            "/param/rho",
            rho,
            desc="Mass density; homogeneous",
        )

        if scale_alpha != 0 and scale_alpha is not None:
            storage.dump_with_atttrs(
                file,
                "/param/alpha",
                scale_alpha * alpha,
                desc="Damping coefficient (density); homogeneous",
            )

        if eta is not None:
            storage.dump_with_atttrs(
                file,
                "/param/eta",
                eta,
                desc="Damping coefficient at the interface; homogeneous",
            )

        storage.dump_with_atttrs(
            file,
            "/param/elastic/K",
            K,
            desc="Bulk modulus for elastic (non-plastic) elements; homogeneous",
        )

        storage.dump_with_atttrs(
            file,
            "/param/elastic/G",
            G,
            desc="Shear modulus for elastic (non-plastic) elements; homogeneous",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/K",
            K,
            desc="Bulk modulus for elements in '/cusp/elem'; homogeneous",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/G",
            G,
            desc="Shear modulus for elements in '/cusp/elem'; homogeneous",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/nchunk",
            nchunk,
            desc="Chunk size",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/initstate",
            initstate,
            desc="State to initialise prrng.pcg32_array",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/initseq",
            initseq,
            desc="Sequence to initialise prrng.pcg32_array",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/weibull/k",
            k,
            desc="Shape factor of Weibull distribution",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/weibull/typical",
            eps0,
            desc="Normalisation: epsy(i + 1) - epsy(i) = 2 * typical * random + offset",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/weibull/offset",
            eps_offset,
            desc="Offset, see '/param/cusp/epsy/weibull/typical'",
        )

        storage.dump_with_atttrs(
            file,
            "/param/cusp/epsy/deps",
            eps0 * 2e-4,
            desc="Strain kick to apply",
        )

        assert np.min(np.diff(read_epsy(file), axis=1)) > file["/param/cusp/epsy/deps"][...]

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/l",
            h,
            desc="Elementary block size",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        storage.dump_with_atttrs(
            file,
            "/param/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        create_check_meta(file, "/meta/QuasiStatic_Generate", dev=dev)

        if init_run:
            _init_run_state(root=file.create_group("QuasiStatic"), u=np.zeros_like(coor))


def Generate(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=4 * (3**6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("outdir", type=str, help="Output directory")

    args = tools._parse(parser, cli_args)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []

    for i in range(args.start, args.start + args.nsim):
        files += [f"id={i:03d}.h5"]
        generate(
            filepath=str(outdir / f"id={i:03d}.h5"),
            N=args.size,
            seed=i * args.size,
            dev=args.develop,
        )

    commands = [f"QuasiStatic_Run {file}" for file in files]
    shelephant.yaml.dump(outdir / "commands_run.yaml", commands)


def _compare_versions(ver, cmpver):
    if tag.greater_equal(cmpver, "14.0"):
        if tag.greater_equal(ver, cmpver):
            return True
    else:
        return tag.equal(ver, cmpver)

    return False


def create_check_meta(
    file: h5py.File = None,
    path: str = None,
    dev: bool = False,
    **kwargs,
) -> h5py.Group:
    """
    Create or read/check metadata. This function asserts that:

    -   There are no uncommitted changes.
    -   There are no version changes.

    It create metadata as attributes to a group ``path`` as follows::

        "uuid": A unique identifier that can be used to distinguish simulations if needed.
        "version": The current version of this code (see below).
        "dependencies": The current version of all relevant dependencies (see below).
        "compiler": Compiler information.

    :param file: HDF5 archive.
    :param path: Path in ``file`` to store/read metadata.
    :param dev: Allow uncommitted changes.
    :return: Group to metadata.
    """

    deps = sorted(list(set(list(model.version_dependencies()) + ["prrng=" + prrng.version()])))

    assert dev or not tag.has_uncommitted(version)
    assert dev or not tag.any_has_uncommitted(deps)

    if file is None:
        return None

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = model.version_compiler()
        for key in kwargs:
            meta.attrs[key] = kwargs[key]
        return meta

    meta = file[path]
    if file.mode in ["r+", "w", "a"]:
        assert dev or _compare_versions(version, meta.attrs["version"])
        assert dev or tag.all_greater_equal(deps, meta.attrs["dependencies"])
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = model.version_compiler()
    else:
        assert dev or tag.equal(version, meta.attrs["version"])
        assert dev or tag.all_equal(deps, meta.attrs["dependencies"])
    return meta


def MoveMeta(cli_args=None):
    """
    Create a copy of meta-data, and overwrite the version information with the current information
    and a new UUID.
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--uncomplete", action="store_true", help="Unmarked as completed")
    parser.add_argument("old_name", type=str, help="Former name (overwritten with new versions)")
    parser.add_argument("new_name", type=str, help="Former name (overwritten with new versions)")
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    deps = sorted(list(model.version_dependencies()) + ["prrng=" + prrng.version()])
    compiler = model.version_compiler()

    assert args.develop or not tag.has_uncommitted(version)
    assert args.develop or not tag.any_has_uncommitted(deps)

    with h5py.File(args.file, "a") as file:
        assert args.old_name in file

        g5.copy(file, file, args.old_name, args.new_name)

        meta = file[args.old_name]
        meta.attrs["uuid"] = str(uuid.uuid4())
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = deps
        meta.attrs["compiler"] = compiler

        if args.uncomplete:
            meta.attrs["completed"] = 0


def Run(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    basename = os.path.basename(args.file)

    with h5py.File(args.file, "a") as file:
        system = System(file)
        meta = create_check_meta(file, "/meta/QuasiStatic_Run", dev=args.develop)

        if "completed" in meta.attrs:
            if meta.attrs["completed"]:
                print(f"{basename}: marked completed, skipping")
                return 1

        root = file["QuasiStatic"]
        step = root["kick"].size
        deps = file["/param/cusp/epsy/deps"][...]
        system.restore_quasistatic_step(root, step - 1)
        kick = root["kick"][step - 1]
        system.initEventDrivenSimpleShear()

        nchunk = system.nchunk - 5
        pbar = tqdm.tqdm(
            total=nchunk,
            desc=f"{basename}: step = {step:8d}, niter = {'-':8s}",
        )

        for step in range(step, sys.maxsize):
            kick = not kick
            system.eventDrivenStep(deps, kick)

            if kick:
                inc_n = system.inc
                ret = system.minimise(nmargin=5)

                if ret < 0:
                    break

                pbar.n = np.max(system.plastic.i)
                nstep = system.inc - inc_n
                pbar.set_description(f"{basename}: step = {step:8d}, nstep = {nstep:8d}")
                pbar.refresh()

            if not kick:
                if np.any(system.plastic.i >= nchunk):
                    break

            storage.dset_extend1d(root, "inc", step, system.inc)
            storage.dset_extend1d(root, "kick", step, kick)
            root["u"][str(step)] = system.u
            file.flush()

        pbar.set_description(f"{basename}: step = {step:8d}, {'completed':16s}")
        meta.attrs["completed"] = 1


def SimulationStatus(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

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
                continue

            if "completed" in file[args.key].attrs:
                if file[args.key].attrs["completed"]:
                    ret["completed"].append(filepath)
                    continue

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
    eps: ArrayLike, sig: ArrayLike, kick: ArrayLike, A: ArrayLike, N: int, **kwargs
) -> int:
    """
    Estimate the first step of the steady-state. Constraints:

    -   Start with elastic loading.
    -   Sufficiently low tangent modulus.
    -   All blocks yielded at least once.

    .. note::

        Keywords arguments that are not explicitly listed are ignored.

    :param eps: Macroscopic strain [nstep].
    :param sig: Macroscopic stress [nstep].
    :param kick: Whether a kick was applied [nstep].
    :param A: Number of blocks that yielded at least once [nstep].
    :param N: Number of blocks.
    :return: Increment number.
    """

    if sig.size <= 2:
        return None

    tangent = np.empty_like(sig)
    tangent[0] = np.inf
    tangent[1:] = (sig[1:] - sig[0]) / (eps[1:] - eps[0])

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
        The output is dimensionless, and consists of::
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
                system = System(file)
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

            for j, step in enumerate(filepaths[filepath]):
                if read_disp:
                    with h5py.File(read_disp[filepath][j], "r") as disp:
                        system.u = disp[f"/QuasiStatic/u/{step:d}"][...]
                else:
                    system.u = file[f"/QuasiStatic/u/{step:d}"][...]

                Sig = np.average(system.plastic.Sig / system.sig0, weights=dV2, axis=1)
                Eps = np.average(system.plastic.Eps / system.eps0, weights=dV2, axis=1)
                ret["sig_xx"][i, :] = Sig[:, 0, 0]
                ret["sig_xy"][i, :] = Sig[:, 0, 1]
                ret["sig_yy"][i, :] = Sig[:, 1, 1]
                ret["eps_xx"][i, :] = Eps[:, 0, 0]
                ret["eps_xy"][i, :] = Eps[:, 0, 1]
                ret["eps_yy"][i, :] = Eps[:, 1, 1]
                ret["epsp"][i, :] = np.average(
                    system.plastic.epsp / system.eps0, weights=dV, axis=1
                )
                ret["S"][i, :] = system.plastic.i[:, 0]
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
        eta0: Normalisation of the viscosity of the interface (float).
    """

    ret = {}

    root = file["param"]["normalisation"]
    N = root["N"][...]
    eps0 = root["eps"][...]
    sig0 = root["sig"][...]
    ret["l0"] = root["l"][...]
    ret["G"] = root["G"][...]
    ret["K"] = root["K"][...]
    ret["rho"] = root["rho"][...]
    ret["seed"] = None

    if "realisation" in file:
        ret["seed"] = file["realisation"]["seed"][...]

    # interpret / store additional normalisation
    kappa = ret["K"] / 3.0
    mu = ret["G"] / 2.0
    ret["cs"] = np.sqrt(mu / ret["rho"])
    ret["cd"] = np.sqrt((kappa + 4.0 / 3.0 * mu) / ret["rho"])
    ret["sig0"] = sig0
    ret["eps0"] = eps0
    ret["N"] = N
    ret["t0"] = ret["l0"] / ret["cs"]
    ret["dt"] = file["param"]["dt"][...]
    ret["eta0"] = ret["G"] * ret["t0"]

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def basic_output(
    system: System, file: h5py.File, root: str = "QuasiStatic", verbose: bool = True
) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.

    :return: Basic output as follows::
        eps: Macroscopic strain (dimensionless) [nstep].
        sig: Macroscopic stress (dimensionless) [nstep].
        sig_broken: Average stress in blocks that have yielded during this event [nstep].
        sig_unbroken: Average stress in blocks that have not yielded during this event [nstep].
        delta_sig_broken: Same as `sig_broken` but relative to the previous step [nstep].
        delta_sig_unbroken: Same as `sig_unbroken` but relative to the previous step [nstep].
        S: Number of times a block yielded [nstep].
        A: Number of blocks that yielded at least once [nstep].
        xi: Largest extension corresponding to A [nstep].
        duration: Duration of the event (dimensionless) [nstep].
        dinc: Same as `duration` but in number of time steps [nstep].
        kick: True is step started with a kick [nstep].
        step: Step numbers == np.arange(nstep).
        steadystate: Step number where the steady state starts (int).
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
        eta0: Normalisation of the viscosity of the interface (float).
    """

    root = file[root]
    steps = np.arange(root["inc"].size)
    nstep = steps.size

    ret = dict(system.normalisation)
    ret["eps"] = np.empty((nstep), dtype=float)
    ret["sig"] = np.empty((nstep), dtype=float)
    ret["sig_broken"] = np.zeros((nstep), dtype=float)
    ret["sig_unbroken"] = np.zeros((nstep), dtype=float)
    ret["delta_sig_broken"] = np.zeros((nstep), dtype=float)
    ret["delta_sig_unbroken"] = np.zeros((nstep), dtype=float)
    ret["S"] = np.zeros((nstep), dtype=int)
    ret["A"] = np.zeros((nstep), dtype=int)
    ret["xi"] = np.zeros((nstep), dtype=int)
    ret["kick"] = root["kick"][...].astype(bool)
    ret["step"] = steps

    dV = system.dV(rank=2)
    dV_plas = system.plastic_dV(rank=2)
    i_n = None

    opts = {}
    if not verbose:
        opts["disable"] = True

    for step in tqdm.tqdm(steps, **opts):
        system.restore_quasistatic_step(root=root, step=step)

        if i_n is None:
            i_n = np.copy(system.plastic.i.astype(int)[:, 0])
            Sig_plas_n = system.plastic.Sig / system.sig0

        Sig = system.Sig() / system.sig0
        Eps = system.Eps() / system.eps0
        i = system.plastic.i.astype(int)[:, 0]
        Sig_plas = system.plastic.Sig / system.sig0
        dSig_plas = Sig_plas - Sig_plas_n
        broken = tools.fill_avalanche(i != i_n)

        ret["S"][step] = np.sum(i - i_n)
        ret["A"][step] = np.sum(i != i_n)
        ret["xi"][step] = np.sum(broken)
        ret["eps"][step] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sig"][step] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        if np.any(~broken):
            ret["sig_unbroken"][step] = GMat.Sigd(
                np.average(Sig_plas[~broken, ...], weights=dV_plas[~broken, ...], axis=(0, 1))
            )
            ret["delta_sig_unbroken"][step] = GMat.Sigd(
                np.average(dSig_plas[~broken, ...], weights=dV_plas[~broken, ...], axis=(0, 1))
            )

        if np.any(broken):
            ret["sig_broken"][step] = GMat.Sigd(
                np.average(Sig_plas[broken, ...], weights=dV_plas[broken, ...], axis=(0, 1))
            )
            ret["delta_sig_broken"][step] = GMat.Sigd(
                np.average(dSig_plas[broken, ...], weights=dV_plas[broken, ...], axis=(0, 1))
            )

        i_n = np.copy(i)
        Sig_plas_n = np.copy(Sig_plas)

    ret["dinc"] = np.diff(root["inc"][...].astype(np.int64), prepend=0)
    ret["duration"] = np.diff(root["inc"][...] * ret["dt"][...], prepend=0) / ret["t0"]
    ret["steadystate"] = steadystate(**ret)

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def EnsembleInfo(cli_args=None):
    """
    Read information (avalanche size, stress, strain, ...) of an ensemble,
    see :py:func:`basic_output`.
    Store into a single output file.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument(
        "-o", "--output", type=str, default="QuasiStatic_EnsembleInfo.h5", help="Output file"
    )
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

    floats = [
        "eps",
        "sig",
        "sig_broken",
        "sig_unbroken",
        "delta_sig_broken",
        "delta_sig_unbroken",
    ]
    fields_full = ["S", "A", "kick", "step"] + floats
    combine_load = {key: [] for key in fields_full}
    combine_kick = {key: [] for key in fields_full}
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
                    system = System(file)
                else:
                    system.reset(file)

                out = basic_output(system, file, verbose=False)

                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]
                if out["steadystate"] is not None:
                    output[f"/full/{filename}/steadystate"] = out["steadystate"]
                output.flush()

                info["seed"].append(out["seed"])

                meta = file["/meta/QuasiStatic_Run"]
                for key in ["uuid", "version", "dependencies"]:
                    info[key].append(meta.attrs[key])

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

        for key in ["A", "step"]:
            combine_load[key] = np.array(combine_load[key], dtype=np.uint64)
            combine_kick[key] = np.array(combine_kick[key], dtype=np.uint64)

        for key in floats:
            combine_load[key] = np.array(combine_load[key])
            combine_kick[key] = np.array(combine_kick[key])

        for key in combine_load:
            output[f"/loading/{key}"] = combine_load[key]
            output[f"/avalanche/{key}"] = combine_kick[key]

        for key, value in system.normalisation.items():
            output[f"/normalisation/{key}"] = value

        # extract ensemble averages

        ss = np.equal(combine_kick["A"], system.N)
        assert all(np.equal(combine_kick["step"][ss], combine_load["step"][ss] + 1))
        output["/averages/sig_top"] = np.mean(combine_load["sig"][ss])
        output["/averages/sig_bottom"] = np.mean(combine_kick["sig"][ss])

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

        meta = create_check_meta(output, "/meta/QuasiStatic_EnsembleInfo", dev=args.develop)


def Plot(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("-m", "--marker", type=str, help="Use marker")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file, "r") as file:
        data = basic_output(System(file), file)

    _, ax = plt.subplots()

    opts = {}
    if args.marker:
        opts["marker"] = args.marker

    ax.plot(data["eps"], data["sig"], **opts)

    lim = ax.get_ylim()
    lim = [0, lim[-1]]
    ax.set_ylim(lim)

    if data["steadystate"] is not None:
        e = data["eps"][data["steadystate"]]
        ax.plot([e, e], lim, c="r", lw=1)

    plt.show()


def MakeJobEventMapOfSystemSpanning(cli_args=None):
    """
    Generate a job to rerun all system-spanning events and generate an event-map.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))
    parser.add_argument("-f", "--force", action="store_true", help="Force clean output directory")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="QuasiStatic_EventMapOfSystemSpanning",
        help="Output directory",
    )
    parser.add_argument("-t", "--truncate", action="store_true", help="Truncate at known Smax")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._create_or_clear_directory(args.outdir, args.force)
    outdir = pathlib.Path(args.outdir)

    with h5py.File(args.EnsembleInfo, "r") as file:
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        S = file["/avalanche/S"][...]
        steps = file["/avalanche/step"][...]
        ifile = file["/avalanche/file"][...]
        files = file["/files"].asstr()[...]

    keep = A == N
    S = S[keep]
    steps = steps[keep]
    ifile = ifile[keep]

    commands = []
    basedir = os.path.dirname(args.EnsembleInfo)
    basedir = basedir if basedir else "."
    relpath = os.path.relpath(basedir, args.outdir)

    for s, step, f in zip(S, steps, ifile):
        fname = files[f]
        basename = os.path.splitext(os.path.basename(fname))[0]
        cmd = ["EventMap_Run -o", f"{basename}_step={step:d}.h5", "--step", f"{step:d}"]
        if args.truncate:
            cmd += ["-s", f"{s:d}"]
        cmd += [os.path.join(relpath, fname)]
        commands.append(" ".join(cmd))

    shelephant.yaml.dump(outdir / "commands_system-spanning_event.yaml", commands)

    if cli_args is not None:
        return commands


def MakeJobDynamicsOfSystemSpanning(cli_args=None):
    """
    Generate a job to rerun all system-spanning events and measure the dynamics.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))
    parser.add_argument("-f", "--force", action="store_true", help="Force clean output directory")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="QuasiStatic_DynamicsOfSystemSpanning",
        help="Output directory",
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._create_or_clear_directory(args.outdir, args.force)
    outdir = pathlib.Path(args.outdir)

    with h5py.File(args.EnsembleInfo, "r") as file:
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        steps = file["/avalanche/step"][...]
        ifile = file["/avalanche/file"][...]
        files = file["/files"].asstr()[...]

    keep = A == N
    steps = steps[keep]
    ifile = ifile[keep]

    commands = []
    basedir = os.path.dirname(args.EnsembleInfo)
    basedir = basedir if basedir else "."
    relpath = os.path.relpath(basedir, args.outdir)

    for step, f in zip(steps, ifile):
        fname = files[f]
        basename = os.path.splitext(os.path.basename(fname))[0]
        cmd = ["Dynamics_Run -o", f"{basename}_step={step:d}.h5", "--step", f"{step:d}"]
        cmd += [os.path.join(relpath, fname)]
        commands.append(" ".join(cmd))

    shelephant.yaml.dump(outdir / "commands_system-spanning_dynamics.yaml", commands)

    if cli_args is not None:
        return commands


def StateAfterSystemSpanning(cli_args=None):
    """
    Extract state after system-spanning avalanches.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))
    parser.add_argument("--all", action="store_true", help="Store all output")
    parser.add_argument("--sig", action="store_true", help="Include sig in output")
    parser.add_argument("--eps", action="store_true", help="Include eps in output")
    parser.add_argument("--epsp", action="store_true", help="Include epsp in output")
    parser.add_argument("--size", action="store_true", help="Include S in output")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="QuasiStatic_StateAfterSystemSpanning.h5",
        help="Output file",
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("EnsembleInfo", type=str, help="EnsembleInfo")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.EnsembleInfo)
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.EnsembleInfo) as file:
        files = file["/files"].asstr()[...]
        step = file["/avalanche/step"][...]
        fid = file["/loading/file"][...]
        A = file["/avalanche/A"][...]
        N = int(file["/normalisation/N"][...])

    keep = A == N
    step = step[keep]
    fid = fid[keep]
    A = A[keep]

    dirname = os.path.dirname(args.EnsembleInfo)
    select = {}

    for f in np.unique(fid):
        filepath = os.path.join(dirname, files[f])
        select[filepath] = step[fid == f]

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


def transform_deprecated_param(src, dest, paths, source_root: str = "/"):
    """
    Transform deprecated parameters.
    This code is considered 'non-maintained'.
    """

    # read and remove from list of paths
    def read_clear(file, key, paths):
        value = file[key][...]
        try:
            paths.remove(key)
        except ValueError:
            raise ValueError(f"{key} not found in paths")
        return value

    source_root = g5.abspath(source_root)
    dest.create_group("param")

    for key in ["/alpha", "/eta", "/rho", "/elastic/K", "/elastic/G", "/cusp/K", "/cusp/G"]:
        if g5.join(source_root, key) not in src:
            continue
        data = read_clear(src, g5.join(source_root, key), paths)
        value = np.atleast_1d(data)[0]
        assert np.allclose(value, data)
        dest[g5.join("param", key, root=True)] = value

    assert g5.join(source_root, "dofsP") not in src, "WIP: please implement when needed"

    for key in ["/coor", "/conn", "/dofs", "/iip"]:
        g5.copy(src, dest, key, root="param", source_root=source_root)
        paths.remove(g5.join(source_root, key))

    plastic = read_clear(src, g5.join(source_root, "/cusp/elem"), paths)
    elastic = np.setdiff1d(np.arange(src[g5.join(source_root, "conn")].shape[0]), plastic)
    assert np.all(read_clear(src, g5.join(source_root, "/elastic/elem"), paths) == elastic)
    g5.copy(src, dest, "/cusp/elem", "/param/cusp/elem", source_root=source_root)

    assert not isinstance(
        src[g5.join(source_root, "/cusp/epsy")], h5py.Dataset
    ), "WIP: please implement when needed"

    seed = src[g5.join(source_root, "/meta/seed_base")][...]
    initstate = read_clear(src, g5.join(source_root, "/cusp/epsy/initstate"), paths)
    initseq = read_clear(src, g5.join(source_root, "/cusp/epsy/initseq"), paths)
    new_initstate = np.arange(initstate.size, dtype=initstate.dtype)
    new_initseq = np.zeros_like(initseq)
    assert np.all(initstate == seed + new_initstate)
    assert np.all(initseq == new_initseq)
    dest["/param/cusp/epsy/initstate"] = new_initstate
    g5.copy(src, dest, "/cusp/epsy/initseq", "/param/cusp/epsy/initseq", source_root=source_root)

    rename = {
        "/cusp/epsy/nchunk": "/param/cusp/epsy/nchunk",
        "/cusp/epsy/eps_offset": "/param/cusp/epsy/weibull/offset",
        "/cusp/epsy/eps0": "/param/cusp/epsy/weibull/typical",
        "/cusp/epsy/k": "/param/cusp/epsy/weibull/k",
        "/run/dt": "/param/dt",
        "/meta/seed_base": "/realisation/seed",
        "/run/epsd/kick": "/param/cusp/epsy/deps",
    }

    for key in rename:
        if g5.join(source_root, key) not in src:
            continue
        g5.copy(src, dest, g5.join(source_root, key), rename[key])
        paths.remove(g5.join(source_root, key))

    return paths


def TransformDeprecated(cli_args=None):
    """
    Transform old data structure to the current one.
    This code is considered 'non-maintained'.

    To check::

        G5compare \
            -r "/meta/seed_base" "/realisation/seed" \
            -r "/meta/normalisation" "/param/normalisation" \
            -r "/alpha" "/param/alpha" \
            -r "/rho" "/param/rho" \
            -r "/conn" "/param/conn" \
            -r "/coor" "/param/coor" \
            -r "/dofs" "/param/dofs" \
            -r "/iip" "/param/iip" \
            -r "/cusp" "/param/cusp" \
            -r "/cusp/epsy/k" "/param/cusp/epsy/weibull/k" \
            -r "/cusp/epsy/eps0" "/param/cusp/epsy/weibull/typical" \
            -r "/cusp/epsy/eps_offset" "/param/cusp/epsy/weibull/offset" \
            -r "/elastic" "/param/elastic" \
            -r "/run/dt" "/param/dt" \
            -r "/run/epsd/kick" "/param/cusp/epsy/deps" \
            -r "/disp" "/QuasiStatic/u" \
            -r "/kick" "/QuasiStatic/kick" \
            -r "/t" "/QuasiStatic/inc" \
            -r "/meta/Run_generate" "/meta/QuasiStatic_Generate" \
            -r "/meta/Run" "/meta/QuasiStatic_Run" \
            foo.h5.bak foo.h5
    """

    class MyFmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.MetavarTypeHelpFormatter,
    ):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="File to transform: .bak appended")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert not os.path.isfile(args.file + ".bak")
    os.rename(args.file, args.file + ".bak")

    with h5py.File(args.file + ".bak") as src, h5py.File(args.file, "w") as dest:
        paths = list(g5.getdatapaths(src, fold="/meta/normalisation", fold_symbol=""))
        paths = transform_deprecated_param(src, dest, paths)
        dest.create_group("QuasiStatic")

        rename = {f"/disp/{i}": f"/QuasiStatic/u/{i}" for i in src["disp"]}
        rename["/kick"] = "/QuasiStatic/kick"
        rename["/meta/EnsembleInfo"] = "/meta/QuasiStatic_EnsembleInfo"
        rename["/meta/Run_generate"] = "/meta/QuasiStatic_Generate"
        rename["/meta/Run"] = "/meta/QuasiStatic_Run"
        rename["/meta/normalisation"] = "/param/normalisation"

        for key in rename:
            if key not in src:
                continue
            g5.copy(src, dest, key, rename[key])
            paths.remove(key)

        dest["/QuasiStatic/inc"] = np.round(src["/t"][...] / src["/run/dt"][...]).astype(np.uint64)
        paths.remove("/t")

        assert "/param/normalisation" in dest
        assert np.all(src["/stored"][...] == np.arange(dest["/QuasiStatic/inc"].size))
        paths.remove("/stored")
        paths.remove("/disp")

        dest.create_group("/meta/QuasiStatic_TransformDeprecated").attrs["version"] = version

        if "uuid" not in dest["/meta/QuasiStatic_Run"].attrs:
            dest["/meta/QuasiStatic_Run"].attrs["uuid"] = str(uuid.uuid4())

        assert len(paths) == 0
