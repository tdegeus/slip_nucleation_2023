"""
Rerun dynamics.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM
import h5py
import numpy as np
import tqdm
from numpy.typing import ArrayLike

from . import QuasiStatic
from . import storage
from . import tools
from ._version import version

entry_points = dict(
    cli_run="MeasureDynamics_run",
    cli_ensembleinfo="MeasureDynamics_EnsembleInfo",
    cli_spatialaverage_syncA="MeasureDynamics_SpatialAverage_syncA",
)

file_defaults = dict(
    cli_ensembleinfo="MeasureDynamics_EnsembleInfo.h5",
    cli_spatialaverage_syncA="MeasureDynamics_SpatialAverage_syncA.h5",
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def elements_at_height(coor: ArrayLike, conn: ArrayLike, height: float) -> np.ndarray:
    """
    Get elements at a 'normal' row of elements at a certain height above the interface.

    :param coor: Nodal coordinates [nnode, ndim].
    :param conn: Connectivity [nelem, nne].
    :param height: Height in units of the linear block size of the middle layer.
    :return: List of element numbers.
    """

    mesh = GooseFEM.Mesh.Quad4.FineLayer(coor=coor, conn=conn)

    dy = mesh.elemrow_nhy
    normal = mesh.elemrow_type == -1
    n = int((dy.size - dy.size % 2) / 2)
    dy = dy[n:]
    normal = normal[n:]
    y = np.cumsum(dy).astype(float) - dy[0]
    i = np.argmin(np.abs(y - height))

    while not normal[i]:
        if i == 1:
            i = 0
            break
        if i == normal.size - 1:
            break
        if np.abs(y[i - 1] - height) < np.abs(y[i + 1] - height):
            i -= 1
        else:
            i += 1

    return mesh.elementsLayer(n + i)


def cli_run(cli_args=None):
    """
    Rerun one increment and store output at different events sizes and/or times.
    By default output is stored at given event sizes ``A`` (interval controlled by ``--A-step``)
    and then at given time-steps (interval controlled by ``--t-step``)
    since the event was system spanning.
    The default behaviour can be modified as follows:

    *   ``--t-step=0``: Stop when ``A = N``
    *   ``--A-step=0``: Store are fixed time intervals

    By default, the macroscopic stress and stain and displacements at the weak layer are stored.
    Using the latter the full state of the interface can be restored
    (but nowhere else in the system).
    In stead, if ``--height`` is used, the displacements field for an element row at a given height
    above the interface is selected (see :py:func:`elements_at_height`).
    In that case the state of the interface cannot be restored.
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
    progname = entry_points[funcname]

    parser.add_argument("--A-step", type=int, default=1, help="Control sync-A storage")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--t-step", type=int, default=500, help="Control sync-A storage")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-i", "--inc", required=True, type=int, help="Increment number")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-y", "--height", type=float, help="Select element row")
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)
    assert args.A_step > 0 or args.t_step > 0

    # restore state

    with h5py.File(args.file, "r") as file:

        system = QuasiStatic.System(file)
        system.restore_step(file, args.inc - 1)
        deps = file["/run/epsd/kick"][...]
        i_n = np.copy(system.plastic.i[:, 0].astype(int))
        maxiter = int((file["/t"][args.inc] - file["/t"][args.inc - 1]) / file["/run/dt"][...])

        if "trigger" in file:
            element = file["/trigger/element"][args.inc]
            kick = None
        else:
            kick = file["/kick"][args.inc]

    # variables needed to write output

    if args.height is not None:
        element_list = elements_at_height(system.coor, system.conn, args.height)
    else:
        element_list = system.plastic_elem

    partial = tools.PartialDisplacement(
        conn=system.conn,
        dofs=system.dofs,
        element_list=element_list,
    )
    dofstore = partial.dof_is_stored()
    doflist = partial.dof_list()
    dV = system.dV(rank=2)
    N = system.N

    # rerun dynamics and store every other time

    pbar = tqdm.tqdm(total=maxiter)
    pbar.set_description(args.output)

    with h5py.File(args.output, "w") as file:

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = os.path.basename(args.file)

        A = 0  # maximal A encountered so far
        A_next = 0  # next A at which to write output
        A_istore = 0  # storage index for sync-A storage
        t_istore = 0  # storage index for sync-t storage
        A_check = args.A_step > 0  # switch to store sync-A
        trigger = True  # signal if trigger is needed
        store = True  # signal if storage is needed
        iiter = 0  # total number of elapsed iterations
        istore = 0  # index of the number of storage steps
        stop = False
        last_iiter = -1  # last written increment

        storage.create_extendible(file, "/t", float, desc="Time of each stored step (real units)")
        storage.create_extendible(file, "/A", np.uint64, desc="'A' of each stored step")
        storage.create_extendible(file, "/stored", np.uint64, desc="Stored steps")
        storage.create_extendible(file, "/sync-A/stored", np.uint64, desc="Stored step")
        storage.create_extendible(file, "/sync-t/stored", np.uint64, desc="Stored step")
        storage.dump_with_atttrs(file, "/doflist", doflist, desc="Index of each of the stored DOFs")

        while True:

            if store:

                if iiter != last_iiter:
                    file[f"/Eps/{iiter:d}"] = np.average(system.Eps(), weights=dV, axis=(0, 1))
                    file[f"/Sig/{iiter:d}"] = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    file[f"/u/{iiter:d}"] = system.vector.AsDofs(system.u)[dofstore]
                    storage.dset_extend1d(file, "/t", istore, system.t)
                    storage.dset_extend1d(file, "/A", istore, A)
                    storage.dset_extend1d(file, "/stored", istore, iiter)
                    file.flush()
                    istore += 1
                    last_iiter = iiter
                store = False
                pbar.n = iiter
                pbar.refresh()

            if stop:

                break

            if trigger:

                trigger = False
                if kick is None:
                    system.triggerElementWithLocalSimpleShear(deps, element)
                else:
                    system.initEventDrivenSimpleShear()
                    system.eventDrivenStep(deps, kick)

            if A_check:

                niter = system.timeStepsUntilEvent()
                iiter += niter
                stop = niter == 0
                i = system.plastic.i[:, 0].astype(int)
                a = np.sum(np.not_equal(i, i_n))
                A = max(A, a)

                if (A >= A_next and A % args.A_step == 0) or A == N:
                    storage.dset_extend1d(file, "/sync-A/stored", A_istore, iiter)
                    A_istore += 1
                    store = True
                    A_next += args.A_step

                if A == N:
                    storage.dset_extend1d(file, "/sync-t/stored", t_istore, iiter)
                    t_istore += 1
                    store = True
                    A_check = False
                    if args.t_step == 0:
                        stop = True

            else:

                inc_n = system.inc
                ret = system.minimise(max_iter=args.t_step, max_iter_is_error=False)
                iiter += system.inc - inc_n
                stop = ret == 0
                storage.dset_extend1d(file, "/sync-t/stored", t_istore, iiter)
                t_istore += 1
                store = True

        file["Eps"].attrs["desc"] = "Average strain of each stored step (real units)"
        file["Sig"].attrs["desc"] = "Average stress of each stored step (real units)"

        meta.attrs["completed"] = 1


def basic_output(system: QuasiStatic.System, file: h5py.File, verbose: bool = True) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.

    :return: Basic information as follows::
        Epsbar: Macroscopic strain tensor [ninc].
        Sigbar: Macroscopic stress tensor [ninc].
        Eps: Strain tensor, averaged on the interface [ninc].
        Sig: Stress tensor, averaged on the interface [ninc].
        epsp: Plastic strain, averaged on the interface [ninc].
        Eps_moving: Strain tensor, averaged on moving blocks [ninc].
        Sig_moving: Stress tensor, averaged on moving blocks [ninc].
        epsp_moving: Plastic strain, averaged on moving blocks [ninc].
        S: Number of times a block yielded [ninc].
        A: Number of blocks that yielded at least once [ninc].
        t: Time [ninc].
    """

    doflist = file["/doflist"][...]
    udof = np.zeros(system.vector.shape_dofval())
    dVs = system.plastic_dV()
    dV = system.plastic_dV(rank=2)

    n = file["/stored"].size
    ret = {}
    ret["Epsbar"] = np.zeros((n, 2, 2), dtype=float)
    ret["Sigbar"] = np.zeros((n, 2, 2), dtype=float)
    ret["Eps"] = np.zeros((n, 2, 2), dtype=float)
    ret["Sig"] = np.zeros((n, 2, 2), dtype=float)
    ret["epsp"] = np.zeros(n, dtype=float)
    ret["Eps_moving"] = np.zeros((n, 2, 2), dtype=float)
    ret["Sig_moving"] = np.zeros((n, 2, 2), dtype=float)
    ret["epsp_moving"] = np.zeros(n, dtype=float)
    ret["S"] = np.zeros(n, dtype=int)
    ret["A"] = np.zeros(n, dtype=int)

    for step, iiter in enumerate(file["/stored"][...]):

        udof[doflist] = file[f"/u/{iiter:d}"]
        system.u = system.vector.AsNode(udof)

        if step == 0:
            i_n = np.copy(system.plastic.i.astype(int)[:, 0])

        i = system.plastic.i.astype(int)[:, 0]
        c = i != i_n

        Eps = system.plastic.Eps / system.eps0
        Sig = system.plastic.Sig / system.sig0
        epsp = system.plastic.epsp / system.eps0

        ret["Epsbar"][step, ...] = file[f"/Eps/{iiter:d}"][...] / system.eps0
        ret["Sigbar"][step, ...] = file[f"/Sig/{iiter:d}"][...] / system.sig0
        ret["Eps"][step, ...] = np.average(Eps, weights=dV, axis=(0, 1))
        ret["Sig"][step, ...] = np.average(Sig, weights=dV, axis=(0, 1))
        ret["epsp"][step] = np.average(epsp, weights=dVs, axis=(0, 1))
        if np.sum(c) > 0:
            ret["Eps_moving"][step, ...] = np.average(Eps[c], weights=dV[c], axis=(0, 1))
            ret["Sig_moving"][step, ...] = np.average(Sig[c], weights=dV[c], axis=(0, 1))
            ret["epsp_moving"][step] = np.average(epsp[c], weights=dVs[c], axis=(0, 1))
        ret["S"][step] = np.sum(i - i_n)
        ret["A"][step] = np.sum(i != i_n)

    assert np.all(file["/A"][...] >= ret["A"])
    ret["t"] = file["/t"][...] / system.t0

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def basic_spatial_sync_A(system: QuasiStatic.System, file: h5py.File, verbose: bool = True) -> dict:
    """
    Read basic output from simulation.
    The output is given at fixed "A" and per block along the interface.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.

    :return: Basic information [A, block, ...]::
        Eps: Strain tensor [N + 1, N, 2, 2].
        Sig: Stress tensor [N + 1, N, 2, 2].
        epsp: Plastic strain [N + 1, N].
        s: Number of times a block yielded [N + 1, N].
        mask: If ``True`` no data was read for that array [N + 1].
    """

    doflist = file["/doflist"][...]
    udof = np.zeros(system.vector.shape_dofval())
    N = system.N

    ret = {}
    ret["Eps"] = np.empty((N + 1, N, 2, 2), dtype=float)
    ret["Sig"] = np.empty((N + 1, N, 2, 2), dtype=float)
    ret["epsp"] = np.empty((N + 1, N), dtype=float)
    ret["s"] = np.empty((N + 1, N), dtype=int)
    ret["mask"] = np.ones((N + 1), dtype=bool)

    for step, iiter in enumerate(file["/stored"][...]):

        udof[doflist] = file[f"/u/{iiter:d}"]
        system.u = system.vector.AsNode(udof)

        if step == 0:
            i_n = np.copy(system.plastic.i.astype(int)[:, 0])

        i = system.plastic.i.astype(int)[:, 0]
        A = np.sum(i != i_n)

        if not ret["mask"][A]:
            continue

        ret["mask"][A] = False
        ret["Eps"][A, ...] = np.mean(system.plastic.Eps / system.eps0, axis=1)
        ret["Sig"][A, ...] = np.mean(system.plastic.Sig / system.sig0, axis=1)
        ret["epsp"][A, ...] = np.mean(system.plastic.epsp / system.eps0, axis=1)
        ret["s"][A, ...] = i - i_n

        if A >= N:
            break

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def cli_ensembleinfo(cli_args=None):
    """
    Read and store basic info from individual runs.
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
    entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-p", "--sourcedir", type=str, default=".", help="Directory with sim files")
    parser.add_argument("files", nargs="*", type=str, help="See " + entry_points["cli_run"])

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm(args.files)
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:

        for ifile, filepath in enumerate(pbar):

            pbar.set_description(fmt.format(filepath), refresh=True)

            with h5py.File(filepath, "r") as file:

                meta = file[f"/meta/{entry_points['cli_run']}"]
                sourcepath = os.path.join(args.sourcedir, meta.attrs["file"])
                assert os.path.isfile(sourcepath)

                with h5py.File(sourcepath, "r") as source:
                    if ifile == 0:
                        system = QuasiStatic.System(source)
                    else:
                        system.reset(source)

                if ifile == 0:
                    partial = tools.PartialDisplacement(
                        conn=system.conn,
                        dofs=system.dofs,
                        dof_list=file["/doflist"][...],
                    )
                    assert np.all(np.in1d(system.plastic_elem, partial.element_list()))

                try:
                    out = basic_output(system, file, verbose=False)
                except AssertionError:
                    print(f'Treating "{filepath}" as broken')
                    continue

                for key in out:
                    output[f"/full/{os.path.basename(filepath)}/{key}"] = out[key]

        output["/stored"] = [os.path.basename(i) for i in args.files]


def cli_spatialaverage_syncA(cli_args=None):
    """
    Collect data to get the spatial average of growing events.
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
    entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-p", "--sourcedir", type=str, default=".", help="Source directory")
    parser.add_argument("files", nargs="*", type=str, help="See " + entry_points["cli_run"])

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm(args.files)
    pbar.set_description(fmt.format(""))

    with h5py.File(args.output, "w") as output:

        for ifile, filepath in enumerate(pbar):

            pbar.set_description(fmt.format(filepath), refresh=True)

            with h5py.File(filepath, "r") as file:

                meta = file[f"/meta/{entry_points['cli_run']}"]
                sourcepath = os.path.join(args.sourcedir, meta.attrs["file"])
                assert os.path.isfile(sourcepath)

                with h5py.File(sourcepath, "r") as source:
                    if ifile == 0:
                        system = QuasiStatic.System(source)
                    else:
                        system.reset(source)

                if ifile == 0:
                    partial = tools.PartialDisplacement(
                        conn=system.conn,
                        dofs=system.dofs,
                        dof_list=file["/doflist"][...],
                    )
                    assert np.all(np.in1d(system.plastic_elem, partial.element_list()))

                try:
                    out = basic_spatial_sync_A(system, file, verbose=False)
                except AssertionError:
                    print(f'Treating "{filepath}" as broken')
                    continue

                for key in out:
                    output[f"/full/{os.path.basename(filepath)}/{key}"] = out[key]

        output["/stored"] = [os.path.basename(i) for i in args.files]
