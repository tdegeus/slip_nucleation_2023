"""
Rerun dynamics.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import enstat
import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM
import GooseHDF5
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
    cli_average="MeasureDynamics_average",
)

file_defaults = dict(
    cli_average="MeasureDynamics_average.h5",
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
    Rerun an event and store output at different increments that are selected at:
    *   Given event sizes "A" unit the event is system spanning (``--A-step`` controls interval).
    *   Given time-steps if no longer checking at "A" (interval controlled by ``--t-step``).

    Customisation:
    *   ``--t-step=0``: Break simulation when ``A = N``.
    *   ``--A-step=0``: Store at fixed time intervals from the beginning.

    Storage:
    *   An exact copy of the input file.
    *   The macroscopic stress tensor ("/dynamics/Sigbar/{iiter:d}").
    *   The macroscopic strain tensor ("/dynamics/Epsbar/{iiter:d}").
    *   The displacement ("/dynamics/u/{iiter:d}") of a selection of DOFs ("/dynamics/doflist") from:
        - elements along the weak layer,
        - element row(s) at (a) given height(s) above the weak layer.
        The mechanical state can be fully restored for the saved elements (but nowhere else).
    *   Metadata:
        - "/dynamics/t": Time.
        - "/dynamics/A": Actual number of blocks that yielded at least once.
        - "/dynamics/stored": The stored "iiter".
        - "/dynamics/sync-A": List of "iiter" stored due to given "A".
        - "/dynamics/sync-t": List of "iiter" stored due to given "t" after checking for "A".
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
    parser.add_argument("--t-step", type=int, default=500, help="Control sync-t storage")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-i", "--inc", required=True, type=int, help="Increment number")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--height", type=float, action="append", help="Add element row(s)")
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    args.height = [] if args.height is None else args.height
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output, args.force)
    assert args.A_step > 0 or args.t_step > 0

    # copy file

    if os.path.realpath(args.file) != os.path.realpath(args.output):

        with h5py.File(args.file) as src, h5py.File(args.output, "w") as dest:
            paths = list(GooseHDF5.getdatasets(src, fold="/disp"))
            assert "/disp/..." in paths
            paths.remove("/disp/...")
            GooseHDF5.copy(src, dest, paths, expand_soft=False)
            dest[f"/disp/{args.inc - 1:d}"] = src[f"/disp/{args.inc - 1:d}"][:]

    with h5py.File(args.output, "a") as file:

        # metadata & storage preparation

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        meta.attrs["file"] = os.path.basename(args.file)
        meta.attrs["A-step"] = args.A_step
        meta.attrs["t-step"] = args.t_step
        meta.attrs["height"] = args.height

        storage.create_extendible(
            file, "/dynamics/stored", np.uint64, desc='"iiter" of stored steps'
        )

        storage.create_extendible(
            file, "/dynamics/t", float, desc="Time of each stored step (real units)"
        )

        storage.create_extendible(
            file, "/dynamics/A", np.uint64, desc='Size "A" of each stored step'
        )

        storage.create_extendible(
            file, "/dynamics/sync-A", np.uint64, desc="Steps stored due to sync-A"
        )

        storage.create_extendible(
            file, "/dynamics/sync-t", np.uint64, desc="Steps stored due to sync-t"
        )

        storage.symtens2_create(file, "/dynamics/Epsbar", float, desc="Macroscopic strain tensor")
        storage.symtens2_create(file, "/dynamics/Sigbar", float, desc="Macroscopic stress tensor")

        file_u = file["dynamics"].create_group("u")
        file_u.attrs["desc"] = 'Displacement of selected DOFs (see "/doflist")'
        file_u.attrs["goal"] = "Reconstruct (part of) the system at that instance"
        file_u.attrs["groups"] = 'Items in "/stored"'

        # restore state

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

        element_list = list(system.plastic_elem)
        for height in args.height:
            element_list += list(elements_at_height(system.coor, system.conn, height))

        partial = tools.PartialDisplacement(
            conn=system.conn,
            dofs=system.dofs,
            element_list=np.unique(element_list),
        )

        del element_list

        keepdof = partial.dof_is_stored()
        dV = system.dV(rank=2)
        N = system.N

        storage.dump_with_atttrs(
            file,
            "/dynamics/doflist",
            partial.dof_list(),
            desc='List of stored DOFs (order corresponds to "u"',
        )

        # rerun dynamics and store every other time

        pbar = tqdm.tqdm(total=maxiter)
        pbar.set_description(args.output)

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
        last_stored_iiter = -1  # last written increment

        while True:

            if store:

                if iiter != last_stored_iiter:
                    file[f"/dynamics/u/{iiter:d}"] = system.vector.AsDofs(system.u)[keepdof]
                    Epsbar = np.average(system.Eps(), weights=dV, axis=(0, 1))
                    Sigbar = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    storage.symtens2_extend(file, "/dynamics/Epsbar", istore, Epsbar)
                    storage.symtens2_extend(file, "/dynamics/Sigbar", istore, Sigbar)
                    storage.dset_extend1d(file, "/dynamics/t", istore, system.t)
                    storage.dset_extend1d(file, "/dynamics/A", istore, A)
                    storage.dset_extend1d(file, "/dynamics/stored", istore, iiter)
                    file.flush()
                    istore += 1
                    last_stored_iiter = iiter
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
                    storage.dset_extend1d(file, "/dynamics/sync-A", A_istore, iiter)
                    A_istore += 1
                    store = True
                    A_next += args.A_step

                if A == N:
                    storage.dset_extend1d(file, "/dynamics/sync-t", t_istore, iiter)
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
                storage.dset_extend1d(file, "/dynamics/sync-t", t_istore, iiter)
                t_istore += 1
                store = True

        meta.attrs["completed"] = 1


def cli_average(cli_args=None):
    """
    ???
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
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("files", nargs="*", type=str, help="See " + entry_points["cli_run"])

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    # get duration of each event and allocate binning on duration since system spanning

    t_start = []
    t_end = []

    for ifile, filepath in enumerate(args.files):

        with h5py.File(filepath, "r") as file:

            if ifile == 0:

                system = QuasiStatic.System(file)
                N = system.N
                dV = system.dV()
                dV2 = system.dV(rank=2)

                height = file[f"/meta/{entry_points['cli_run']}"].attrs["height"]
                t_step = file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                dt = float(file["/run/dt"][...])

                element_list = list(system.plastic_elem)
                eh = []

                for h in height:
                    elem = elements_at_height(system.coor, system.conn, h)
                    element_list += list(elem)
                    eh.append(elem)

                partial = tools.PartialDisplacement(
                    conn=system.conn,
                    dofs=system.dofs,
                    element_list=np.unique(element_list),
                )

                doflist = partial.dof_list()

            else:

                assert N == system.N
                assert height == file[f"/meta/{entry_points['cli_run']}"].attrs["height"]
                assert t_step == file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                assert dt == float(file["/run/dt"][...])
                assert np.all(doflist, file["/dynamics/doflist"][...])

            t = file["/dynamics/t"][...] / system.t0
            A = file["/dynamics/A"][...]
            t = np.sort(t) - np.min(t[A == N])
            t_start.append(t[0])
            t_end.append(t[-1])

    Dt = t_step * dt
    t_bin = np.arange(np.min(t_start) - Dt, np.max(t_end) + 2 * Dt, Dt)
    t_bin = t_bin - np.min(t_bin[t_bin > 0])
    t_mid = 0.5 * (t_bin[1:] + t_bin[:-1])

    def allocate(n, height):
        ret = dict(
            delta_t=enstat.static(shape=[n]),
            macroscopic={
                "Eps": enstat.static(shape=[n, 2, 2]),
                "Sig": enstat.static(shape=[n, 2, 2]),
            },
            weak={
                "Eps": enstat.static(shape=[n, 2, 2]),
                "Sig": enstat.static(shape=[n, 2, 2]),
                "epsp": enstat.static(shape=[n]),
                "Eps_moving": enstat.static(shape=[n, 2, 2]),
                "Sig_moving": enstat.static(shape=[n, 2, 2]),
                "epsp_moving": enstat.static(shape=[n]),
            },
        )

        for i in range(len(height)):
            ret[i] = {
                "Eps": enstat.static(shape=[n, 2, 2]),
                "Sig": enstat.static(shape=[n, 2, 2]),
            }

        return ret

    def add_my_sample(average, index, data):
        average.first[index, ...] += data
        average.second[index, ...] += data**2
        average.norm[index, ...] += 1

    synct = allocate(t_bin.size - 1, height)
    syncA = allocate(N + 1, height)
    syncA["align"] = {}
    syncA["align"]["weak"] = dict(
        eps_xx=enstat.static(shape=[N + 1, N]),
        eps_xy=enstat.static(shape=[N + 1, N]),
        eps_yy=enstat.static(shape=[N + 1, N]),
        sig_xx=enstat.static(shape=[N + 1, N]),
        sig_xy=enstat.static(shape=[N + 1, N]),
        sig_yy=enstat.static(shape=[N + 1, N]),
        epsp=enstat.static(shape=[N + 1, N]),
        s=enstat.static(shape=[N + 1, N]),
    )
    for i in range(len(height)):
        syncA["align"][i] = dict(
            eps_xx=enstat.static(shape=[N + 1, N]),
            eps_xy=enstat.static(shape=[N + 1, N]),
            eps_yy=enstat.static(shape=[N + 1, N]),
            sig_xx=enstat.static(shape=[N + 1, N]),
            sig_xy=enstat.static(shape=[N + 1, N]),
            sig_yy=enstat.static(shape=[N + 1, N]),
        )

    # averages

    with h5py.File(args.output, "w") as output:

        fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
        pbar = tqdm.tqdm(args.files)
        pbar.set_description(fmt.format(""))

        for ifile, filepath in enumerate(pbar):

            pbar.set_description(fmt.format(filepath), refresh=True)

            with h5py.File(filepath, "r") as file:

                if ifile > 0:
                    system.reset(file)

                # determine duration bin, ensure that only one measurement per bin is added
                # (take the one closest to the middle of the bin)

                steps_syncA = file["/dynamics/sync-A"][...]
                A = file["/dynamics/A"][...]
                t = file["/dynamics/t"][...] / system.t0
                delta_t = t - np.min(t[A == N])
                t_ibin = np.digitize(delta_t, t_bin) - 1
                d = np.abs(delta_t - t_mid[t_ibin])

                for ibin in np.unique(t_ibin):
                    idx = np.argwhere(t_ibin == ibin).ravel()
                    if len(idx) <= 1:
                        continue
                    jdx = idx[np.argmin(d[idx])]
                    t_ibin[idx] = -1
                    t_ibin[jdx] = ibin

                del d

                # add averages

                Epsbar = storage.symtens2_read(file, "/dynamics/Epsbar") / system.eps0
                Sigbar = storage.symtens2_read(file, "/dynamics/Sigbar") / system.sig0

                keep = t_ibin >= 0
                add_my_sample(synct["delta_t"], t_ibin[keep], delta_t[keep])
                add_my_sample(synct["macroscopic"]["Eps"], t_ibin[keep], Epsbar[keep])
                add_my_sample(synct["macroscopic"]["Sig"], t_ibin[keep], Sigbar[keep])

                for step, iiter in enumerate(file["/dynamics/stored"][...]):

                    udof = np.zeros(system.vector.shape_dofval())
                    udof[doflist] = file[f"/dynamics/u/{iiter:d}"]
                    system.u = system.vector.AsNode(udof)

                    if step == 0:
                        i_n = np.copy(system.plastic.i.astype(int)[:, 0])

                    i = system.plastic.i.astype(int)[:, 0]
                    broken = i != i_n
                    assert np.sum(broken) == file["/dynamics/A"][step]

                    if iiter in steps_syncA or t_ibin[step] >= 0:

                        Eps = system.Eps() / system.eps0
                        Sig = system.Sig() / system.sig0

                        plas = system.plastic_elem
                        epsp = system.plastic.epsp
                        moving = plas[broken]

                    # synct

                    if t_ibin[step] >= 0:

                        add_my_sample(
                            synct["weak"]["Eps"],
                            t_ibin[step],
                            np.average(Eps[plas, ...], weights=dV2[plas, ...], axis=(0, 1)),
                        )
                        add_my_sample(
                            synct["weak"]["Sig"],
                            t_ibin[step],
                            np.average(Sig[plas, ...], weights=dV2[plas, ...], axis=(0, 1)),
                        )
                        add_my_sample(
                            synct["weak"]["epsp"],
                            t_ibin[step],
                            np.average(epsp, weights=dV[plas, ...], axis=(0, 1)),
                        )

                        if np.sum(broken) > 0:
                            add_my_sample(
                                synct["weak"]["Eps_moving"],
                                t_ibin[step],
                                np.average(Eps[moving, ...], weights=dV2[moving, ...], axis=(0, 1)),
                            )
                            add_my_sample(
                                synct["weak"]["Sig_moving"],
                                t_ibin[step],
                                np.average(Sig[moving, ...], weights=dV2[moving, ...], axis=(0, 1)),
                            )
                            add_my_sample(
                                synct["weak"]["epsp_moving"],
                                t_ibin[step],
                                np.average(epsp[broken], weights=dV[moving, ...], axis=(0, 1)),
                            )

                        for i in range(len(height)):
                            add_my_sample(
                                synct[i]["Eps"],
                                t_ibin[step],
                                np.average(Eps[eh[i], ...], weights=dV2[eh[i], ...], axis=(0, 1)),
                            )
                            add_my_sample(
                                synct[i]["Sig"],
                                t_ibin[step],
                                np.average(Sig[eh[i], ...], weights=dV2[eh[i], ...], axis=(0, 1)),
                            )

                    # syncA

                    if iiter not in steps_syncA:
                        continue

                    roll = tools.center_avalanche(broken)

                    add_my_sample(
                        syncA["weak"]["Eps"],
                        A[step],
                        np.average(Eps[plas, ...], weights=dV2[plas, ...], axis=(0, 1)),
                    )
                    add_my_sample(
                        syncA["weak"]["Sig"],
                        A[step],
                        np.average(Sig[plas, ...], weights=dV2[plas, ...], axis=(0, 1)),
                    )
                    add_my_sample(
                        syncA["weak"]["epsp"],
                        A[step],
                        np.average(epsp, weights=dV[plas, ...], axis=(0, 1)),
                    )

                    if np.sum(broken) > 0:
                        add_my_sample(
                            syncA["weak"]["Eps_moving"],
                            A[step],
                            np.average(Eps[moving, ...], weights=dV2[moving, ...], axis=(0, 1)),
                        )
                        add_my_sample(
                            syncA["weak"]["Sig_moving"],
                            A[step],
                            np.average(Sig[moving, ...], weights=dV2[moving, ...], axis=(0, 1)),
                        )
                        add_my_sample(
                            syncA["weak"]["epsp_moving"],
                            A[step],
                            np.average(epsp[broken], weights=dV[moving, ...], axis=(0, 1)),
                        )

                    for i in range(len(height)):
                        add_my_sample(
                            syncA[i]["Eps"],
                            A[step],
                            np.average(Eps[eh[i], ...], weights=dV2[eh[i], ...], axis=(0, 1)),
                        )
                        add_my_sample(
                            syncA[i]["Sig"],
                            A[step],
                            np.average(Sig[eh[i], ...], weights=dV2[eh[i], ...], axis=(0, 1)),
                        )

                    d = np.average(Eps[plas, ...], weights=dV2[plas, ...], axis=1)
                    add_my_sample(syncA["align"]["weak"]["eps_xx"], A[step], np.roll(d[..., 0, 0], roll))
                    add_my_sample(syncA["align"]["weak"]["eps_xy"], A[step], np.roll(d[..., 0, 1], roll))
                    add_my_sample(syncA["align"]["weak"]["eps_yy"], A[step], np.roll(d[..., 1, 1], roll))

                    d = np.average(Sig[plas, ...], weights=dV2[plas, ...], axis=1)
                    add_my_sample(syncA["align"]["weak"]["sig_xx"], A[step], np.roll(d[..., 0, 0], roll))
                    add_my_sample(syncA["align"]["weak"]["sig_xy"], A[step], np.roll(d[..., 0, 1], roll))
                    add_my_sample(syncA["align"]["weak"]["sig_yy"], A[step], np.roll(d[..., 1, 1], roll))

                    d = np.average(epsp, weights=dV[plas, ...], axis=1)
                    add_my_sample(syncA["align"]["weak"]["epsp"], A[step], np.roll(d, roll))
                    add_my_sample(syncA["align"]["weak"]["s"], A[step], np.roll(i - i_n, roll))

                    for i in range(len(height)):

                        assert np.all(np.diff(eh[i]) == 1)

                        c = np.average(Eps[eh[i], ...], weights=dV2[eh[i], ...], axis=1)
                        n = int(plas.size / len(eh[i]))
                        d_xx = np.empty(plas.size)
                        d_xy = np.empty(plas.size)
                        d_yy = np.empty(plas.size)
                        for j in range(n):
                            d_xx[j::n] = c[..., 0, 0]
                            d_xy[j::n] = c[..., 0, 1]
                            d_yy[j::n] = c[..., 1, 1]

                        add_my_sample(syncA["align"]["weak"]["eps_xx"], A[step], np.roll(d_xx, roll))
                        add_my_sample(syncA["align"]["weak"]["eps_xy"], A[step], np.roll(d_xy, roll))
                        add_my_sample(syncA["align"]["weak"]["eps_yy"], A[step], np.roll(d_yy, roll))

                        c = np.average(Sig[eh[i], ...], weights=dV2[eh[i], ...], axis=1)
                        n = int(plas.size / len(eh[i]))
                        d_xx = np.empty(plas.size)
                        d_xy = np.empty(plas.size)
                        d_yy = np.empty(plas.size)
                        for j in range(n):
                            d_xx[j::n] = c[..., 0, 0]
                            d_xy[j::n] = c[..., 0, 1]
                            d_yy[j::n] = c[..., 1, 1]

                        add_my_sample(syncA["align"]["weak"]["sig_xx"], A[step], np.roll(d_xx, roll))
                        add_my_sample(syncA["align"]["weak"]["sig_xy"], A[step], np.roll(d_xy, roll))
                        add_my_sample(syncA["align"]["weak"]["sig_yy"], A[step], np.roll(d_yy, roll))




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
        t: Elapsed time since start of the step [N + 1].
        mask: If ``True`` no data was read for that array [N + 1].
    """

    t = file["t"][...]
    doflist = file["/doflist"][...]
    udof = np.zeros(system.vector.shape_dofval())
    N = system.N

    ret = {}
    ret["Eps"] = np.empty((N + 1, N, 2, 2), dtype=float)
    ret["Sig"] = np.empty((N + 1, N, 2, 2), dtype=float)
    ret["epsp"] = np.empty((N + 1, N), dtype=float)
    ret["s"] = np.empty((N + 1, N), dtype=int)
    ret["t"] = np.zeros((N + 1), dtype=float)
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
        ret["t"][A] = t[step] / system.t0
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


def cli_average_bak(cli_args=None):
    """
    Compute the spatial average of growing events.
    At each "A" the avalanche is aligned and the ensemble average is taken.
    This approximates:
    *   The average is taken at different times, with the synchochronisation different at each "A".
    *   The alignment of avalanches is different at each "A".
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


def cli_spatialaverage_syncA_data(cli_args=None):
    """
    Treat raw data from :py:func:`cli_spatialaverage_syncA_raw` to get the spatial averages.
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
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("raw", type=str, help="See " + entry_points["cli_spatialaverage_syncA_raw"])

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.raw)
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.raw) as raw:

        for i, simid in enumerate(tqdm.tqdm(raw["full"])):

            spatial = dict(
                eps_xx=raw["full"][simid]["Eps"][...][..., 0, 0],
                eps_xy=raw["full"][simid]["Eps"][...][..., 0, 1],
                eps_yy=raw["full"][simid]["Eps"][...][..., 1, 1],
                sig_xx=raw["full"][simid]["Sig"][...][..., 0, 0],
                sig_xy=raw["full"][simid]["Sig"][...][..., 0, 1],
                sig_yy=raw["full"][simid]["Sig"][...][..., 1, 1],
                epsp=raw["full"][simid]["epsp"][...],
                s=raw["full"][simid]["s"][...],
            )
            t = raw["full"][simid]["t"][...]
            mask = raw["full"][simid]["mask"][...]
            istart = np.argmin(t[~mask])
            t = t - t[istart]
            spatial["epsp"] = spatial["epsp"] - spatial["epsp"][istart, :]

            shift = tools.center_avalanche_per_row(spatial["s"])
            for key in spatial:
                spatial[key] = tools.indep_roll(spatial[key], shift, axis=1)

            if i == 0:
                average = {
                    key: enstat.static(shape=spatial[key].shape, dtype=spatial[key].dtype)
                    for key in spatial
                }
                averate_t = enstat.static(shape=t.shape, dtype=t.dtype)
                N = spatial["epsp"].shape[1]

            for key in spatial:
                average[key].add_sample(spatial[key], mask=mask)
            averate_t.add_sample(t, mask=mask)

    with h5py.File(args.output, "w") as output:

        Eps = np.empty([N + 1, N, 2, 2])
        Eps[..., 0, 0] = average["eps_xx"].mean()
        Eps[..., 0, 1] = average["eps_xy"].mean()
        Eps[..., 1, 0] = average["eps_xy"].mean()
        Eps[..., 1, 1] = average["eps_yy"].mean()

        Sig = np.empty([N + 1, N, 2, 2])
        Sig[..., 0, 0] = average["sig_xx"].mean()
        Sig[..., 0, 1] = average["sig_xy"].mean()
        Sig[..., 1, 0] = average["sig_xy"].mean()
        Sig[..., 1, 1] = average["sig_yy"].mean()

        output["Eps"] = Eps
        output["Sig"] = Sig
        output["s"] = average["s"].mean()
        output["epsp"] = average["epsp"].mean()
        output["t"] = averate_t.mean()
