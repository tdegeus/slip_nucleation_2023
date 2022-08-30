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
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_average_systemspanning="MeasureDynamics_average_systemspanning",
    cli_run="MeasureDynamics_run",
)

file_defaults = dict(
    cli_average_systemspanning="MeasureDynamics_average_systemspanning.h5",
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
            file, "/dynamics/stored", np.uint64, desc="List with stored items"
        )
        storage.create_extendible(
            file, "/dynamics/t", float, desc="Time of each stored item (real units)"
        )
        storage.create_extendible(
            file, "/dynamics/A", np.uint64, desc='Size "A" of each stored item'
        )
        storage.create_extendible(
            file, "/dynamics/sync-A", np.uint64, desc="Items stored due to sync-A"
        )
        storage.create_extendible(
            file, "/dynamics/sync-t", np.uint64, desc="Items stored due to sync-t"
        )
        storage.symtens2_create(
            file, "/dynamics/Epsbar", float, desc="Macroscopic strain tensor per item (real units)"
        )
        storage.symtens2_create(
            file, "/dynamics/Sigbar", float, desc="Macroscopic stress tensor per item (real units)"
        )

        file_u = file["dynamics"].create_group("u")
        file_u.attrs["desc"] = 'Displacement of selected DOFs (see "/doflist")'
        file_u.attrs["goal"] = "Reconstruct (part of) the system at that instance"
        file_u.attrs["groups"] = 'Items in "/stored"'

        # restore state

        system = QuasiStatic.System(file)
        system.restore_step(file, args.inc - 1)
        deps = file["/run/epsd/kick"][...]
        i_n = np.copy(system.plastic.i[:, 0].astype(int))
        i = np.copy(i_n)
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
            desc='List of stored DOFs (same order as "u")',
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
                    file[f"/dynamics/u/{istore:d}"] = system.vector.AsDofs(system.u)[keepdof]
                    Epsbar = np.average(system.Eps(), weights=dV, axis=(0, 1))
                    Sigbar = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    storage.symtens2_extend(file, "/dynamics/Epsbar", istore, Epsbar)
                    storage.symtens2_extend(file, "/dynamics/Sigbar", istore, Sigbar)
                    storage.dset_extend1d(file, "/dynamics/t", istore, system.t)
                    storage.dset_extend1d(file, "/dynamics/A", istore, np.sum(np.not_equal(i, i_n)))
                    storage.dset_extend1d(file, "/dynamics/stored", istore, istore)
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
                    storage.dset_extend1d(file, "/dynamics/sync-A", A_istore, istore)
                    A_istore += 1
                    store = True
                    A_next += args.A_step

                if A == N:
                    storage.dset_extend1d(file, "/dynamics/sync-t", t_istore, istore)
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
                storage.dset_extend1d(file, "/dynamics/sync-t", t_istore, istore)
                t_istore += 1
                store = True

        meta.attrs["completed"] = 1


class BasicAverage(enstat.static):
    """
    Support class for :py:func:`cli_average_systemspanning`.
    This class writes on item at a time using :py:func:`BasicAverage.add_subsample`,
    and it does so by selecting the relevant elements from a field and
    computing the relevant volume average.
    """

    def __init__(self, shape, elements=None, dV=None):
        """
        :param shape: Shape of the averaged field: ``[nitem, ...]``.
        :param elements: Optional: list of elements to take from input data.
        :param dV: Optional: volume of each integration point, selected: ``dV[elements, :]``.
        """

        super().__init__(shape=shape)

        self.elem = elements
        self.dV = None if dV is None else dV[self.elem, ...]

        if self.elem is not None:
            assert np.all(np.equal(self.elem, np.sort(self.elem)))

    def add_subsample(self, index, data, moving=None):
        """
        :param index: Index of the item to add to the average.
        :param data: Data to add to the average.
        :param moving: Optional: the index of moving blocks (index relative to ``elements``).
        """

        if self.dV is not None:
            if moving is None:
                data = np.average(data[self.elem, ...], weights=self.dV, axis=(0, 1))
            else:
                if len(moving) == 0:
                    return
                data = np.average(
                    data[self.elem[moving], ...], weights=self.dV[moving, ...], axis=(0, 1)
                )

        self.first[index, ...] += data
        self.second[index, ...] += data**2
        self.norm[index, ...] += 1


class AlignedAverage(BasicAverage):
    """
    Support class for :py:func:`cli_average_systemspanning`.
    Similar to :py:class:`BasicAverage`, but it aligns blocks and averages per blocks
    (not on all elements).
    """

    def __init__(self, shape, elements, dV):
        super().__init__(shape=shape, elements=elements, dV=dV)
        self.n = int(shape[1] / self.elem.size)

    def add_subsample(self, index, data, roll, broken=None):
        """
        :param index: Index of the item to add to the average.
        :param data: Data to add to the average.
        :param roll: Roll to apply to align the data.
        :param broken: Array with per weak element whether the element is broken.
        """

        data = np.average(data[self.elem, ...], weights=self.dV, axis=1)

        if self.n > 1:
            tmp = np.empty(self.shape[1:], dtype=data.dtype)
            for i in range(self.n):
                tmp[i :: self.n, ...] = data  # noqa: E203
            data = tmp

        data = np.roll(data, roll, axis=0)

        if broken is None:
            self.first[index, ...] += data
            self.second[index, ...] += data**2
            self.norm[index, ...] += 1
        else:
            incl = np.roll(broken, roll)
            self.first[index, incl, ...] += data[incl]
            self.second[index, incl, ...] += data[incl] ** 2
            self.norm[index, incl, ...] += 1


def cli_average_systemspanning(cli_args=None):
    """
    Compute averages from output of :py:func:`cli_run`:

    -   'Simple' averages (macroscopic, per element row, on moving blocks):

        *   For bins of time compared to the time when the event is system-spanning.
        *   For fixed ``A``.

    -   'Aligned' averages (for different element rows), for fixed A.
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

                element_list = [system.plastic_elem]
                for h in height:
                    element_list.append(elements_at_height(system.coor, system.conn, h))

                partial = tools.PartialDisplacement(
                    conn=system.conn,
                    dofs=system.dofs,
                    element_list=np.unique([item for sublist in element_list for item in sublist]),
                )

                doflist = partial.dof_list()

            else:

                assert N == system.N
                assert np.all(
                    np.equal(height, file[f"/meta/{entry_points['cli_run']}"].attrs["height"])
                )
                assert t_step == file[f"/meta/{entry_points['cli_run']}"].attrs["t-step"]
                assert dt == float(file["/run/dt"][...])
                assert np.all(np.equal(doflist, file["/dynamics/doflist"][...]))

            t = file["/dynamics/t"][...] / system.t0
            A = file["/dynamics/A"][...]
            assert np.sum(A == N) > 0
            t = np.sort(t) - np.min(t[A == N])
            t_start.append(t[0])
            t_end.append(t[-1])

    Dt = t_step * dt / system.t0
    t_bin = np.arange(np.min(t_start), np.max(t_end) + 3 * Dt, Dt)
    t_bin = t_bin - np.min(t_bin[t_bin > 0]) - 0.5 * Dt
    t_mid = 0.5 * (t_bin[1:] + t_bin[:-1])

    # allocate averages

    def allocate(n, element_list, dV, dV2):

        ret = dict(
            delta_t=BasicAverage(shape=n),
            Epsbar=BasicAverage(shape=(n, 2, 2)),
            Sigbar=BasicAverage(shape=(n, 2, 2)),
        )

        for i in range(len(element_list)):
            ret[i] = {
                "Eps": BasicAverage(shape=(n, 2, 2), elements=element_list[i], dV=dV2),
                "Sig": BasicAverage(shape=(n, 2, 2), elements=element_list[i], dV=dV2),
            }

        ret[0]["epsp"] = BasicAverage(shape=n, elements=element_list[0], dV=dV)
        ret[0]["epsp_moving"] = BasicAverage(shape=n, elements=element_list[0], dV=dV)
        ret[0]["Eps_moving"] = BasicAverage(shape=(n, 2, 2), elements=element_list[0], dV=dV2)
        ret[0]["Sig_moving"] = BasicAverage(shape=(n, 2, 2), elements=element_list[0], dV=dV2)

        return ret

    synct = allocate(t_bin.size - 1, element_list, dV, dV2)
    syncA = allocate(N + 1, element_list, dV, dV2)

    syncA["align"] = {}
    for i in range(len(element_list)):
        syncA["align"][i] = {
            "Eps": AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[i], dV=dV2),
            "Sig": AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[i], dV=dV2),
        }
    syncA["align"][0]["s"] = AlignedAverage(shape=(N + 1, N), elements=element_list[0], dV=dV)
    syncA["align"][0]["epsp"] = AlignedAverage(shape=(N + 1, N), elements=element_list[0], dV=dV)

    # averages

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

            ver = file[f"/meta/{entry_points['cli_run']}"].attrs["version"]

            nitem = file["/dynamics/stored"].size
            assert np.all(np.equal(file["/dynamics/stored"][...], np.arange(nitem)))

            items_syncA = file["/dynamics/sync-A"][...]
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
            synct["delta_t"].add_subsample(t_ibin[keep], delta_t[keep])
            synct["Epsbar"].add_subsample(t_ibin[keep], Epsbar[keep])
            synct["Sigbar"].add_subsample(t_ibin[keep], Sigbar[keep])
            syncA["delta_t"].add_subsample(A[items_syncA], delta_t[items_syncA])
            syncA["Epsbar"].add_subsample(A[items_syncA], Epsbar[items_syncA])
            syncA["Sigbar"].add_subsample(A[items_syncA], Sigbar[items_syncA])

            for item in tqdm.tqdm(range(nitem)):

                if item not in items_syncA and t_ibin[item] < 0 and item > 0:
                    continue

                udof = np.zeros(system.vector.shape_dofval())
                udof[doflist] = file[f"/dynamics/u/{item:d}"]
                system.u = system.vector.AsNode(udof)

                if item == 0:
                    i_n = np.copy(system.plastic.i.astype(int)[:, 0])

                i = system.plastic.i.astype(int)[:, 0]
                broken = i != i_n
                moving = np.argwhere(broken).ravel()
                if tag.greater(ver, "12.3"):
                    assert np.sum(broken) == file["/dynamics/A"][item]

                Eps = system.Eps() / system.eps0
                Sig = system.Sig() / system.sig0

                # convert epsp: [N, nip] -> [nelem, nip] (for simplicity below)
                epsp = np.zeros(Eps.shape[:2], dtype=float)
                epsp[system.plastic_elem] = system.plastic.epsp

                # convert s: [N] -> [nelem, nip] (for simplicity below)
                if item in items_syncA:
                    s = np.zeros(epsp.shape, np.int64)
                    s[system.plastic_elem] = (i - i_n).reshape(-1, 1)

                # synct

                if t_ibin[item] >= 0:

                    for i in range(len(element_list)):
                        synct[i]["Eps"].add_subsample(t_ibin[item], Eps)
                        synct[i]["Sig"].add_subsample(t_ibin[item], Sig)

                    synct[0]["epsp"].add_subsample(t_ibin[item], epsp)
                    synct[0]["epsp_moving"].add_subsample(t_ibin[item], epsp, moving)
                    synct[0]["Eps_moving"].add_subsample(t_ibin[item], Eps, moving)
                    synct[0]["Sig_moving"].add_subsample(t_ibin[item], Sig, moving)

                # syncA

                if item in items_syncA:

                    for i in range(len(element_list)):
                        syncA[i]["Eps"].add_subsample(A[item], Eps)
                        syncA[i]["Sig"].add_subsample(A[item], Sig)

                    syncA[0]["epsp"].add_subsample(A[item], epsp)
                    syncA[0]["epsp_moving"].add_subsample(A[item], epsp, moving)
                    syncA[0]["Eps_moving"].add_subsample(A[item], Eps, moving)
                    syncA[0]["Sig_moving"].add_subsample(A[item], Sig, moving)

                    roll = tools.center_avalanche(broken)

                    for i in range(1, len(element_list)):
                        syncA["align"][i]["Eps"].add_subsample(A[item], Eps, roll)
                        syncA["align"][i]["Sig"].add_subsample(A[item], Sig, roll)

                    syncA["align"][0]["Eps"].add_subsample(A[item], Eps, roll, broken)
                    syncA["align"][0]["Sig"].add_subsample(A[item], Sig, roll, broken)
                    syncA["align"][0]["epsp"].add_subsample(A[item], epsp, roll, broken)
                    syncA["align"][0]["s"].add_subsample(A[item], s, roll, broken)

    with h5py.File(args.output, "w") as output:

        for title, data in zip(["sync-t", "sync-A"], [synct, syncA]):

            for key in ["delta_t", "Epsbar", "Sigbar"]:
                output[f"/{title}/{key}/first"] = data[key].first
                output[f"/{title}/{key}/second"] = data[key].second
                output[f"/{title}/{key}/norm"] = data[key].norm

            for i in range(len(element_list)):
                for key in data[i]:
                    output[f"/{title}/{i}/{key}/first"] = data[i][key].first
                    output[f"/{title}/{i}/{key}/second"] = data[i][key].second
                    output[f"/{title}/{i}/{key}/norm"] = data[i][key].norm

        for i in range(len(element_list)):
            for key in syncA["align"][i]:
                output[f"/sync-A/align/{i}/{key}/first"] = syncA["align"][i][key].first
                output[f"/sync-A/align/{i}/{key}/second"] = syncA["align"][i][key].second
                output[f"/sync-A/align/{i}/{key}/norm"] = syncA["align"][i][key].norm
