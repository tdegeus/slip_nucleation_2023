"""
Rerun step (quasi-static step, or trigger) to extract the dynamic evolution of fields.
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
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm
import XDMFWrite_h5py as xh
from numpy.typing import ArrayLike

from . import QuasiStatic
from . import storage
from . import tag
from . import tools
from ._version import version

file_defaults = dict(
    AverageSystemSpanning="MeasureDynamics_average_systemspanning.h5",
    PlotMeshHeight="MeasureDynamics_plot_height",
)


def elements_at_height(
    coor: ArrayLike, conn: ArrayLike, height: float, return_type: bool = False
) -> np.ndarray:
    """
    Get elements at a 'normal' row of elements at a certain height above the interface.

    :param coor: Nodal coordinates [nnode, ndim].
    :param conn: Connectivity [nelem, nne].
    :param height: Height in units of the linear block size of the middle layer.
    :param return_type: Extra return argument: layer if normal (``True``) or refinement (``False``).
    :return: List of element numbers.
    """

    mesh = GooseFEM.Mesh.Quad4.FineLayer(coor=coor, conn=conn)

    dy = mesh.elemrow_nhy
    normal = mesh.elemrow_type == -1
    mid = int((dy.size - dy.size % 2) / 2)
    dy = dy[mid:]
    normal = normal[mid:]
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

    if return_type:
        return mesh.elementsLayer(mid + i), normal[i]

    return mesh.elementsLayer(mid + i)


def PlotMeshHeight(cli_args=None):
    """
    Plot geometry with elements at certain heights marked.
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

    # developer options
    parser.add_argument("-v", "--version", action="version", version=version)

    # output
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument(
        "-o", "--output", type=str, default=file_defaults[funcname], help="Basename of output"
    )

    # input
    parser.add_argument("--height", type=float, action="append", help="Add element row(s)")
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    args.height = [] if args.height is None else args.height
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(args.output + ".h5", args.force)
    tools._check_overwrite_file(args.output + ".xdmf", args.force)

    with h5py.File(args.file) as file:
        system = QuasiStatic.System(file)
        coor = np.copy(system.coor)
        conn = np.copy(system.conn)

    element_list = [system.plastic_elem]
    name = ["weak"]
    for height in args.height:
        element_list.append(elements_at_height(coor, conn, height))
        name.append(str(height))

    with h5py.File(args.output + ".h5", "w") as file:
        file["/coor"] = coor
        file["/conn"] = conn
        opts = []
        for i in range(len(element_list)):
            e = np.zeros(conn.shape[0], dtype=np.bool)
            e[element_list[i]] = True
            file[f"/elements/{i:d}"] = e
            opts += [xh.Attribute(file, f"/elements/{i:d}", "Cell", name[i])]

        grid = xh.Grid(xh.Unstructured(file, "/coor", "/conn", "Quadrilateral"), *opts)
        xh.write(grid, args.output + ".xdmf")


def Run(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))

    # developer options
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-v", "--version", action="version", version=version)

    # output selection
    parser.add_argument("--A-step", type=int, default=1, help="Control sync-A storage")
    parser.add_argument("--t-step", type=int, default=500, help="Control sync-t storage")
    parser.add_argument("--height", type=float, action="append", help="Add element row(s)")

    # input selection
    parser.add_argument("--step", required=True, type=int, help="Quasistatic step to run")

    # output file
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # input files
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    args.height = [] if args.height is None else args.height
    assert os.path.isfile(args.file)
    assert args.A_step > 0 or args.t_step > 0

    with h5py.File(args.file) as src, h5py.File(args.output, "w") as file:
        g5.copy(src, file, ["/param", "/realisation", "/meta"])

        meta = QuasiStatic.create_check_meta(file, "/meta/Dynamics_Run", dev=args.develop)
        meta.attrs["file"] = os.path.basename(args.file)
        meta.attrs["step"] = args.step
        meta.attrs["A-step"] = args.A_step
        meta.attrs["t-step"] = args.t_step
        meta.attrs["height"] = args.height

        if "QuasiStatic" in src:
            typename = "QuasiStatic"
            sroot = src[typename]
            kick = sroot["kick"][args.step]
        elif "Trigger" in src:
            typename = "Trigger"
            sroot = src[typename]
            element = sroot["element"][args.step]
            meta.attrs["element"] = element
            assert not sroot["truncated"][args.step - 1]
            assert element >= 0

        # metadata & storage preparation

        meta.attrs["type"] = typename

        root = file.create_group("Dynamics")
        root.create_dataset("inc", maxshape=(None,), data=[sroot["inc"][args.step - 1]])
        storage.create_extendible(root, "A", np.uint64, desc='Size "A" of each stored item')
        storage.create_extendible(root, "sync-A", np.uint64, desc="Items stored due to sync-A")
        storage.create_extendible(root, "sync-t", np.uint64, desc="Items stored due to sync-t")
        storage.symtens2_create(
            root, "Epsbar", float, desc="Macroscopic strain tensor per item (real units)"
        )
        storage.symtens2_create(
            root, "Sigbar", float, desc="Macroscopic stress tensor per item (real units)"
        )

        root.create_group("u").attrs["desc"] = 'Displacement of selected DOFs (see "/doflist")'

        # restore state

        system = QuasiStatic.System(file)
        system.restore_quasistatic_step(sroot, args.step - 1)
        deps = file["/param/cusp/epsy/deps"][...]
        i_n = np.copy(system.plastic.i[:, 0].astype(int))
        i = np.copy(i_n)
        maxiter = sroot["inc"][args.step] - sroot["inc"][args.step - 1]

    with h5py.File(args.output, "a") as file:
        root = file["Dynamics"]

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
            root,
            "doflist",
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
                    root["u"][str(istore)] = system.vector.AsDofs(system.u)[keepdof]
                    Epsbar = np.average(system.Eps(), weights=dV, axis=(0, 1))
                    Sigbar = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    storage.symtens2_extend(root, "Epsbar", istore, Epsbar)
                    storage.symtens2_extend(root, "Sigbar", istore, Sigbar)
                    storage.dset_extend1d(root, "inc", istore, system.inc)
                    storage.dset_extend1d(root, "A", istore, np.sum(np.not_equal(i, i_n)))
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
                if typename == "Trigger":
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
                    storage.dset_extend1d(root, "sync-A", A_istore, istore)
                    A_istore += 1
                    store = True
                    A_next += args.A_step

                if A == N:
                    storage.dset_extend1d(root, "sync-t", t_istore, istore)
                    t_istore += 1
                    store = True
                    A_check = False
                    if args.t_step == 0:
                        stop = True

            else:
                inc_n = system.inc
                ret = system.minimise(max_iter=args.t_step, max_iter_is_error=False, nmargin=5)
                stop = ret == 0
                iiter += system.inc - inc_n
                storage.dset_extend1d(root, "sync-t", t_istore, istore)
                t_istore += 1
                store = True

        file["/meta/Dynamics_Run"].attrs["completed"] = 1


def RunHighFrequency(cli_args=None):
    """
    Perform a high-frequency measurement on very few observables:

    -   Average stress at the boundary.
    -   Displacement, velocity, and acceleration sensors at nodes at some ("x", "y") coordinates.
        These nodes are selected by specifying the number of sensors in each direction.
        The vertical sensors are placed in the top half of the domain.
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

    # developer options
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-v", "--version", action="version", version=version)

    # output selection
    parser.add_argument("--nx", type=int, default=6, help="Number of sensors in x-direction")
    parser.add_argument("--ny", type=int, default=2, help="Number of sensors in y-direction")
    parser.add_argument(
        "--height",
        type=int,
        action="append",
        default=[],
        help="Height of extra sensors (units: block size)",
    )

    # input selection
    parser.add_argument("--step", required=True, type=int, help="Quasistatic step to run")

    # output file
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # input files
    parser.add_argument("file", type=str, help="Simulation from which to run (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert args.nx > 0
    assert args.ny > 0

    with h5py.File(args.file) as src, h5py.File(args.output, "w") as file:
        g5.copy(src, file, ["/param", "/realisation", "/meta"])

        meta = QuasiStatic.create_check_meta(
            file, "/meta/Dynamics_RunHighFrequency", dev=args.develop
        )
        meta.attrs["file"] = os.path.basename(args.file)
        meta.attrs["step"] = args.step

        # select nodes to store

        coor = file["/param/coor"][...]
        conn = file["/param/conn"][...]
        mesh = GooseFEM.Mesh.Quad4.FineLayer(coor, conn)
        dy = mesh.elemrow_nhy
        mid = int((dy.size - 1) / 2)
        dy = dy[mid:]
        y = np.cumsum(dy) - dy[0]
        nodes = []

        # N.B. y[-1] is boring as the nodes are fixed
        for height in args.height + list(np.linspace(0, y[-2], args.ny)):
            elements, normal = elements_at_height(coor, conn, height, return_type=True)
            if not normal:
                elements = elements.reshape(-1, 4)[:, -1]
            nds = conn[elements, 3]
            idx = np.round(np.linspace(0, len(nds) - 1, args.nx)).astype(int)
            nodes += list(nds[idx])
        nodes = np.unique(np.sort(nodes))
        top = mesh.nodesTopEdge
        L = coor[top[1], 0] - coor[top[0], 0]

        meta.attrs["nodes"] = nodes
        meta.attrs["top"] = top

        # sort of triggering

        if "QuasiStatic" in src:
            typename = "QuasiStatic"
            sroot = src[typename]
            kick = sroot["kick"][args.step]
        elif "Trigger" in src:
            typename = "Trigger"
            sroot = src[typename]
            element = sroot["element"][args.step]
            meta.attrs["element"] = element
            assert not sroot["truncated"][args.step - 1]
            assert element >= 0

        # restore state

        system = QuasiStatic.System(file)
        system.restore_quasistatic_step(sroot, args.step - 1)
        deps = file["/param/cusp/epsy/deps"][...]

    with h5py.File(args.output, "a") as file:
        root = file.create_group("DynamicsHighFrequency")

        root["u0_x"] = coor[nodes, 0]
        root["u0_y"] = coor[nodes, 1]

        if typename == "Trigger":
            system.triggerElementWithLocalSimpleShear(deps, element)
        else:
            system.initEventDrivenSimpleShear()
            system.eventDrivenStep(deps, kick)

        fext, u_x, u_y = system.minimise_highfrequency(nodes, top, nmargin=1)
        root["fext"] = fext / L
        root["u_x"] = u_x
        root["u_y"] = u_y


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
    (not on all blocks).
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


def AverageSystemSpanning(cli_args=None):
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
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=textwrap.dedent(doc))
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("files", nargs="*", type=str, help="See Dynamics_Run")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)
    QuasiStatic.create_check_meta(dev=args.develop)

    # get duration of each event and allocate binning on duration since system spanning

    t_start = []
    t_end = []

    for ifile, filepath in enumerate(args.files):
        with h5py.File(filepath, "r") as file:
            root = file["Dynamics"]

            if ifile == 0:
                system = QuasiStatic.System(file)
                N = system.N
                dV = system.dV()
                dV2 = system.dV(rank=2)

                height = file["/meta/Dynamics_Run"].attrs["height"]
                t_step = file["/meta/Dynamics_Run"].attrs["t-step"]
                dt = float(file["/param/dt"][...])

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
                assert np.all(np.equal(height, file["/meta/Dynamics_Run"].attrs["height"]))
                assert t_step == file["/meta/Dynamics_Run"].attrs["t-step"]
                assert dt == float(file["/param/dt"][...])
                assert np.all(np.equal(doflist, root["doflist"][...]))

            t = root["inc"][...] * dt / system.t0
            A = root["A"][...]
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

    for title in ["align", "align_moving"]:
        syncA[title] = {}
        syncA[title][0] = dict(
            Eps=AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[0], dV=dV2),
            Sig=AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[0], dV=dV2),
            s=AlignedAverage(shape=(N + 1, N), elements=element_list[0], dV=dV),
            epsp=AlignedAverage(shape=(N + 1, N), elements=element_list[0], dV=dV),
        )
        for i in range(1, len(element_list)):
            syncA["align"][i] = {
                "Eps": AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[i], dV=dV2),
                "Sig": AlignedAverage(shape=(N + 1, N, 2, 2), elements=element_list[i], dV=dV2),
            }

    # averages

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm(args.files)
    pbar.set_description(fmt.format(""))

    for ifile, filepath in enumerate(pbar):
        pbar.set_description(fmt.format(filepath), refresh=True)

        with h5py.File(filepath, "r") as file:
            root = file["Dynamics"]

            if ifile > 0:
                system.reset(file)

            # determine duration bin, ensure that only one measurement per bin is added
            # (take the one closest to the middle of the bin)

            ver = file["/meta/Dynamics_Run"].attrs["version"]

            nitem = root["inc"].size
            items_syncA = root["sync-A"][...]
            A = root["A"][...]
            t = root["inc"][...] * dt / system.t0
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

            Epsbar = storage.symtens2_read(root, "Epsbar") / system.eps0
            Sigbar = storage.symtens2_read(root, "Sigbar") / system.sig0

            for i in range(2):
                for j in range(2):
                    Epsbar[:, i, j] -= Epsbar[0, i, j]

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

                udof = np.zeros(system.vector.shape_dofval)
                udof[doflist] = root["u"][str(item)][...]
                system.u = system.vector.AsNode(udof)

                if item == 0:
                    i_n = np.copy(system.plastic.i.astype(int)[:, 0])
                    epsp_n = np.copy(system.plastic.epsp)
                    Eps_n = system.Eps()

                i = system.plastic.i.astype(int)[:, 0]
                broken = i != i_n
                moving = np.argwhere(broken).ravel()
                if tag.greater(ver, "12.3"):
                    assert np.sum(broken) == root["A"][item]

                Eps = (system.Eps() - Eps_n) / system.eps0
                Sig = system.Sig() / system.sig0

                # convert epsp: [N, nip] -> [nelem, nip] (for simplicity below)
                epsp = np.zeros(Eps.shape[:2], dtype=float)
                epsp[system.plastic_elem] = (system.plastic.epsp - epsp_n) / system.eps0

                # convert s: [N] -> [nelem, nip] (for simplicity below)
                if item in items_syncA:
                    s = np.zeros(epsp.shape, np.int64)
                    s[system.plastic_elem] = (i - i_n).reshape(-1, 1)

                # synct / syncA

                for data, store, j in zip(
                    [synct, syncA],
                    [t_ibin[item] >= 0, item in items_syncA],
                    [t_ibin[item], A[item]],
                ):
                    if not store:
                        continue

                    for i in range(len(element_list)):
                        data[i]["Eps"].add_subsample(j, Eps)
                        data[i]["Sig"].add_subsample(j, Sig)

                    data[0]["epsp"].add_subsample(j, epsp)

                    if np.sum(broken) == 0:
                        continue

                    data[0]["epsp_moving"].add_subsample(j, epsp, moving)
                    data[0]["Eps_moving"].add_subsample(j, Eps, moving)
                    data[0]["Sig_moving"].add_subsample(j, Sig, moving)

                # syncA["align"]

                if item in items_syncA and np.sum(broken) > 0:
                    j = A[item]
                    roll = tools.center_avalanche(broken)

                    for i in range(1, len(element_list)):
                        syncA["align"][i]["Eps"].add_subsample(j, Eps, roll)
                        syncA["align"][i]["Sig"].add_subsample(j, Sig, roll)

                    syncA["align"][0]["Eps"].add_subsample(j, Eps, roll)
                    syncA["align"][0]["Sig"].add_subsample(j, Sig, roll)
                    syncA["align"][0]["epsp"].add_subsample(j, epsp, roll)
                    syncA["align"][0]["s"].add_subsample(j, s, roll)

                    syncA["align_moving"][0]["Eps"].add_subsample(j, Eps, roll, broken)
                    syncA["align_moving"][0]["Sig"].add_subsample(j, Sig, roll, broken)
                    syncA["align_moving"][0]["epsp"].add_subsample(j, epsp, roll, broken)
                    syncA["align_moving"][0]["s"].add_subsample(j, s, roll, broken)

    with h5py.File(args.output, "w") as file:
        QuasiStatic.create_check_meta(
            file, "/meta/Dynamics_AverageSystemSpanning", dev=args.develop
        )

        for title, data in zip(["sync-t", "sync-A"], [synct, syncA]):
            for key in ["delta_t", "Epsbar", "Sigbar"]:
                file[f"/{title}/{key}/first"] = data[key].first
                file[f"/{title}/{key}/second"] = data[key].second
                file[f"/{title}/{key}/norm"] = data[key].norm

            for i in range(len(element_list)):
                for key in data[i]:
                    file[f"/{title}/{i}/{key}/first"] = data[i][key].first
                    file[f"/{title}/{i}/{key}/second"] = data[i][key].second
                    file[f"/{title}/{i}/{key}/norm"] = data[i][key].norm

        for title in ["align", "align_moving"]:
            for i in syncA[title]:
                for key in syncA[title][i]:
                    file[f"/sync-A/{title}/{i}/{key}/first"] = syncA[title][i][key].first
                    file[f"/sync-A/{title}/{i}/{key}/second"] = syncA[title][i][key].second
                    file[f"/sync-A/{title}/{i}/{key}/norm"] = syncA[title][i][key].norm


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
            -r "/kick" "/Dynamics/kick" \
            -r "/dynamics" "/Dynamics" \
            -r "/meta/MeasureDynamics_run" "/meta/Dynamics_Run" \
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
        fold = ["/meta/normalisation", "/dynamics"]
        paths = list(g5.getdatapaths(src, fold=fold, fold_symbol=""))
        paths = QuasiStatic.transform_deprecated_param(src, dest, paths)
        dest.create_group("Flow")

        rename = {}
        rename["/meta/Run_generate"] = "/meta/QuasiStatic_Generate"
        rename["/meta/Run"] = "/meta/QuasiStatic_Run"
        rename["/meta/MeasureDynamics_run"] = "/meta/Dynamics_Run"
        rename["/meta/normalisation"] = "/param/normalisation"
        rename["/kick"] = "/Dynamics/kick"

        for key in src["dynamics"]:
            if key not in ["t", "stored"]:
                rename[f"/dynamics/{key}"] = f"/Dynamics/{key}"
                paths.append(f"/dynamics/{key}")

        for key in rename:
            if key not in src:
                continue
            g5.copy(src, dest, key, rename[key])
            paths.remove(key)

        t = src["/dynamics/t"][...]
        dt = src["/run/dt"][...]
        dest["/Dynamics/inc"] = np.round(t / dt).astype(np.uint64)
        paths.remove("/t")

        dest["/meta/Dynamics_Run"].attrs["element"] = src["/trigger/element"][1]
        paths.remove("/trigger/element")
        paths.remove("/trigger/branched")
        paths.remove("/trigger/truncated")

        assert "/param/normalisation" in dest
        assert np.all(src["/dynamics/stored"][...] == np.arange(dest["/Dynamics/inc"].size))
        paths.remove("/stored")
        paths.remove("/disp/0")
        paths.remove("/dynamics")

        if "/dynamics" in paths:
            paths.remove("/dynamics")

        dest.create_group("/meta/Dynamics_TransformDeprecated").attrs["version"] = version

        if len(paths) != 0:
            print(paths)

        assert len(paths) == 0
