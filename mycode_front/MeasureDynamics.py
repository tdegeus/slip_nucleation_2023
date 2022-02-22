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

from . import storage
from . import System
from . import tools
from ._version import version

entry_points = dict(
    cli_run="MeasureDynamics_run",
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

    dy = mesh.elemrow_nhy()
    normal = mesh.elemrow_type() == -1
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

        system = System.init(file)
        System._restore_inc(file, system, args.inc - 1)
        deps = file["/run/epsd/kick"][...]
        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
        maxiter = int((file["/t"][args.inc] - file["/t"][args.inc - 1]) / file["/run/dt"][...])

        if "trigger" in file:
            element = file["/trigger/element"][args.inc]
            kick = None
        else:
            kick = file["/kick"][args.inc]

    # variables needed to write output

    if args.height is not None:
        element_list = elements_at_height(system.coor(), system.conn(), args.height)
    else:
        element_list = system.plastic()

    vector = system.vector()
    partial = tools.PartialDisplacement(
        conn=system.conn(),
        dofs=system.dofs(),
        element_list=element_list,
    )
    dofstore = partial.dof_is_stored()
    doflist = partial.dof_list()
    dV = system.quad().AsTensor(2, system.quad().dV())
    N = system.plastic().size

    # rerun dynamics and store every other time

    pbar = tqdm.tqdm(total=maxiter)
    pbar.set_description(args.output)

    with h5py.File(args.output, "w") as file:

        System.create_check_meta(file, f"/meta/{progname}", dev=args.develop)

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

        storage.create_extendible(file, "/t", float, desc="Time of each stored time-step")
        storage.create_extendible(file, "/A", np.uint64, desc="'A' of each stored time-step")
        storage.create_extendible(file, "/stored", np.uint64, desc="Stored time-steps")
        storage.create_extendible(file, "/sync-A/stored", np.uint64, desc="Stored time-steps")
        storage.create_extendible(file, "/sync-t/stored", np.uint64, desc="Stored time-steps")
        storage.dump_with_atttrs(file, "/doflist", doflist, desc="Index of each of the stored DOFs")

        while True:

            if store:

                if iiter != last_iiter:
                    file[f"/Eps/{iiter:d}"] = np.average(system.Eps(), weights=dV, axis=(0, 1))
                    file[f"/Sig/{iiter:d}"] = np.average(system.Sig(), weights=dV, axis=(0, 1))
                    file[f"/u/{iiter:d}"] = vector.AsDofs(system.u())[dofstore]
                    storage.dset_extend1d(file, "/t", istore, system.t())
                    storage.dset_extend1d(file, "/A", istore, A)
                    storage.dset_extend1d(file, "/stored", istore, iiter)
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
                idx = system.plastic_CurrentIndex()[:, 0].astype(int)
                a = np.sum(np.not_equal(idx, idx_n))
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

                niter = system.timeSteps_residualcheck(args.t_step)
                iiter += niter
                stop = niter == 0
                storage.dset_extend1d(file, "/sync-t/stored", t_istore, iiter)
                t_istore += 1
                store = True