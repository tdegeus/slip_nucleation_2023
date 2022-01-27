from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap
from collections import defaultdict

import click
import enstat
import FrictionQPotFEM.UniformSingleLayer2d as model
import h5py
import numpy as np
import tqdm

from . import storage
from . import System
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_ensembleinfo="TriggerAvalanche_EnsembleInfo",
    cli_spatialprofile="TriggerAvalanche_SpatialProfile",
    cli_enstataverage_sync_A="TriggerAvalanche_enstataverage_sync_A",
)

file_defaults = dict(
    cli_ensembleinfo="TriggerAvalanche_EnsembleInfo.h5",
    cli_spatialprofile="TriggerAvalanche_SpatialProfile.h5",
    cli_enstataverage_sync_A="TriggerAvalanche_enstataverage_sync_A.h5",
)


def replace_ep(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def trigger_and_run(
    system: model.System,
    element: int,
    deps_kick: float,
    overview_interval: int = 500,
    snapshot_interval: int = 2000,
) -> dict:
    """
    .. todo::

        Complete

    Trigger and element and run the dynamics.

    :system: Fully initialised system with the relevant increment loaded.
    :element: Index of plastic element to trigger.
    :deps_kick: Strain amplitude to use for trigger.
    """

    ret = defaultdict(lambda: defaultdict(dict))

    ret["disp"]["0"] = system.u()

    # system.t()

    # idx = system.plastic_CurrentIndex()[:, 0].astype(np.int64)
    # idx_n = np.array(idx, copy=True)
    # idx_last = np.array(idx, copy=True)

    system.triggerElementWithLocalSimpleShear(deps_kick, element)

    inc = 0

    with True:

        n = min(
            [
                overview_interval - inc % overview_interval,
                snapshot_interval - inc % snapshot_interval,
            ]
        )

        niter = system.timeStepsUntilEvent(max_iter=n)
        print(niter)


def trigger_avalanche_ensembleinfo(
    filepaths: list[str],
    ensembleinfo: str,
) -> dict:
    """
    Read the general ensemble information of triggered avalanches
    (previously known as ``EventEvolution``).

    :param filepaths:
        List of file paths.

    :param ensembleinfo:
        Path to global EnsembleInfo (for normalisation),
        see :py:func:`mycode_front.System.cli_ensembleinfo`

    :return:
        A dictionary as follows::

            xi: size of connect region @ final step
            A: avalanche area @ final step
            S: avalanche size @ final step
            t: avalanche duration; time between first and last event
            "t:xi": value of ``xi`` corresponding to each entry in ``t``
            "t:A": value of ``A`` corresponding to each entry in ``t``
            "t:S": value of ``S`` corresponding to each entry in ``t``
            t=0_Sigma: macroscopic stress @ triggering
            t=t_Sigma: macroscopic stress @ last yielding event
            t=e_Sigma: macroscopic stress @ equilibrium
            t=0_sigmar: stress inside yielding blocks @ triggering
            t=t_sigmar: stress inside yielding blocks @ last yielding event
            t=e_sigmar: stress inside yielding blocks @ equilibrium
            t=0_sigmar_connected: stress inside connected yielding region @ triggering
            t=t_sigmar_connected: stress inside connected yielding region @ last yielding event
            t=e_sigmar_connected: stress inside connected yielding region @ equilibrium
            t=t_sigmar_connected_error: expected read error in ``t=t_sigmar_connected``
            t=e_sigmar_connected_error: expected read error in ``t=e_sigmar_connected``

        Note that the duration is only measured for non-system-spanning events,
        as for other events the run is truncated.
        The field ``t:A`` and ``t:S`` are returned for this purpose.
    """

    with h5py.File(ensembleinfo, "r") as file:

        N = int(file["/normalisation/N"][...])
        dt = float(file["/normalisation/dt"][...])
        l0 = float(file["/normalisation/l0"][...])
        cs = float(file["/normalisation/cs"][...])
        sig0 = float(file["/normalisation/sig0"][...])

    ret = defaultdict(list)

    for filepath in tqdm.tqdm(filepaths):

        with h5py.File(filepath, "r") as file:

            imin = file["/snapshot/storage/snapshot"][0]
            imax = file["/snapshot/storage/snapshot"][-1]
            iiter_last = file["/event/global/iiter"][-1]
            v = file["/snapshot/storage/iiter/values"][...]
            i = file["/snapshot/storage/iiter/index"][...]
            iread = i[np.argmin(np.abs(v - iiter_last))]

            idx_n = file[f"/snapshot/plastic/{imin:d}/idx"][...]
            idx = file[f"/snapshot/plastic/{imax:d}/idx"][...]

            connected = tools.fill_avalanche(idx_n != idx)
            xi = np.sum(connected)

            Sig = file[f"/snapshot/plastic/{imin:d}/sig"][...]
            Sig_avalanche = np.mean(Sig[:, idx_n != idx], axis=1)
            Sig_connected = np.mean(Sig[:, connected], axis=1)

            ret["t=0_sigmar"].append(
                tools.sigd(
                    xx=Sig_avalanche[0] / sig0,
                    xy=Sig_avalanche[1] / sig0,
                    yy=Sig_avalanche[2] / sig0,
                )
            )

            ret["t=0_sigmar_connected"].append(
                tools.sigd(
                    xx=Sig_connected[0] / sig0,
                    xy=Sig_connected[1] / sig0,
                    yy=Sig_connected[2] / sig0,
                )
            )

            A = file["/event/global/A"][-1]
            S = file["/event/global/S"][-1]

            ret["xi"].append(xi)
            ret["A"].append(A)
            ret["S"].append(S)

            ret["t=0_Sigma"].append(
                tools.sigd(
                    xx=file["/event/global/sig"][0, 0] / sig0,
                    xy=file["/event/global/sig"][1, 0] / sig0,
                    yy=file["/event/global/sig"][2, 0] / sig0,
                )
            )

            ret["t=t_Sigma"].append(
                tools.sigd(
                    xx=file["/event/global/sig"][0, -1] / sig0,
                    xy=file["/event/global/sig"][1, -1] / sig0,
                    yy=file["/event/global/sig"][2, -1] / sig0,
                )
            )

            ret["t=e_Sigma"].append(
                tools.sigd(
                    xx=file["/overview/global/sig"][0, -1] / sig0,
                    xy=file["/overview/global/sig"][1, -1] / sig0,
                    yy=file["/overview/global/sig"][2, -1] / sig0,
                )
            )

            ret["t=t_sigmar"].append(
                tools.sigd(
                    xx=file["/event/crack/sig"][0, -1] / sig0,
                    xy=file["/event/crack/sig"][1, -1] / sig0,
                    yy=file["/event/crack/sig"][2, -1] / sig0,
                )
            )

            Sig = file[f"/snapshot/plastic/{iread:d}/sig"][...]
            Sig_avalanche = np.mean(Sig[:, idx_n != idx], axis=1)
            Sig_connected = np.mean(Sig[:, connected], axis=1)

            sigmar = tools.sigd(
                xx=Sig_avalanche[0] / sig0,
                xy=Sig_avalanche[1] / sig0,
                yy=Sig_avalanche[2] / sig0,
            )

            ret["t=t_sigmar_connected_error"].append(
                (sigmar - ret["t=t_sigmar"][-1]) / ret["t=t_sigmar"][-1]
            )

            ret["t=t_sigmar_connected"].append(
                tools.sigd(
                    xx=Sig_connected[0] / sig0,
                    xy=Sig_connected[1] / sig0,
                    yy=Sig_connected[2] / sig0,
                )
            )

            ret["t=e_sigmar"].append(
                tools.sigd(
                    xx=file["/overview/crack/sig"][0, -1] / sig0,
                    xy=file["/overview/crack/sig"][1, -1] / sig0,
                    yy=file["/overview/crack/sig"][2, -1] / sig0,
                )
            )

            Sig = file[f"/snapshot/plastic/{imax:d}/sig"][...]
            Sig_avalanche = np.mean(Sig[:, idx_n != idx], axis=1)
            Sig_connected = np.mean(Sig[:, connected], axis=1)

            sigmar = tools.sigd(
                xx=Sig_avalanche[0] / sig0,
                xy=Sig_avalanche[1] / sig0,
                yy=Sig_avalanche[2] / sig0,
            )

            ret["t=e_sigmar_connected_error"].append(
                (sigmar - ret["t=e_sigmar"][-1]) / ret["t=e_sigmar"][-1]
            )

            ret["t=e_sigmar_connected"].append(
                tools.sigd(
                    xx=Sig_connected[0] / sig0,
                    xy=Sig_connected[1] / sig0,
                    yy=Sig_connected[2] / sig0,
                )
            )

            t0 = file["/event/global/iiter"][0]
            t1 = file["/event/global/iiter"][-1]

            if t1 - t0 > 0 and A != N:
                ret["t"].append((t1 - t0) * dt * cs / (l0))
                ret["t:xi"].append(xi)
                ret["t:A"].append(A)
                ret["t:S"].append(S)

    ret = dict(ret)

    for key in ret:
        ret[key] = np.array(ret[key])

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


def cli_ensembleinfo(cli_args=None):
    """
    Read and store the general ensemble information of triggered avalanches
    (previously known as ``EventEvolution``).
    See :py:func:`trigger_avalanche_ensembleinfo` and datasets and their attributes.

    .. note::

        The output of :py:func:`mycode_front.System.cli_ensembleinfo`
        is needed for normalisation.
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
    parser.add_argument("-e", "--ensembleinfo", required=True, type=str, help="Basic EnsembleInfo")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation output")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.exists(args.ensembleinfo)
    assert len(args.files) > 0
    assert np.all([os.path.exists(f) for f in args.files])
    assert args.develop or not tag.has_uncommitted(version)

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    data = trigger_avalanche_ensembleinfo(args.files, args.ensembleinfo)

    with h5py.File(args.output, "w") as file:

        meta = file.create_group("meta")
        meta.attrs["version"] = version

        file["/meta/files"] = args.files

        file["S"] = data["S"]
        file["A"] = data["A"]
        file["xi"] = data["xi"]
        file["t"] = data["t"]
        file["t"].attrs["S"] = data["t:S"]
        file["t"].attrs["A"] = data["t:A"]
        file["t"].attrs["xi"] = data["t:xi"]

        storage.dump_with_atttrs(
            file,
            "/initial/Sigma",
            data["t=0_Sigma"],
            desc="Macroscopic stress @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/initial/sigmar",
            data["t=0_sigmar"],
            desc="Residual stress inside moving blocks @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/initial/sigmar_connected",
            data["t=0_sigmar_connected"],
            desc="Residual stress inside connected moving region @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/last-event/Sigma",
            data["t=t_Sigma"],
            desc="Macroscopic stress @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/last-event/sigmar",
            data["t=t_sigmar"],
            desc="Residual stress inside moving blocks @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/last-event/sigmar_connected",
            data["t=t_sigmar_connected"],
            desc="Residual stress inside connected moving region @ last yielding event",
        )

        file["/last-event/sigmar_connected"].attrs["error"] = data["t=t_sigmar_connected_error"]

        storage.dump_with_atttrs(
            file,
            "/equilibrium/Sigma",
            data["t=e_Sigma"],
            desc="Macroscopic stress @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/equilibrium/sigmar",
            data["t=e_sigmar"],
            desc="Residual stress inside moving blocks @ last yielding event",
        )

        storage.dump_with_atttrs(
            file,
            "/equilibrium/sigmar_connected",
            data["t=e_sigmar_connected"],
            desc="Residual stress inside connected moving region @ last yielding event",
        )

        file["/equilibrium/sigmar_connected"].attrs["error"] = data["t=e_sigmar_connected_error"]


def cli_spatialprofile(cli_args=None):
    """
    Collect the spatial profile of triggered avalanches
    (previously known as ``EventEvolution``)
    in a single matrix, see :py:func:`mycode_front.System.interface_state`.
    Note that:
    *   System-spanning events are excluded.
    *   The stress state before and after avalanche is stored (if selected).
    *   Only the (plastic) strain drop is stored (if selected).

    .. note::

        The output of :py:func:`mycode_front.System.cli_ensembleinfo`
        is needed for normalisation.
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
    parser.add_argument("--store-epsp", action="store_true", help="Store diff. of plastic strain")
    parser.add_argument("--store-strain", action="store_true", help="Store diff. of strain tensor")
    parser.add_argument("--store-stress", action="store_true", help="Store stress tensor")
    parser.add_argument("-e", "--ensembleinfo", required=True, type=str, help="Basic EnsembleInfo")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation output")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.exists(args.ensembleinfo)
    assert len(args.files) > 0
    assert np.all([os.path.exists(f) for f in args.files])
    assert args.develop or not tag.has_uncommitted(version)

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    with h5py.File(args.ensembleinfo, "r") as file:
        N = int(file["/normalisation/N"][...])
        files = file["/files"].asstr()[...]
        asext = {os.path.splitext(f)[0]: f for f in files}

    sourcedir = os.path.dirname(args.ensembleinfo)
    filepaths = defaultdict(list)
    read_disp = defaultdict(list)

    for filepath in args.files:
        info = tools.read_parameters(filepath)
        sourcepath = os.path.join(sourcedir, asext["id={id}".format(**info)])
        filepaths[sourcepath].append(0)
        read_disp[sourcepath].append(filepath)

    info = trigger_avalanche_ensembleinfo(args.files, args.ensembleinfo)
    keep = info["A"] < N

    data_0 = System.interface_state(filepaths, read_disp)

    for key in filepaths:
        filepaths[key] = [1 for i in filepaths[key]]

    data_1 = System.interface_state(filepaths, read_disp)

    with h5py.File(args.output, "w") as file:

        meta = file.create_group("meta")
        meta.attrs["version"] = version

        file["/meta/files"] = args.files

        if args.store_stress:
            file["/sig_xx/0"] = data_0["sig_xx"][keep]
            file["/sig_xy/0"] = data_0["sig_xy"][keep]
            file["/sig_yy/0"] = data_0["sig_yy"][keep]
            file["/sig_xx/1"] = data_1["sig_xx"][keep]
            file["/sig_xy/1"] = data_1["sig_xy"][keep]
            file["/sig_yy/1"] = data_1["sig_yy"][keep]

        if args.store_strain:
            file["deps_xx"] = data_1["eps_xx"][keep] - data_0["eps_xx"][keep]
            file["deps_xy"] = data_1["eps_xy"][keep] - data_0["eps_xy"][keep]
            file["deps_yy"] = data_1["eps_yy"][keep] - data_0["eps_yy"][keep]

        if args.store_epsp:
            file["depsp"] = data_1["epsp"][keep] - data_0["epsp"][keep]

        file["S"] = data_1["S"][keep] - data_0["S"][keep]


def enstataverage_sync_A(
    filepaths: list[str],
    ensembleinfo: str,
    delta_A: int = 50,
) -> dict:
    """
    Get the sums of the first and second statistical moments and the norm, at varying ``A``.

    :param filepaths:
        List of file paths.

    :param ensembleinfo:
        Path to global EnsembleInfo (for normalisation),
        see :py:func:`mycode_front.System.cli_ensembleinfo`.

    :param delta_A:
        Compute plastic strain rate based on the plastic strain and time difference between ``A``
        and ``A - delta_A``. For small ``A`` the following is used ``delta_A = min(A, delta_A)``.

    .. todo::

        Write test.

    :return:

        Dictionary with the following output.
        Each variable is stored as an ``enstat.static`` object with shape ``[N + 1, N]```
        (or ``[N + 1]`` for the time), with each row corresponding to a different ``A``.
        The following is collected::

            {
                # aligned avalanche but not masking in space
                "layer": {
                    "epsp": ..., # plastic strain accumulated in the event
                    "epspdot": ..., # plastic strain rate
                    "moved": ..., # block has moved or not
                    "S": ..., # total number of times the block yielded
                    "sig_xx": ..., # local stress: xx-component
                    "sig_xy": ..., # local stress: xy-component
                    "sig_yy": ..., # local stress: yy-component
                    "t": ..., # time
                },
                # same as "layers", but blocks outside largest connected 'crack' are masked
                "crack": {...},
                # same as "layers", but blocks that have not yielded are masked
                "moved": {...},
            }
    """

    with h5py.File(ensembleinfo, "r") as file:

        N = int(file["/normalisation/N"][...])
        t0 = float(file["/normalisation/t0"][...])
        dt = float(file["/normalisation/dt"][...])
        eps0 = float(file["/normalisation/eps0"][...])
        sig0 = float(file["/normalisation/sig0"][...])
        files = file["files"].asstr()[...]
        asext = {os.path.splitext(f)[0]: f for f in files}

    sourcedir = os.path.dirname(ensembleinfo)

    # allocate ensemble averages
    ret = {}
    for mode in ["layer", "crack", "moved"]:
        ret[mode] = {
            "epsp": enstat.static(shape=(N + 1, N), dtype=np.float),
            "epspdot": enstat.static(shape=(N + 1, N), dtype=np.float),
            "moved": enstat.static(shape=(N + 1, N), dtype=np.int64),
            "S": enstat.static(shape=(N + 1, N), dtype=np.int64),
            "sig_xx": enstat.static(shape=(N + 1, N), dtype=np.float),
            "sig_xy": enstat.static(shape=(N + 1, N), dtype=np.float),
            "sig_yy": enstat.static(shape=(N + 1, N), dtype=np.float),
        }

    ret["global"] = {
        "t": enstat.static(shape=(N + 1), dtype=np.float),
        "sig_xx": enstat.static(shape=(N + 1), dtype=np.float),
        "sig_xy": enstat.static(shape=(N + 1), dtype=np.float),
        "sig_yy": enstat.static(shape=(N + 1), dtype=np.float),
    }

    # pre-allocate output per realisation (not averaged)
    crack = np.empty((N + 1, N), dtype=bool)
    moved = np.zeros((N + 1, N), dtype=bool)
    epsp = np.zeros((N + 1, N), dtype=float)
    epspdot = np.zeros((N + 1, N), dtype=float)
    S = np.zeros((N + 1, N), dtype=np.int64)
    sig_xx = np.zeros((N + 1, N), dtype=float)
    sig_xy = np.zeros((N + 1, N), dtype=float)
    sig_yy = np.zeros((N + 1, N), dtype=float)

    mask = np.empty((N + 1), dtype=bool)
    t = np.zeros((N + 1), dtype=float)
    sigbar_xx = np.zeros((N + 1), dtype=float)
    sigbar_xy = np.zeros((N + 1), dtype=float)
    sigbar_yy = np.zeros((N + 1), dtype=float)

    # array indices (used below)
    edx = np.empty((2, N), dtype=int)
    edx[0, :] = np.arange(N)

    for ifile, filepath in enumerate(tqdm.tqdm(filepaths)):

        info = tools.read_parameters(filepath)
        sourcepath = os.path.join(sourcedir, asext["id={id}".format(**info)])

        with h5py.File(sourcepath, "r") as file:
            epsy = System.read_epsy(file)
            epsy = np.hstack((-epsy[:, 0].reshape(-1, 1), epsy))

        with h5py.File(filepath, "r") as file:

            if ifile == 0:
                plastic = file["/meta/plastic"][...]

            A = file["/sync-A/stored"][...].astype(np.int64)
            t[A] = file["/sync-A/global/iiter"][A] * dt / t0
            sigbar_xx[A] = file["/sync-A/global/sig_xx"][A] / sig0
            sigbar_xy[A] = file["/sync-A/global/sig_xy"][A] / sig0
            sigbar_yy[A] = file["/sync-A/global/sig_yy"][A] / sig0

            mask.fill(True)
            mask[A] = False
            crack[mask] = False
            moved[mask] = False

            for ia, a in enumerate(A):

                # path aliases to shorten code

                root = f"/sync-A/plastic/{a:d}"
                sroot = None

                if f"/sync-A/element/{a:d}/sig_xx" in file:
                    sroot = f"/sync-A/element/{a:d}"

                # current index & avalanche properties

                idx = file[f"{root}/idx"][...].astype(np.int64)

                if ia == 0:
                    idx0 = np.array(idx, copy=True)

                m = idx0 != idx
                crack[a, :] = tools.fill_avalanche(m)
                moved[a, :] = m
                S[a, :] = idx - idx0

                # plastic strain

                edx[1, :] = idx
                i = np.ravel_multi_index(edx, epsy.shape)
                epsy_l = epsy.flat[i]
                epsy_r = epsy.flat[i + 1]
                epsp[a, :] = 0.5 * (epsy_l + epsy_r) / eps0

                if f"{root}/epsp" in file and a > 0:
                    assert np.allclose(epsp[a, :], file[f"{root}/epsp"][...] / eps0)

                # stress

                if sroot:
                    sig_xx[a, :] = file[f"{sroot}/sig_xx"][...][plastic] / sig0
                    sig_xy[a, :] = file[f"{sroot}/sig_xy"][...][plastic] / sig0
                    sig_yy[a, :] = file[f"{sroot}/sig_yy"][...][plastic] / sig0
                else:
                    sig_xx[a, :] = file[f"{root}/sig_xx"][...] / sig0
                    sig_xy[a, :] = file[f"{root}/sig_xy"][...] / sig0
                    sig_yy[a, :] = file[f"{root}/sig_yy"][...] / sig0

        # plastic strain rate

        i_delta = np.argmin(tools.distance1d(A - delta_A, A), axis=1)

        for a, i_n in zip(A, i_delta):
            a_n = A[i_n]
            epspdot[a, :] = (epsp[a, :] - epsp[a_n, :]) / (t[a] - t[a_n])

        # total plastic strain -> accumulated plastic strain

        epsp = epsp - epsp[np.min(A), :]

        # save

        roll = tools.center_avalanche_per_row(moved)

        crack = tools.indep_roll(crack, roll)
        moved = tools.indep_roll(moved, roll)
        epsp = tools.indep_roll(epsp, roll)
        epspdot = tools.indep_roll(epspdot, roll)
        S = tools.indep_roll(S, roll)
        sig_xx = tools.indep_roll(sig_xx, roll)
        sig_xy = tools.indep_roll(sig_xy, roll)
        sig_yy = tools.indep_roll(sig_yy, roll)

        for key in ret["layer"]:
            if key not in ["epspdot"]:
                ret["layer"][key].add_sample(locals()[key], mask=mask)
                ret["crack"][key].add_sample(locals()[key], mask=np.logical_not(crack))
                ret["moved"][key].add_sample(locals()[key], mask=np.logical_not(moved))

        ret["global"]["t"].add_sample(t, mask=mask)
        ret["global"]["sig_xx"].add_sample(sigbar_xx, mask=mask)
        ret["global"]["sig_xy"].add_sample(sigbar_xy, mask=mask)
        ret["global"]["sig_yy"].add_sample(sigbar_yy, mask=mask)

        mask[A[A <= A[i_delta]]] = True
        crack[mask] = False
        moved[mask] = False

        for key in ["epspdot"]:
            ret["layer"][key].add_sample(locals()[key], mask=mask)
            ret["crack"][key].add_sample(locals()[key], mask=np.logical_not(crack))
            ret["moved"][key].add_sample(locals()[key], mask=np.logical_not(moved))

    return ret


def cli_enstataverage_sync_A(cli_args=None):
    """
    Read the ensemble average synchronized at "A".

    .. note::

        The output of :py:func:`mycode_front.System.cli_ensembleinfo`
        is needed for normalisation.
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
    parser.add_argument("-e", "--ensembleinfo", required=True, type=str, help="Basic EnsembleInfo")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation output")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.exists(args.ensembleinfo)
    assert len(args.files) > 0
    assert np.all([os.path.exists(f) for f in args.files])
    assert args.develop or not tag.has_uncommitted(version)

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    data = enstataverage_sync_A(args.files, args.ensembleinfo)

    with h5py.File(args.output, "w") as file:

        meta = file.create_group("meta")
        meta.attrs["version"] = version

        file["/meta/files"] = args.files

        for mode in data:
            for key in data[mode]:
                file[f"/{mode}/{key}/first"] = data[mode][key].first()
                file[f"/{mode}/{key}/second"] = data[mode][key].second()
                file[f"/{mode}/{key}/norm"] = data[mode][key].norm()
