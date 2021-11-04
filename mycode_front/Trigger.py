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
    cli_trigger_avalanche_ensembleinfo="TriggerAvalanche_EnsembleInfo",
    cli_enstataverage_sync_A="TriggerAvalanche_enstataverage_sync_A",
)

file_defaults = dict(
    cli_trigger_avalanche_ensembleinfo="TriggerAvalanche_EnsembleInfo.h5",
    cli_enstataverage_sync_A="TriggerAvalanche_enstataverage_sync_A.h5",
)


def replace_ep(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def trigger_avalanche(
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
    Read the general ensemble information of triggered avalanches.

    :param filepaths:
        List of file paths.

    :param ensembleinfo:
        Path to global EnsembleInfo (for normalisation), see :py:func:`System.cli_ensembleinfo`

    :return:

        Dictionary with the following output::

            {
                "A": [...], # avalanche area @ final step
                "S": [...], # avalanche size @ final step
                "t": [...], # avalanche duration: time between first and last event
                "t:A": [...], # value of "A" corresponding to each entry in "t"
                "t:S": [...], # value of "S" corresponding to each entry in "t"
                "sigmar": [...], # stress inside yielding blocks @ final step
                "Sigma": [...], # macroscopic stress @ final step
                "t=0_Sigma": [...], # macroscopic stress @ triggering
                "t=0_sigmar": [...], # stress 'inside' avalanche @ triggering
            }

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

            idx_n = file[f"/snapshot/plastic/{imin:d}/idx"][...]
            idx = file[f"/snapshot/plastic/{imax:d}/idx"][...]

            Sig = file[f"/snapshot/plastic/{imin:d}/sig"][...]
            Sig = np.mean(Sig[:, idx_n != idx], axis=1)

            ret["t=0_sigmar"].append(
                tools.sigd(
                    xx=Sig[0] / sig0,
                    xy=Sig[1] / sig0,
                    yy=Sig[2] / sig0,
                )
            )

            A = file["/event/global/A"][-1]
            S = file["/event/global/S"][-1]

            ret["A"].append(A)
            ret["S"].append(S)

            ret["t=0_Sigma"].append(
                tools.sigd(
                    xx=file["/event/global/sig"][0, 0] / sig0,
                    xy=file["/event/global/sig"][1, 0] / sig0,
                    yy=file["/event/global/sig"][2, 0] / sig0,
                )
            )

            ret["Sigma"].append(
                tools.sigd(
                    xx=file["/overview/global/sig"][0, -1] / sig0,
                    xy=file["/overview/global/sig"][1, -1] / sig0,
                    yy=file["/overview/global/sig"][2, -1] / sig0,
                )
            )

            ret["sigmar"].append(
                tools.sigd(
                    xx=file["/event/crack/sig"][0, -1] / sig0,
                    xy=file["/event/crack/sig"][1, -1] / sig0,
                    yy=file["/event/crack/sig"][2, -1] / sig0,
                )
            )

            t0 = file["/event/global/iiter"][0]
            t1 = file["/event/global/iiter"][-1]

            if t1 - t0 > 0 and A != N:
                ret["t"].append((t1 - t0) * dt * cs / (l0))
                ret["t:A"].append(A)
                ret["t:S"].append(S)

    ret = dict(ret)

    for key in ret:
        ret[key] = np.array(ret[key])

    return ret


def cli_trigger_avalanche_ensembleinfo(cli_args=None):
    """
    Read and store the general ensemble information of triggered avalanches.
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
    parser.add_argument("-e", "--ensembleinfo", type=str, help="EnsembleInfo for normalisation")
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
        file["t"] = data["t"]
        file["t"].attrs["S"] = data["t:S"]
        file["t"].attrs["A"] = data["t:A"]

        storage.dump_with_atttrs(
            file, "Sigma", data["Sigma"], desc="Macroscopic stress @ final step"
        )

        storage.dump_with_atttrs(
            file,
            "/sigmar",
            data["sigmar"],
            desc="Residual stress inside the avalanche @ final step",
        )

        storage.dump_with_atttrs(
            file, "/initial/Sigma", data["t=0_Sigma"], desc="Macroscopic stress @ triggering"
        )

        storage.dump_with_atttrs(
            file,
            "/initial/sigmar",
            data["t=0_sigmar"],
            desc="Stress 'inside' the avalanche @ triggering",
        )


def enstataverage_sync_A(
    filepaths: list[str],
    ensembleinfo: str,
    delta_A: int = 50,
) -> dict:
    """
    Get the sums of the first and second statistical moments and the norm.

    :param filepaths:
        List of file paths.

    :param ensembleinfo:
        Path to global EnsembleInfo (for normalisation), see :py:func:`System.cli_ensembleinfo`

    :param delta_A:
        Compute plastic strain rate based on the plastic strain and time difference between ``A``
        and ``A - delta_A``. For small ``A`` the following is used ``delta_A = min(A, delta_A)``.

    .. todo::

        Write test.

    :return:

        Dictionary with the following output::

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

        Each variable is stored as an ``enstat.static`` object with shape ``[N + 1, N]```
        (or ``[N + 1]`` for the time), with each row corresponding to a different ``A``.
    """

    with h5py.File(ensembleinfo, "r") as file:

        N = int(file["/normalisation/N"][...])
        t0 = float(file["/normalisation/t0"][...])
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
            t[A] = file["/sync-A/global/iiter"][A] / t0
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
    parser.add_argument("-e", "--ensembleinfo", type=str, help="EnsembleInfo for normalisation")
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
