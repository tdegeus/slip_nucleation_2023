from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import textwrap
import warnings
from collections import defaultdict

import click
import enstat
import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numpy.typing import ArrayLike

from . import slurm
from . import storage
from . import System
from . import tag
from ._version import version

plt.style.use(["goose", "goose-latex"])


entry_points = dict(
    cli_branch_velocityjump="Flow_branch_velocityjump",
    cli_ensembleinfo="Flow_ensembleinfo",
    cli_ensembleinfo_velocityjump="Flow_ensembleinfo_velocityjump",
    cli_generate="Flow_generate",
    cli_plot="Flow_plot",
    cli_plot_velocityjump="Flow_plot_velocityjump",
    cli_run="Flow_run",
    cli_update_branch_velocityjump="Flow_update_branch_velocityjump",
    cli_update_generate="Flow_update_generate",
    cli_update_run="Flow_update_run",
)

file_defaults = dict(
    cli_ensembleinfo="Flow_EnsembleInfo.h5",
    cli_ensembleinfo_velocityjump="Flow_EnsembleInfo_velocityjump.h5",
)


def replace_ep(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def _interpret(part: list[str], convert: bool = False) -> dict:
    """
    Convert to useful information by splitting at "=".

    :param: List with strings ``key=value``.
    :param convert: Convert to numerical values depending on the variable.
    :return: Parameters as dictionary.
    """

    info = {}

    for i in part:
        if len(i.split("=")) > 1:
            key, value = i.split("=")
            info[key] = value

    if convert:
        for key in info:
            if key in ["gammadot", "jump"]:
                info[key] = float(info[key])
            else:
                info[key] = int(info[key])

    return info


def interpret_key(key: str, convert: bool = False) -> dict:
    """
    Split a key in useful information.

    :param key: ``key=value`` separated by ``/`` or ``_``.
    :param convert: Convert to numerical values.
    :return: Parameters as dictionary.
    """

    return _interpret(re.split("_|/", key), convert=convert)


def interpret_filename(filepath: str, convert: bool = False) -> dict:
    """
    Split filepath in useful information.

    :param filepath: Filepath of which only the basename is considered.
    :param convert: Convert to numerical values.
    :return: Parameters as dictionary.
    """

    part = os.path.splitext(os.path.basename(filepath))[0]
    return _interpret(re.split("_|/", part), convert=convert)


def generate(*args, **kwargs):
    """
    Generate input file.
    See :py:func:`System.generate`. On top of that:

    :param gammadot: The shear-rate to prescribe.
    :param output: Output storage interval.
    :param restart: Restart storage interval.
    :param snapshot: Snapshot storage interval.
    """

    kwargs.setdefault("init_run", False)
    gammadot = kwargs.pop("gammadot")
    output = kwargs.pop("output", int(500))
    restart = kwargs.pop("restart", int(5000))
    snapshot = kwargs.pop("snapshot", int(500 * 500))

    progname = entry_points["cli_generate"]
    System.generate(*args, **kwargs)

    test_mode = kwargs.pop("test_mode", False)
    assert test_mode or not tag.has_uncommitted(version)

    with h5py.File(kwargs["filepath"], "a") as file:

        meta = file.create_group(f"/meta/{progname}")
        meta.attrs["version"] = version

        storage.dump_with_atttrs(
            file,
            "/gammadot",
            gammadot,
            desc="Applied shear-rate",
        )

        storage.dump_with_atttrs(
            file,
            "/boundcheck",
            10,
            desc="Stop at n potentials before running out of bounds",
        )

        storage.dump_with_atttrs(
            file,
            "/restart/interval",
            restart,
            desc="Restart storage interval",
        )

        storage.dump_with_atttrs(
            file,
            "/output/interval",
            output,
            desc="Output storage interval",
        )

        storage.dump_with_atttrs(
            file,
            "/snapshot/interval",
            snapshot,
            desc="Snapshot storage interval",
        )


def default_gammadot():
    """
    Shear rates to simulate.

    :return: List of shear rates, and a typical shear rate.
    """

    eps0_new = 1.0e-3 / 2.0 / 10.0
    # eps0_old = 5e-4 / 8.0
    # Gammadot = np.linspace(1e-9, 20e-9, 20) / 8.0
    # Gammadot = Gammadot
    # Gammadot *= eps0_new / eps0_old
    Gammadot = [1e-13, 5e-13, 1e-12, 5e-12, 1e-11, 5e-11] + np.linspace(1e-10, 2e-9, 20).tolist()

    return np.array(Gammadot), eps0_new


def cli_generate(cli_args=None):
    """
    Generate IO files, including job-scripts to run simulations.
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

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=2 * (3**6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("outdir", type=str, help="Output directory")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isdir(args.outdir)

    Gammadot, _ = default_gammadot()
    filenames = []
    filepaths = []

    for i in tqdm.tqdm(range(args.start, args.start + args.nsim)):

        for gammadot in Gammadot:

            filename = f"id={i:03d}_gammadot={gammadot:.2e}.h5"
            filepath = os.path.join(args.outdir, filename)
            filenames.append(filename)
            filepaths.append(filepath)

            generate(
                filepath=filepath,
                gammadot=gammadot,
                N=args.size,
                seed=i * args.size,
                test_mode=args.develop,
                dev=args.develop,
            )

            # warning: Gammadot hard-coded here, check that yield strains did not change
            with h5py.File(filepath, "r") as file:
                assert not isinstance(file["/cusp/epsy"], h5py.Dataset)
                assert np.isclose(file["/cusp/epsy/eps0"][...], default_gammadot()[1])

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in filenames]
    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return filepaths


def cli_update_generate(cli_args=None):
    """
    Apply updates and bugfixes to old files:

    *   <= 5.3 : remove :py:func:`System.run` specific files.
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
    progname = entry_points["cli_generate"]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("files", nargs="*", type=str, help="Simulations to update")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    outpaths = [os.path.join(args.outdir, f) for f in args.files]
    assert np.all([os.path.exists(f) for f in args.files])
    assert not np.any([os.path.exists(f) for f in outpaths])
    assert args.develop or not tag.has_uncommitted(version)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    for filepath, outpath in zip(tqdm.tqdm(args.files), outpaths):

        with h5py.File(filepath, "r") as source, h5py.File(outpath, "w") as dest:

            assert f"/meta/{progname}" in source
            meta = source[f"/meta/{progname}"]
            ver = meta.attrs["version"]
            updated = [ver]
            if "updated" in meta.attrs:
                updated += list(meta.attrs["updated"])

            if tag.less_equal(ver, "5.3"):

                paths = g5.getdatapaths(source)
                paths.remove("/run/epsd/kick")
                paths.remove("/stored")
                paths.remove("/t")
                paths.remove("/kick")
                paths.remove("/disp/0")
                paths.remove("/disp")
                paths.remove(f"/meta/{progname}")

                g5.copy(source, dest, paths)

                dest_meta = dest.create_group(f"/meta/{progname}")
                dest_meta.attrs["version"] = version
                dest_meta.attrs["updated"] = updated

                for key in meta.attrs:
                    if key not in ["version", "updated"]:
                        dest_meta.attrs[key] = meta.attrs[key]

    if cli_args is not None:
        return outpaths


def run_create_extendible(file: h5py.File):
    """
    Create extendible datasets used in :py:func:`run`.
    """

    flatindex = dict(
        xx=0,
        xy=1,
        yy=2,
    )

    storage.create_extendible(file, "/output/epsp", np.float64)
    storage.create_extendible(file, "/output/global/sig", np.float64, ndim=2, **flatindex)
    storage.create_extendible(file, "/output/inc", np.uint32)
    storage.create_extendible(file, "/output/S", np.int64)
    storage.create_extendible(file, "/output/weak/sig", np.float64, ndim=2, **flatindex)
    storage.create_extendible(file, "/restart/inc", np.uint32)
    storage.create_extendible(file, "/snapshot/inc", np.uint32)


def run(filepath: str, dev: bool = False, progress: bool = True):
    """
    Run flow simulation.

    :param filepath: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    :param progress: Show progress bar.
    """

    basename = os.path.basename(filepath)
    progname = entry_points["cli_run"]

    with h5py.File(filepath, "a") as file:

        system = System.init(file)
        meta = System.create_check_meta(file, f"/meta/{progname}", dev=dev)

        if "completed" in meta:
            print(f'"{basename}": marked completed, skipping')
            return 1

        run_create_extendible(file)

        inc = 0
        output = file["/output/interval"][...]
        restart = file["/restart/interval"][...]
        snapshot = file["/snapshot/interval"][...] if "/snapshot/interval" in file else sys.maxsize
        v = system.affineSimpleShearCentered(file["/gammadot"][...])

        plastic = system.plastic()
        dV = system.quad().AsTensor(2, system.quad().dV())
        dV_plas = dV[plastic, ...]

        if "/restart/u" in file:
            system.setU(file["/restart/u"][...])
            system.setV(file["/restart/v"][...])
            system.setA(file["/restart/a"][...])
            inc = file["/restart/inc"][...]
            print(f"{basename}: restarting, inc = {inc:d}")
        else:
            print(f"{basename}: starting")

        boundcheck = file["/boundcheck"][...]
        nchunk = file["/cusp/epsy/nchunk"][...] - boundcheck
        pbar = tqdm.tqdm(total=nchunk, disable=not progress)

        while True:

            n = min(
                [
                    output - inc % output,
                    restart - inc % restart,
                    snapshot - inc % snapshot,
                ]
            )

            if not system.flowSteps_boundcheck(n, v, boundcheck):
                break

            if progress:
                pbar.n = np.max(system.plastic_CurrentIndex())
                pbar.refresh()

            inc += n

            if inc % snapshot == 0:

                i = int(inc / snapshot)

                for key in ["/snapshot/inc"]:
                    file[key].resize((i + 1,))

                file["/snapshot/inc"][i] = inc
                file[f"/snapshot/u/{inc:d}"] = system.u()
                file[f"/snapshot/v/{inc:d}"] = system.v()
                file[f"/snapshot/a/{inc:d}"] = system.a()

            if inc % output == 0:

                i = int(inc / output)

                for key in ["/output/inc", "/output/S", "/output/epsp"]:
                    file[key].resize((i + 1,))

                for key in ["/output/global/sig", "/output/weak/sig"]:
                    file[key].resize((3, i + 1))

                Sig_bar = np.average(system.Sig(), weights=dV, axis=(0, 1))
                Sig_weak = np.average(system.plastic_Sig(), weights=dV_plas, axis=(0, 1))

                file["/output/inc"][i] = inc
                file["/output/S"][i] = np.sum(system.plastic_CurrentIndex()[:, 0])
                file["/output/epsp"][i] = np.mean(system.plastic_Epsp()[:, 0])
                file["/output/global/sig"][:, i] = Sig_bar.ravel()[[0, 1, 3]]
                file["/output/weak/sig"][:, i] = Sig_weak.ravel()[[0, 1, 3]]

            if inc % restart == 0:

                storage.dump_overwrite(file, "/restart/u", system.u())
                storage.dump_overwrite(file, "/restart/v", system.v())
                storage.dump_overwrite(file, "/restart/a", system.a())
                storage.dump_overwrite(file, "/restart/inc", inc)


def cli_run(cli_args=None):
    """
    Run flow simulation.
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

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)
    run(args.file, dev=args.develop, progress=not args.quiet)


def cli_update_run(cli_args=None):
    """
    Apply updates and bugfixes to old files:

    *   <= 5.3 : Fix "/output/epsp" which was not normalised by the number of blocs.
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
    progname = entry_points["cli_run"]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("files", nargs="*", type=str, help="Simulations to update")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    outpaths = [os.path.join(args.outdir, f) for f in args.files]
    assert np.all([os.path.exists(f) for f in args.files])
    assert not np.any([os.path.exists(f) for f in outpaths])
    assert args.develop or not tag.has_uncommitted(version)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    for filepath, outpath in zip(tqdm.tqdm(args.files), outpaths):

        with h5py.File(filepath, "r") as source, h5py.File(outpath, "w") as dest:

            assert f"/meta/{progname}" in source
            meta = source[f"/meta/{progname}"]
            ver = meta.attrs["version"]
            updated = [ver]
            if "updated" in meta.attrs:
                updated += list(meta.attrs["updated"])

            if tag.less_equal(ver, "5.3"):

                paths = g5.getdatapaths(source)
                paths.remove(f"/meta/{progname}")

                g5.copy(source, dest, paths)

                N = dest["/meta/normalisation/N"][...]
                dest["/output/epsp"][...] /= float(N)

                dest_meta = dest.create_group(f"/meta/{progname}")
                dest_meta.attrs["version"] = version
                dest_meta.attrs["updated"] = updated

                for key in meta.attrs:
                    if key not in ["version", "updated"]:
                        dest_meta.attrs[key] = meta.attrs[key]

    if cli_args is not None:
        return outpaths


def basic_output(file: h5py.File) -> dict:
    """
    Extract basic output.

    :param file: HDF5-archive.
    :return: Basic output.
    """

    ret = defaultdict(lambda: defaultdict(dict))

    gammadot = file["/gammadot"][...]
    dt = file["/run/dt"][...]
    sig0 = file["/meta/normalisation/sig"][...] if "/meta/normalisation/sig" in file else 1.0
    eps0 = file["/meta/normalisation/eps"][...] if "/meta/normalisation/eps" in file else 1.0
    iout = file["/output/inc"][...]
    isnap = file["/snapshot/inc"][...]
    ret["meta"]["inc2eps"] = gammadot * dt / eps0

    ret["gammadot"] = gammadot
    ret["dt"] = dt

    if "/meta/normalisation" in file:
        ret["normalisation"]["eps0"] = eps0
        ret["normalisation"]["sig0"] = sig0
        mu = file["/meta/normalisation/G"][...] / 2.0
        rho = file["/meta/normalisation/rho"][...]
        cs = np.sqrt(mu / rho)
        l0 = file["/meta/normalisation/l"][...]
        t0 = float(l0 / cs)
        ret["normalisation"]["cs"] = cs
        ret["normalisation"]["l0"] = l0
        ret["normalisation"]["t0"] = t0
        ret["normalisation"]["N"] = file["/meta/normalisation/N"][...]
    else:
        t0 = 1.0

    n = file["/output/weak/sig"].shape[1]
    xx = file["/output/weak/sig"].attrs["xx"]
    xy = file["/output/weak/sig"].attrs["xy"]
    yy = file["/output/weak/sig"].attrs["yy"]

    Sig = np.empty((n, 2, 2), dtype=float)
    Sig[:, 0, 0] = file["/output/weak/sig"][xx, :]
    Sig[:, 0, 1] = file["/output/weak/sig"][xy, :]
    Sig[:, 1, 0] = Sig[:, 0, 1]
    Sig[:, 1, 1] = file["/output/weak/sig"][yy, :]
    sig = GMat.Sigd(Sig) / sig0
    ret["weak"]["sig"] = sig
    ret["weak"]["epsp"] = file["/output/epsp"][...] / eps0
    ret["weak"]["inc"] = iout
    ret["weak"]["t"] = iout * dt / t0
    ret["weak"]["eps"] = gammadot * dt * iout / eps0
    ret["snapshot"]["inc"] = isnap
    ret["snapshot"]["eps"] = gammadot * dt * isnap / eps0

    dout = np.diff(iout)
    dsnap = np.diff(isnap)

    assert np.all(dout == dout[0])
    assert np.all(dsnap == dsnap[0])

    dout = dout[0]
    dsnap = dsnap[0]

    assert dsnap % dout == 0

    d = int(dsnap / dout)
    m = np.zeros(int(iout.size / d))
    s = np.zeros(int(iout.size / d))

    for i in range(0, iout.size, d):
        u = i + d
        m[int(i / d - 1)] = np.mean(sig[i:u])
        s[int(i / d - 1)] = np.std(sig[i:u])

    mtarget = np.mean(m[-3:])
    starget = np.mean(s[-3:])
    ss = np.argmax(np.isclose(m, mtarget, 1e-2, 1e-3 * starget)) + 2
    assert ss < ret["snapshot"]["inc"].size

    ret["snapshot"]["steadystate"] = ss
    ret["weak"]["steadystate"] = int(ss * d)

    return ret


def steadystate_output(files: list[str], normalisation: bool = True) -> dict:
    """
    Extract relevant averages from the steady-state of the output read by :py:func:`basic_output`.

    .. todo::

        For now this simply takes an average on the last 400 steps, this could be made more cleaver.

    :param files: List of files.
    :param normalisation: Read/check normalisation.

    :return:

        .. code-block:: none

            sigma_weak:
                gammadot=XXX": [...],
                gammadot=XXX": [...]

            deps: ...

            dt: ...

            epspot: ...

            normalisation:
                t0: ...
    """

    ret = defaultdict(lambda: defaultdict(list))

    for filepath in tqdm.tqdm(files):

        info = interpret_filename(os.path.basename(filepath))
        name = info["gammadot"]

        with h5py.File(filepath, "r") as file:
            data = basic_output(file)

        if normalisation:
            if "normalisation" not in ret:
                for key in data["normalisation"]:
                    ret["normalisation"][key] = data["normalisation"][key]
            for key in data["normalisation"]:
                assert np.isclose(ret["normalisation"][key], data["normalisation"][key])

        ret["sigma_weak"][name].append(np.mean(data["weak"]["sig"][-400:]))
        ret["depsp"][name].append(data["weak"]["epsp"][-1] - data["weak"]["epsp"][-400])
        ret["dt"][name].append(data["weak"]["t"][-1] - data["weak"]["t"][-400])

    for key in ret["depsp"]:
        ret["sigma_weak"][key] = np.array(ret["sigma_weak"][key])
        ret["depsp"][key] = np.array(ret["depsp"][key])
        ret["dt"][key] = np.array(ret["dt"][key])
        ret["epspdot"][key] = ret["depsp"][key] / ret["dt"][key]

    return ret


def cli_ensembleinfo(cli_args=None):
    """
    Collect basic ensemble information.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation output")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert len(args.files) > 0
    assert np.all([os.path.exists(f) for f in args.files])

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    data = steadystate_output(args.files)
    avr = defaultdict(list)
    std = defaultdict(list)
    gammadot = []

    for key in sorted(data["depsp"], key=lambda key: float(key)):
        gammadot.append(key)
        for field in [k for k in data.keys() if k not in ["normalisation"]]:
            avr[field].append(np.mean(data[field][key]))
            std[field].append(np.std(data[field][key]))

    with h5py.File(args.output, "w") as file:
        file["/gammadot"] = [float(i) for i in gammadot]
        file["/gammadot"].attrs["keys"] = gammadot
        for field in avr:
            file[g5.join(field, "mean", root=True)] = avr[field]
            file[g5.join(field, "std", root=True)] = std[field]
        if "normalisation" in data:
            for key in data["normalisation"]:
                file[f"/normalisation/{key}"] = data["normalisation"][key]


def cli_ensembleinfo_velocityjump(cli_args=None):
    """
    Collect ensemble info for velocity jumpy experiments.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulations to read")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert len(args.files) > 0
    assert np.all([os.path.exists(f) for f in args.files])

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    start = []
    stop = []

    for file in args.files:
        info = interpret_filename(file, convert=False)
        start.append(info["gammadot"])
        stop.append(info["jump"])

    start = np.array(start)
    stop = np.array(stop)
    files = np.array(args.files)
    Start = np.unique(start)
    Stop = np.unique(stop)

    with h5py.File(args.output, "w") as dest:

        for jump in tqdm.tqdm(Stop):
            for gammadot in Start:

                select = np.logical_and(start == gammadot, stop == jump)
                sigma = enstat.dynamic1d()
                tau = enstat.dynamic1d()
                for filepath in files[select]:
                    with h5py.File(filepath, "r") as file:
                        data = basic_output(file)
                        sigma.add_sample(data["weak"]["sig"])
                        tau.add_sample(data["weak"]["t"])

                n = sigma.norm()

                if n is None:
                    continue

                base = f"/jump={jump}/gammadot={gammadot}"

                dest[f"{base}/sigma_weak/mean"] = sigma.mean()[n == n[0]]
                dest[f"{base}/sigma_weak/std"] = sigma.std()[n == n[0]]

                dest[f"{base}/t/mean"] = tau.mean()[n == n[0]]
                dest[f"{base}/t/std"] = tau.std()[n == n[0]]


def cli_branch_velocityjump(cli_args=None):
    """
    Branch simulation to a velocity jump experiment:
    Copies a snapshot as restart.
    To run simply use :py:func:`cli_run`.
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

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("files", nargs="*", type=str, help="Simulations to branch")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.exists(f) for f in args.files])
    assert args.develop or not tag.has_uncommitted(version)

    Gammadot, scale = default_gammadot()
    inp_paths = []
    out_names = []
    out_paths = []
    out_gammadots = []

    for filepath in args.files:

        info = interpret_filename(os.path.basename(filepath))
        include = np.logical_not(np.isclose(Gammadot / float(info["gammadot"]), 1.0))

        for gammadot in Gammadot[include]:

            name = "id={id}_gammadot={gammadot}_jump={jump:.2e}.h5".format(jump=gammadot, **info)
            inp_paths.append(filepath)
            out_names.append(name)
            out_paths.append(os.path.join(args.outdir, name))
            out_gammadots.append(gammadot)

    assert not np.any([os.path.exists(f) for f in out_paths])

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    for out_gammadot, out_path, inp_path in zip(tqdm.tqdm(out_gammadots), out_paths, inp_paths):

        with h5py.File(inp_path, "r") as source, h5py.File(out_path, "w") as dest:

            output = basic_output(source)
            if args.develop:
                inc = output["snapshot"]["inc"]
                inc = inc[int(inc.size / 2)]
            else:
                inc = output["snapshot"]["inc"][output["snapshot"]["steadystate"]]

            meta = "/meta/{:s}".format(entry_points["cli_run"])
            paths = g5.getdatapaths(source)
            paths = [p for p in paths if not re.match(r"(/snapshot/)(.*)", p)]
            paths = [p for p in paths if not re.match(r"(/output/)(.*)", p)]
            paths = [p for p in paths if not re.match(r"(/restart/)(.*)", p)]
            paths = [p for p in paths if not re.match(f"({meta})(.*)", p)]
            g5.copy(source, dest, paths)

            for t in ["restart", "output", "snapshot"]:
                g5.copy(source, dest, [f"/{t}/interval"])

            g5.copy(
                source,
                dest,
                ["/meta/{:s}".format(entry_points["cli_run"])],
                ["/meta/{:s}_source".format(entry_points["cli_run"])],
            )
            if tag.greater(g5.version, "0.14.0"):
                warnings.warn("Remove np.array conversion in previous command", FutureWarning)

            meta = dest.create_group(f"/meta/{progname}")
            meta.attrs["version"] = version
            meta.attrs["inc"] = inc

            dest["/gammadot"][...] = out_gammadot
            dest["/restart/inc"] = 0
            dest["/restart/u"] = source[f"/snapshot/u/{inc:d}"][...]
            dest["/restart/v"] = source[f"/snapshot/v/{inc:d}"][...]
            dest["/restart/a"] = source[f"/snapshot/a/{inc:d}"][...]
            dest["/snapshot/u/0"] = source[f"/snapshot/u/{inc:d}"][...]
            dest["/snapshot/v/0"] = source[f"/snapshot/v/{inc:d}"][...]
            dest["/snapshot/a/0"] = source[f"/snapshot/a/{inc:d}"][...]

            run_create_extendible(dest)

            i = int(inc / source["/output/interval"][...])

            for key in ["/output/inc"]:
                dest[key].resize((1,))
                dest[key][0] = 0

            for key in ["/output/S", "/output/epsp"]:
                dest[key].resize((1,))
                dest[key][0] = source[key][i]

            for key in ["/output/global/sig", "/output/weak/sig"]:
                dest[key].resize((3, 1))
                dest[key][:, 0] = source[key][:, i]

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in out_names]
    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return out_paths


def cli_update_branch_velocityjump(cli_args=None):
    """
    Not anymore relevant.
    """

    raise OSError("Not any more relevant, by cli_update_generate is")


def moving_average(a: ArrayLike, n: int) -> ArrayLike:
    """
    Return the moving average.

    :param a: Array.
    :param n: Moving average over n entries.
    :return: Averaged array.
    """

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    s = n - 1
    return ret[s:] / n


def moving_average_y(x: ArrayLike, y: ArrayLike, n: int) -> ArrayLike:
    """
    Return the moving average of y while modifying the size of x.

    :param x: Array.
    :param y: Array to average.
    :param n: Moving average over n entries.
    :return: x, y
    """

    if n is None:
        return x, y

    assert n > 0

    s = n - 1
    return x[s:], moving_average(y, n)


def cli_plot(cli_args=None):
    """
    Plot overview of flow simulation.
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

    parser.add_argument("-a", "--moving-average", type=int, help="Apply moving average")
    parser.add_argument("-s", "--save", type=str, help="Save the image")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    with h5py.File(args.file, "r") as file:
        data = basic_output(file)

    fig, ax = plt.subplots()

    s = data["weak"]["steadystate"]
    ax.plot(
        *moving_average_y(data["weak"]["eps"][:s], data["weak"]["sig"][:s], args.moving_average),
        c=0.5 * np.ones(3),
    )
    ax.plot(
        *moving_average_y(data["weak"]["eps"][s:], data["weak"]["sig"][s:], args.moving_average),
        c="k",
    )

    lim = ax.get_ylim()
    ax.set_ylim([0, lim[-1]])

    s = data["snapshot"]["steadystate"]

    eps = data["snapshot"]["eps"][:s]
    x = np.zeros((2, eps.size), dtype=float)
    y = np.zeros((2, eps.size), dtype=float)
    x[0, :] = eps
    x[1, :] = eps
    y[1, :] = lim[-1]

    ax.plot(x, y, c="r", lw=1)

    eps = data["snapshot"]["eps"][s:]
    x = np.zeros((2, eps.size), dtype=float)
    y = np.zeros((2, eps.size), dtype=float)
    x[0, :] = eps
    x[1, :] = eps
    y[1, :] = lim[-1]

    ax.plot(x, y, c="b", lw=1)

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\sigma$")

    if args.save:
        fig.savefig(args.save)
    else:
        plt.show()

    plt.close(fig)


def cli_plot_velocityjump(cli_args=None):
    """
    Plot overview of flow simulation.
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

    parser.add_argument("-a", "--moving-average", type=int, help="Apply moving average")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=str, help="The initial simulation")
    parser.add_argument("jump", type=str, help="The velocity jump")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.source)
    assert os.path.isfile(args.jump)

    with h5py.File(args.source, "r") as file:
        source = basic_output(file)

    with h5py.File(args.jump, "r") as file:
        jump = basic_output(file)
        prog = entry_points["cli_branch_velocityjump"]
        inc0 = file[f"/meta/{prog}"].attrs["inc"]
        eps0 = jump["meta"]["inc2eps"] * inc0

    fig, ax = plt.subplots()

    s = source["weak"]["steadystate"]

    x, y = moving_average_y(
        source["weak"]["eps"][:s], source["weak"]["sig"][:s], args.moving_average
    )
    ax.plot(x, y, c=0.5 * np.ones(3))

    x, y = moving_average_y(
        source["weak"]["eps"][s:], source["weak"]["sig"][s:], args.moving_average
    )
    ax.plot(x, y, c="k")

    x, y = moving_average_y(eps0 + jump["weak"]["eps"], jump["weak"]["sig"], args.moving_average)
    ax.plot(x, y, c="r")

    ax.plot(eps0 * np.ones(2), ax.get_ylim(), c="g")

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\sigma$")

    plt.show()
