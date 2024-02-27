from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import re
import sys
import textwrap
import uuid

import click
import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import shelephant
import tqdm
from numpy.typing import ArrayLike

from . import QuasiStatic
from . import storage
from . import tag
from . import tools
from ._version import version


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
            if key in ["N"]:
                info[key] = {"3^7": 3**7, "3^6x4": 3**6 * 4, "3^6x2": 3**6 * 4}[info[key]]
            elif key in ["gammadot", "jump", "alpha", "eta"]:
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

    filepath = os.path.basename(filepath)

    if filepath.endswith(".h5"):
        filepath = filepath[:-3]
    if filepath.endswith(".hdf5"):
        filepath = filepath[:-5]

    return _interpret(re.split("_|/", filepath), convert=convert)


def generate(*args, **kwargs):
    """
    Generate input file.
    See :py:func:`slip_nucleation_2023.QuasiStatic.generate`. On top of that:

    :param v: Slip-rate to apply.
    :param output: Output storage interval.
    :param restart: Restart storage interval.
    :param snapshot: Snapshot storage interval.
    """

    kwargs.setdefault("init_run", False)
    v = kwargs.pop("v", None)
    gammadot = kwargs.pop("gammadot", None)
    output = kwargs.pop("output")
    restart = kwargs.pop("restart")
    snapshot = kwargs.pop("snapshot")
    assert v is not None or gammadot is not None

    QuasiStatic.generate(*args, **kwargs)

    with h5py.File(kwargs["filepath"], "a") as file:
        meta = file.create_group("/meta/Flow_Generate")
        meta.attrs["version"] = version

        if gammadot is None:
            norm = QuasiStatic.normalisation(file)
            y = file["/param/coor"][:, 1]
            L = np.max(y) - np.min(y)
            t0 = norm["t0"]
            eps0 = norm["eps0"]
            vtop = 2 * v * norm["l0"]
            gammadot = vtop / (L / eps0 * t0)

        storage.dump_with_atttrs(
            file,
            "/Flow/gammadot",
            gammadot,
            desc="Applied shear-rate",
        )

        storage.dump_with_atttrs(
            file,
            "/Flow/boundcheck",
            10,
            desc="Stop at n potentials before running out of bounds",
        )

        storage.dump_with_atttrs(
            file,
            "/Flow/restart/interval",
            restart,
            desc="Restart storage interval",
        )

        storage.dump_with_atttrs(
            file,
            "/Flow/output/interval",
            output,
            desc="Output storage interval",
        )

        storage.dump_with_atttrs(
            file,
            "/Flow/snapshot/interval",
            snapshot,
            desc="Snapshot storage interval",
        )


class DefaultEnsemble:
    v = np.linspace(0.0, 0.4, 21)[1:]
    output = 500 * np.ones(v.shape, dtype=int)
    restart = 10000 * np.ones(v.shape, dtype=int)
    snapshot = 500 * 500 * np.ones(v.shape, dtype=int)
    n = v.size


@tools.docstring_append_cli()
def Generate(cli_args: list = None, _return_parser: bool = False):
    """
    Generate IO files, including job-scripts to run simulations.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("--force", action="store_true", help="Overwrite job")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--scale-alpha", type=float, help="Scale general damping")
    parser.add_argument("--eta", type=float, help="Damping at the interface")
    parser.add_argument(
        "--slip-rate", type=float, action="append", default=[], help="Run at specific slip rate"
    )
    parser.add_argument(
        "--gammadot", type=float, action="append", default=[], help="Run at specific gammadot"
    )
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=4 * (3**6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("outdir", type=str, help="Output directory")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert not (len(args.slip_rate) > 0 and len(args.gammadot) > 0)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.scale_alpha is None and args.eta is None:
        args.scale_alpha = 1.0

    Ensemble = DefaultEnsemble
    filenames = []
    filepaths = []

    if len(args.slip_rate) > 0:
        Ensemble.v = np.array(args.slip_rate)

    if len(args.gammadot) > 0:
        Ensemble.v = np.array(args.gammadot)

    if len(args.slip_rate) > 0 or len(args.gammadot) > 0:
        Ensemble.output = Ensemble.output[0] * np.ones(
            Ensemble.v.shape, dtype=Ensemble.output.dtype
        )
        Ensemble.restart = Ensemble.restart[0] * np.ones(
            Ensemble.v.shape, dtype=Ensemble.restart.dtype
        )
        Ensemble.snapshot = Ensemble.snapshot[0] * np.ones(
            Ensemble.v.shape, dtype=Ensemble.snapshot.dtype
        )
        Ensemble.n = Ensemble.v.size

    for i in tqdm.tqdm(range(args.start, args.start + args.nsim)):
        for j in range(Ensemble.n):
            filename = f"id={i:03d}_v={Ensemble.v[j]:.4f}".replace(".", ",") + ".h5"
            filepath = str(outdir / filename)
            assert not pathlib.Path(filepath).exists()
            filenames.append(filename)
            filepaths.append(filepath)

            if len(args.gammadot) > 0:
                opts = {"gammadot": Ensemble.v[j]}
            else:
                opts = {"v": Ensemble.v[j]}

            generate(
                filepath=filepath,
                output=Ensemble.output[j],
                restart=Ensemble.restart[j],
                snapshot=Ensemble.snapshot[j],
                N=args.size,
                seed=i * args.size,
                scale_alpha=args.scale_alpha,
                eta=args.eta,
                dev=args.develop,
                **opts,
            )

    commands = [f"Flow_Run {file}" for file in filenames]
    shelephant.yaml.dump(outdir / "commands.yaml", commands, args.force)

    if cli_args is not None:
        return filepaths


def run_create_extendible(file: h5py.File):
    """
    Create extendible datasets used in :py:func:`run`.
    """

    flatindex = dict(
        xx=0,
        xy=1,
        yy=2,
    )

    output = file["/Flow/output"]
    snapshot = file["/Flow/snapshot"]

    # reaction force
    storage.create_extendible(output, "fext", np.float64, unit="sig0 (normalised stress)")

    # averaged on the weak layer
    storage.create_extendible(output, "sig_visco", np.float64, ndim=2, unit="sig0", **flatindex)
    storage.create_extendible(output, "sig", np.float64, ndim=2, unit="sig0", **flatindex)
    storage.create_extendible(output, "eps", np.float64, ndim=2, unit="eps0", **flatindex)
    storage.create_extendible(output, "epsp", np.float64, unit="eps0")

    # book-keeping
    storage.create_extendible(output, "inc", np.uint32)
    storage.create_extendible(snapshot, "inc", np.uint32)


def __velocity_preparation(system, gammadot):
    return system.affineSimpleShear(gammadot)


def __velocity_steadystate(system, gammadot):
    conn = system.conn
    coor = system.coor
    mesh = GooseFEM.Mesh.Quad4.FineLayer(coor=coor, conn=conn)
    H = np.max(coor[:, 1]) - np.min(coor[:, 0])

    v = np.zeros_like(coor)
    v[mesh.nodesTopEdge, 0] = gammadot * H

    return v


def run(filepath: str, dev: bool = False, progress: bool = True):
    """
    Run flow simulation.

    :param filepath: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    :param progress: Show progress bar.
    """

    basename = os.path.basename(filepath)
    opts = {}
    if not progress:
        opts["disable"] = True

    with h5py.File(filepath, "a") as file:
        system = QuasiStatic.System(file)
        meta = QuasiStatic.create_check_meta(file, "/meta/Flow_Run", dev=dev)
        dV = system.plastic_dV(rank=2)

        if "completed" in meta.attrs:
            if meta.attrs["completed"]:
                print(f'"{basename}": marked completed, skipping')
                return 1

        run_create_extendible(file)

        inc = 0
        output = file["/Flow/output/interval"][...]
        restart = file["/Flow/restart/interval"][...]
        snapshot = (
            file["/Flow/snapshot/interval"][...]
            if "/Flow/snapshot/interval" in file
            else sys.maxsize
        )
        v = __velocity_preparation(system, file["/Flow/gammadot"][...])
        init = True
        i_n = np.copy(system.plastic.i.astype(int))

        boundcheck = file["/Flow/boundcheck"][...]
        nchunk = file["/param/cusp/epsy/nchunk"][...] + 2 - boundcheck
        pbar = tqdm.tqdm(total=nchunk, desc=filepath, **opts)

        if "/Flow/restart/u" in file:
            system.u = file["/Flow/restart/u"][...]
            system.v = file["/Flow/restart/v"][...]
            system.a = file["/Flow/restart/a"][...]
            inc = int(file["/Flow/restart/inc"][...])
            pbar.n = np.max(system.plastic.i)
            pbar.refresh()

        mesh = GooseFEM.Mesh.Quad4.FineLayer(system.coor, system.conn)
        top = mesh.nodesTopEdge
        h = system.coor[top[1], 0] - system.coor[top[0], 0]

        while True:
            if init:
                if np.all(system.plastic.i.astype(int) - i_n > 1):
                    v = __velocity_steadystate(system, file["/Flow/gammadot"][...])
                    init = False

            n = min(
                [
                    output - inc % output,
                    restart - inc % restart,
                    snapshot - inc % snapshot,
                ]
            )

            ret = system.flowSteps(n, v, nmargin=boundcheck)

            if ret < 0:
                break

            pbar.n = np.max(system.plastic.i)
            pbar.refresh()
            inc += n

            if inc % snapshot == 0:
                i = int(inc / snapshot)

                for key in ["/Flow/snapshot/inc"]:
                    file[key].resize((i + 1,))

                file["/Flow/snapshot/inc"][i] = inc
                file[f"/Flow/snapshot/u/{inc:d}"] = system.u
                file[f"/Flow/snapshot/v/{inc:d}"] = system.v
                file[f"/Flow/snapshot/a/{inc:d}"] = system.a
                file.flush()

            if inc % output == 0:
                i = int(inc / output)

                for key in ["/Flow/output/inc", "/Flow/output/fext", "/Flow/output/epsp"]:
                    file[key].resize((i + 1,))

                for key in ["/Flow/output/sig_visco", "/Flow/output/sig", "/Flow/output/eps"]:
                    file[key].resize((3, i + 1))

                Sigdamp_weak = np.average(
                    system.eta * system.Epsdot()[system.plastic_elem, ...] / system.sig0,
                    weights=dV,
                    axis=(0, 1),
                )
                Eps_weak = np.average(system.plastic.Eps / system.eps0, weights=dV, axis=(0, 1))
                Sig_weak = np.average(system.plastic.Sig / system.sig0, weights=dV, axis=(0, 1))

                fext = system.fext[top, 0]
                fext[0] += fext[-1]
                fext = 2 * np.mean(fext[:-1]) / h / system.normalisation["sig0"]
                # factor 2 because of normalisation of equivalent stress

                file["/Flow/output/inc"][i] = inc
                file["/Flow/output/fext"][i] = fext
                file["/Flow/output/epsp"][i] = np.mean(system.plastic.epsp / system.eps0)
                file["/Flow/output/eps"][:, i] = Eps_weak.ravel()[[0, 1, 3]]
                file["/Flow/output/sig"][:, i] = Sig_weak.ravel()[[0, 1, 3]]
                file["/Flow/output/sig_visco"][:, i] = Sigdamp_weak.ravel()[[0, 1, 3]]
                file.flush()

            if inc % restart == 0:
                storage.dump_overwrite(file, "/Flow/restart/u", system.u)
                storage.dump_overwrite(file, "/Flow/restart/v", system.v)
                storage.dump_overwrite(file, "/Flow/restart/a", system.a)
                storage.dump_overwrite(file, "/Flow/restart/inc", inc)
                file.flush()

        meta.attrs["completed"] = 1


@tools.docstring_append_cli()
def Run(cli_args: list = None, _return_parser: bool = False):
    """
    Run flow simulation.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    run(args.file, dev=args.develop, progress=not args.quiet)


def basic_output(file: h5py.File) -> dict:
    """
    Extract basic output.
    Strain rate are obtained simply by taking the ratio of the differences in strain rate
    and time interval of to subsequent stored states (interval controlled in ``/output/interval``).

    :param file: HDF5-archive.

    :return: Basic output as follows::
        v: Imposed slip rate at the boundary (scalar).
        t: Time [n].
        eps_applied: Applied strain [n].
        sig_visco: Stress due to viscous damping [n].
        fext: External force (stress at the boundary) [n].
        sig: Mean stress at the interface [n].
        eps: Mean strain at the interface [n].
        epsp: Mean plastic strain at the interface [n].
        epsdot: Strain rate at the interface [n].
        epspdot: Plastic strain rate at the interface [n].
    """

    gammadot = file["/Flow/gammadot"][...]
    norm = QuasiStatic.normalisation(file)

    dt = norm["dt"]
    t0 = norm["t0"]
    eps0 = norm["eps0"]
    y = file["/param/coor"][:, 1]
    L = np.max(y) - np.min(y)
    inc = file["/Flow/output/inc"][...]
    vtop = gammadot * L / eps0 * t0

    sig = file["/Flow/output/sig"][...]
    eps = file["/Flow/output/eps"][...]

    if "sig_visco" in file["/Flow/output"]:
        sig_visco = file["/Flow/output/sig_visco"][...]
    elif "sig_damp" in file["/Flow/output"]:
        assert tag.less_equal(file["/meta/Flow_Run"].attrs["version"], "15.13")
        sig_visco = file["/Flow/output/sig_damp"][...]
    else:
        epsdot = np.diff(eps * eps0, prepend=0, axis=1) / np.diff(inc * dt, prepend=1)
        sig_visco = file["/param/eta"][...] * epsdot / norm["sig0"]

    if sig.size == 0:
        return {}

    ret = {}
    ret["fext"] = file["/Flow/output/fext"][...]
    ret["epsp"] = file["/Flow/output/epsp"][...]
    ret["sig_visco"] = tools.sigd(
        xx=sig_visco[0, :], xy=sig_visco[1, :], yy=sig_visco[2, :]
    ).ravel()
    ret["sig"] = tools.sigd(xx=sig[0, :], xy=sig[1, :], yy=sig[2, :]).ravel()
    ret["eps"] = tools.epsd(xx=eps[0, :], xy=eps[1, :], yy=eps[2, :]).ravel()
    ret["t"] = inc * dt / t0
    ret["eps_applied"] = inc * dt * file["/Flow/gammadot"][...] / eps0
    ret["epsdot"] = np.diff(ret["eps"], prepend=0) / np.diff(ret["t"], prepend=1)
    ret["epspdot"] = np.diff(ret["epsp"], prepend=0) / np.diff(ret["t"], prepend=1)
    ret["v"] = 0.5 * vtop / norm["l0"]

    if tag.less_equal(file["/meta/Flow_Run"].attrs["version"], "15.12"):
        ret["fext"] *= 2

    dinc = np.diff(inc)
    assert np.all(dinc == dinc[0])

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    tools.check_docstring(doc, ret, ":return:")

    return ret


@tools.docstring_append_cli()
def EnsembleInfo(cli_args: list = None, _return_parser: bool = False):
    """
    Collect basic ensemble information.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument(
        "-o", "--output", type=str, default="Flow_EnsembleInfo.h5", help="Output file"
    )
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Simulation output")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert np.all([os.path.isfile(i) for i in args.files])
    tools._check_overwrite_file(args.output, args.force)

    pbar = tqdm.tqdm(args.files)

    with h5py.File(args.output, "w") as output:
        for ifile, filename in enumerate(pbar):
            pbar.set_description(filename)
            pbar.refresh()

            prefix = os.path.relpath(filename, os.path.dirname(args.output))

            with h5py.File(filename) as file:
                if ifile == 0:
                    g5.copy(file, output, "/param")
                else:
                    test = g5.compare(
                        file,
                        output,
                        g5.getdatapaths(file, root="/param"),
                        g5.getdatapaths(output, root="/param"),
                        attrs=False,
                        close=True,
                    )

                    if "/param/cusp/epsy/deps" in test["->"]:
                        g5.copy(file, output, "/param/cusp/epsy/deps")
                        test["->"].remove("/param/cusp/epsy/deps")
                        test["=="].append("/param/cusp/epsy/deps")
                    if "/param/cusp/epsy/deps" in test["<-"]:
                        test["<-"].remove("/param/cusp/epsy/deps")

                    assert len(test["!="]) == 0
                    assert len(test["->"]) == 0
                    assert len(test["<-"]) == 0
                    assert len(test["=="]) > 0

                out = basic_output(file)
                for key in out:
                    output[g5.join(f"/full/{prefix}/{key}")] = out[key]

                src = g5.getdatapaths(file, root="/realisation")
                dest = [g5.join(f"/full/{prefix}/{i.split('/realisation/')[1]}") for i in src]
                g5.copy(file, output, src, dest)


@tools.docstring_append_cli()
def VelocityJump_Branch(cli_args: list = None, _return_parser: bool = False):
    """
    Branch simulation to a velocity jump experiment:
    Copies a snapshot as restart.
    To run simply use :py:func:`Run`.
    Note that if no new flow rate(s) is specified the default ensemble is again generated.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-i", "--inc", type=int, required=True, help="Increment to branch")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Flow simulation to branch from")
    parser.add_argument("gammadot", type=float, nargs="*", help="New flow rate")

    if _return_parser:
        return parser

    raise NotImplementedError("Reimplement for v.")
    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file) as source:
        assert str(args.inc) in source["Flow"]["snapshot"]["u"]

    Gammadot = np.array(args.gammadot)

    out_names = []
    out_paths = []
    out_gammadots = []

    info = interpret_filename(os.path.basename(args.file))
    include = np.logical_not(np.isclose(Gammadot / float(info["gammadot"]), 1.0))

    for gammadot in Gammadot[include]:
        name = "id={id}_gammadot={gammadot}_jump={jump:.2e}.h5".format(jump=gammadot, **info)
        out_names.append(name)
        out_paths.append(str(outdir / name))
        out_gammadots.append(gammadot)

    assert not np.any([os.path.exists(f) for f in out_paths])

    with h5py.File(args.file) as source:
        for out_gammadot, out_path in zip(tqdm.tqdm(out_gammadots), out_paths):
            with h5py.File(out_path, "w") as dest:
                meta = "/meta/Flow_Run"
                paths = g5.getdatapaths(source)
                paths = [p for p in paths if not re.match(r"(/Flow/snapshot/)(.*)", p)]
                paths = [p for p in paths if not re.match(r"(/Flow/output/)(.*)", p)]
                paths = [p for p in paths if not re.match(r"(/Flow/restart/)(.*)", p)]
                paths = [p for p in paths if not re.match(f"({meta})(.*)", p)]
                g5.copy(source, dest, paths)

                for t in ["restart", "output", "snapshot"]:
                    g5.copy(source, dest, f"/Flow/{t}/interval")

                g5.copy(
                    source,
                    dest,
                    "/meta/Flow_Run",
                    "/meta/Flow_Run_source",
                )

                meta = QuasiStatic.create_check_meta(
                    dest, "/meta/Flow_VelocityJump_Branch", dev=args.develop
                )
                meta.attrs["inc"] = args.inc

                dest["/Flow/gammadot"][...] = out_gammadot
                dest["/Flow/restart/inc"] = 0
                dest["/Flow/restart/u"] = source[f"/Flow/snapshot/u/{args.inc:d}"][...]
                dest["/Flow/restart/v"] = source[f"/Flow/snapshot/v/{args.inc:d}"][...]
                dest["/Flow/restart/a"] = source[f"/Flow/snapshot/a/{args.inc:d}"][...]
                dest["/Flow/snapshot/u/0"] = source[f"/Flow/snapshot/u/{args.inc:d}"][...]
                dest["/Flow/snapshot/v/0"] = source[f"/Flow/snapshot/v/{args.inc:d}"][...]
                dest["/Flow/snapshot/a/0"] = source[f"/Flow/snapshot/a/{args.inc:d}"][...]

                run_create_extendible(dest)

                i = int(args.inc / source["/Flow/output/interval"][...])

                for key in ["/Flow/output/inc"]:
                    dest[key].resize((1,))
                    dest[key][0] = 0

                for key in ["/Flow/output/fext", "/Flow/output/epsp"]:
                    dest[key].resize((1,))
                    dest[key][0] = source[key][i]

                for key in ["/Flow/output/sig", "/Flow/output/eps"]:
                    dest[key].resize((3, 1))
                    dest[key][:, 0] = source[key][:, i]

    commands = ["Flow_Run {file}" for file in out_names]
    shelephant.yaml.dump(str(outdir / "commands.yaml"), commands, args.develop)

    if cli_args is not None:
        return out_paths


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


@tools.docstring_append_cli()
def Paraview(cli_args: list = None, _return_parser: bool = False):
    """
    Prepare snapshots to be viewed with ParaView.
    """

    import XDMFWrite_h5py as xh

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Appended xdmf/h5py")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    tools._check_overwrite_file(f"{args.output}.h5", args.force)
    tools._check_overwrite_file(f"{args.output}.xdmf", args.force)

    with h5py.File(args.file) as file, h5py.File(f"{args.output}.h5", "w") as output, xh.TimeSeries(
        f"{args.output}.xdmf"
    ) as xdmf:
        system = QuasiStatic.System(file)

        output["/coor"] = system.coor
        output["/conn"] = system.conn

        for inc in file["/Flow/snapshot/inc"][...]:
            if inc == 0 and f"/Flow/snapshot/u/{inc:d}" not in file:
                system.u = np.zeros_like(system.coor)
                system.v = np.zeros_like(system.coor)
                system.a = np.zeros_like(system.coor)
            else:
                system.u = file[f"/Flow/snapshot/u/{inc:d}"][...]
                system.v = file[f"/Flow/snapshot/v/{inc:d}"][...]
                system.a = file[f"/Flow/snapshot/a/{inc:d}"][...]

            output[f"/disp/{inc:d}"] = xh.as3d(system.u)
            output[f"/Sig/{inc:d}"] = GMat.Sigd(np.mean(system.Sig() / system.sig0, axis=1))
            output[f"/Eps/{inc:d}"] = GMat.Epsd(np.mean(system.Eps() / system.eps0, axis=1))
            output[f"/Epsdot/{inc:d}"] = GMat.Epsd(
                np.mean(system.Epsdot() / system.eps0 * system.t0, axis=1)
            )
            output[f"/Epsddot/{inc:d}"] = GMat.Epsd(
                np.mean(system.Epsddot() / system.eps0 * system.t0**2, axis=1)
            )

            xdmf += xh.TimeStep()
            xdmf += xh.Unstructured(output["/coor"], output["/conn"], "Quadrilateral")
            xdmf += xh.Attribute(output[f"/disp/{inc:d}"], "Node", name="Displacement")
            xdmf += xh.Attribute(output[f"/Sig/{inc:d}"], "Cell", name="Stress")
            xdmf += xh.Attribute(output[f"/Eps/{inc:d}"], "Cell", name="Strain")
            xdmf += xh.Attribute(output[f"/Epsdot/{inc:d}"], "Cell", name="Strain rate")
            xdmf += xh.Attribute(
                output[f"/Epsddot/{inc:d}"], "Cell", name="Symgrad of accelerations"
            )


@tools.docstring_append_cli()
def Plot(cli_args: list = None, _return_parser: bool = False):
    """
    Plot overview of flow simulation.
    """

    import GooseMPL as gplt
    import matplotlib.pyplot as plt

    plt.style.use(["goose", "goose-latex"])
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("--sigma-max", type=float, help="Set limit of y-axis of left panel")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, help="Save the image")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file) as file:
        out = basic_output(file)
        N = file["/param/normalisation/N"][...]

    fig, axes = gplt.subplots(ncols=3)

    ax = axes[0]

    ax.plot(out["eps_applied"], out["fext"], c="k", label=r"$f_\text{ext}$", zorder=100)
    ax.plot(out["eps_applied"], out["sig"], c="r", label=r"$\sigma_\text{interface}$")
    ax.plot(
        out["eps_applied"],
        out["sig"] + out["sig_visco"],
        c="g",
        label=r"$\sigma_\text{interface} + \sigma_\text{damping}$",
    )

    ax.set_xlim([0, ax.get_xlim()[-1]])

    if args.sigma_max is not None:
        ax.set_ylim([0, args.sigma_max])
    else:
        ax.set_ylim([0, ax.get_ylim()[-1]])

    ax.set_xlabel(r"$\bar{\varepsilon}$")
    ax.set_ylabel(r"$\sigma$")

    ax.legend()

    ax = axes[1]

    n = r"\dot{\varepsilon}"
    ax.plot(out["eps_applied"], out["epsdot"], c="k", label=rf"${n}_\text{{interface}}$")
    ax.axhline(out["v"], c="r", label=rf"${n}_\text{{applied}}$")

    ax.set_xlabel(r"$\bar{\varepsilon}$")
    ax.set_ylabel(r"$\dot{\varepsilon}$")

    ax.legend()

    ax = axes[2]

    ax.plot(out["eps_applied"], out["epsp"] / N, c="k")

    ax.set_xlabel(r"$\bar{\varepsilon}$")
    ax.set_ylabel(r"$\varepsilon_\mathrm{p} / N$")

    if args.output:
        tools._check_overwrite_file(args.output, args.force)
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)


@tools.docstring_append_cli()
def TransformDeprecated(cli_args: list = None, _return_parser: bool = False):
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
            -r "/gammadot" "/Flow/gammadot" \
            -r "/output" "/Flow/output" \
            -r "/snapshot" "/Flow/snapshot" \
            -r "/restart" "/Flow/restart" \
            -r "/boundcheck" "/Flow/boundcheck" \
            -r "/meta/Flow_generate" "/meta/Flow_Generate" \
            -r "/meta/Flow_run" "/meta/Flow_Run" \
            -r "/meta/Run_generate" "/meta/QuasiStatic_Generate" \
            foo.h5.bak foo.h5
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="File to transform: .bak appended")

    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert not os.path.isfile(args.file + ".bak")
    os.rename(args.file, args.file + ".bak")

    with h5py.File(args.file + ".bak") as src, h5py.File(args.file, "w") as dest:
        old = ["/boundcheck", "/gammadot", "/output", "/snapshot", "/restart"]
        fold = old + ["/meta/normalisation"]
        paths = list(g5.getdatapaths(src, fold=fold, fold_symbol=""))
        paths = QuasiStatic.transform_deprecated_param(src, dest, paths)
        dest.create_group("Flow")

        rename = {i: g5.join("/Flow", i) for i in old}
        rename["/meta/Run_generate"] = "/meta/QuasiStatic_Generate"
        rename["/meta/Flow_generate"] = "/meta/Flow_Generate"
        rename["/meta/Flow_run"] = "/meta/Flow_Run"
        rename["/meta/normalisation"] = "/param/normalisation"

        for key in rename:
            if key not in src:
                continue
            g5.copy(src, dest, key, rename[key])
            paths.remove(key)

        assert "/param/normalisation" in dest

        dest.create_group("/meta/Flow_TransformDeprecated").attrs["version"] = version

        if "Flow_Run" in dest["meta"]:
            if "uuid" not in dest["/meta/Flow_Run"].attrs:
                dest["/meta/Flow_Run"].attrs["uuid"] = str(uuid.uuid4())

        assert len(paths) == 0


@tools.docstring_append_cli()
def Rename(cli_args: list = None, _return_parser: bool = False):
    """
    Update file name to the current version.
    """
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=tools.MyFmt, description=tools._fmt_doc(doc))

    parser.add_argument("files", type=str, nargs="*", help="Files")
    if _return_parser:
        return parser

    args = tools._parse(parser, cli_args)
    newnames = []

    for filename in args.files:
        assert len(pathlib.Path(filename).stem.split("id")) == 2
        assert len(pathlib.Path(filename).stem.split("gammadot")) == 2

        with h5py.File(filename) as file:
            gammadot = file["/Flow/gammadot"][...]
            norm = QuasiStatic.normalisation(file)

            t0 = norm["t0"]
            eps0 = norm["eps0"]
            y = file["/param/coor"][:, 1]
            L = np.max(y) - np.min(y)
            vtop = gammadot * L / eps0 * t0
            v = 0.5 * vtop / norm["l0"]

        i = int(pathlib.Path(filename).stem.split("id=")[1].split("_")[0])
        newnames.append(f"id={i:03d}_v={v:.4f}".replace(".", ",") + ".h5")

    for old, new in zip(args.files, newnames):
        print(f"{old} -> {new}")

    if not click.confirm("Continue?"):
        raise OSError("Cancelled")

    for old, new in zip(args.files, newnames):
        os.rename(old, new)


# <autodoc> generated by docs/conf.py


def _EnsembleInfo_parser() -> argparse.ArgumentParser:
    return EnsembleInfo(_return_parser=True)


def _Generate_parser() -> argparse.ArgumentParser:
    return Generate(_return_parser=True)


def _Paraview_parser() -> argparse.ArgumentParser:
    return Paraview(_return_parser=True)


def _Plot_parser() -> argparse.ArgumentParser:
    return Plot(_return_parser=True)


def _Rename_parser() -> argparse.ArgumentParser:
    return Rename(_return_parser=True)


def _Run_parser() -> argparse.ArgumentParser:
    return Run(_return_parser=True)


def _TransformDeprecated_parser() -> argparse.ArgumentParser:
    return TransformDeprecated(_return_parser=True)


def _VelocityJump_Branch_parser() -> argparse.ArgumentParser:
    return VelocityJump_Branch(_return_parser=True)


# </autodoc>
