from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import textwrap

import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numpy.typing import ArrayLike

from . import QuasiStatic
from . import slurm
from . import storage
from . import tools
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
    See :py:func:`mycode_front.QuasiStatic.generate`. On top of that:

    :param gammadot: The shear-rate to prescribe.
    :param output: Output storage interval.
    :param restart: Restart storage interval.
    :param snapshot: Snapshot storage interval.
    """

    kwargs.setdefault("init_run", False)
    gammadot = kwargs.pop("gammadot")
    output = kwargs.pop("output")
    restart = kwargs.pop("restart")
    snapshot = kwargs.pop("snapshot")

    progname = entry_points["cli_generate"]
    QuasiStatic.generate(*args, **kwargs)

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


class DefaultEnsemble:

    gammadot = np.array([5e-11, 8e-11] + np.linspace(1e-10, 2e-9, 20).tolist())
    eps0 = 1.0e-3 / 2.0 / 10.0
    output = 500 * np.ones(gammadot.shape, dtype=int)
    restart = 10000 * np.ones(gammadot.shape, dtype=int)
    snapshot = 500 * 500 * np.ones(gammadot.shape, dtype=int)
    n = gammadot.size


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

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--scale-alpha", type=float, default=1.0, help="Scale general damping")
    parser.add_argument("--eta", type=float, help="Damping at the interface")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=2 * (3**6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("outdir", type=str, help="Output directory")

    args = tools._parse(parser, cli_args)
    assert os.path.isdir(args.outdir)

    Ensemble = DefaultEnsemble
    filenames = []
    filepaths = []

    for i in tqdm.tqdm(range(args.start, args.start + args.nsim)):

        for j in range(Ensemble.n):

            filename = f"id={i:03d}_gammadot={Ensemble.gammadot[j]:.2e}.h5"
            filepath = os.path.join(args.outdir, filename)
            filenames.append(filename)
            filepaths.append(filepath)

            generate(
                filepath=filepath,
                gammadot=Ensemble.gammadot[j],
                output=Ensemble.output[j],
                restart=Ensemble.restart[j],
                snapshot=Ensemble.snapshot[j],
                N=args.size,
                seed=i * args.size,
                scale_alpha=args.scale_alpha,
                eta=args.eta,
                test_mode=args.develop,
                dev=args.develop,
            )

            # warning: Gammadot hard-coded here, check that yield strains did not change
            with h5py.File(filepath, "r") as file:
                assert not isinstance(file["/cusp/epsy"], h5py.Dataset)
                assert np.isclose(file["/cusp/epsy/eps0"][...], Ensemble.eps0)

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in filenames]
    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

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

    # reaction force
    storage.create_extendible(file, "/output/fext", np.float64)

    # averaged on the weak layer
    storage.create_extendible(file, "/output/sig", np.float64, ndim=2, **flatindex)
    storage.create_extendible(file, "/output/eps", np.float64, ndim=2, **flatindex)
    storage.create_extendible(file, "/output/epsp", np.float64)

    # book-keeping
    storage.create_extendible(file, "/output/inc", np.uint32)
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

        system = QuasiStatic.DimensionlessSystem(file)
        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=dev)
        dV = system.plastic_dV(rank=2)

        if "completed" in meta.attrs:
            print(f'"{basename}": marked completed, skipping')
            return 1

        run_create_extendible(file)

        inc = 0
        output = file["/output/interval"][...]
        restart = file["/restart/interval"][...]
        snapshot = file["/snapshot/interval"][...] if "/snapshot/interval" in file else sys.maxsize
        v = system.affineSimpleShear(file["/gammadot"][...])

        boundcheck = file["/boundcheck"][...]
        nchunk = file["/cusp/epsy/nchunk"][...] - boundcheck
        pbar = tqdm.tqdm(total=nchunk, disable=not progress, desc=filepath)

        if "/restart/u" in file:
            system.setU(file["/restart/u"][...])
            system.setV(file["/restart/v"][...])
            system.setA(file["/restart/a"][...])
            inc = int(file["/restart/inc"][...])
            pbar.n = np.max(system.plastic_CurrentIndex())
            pbar.refresh()

        mesh = GooseFEM.Mesh.Quad4.FineLayer(system.coor(), system.conn())
        top = mesh.nodesTopEdge()
        h = system.coor()[top[1], 0] - system.coor()[top[0], 0]

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
                file.flush()

            if inc % output == 0:

                i = int(inc / output)

                for key in ["/output/inc", "/output/fext", "/output/epsp"]:
                    file[key].resize((i + 1,))

                for key in ["/output/sig", "/output/eps"]:
                    file[key].resize((3, i + 1))

                Eps_weak = np.average(system.plastic_Eps(), weights=dV, axis=(0, 1))
                Sig_weak = np.average(system.plastic_Sig(), weights=dV, axis=(0, 1))

                fext = system.fext()[top, 0]
                fext[0] += fext[-1]
                fext = np.mean(fext[:-1]) / h / system.normalisation["sig0"]

                file["/output/inc"][i] = inc
                file["/output/fext"][i] = fext
                file["/output/epsp"][i] = np.mean(system.plastic_Epsp())
                file["/output/eps"][:, i] = Eps_weak.ravel()[[0, 1, 3]]
                file["/output/sig"][:, i] = Sig_weak.ravel()[[0, 1, 3]]
                file.flush()

            if inc % restart == 0:

                storage.dump_overwrite(file, "/restart/u", system.u())
                storage.dump_overwrite(file, "/restart/v", system.v())
                storage.dump_overwrite(file, "/restart/a", system.a())
                storage.dump_overwrite(file, "/restart/inc", inc)
                file.flush()

        meta.attrs["completed"] = 1


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

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    run(args.file, dev=args.develop, progress=not args.quiet)


def basic_output(file: h5py.File, interval=400) -> dict:
    """
    Extract basic output averaged on a time interval.
    Strain rate are obtained simply by taking the ratio of the differences in strain rate
    and time interval of to subsequent stored states (interval controlled in ``/output/interval``).

    :param file: HDF5-archive.

    :param interval:
        Number of stored increment to average.
        Note that the time window that will be averaged will depend on the interval at which
        ``/output/interval``.

    :return: Basic output as follows::
        sig: Mean stress at the interface [n].
        eps: Mean strain at the interface [n].
        fext: External force [n].
        epsdot: Strain rate [n].
        epspdot: Plastic strain rate [n].
        epsdot_remote: Strain rate imposed at the boundary [n].
        fext_std: Standard deviation corresponding to time averaging ``fext`` [n].
    """

    gammadot = file["/gammadot"][...]
    norm = QuasiStatic.normalisation(file)

    dt = norm["dt"]
    t0 = norm["t0"]
    eps0 = norm["eps0"]
    inc = file["/output/inc"][...]

    data = {}
    sig = file["/output/sig"][...]
    eps = file["/output/eps"][...]
    data["fext"] = file["/output/fext"][...]
    data["epsp"] = file["/output/epsp"][...]
    data["sig"] = tools.sigd(xx=sig[0, :], xy=sig[1, :], yy=sig[2, :]).ravel()
    data["eps"] = tools.epsd(xx=eps[0, :], xy=eps[1, :], yy=eps[2, :]).ravel()
    data["t"] = inc * dt / t0
    data["epsdot"] = np.diff(data["eps"], prepend=0) / np.diff(data["t"], prepend=1)
    data["epspdot"] = np.diff(data["epsp"], prepend=0) / np.diff(data["t"], prepend=1)
    data["eps_remote"] = gammadot * dt * inc / eps0
    data["epsdot_remote"] = np.diff(data["eps_remote"], prepend=0) / np.diff(data["t"], prepend=1)

    dinc = np.diff(inc)
    assert np.all(dinc == dinc[0])
    dinc = dinc[0]

    store = ["eps", "sig", "fext", "epsdot", "epspdot", "epsdot_remote"]

    n = int((inc.size - inc.size % interval) / interval)
    ret = {}
    for key in store:
        ret[key] = np.zeros(n, dtype=float)
        ret[f"{key}_std"] = np.zeros(n, dtype=float)

    for i in range(0, inc.size, interval):
        u = i + interval
        for key in store:
            ret[key][int(i / interval - 1)] = np.mean(data[key][i:u])
            ret[f"{key}_std"][int(i / interval - 1)] = np.std(data[key][i:u])

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

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert np.all([os.path.isfile(i) for i in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        for filename in tqdm.tqdm(args.files):
            with h5py.File(filename) as file:
                out = basic_output(file)
                for key in out:
                    output[g5.join(f"/full/{filename}/{key}")] = out[key]


def cli_branch_velocityjump(cli_args=None):
    """
    Branch simulation to a velocity jump experiment:
    Copies a snapshot as restart.
    To run simply use :py:func:`cli_run`.
    Note that if no new flow rate(s) is specified the default ensemble is again generated.
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

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-i", "--inc", type=int, required=True, help="Increment to branch")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("file", type=str, help="Flow simulation to branch from")
    parser.add_argument("gammadot", type=float, nargs="*", help="New flow rate")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file, "r") as source:
        assert str(args.inc) in source["snapshot"]["u"]

    if len(args.gammadot) == 0:
        Gammadot = DefaultEnsemble.gammadot
    else:
        Gammadot = np.array(args.gammadot)

    out_names = []
    out_paths = []
    out_gammadots = []

    info = interpret_filename(os.path.basename(args.file))
    include = np.logical_not(np.isclose(Gammadot / float(info["gammadot"]), 1.0))

    for gammadot in Gammadot[include]:

        name = "id={id}_gammadot={gammadot}_jump={jump:.2e}.h5".format(jump=gammadot, **info)
        out_names.append(name)
        out_paths.append(os.path.join(args.outdir, name))
        out_gammadots.append(gammadot)

    assert not np.any([os.path.exists(f) for f in out_paths])

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    with h5py.File(args.file, "r") as source:

        for out_gammadot, out_path in zip(tqdm.tqdm(out_gammadots), out_paths):

            with h5py.File(out_path, "w") as dest:

                meta = f"/meta/{entry_points['cli_run']}"
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
                    [f"/meta/{entry_points['cli_run']}"],
                    [f"/meta/{entry_points['cli_run']}_source"],
                )

                meta = QuasiStatic.create_check_meta(dest, f"/meta/{progname}", dev=args.develop)
                meta.attrs["inc"] = args.inc

                dest["/gammadot"][...] = out_gammadot
                dest["/restart/inc"] = 0
                dest["/restart/u"] = source[f"/snapshot/u/{args.inc:d}"][...]
                dest["/restart/v"] = source[f"/snapshot/v/{args.inc:d}"][...]
                dest["/restart/a"] = source[f"/snapshot/a/{args.inc:d}"][...]
                dest["/snapshot/u/0"] = source[f"/snapshot/u/{args.inc:d}"][...]
                dest["/snapshot/v/0"] = source[f"/snapshot/v/{args.inc:d}"][...]
                dest["/snapshot/a/0"] = source[f"/snapshot/a/{args.inc:d}"][...]

                run_create_extendible(dest)

                i = int(args.inc / source["/output/interval"][...])

                for key in ["/output/inc"]:
                    dest[key].resize((1,))
                    dest[key][0] = 0

                for key in ["/output/fext", "/output/epsp"]:
                    dest[key].resize((1,))
                    dest[key][0] = source[key][i]

                for key in ["/output/sig", "/output/eps"]:
                    dest[key].resize((3, 1))
                    dest[key][:, 0] = source[key][:, i]

    executable = entry_points["cli_run"]
    commands = [f"{executable} {file}" for file in out_names]
    slurm.serial_group(
        commands,
        basename=executable,
        group=1,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

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

    parser.add_argument("-s", "--save", type=str, help="Save the image")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    with h5py.File(args.file, "r") as file:
        Eps = file["output"]["eps"][...]
        Sig = file["output"]["sig"][...]
        fext = file["output"]["fext"][...]
        out = basic_output(file)

    meps = out["eps"][...]
    msig = out["sig"][...]
    mfext = out["fext"][...]

    eps = tools.epsd(xx=Eps[0, :], xy=Eps[1, :], yy=Eps[2, :]).ravel()
    sig = tools.sigd(xx=Sig[0, :], xy=Sig[1, :], yy=Sig[2, :]).ravel()

    i = np.argsort(eps)
    eps = eps[i]
    sig = sig[i]
    fext = fext[i]

    fig, ax = plt.subplots()
    ax.plot(eps, sig, c="k", label=r"$\sigma_\text{interface}$")
    ax.plot(eps, fext, c="r", label=r"$f_\text{ext}$")
    ax.plot(meps, msig, c="b", marker="o", ls="none", label=r"$\bar{\sigma}_\text{interface}$")
    ax.plot(meps, mfext, c="tab:orange", marker="o", ls="none", label=r"$\bar{f}_\text{ext}$")
    ax.set_xlim([0, ax.get_xlim()[-1]])
    ax.set_ylim([0, ax.get_ylim()[-1]])
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\sigma$")
    ax.legend()

    if args.save:
        fig.savefig(args.save)
    else:
        plt.show()

    plt.close(fig)
