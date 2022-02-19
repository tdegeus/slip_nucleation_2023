"""
Take the system to a certain state and trigger an event.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import click
import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM  # noqa: F401
import h5py
import numpy as np
import tqdm

from . import slurm
from . import storage
from . import System
from . import tools
from ._version import version

entry_points = dict(
    cli_run="Trigger_run",
    cli_job_strain="Trigger_JobStrain",
    cli_job_deltasigma="Trigger_JobDeltaSigma",
    cli_ensembleinfo="Trigger_EnsembleInfo",
)

file_defaults = dict(
    cli_ensembleinfo="Trigger_EnsembleInfo.h5",
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def cli_run(cli_args=None):
    """
    Trigger event and minimise energy.

    An option is provided to truncate the simulation when an event is system-spanning.
    In that case the ``truncated`` meta-attribute will be ``True``.
    The displacement field will not correspond to a mechanical equilibrium, while the state
    at truncation will be stored under ``restart``.
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
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    pbar = tqdm.tqdm(total=1)
    pbar.set_description(args.file)

    with h5py.File(args.file, "a") as file:

        System.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        system = System.init(file)

        inc = int(file["/stored"][-1])
        System._restore_inc(file, system, inc)
        idx_n = system.plastic_CurrentIndex()[:, 0]

        assert not file["/trigger/truncated"][inc]
        if file["/trigger/element"].size - 1 == inc:
            storage.dset_extend1d(file, "/trigger/element", inc + 1, file["/trigger/element"][inc])

        system.triggerElementWithLocalSimpleShear(
            file["/run/epsd/kick"][...], file["/trigger/element"][inc + 1]
        )

        if args.truncate_system_spanning:
            niter = system.minimise_truncate(idx_n=idx_n, A_truncate=system.plastic().size)
        else:
            niter = system.minimise()

        inc += 1
        storage.dset_extend1d(file, "/stored", inc, inc)
        storage.dset_extend1d(file, "/t", inc, system.t())
        storage.dset_extend1d(file, "/kick", inc, True)
        storage.dset_extend1d(file, "/trigger/branched", inc, False)
        storage.dset_extend1d(file, "/trigger/truncated", inc, niter == 0)
        file[f"/disp/{inc:d}"] = system.u()

        # in case that the event was truncated at a given "A":
        # store state from which a restart from the moment of truncation is possible
        if niter == 0:
            file["/restart/u"] = system.u()
            file["/restart/v"] = system.v()
            file["/restart/a"] = system.a()
            file["/restart/t"] = system.t()

        pbar.n = niter
        pbar.refresh()

    return args.file


def cli_ensembleinfo(cli_args=None):
    """
    Read and store basic info from individual pushes.
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
    output = file_defaults[funcname]

    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    ret = dict(
        S=[],
        A=[],
        xi=[],
        epsd=[],
        epsd0=[],
        sigd=[],
        sigd0=[],
        duration=[],
        truncated=[],
        element=[],
        version=[],
        dependencies=[],
        file=[],
        inc=[],
        incc=[],
        stress=[],
    )

    for i, filepath in enumerate(tqdm.tqdm(args.files)):

        with h5py.File(filepath, "r") as file:

            if i == 0:
                system = System.init(file)
            else:
                system.reset_epsy(System.read_epsy(file))

            out = System.basic_output(system, file, verbose=False)
            assert len(out["S"]) == 2
            assert file["/trigger/branched"][0]
            assert not file["/trigger/branched"][1]

            meta = file[f"/meta/{entry_points['cli_run']}"]
            branch = file["/meta/branch_fixed_stress"]

            ret["S"].append(out["S"][1])
            ret["A"].append(out["A"][1])
            ret["xi"].append(out["xi"][1])
            ret["epsd"].append(out["epsd"][1])
            ret["epsd0"].append(out["epsd"][0])
            ret["sigd"].append(out["epsd"][1])
            ret["sigd0"].append(out["epsd"][0])
            ret["duration"].append(out["duration"][1])
            ret["truncated"].append(file["/trigger/truncated"][1])
            ret["element"].append(file["/trigger/element"][1])
            ret["version"].append(meta.attrs["version"])
            ret["dependencies"].append(";".join(meta.attrs["dependencies"]))
            ret["file"].append(branch.attrs["file"])
            ret["inc"].append(branch.attrs["inc"] if "inc" in branch.attrs else int(-1))
            ret["incc"].append(branch.attrs["incc"] if "incc" in branch.attrs else int(-1))
            ret["stress"].append(branch.attrs["stress"] if "stress" in branch.attrs else int(-1))

    with h5py.File(args.output, "w") as output:

        for key in ["file", "version"]:
            tools.h5py_save_unique(data=ret.pop(key), file=output, path=f"/{key}", asstr=True)

        for key in ["dependencies"]:
            tools.h5py_save_unique(data=ret.pop(key), file=output, path=f"/{key}", split=";")

        for key in ret:
            output[key] = ret[key]

        System.create_check_meta(output, f"/meta/{progname}", dev=args.develop)


def restore_from_ensembleinfo(
    ensembleinfo: h5py.File, index: int, destpath: str, sourcedir: str = None, dev: bool = False
):
    """
    Restore the begin state of a specific push.

    :param ensembleinfo: Opened Trigger-EnsembleInfo, see :py:func:`cli_ensembleinfo`.
    :param index: Item from ``ensembleinfo``.
    :param destpath: Path where to write restored state.
    :param dev: Allow uncommitted changes.
    """

    sourcepath = tools.h5py_read_unique(ensembleinfo, "file", asstr=True)[index]

    if sourcedir is not None:
        sourcepath = os.path.join(sourcedir, sourcepath)
    elif not os.path.isfile(sourcepath):
        sourcedir = os.path.dirname(ensembleinfo.filename)
        sourcepath = os.path.join(sourcedir, sourcepath)

    assert os.path.isfile(sourcepath)

    with h5py.File(sourcepath, "r") as source, h5py.File(destpath, "w") as dest:

        System.branch_fixed_stress(
            source=source,
            dest=dest,
            inc=ensembleinfo["inc"][index],
            incc=ensembleinfo["incc"][index],
            stress=ensembleinfo["stress"][index],
            normalised=True,
            system=System.init(source),
            initialise=True,
            dev=dev,
        )

        _writeinitbranch(dest, ensembleinfo["element"][index])
        storage.dset_extend1d(dest, "/t", 1, ensembleinfo["duration"][index])


def _writeinitbranch(dest: h5py.File, element: int):
    """
    Write :py:mod:`Trigger` specific fields.
    """

    storage.create_extendible(
        dest,
        "/trigger/element",
        np.uint64,
        desc="Plastic element to trigger",
    )

    storage.create_extendible(
        dest,
        "/trigger/truncated",
        bool,
        desc="Flag if run was truncated before equilibrium",
    )

    storage.create_extendible(
        dest,
        "/trigger/branched",
        bool,
        desc="Flag if configuration followed from a branch",
    )

    storage.dset_extend1d(dest, "/trigger/element", 0, element)
    storage.dset_extend1d(dest, "/trigger/truncated", 0, False)
    storage.dset_extend1d(dest, "/trigger/branched", 0, True)


def _write_job(ret: dict, basename: str, **kwargs):
    """
    Write jobs:
    *   Branch at a given increment or fixed stress.
    *   Write slurm scripts.
    """

    if kwargs["nmax"] is not None:
        n = kwargs["nmax"]
        for key in ret:
            ret[key] = ret[key][:n]

    if not kwargs["force"]:
        if any([os.path.isfile(i) for i in ret["dest"]]):
            if not click.confirm("Overwrite output files?"):
                raise OSError("Cancelled")

    for i in tqdm.tqdm(range(len(ret["command"]))):

        s = ret["source"][i]
        d = os.path.join(kwargs["outdir"], ret["dest"][i])

        with h5py.File(s, "r") as source, h5py.File(d, "w") as dest:

            if i == 0:
                system = System.init(source)
                initialise = False
            else:
                initialise = ret["source"][i] != ret["source"][i - 1]

            System.branch_fixed_stress(
                source=source,
                dest=dest,
                inc=ret["inc"][i],
                incc=ret["incc"][i],
                stress=ret["stress"][i],
                normalised=True,
                system=system,
                initialise=initialise,
                dev=kwargs["develop"],
            )

            _writeinitbranch(dest, ret["element"][i])

    slurm.serial_group(
        ret["command"],
        basename=basename,
        group=kwargs["group"],
        outdir=kwargs["outdir"],
        conda=dict(condabase=kwargs["conda"]),
        sbatch={"time": kwargs["time"]},
    )

    return ret


def cli_job_deltasigma(cli_args=None):
    """
    Create jobs to trigger at fixed stress increase ``delta_sigma``
    after the last system-spanning event:
    ``stress[i] = sigma_c[i] + j * delta_sigma`` with ``j = 0, 1, ...``.
    The highest stress is thereby always lower than that where the next system spanning event.
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
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-d", "--delta-sigma", type=float, required=True, help="delta_sigma")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#simulations to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=3, help="#elements per configuration")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)
    assert os.path.isdir(args.outdir)

    basedir = os.path.dirname(args.ensembleinfo)
    executable = entry_points["cli_run"]

    with h5py.File(args.ensembleinfo, "r") as file:

        files = [os.path.join(basedir, f) for f in file["/files"].asstr()[...]]
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        inc = file["/avalanche/inc"][...]
        sigd = file["/avalanche/sigd"][...]
        ifile = file["/avalanche/file"][...]
        inc_loading = file["/loading/inc"][...]
        sigd_loading = file["/loading/sigd"][...]
        ifile_loading = file["/loading/file"][...]

    keep = A == N
    inc = inc[keep]
    sigd = sigd[keep]
    ifile = ifile[keep]
    inc_loading = inc_loading[keep]
    sigd_loading = sigd_loading[keep]
    ifile_loading = ifile_loading[keep]
    assert all(inc - 1 == inc_loading)
    elements = np.linspace(0, N + 1, args.pushes + 1)[:-1].astype(int)

    ret = dict(
        command=[],
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
        element=[],
    )

    basecommand = [executable]
    if args.truncate_system_spanning:
        basecommand += ["--truncate-system-spanning"]

    for i in range(sigd.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        simid = os.path.basename(os.path.splitext(filepath)[0])
        assert sigd_loading[i + 1] > sigd[i]
        stress = sigd[i] + args.delta_sigma * np.arange(100, dtype=float)
        stress = stress[stress < sigd_loading[i + 1]]

        for istress, s in enumerate(stress):

            if istress == 0:
                j = inc[i]  # directly after system-spanning events
            else:
                j = None  # at fixed stress

            for e in elements:
                bse = f"deltasigma={args.delta_sigma:.3f}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}_istep={istress:02d}.h5"
                ret["command"].append(" ".join(basecommand + [f"{out:s}"]))
                ret["source"].append(filepath)
                ret["dest"].append(out)
                ret["inc"].append(j)
                ret["incc"].append(inc[i])
                ret["stress"].append(s)
                ret["element"].append(e)

    ret = _write_job(ret, executable, **vars(args))

    if cli_args is not None:
        return ret


def cli_job_strain(cli_args=None):
    """
    Create jobs to trigger at fixed intervals between system-spanning events.
    The stress interval between two system spanning events is
    ``delta_sigma_i = (sigma_n[i] - sigma_c[i]) / (steps + 1)``
    with triggers at ``j * delta_sigma_i`` with ``j = 0, 1, ..., steps``.
    This implies that there is no push that coincides with the next system-spanning event.
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
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#simulations to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=3, help="#elements per configuration")
    parser.add_argument("-s", "--steps", type=int, default=10, help="#pushes between ss-events")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)
    assert os.path.isdir(args.outdir)

    basedir = os.path.dirname(args.ensembleinfo)
    executable = entry_points["cli_run"]

    with h5py.File(args.ensembleinfo, "r") as file:

        files = [os.path.join(basedir, f) for f in file["/files"].asstr()[...]]
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        inc = file["/avalanche/inc"][...]
        sigd = file["/avalanche/sigd"][...]
        ifile = file["/avalanche/file"][...]
        inc_loading = file["/loading/inc"][...]
        sigd_loading = file["/loading/sigd"][...]
        ifile_loading = file["/loading/file"][...]

    keep = A == N
    inc = inc[keep]
    sigd = sigd[keep]
    ifile = ifile[keep]
    inc_loading = inc_loading[keep]
    sigd_loading = sigd_loading[keep]
    ifile_loading = ifile_loading[keep]
    assert all(inc - 1 == inc_loading)
    elements = np.linspace(0, N + 1, args.pushes + 1)[:-1].astype(int)

    ret = dict(
        command=[],
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
        element=[],
    )

    basecommand = [executable]
    if args.truncate_system_spanning:
        basecommand += ["--truncate-system-spanning"]

    for i in range(sigd.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        simid = os.path.basename(os.path.splitext(filepath)[0])
        assert sigd_loading[i + 1] > sigd[i]
        stress = np.linspace(sigd[i], sigd_loading[i + 1], args.steps + 1)[:-1]

        for istress, s in enumerate(stress):

            if istress == 0:
                j = inc[i]  # directly after system-spanning events
            else:
                j = None  # at fixed stress

            for e in elements:
                bse = f"strainsteps={args.steps:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}_istep={istress:02d}.h5"
                ret["command"].append(" ".join(basecommand + [f"{out:s}"]))
                ret["source"].append(filepath)
                ret["dest"].append(out)
                ret["inc"].append(j)
                ret["incc"].append(inc[i])
                ret["stress"].append(s)
                ret["element"].append(e)

    ret = _write_job(ret, executable, **vars(args))

    if cli_args is not None:
        return ret
