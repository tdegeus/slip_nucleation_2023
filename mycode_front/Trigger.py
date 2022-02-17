"""
Take the system to a certain state and trigger an event.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import click
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
    parser.add_argument("--element", type=int, required=True, help="Plastic element to push")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("--silent", action="store_true", help="No screen output")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)

    pbar = tqdm.tqdm(total=1, disable=args.silent)
    pbar.set_description(args.file)

    with h5py.File(args.file, "a") as file:

        inc = int(file["/stored"][-1])
        system = System.init(file)
        meta = System.create_check_meta(file, f"/meta/{progname}/{inc + 1:d}", dev=args.develop)

        deps = file["/run/epsd/kick"][...]
        System._restore_inc(file, system, inc)
        idx_n = system.plastic_CurrentIndex()[:, 0]

        system.triggerElementWithLocalSimpleShear(deps, args.element)

        if args.truncate_system_spanning:
            niter = system.minimise_truncate(idx_n=idx_n, A_truncate=system.plastic().size)
        else:
            niter = system.minimise()

        inc += 1
        storage.dset_extend1d(file, "/stored", inc, inc)
        storage.dset_extend1d(file, "/t", inc, system.t())
        storage.dset_extend1d(file, "/kick", inc, True)
        file[f"/disp/{inc:d}"] = system.u()

        # in case that the event was truncated at a given "A":
        # store state from which a restart from the moment of truncation is possible
        if niter == 0:
            file["/restart/u"] = system.u()
            file["/restart/v"] = system.v()
            file["/restart/a"] = system.a()
            file["/restart/t"] = system.t()

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        idx_n = idx_n.astype(int)

        if not args.silent:
            pbar.n = niter
            pbar.refresh()

        meta.attrs["truncated"] = niter == 0
        meta.attrs["niter"] = niter
        meta.attrs["element"] = args.element
        meta.attrs["S"] = np.sum(idx - idx_n)
        meta.attrs["A"] = np.sum(idx != idx_n)

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

            meta = file[f"/meta/{entry_points['cli_run']}/1"]
            assert out["S"][1] == meta.attrs["S"]
            assert out["A"][1] == meta.attrs["A"]

            branch = file["/meta/branch_fixed_stress"]

            ret["S"].append(out["S"][1])
            ret["A"].append(out["A"][1])
            ret["xi"].append(out["xi"][1])
            ret["epsd"].append(out["epsd"][1])
            ret["epsd0"].append(out["epsd"][0])
            ret["sigd"].append(out["epsd"][1])
            ret["sigd0"].append(out["epsd"][0])
            ret["truncated"].append(meta.attrs["truncated"])
            ret["element"].append(meta.attrs["element"])
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


def _write(ret: dict, basename: str, **kwargs):
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
    Create jobs to trigger at fixed stress increase after the last system-spanning event.
    Thereby: ``delta_sigma = (sigma_top - sigma_bottom) / (pushes - 1)``, with pushes at
    ``sigma_c[i] +  * delta_sigma`` with ``j = 0, 1, ...``
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

    h = "#elements triggered per configuration"
    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--pushes-per-config", type=int, default=3, help=h)
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#pushes to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=10, help="#pushes")
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
        sig_bot = file["/averages/sigd_bottom"][...]
        sig_top = file["/averages/sigd_top"][...]
        delta_sig = (sig_top - sig_bot) / (args.pushes - 1)

    keep = A == N
    inc = inc[keep]
    sigd = sigd[keep]
    ifile = ifile[keep]
    inc_loading = inc_loading[keep]
    sigd_loading = sigd_loading[keep]
    ifile_loading = ifile_loading[keep]
    assert all(inc - 1 == inc_loading)
    elements = np.linspace(0, N + 1, args.pushes_per_config + 1)[:-1].astype(int)

    ret = dict(
        command=[],
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
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
        stress = sigd[i] + delta_sig * np.arange(100, dtype=float)
        stress = stress[stress < sigd_loading[i + 1]]

        for istress, s in enumerate(stress):

            if istress == 0:
                j = inc[i] # directly after system-spanning events
            else:
                j = None # at fixed stress

            for e in elements:
                bse = f"deltasigmapushes={args.pushes:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}_ipush={istress:02d}.h5"
                ret["command"].append(" ".join(basecommand + [f"--element {e:d}", f"{out:s}"]))
                ret["source"].append(filepath)
                ret["dest"].append(out)
                ret["inc"].append(j)
                ret["incc"].append(inc[i])
                ret["stress"].append(s)

    ret = _write(ret, executable, **vars(args))

    if cli_args is not None:
        return ret


def cli_job_strain(cli_args=None):
    """
    Create jobs to trigger at fixed intervals between system-spanning events.
    The stress interval between two system spanning events is
    ``delta_sigma_i = (sigma_n[i] - sigma_c[i]) / (pushes + 1)``
    with pushes at ``j * delta_sigma_i`` with ``j = 0, 1, ..., pushes``.
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

    h = "#elements triggered per configuration"
    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--pushes-per-config", type=int, default=3, help=h)
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#pushes to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=10, help="#pushes")
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
    elements = np.linspace(0, N + 1, args.pushes_per_config + 1)[:-1].astype(int)

    ret = dict(
        command=[],
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
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
        stress = np.linspace(sigd[i], sigd_loading[i + 1], args.pushes + 1)[:-1]

        for istress, s in enumerate(stress):

            if istress == 0:
                j = inc[i] # directly after system-spanning events
            else:
                j = None # at fixed stress

            for e in elements:
                bse = f"strainpushes={args.pushes:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}_ipush={istress:02d}.h5"
                ret["command"].append(" ".join(basecommand + [f"--element {e:d}", f"{out:s}"]))
                ret["source"].append(filepath)
                ret["dest"].append(out)
                ret["inc"].append(j)
                ret["incc"].append(inc[i])
                ret["stress"].append(s)

    ret = _write(ret, executable, **vars(args))

    if cli_args is not None:
        return ret
