"""
Take the system to a certain state and trigger an event.
"""
from __future__ import annotations

import argparse
import inspect
import os
import textwrap

import click
import FrictionQPotFEM.UniformSingleLayer2d as model
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import slurm
from . import System
from . import tag
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
    Trigger event. The protocol is as follows:

    1.  Restore system at a given increment (using `--inc``) or
        at fixed stress after a system-spanning event (using ``--incc`` and ``--stress``).
    2.  Store the displacement field (`/disp/0``).
    3.  Push a specific element and minimise energy.
    4.  Store the new displacement field (`/disp/1``).

    An option is provided to truncate the simulation for example when an event is system-spanning.
    In that case the ``truncated`` meta-attribute will be ``True``,
    The displacement field will not correspond to a mechanical equilibrium and the state of
    the system will be stored under ``restart``.
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
    parser.add_argument("--inc", type=int, help="Trigger at specific increment")
    parser.add_argument("--incc", type=int, required=True, help="Last system-spanning event")
    parser.add_argument("--stress", type=float, required=True, help="Trigger stress (real units)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-i", "--input", type=str, help="Simulation (read-only)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file (overwritten)")
    parser.add_argument("-v", "--version", action="version", version=version)

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.input)
    assert os.path.realpath(args.input) != os.path.realpath(args.output)
    assert args.develop or not tag.has_uncommitted(version)
    tools._check_overwrite_file(args.output, args.force)

    print("starting:", args.output)

    with h5py.File(args.output, "w") as output:

        meta = output.create_group(f"/meta/{progname}")

        # (*) Initialise system

        with h5py.File(args.input, "r") as file:

            system = System.init(file)
            eps_kick = file["/run/epsd/kick"][...]

            copy = ["/run/dt"]
            if "uuid" in file:
                copy += ["/uuid"]
            else:
                copy += ["/meta/seed_base", "/meta/normalisation"]
            g5.copy(file, output, copy)

            m = "/meta/{:s}".format(System.entry_points["cli_run"])
            if "uuid" in file[m].attrs:
                meta["file_uuid"] = file[m].attrs["uuid"]

            if args.inc is not None:

                system.setU(file[f"/disp/{args.inc:d}"])

                d = System.basic_output(system, file, verbose=False)
                if args.stress:
                    assert np.isclose(args.stress, d["sigd"][args.inc] * d["sig0"])
                if args.incc:
                    i = d["inc"][d["A"] == d["N"]]
                    assert i[i >= args.incc][1] > args.inc

            else:

                # determine at which increment a push could be applied

                inc_system, inc_push = System.pushincrements(system, file, args.stress)

                # reload specific increment based on target stress and system-spanning increment

                assert args.incc in inc_system
                i = np.argmax(np.logical_and(args.incc == inc_system, args.incc <= inc_push))
                inc = inc_push[i]
                assert args.incc == inc_system[i]

                system.setU(file[f"/disp/{inc:d}"])
                idx_n = system.plastic_CurrentIndex()
                system.addSimpleShearToFixedStress(args.stress)
                idx = system.plastic_CurrentIndex()
                assert np.all(idx == idx_n)

        # (*) Apply push and minimise energy

        output["/disp/0"] = system.u()
        idx_n = system.plastic_CurrentIndex()[:, 0]

        system.triggerElementWithLocalSimpleShear(eps_kick, args.element)

        if args.truncate_system_spanning:
            niter = system.minimise_truncate(idx_n=idx_n, A_truncate=system.plastic().size)
        else:
            niter = system.minimise()

        output["/disp/1"] = system.u()
        output["/stored"] = [0, 1]
        output["/kick"] = [False, True]

        # in case that the event was truncated at a given "A":
        # store state from which a restart from the moment of truncation is possible
        if niter == 0:
            output["/restart/u"] = system.u()
            output["/restart/v"] = system.v()
            output["/restart/a"] = system.a()
            output["/restart/t"] = system.t()

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        idx_n = idx_n.astype(int)

        print("done:", args.output, ", niter = ", niter)

        meta.attrs["file"] = os.path.basename(args.input)
        meta.attrs["filepath"] = args.input
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = System.dependencies(model)
        meta.attrs["truncated"] = niter == 0
        meta.attrs["target_stress"] = args.stress
        meta.attrs["target_inc_system"] = args.incc
        meta.attrs["target_element"] = args.element
        if args.inc is not None:
            meta.attrs["target_inc"] = args.inc
        meta.attrs["niter"] = niter
        meta.attrs["S"] = np.sum(idx - idx_n)
        meta.attrs["A"] = np.sum(idx != idx_n)

    if cli_args is not None:
        return args.output


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

    h = "Basic EnsembleInfo (not read, only its path is used to find where source files are stored)"
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-e", "--ensembleinfo", type=str, help=h)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    assert args.develop or not tag.has_uncommitted(version)
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
        inc_c=[],
        file=[],
        version=[],
        dependencies=[],
    )

    for i, filepath in enumerate(tqdm.tqdm(args.files)):

        with h5py.File(filepath, "r") as file:

            meta = file[f"/meta/{entry_points['cli_run']}"]
            sourcepath = meta.attrs["filepath"]

            if not os.path.isfile(sourcepath):
                assert args.ensembleinfo
                basedir = os.path.dirname(args.ensembleinfo)
                sourcepath = os.path.join(basedir, os.path.basename(sourcepath))
                assert os.path.isfile(sourcepath)

            with h5py.File(sourcepath, "r") as source:

                if i == 0:
                    system = System.init(source)
                else:
                    system.reset_epsy(System.read_epsy(source))

            out = System.basic_output(system, file, verbose=False)
            assert out["S"][1] == meta.attrs["S"]
            assert out["A"][1] == meta.attrs["A"]

            ret["S"].append(out["S"][1])
            ret["A"].append(out["A"][1])
            ret["xi"].append(out["xi"][1])
            ret["epsd"].append(out["epsd"][1])
            ret["epsd0"].append(out["epsd"][0])
            ret["sigd"].append(out["epsd"][1])
            ret["sigd0"].append(out["epsd"][0])
            ret["truncated"].append(meta.attrs["truncated"])
            ret["element"].append(meta.attrs["target_element"])
            ret["inc_c"].append(meta.attrs["target_inc_system"])
            ret["file"].append(meta.attrs["file"])
            ret["version"].append(meta.attrs["version"])
            ret["dependencies"].append(";".join(meta.attrs["dependencies"]))

    with h5py.File(args.output, "w") as output:

        for key in ["file", "version", "dependencies"]:
            value, index = np.unique(ret.pop(key), return_inverse=True)
            if key == "dependencies":
                value = [str(i).split(";") for i in value]
            else:
                value = [str(i) for i in value]

            output[f"/{key}/index"] = index
            output[f"/{key}/value"] = value

        for key in ret:
            output[key] = ret[key]

        meta = output.create_group(f"/meta/{progname}")
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = System.dependencies(model)


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
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("--pushes-per-config", type=int, default=3, help=h)
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
        sig0 = file["/normalisation/sig0"][...]
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

    commands = []
    outfiles = []
    basecommand = [executable]
    meta_sigma = []
    meta_deltasigma = []

    if args.truncate_system_spanning:
        basecommand += ["--truncate-system-spanning"]

    for i in range(sigd.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        simid = os.path.basename(os.path.splitext(filepath)[0])
        assert sigd_loading[i + 1] > sigd[i]
        stress = sigd[i] + delta_sig * np.arange(100, dtype=float)
        stress = stress[stress < sigd_loading[i + 1]] * sig0
        meta_sigma += (stress / sig0).tolist()
        meta_deltasigma += ((stress - sigd[i]) / sig0).tolist()

        # directly after system-spanning events
        for e in elements:
            bse = f"deltasigma=00_pushes={args.pushes:02d}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
            cmd = " ".join(
                basecommand
                + [
                    "-f",
                    f"--inc {inc[i]:d}",
                    f"--incc {inc[i]:d}",
                    f"--element {e:d}",
                    f"--stress {stress[0]:.8e}",
                    f"--input {filepath:s}",
                    f"--output {out:s}",
                ]
            )
            commands.append(cmd)
            outfiles.append(out)

        # intermediate stresses
        for istress, s in enumerate(stress[1:]):
            for e in elements:
                bse = f"deltasigma={istress + 1:02d}_pushes={args.pushes:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
                cmd = " ".join(
                    basecommand
                    + [
                        "-f",
                        f"--incc {inc[i]:d}",
                        f"--element {e:d}",
                        f"--stress {s:.8e}",
                        f"--input {filepath:s}",
                        f"--output {out:s}",
                    ]
                )
                commands.append(cmd)
                outfiles.append(out)

    if not args.force:
        if any([os.path.isfile(i) for i in outfiles]):
            if not click.confirm("Overwrite output files?"):
                raise OSError("Cancelled")

    if cli_args is not None:
        return dict(
            commands=[i.replace("--output ", f"--output {args.outdir}/") for i in commands],
            sigma=np.array(meta_sigma),
            delta_sigma=np.array(meta_deltasigma),
        )

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )


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
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("--pushes-per-config", type=int, default=3, help=h)
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
        sig0 = file["/normalisation/sig0"][...]
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

    commands = []
    outfiles = []
    basecommand = [executable]
    meta_sigma = []

    if args.truncate_system_spanning:
        basecommand += ["--truncate-system-spanning"]

    for i in range(sigd.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        simid = os.path.basename(os.path.splitext(filepath)[0])
        assert sigd_loading[i + 1] > sigd[i]
        stress = np.linspace(sigd[i], sigd_loading[i + 1], args.pushes + 1) * sig0
        meta_sigma += (stress[:-1] / sig0).tolist()

        # directly after system-spanning events
        for e in elements:
            bse = f"strain=00_pushes={args.pushes:02d}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
            cmd = " ".join(
                basecommand
                + [
                    "-f",
                    f"--inc {inc[i]:d}",
                    f"--incc {inc[i]:d}",
                    f"--element {e:d}",
                    f"--stress {stress[0]:.8e}",
                    f"--input {filepath:s}",
                    f"--output {out:s}",
                ]
            )
            commands.append(cmd)
            outfiles.append(out)

        # intermediate stresses
        for istress, s in enumerate(stress[1:-1]):
            for e in elements:
                bse = f"strain={istress + 1:02d}_pushes={args.pushes:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
                cmd = " ".join(
                    basecommand
                    + [
                        "-f",
                        f"--incc {inc[i]:d}",
                        f"--element {e:d}",
                        f"--stress {s:.8e}",
                        f"--input {filepath:s}",
                        f"--output {out:s}",
                    ]
                )
                commands.append(cmd)
                outfiles.append(out)

    if not args.force:
        if any([os.path.isfile(i) for i in outfiles]):
            if not click.confirm("Overwrite output files?"):
                raise OSError("Cancelled")

    if cli_args is not None:
        return dict(
            commands=[i.replace("--output ", f"--output {args.outdir}/") for i in commands],
            sigma=np.array(meta_sigma),
        )

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )
