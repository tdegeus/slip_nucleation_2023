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
import h5py
import numpy as np

from . import slurm
from . import System
from . import tools
from ._version import version

entry_points = dict(
    cli_run="Trigger_run",  # silence formatter
    cli_job_strain="Trigger_JobStrain",  # silence formatter
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def cli_run(cli_args=None):
    """
    Run simulation. The protocol is as follows:

    1.  Restore system at a given increment.
    2.  Push a specific element and minimise energy.
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

    parser.add_argument("--element", type=int, required=True, help="Plastic element to push")
    parser.add_argument("--incc", type=int, required=True, help="Last system-spanning event")
    parser.add_argument("--stress", type=float, required=True, help="Trigger stress (real units)")
    parser.add_argument("--inc", type=int, help="Trigger at specific element")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-i", "--input", type=str, help="Simulation (read-only)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file (overwritten)")
    parser.add_argument("-v", "--version", action="version", version=version)

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.input)
    assert os.path.realpath(args.input) != os.path.realpath(args.output)
    tools._check_overwrite_file(args.output, args.force)

    seed = None
    uuid = None

    print("starting:", args.output)

    with h5py.File(args.input, "r") as file:

        system = System.init(file)
        eps_kick = file["/run/epsd/kick"][...]

        if "uuid" in file:
            uuid = str(file["/uuid"].asstr()[...])
        else:
            seed = file["/meta/seed_base"][...]

        if args.inc is not None:

            system.setU(file[f"/disp/{args.inc:d}"])

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

    with h5py.File(args.output, "w") as output:

        output["/disp/0"] = system.u()
        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

        system.triggerElementWithLocalSimpleShear(eps_kick, args.element)
        niter = system.minimise()

        output["/disp/1"] = system.u()
        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

        print("done:", args.output, ", niter = ", niter)

        meta = output.create_group(f"/meta/{progname}")
        meta.attrs["file"] = os.path.relpath(args.input, os.path.dirname(args.output))
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = System.dependencies(model)
        meta.attrs["target_stress"] = args.stress
        meta.attrs["target_inc_system"] = args.incc
        meta.attrs["target_element"] = args.element
        if args.inc is not None:
            meta.attrs["target_inc"] = args.inc
        meta.attrs["S"] = np.sum(idx - idx_n)
        meta.attrs["A"] = np.sum(idx != idx_n)

        if seed:
            meta.attrs["seed_base"] = seed
        else:
            meta.attrs["uuid"] = uuid


def cli_job_strain(cli_args=None):
    """
    Create jobs to run at fixed strain intervals between system-spanning events.
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

    parser.add_argument("-e", "--estep", type=int, default=365, help="Elements between triggers")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#pushes to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=11, help="#pushes between ss-events")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Basename")
    parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.info)
    assert os.path.isdir(args.outdir)

    basedir = os.path.dirname(args.info)
    executable = entry_points["cli_run"]

    with h5py.File(args.info, "r") as file:

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
    elements = args.estep * np.arange(100, dtype=int)
    elements = elements[elements < N]

    commands = []
    outfiles = []

    for i in range(sigd.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        simid = os.path.basename(os.path.splitext(filepath)[0])
        assert sigd_loading[i + 1] > sigd[i]
        stress = np.linspace(sigd[i], sigd_loading[i + 1], args.pushes) * sig0

        for istress, s in enumerate(stress[1:-1]):
            for e in elements:
                bse = f"strain={istress + 1:02d}d{args.pushes - 1:02d}"
                out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
                cmd = " ".join(
                    [
                        f"{executable}",
                        f"--incc {inc[i]:d}",
                        f"--element {e:d}",
                        f"--stress {s:.8e}",
                        f"--input {filepath:s}",
                        f"--output {out:s}",
                    ]
                )
                commands.append(cmd)
                outfiles.append(out)

        for e in elements:
            bse = f"strain=00d{args.pushes - 1:02d}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
            cmd = " ".join(
                [
                    f"{executable}",
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

        for e in elements:
            bse = f"strain={args.pushes - 1:02d}d{args.pushes - 1:02d}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={e:d}.h5"
            cmd = " ".join(
                [
                    f"{executable}",
                    f"--inc {inc[i + 1]:d}",
                    f"--incc {inc[i]:d}",
                    f"--element {e:d}",
                    f"--stress {stress[-1]:.8e}",
                    f"--input {filepath:s}",
                    f"--output {out:s}",
                ]
            )
            commands.append(cmd)
            outfiles.append(out)

    if any([os.path.isfile(i) for i in outfiles]):
        if not click.confirm("Overwrite output files?"):
            raise OSError("Cancelled")

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return commands
