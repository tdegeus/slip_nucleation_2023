"""
Take the system to a certain state and trigger an event.
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import logging
import os
import pathlib
import re
import shutil
import tempfile
import textwrap

import click
import FrictionQPotFEM  # noqa: F401
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm

from . import Dynamics
from . import EventMap
from . import QuasiStatic
from . import slurm
from . import storage
from . import tools
from ._version import version

entry_points = dict(
    cli_ensembleinfo="Trigger_EnsembleInfo",
    cli_ensemblepack="Trigger_EnsemblePack",
    cli_ensemblepack_merge="Trigger_EnsemblePackMerge",
    cli_job_deltasigma="Trigger_JobDeltaSigma",
    cli_job_rerun_dynamics="Trigger_JobRerunDynamics",
    cli_job_rerun_eventmap="Trigger_JobRerunEventMap",
    cli_run="Trigger_Run",
    cli_transform_deprecated_pack="Trigger_TransformDeprecatedEnsemblePack",
    cli_transform_deprecated_pack2="Trigger_TransformDeprecatedEnsemblePack2",
)

file_defaults = dict(
    cli_ensembleinfo="Trigger_EnsembleInfo.h5",
    cli_ensemblepack="Trigger_EnsemblePack.h5",
)


def replace_ep(doc: str) -> str:
    """
    Replace ``:py:func:`...``` with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(rf":py:func:`{ep:s}`", entry_points[ep])
    return doc


def interpret_filename(filename: str) -> dict:
    """
    Split filename in useful information.
    """

    part = re.split("_|/", os.path.splitext(filename)[0])
    info = {}

    for i in part:
        key, value = i.split("=")
        info[key] = value

    for key in ["id", "incc", "element", "istep"]:
        info[key] = int(info[key])

    for key in ["deltasigma"]:
        info[key] = float(info[key])

    return info


def cli_run(cli_args=None):
    """
    Trigger event and minimise energy.

    This function will run ``--niter`` time steps to see if the trigger lead to any plastic event.
    If not, recursively try the neighbouring element until a plastic event is found.
    Due to this feature, the time of the event may be overestimated for small events.
    To skip this functionality:
    -   Use ``--retry=0``.
    -   Specify `file["/Trigger/element"][-1] >= 0`
        (you are responsible to have something meaningful in `file["/Trigger/try_element"][-1]`).

    An option is provided to truncate the simulation when an event is system-spanning.
    In that case the ``truncated`` meta-attribute will be ``True``.
    The displacement field will not correspond to a mechanical equilibrium.
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
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument(
        "--truncate-system-spanning", action="store_true", help="Truncate as soon as A == N"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Exact rerun: run at least --niter iterations"
    )
    parser.add_argument(
        "-r", "--retry", type=int, default=50, help="Maximum number of elements to try"
    )
    parser.add_argument(
        "-t", "--niter", type=int, default=20000, help="#iterations to use to try element"
    )
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert args.retry >= 0
    assert args.niter > 0

    pbar = tqdm.tqdm(total=1, desc=args.file)

    with h5py.File(args.file, "a") as file:

        root = file["Trigger"]
        assert root["element"].size == root["try_element"].size
        assert root["kick"].size == root["element"].size - 1

        step = root["kick"].size - 1
        element = int(root["element"][step + 1])
        try_element = int(root["try_element"][step + 1])
        assert not root["truncated"][step], "Cannot run is last step was not minimised"

        meta = QuasiStatic.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        system = QuasiStatic.System(file)
        system.restore_quasistatic_step(root, step)
        idx_n = np.copy(system.plastic.i[:, 0])
        N = system.N
        search_element = True

        if element >= 0:
            try_element = element
            search_element = False

        assert try_element >= 0
        system.triggerElementWithLocalSimpleShear(file["/param/cusp/epsy/deps"][...], try_element)

        # search for the element to run

        if search_element:

            element = try_element

            for itry in range(args.retry + 1):

                assert itry <= args.retry, "Maximum number of tries reached"
                assert itry == 0 or element != try_element, "All elements tried"

                system.timeSteps(args.niter)

                if np.sum(np.not_equal(system.plastic.i[:, 0], idx_n)) > 0:
                    break

                if element + 1 >= N:
                    element -= N

                element += 1
                system.restore_quasistatic_step(root, step)
                system.triggerElementWithLocalSimpleShear(
                    file["/param/cusp/epsy/deps"][...], element
                )

            root["element"][step + 1] = element

        elif args.rerun:

            system.timeSteps(args.niter)

        # run the dynamics

        if args.truncate_system_spanning:
            ret = system.minimise_truncate(idx_n=idx_n, A_truncate=system.N)
            truncated = ret == 0
        else:
            ret = system.minimise()
            assert ret == 0, "Out-of-bounds"
            truncated = False

        # store the output, and prepare for the next push

        storage.dset_extend1d(root, "inc", step + 1, system.inc)
        storage.dset_extend1d(root, "kick", step + 1, True)
        storage.dset_extend1d(root, "branched", step + 1, False)
        storage.dset_extend1d(root, "truncated", step + 1, truncated)
        root["u"][str(step + 1)] = system.u

        if root["element"].size == step + 2:
            storage.dset_extend1d(root, "element", step + 2, -1)
            storage.dset_extend1d(root, "try_element", step + 2, try_element)

        meta.attrs["completed"] = 1
        pbar.n = 1
        pbar.refresh()

    return args.file


def cli_ensemblepack_merge(cli_args=None):
    """
    Merge files created by :py:func:`cli_ensemblepack`.
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument("files", nargs="*", type=str, help="Files to merge")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    tools._check_overwrite_file(args.output, args.force)

    with h5py.File(args.output, "w") as output:
        for ifile, filepath in enumerate(tqdm.tqdm(args.files)):
            with h5py.File(filepath) as file:
                if ifile == 0:
                    g5.copy(file, output, "/param")
                    if "event" not in output:
                        output.create_group("event")
                else:
                    assert g5.allequal(file, output, g5.getdatapaths(file, "/param"))

                for path in file["event"]:
                    if path in output["event"]:
                        assert g5.allequal(file, output, g5.getdatapaths(file, f"/event/{path}"))
                    else:
                        g5.copy(file, output, f"/event/{path}", expand_soft=False)


def cli_ensemblepack(cli_args=None):
    """
    Pack pushes into a single file with soft links.
    The individual pushes are listed as::

        /event/filename_of_trigger/...

    Thereby ``...`` houses all fields that were present in the source file.
    However, the data common to the ensemble are included as a soft link::

        /event/filename_of_trigger/param  ->  /param
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
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-a", "--append", action="store_true", help="Append output file")
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = tools._parse(parser, cli_args)
    assert len(args.files) > 0
    assert all([os.path.isfile(file) for file in args.files])
    if not args.append:
        tools._check_overwrite_file(args.output, args.force)

    fmt = "{:" + str(max(len(i) for i in args.files)) + "s}"
    pbar = tqdm.tqdm(args.files)
    pbar.set_description(fmt.format(""))

    if args.append:
        with h5py.File(args.output, "r") as output:
            for filepath in args.files:
                assert filepath not in output["event"]

    with h5py.File(args.output, "a" if args.append else "w") as output:

        for ifile, filepath in enumerate(pbar):

            pbar.set_description(fmt.format(filepath), refresh=True)

            with h5py.File(filepath, "r") as file:

                # copy/check global ensemble data
                if ifile == 0 and not args.append:
                    g5.copy(file, output, "/param")
                else:
                    assert g5.allequal(file, output, list(g5.getdatapaths(file, "/param")))

                # test that all data is copied/linked
                for path in file:
                    assert path in ["param", "realisation", "meta", "Trigger"]

                # skip non-completed runs
                if "completed" not in file[f"/meta/{entry_points['cli_run']}"].attrs:
                    continue
                if not file[f"/meta/{entry_points['cli_run']}"].attrs["completed"]:
                    continue

                # copy/link event data
                tid = os.path.basename(filepath)
                g5.copy(file, output, ["/realisation", "/meta", "/Trigger"], root=f"/event/{tid}")
                output[f"/event/{tid}/param"] = h5py.SoftLink("/param")


def cli_ensembleinfo(cli_args=None):
    """
    Read and store basic info from individual pushes.
    Can only be read from output of :py:func:`cli_ensemblepack`.

    .. warning::

        This function currently just extracts the first push.
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
    parser.add_argument("ensemblepack", type=str, help="File to read")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensemblepack)
    tools._check_overwrite_file(args.output, args.force)

    ret = dict(
        S=[],
        A=[],
        xi=[],
        eps=[],
        eps0=[],
        sig=[],
        sig0=[],
        sig_broken=[],
        sig_unbroken=[],
        delta_sig_broken=[],
        delta_sig_unbroken=[],
        duration=[],
        dinc=[],
        truncated=[],
        element=[],
        try_element=[],
        run_version=[],
        run_dependencies=[],
        source=[],
        file=[],
        step=[],
        inci=[],
        step_c=[],
        stress=[],
    )

    with h5py.File(args.ensemblepack, "r") as pack, h5py.File(args.output, "w") as output:

        files = [i for i in pack["event"]]
        assert len(files) > 0
        simid = [int(tools.read_parameters(i)["id"]) for i in files]
        index = np.argsort(simid)
        fmt = "{:" + str(max(len(i) for i in files)) + "s}"
        pbar = tqdm.tqdm(index)
        pbar.set_description(fmt.format(""))

        for i, idx in enumerate(pbar):

            filepath = files[idx]
            pbar.set_description(fmt.format(filepath), refresh=True)
            file = pack["event"][filepath]

            if i == 0:
                system = QuasiStatic.System(file)
                N = system.N
            elif simid[idx] != simid[index[i - 1]]:
                system.reset(file)

            out = QuasiStatic.basic_output(system, file, root="Trigger", verbose=False)
            assert len(out["S"]) >= 2
            assert file["Trigger"]["branched"][0]
            assert not file["Trigger"]["branched"][1]

            meta = file["meta"][entry_points["cli_run"]]
            branch = file["meta"]["branch_fixed_stress"]

            # see QuasiStatic.branch_fixed_stress
            for i in range(1, 999):
                if f"branch_fixed_stress_{i:d}" in file["meta"]:
                    branch = file["meta"][f"branch_fixed_stress_{i:d}"]
                else:
                    break

            if "try_element" in file["Trigger"]:
                try_element = file["Trigger"]["try_element"][1]
            else:
                try_element = int(os.path.basename(filepath).split("element=")[1].split("_")[0])

            ret["S"].append(out["S"][1])
            ret["A"].append(out["A"][1])
            ret["xi"].append(out["xi"][1])
            ret["eps"].append(out["eps"][1])
            ret["eps0"].append(out["eps"][0])
            ret["sig"].append(out["sig"][1])
            ret["sig0"].append(out["sig"][0])
            ret["sig_broken"].append(out["sig_broken"][1])
            ret["sig_unbroken"].append(out["sig_unbroken"][1])
            ret["delta_sig_broken"].append(out["delta_sig_broken"][1])
            ret["delta_sig_unbroken"].append(out["delta_sig_unbroken"][1])
            ret["duration"].append(out["duration"][1])
            ret["dinc"].append(out["dinc"][1])
            ret["truncated"].append(file["Trigger"]["truncated"][1])
            ret["element"].append(file["Trigger"]["element"][1])
            ret["try_element"].append(try_element)
            ret["run_version"].append(meta.attrs["version"])
            ret["run_dependencies"].append(";".join(meta.attrs["dependencies"]))
            ret["source"].append(os.path.basename(filepath))
            ret["file"].append(branch.attrs["file"])
            ret["step"].append(branch.attrs["step"] if "step" in branch.attrs else int(-1))
            ret["inci"].append(branch.attrs["inci"] if "inci" in branch.attrs else int(-1))
            ret["step_c"].append(branch.attrs["step_c"] if "step_c" in branch.attrs else int(-1))
            ret["stress"].append(branch.attrs["stress"] if "stress" in branch.attrs else int(-1))

        if "param" in pack:
            g5.copy(pack, output, "/param")

        for key in ["file", "run_version"]:
            tools.h5py_save_unique(data=ret.pop(key), file=output, path=f"/{key}", asstr=True)

        for key in ["run_dependencies"]:
            tools.h5py_save_unique(data=ret.pop(key), file=output, path=f"/{key}", split=";")

        for key in ret:
            output[key] = ret[key]

        QuasiStatic.create_check_meta(output, f"/meta/{progname}", dev=args.develop)
        output["/meta/normalisation/N"] = N


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
    step = ensembleinfo["step"][index]
    step_c = ensembleinfo["step_c"][index]
    element = ensembleinfo["element"][index]
    try_element = ensembleinfo["try_element"][index]
    dinc = ensembleinfo["dinc"][index]

    if sourcedir is not None:
        sourcepath = pathlib.Path(sourcedir) / sourcepath
    elif not os.path.isfile(sourcepath):
        sourcedir = pathlib.Path(ensembleinfo.filename).parent
        sourcepath = sourcedir / sourcepath

    if not os.path.isfile(sourcepath):
        raise OSError(f'File not found: "{sourcepath}"')

    with h5py.File(sourcepath, "r") as source, h5py.File(destpath, "w") as dest:

        QuasiStatic.branch_fixed_stress(
            source=source,
            dest=dest,
            root="Trigger",
            step=step if step > 0 else None,
            step_c=step_c,
            stress=ensembleinfo["stress"][index],
            normalised=True,
            system=QuasiStatic.System(source),
            dev=dev,
        )
        root = dest["Trigger"]
        storage.dset_extend1d(root, "inc", 1, root["inc"][0] + dinc)

        _writeinitbranch(dest, try_element=try_element, element=element, dev=dev)

    return destpath


def __job_rerun(file, sims, basename, executable, args):

    if not args.force:
        if any([os.path.isfile(i) for i in sims["replica"]]) or any(
            [os.path.isfile(i) for i in sims["output"]]
        ):
            if not click.confirm("Overwrite existing?"):
                raise OSError("Cancelled")

    for dirname in list({os.path.dirname(i) for i in sims["replica"]}):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    commands = []
    ret = []
    height = " "

    if hasattr(args, "height"):
        height = " " + " ".join(["--height " + str(h) for h in args.height]) + " "

    for i in tqdm.tqdm(range(len(sims["replica"]))):
        index = sims["index"][i]
        replica = sims["replica"][i]
        output = sims["output"][i]
        restore_from_ensembleinfo(file, index, replica, args.sourcedir, args.develop)
        o = os.path.relpath(output, args.outdir)
        r = os.path.relpath(replica, args.outdir)
        commands += [f"{executable} -o {o} --step 1" + height + f"{r}"]
        ret += [f"{executable} -o {output} --step 1" + height + f"{replica}"]

    slurm.serial_group(
        commands,
        basename=basename,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

    return ret


def cli_job_rerun_eventmap(cli_args=None):
    """
    Rerun to get an event-map for avalanches resulting from triggers after system spanning events.
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
    parser.add_argument("--group", type=int, default=5, help="#simulations to group")
    parser.add_argument("--time", type=str, default="24h", help="Walltime")
    parser.add_argument("--amin", type=int, default=20, help="Minimal avalanche size")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--nsim", type=int, help="Number of simulations")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("-s", "--sourcedir", type=str, default=".", help="Path to sim-dir.")
    parser.add_argument("ensembleinfo", type=str, help=f"{entry_points['cli_ensembleinfo']} (read)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    ret = []

    with h5py.File(args.ensembleinfo) as file:

        if "/meta/normalisation/N" not in file:
            A = file["A"][...]
            truncated = file["truncated"][...]
            N = np.unique(A[truncated])
            assert N.size == 1
            N = N[0]
        else:
            N = int(file["/meta/normalisation/N"][...])

        A = file["A"][...]
        step = file["step"][...]
        step_c = file["step_c"][...]
        element = file["element"][...]
        sid = [
            QuasiStatic.interpret_filename(i)["id"]
            for i in tools.h5py_read_unique(file, "file", asstr=True)
        ]

        select = np.argwhere((A > args.amin) * (A < N) * (step == step_c)).ravel()

        sims = dict(
            replica=[],
            output=[],
            index=[],
        )

        for index in select:
            base = f"id={sid[index]:d}_incc={step_c[index]:d}_element={element[index]:d}.h5"
            sims["replica"].append(os.path.join(args.outdir, "src", base))
            sims["output"].append(os.path.join(args.outdir, base))
            sims["index"].append(index)

        executable = EventMap.entry_points["cli_run"]
        ret = __job_rerun(file, sims, "TriggerEventMap_conda={conda:s}", executable, args)

    if cli_args is not None:
        return ret


def cli_job_rerun_dynamics(cli_args=None):
    """
    Create job to rerun and measure the dynamics of system spanning (A == N) events.
    Instead of :py:func:`Dynamics.cli_run` one of the following measurements can be performed:

    -    ``--eventmap``: use :py:func:`EventMap.cli_run`.
    -    ``--highfreq``: use :py:func:`Dyanmics.cli_run_highfrequency`.

    In addition, select ``--avalanche`` and ``--amin`` to run avalanches instead of system spanning
    events.

    Possible usage:

        :py:func:`cli_job_rerun_dynamics`
            --ibin=1
            --nsim=100
            --height=20
            -o ../Dynamics/bin=1/src
            -s ../Run
            Trigger_EnsembleInfo.h5
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

    # development mode
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--test", action="store_true", help="Test mode")

    # job settings
    parser.add_argument("--group", type=int, default=5, help="Job: #simulations to group")
    parser.add_argument("--time", type=str, default="24h", help="Job: walltime")
    parser.add_argument(
        "--conda",
        type=str,
        default=slurm.default_condabase,
        help="Job: basename of conda environement",
    )

    # ensemble definition
    parser.add_argument("--bins", type=int, default=7, help="Number of stress bins")
    parser.add_argument("--skip", type=int, default=0, help="Number bins to skip")
    parser.add_argument("--ibin", type=int, help="Choose specific bin")
    parser.add_argument(
        "-n",
        "--nsim",
        type=int,
        default=100,
        help="Number of simulations (first nsim in order of distance to target stress)",
    )
    parser.add_argument("--eventmap", action="store_true", help="Run to get event-map")
    parser.add_argument("--highfreq", action="store_true", help="Run high frequency measurement")
    parser.add_argument(
        "--height",
        type=float,
        action="append",
        help="Add element row(s), see " + Dynamics.entry_points["cli_run"],
    )
    parser.add_argument(
        "--avalanche", action="store_true", help="Run avalanches, not system spanning events"
    )
    parser.add_argument(
        "--amin",
        type=int,
        default=20,
        help="Minimal avalanche size, only used if --avalanche is selected",
    )

    # paths
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-o", "--outdir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "-s",
        "--sourcedir",
        type=str,
        default=".",
        help="Directory with quasi-static simulations on which pushes were based",
    )

    parser.add_argument(
        "ensembleinfo", type=str, help="Input, see " + entry_points["cli_ensembleinfo"]
    )

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)
    assert not args.eventmap or (args.eventmap and args.height is None)
    args.height = [] if args.height is None else args.height

    with h5py.File(args.ensembleinfo) as file:

        if "/meta/normalisation/N" not in file:
            A = file["A"][...]
            truncated = file["truncated"][...]
            N = np.unique(A[truncated])
            assert N.size == 1
            N = N[0]
        else:
            N = int(file["/meta/normalisation/N"][...])

        sigmastar = file["sig0"][...]
        A = file["A"][...]
        step_c = file["step_c"][...]
        element = file["element"][...]
        sid = [
            QuasiStatic.interpret_filename(i)["id"]
            for i in tools.h5py_read_unique(file, "file", asstr=True)
        ]

        bin_edges = np.linspace(np.min(sigmastar), np.max(sigmastar), args.bins + 1)
        bins = bin_edges.size - 1
        bin_index = np.digitize(sigmastar, bin_edges) - 1
        bin_index[bin_index == bins] = bins - 1

        sims = dict(
            replica=[],
            output=[],
            index=[],
        )

        for ibin in range(args.skip, args.bins):

            if args.ibin is not None:
                if ibin != args.ibin:
                    continue

            keep = bin_index == ibin
            if np.sum(keep) == 0:
                continue
            target = np.mean(sigmastar[keep])
            sorter = np.argsort(np.abs(sigmastar - target))
            if args.avalanche or args.test:
                sorter = sorter[np.logical_and(A[sorter] >= args.amin, A[sorter] < N)]
            elif not args.test:
                sorter = sorter[A[sorter] == N]
            for index in sorter[: args.nsim]:
                base = f"id={sid[index]:d}_incc={step_c[index]:d}_element={element[index]:d}.h5"
                sims["replica"].append(os.path.join(args.outdir, f"bin={ibin:02d}", "src", base))
                sims["output"].append(os.path.join(args.outdir, f"bin={ibin:02d}", base))
                sims["index"].append(index)

        executable = Dynamics.entry_points["cli_run"]

        if args.eventmap:
            executable = EventMap.entry_points["cli_run"]
        elif args.highfreq:
            executable = Dynamics.entry_points["cli_run_highfrequency"]

        ret = __job_rerun(file, sims, "TriggerDynamics_conda={conda:s}", executable, args)

    if cli_args is not None:
        return ret


def _writeinitbranch(
    file: h5py.File,
    try_element: int,
    element: int = -1,
    meta: tuple[str, dict] = None,
    dev: bool = False,
):
    """
    Write :py:mod:`Trigger` specific fields.

    :param try_element: Element to try trigger.
    :param element: Element to trigger, without trying others.
    :param meta: Extra metadata to write ``("/path/to/group", {"mykey": myval, ...})``.
    :param dev: If True, allow uncommited changes.
    """

    root = file["Trigger"]

    storage.create_extendible(
        root,
        "element",
        np.int64,
        desc="Plastic element triggered",
    )

    storage.create_extendible(
        root,
        "try_element",
        np.int64,
        desc="Plastic element to start trying triggering",
    )

    storage.create_extendible(
        root,
        "truncated",
        bool,
        desc="Flag if run was truncated before equilibrium",
    )

    storage.create_extendible(
        root,
        "branched",
        bool,
        desc="Flag if configuration followed from a branch",
    )

    storage.dset_extend1d(root, "element", 0, -1)
    storage.dset_extend1d(root, "element", 1, element)

    storage.dset_extend1d(root, "try_element", 0, -1)
    storage.dset_extend1d(root, "try_element", 1, try_element)

    storage.dset_extend1d(root, "truncated", 0, False)
    storage.dset_extend1d(root, "branched", 0, True)

    if meta is not None:
        g = QuasiStatic.create_check_meta(file, meta[0], dev=dev)
        for key in meta[1]:
            g.attrs[key] = meta[1][key]


def _write_configurations(
    try_element: int,
    info: h5py.File,
    force: bool = False,
    dev: bool = False,
    source: list[str] = None,
    dest: list[str] = None,
    step: list[int] = None,
    step_c: list[int] = None,
    stress: list[float] = None,
    meta: tuple[str, dict] = None,
):
    """
    Branch at a given increment or fixed stress.

    :param try_element: Element to try to trigger first.
    :param info: EnsembleInfo to read configuration data from.
    :param force: Force overwrite of existing files.
    :param dev: Allow uncommitted changes.
    :param source: List with source file-paths.
    :param dest: List with destination file-paths.
    :param step: List with fixed increment at which to push (entries can be ``None``).
    :param step_c: List with system spanning events after which to load (entries can be ``None``).
    :param stress: List with stress at which to load (entries can be ``None``).
    :param meta: Extra metadata to write ``("/path/to/group", {"mykey": myval, ...})``.
    """

    if len(dest) == 0:
        return

    if not force:
        if any([os.path.isfile(i) for i in dest]):
            if not click.confirm("Overwrite output files?"):
                raise OSError("Cancelled")

    for i in dest:
        if os.path.isfile(i):
            os.remove(i)

    fmt = "{:" + str(max(len(str(i)) for i in source)) + "s}"
    index = np.argsort(source)
    pbar = tqdm.tqdm(index)
    tmp = tempfile.mkstemp(suffix=".h5", dir=os.path.dirname(dest[0]))[1]

    for i, idx in enumerate(pbar):

        s = source[idx]
        d = dest[idx]
        pbar.set_description(fmt.format(str(s)), refresh=True)

        with h5py.File(s, "r") as source_file:

            if i == 0:
                system = QuasiStatic.System(source_file)
                output = {
                    "N": int(info["/normalisation/N"][...]),
                    "sig0": float(info["/normalisation/sig0"][...]),
                }
                init_system = True
            else:
                init_system = source[idx] != source[index[i - 1]]

            if init_system:
                p = f"/full/{os.path.split(s)[-1]:s}"
                output["kick"] = info[f"{p:s}/kick"][...]
                output["sig"] = info[f"{p:s}/sig"][...]
                output["A"] = info[f"{p:s}/A"][...]
                output["step"] = info[f"{p:s}/step"][...]
                with h5py.File(tmp, "w") as dest_file:
                    g5.copy(source_file, dest_file, ["/param", "/realisation", "/meta"])
                    QuasiStatic._init_run_state(
                        root=dest_file.create_group("Trigger"),
                        u=source_file["/QuasiStatic/u/0"][...],
                    )
                    _writeinitbranch(dest_file, try_element=try_element, meta=meta, dev=dev)

            shutil.copy(tmp, d)

            with h5py.File(d, "a") as dest_file:

                QuasiStatic.branch_fixed_stress(
                    source=source_file,
                    dest=dest_file,
                    root="Trigger",
                    step=step[idx],
                    step_c=step_c[idx],
                    stress=stress[idx],
                    normalised=True,
                    system=system,
                    init_system=init_system,
                    init_dest=False,
                    output=output,
                    dev=dev,
                )

    os.remove(tmp)


def _copy_configurations(try_element: int, source: list[str], dest: list[str], force: bool = False):
    """
    Copy configurations written by :py:func:`_write_configurations` and
    overwrite the triggered element.

    :param try_element: Element to try to trigger first.
    :param source: List with source file-paths.
    :param dest: List with destination file-paths.
    :param force: Force overwrite of existing files.
    """

    if not force:
        if any([os.path.isfile(i) for i in dest]):
            if not click.confirm("Overwrite output files?"):
                raise OSError("Cancelled")

    for s, d in zip(source, dest):

        shutil.copy(s, d)

        with h5py.File(d, "a") as dest:
            dest["/Trigger/element"][1] = -1
            dest["/Trigger/try_element"][1] = try_element


def __filter(ret, filepath, meta):
    """
    Filter already run simulations.
    """

    assert os.path.isfile(filepath)

    with h5py.File(filepath, "r") as file:
        # cli_ensembleinfo
        if "sig0" in file:
            raise OSError("Not yet implemented (not very hard though)")
        # cli_ensemblepack
        else:
            if g5.join("/ensemble", meta[0]) in file:
                m = file[g5.join("/ensemble", meta[0])]
                for key in meta[1]:
                    assert m.attrs[key] == meta[1][key]

            present = sorted(i for i in file["event"])
            ensemble = [os.path.basename(i) for i in ret["dest"]]
            keep = ~np.in1d(ensemble, present)
            for key in ret:
                ret[key] = list(itertools.compress(ret[key], keep))

    return ret


def cli_job_deltasigma(cli_args=None):
    """
    Create jobs to trigger at fixed stress increase ``delta_sigma``
    since the last system-spanning event:
    ``stress[i] = sigma_c[i] + j * delta_sigma`` with ``j = 0, 1, ...``.
    The highest stress is thereby always lower than that of the next system spanning event.
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
    parser.add_argument("--filter", type=str, help="Filter completed jobs")
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument(
        "--istress",
        type=int,
        action="append",
        help="Select only specific stress for the list of stresses",
    )
    parser.add_argument("-d", "--delta-sigma", type=float, required=True, help="delta_sigma")
    parser.add_argument("-e", "--element", type=int, action="append", help="Specify element(s)")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#simulations to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=3, help="#elements per configuration")
    parser.add_argument("-r", "--subdir", action="store_true", help="Separate in directories")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    basedir = pathlib.Path(args.ensembleinfo).parent
    executable = entry_points["cli_run"]

    with h5py.File(args.ensembleinfo, "r") as file:
        files = [basedir / f for f in file["/files"].asstr()[...]]
        N = file["/normalisation/N"][...]
        A = file["/avalanche/A"][...]
        step = file["/avalanche/step"][...]
        sig = file["/avalanche/sig"][...]
        ifile = file["/avalanche/file"][...]
        step_loading = file["/loading/step"][...]
        sig_loading = file["/loading/sig"][...]
        ifile_loading = file["/loading/file"][...]

    keep = A == N
    step = step[keep]
    sig = sig[keep]
    ifile = ifile[keep]
    step_loading = step_loading[keep]
    sig_loading = sig_loading[keep]
    ifile_loading = ifile_loading[keep]
    assert all(step - 1 == step_loading)
    assert args.delta_sigma > 0
    assert args.delta_sigma < np.max(sig_loading - sig)
    elements = np.linspace(0, N + 1, args.pushes + 1)[:-1].astype(int)

    if args.element:
        elements = args.element

    ret = dict(
        source=[],
        dest=[],
        step=[],
        step_c=[],
        stress=[],
    )

    basecommand = [executable]
    if args.truncate_system_spanning:
        basecommand += ["--truncate-system-spanning"]

    for i in range(sig.size - 1):

        if ifile[i] != ifile_loading[i + 1]:
            continue

        filepath = files[ifile[i]]
        sid = filepath.stem
        assert sig_loading[i + 1] > sig[i]
        stress = sig[i] + args.delta_sigma * np.arange(100, dtype=float)
        stress = stress[stress < sig_loading[i + 1]]

        if args.istress:
            stress = [stress[i] for i in args.istress]

        for istress, s in enumerate(stress):

            if istress == 0:
                j = step[i]  # directly after system-spanning events
            else:
                j = None  # at fixed stress

            if args.subdir:
                bse = f"{args.outdir}/{sid}/deltasigma={args.delta_sigma:.3f}"
                if not os.path.isdir(f"{args.outdir}/{sid}"):
                    os.makedirs(f"{args.outdir}/{sid}")
            else:
                bse = f"{args.outdir}/deltasigma={args.delta_sigma:.3f}"
            out = f"{bse}_{sid}_incc={step[i]:d}_element={elements[0]:d}_istep={istress:02d}.h5"
            ret["source"].append(filepath)
            ret["dest"].append(out)
            ret["step"].append(j)
            ret["step_c"].append(step[i])
            ret["stress"].append(s)

    # -----------
    # write files
    # -----------

    meta = {
        "deltasigma": args.delta_sigma,
        "pushes": args.pushes,
    }

    meta = (f"/meta/{progname}", meta)

    if args.nmax is not None:
        for key in ret:
            ret[key] = ret[key][: args.nmax]

    # when previously present data have to be filtered the trick of
    # generating for one element and then copying cannot be used
    # instead generate per element after filtering
    # (this could be made more clever, but it would take some more coding)
    if args.filter:

        data = ret.copy()
        e0 = elements[0]
        outfiles = []

        with h5py.File(args.ensembleinfo, "r") as file:
            for e in elements:
                r = data.copy()
                r["dest"] = [i.replace(f"_element={e0}_", f"_element={e:d}_") for i in r["dest"]]
                r = __filter(r, args.filter, meta)
                _write_configurations(e, file, args.force, args.develop, meta=meta, **r)
                outfiles += [i for i in r["dest"]]

    # if no filter is applied: generate for one element and copy + modify for all the other elements
    else:

        with h5py.File(args.ensembleinfo, "r") as file:
            _write_configurations(elements[0], file, args.force, args.develop, meta=meta, **ret)
            outfiles = [i for i in ret["dest"]]

        for e in elements[1:]:
            d = [i.replace(f"_element={elements[0]}_", f"_element={e:d}_") for i in ret["dest"]]
            _copy_configurations(e, ret["dest"], d, args.force)
            outfiles += d

    cmd = [executable]
    if args.truncate_system_spanning:
        cmd.append("--truncate-system-spanning")
    if args.develop:
        cmd.append("--develop")

    slurm.serial_group(
        [" ".join(cmd + [os.path.relpath(i, args.outdir)]) for i in outfiles],
        basename=executable,
        group=args.group,
        outdir=args.outdir,
        conda=dict(condabase=args.conda),
        sbatch={"time": args.time},
    )

    if cli_args is not None:
        return [" ".join(cmd + [i]) for i in outfiles]


def cli_transform_deprecated_pack(cli_args=None):
    """
    Transform old data structure to the current one.
    This code is considered 'non-maintained'.
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

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--log", default=entry_points[funcname] + ".log", help="Log file")
    parser.add_argument("source", type=str, help="Source (read only)")
    parser.add_argument("dest", type=str, help="Destination (overwritten)")

    args = tools._parse(parser, cli_args)

    logger = logging.getLogger("mylogger")
    logger.setLevel(logging.WARNING)
    handler = logging.FileHandler(args.log)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_file = os.path.join(temp_dir, "tmp.h5")
        temp_file = "mytmp.h5"

        with h5py.File(args.source) as src, h5py.File(args.dest, "w-") as dest:

            paths = list(g5.getdatasets(src, fold="event", fold_symbol=""))
            paths.remove("/event")

            if len(paths) > 0:
                logger.warning("Potentially not be copied:\n" + "\n".join(paths))

            pbar = tqdm.tqdm(src["event"])

            for ifile, name in enumerate(pbar):

                pbar.set_description(f"{name}")
                root = f"/event/{name}"
                dest.create_group(root)
                allow_nonempty = False
                fold = ["/meta/normalisation", "/trigger"]

                if ifile == 0:
                    paths = list(g5.getdatapaths(src, root=root, fold=fold, fold_symbol=""))
                    paths_bak = [i for i in paths]
                    root_bak = root

                try:
                    paths = list(g5.getdatapaths(src, root=root, fold=fold, fold_symbol=""))
                except:
                    paths = [g5.join(root, i.split(root_bak)[1]) for i in paths_bak]
                    logger.warning(f'Failed to read paths "{name}"')

                # copy/check parameters
                try:

                    with h5py.File(temp_file, "w") as tmp:

                        paths = QuasiStatic.transform_deprecated_param(
                            src, tmp, paths, source_root=root
                        )

                        for key in g5.getdatapaths(src, root=g5.join(root, "/meta/normalisation")):
                            g5.copy(src, tmp, key, key.split(root)[1].replace("/meta", "/param"))

                        paths.remove(g5.join(root, "/meta/normalisation"))
                        g5.copy(tmp, dest, "/realisation/seed", root=root)

                        if ifile == 0:
                            g5.copy(tmp, dest, "/param")
                        else:
                            check = list(g5.getdatapaths(tmp))
                            check.remove("/realisation/seed")
                            assert g5.allequal(tmp, dest, check)

                except:  # noqa: E722

                    if ifile > 0:
                        logger.warning(f'Failed to read parameters "{name}"')
                        allow_nonempty = True
                    else:
                        raise OSError("Cannot read parameters")

                # link parameters
                dest[g5.join(root, "param")] = h5py.SoftLink("/param")

                # basic trigger fields
                rename = {"/trigger": "/Trigger"}

                for key in rename:
                    if g5.join(root, key) not in src:
                        logger.warning(f'No trigger in "{name}"')
                        continue
                    g5.copy(src, dest, g5.join(root, key), g5.join(root, rename[key]))
                    paths.remove(g5.join(root, key))

                # fields that were previously not grouped under trigger
                rename = {f"/disp/{i}": f"/Trigger/u/{i}" for i in src[g5.join(root, "disp")]}
                rename["/kick"] = "/Trigger/kick"

                for key in rename:
                    g5.copy(src, dest, g5.join(root, key), g5.join(root, rename[key]))
                    paths.remove(g5.join(root, key))

                # optional metadata
                rename = {}
                rename["/meta/EnsembleInfo"] = "/meta/QuasiStatic_EnsembleInfo"
                rename["/meta/Run_generate"] = "/meta/QuasiStatic_Generate"
                rename["/meta/Run"] = "/meta/QuasiStatic_Run"
                rename["/meta/Trigger_run"] = "/meta/Trigger_Run"
                rename["/meta/branch_fixed_stress"] = "/meta/branch_fixed_stress"

                for key in rename:
                    if g5.join(root, key) not in src:
                        continue
                    g5.copy(src, dest, g5.join(root, key), g5.join(root, rename[key]))
                    paths.remove(g5.join(root, key))

                t = src[g5.join(root, "/t")][...]
                dt = src[g5.join(root, "/run/dt")][...]
                dest[g5.join(root, "/Trigger/inc")] = np.round(t / dt).astype(np.uint64)
                paths.remove(g5.join(root, "/t"))

                # adding meta data
                dest.create_group(g5.join(root, f"/meta/{progname}")).attrs["version"] = version

                # assertions
                assert "/param/normalisation" in dest

                n = dest[g5.join(root, "/Trigger/inc")].size
                assert np.all(src[g5.join(root, "stored")][...] == np.arange(n))

                paths.remove(g5.join(root, "/stored"))
                paths.remove(g5.join(root, "/disp"))

                if len(paths) > 0:
                    logger.warning("Potentially not be copied:\n" + "\n".join(paths))

                assert len(paths) == 0 or allow_nonempty


def cli_transform_deprecated_pack(cli_args=None):
    """
    Add seed
    This code is considered 'non-maintained'.
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

    parser.add_argument("pack", type=str, help="EnsemblePack")
    parser.add_argument("--source", type=str, help="Source dir")

    args = parser.parse_args(cli_args)
    sourcedir = pathlib.Path(args.source)
    assert sourcedir.is_dir()

    with h5py.File(args.pack) as file:

        for event in file["event"]:

            if f"/event/{event}/realisation/seed" in file:
                continue

            sid = interpret_filename(event)["id"]

            with h5py.File(sourcedir / f"{sid:03d}.h5") as src:
                print(src["/realisation/seed"][...])
                # file[f"/event/{event}/realisation/seed"] = src["/realisation/seed"][...]
