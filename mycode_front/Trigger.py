"""
Take the system to a certain state and trigger an event.
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import os
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

from . import slurm
from . import storage
from . import System
from . import tag
from . import tools
from ._version import version

entry_points = dict(
    cli_run="Trigger_run",
    cli_job_strain="Trigger_JobStrain",
    cli_job_deltasigma="Trigger_JobDeltaSigma",
    cli_ensembleinfo="Trigger_EnsembleInfo",
    cli_ensemblepack="Trigger_EnsemblePack",
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


def cli_run(cli_args=None):
    """
    Trigger event and minimise energy.

    This function will run ``--niter`` time steps to see if the trigger lead to any plastic event.
    If that is not the case, it will retry the neighbouring element recursively until a plastic
    event is found.
    Due to this feature, the time of the event may be overestimated for very small events.
    If ``--retry=0`` this functionality with be skipped.

    An option is provided to truncate the simulation when an event is system-spanning.
    In that case the ``truncated`` meta-attribute will be ``True``.
    The displacement field will not correspond to a mechanical equilibrium,
    while the state at truncation will be stored under ``restart``.
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
    parser.add_argument("-r", "--retry", type=int, default=50, help="Maximum number of tries")
    parser.add_argument("-t", "--niter", type=int, default=20000, help="Trial number of iterations")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Input/output file")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.file)
    assert args.retry >= 0
    assert args.niter > 0

    pbar = tqdm.tqdm(total=1, desc=args.file)

    with h5py.File(args.file, "a") as file:

        element = int(file["/trigger/element"][-1])
        inc = int(file["/stored"][-1])
        assert not file["/trigger/truncated"][inc]

        meta = System.create_check_meta(file, f"/meta/{progname}", dev=args.develop)
        system = System.init(file)
        System._restore_inc(file, system, inc)
        idx_n = system.plastic_CurrentIndex()[:, 0]
        N = system.plastic().size
        system.triggerElementWithLocalSimpleShear(file["/run/epsd/kick"][...], element)

        for trial in range(args.retry):

            system.timeSteps(args.niter)

            if np.sum(np.not_equal(system.plastic_CurrentIndex()[:, 0], idx_n)) > 0:
                break
            if element + 1 == N:
                break

            element += 1
            System._restore_inc(file, system, inc)
            system.triggerElementWithLocalSimpleShear(file["/run/epsd/kick"][...], element)

        if args.truncate_system_spanning:
            niter = system.minimise_truncate(idx_n=idx_n, A_truncate=system.plastic().size)
        else:
            niter = system.minimise()

        inc += 1
        storage.dset_extend1d(file, "/stored", inc, inc)
        storage.dset_extend1d(file, "/t", inc, system.t())
        storage.dset_extend1d(file, "/kick", inc, True)
        storage.dset_extend1d(file, "/trigger/element", inc, element)
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

        meta.attrs["completed"] = 1
        pbar.n = 1
        pbar.refresh()

    return args.file


def cli_ensemblepack(cli_args=None):
    """
    Pack pushes into a single file with soft links.
    The individual pushes are listed as::

        /event/filename_of_trigger/...

    Thereby ``...`` houses all fields that were present in the source file.
    However, for common data this is only a soft-link, to::

        /realisation/filename_of_realisation/...
        /source/...
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

    # categorise datasets
    restart = [
        "/restart/u",
        "/restart/v",
        "/restart/a",
        "/restart/t",
    ]
    # - copy/link on "/realisation/{sid}"
    realisation = [
        "/cusp/epsy/initstate",
        "/meta/seed_base",
    ]
    # - copy on "/event/{tid}"
    event_meta = [
        "/meta/branch_fixed_stress",
        f"/meta/{entry_points['cli_run']}",
        f"/meta/{System.entry_points['cli_run']}",
        f"/meta/{System.entry_points['cli_generate']}",
        "/trigger/branched",
        "/trigger/element",
        "/trigger/truncated",
    ]
    # - copy on "/event/{tid}"
    event_data = [
        "/disp",
        "/stored",
        "/t",
        "/kick",
    ]
    # - copy on "/ensemble"
    ensemble_meta = [
        f"/meta/{entry_points['cli_job_strain']}",
        f"/meta/{entry_points['cli_job_deltasigma']}",
    ]
    skip = realisation + event_meta + restart

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
                    datasets = System.clone(file, output, skip=skip, root="/source")
                    ensemble_meta = [i for i in ensemble_meta if i in file]
                    g5.copy(file, output, ensemble_meta, root="/ensemble")
                elif ifile == 0 and args.append:
                    datasets = System.clone(file, output, skip=skip, root="/source", dry_run=True)
                    ensemble_meta = [i for i in ensemble_meta if i in file]

                for path in datasets:
                    if not g5.equal(file, output, path, root="/source"):
                        print(path)

                assert g5.allequal(file, output, datasets, root="/source")
                assert g5.allequal(file, output, ensemble_meta, root="/ensemble")

                # test that all data is copied/linked
                present = g5.getdatapaths(file)
                present = list(itertools.filterfalse(re.compile("^/disp.*$").match, present))
                present = list(itertools.filterfalse(re.compile("^/restart.*$").match, present))
                copied = datasets + ensemble_meta + realisation + event_data + event_meta
                assert np.all(np.in1d(present, copied))

                sid = file["/meta/branch_fixed_stress"].attrs["file"]
                tid = os.path.basename(filepath)

                if f"/realisation/{sid}" not in output:
                    g5.copy(file, output, realisation, root=f"/realisation/{sid}")
                else:
                    g5.compare(
                        file,
                        output,
                        realisation,
                        [g5.join(f"/realisation/{sid}", i) for i in realisation],
                    )

                if "/disp/1" not in file:
                    continue
                if "/meta/Trigger_run" not in file:
                    continue
                if tag.greater_equal(file["/meta/Trigger_run"].attrs["version"], "7.4"):
                    if "completed" not in file["/meta/Trigger_run"].attrs:
                        continue

                g5.copy(file, output, event_data, root=f"/event/{tid}")
                g5.copy(file, output, event_meta, root=f"/event/{tid}", skip=True)

                for path in datasets:
                    output[g5.join(f"/event/{tid}", path)] = h5py.SoftLink(g5.join("/source", path))

                for path in realisation:
                    output[g5.join(f"/event/{tid}", path)] = h5py.SoftLink(
                        g5.join(f"/realisation/{sid}", path)
                    )


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
        epsd=[],
        epsd0=[],
        sigd=[],
        sigd0=[],
        duration=[],
        truncated=[],
        element=[],
        run_version=[],
        run_dependencies=[],
        file=[],
        inc=[],
        inci=[],
        incc=[],
        stress=[],
    )

    with h5py.File(args.ensemblepack, "r") as pack, h5py.File(args.output, "w") as output:

        files = [i for i in pack["event"]]
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
                system = System.init(file)
            elif simid[idx] != simid[index[i - 1]]:
                system.reset_epsy(System.read_epsy(file))

            out = System.basic_output(system, file, verbose=False)
            assert len(out["S"]) >= 2
            assert file["trigger"]["branched"][0]
            assert not file["trigger"]["branched"][1]

            meta = file["meta"][entry_points["cli_run"]]
            branch = file["meta"]["branch_fixed_stress"]

            ret["S"].append(out["S"][1])
            ret["A"].append(out["A"][1])
            ret["xi"].append(out["xi"][1])
            ret["epsd"].append(out["epsd"][1])
            ret["epsd0"].append(out["epsd"][0])
            ret["sigd"].append(out["epsd"][1])
            ret["sigd0"].append(out["epsd"][0])
            ret["duration"].append(out["duration"][1])
            ret["truncated"].append(file["trigger"]["truncated"][1])
            ret["element"].append(file["trigger"]["element"][1])
            ret["run_version"].append(meta.attrs["version"])
            ret["run_dependencies"].append(";".join(meta.attrs["dependencies"]))
            ret["file"].append(branch.attrs["file"])
            ret["inc"].append(branch.attrs["inc"] if "inc" in branch.attrs else int(-1))
            ret["inci"].append(branch.attrs["inci"] if "inci" in branch.attrs else int(-1))
            ret["incc"].append(branch.attrs["incc"] if "incc" in branch.attrs else int(-1))
            ret["stress"].append(branch.attrs["stress"] if "stress" in branch.attrs else int(-1))

        if "ensemble" in pack:
            g5.copy(pack, output, ["/ensemble"])

        for key in ["file", "run_version"]:
            tools.h5py_save_unique(data=ret.pop(key), file=output, path=f"/{key}", asstr=True)

        for key in ["run_dependencies"]:
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
            dev=dev,
        )

        _writeinitbranch(dest, ensembleinfo["element"][index])
        storage.dset_extend1d(dest, "/t", 1, ensembleinfo["duration"][index])


def _writeinitbranch(file: h5py.File, element: int, meta: tuple[str, dict] = None):
    """
    Write :py:mod:`Trigger` specific fields.

    :oaram element: Element to trigger.
    :param meta: Extra metadata to write ``("/path/to/group", {"mykey": myval, ...})``.
    """

    storage.create_extendible(
        file,
        "/trigger/element",
        np.uint64,
        desc="Plastic element to trigger",
    )

    storage.create_extendible(
        file,
        "/trigger/truncated",
        bool,
        desc="Flag if run was truncated before equilibrium",
    )

    storage.create_extendible(
        file,
        "/trigger/branched",
        bool,
        desc="Flag if configuration followed from a branch",
    )

    storage.dset_extend1d(file, "/trigger/element", 0, element)
    storage.dset_extend1d(file, "/trigger/truncated", 0, False)
    storage.dset_extend1d(file, "/trigger/branched", 0, True)

    if meta is not None:
        g = file.create_group(meta[0])
        for key in meta[1]:
            g.attrs[key] = meta[1][key]


def _write_configurations(
    element: int,
    info: h5py.File,
    force: bool = False,
    dev: bool = False,
    source: list[str] = None,
    dest: list[str] = None,
    inc: list[int] = None,
    incc: list[int] = None,
    stress: list[float] = None,
    meta: tuple[str, dict] = None,
):
    """
    Branch at a given increment or fixed stress.

    :param element: Element to trigger.
    :param info: EnsembleInfo to read configuration data from.
    :param force: Force overwrite of existing files.
    :param dev: Allow uncommitted changes.
    :param source: List with source file-paths.
    :param dest: List with destination file-paths.
    :param inc: List with fixed increment at which to push (entries can be ``None``).
    :param incc: List with system spanning events after which to load (entries can be ``None``).
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

    fmt = "{:" + str(max(len(i) for i in source)) + "s}"
    index = np.argsort(source)
    pbar = tqdm.tqdm(index)
    tmp = tempfile.mkstemp(suffix=".h5", dir=os.path.dirname(dest[0]))[1]

    for i, idx in enumerate(pbar):

        s = source[idx]
        d = dest[idx]
        pbar.set_description(fmt.format(s), refresh=True)

        with h5py.File(s, "r") as source_file:

            if i == 0:
                system = System.init(source_file)
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
                output["sigd"] = info[f"{p:s}/sigd"][...]
                output["A"] = info[f"{p:s}/A"][...]
                output["inc"] = info[f"{p:s}/inc"][...]
                with h5py.File(tmp, "w") as dest_file:
                    System.clone(source_file, dest_file)
                    System._init_run_state(dest_file)
                    _writeinitbranch(dest_file, element, meta)

            shutil.copy(tmp, d)

            with h5py.File(d, "a") as dest_file:

                System.branch_fixed_stress(
                    source=source_file,
                    dest=dest_file,
                    inc=inc[idx],
                    incc=incc[idx],
                    stress=stress[idx],
                    normalised=True,
                    system=system,
                    init_system=init_system,
                    init_dest=False,
                    output=output,
                    dev=dev,
                )

    os.remove(tmp)


def _copy_configurations(element: int, source: list[str], dest: list[str], force: bool = False):
    """
    Copy configurations written by :py:func:`_write_configurations` and
    overwrite the triggered element.

    :param element: Element to trigger.
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
            dest["/trigger/element"][0] = element


def __write(elements, ret, args, executable, cli_args, meta):
    """
    Internal use only.
    Just to avoid duplicate code.
    """

    if args.nmax is not None:
        for key in ret:
            ret[key] = ret[key][: args.nmax]

    # when previously present data have to be filtered the trick of generating for one element
    # and then copying cannot be used
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


def __filter(ret, filepath, meta):
    """
    Filter already run simulations.
    """

    assert os.path.isfile(filepath)

    with h5py.File(filepath, "r") as file:
        # cli_ensembleinfo
        if "sigd0" in file:
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
    progname = entry_points[funcname]

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--filter", type=str, help="Filter completed jobs")
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-d", "--delta-sigma", type=float, required=True, help="delta_sigma")
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
    assert args.delta_sigma > 0
    assert args.delta_sigma < np.max(sigd_loading - sigd)
    elements = np.linspace(0, N + 1, args.pushes + 1)[:-1].astype(int)

    ret = dict(
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
    )

    meta = {
        "deltasigma": args.delta_sigma,
        "pushes": args.pushes,
    }

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

            if args.subdir:
                bse = f"{args.outdir}/{simid}/deltasigma={args.delta_sigma:.3f}"
                if not os.path.isdir(f"{args.outdir}/{simid}"):
                    os.makedirs(f"{args.outdir}/{simid}")
            else:
                bse = f"{args.outdir}/deltasigma={args.delta_sigma:.3f}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={elements[0]:d}_istep={istress:02d}.h5"
            ret["source"].append(filepath)
            ret["dest"].append(out)
            ret["inc"].append(j)
            ret["incc"].append(inc[i])
            ret["stress"].append(s)

    return __write(elements, ret, args, executable, cli_args, meta=(f"/meta/{progname}", meta))


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
    progname = entry_points[funcname]

    parser.add_argument("--conda", type=str, default=slurm.default_condabase, help="Env-basename")
    parser.add_argument("--develop", action="store_true", help="Development mode")
    parser.add_argument("--filter", type=str, help="Filter completed jobs")
    parser.add_argument("--nmax", type=int, help="Keep first nmax jobs (mostly for testing)")
    parser.add_argument("--truncate-system-spanning", action="store_true", help="Stop large events")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite output")
    parser.add_argument("-n", "--group", type=int, default=50, help="#simulations to group")
    parser.add_argument("-o", "--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("-p", "--pushes", type=int, default=3, help="#elements per configuration")
    parser.add_argument("-r", "--subdir", action="store_true", help="Separate in directories")
    parser.add_argument("-s", "--steps", type=int, default=10, help="#pushes between ss-events")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("ensembleinfo", type=str, help="EnsembleInfo (read-only)")

    args = tools._parse(parser, cli_args)
    assert os.path.isfile(args.ensembleinfo)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

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
        source=[],
        dest=[],
        inc=[],
        incc=[],
        stress=[],
    )

    meta = {
        "strain_steps": args.steps,
        "pushes": args.pushes,
    }

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

            if args.subdir:
                bse = f"{args.outdir}/{simid}/strainsteps={args.steps:02d}"
                if not os.path.isdir(f"{args.outdir}/{simid}"):
                    os.makedirs(f"{args.outdir}/{simid}")
            else:
                bse = f"{args.outdir}/strainsteps={args.steps:02d}"
            out = f"{bse}_{simid}_incc={inc[i]:d}_element={elements[0]:d}_istep={istress:02d}.h5"
            ret["source"].append(filepath)
            ret["dest"].append(out)
            ret["inc"].append(j)
            ret["incc"].append(inc[i])
            ret["stress"].append(s)

    return __write(elements, ret, args, executable, cli_args, meta=(f"/meta/{progname}", meta))
