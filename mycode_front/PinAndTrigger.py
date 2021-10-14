import argparse
import inspect
import itertools
import os
import re
import shutil
import sys
import textwrap
from collections import defaultdict

import click
import enstat
import FrictionQPotFEM.UniformSingleLayer2d as model
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import QPot  # noqa: F401
import shelephant
import tqdm
import yaml
from nested_dict import nested_dict

from . import slurm
from . import System
from . import tag
from . import tools
from ._version import version


entry_points = dict(
    cli_collect="PinAndTrigger_collect",
    cli_collect_combine="PinAndTrigger_collect_combine",
    cli_collect_update="PinAndTrigger_collect_update",
    cli_getdynamics_sync_A="PinAndTrigger_getdynamics_sync_A",
    cli_getdynamics_sync_A_average="PinAndTrigger_getdynamics_sync_A_average",
    cli_getdynamics_sync_A_check="PinAndTrigger_getdynamics_sync_A_check",
    cli_getdynamics_sync_A_combine="PinAndTrigger_getdynamics_sync_A_combine",
    cli_getdynamics_sync_A_job="PinAndTrigger_getdynamics_sync_A_job",
    cli_job="PinAndTrigger_job",
    cli_main="PinAndTrigger",
    cli_output_scalar="PinAndTrigger_output_scalar",
    cli_output_spatial="PinAndTrigger_output_spatial",
)


def replace_ep(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def interpret_filename(filename):
    """
    Split filename in useful information.
    """

    part = re.split("_|/", os.path.splitext(filename)[0])
    info = {}

    for i in part:
        key, value = i.split("=")
        info[key] = value

    for key in info:
        if key in ["stress"]:
            continue
        else:
            info[key] = int(info[key])

    return info


def pinning(system: model.System, target_element: int, target_A: int) -> np.ndarray:
    r"""
    Return pinning used in ``pinsystem``.

    :param system:
        The system (unchanged).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

    :return: Per element: pinned (``True``) or not (``False``)
    """

    plastic = system.plastic()
    N = plastic.size

    assert target_A <= N
    assert target_element <= N

    i = int(N - target_A / 2)
    pinned = np.ones((3 * N), dtype=bool)

    ii = i
    jj = i + target_A
    pinned[ii:jj] = False

    ii = N + i
    jj = N + i + target_A
    pinned[ii:jj] = False

    ii = N
    jj = 2 * N
    pinned = pinned[ii:jj]

    pinned = np.roll(pinned, target_element)

    return pinned


def pinsystem(system: model.System, target_element: int, target_A: int) -> np.ndarray:
    r"""
    Pin down part of the system by converting blocks to being elastic:
    having a single parabolic potential with the minimum equal to the current minimum.

    :param system:
        The system (modified: yield strains changed).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

    :return: Per element: pinned (``True``) or not (``False``)
    """

    plastic = system.plastic()
    nip = system.quad().nip()
    pinned = pinning(system, target_element, target_A)
    idx = system.plastic_CurrentIndex()
    material = system.material()
    material_plastic = system.material_plastic()

    for i, e in enumerate(plastic):
        if pinned[i]:
            for q in range(nip):
                for cusp in [
                    material.refCusp([e, q]),
                    material_plastic.refCusp([i, q]),
                ]:
                    chunk = cusp.refQPotChunked()
                    y = chunk.y()
                    ymax = y[-1]  # get some scale
                    ii = int(idx[i, q])
                    jj = int(idx[i, q] + 2)  # slicing is up to not including
                    y = y[ii:jj]
                    ymin = 0.5 * sum(y)  # current minimum
                    chunk.set_y([ymin - 2 * ymax, ymin + 2 * ymax])

    return pinned


def cli_main(cli_args=None):
    """
    1.  Restore system at a given increment.
    2.  Pin down part of the system.
    3.  Push a specific element and minimise energy.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-a", "--size", type=int, help="#elements to keep unpinned")
    parser.add_argument("-e", "--element", type=int, help="Plastic element to push")
    parser.add_argument("-f", "--file", type=str, help="Simulation (read-only)")
    parser.add_argument("-i", "--incc", type=int, help="Last system-spanning event")
    parser.add_argument("-o", "--output", type=str, help="Output file (overwritten)")
    parser.add_argument("-s", "--stress", type=float, help="Trigger stress (real unit)")
    parser.add_argument("-v", "--version", action="version", version=version)

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)
    assert os.path.realpath(args.file) != os.path.realpath(args.output)

    print("starting:", args.output)

    target_stress = args.stress
    target_inc_system = args.incc
    target_A = args.size  # number of blocks to keep unpinned
    target_element = args.element  # element to trigger

    with h5py.File(args.file, "r") as file:

        system = System.init(file)
        eps_kick = file["/run/epsd/kick"][...]

        # (*) Determine at which increment a push could be applied

        inc_system, inc_push = System.pushincrements(system, file, target_stress)

        # (*) Reload specific increment based on target stress and system-spanning increment

        assert target_inc_system in inc_system
        i = np.argmax((target_inc_system == inc_system) * (target_inc_system <= inc_push))
        inc = inc_push[i]
        assert target_inc_system == inc_system[i]

        system.setU(file[f"/disp/{inc:d}"])
        idx_n = system.plastic_CurrentIndex()
        system.addSimpleShearToFixedStress(target_stress)
        idx = system.plastic_CurrentIndex()
        assert np.all(idx == idx_n)

        # (*) Pin down a fraction of the system

        pinsystem(system, target_element, target_A)

    # (*) Apply push and minimise energy

    with h5py.File(args.output, "w") as output:

        output["/disp/0"] = system.u()
        idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

        system.triggerElementWithLocalSimpleShear(eps_kick, target_element)
        niter = system.minimise()

        output["/disp/1"] = system.u()
        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

        print("done:", args.output, ", niter = ", niter)

        meta = output.create_group(f"/meta/{progname}")
        meta.attrs["file"] = os.path.relpath(args.file, os.path.dirname(args.output))
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = System.dependencies(model)
        meta.attrs["target_stress"] = target_stress
        meta.attrs["target_inc_system"] = target_inc_system
        meta.attrs["target_A"] = target_A
        meta.attrs["target_element"] = target_element
        meta.attrs["S"] = np.sum(idx - idx_n)
        meta.attrs["A"] = np.sum(idx != idx_n)


def cli_collect(cli_args=None):
    """
    Collect output of several pushes in a single output-file.
    Requires files to be named "stress=X_A=X_id=X_incc=X_element=X" (in any order).
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument(
        "-a",
        "--min-a",
        type=int,
        help="Save events only with A > ... (to save disk space)",
        default=10,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file (appended)",
        default=f"{progname}.h5",
    )

    parser.add_argument(
        "-e",
        "--error",
        type=str,
        help="List corrupted files (if found)",
        default=f"{progname}.yaml",
    )

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", type=str, nargs="*", help="Files to add")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.isfile(file) for file in args.files])
    assert len(args.files) > 0

    corrupted = []
    existing = []

    with h5py.File(args.output, "a") as output:

        if "meta" not in output:
            meta = output.create_group("meta")
            meta.attrs["version"] = version
            meta.attrs["dependencies"] = System.dependencies(model)
        else:
            meta = output["meta"]
            assert meta.attrs["version"] == version
            assert list(meta.attrs["dependencies"]) == System.dependencies(model)

        for filepath in tqdm.tqdm(args.files):

            try:
                with h5py.File(filepath, "r") as file:
                    pass
            except:
                corrupted += [filepath]
                continue

            with h5py.File(filepath, "r") as file:
                paths = list(g5.getdatasets(file))
                verify = g5.verify(file, paths)
                if paths != verify:
                    corrupted += [filepath]
                    continue

            with h5py.File(filepath, "r") as file:

                info = interpret_filename(os.path.basename(filepath))
                meta = file["/meta/{:s}".format(entry_points["cli_main"])]
                assert meta.attrs["version"] == version
                assert list(meta.attrs["dependencies"]) == System.dependencies(model)
                assert meta.attrs["target_inc_system"] == info["incc"]
                assert meta.attrs["target_A"] == info["A"]
                assert meta.attrs["target_element"] == info["element"]
                assert interpret_filename(meta.attrs["file"])["id"] == info["id"]

                root = (
                    "/data"
                    "/stress={stress:s}"
                    "/A={A:d}"
                    "/id={id:03d}"
                    "/incc={incc:d}"
                    "/element={element:d}"
                ).format(**info)

                if root in output:
                    existing += [filepath]
                    continue

                datasets = ["meta"]
                if meta.attrs["A"] >= args.min_a:
                    datasets += ["/disp/0", "/disp/1"]

                g5.copy(file, output, datasets, root=root)

    if len(corrupted) > 0 or len(existing) > 0:
        shelephant.yaml.dump(args.error, dict(corrupted=corrupted, existing=existing), force=True)


def cli_collect_update(cli_args=None):
    """
    Update older files collected with :py:func:`cli_collect` to latest file layout.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points["cli_main"]

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Files to convert")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    with h5py.File(args.file, "r") as file:

        vers = "None"

        if "/meta/version" in file:
            vers = str(file["/meta/version"].asstr()[...])

        if tag.equal(vers, "2.4"):

            with h5py.File(args.output, "w") as output:

                meta = output.create_group("meta")
                meta.attrs["version"] = version
                meta.attrs["dependencies"] = System.dependencies(model)

                vers = str(file["/meta/version"].asstr()[...])
                deps = sorted(str(d) for d in file["/meta/version_dependencies"].asstr()[...])

                paths = list(g5.getdatasets(file, root="data", max_depth=5))
                paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

                for path in tqdm.tqdm(paths):
                    info = interpret_filename(path)
                    root = file[g5.join("data", path, root=True)]

                    meta = output.create_group(g5.join("data", path, "meta", progname, root=True))
                    meta.attrs["file"] = str(root["file"].asstr()[...])
                    meta.attrs["version"] = vers
                    meta.attrs["dependencies"] = deps
                    meta.attrs["target_stress"] = root["target_stress"][...]
                    meta.attrs["target_inc_system"] = info["incc"]
                    meta.attrs["target_A"] = info["A"]
                    meta.attrs["target_element"] = info["element"]
                    meta.attrs["S"] = root["S"][...]
                    meta.attrs["A"] = root["A"][...]

                    if "disp" not in root:
                        continue

                    p = [g5.join("data", path, "disp", str(i), root=True) for i in range(2)]
                    g5.copy(file, output, p)

                return 0

    raise OSError("Don't know how to interpret the data")


def cli_collect_combine(cli_args=None):
    """
    Combine two or more collections, see :py:func:`cli_collect` to obtain them from
    individual runs.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, help="Output file", default=f"{progname}.h5")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", type=str, nargs="*", help="Files to add")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.isfile(file) for file in args.files])
    assert len(args.files) > 0

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    shutil.copyfile(args.files[0], args.output)

    with h5py.File(args.output, "a") as output:

        for filename in tqdm.tqdm(args.files[1:]):

            with h5py.File(filename, "r") as file:

                m = file["meta"]
                n = output["meta"]

                for key in ["version", "dependencies"]:
                    assert list(m.attrs[key]) == list(n.attrs[key])

                paths = list(g5.getdatapaths(file))
                paths.remove("/meta")
                g5.copy(file, output, paths)


def cli_job(cli_args=None):
    """
    Run for fixed A by pushing two different elements (0 and N / 2).
    Fixed A implies that N - A are pinned, leaving A / 2 unpinned blocks on both sides of the
    pushed element.
    Note that the assumption is made that the push on the different elements of the same system
    in the same still still results in sufficiently independent measurements.

    Use "?? (todo)" to run for A that are smaller than the one used here.
    That function skips all events that are know to be too small,
    and therefore less time is waisted on computing small events.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-a", "--size", type=int, default=1200, help="Size to keep unpinned")
    parser.add_argument("-n", "--group", type=int, default=100, help="#pushes to group")
    parser.add_argument("-o", "--output", type=str, default=".", help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")

    parser.add_argument(
        "-f",
        "--finished",
        type=str,
        help="Result of {:s}".format(entry_points["cli_collect"]),
    )

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.info)
    assert os.path.isdir(args.output)
    if args.finished:
        assert os.path.isfile(args.finished)

    basedir = os.path.dirname(args.info)
    executable = entry_points["cli_main"]

    simpaths = []
    if args.finished:
        with h5py.File(args.finished, "r") as file:
            simpaths = list(g5.getpaths(file, max_depth=6))
            simpaths = [path.replace("/...", "") for path in simpaths]

    with h5py.File(args.info, "r") as file:

        files = [os.path.join(basedir, f) for f in file["/files"].asstr()[...]]
        N = file["/normalisation/N"][...]
        sig0 = file["/normalisation/sig0"][...]
        sigc = file["/averages/sigd_bottom"][...] * sig0
        sign = file["/averages/sigd_top"][...] * sig0

        stress_names = [
            "stress=0d6",
            "stress=1d6",
            "stress=2d6",
            "stress=3d6",
            "stress=4d6",
            "stress=5d6",
            "stress=6d6",
        ]

        stresses = [
            0.0 * (sign - sigc) / 6.0 + sigc,
            1.0 * (sign - sigc) / 6.0 + sigc,
            2.0 * (sign - sigc) / 6.0 + sigc,
            3.0 * (sign - sigc) / 6.0 + sigc,
            4.0 * (sign - sigc) / 6.0 + sigc,
            5.0 * (sign - sigc) / 6.0 + sigc,
            6.0 * (sign - sigc) / 6.0 + sigc,
        ]

        system = None
        commands = []

        for filename, (stress, stress_name) in itertools.product(
            files, zip(stresses, stress_names)
        ):

            with h5py.File(filename, "r") as file:

                if system is None:
                    system = System.init(file)
                else:
                    system.reset_epsy(System.read_epsy(file))

                trigger, _ = System.pushincrements(system, file, stress)

            simid = os.path.basename(os.path.splitext(filename)[0])
            filepath = os.path.relpath(filename, args.output)

            for element, A, incc in itertools.product([0, int(N / 2)], [args.size], trigger):

                root = (
                    f"/data"
                    f"/{stress_name}"
                    f"/A={A:d}"
                    f"/{simid}"
                    f"/incc={incc:d}"
                    f"/element={element:d}"
                )

                if root in simpaths:
                    continue

                output = f"{stress_name}_A={A:d}_{simid}_incc={incc:d}_element={element:d}.h5"
                cmd = " ".join(
                    [
                        f"{executable}",
                        f"-f {filepath}",
                        f"-o {output}",
                        f"-s {stress:.8e}",
                        f"-i {incc:d}",
                        f"-e {element:d}",
                        f"-a {A:d}",
                    ]
                )
                commands += [cmd]

    slurm.serial_group(
        commands,
        basename=executable,
        group=args.group,
        outdir=args.output,
        sbatch={"time": args.time},
    )

    return commands


def output_scalar(filepath: str, sig0: float):
    """
    Interpret scalar data of an ensemble, collected by :py:func:`cli_collect`.

    :param filepath: File with the ensemble.
    :param sig0: Stress normalisation.
    :return: Dictionary with output per stress/A.
    """

    system = None
    ret = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dirname = os.path.dirname(filepath)

    with h5py.File(filepath, "r") as file:

        # list with realisations
        paths = list(g5.getdatasets(file, root="data", max_depth=5))
        paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

        for path in tqdm.tqdm(paths):

            info = interpret_filename(path)
            root = g5.join("data", path, root=True)
            root = file[root]
            meta = g5.join("data", path, "meta", entry_points["cli_main"], root=True)
            meta = file[meta]

            if "disp" not in root:
                continue

            with h5py.File(os.path.join(dirname, meta.attrs["file"]), "r") as simfile:
                if system is None:
                    system = System.init(simfile)
                    dV = system.quad().AsTensor(2, system.quad().dV())
                    plastic = system.plastic()
                    plastic_dV = dV[plastic, ...]
                else:
                    system.reset_epsy(System.read_epsy(simfile))

            system.setU(root["disp"]["0"][...])
            pinned = pinsystem(system, meta.attrs["target_element"], meta.attrs["target_A"])
            idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

            system.setU(root["disp"]["1"][...])
            idx = system.plastic_CurrentIndex()[:, 0].astype(int)

            s = np.sum(idx - idx_n)
            a = np.sum(idx != idx_n)
            Sig = np.average(system.Sig(), weights=dV, axis=(0, 1)) / sig0
            plastic_sig = np.average(system.plastic_Sig(), weights=plastic_dV, axis=1)
            Sig /= sig0
            plastic_sig /= sig0

            stress = "stress={stress:s}".format(**info)
            A = "A={A:d}".format(**info)
            ret[stress][A]["S"].append(s)
            ret[stress][A]["A"].append(a)
            ret[stress][A]["Sig_xx"].append(Sig[0, 0])
            ret[stress][A]["Sig_xy"].append(Sig[0, 1])
            ret[stress][A]["Sig_yy"].append(Sig[1, 1])

            if s > 0:
                w = idx != idx_n
                moving_sig = np.average(plastic_sig, weights=w, axis=0)
                ret[stress][A]["moving_sig_xx"].append(moving_sig[0, 0])
                ret[stress][A]["moving_sig_xy"].append(moving_sig[0, 1])
                ret[stress][A]["moving_sig_yy"].append(moving_sig[1, 1])

                w = tools.fill_avalanche(idx != idx_n)
                crack_sig = np.average(plastic_sig, weights=w, axis=0)
                ret[stress][A]["crack_sig_xx"].append(crack_sig[0, 0])
                ret[stress][A]["crack_sig_xy"].append(crack_sig[0, 1])
                ret[stress][A]["crack_sig_yy"].append(crack_sig[1, 1])

                w = np.logical_not(pinned)
                unpinned_sig = np.average(plastic_sig, weights=w, axis=0)
                ret[stress][A]["unpinned_sig_xx"].append(unpinned_sig[0, 0])
                ret[stress][A]["unpinned_sig_xy"].append(unpinned_sig[0, 1])
                ret[stress][A]["unpinned_sig_yy"].append(unpinned_sig[1, 1])
            else:
                for t in ["crack", "unpinned", "moving"]:
                    for d in ["xx", "xy", "yy"]:
                        ret[stress][A][f"{t}_sig_{d}"].append(np.NaN)

    return ret


def cli_output_scalar(cli_args=None):
    """
    Interpret scalar data of an ensemble, collected by :py:func:`cli_collect`.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, default=f"{progname}.h5", help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Input file")

    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="{:s}.h5".format(System.entry_points["cli_ensembleinfo"]),
        help="EnsembleInfo to read normalisation (read-only)",
    )

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)
    assert os.path.isfile(args.info)

    with h5py.File(args.info, "r") as file:
        sig0 = file["/normalisation/sig0"][...]

    data = output_scalar(args.file, sig0)

    with h5py.File(args.output, "w") as output:
        g5.dump(output, data)


def output_spatial(filepath: str, sig0: float):
    """
    Interpret spatial average of an ensemble, collected by :py:func:`cli_collect`.
    Note that this implies interpolation to a regular grid.

    :param filepath: File with the ensemble.
    :param sig0: Stress normalisation.
    :return: Dictionary with output per stress/A (as 'matrix').
    """

    system = None
    ret = defaultdict(lambda: defaultdict(lambda: defaultdict(enstat.mean.StaticNd)))
    dirname = os.path.dirname(filepath)

    with h5py.File(filepath, "r") as file:

        # list with realisations
        paths = list(g5.getdatasets(file, root="data", max_depth=5))
        paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

        for path in tqdm.tqdm(paths):

            info = interpret_filename(path)
            root = g5.join("data", path, root=True)
            root = file[root]
            meta = g5.join("data", path, "meta", entry_points["cli_main"], root=True)
            meta = file[meta]

            if "disp" not in root:
                continue

            with h5py.File(os.path.join(dirname, meta.attrs["file"]), "r") as simfile:
                if system is None:
                    system = System.init(simfile)
                    dV = system.quad().AsTensor(2, system.quad().dV())
                    plastic = system.plastic()
                    mesh = GooseFEM.Mesh.Quad4.FineLayer(system.coor(), system.conn())
                    mapping = GooseFEM.Mesh.Quad4.Map.FineLayer2Regular(mesh)
                    regular = mapping.getRegularMesh()
                    elmat = regular.elementgrid()
                    isplastic = np.zeros((system.conn().shape[0]), dtype=bool)
                    isplastic[plastic] = True
                else:
                    system.reset_epsy(System.read_epsy(simfile))

            system.setU(root["disp"]["0"][...])
            pinsystem(system, meta.attrs["target_element"], meta.attrs["target_A"])
            idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

            system.setU(root["disp"]["1"][...])
            idx = system.plastic_CurrentIndex()[:, 0].astype(int)

            shift = tools.center_avalanche(idx != idx_n)
            _elmat = np.roll(elmat, shift, axis=1).ravel()
            s = np.roll(idx - idx_n, shift)
            a = np.roll(idx != idx_n, shift)
            Sig = system.Sig() / sig0
            Sig = np.average(Sig, weights=dV, axis=(1,))

            def to_regular(field):
                return mapping.mapToRegular(field)[_elmat].reshape(elmat.shape)

            stress = "stress={stress:s}".format(**info)
            A = "A={A:d}".format(**info)
            ret[stress][A]["S"].add_sample(s)
            ret[stress][A]["A"].add_sample(a.astype(int))
            ret[stress][A]["sig_xx"].add_sample(to_regular(Sig[:, 0, 0]))
            ret[stress][A]["sig_xy"].add_sample(to_regular(Sig[:, 0, 1]))
            ret[stress][A]["sig_yy"].add_sample(to_regular(Sig[:, 1, 1]))
            ret[stress][A]["isplastic"].add_sample(to_regular(isplastic))

    return ret


def cli_output_spatial(cli_args=None):
    """
    Interpret spatial average an ensemble, collected by :py:func:`cli_collect`.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, default=f"{progname}.h5", help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Input file")

    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="{:s}.h5".format(System.entry_points["cli_ensembleinfo"]),
        help="EnsembleInfo to read normalisation (read-only)",
    )

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)
    assert os.path.isfile(args.info)

    with h5py.File(args.info, "r") as file:
        sig0 = file["/normalisation/sig0"][...]

    data = output_spatial(args.file, sig0)

    with h5py.File(args.output, "w") as output:
        for stress in data:
            for A in data[stress]:
                for key in data[stress][A]:
                    root = f"{stress}/{A}/{key}/"
                    output[root + "mean"] = data[stress][A][key].mean()
                    output[root + "variance"] = data[stress][A][key].variance()
                    output[root + "norm"] = data[stress][A][key].norm()
                    output[root + "first"] = data[stress][A][key].first()
                    output[root + "second"] = data[stress][A][key].second()


def getdynamics_sync_A(
    system: model.System,
    target_element: int,
    target_A: int,
    eps_kick: float,
    sig0: float,
    t0: float,
) -> dict:
    """
    Run the dynamics of an event, saving the state at the interface at every "A".

    :param system:
        The initialised system, initialised to the proper displacement
        (but not pinned down yet).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

    :param sig0
        Stress normalisation.

    :param t0
        Time normalisation.

    :param eps_kick:
        Strain kick to use.

    :return:
        Dictionary with the following fields:
        *   Shape ``(target_A, N)``:
            -   sig_xx, sig_xy, sig_yy: the average stress along the interface.
        *   Shape ``(target_A, target_A)``:
            -   idx: the current potential index along the interface.
        *   Shape ``(target_A)``:
            -   t: the duration since nucleating the event.
            -   Sig_xx, Sig_xy, Sig_yy: the macroscopic stress.
    """

    plastic = system.plastic()
    N = plastic.size
    dV = system.quad().AsTensor(2, system.quad().dV())
    plastic_dV = dV[plastic, ...]
    pinned = pinsystem(system, target_element, target_A)
    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
    system.triggerElementWithLocalSimpleShear(eps_kick, target_element)

    a_n = 0
    a = 0

    ret = dict(
        pinned=pinned,
        sig_xx=np.zeros((target_A, N), dtype=np.float64),
        sig_xy=np.zeros((target_A, N), dtype=np.float64),
        sig_yy=np.zeros((target_A, N), dtype=np.float64),
        idx=np.zeros((target_A, N), dtype=np.uint64),
        t=np.zeros(target_A, dtype=np.float64),
        Sig_xx=np.zeros((target_A), dtype=np.float64),
        Sig_xy=np.zeros((target_A), dtype=np.float64),
        Sig_yy=np.zeros((target_A), dtype=np.float64),
    )

    while True:

        niter = system.timeStepsUntilEvent()
        idx = system.plastic_CurrentIndex()[:, 0].astype(int)

        if np.sum(idx != idx_n) > a:
            a_n = a
            a = np.sum(idx != idx_n)
            sig = np.average(system.Sig(), weights=dV, axis=(0, 1))
            plastic_sig = np.average(system.plastic_Sig(), weights=plastic_dV, axis=1)

            # store to output (broadcast if needed)
            ret["sig_xx"][a_n:a, :] = plastic_sig[:, 0, 0].reshape(1, -1) / sig0
            ret["sig_xy"][a_n:a, :] = plastic_sig[:, 0, 1].reshape(1, -1) / sig0
            ret["sig_yy"][a_n:a, :] = plastic_sig[:, 1, 1].reshape(1, -1) / sig0
            ret["idx"][a_n:a, :] = idx.reshape(1, -1)
            ret["t"][a_n:a] = system.t() / t0
            ret["Sig_xx"][a_n:a] = sig[0, 0] / sig0
            ret["Sig_xy"][a_n:a] = sig[0, 1] / sig0
            ret["Sig_yy"][a_n:a] = sig[1, 1] / sig0

        if a >= target_A:
            break

        if niter == 0:
            break

    ret["idx"] = ret["idx"][:, np.logical_not(pinned)]

    return ret


def cli_getdynamics_sync_A(cli_args=None):
    """
    This script use a configuration file as follows:

    .. code-block:: yaml

        collected: PinAndTrigger_collect.h5
        info: EnsembleInfo.h5
        output: myoutput.h5
        paths:
          - stress=0d6/A=100/id=183/incc=45/element=0
          - stress=0d6/A=100/id=232/incc=41/element=729

    To generate use :py:func`cli_getdynamics_sync_A_job`.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="YAML configuration file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    with open(args.file) as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    assert os.path.isfile(config["info"])

    with h5py.File(config["info"], "r") as file:
        sig0 = file["/normalisation/sig0"][...]
        t0 = file["/normalisation/t0"][...]

    system = None

    with h5py.File(config["output"], "w") as output:

        with h5py.File(config["collected"], "r") as file:

            for path in config["paths"]:

                meta = file["data"][path]["meta"][entry_points["cli_main"]]
                origsim = meta.attrs["file"]

                with h5py.File(origsim, "r") as mysim:
                    if system is None:
                        system = System.init(mysim)
                        eps_kick = mysim["/run/epsd/kick"][...]
                    else:
                        system.reset_epsy(System.read_epsy(mysim))

                system.setU(file["data"][path]["disp"]["0"][...])

                ret = getdynamics_sync_A(
                    system=system,
                    target_element=meta.attrs["target_element"],
                    target_A=meta.attrs["target_A"],
                    eps_kick=eps_kick,
                    sig0=sig0,
                    t0=t0,
                )

                for key in ret:
                    output[f"/data/{path}/{key}"] = ret[key]


def cli_getdynamics_sync_A_job(cli_args=None):
    """
    Generate configuration files to rerun dynamics.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-n", "--group", type=int, default=50, help="#runs to group.")
    parser.add_argument("outdir", type=str, help="Output directory")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=entry_points["cli_getdynamics_sync_A"],
        help="Output base-name (appended with number and extension)",
    )

    parser.add_argument(
        "-c",
        "--collect",
        type=str,
        default="{:s}.h5".format(entry_points["cli_collect"]),
        help="Existing data (see {:s}) (read-only)".format(entry_points["cli_main"]),
    )

    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="{:s}.h5".format(System.entry_points["cli_ensembleinfo"]),
        help="EnsembleInfo to read normalisation (read-only)",
    )

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.collect)
    assert os.path.isfile(args.info)

    config = dict(
        collected=args.collect,
        info=args.info,
    )

    with h5py.File(args.collect, "r") as file:

        # list with realisations
        paths = list(g5.getdatasets(file, root="data", max_depth=5))
        paths = np.array([path.split("data/")[1].split("/...")[0] for path in paths])

        # lists with stress/element/A of each realisation
        stress = []
        element = []
        a_target = []
        a_real = []
        for path in paths:
            info = interpret_filename(path)
            meta = g5.join("data", path, "meta", entry_points["cli_main"], root=True)
            meta = file[meta]
            stress += [info["stress"]]
            element += [info["element"]]
            a_target += [info["A"]]
            a_real += [meta.attrs["A"]]
        stress = np.array(stress)
        element = np.array(element)
        a_target = np.array(a_target)
        a_real = np.array(a_real)

        # lists with possible stress/element/A identifiers (unique)
        Stress = np.unique(stress)
        np.unique(element)
        A_target = np.unique(a_target)

        files = []
        for a, s in itertools.product(A_target, Stress):
            subset = paths[(a_real > 0) * (a_real > a - 10) * (a_real < a + 10) * (stress == s)]
            files += list(subset)

    if len(files) == 0:
        return

    chunks = int(np.ceil(len(files) / float(args.group)))
    devided = np.array_split(files, chunks)
    njob = len(devided)
    fmt = args.output + "_{0:" + str(int(np.ceil(np.log10(njob)))) + "d}-of-" + str(njob)
    ret = []

    for i, group in enumerate(devided):

        bname = fmt.format(i + 1)
        cname = os.path.join(args.outdir, bname + ".yaml")
        oname = os.path.join(args.outdir, bname + ".h5")

        assert not os.path.isfile(cname)
        assert not os.path.isfile(oname)

        config = dict(
            collected=os.path.relpath(args.collect, args.outdir),
            info=os.path.relpath(args.info, args.outdir),
            output=os.path.relpath(oname, args.outdir),
            paths=[str(g) for g in group],
        )

        shelephant.yaml.dump(cname, config)
        slurm.serial(f"{progname:s} {cname:s}", bname, outdir=args.outdir)
        ret += [cname]

    return ret


def cli_getdynamics_sync_A_combine(cli_args=None):
    """
    Combine output from :py:func:`cli_getdynamics_sync_A`
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, default=f"{progname}.h5", help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert len(args.files) > 0
    assert np.all([os.path.isfile(path) for path in args.files])

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    shutil.copyfile(args.files[0], args.output)

    with h5py.File(args.output, "a") as output:

        for filename in tqdm.tqdm(args.files[1:]):

            with h5py.File(filename, "r") as file:

                paths = list(g5.getdatapaths(file))
                exists = np.in1d(paths, list(g5.getdatapaths(output)))

                if np.sum(exists) > 0:
                    paths = np.array(paths)
                    print("The following paths are already present and are skipped")
                    print("\n".join(paths[exists]))
                    paths = list(paths[np.logical_not(exists)])

                if len(paths) > 0:
                    g5.copy(file, output, paths)


def getdynamics_sync_A_check(filepaths: list[str]):
    """
    Check integrity of files spanning the ensemble.

    :param filepaths: Files with the raw data.
    :return: dict with:
        - "duplicate": duplicate paths per file (with pointer to duplicate)
        - "corrupted": corrupted paths per file
        - "unique": paths per file, such that a unique subset is taken.
        - "summary" : a basic summary of the above
    """

    Paths = {}  # non-corrupted paths per file
    Unique = defaultdict(list)  # unique subset of "Paths"
    Corrupted = {}  # corrupted paths per file
    Duplicate = defaultdict(dict)  # duplicate paths per file
    Visited = nested_dict()  # internal check for duplicates

    # get paths, filter corrupted data

    for filepath in tqdm.tqdm(filepaths):

        if os.path.splitext(filepath)[1] in [".yml", ".yaml"]:
            info = shelephant.yaml.read(filepath)
            root = os.path.join(os.path.dirname(filepath), info["output"])
            Paths[root] = info["paths"]
            continue

        with h5py.File(filepath, "r") as file:
            paths = list(g5.getdatasets(file, max_depth=6))
            paths = [path.replace("/...", "").replace("/data/", "") for path in paths]
            d = file["data"]
            has_data = [True if "pinned" in d[path] else False for path in paths]
            paths = np.array(paths)
            has_data = np.array(has_data)
            no_data = np.logical_not(has_data)
            Paths[filepath] = [str(path) for path in paths[has_data]]
            if np.any(no_data):
                Corrupted[filepath] = [str(path) for path in paths[no_data]]

    # check duplicates

    for ifile, filepath in enumerate(tqdm.tqdm(Paths)):

        for path in Paths[filepath]:

            info = interpret_filename(path)
            stress = info["stress"]
            A = info["A"]
            s = info["id"]
            e = info["element"]
            i = info["incc"]

            if i in Visited[stress][s][A][e]:
                other = filepaths[Visited[stress][s][A][e][i]]
                Duplicate[filepath][path] = other
                Duplicate[other][path] = filepath
            else:
                Unique[filepath].append(path)

            Visited[stress][s][A][e][i] = ifile

    Summary = dict(
        unique=0,
        corrupted=0,
        duplicate=0,
    )

    for filepath in Unique:
        Summary["unique"] += len(Unique[filepath])

    for filepath in Corrupted:
        Summary["corrupted"] += len(Corrupted[filepath])

    for filepath in Duplicate:
        Summary["duplicate"] += len(Duplicate[filepath])
    Summary["duplicate"] = int(Summary["duplicate"] / 2)

    return dict(
        duplicate=dict(Duplicate),
        corrupted=dict(Corrupted),
        unique=dict(Unique),
        summary=Summary,
    )


def cli_getdynamics_sync_A_check(cli_args=None):
    """
    Check output from :py:func:`cli_getdynamics_sync_A`
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, default=f"{progname}.yaml", help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.isfile(path) for path in args.files])

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    output = getdynamics_sync_A_check(args.files)
    shelephant.yaml.dump(args.output, output, force=True)


def getdynamics_sync_A_average(paths: dict):
    """
    Get the averages on fixed A.

    :param paths:
        A dictionary with which "path" to read from which file.
        Use e.g. `getdynamics_sync_A_check(filepaths)["unique"]`

    :return: dict, per stress, A, and variable.
    """

    ret = defaultdict(lambda: defaultdict(lambda: defaultdict(enstat.mean.StaticNd)))

    for filepath in paths:

        with h5py.File(filepath, "r") as file:

            for path in tqdm.tqdm(paths[filepath]):

                info = interpret_filename(path)
                stress = info["stress"]
                A = info["A"]

                pinned = file["data"][path]["pinned"][...]
                sig_xx = file["data"][path]["sig_xx"][...]
                sig_xy = file["data"][path]["sig_xy"][...]
                sig_yy = file["data"][path]["sig_yy"][...]
                idx = file["data"][path]["idx"][...].astype(np.int64)
                t = file["data"][path]["t"][...]
                Sig_xx = file["data"][path]["Sig_xx"][...]
                Sig_xy = file["data"][path]["Sig_xy"][...]
                Sig_yy = file["data"][path]["Sig_yy"][...]

                i = np.sum(idx, axis=1)
                mask = np.logical_or(i[1:] == i[:-1], i[1:] == 0)
                mask = np.insert(mask, 0, False)

                S = np.zeros(sig_xx.shape, dtype=np.int64)
                S[:, np.logical_not(pinned)] = idx
                S[1:, :] -= S[0, :].reshape(1, -1)
                S[0, :] = 0

                # weight: 1 if S > 0, 0 otherwise (rows of zeros are replaced by rows of ones)
                w = np.where((np.sum(S, axis=1) > 0)[:, np.newaxis], S > 0, True)
                crack_sig_xx = np.average(sig_xx, weights=w, axis=1)
                crack_sig_xy = np.average(sig_xy, weights=w, axis=1)
                crack_sig_yy = np.average(sig_yy, weights=w, axis=1)

                shift = tools.center_avalanche_per_row(S)
                sig_xx = tools.indep_roll(sig_xx, shift, axis=1)
                sig_xy = tools.indep_roll(sig_xy, shift, axis=1)
                sig_yy = tools.indep_roll(sig_yy, shift, axis=1)
                S = tools.indep_roll(S, shift, axis=1)

                ret[stress][A]["sig_xx"].add_sample(sig_xx, mask=mask)
                ret[stress][A]["sig_xy"].add_sample(sig_xy, mask=mask)
                ret[stress][A]["sig_yy"].add_sample(sig_yy, mask=mask)
                ret[stress][A]["S"].add_sample(S, mask=mask)
                ret[stress][A]["t"].add_sample(t, mask=mask)
                ret[stress][A]["Sig_xx"].add_sample(Sig_xx, mask=mask)
                ret[stress][A]["Sig_xy"].add_sample(Sig_xy, mask=mask)
                ret[stress][A]["Sig_yy"].add_sample(Sig_yy, mask=mask)
                ret[stress][A]["crack_sig_xx"].add_sample(crack_sig_xx, mask=mask)
                ret[stress][A]["crack_sig_xy"].add_sample(crack_sig_xy, mask=mask)
                ret[stress][A]["crack_sig_yy"].add_sample(crack_sig_yy, mask=mask)

    return ret


def cli_getdynamics_sync_A_average(cli_args=None):
    """
    Average output from :py:func:`cli_getdynamics_sync_A`.
    Note that the input may be the YAML configuration files which served to run
    :py:func:`cli_getdynamics_sync_A`.
    See also :py:func:`cli_getdynamics_sync_A_job`.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_ep(doc))
    progname = entry_points[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-o", "--output", type=str, default=f"{progname}.h5", help="Output file")
    parser.add_argument("-s", "--summary", type=str, help="Summary", default=f"{progname}.yaml")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.isfile(path) for path in args.files])

    if not args.force:
        for filepath in [args.output, args.summary]:
            if os.path.isfile(filepath):
                if not click.confirm(f'Overwrite "{filepath}"?'):
                    raise OSError("Cancelled")

    info = getdynamics_sync_A_check(args.files)
    data = getdynamics_sync_A_average(info["unique"])

    shelephant.yaml.dump(args.summary, info, force=True)

    with h5py.File(args.output, "w") as output:

        for stress in data:
            for A in data[stress]:
                for key in data[stress][A]:
                    d = data[stress][A][key]
                    k = f"/stress={stress}/A={A:d}/{key}"
                    output[g5.join(k, "mean", root=True)] = d.mean()
                    output[g5.join(k, "variance", root=True)] = d.variance()
                    output[g5.join(k, "norm", root=True)] = d.norm()
                    output[g5.join(k, "first", root=True)] = d.first()
                    output[g5.join(k, "second", root=True)] = d.second()
