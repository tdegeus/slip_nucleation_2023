import argparse
import itertools
import os
import shutil
import sys
import textwrap

import click
import FrictionQPotFEM.UniformSingleLayer2d as model
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import QPot  # noqa: F401
import shelephant
import tqdm

from . import slurm
from . import System
from ._version import version



entry_points = dict(
    cli_main="PinAndTrigger",
    cli_job="PinAndTrigger_job",
    cli_collect="PinAndTrigger_collect",
    cli_collect_combine="PinAndTrigger_collect_combine",
)


def interpret_filename(filename):
    """
    Split filename in useful information.
    """

    part = os.path.splitext(os.path.basename(filename))[0].split("_")
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

    progname = entry_points["cli_main"]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=textwrap.dedent(cli_main.__doc__)
    )

    parser.add_argument(
        "-f", "--file", type=str, help="Filename of simulation file (read-only)"
    )

    parser.add_argument(
        "-o", "--output", type=str, help="Filename of output file (overwritten)"
    )

    parser.add_argument(
        "-s", "--stress", type=float, help="Stress as which to trigger (in real units)"
    )

    parser.add_argument(
        "-i", "--incc", type=int, help="Increment number of last system-spanning event"
    )

    parser.add_argument(
        "-e", "--element", type=int, help="Element to push (index along the weak layer)"
    )

    parser.add_argument(
        "-a", "--size", type=int, help="Number of elements to keep unpinned"
    )

    parser.add_argument(
        "-v", "--version", action="version", version=version
    )

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))
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
        i = np.argmax(
            (target_inc_system == inc_system) * (target_inc_system <= inc_push)
        )
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
        meta.attrs["file"] = args.file
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

    progname = entry_points["cli_collect"]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=textwrap.dedent(cli_collect.__doc__)
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Files to add"
    )

    parser.add_argument(
        "-a",
        "--min-a",
        type=int,
        help="Save events only with A > ... (to save disk space)",
        default=10
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file ('a')",
        default=f"{progname}.h5",
    )

    parser.add_argument(
        "-e",
        "--error",
        type=str,
        help="Store list of corrupted files",
        default=f"{progname}.yaml",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    assert len(args.files) > 0

    corrupted = []
    existing = []

    with h5py.File(args.output, "a") as output:

        if f"/meta" not in output:
            meta = output.create_group(f"/meta")
            meta.attrs["version"] = version
            meta.attrs["dependencies"] = System.dependencies(model)
        else:
            meta = output[f"/meta"]
            assert meta.attrs["version"] == version
            assert list(meta.attrs["dependencies"]) == System.dependencies(model)

        for filename in tqdm.tqdm(args.files):

            try:
                with h5py.File(filename, "r") as file:
                    pass
            except:
                corrupted += [filename]
                continue

            with h5py.File(filename, "r") as file:
                paths = list(g5.getdatasets(file))
                verify = g5.verify(file, paths)
                if paths != verify:
                    corrupted += [filename]
                    continue

            with h5py.File(filename, "r") as file:

                info = interpret_filename(filename)
                meta = file["/meta/{:s}".format(entry_points["cli_main"])]
                assert meta.attrs["version"] == version
                assert list(meta.attrs["dependencies"]) == System.dependencies(model)
                assert meta.attrs["target_inc_system"] == info["incc"]
                assert meta.attrs["target_A"] == info["A"]
                assert meta.attrs["target_element"] == info["element"]
                assert interpret_filename(meta.attrs["file"])["id"] == info["id"]

                root = "/data/stress={stress:s}/A={A:d}/id={id:03d}/incc={incc:d}/element={element:d}".format(**info)

                if root in output:
                    existing += [filename]
                    continue

                datasets = ["meta"]

                if meta.attrs["A"] >= args.min_a:
                    datasets = ["/disp/0", "/disp/1"] + datasets

                g5.copydatasets(file, output, datasets, root=root)

    if len(corrupted) > 0 or len(existing) > 0:
        shelephant.yaml.dump(
            args.error, dict(corrupted=corrupted, existing=existing), force=True
        )


def cli_collect_combine(cli_args=None):
    """
    Combine two or more collections, see PinAndTrigger_collect to obtain them from
    individual runs.
    """

    progname = entry_points["cli_collect_combine"]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent(cli_collect_combine.__doc__),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file (overwritten)",
        default=f"{progname}.h5",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite of output file"
    )

    parser.add_argument("files", type=str, nargs="*", help="Files to add")

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    assert len(args.files) > 0

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    shutil.copyfile(args.files[0], args.output)

    with h5py.File(args.output, "a") as output:

        for filename in tqdm.tqdm(args.files[1:]):

            with h5py.File(filename, "r") as file:

                for key in ["version", "dependencies"]:
                    assert list(file["meta"].attrs[key]) == list(output["meta"].attrs[key])

                paths = list(g5.getdatasets(file))
                g5.copydatasets(file, output, paths)


def cli_job(cli_args=None):
    """
    Run for fixed A by pushing two different elements (0 and N / 2).
    Fixed A implies that N - A are pinned, leaving A / 2 unpinned blocks on both sides of the
    pushed element.
    Note that the assumption is made that the push on the different elements of the same system
    in the same still still results in sufficiently independent measurements.

    Use "PinAndTrigger_job_compact" to run for A that are smaller than the one used here.
    That function skips all events that are know to be too small,
    and therefore less time is waisted on computing small events.
    """

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=textwrap.dedent(cli_collect.__doc__)
    )

    parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")

    parser.add_argument(
        "-o", "--output", type=str, default=".", help="Output directory"
    )

    parser.add_argument(
        "-a", "--size", type=int, default=1200, help="Size to keep unpinned"
    )

    parser.add_argument(
        "-c",
        "--conda",
        type=str,
        default=slurm.default_condabase,
        help="Base name of the conda environment, appended '_E5v4' and '_s6g1'",
    )

    parser.add_argument(
        "-n",
        "--group",
        type=int,
        default=100,
        help="Number of pushes to group in a single job",
    )

    parser.add_argument(
        "-w",
        "--time",
        type=str,
        default="24h",
        help="Walltime to allocate for the job",
    )

    parser.add_argument(
        "-e", "--executable", type=str, default=entry_points["cli_main"], help="Executable to use"
    )

    parser.add_argument("-f", "--finished", type=str, help="Result of {:s}".format(entry_points["cli_collect"]))

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.info))
    assert os.path.isdir(os.path.realpath(args.output))
    if args.finished:
        assert os.path.isfile(os.path.realpath(args.finished))

    basedir = os.path.dirname(args.info)
    executable = args.executable

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

            for element, A, incc in itertools.product(
                [0, int(N / 2)], [args.size], trigger
            ):

                root = f"/data/{stress_name}/A={A:d}/{simid}/incc={incc:d}/element={element:d}"
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
        basename=args.executable.replace(" ", "_"),
        group=args.group,
        outdir=args.output,
        sbatch={"time": args.time},
    )
