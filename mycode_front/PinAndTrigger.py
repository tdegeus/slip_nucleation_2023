import argparse
import os
import shutil
import sys
import textwrap

import click
import FrictionQPotFEM.UniformSingleLayer2d as model
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import GooseSLURM
import itertools
import h5py
import numpy as np
import QPot  # noqa: F401
import shelephant
import tqdm

from . import slurm
from . import System
from ._version import version


# name of the entry points (used also a default file names)
entry_main = "PinAndTrigger"
entry_collect = "PinAndTrigger_collect"
entry_collect_combine = "PinAndTrigger_collect_combine"


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

    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))
    assert os.path.realpath(args.file) != os.path.realpath(args.output)

    print("starting:", args.output)

    target_stress = args.stress
    target_inc_system = args.incc
    target_A = args.size  # number of blocks to keep unpinned
    target_element = args.element  # element to trigger

    with h5py.File(args.file, "r") as data:

        system = System.init(data)
        eps_kick = data["/run/epsd/kick"][...]

        # (*) Determine at which increment a push could be applied

        inc_system, inc_push = System.pushincrements(system, data, target_stress)

        # (*) Reload specific increment based on target stress and system-spanning increment

        assert target_inc_system in inc_system
        i = np.argmax(
            (target_inc_system == inc_system) * (target_inc_system <= inc_push)
        )
        inc = inc_push[i]
        assert target_inc_system == inc_system[i]

        system.setU(data[f"/disp/{inc:d}"])
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

        root = f"/meta/{entry_main}"
        output[f"{root}/file"] = args.file
        output[f"{root}/version"] = version
        output[f"{root}/version_dependencies"] = model.version_dependencies()
        output[f"{root}/target_stress"] = target_stress
        output[f"{root}/target_inc_system"] = target_inc_system
        output[f"{root}/target_A"] = target_A
        output[f"{root}/target_element"] = target_element
        output[f"{root}/S"] = np.sum(idx - idx_n)
        output[f"{root}/A"] = np.sum(idx != idx_n)


def cli_collect(cli_args=None):
    """
    Collect output of several pushes in a single output-file.
    Requires files to be named "stress=X_A=X_id=X_incc=X_element=X" (in any order).
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

    parser.add_argument(
        "-A", "--min-A", type=int, help="Save events only with A > ...", default=10
    )

    parser.add_argument(
        "-o", "--output", type=str, help="Output file ('a')", default=f"{entry_collect}.h5"
    )

    parser.add_argument(
        "-e",
        "--error",
        type=str,
        help="Store list of corrupted files",
        default=f"{entry_collect}.yaml",
    )

    parser.add_argument("files", type=str, nargs="*", help="Files to add")

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    assert len(args.files) > 0

    corrupted = []
    existing = []
    version = None
    deps = None

    with h5py.File(args.output, "a") as output:

        for file in tqdm.tqdm(args.files):

            try:
                with h5py.File(file, "r") as data:
                    pass
            except:
                corrupted += [file]
                continue

            with h5py.File(file, "r") as data:
                paths = list(g5.getdatasets(data))
                verify = g5.verify(data, paths)
                if paths != verify:
                    corrupted += [file]
                    continue

            with h5py.File(file, "r") as data:

                basename = os.path.splitext(os.path.basename(file))[0]
                stress = basename.split("stress=")[1].split("_")[0]
                A = basename.split("A=")[1].split("_")[0]
                simid = basename.split("id=")[1].split("_")[0]
                incc = basename.split("incc=")[1].split("_")[0]
                element = basename.split("element=")[1].split("_")[0]

                # alias meta-data
                # (allow for typos in previous versions)

                if entry_main in data["meta"]:
                    meta = data["meta"][entry_main]
                    root_meta = f"/meta/{entry_main}"
                elif "PushAndTrigger" in data["meta"]:
                    meta = data["meta"]["PushAndTrigger"]
                    root_meta = "/meta/PushAndTrigger"
                else:
                    raise OSError("Unknown input")

                if version is None:
                    version = meta["version"].asstr()[...]
                    deps = list(meta["version_dependencies"].asstr()[...])
                    output["/meta/version"] = version
                    output["/meta/version_dependencies"] = deps
                else:
                    assert version == meta["version"].asstr()[...]
                    assert deps == list(meta["version_dependencies"].asstr()[...])

                assert int(incc) == meta["target_inc_system"][...]
                assert int(A) == meta["target_A"][...]
                assert int(element) == meta["target_element"][...]
                assert int(simid) == int(
                    os.path.splitext(str(meta["file"].asstr()[...]).split("id=")[1])[0]
                )

                root = f"/data/stress={stress}/A={A}/id={simid}/incc={incc}/element={element}"

                if root in output:
                    existing += [file]
                    continue

                source_datasets = [
                    f"{root_meta:s}/file",
                    f"{root_meta:s}/target_stress",
                    f"{root_meta:s}/S",
                    f"{root_meta:s}/A",
                ]

                dest_datasets = ["/file", "/target_stress", "/S", "/A"]

                if meta["A"][...] >= args.min_A:
                    source_datasets = ["/disp/0", "/disp/1"] + source_datasets
                    dest_datasets = ["/disp/0", "/disp/1"] + dest_datasets

                g5.copydatasets(data, output, source_datasets, dest_datasets, root)

    shelephant.yaml.dump(
        args.error, dict(corrupted=corrupted, existing=existing), force=True
    )


def cli_collect_combine(cli_args=None):
    f"""
    Combine two or more collections, see {entry_collect} to obtain them from
    individual runs.
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

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file (overwritten)",
        default=f"{entry_collect_combine}.h5",
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

        for file in tqdm.tqdm(args.files[1:]):

            with h5py.File(file, "r") as data:

                for key in ["/meta/version", "/meta/version_dependencies"]:
                    assert g5.equal(output, data, key)

                paths = list(g5.getdatasets(data))
                paths.remove("/meta/version")
                paths.remove("/meta/version_dependencies")

                g5.copydatasets(data, output, paths)


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

    parser.add_argument("-A", "--size", type=int, default=1200, help="Size to keep unpinned")

    parser.add_argument("-c", "--conda", type=str, default="code_velocity", help="Base name of the conda environment, appended '_E5v4' and '_s6g1'")

    parser.add_argument("-n", "--group", type=int, default=100, help="Number of pushes to group in a single job")

    parser.add_argument("-w", "--walltime", type=str, default="24h", help="Walltime to allocate for the job")

    parser.add_argument("-e", "--executable", type=str, default=entry_main, help="Executable to use")

    parser.add_argument(
        "-c", "--collection", type=str, help=f"Result of {entry_collect}"
    )

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.info))
    assert os.path.isfile(os.path.realpath(args.collection)) or not args.collection

    basedir = os.path.dirname(args.info)
    executable = args.executable

    simpaths = []
    if args.collection:
        with h5py.File(args.collection, "r") as file:
            simpaths = list(g5.getpaths(file, max_depth=6))
            simpaths = [path.replace("/...", "") for path in simpaths]


    with h5py.File(args.info, "r") as data:

        files = [os.path.join(basedir, f) for f in data["/files"].asstr()[...]]
        N = data["/normalisation/N"][...]
        sig0 = data["/normalisation/sig0"][...]
        sigc = data["/averages/sigd_bottom"][...] * sig0
        sign = data["/averages/sigd_top"][...] * sig0

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

            for element, A, incc in itertools.product([0, int(N / 2)], [args.size], trigger):

                root = (
                    f"/data/{stress_name}/A={A:d}/{simid}/incc={incc:d}/element={element:d}"
                )
                if root in simpaths:
                    continue

                output = (
                    f"{stress_name}_A={A:d}_{simid}_incc={incc:d}_element={element:d}.h5"
                )
                cmd = f"{executable} -f {filename} -o {output} -s {stress:.8e} -i {incc:d} -e {element:d} -a {A:d}"
                commands += [cmd]

    commands = [slurm.script_flush(cmd) for cmd in commands]

    ngroup = int(np.ceil(len(commands) / args.group))
    fmt = str(int(np.ceil(np.log10(ngroup))))

    for group in range(ngroup):

        ii = group * args.group
        jj = (group + 1) * args.group
        c = commands[ii:jj]
        command = "\n".join(c)
        command = slurm.script_exec(command)

        jobname = ("{0:s}-{1:0" + fmt + "d}").format(
            args.executable.replace(" ", "_"), group
        )

        sbatch = {
            "job-name": "_".join([slurm.default_jobbase, jobname]),
            "out": jobname + ".out",
            "nodes": 1,
            "ntasks": 1,
            "cpus-per-task": 1,
            "time": arg.walltime,
            "account": "pcsl",
            "partition": "serial",
        }

        open(jobname + ".slurm", "w").write(
            GooseSLURM.scripts.plain(command=command, **sbatch)
        )

