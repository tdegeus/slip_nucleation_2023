import argparse
import inspect
import itertools
import os
import re
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
import yaml

from . import slurm
from . import System
from ._version import version


entry_points = dict(
    cli_main="PinAndTrigger",
    cli_job="PinAndTrigger_job",
    cli_collect="PinAndTrigger_collect",
    cli_collect_combine="PinAndTrigger_collect_combine",
    cli_getdynamics_sync_A="PinAndTrigger_getdynamics_sync_A",
    cli_getdynamics_sync_A_job="PinAndTrigger_getdynamics_sync_A_job",
    cli_getdynamics_sync_A_combine="PinAndTrigger_getdynamics_sync_A_combine",
)


def replace_entry_point(docstring):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        docstring = docstring.replace(r":py:func:`{:s}`".format(ep), entry_points[ep])
    return docstring


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

    parser.add_argument("-v", "--version", action="version", version=version)

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

    parser.add_argument("files", type=str, nargs="*", help="Files to add")

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

    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    assert len(args.files) > 0

    corrupted = []
    existing = []

    with h5py.File(args.output, "a") as output:

        if "/meta" not in output:
            meta = output.create_group("/meta")
            meta.attrs["version"] = version
            meta.attrs["dependencies"] = System.dependencies(model)
        else:
            meta = output["/meta"]
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

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-v", "--version", action="version", version=version)
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
        "-e",
        "--executable",
        type=str,
        default=entry_points["cli_main"],
        help="Executable to use",
    )

    parser.add_argument(
        "-f",
        "--finished",
        type=str,
        help="Result of {:s}".format(entry_points["cli_collect"]),
    )

    parser.add_argument("-v", "--version", action="version", version=version)

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
        basename=args.executable.replace(" ", "_"),
        group=args.group,
        outdir=args.output,
        sbatch={"time": args.time},
    )

    return commands


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

    To generate use ``PinAndTrigger_getdynamics_sync_A_job``.
    """

    entry_points["cli_getdynamics_sync_A"]

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

    parser.add_argument("file", type=str, help="YAML configuration file")

    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))

    with open(args.file) as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    assert os.path.isfile(os.path.realpath(config["info"]))

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
    Generate configuration file to rerun dynamics.
    """

    progname = entry_points["cli_getdynamics_sync_A"]

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
        "-o",
        "--output",
        type=str,
        default=progname,
        help="Output base-name (appended with number and extension)",
    )

    parser.add_argument(
        "-c",
        "--collect",
        type=str,
        default="{:s}.h5".format(entry_points["cli_collect"]),
        help='Existing data, generated with "{:s}" (read-only)'.format(
            entry_points["cli_main"]
        ),
    )

    parser.add_argument(
        "-i",
        "--info",
        type=str,
        default="{:s}.h5".format(System.entry_points["cli_ensembleinfo"]),
        help="EnsembleInfo to read normalisation (read-only)",
    )

    parser.add_argument(
        "-n",
        "--group",
        type=int,
        default=50,
        help="Number of runs to group in a single job.",
    )

    parser.add_argument("outdir", type=str, help="Output directory")

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.collect))
    assert os.path.isfile(os.path.realpath(args.info))

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
            meta = file[g5.join("data", path, "meta", entry_points["cli_main"], root=True)]
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
            subset = paths[
                (a_real > 0) * (a_real > a - 10) * (a_real < a + 10) * (stress == s)
            ]
            files += list(subset)

    if len(files) == 0:
        return

    chunks = int(np.ceil(len(files) / float(args.group)))
    devided = np.array_split(files, chunks)
    njob = len(devided)
    fmt = (
        args.output + "_{0:" + str(int(np.ceil(np.log10(njob)))) + "d}-of-" + str(njob)
    )
    ret = []

    for i, group in enumerate(devided):

        bname = fmt.format(i + 1)
        cname = os.path.join(args.outdir, bname + ".yaml")
        oname = os.path.join(args.outdir, bname + ".h5")

        assert not os.path.isfile(os.path.realpath(cname))
        assert not os.path.isfile(os.path.realpath(oname))

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

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"{progname}.h5",
        help="Output file",
    )

    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Input files")

    args = parser.parse_args(cli_args)

    assert len(args.files) > 0
    assert np.all([os.path.isfile(os.path.realpath(path)) for path in args.files])

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    shutil.copyfile(args.files[0], args.output)

    with h5py.File(args.output, "a") as output:

        for filename in tqdm.tqdm(args.files[1:]):

            with h5py.File(filename, "r") as file:

                paths = list(g5.getdatapaths(file))
                g5.copy(file, output, paths)

