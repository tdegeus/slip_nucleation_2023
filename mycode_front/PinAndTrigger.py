import argparse
import os
import sys
import textwrap

import FrictionQPotFEM.UniformSingleLayer2d as model
import GooseFEM  # noqa: F401
import GooseHDF5 as g5
import h5py
import numpy as np
import QPot  # noqa: F401
import shelephant
import tqdm

from . import System
from ._version import version

mymodule = "PinAndTrigger"


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

        root = f"/meta/{mymodule}"
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

    basename = os.path.splitext(os.path.basename(__file__))[0]

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
        "-o", "--output", type=str, help="Output file ('a')", default=basename + ".h5"
    )

    parser.add_argument(
        "-e",
        "--error",
        type=str,
        help="Store list of corrupted files",
        default=basename + ".yaml",
    )

    parser.add_argument("files", type=str, nargs="*", help="Files to add")

    args = parser.parse_args(cli_args)

    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    assert len(args.files) > 0

    corrupted = []
    existing = []

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

                if mymodule in data["meta"]:
                    meta = data["meta"][mymodule]
                    root_meta = f"/meta/{mymodule}"
                elif "PushAndTrigger" in data["meta"]:
                    meta = data["meta"]["PushAndTrigger"]
                    root_meta = "/meta/PushAndTrigger"
                else:
                    raise OSError("Unknown input")

                if "/meta/version" in output:
                    assert version == meta["version"].asstr()[...]
                    assert deps == list(meta["version_dependencies"].asstr()[...])
                else:
                    version = meta["version"].asstr()[...]
                    deps = list(meta["version_dependencies"].asstr()[...])
                    output["/meta/version"] = version
                    output["/meta/version_dependencies"] = deps

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
