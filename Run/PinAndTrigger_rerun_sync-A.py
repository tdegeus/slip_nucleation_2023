"""This script use a configuration file as follows:

.. code-block:: yaml

    collected: PinAndTrigger_collect.h5
    info: EnsembleInfo.h5
    output: myoutput.h5
    paths:
      - stress=0d6/A=100/id=183/incc=45/element=0
      - stress=0d6/A=100/id=232/incc=41/element=729

To generate use ``PinAndTrigger_rerun_sync-A_job-serial.py``.
"""
import argparse
import os
import re
import sys

import FrictionQPotFEM.UniformSingleLayer2d as model
import h5py
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import PinAndTrigger  # noqa: E402


def run_dynamics(
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
    pinned = PinAndTrigger.pinsystem(system, target_element, target_A)
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


if __name__ == "__main__":

    basename = os.path.splitext(os.path.basename(__file__))[0]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    parser.add_argument("file", type=str, help="YAML configuration file")
    args = parser.parse_args()

    assert os.path.isfile(os.path.realpath(args.file))

    with open(args.file) as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    assert os.path.isfile(os.path.realpath(config["info"]))

    with h5py.File(config["info"], "r") as data:
        sig0 = data["/normalisation/sig0"][...]
        t0 = data["/normalisation/t0"][...]

    system = None

    with h5py.File(config["output"], "w") as output:

        with h5py.File(config["collected"], "r") as data:

            for path in config["paths"]:

                print(path)

                origsim = str(data["data"][path]["file"].asstr()[...])
                e = int(re.split(r"(element=)([0-9]*)", path)[2])
                a = int(re.split(r"(A=)([0-9]*)", path)[2])

                with h5py.File(origsim, "r") as mysim:
                    if system is None:
                        system = PinAndTrigger.initsystem(mysim)
                        eps_kick = mysim["/run/epsd/kick"][...]
                    else:
                        PinAndTrigger.reset_epsy(system, mysim)

                system.setU(data["data"][path]["disp"]["0"][...])

                ret = run_dynamics(system, e, a, eps_kick, sig0, t0)

                for key in ret:
                    output[f"/data/{path}/{key}"] = ret[key]
