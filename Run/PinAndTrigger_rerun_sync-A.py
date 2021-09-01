import argparse
import itertools
import os
import sys

import enstat.mean
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import tqdm
import FrictionQPotFEM.UniformSingleLayer2d as model
from numpy.typing import ArrayLike

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import PinAndTrigger  # noqa: E402

def run_dynamics(system: model.System, target_element: int, target_A: int) -> dict:
    """
    Run the dynamics of an event, saving the state at the interface at every "A".

    :param system:
        The initialised system, initialised to the proper displacement
        (but not pinned down yet).

    :param target_element:
        The element to trigger.

    :param target_A:
        Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).

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
    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
    pinned = PinAndTrigger.pinsystem(system, target_element, target_A)
    system.triggerElementWithLocalSimpleShear(eps_kick, target_element)

    a_n = 0
    a = 0

    ret = dict(
        sig_xx = np.zeros((target_A, N), dtype=np.float64),
        sig_xy = np.zeros((target_A, N), dtype=np.float64),
        sig_yy = np.zeros((target_A, N), dtype=np.float64),
        idx = np.zeros((target_A, N), dtype=np.uint64),
        t = np.zeros(target_A, dtype=np.float64),
        Sig_xx = np.zeros((target_A), dtype=np.float64),
        Sig_xy = np.zeros((target_A), dtype=np.float64),
        Sig_yy = np.zeros((target_A), dtype=np.float64),
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
            ret["sig_xx"][a_n: a, :] = plastic_sig[:, 0, 0].reshape(1, -1)
            ret["sig_xy"][a_n: a, :] = plastic_sig[:, 0, 1].reshape(1, -1)
            ret["sig_yy"][a_n: a, :] = plastic_sig[:, 1, 1].reshape(1, -1)
            ret["idx"][a_n: a, :] = idx.reshape(1, -1)
            ret["t"][a_n: a] = system.t()
            ret["Sig_xx"][a_n: a] = sig[0, 0]
            ret["Sig_xy"][a_n: a] = sig[0, 1]
            ret["Sig_yy"][a_n: a] = sig[1, 1]

        if a >= target_A:
            break

        if niter == 0:
            break

    ret["idx"] = ret["idx"][:, np.logical_not(pinned)]

    return ret




