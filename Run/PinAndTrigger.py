import argparse
import FrictionQPotFEM.UniformSingleLayer2d as model
import git
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import QPot
import setuptools_scm


def initsystem(data):
    r'''
Read system from file.

:param h5py.File data: Open simulation HDF5 archive (read-only).
:return: FrictionQPotFEM.UniformSingleLayer2d.System
    '''

    system = model.System(
        data["/coor"][...],
        data["/conn"][...],
        data["/dofs"][...],
        data["/dofsP"][...],
        data["/elastic/elem"][...],
        data["/cusp/elem"][...])

    system.setMassMatrix(data["/rho"][...])
    system.setDampingMatrix(data["/damping/alpha"][...])
    system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
    system.setPlastic(data["/cusp/K"][...], data["/cusp/G"][...], data["/cusp/epsy"][...])
    system.setDt(data["/run/dt"][...])

    return system


def pushincrements(system, data, target_stress):
    r'''
Get a list of increment from which the stress can be reached by elastic loading only.

:param FrictionQPotFEM.UniformSingleLayer2d.System system: The system (modified: all increments visited).
:param h5py.File data: Open simulation HDF5 archive (read-only).
:param float target_stress: The stress at which to push (in real units).
:return: 
    ``inc_system`` List of system spanning avalanches.
    ``inc_push`` List of increment from which the stress can be reached by elastic loading only.
    '''

    kick = data["/kick"][...].astype(bool)
    incs = data["/stored"][...].astype(int)
    assert np.all(incs == np.arange(incs.size))
    assert kick.shape == incs.shape
    assert np.all(kick[::2] == False)
    assert np.all(kick[1::2] == True)

    A = np.zeros(incs.shape, dtype=int)
    Strain = np.zeros(incs.shape, dtype=float)
    Stress = np.zeros(incs.shape, dtype=float)

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

    for inc in incs:

        system.setU(data["/disp/{0:d}".format(inc)])

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        Sig = system.Sig()
        Eps = system.Eps()

        A[inc] = np.sum(idx != idx_n)
        Strain[inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        Stress[inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))     

        idx_n = np.array(idx, copy=True)

    # estimate steady-state using secant modulus:
    # - always skip two increments
    # - start with elastic loading
    K = np.empty_like(Stress)
    K[0] = np.inf
    K[1:] = (Stress[1:] - Stress[0]) / (Strain[1:] - Strain[0])
    steadystate = max(2, np.argmax(K <= 0.95 * K[1]))
    if kick[steadystate]: 
        steadystate += 1

    A[:steadystate] = 0

    inc_system = np.argwhere(A == N).ravel()
    inc_push = []
    inc_system_ret = []

    for i in range(inc_system.size - 1):

        # state after elastc loading
        s = Stress[inc_system[i] + 1: inc_system[i + 1]: 2]
        a = A[inc_system[i] + 1: inc_system[i + 1]: 2]
        n = incs[inc_system[i] + 1: inc_system[i + 1]: 2]

        if not np.any(s > target_stress):
            continue
        
        j = np.argmax(s > target_stress)
        ipush = n[j] - 1

        assert Stress[ipush] <= target_stress
        assert kick[ipush + 1] == False

        inc_push += [ipush]
        inc_system_ret += [n[0] - 1]

    inc_push = np.array(inc_push)
    inc_system_ret = np.array(inc_system_ret)

    return inc_system_ret, inc_push


def pinsystem(system, target_element, target_A):
    r'''
Pin down part of the system by converting blocks to being elastic: 
having a single parabolic potential with the minimum equal to the current minimum.

:param FrictionQPotFEM.UniformSingleLayer2d.System system: The system (modified: yield strains changed).
:param int target_element: The element to trigger.
:param int target_A: Number of blocks to keep unpinned (``target_A / 2`` on both sides of ``target_element``).
:return: Per element: pinned (``True``) or not (``False``)
    '''

    plastic = system.plastic()
    N = plastic.size
    nip = system.quad().nip()

    assert target_A <= N 
    assert target_element <= N 

    idx = system.plastic_CurrentIndex()
    i = int(N - target_A / 2)
    pinned = np.ones((3 * N), dtype=bool)
    pinned[i: i + target_A] = False
    pinned[N + i: N + i + target_A] = False
    pinned = pinned[N: 2 * N]
    pinned = np.roll(pinned, target_element)

    material = system.material()
    material_plastic = system.material_plastic()

    for i, e in enumerate(plastic):
        if pinned[i]:
            for q in range(nip):
                for cusp in [material.refCusp([e, q]), material_plastic.refCusp([i, q])]:
                    # cusp = m.refCusp([e, q])
                    chunk = cusp.refQPotChunked()
                    y = chunk.y()
                    ymax = y[-1] # get some scale
                    y = y[int(idx[i, q]): int(idx[i, q] + 2)] # idx is just left, slicing is up to not including 
                    ymin = 0.5 * sum(y) # current mininim
                    chunk.set_y([ymin - 2 * ymax, ymin + 2 * ymax]) 

    return pinned


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Filename of simulation file (read-only)")
    parser.add_argument("-o", "--output", type=str, help="Filename of output file (overwritten)")
    parser.add_argument("-s", "--stress", type=float, help="Stress as which to trigger")
    parser.add_argument("-i", "--incc", type=int, help="Increment number of last system-spanning event")
    parser.add_argument("-e", "--element", type=int, help="Element to push")
    parser.add_argument("-a", "--size", type=int, help="Number of elements to keep unpinned")
    args = parser.parse_args()
    assert os.path.isfile(os.path.realpath(args.file))
    assert os.path.realpath(args.file) != os.path.realpath(args.output)

    print('file =', args.file)
    print('output =', args.output)
    print('stress =', args.stress)
    print('incc =', args.incc)
    print('element =', args.element)
    print('size =', args.size)

    root = git.Repo(os.path.dirname(__file__), search_parent_directories=True).working_tree_dir
    myversion = setuptools_scm.get_version(root=root)

    target_stress = args.stress
    target_inc_system = args.incc
    target_A = args.size # number of blocks to keep unpinned
    target_element = args.element # element to trigger

    with h5py.File(args.file, "r") as data:

        system = initsystem(data)
        eps_kick = data["/run/epsd/kick"][...]
        N = system.plastic().size
        dV = system.quad().AsTensor(2, system.quad().dV())

        # (*) Determine at which increment a push could be applied
        
        inc_system, inc_push = pushincrements(system, data, target_stress)

        # (*) Reload specific increment based on target stress and system-spanning increment

        assert target_inc_system in inc_system
        i = max(np.argwhere(inc_push >= target_inc_system).ravel())
        inc = inc_push[i]
        assert target_inc_system == inc_system[i]

        system.setU(data["/disp/{0:d}".format(inc)])
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

        print('niter =', niter)

        output["/meta/PushAndTrigger/file"] = args.file
        output["/meta/PushAndTrigger/version"] = myversion
        output["/meta/PushAndTrigger/version_dependencies"] = model.version_dependencies()
        output["/meta/PushAndTrigger/target_stress"] = target_stress
        output["/meta/PushAndTrigger/target_inc_system"] = target_inc_system
        output["/meta/PushAndTrigger/target_A"] = target_A
        output["/meta/PushAndTrigger/target_element"] = target_element
        output["/meta/PushAndTrigger/S"] = np.sum(idx - idx_n)
        output["/meta/PushAndTrigger/A"] = np.sum(idx != idx_n)

