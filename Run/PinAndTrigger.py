import argparse
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import QPot

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

def initsystem(data):

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

target_stress = args.stress
target_inc_system = args.incc
target_A = args.size # number of blocks to keep unpinned
target_element = args.element # element to trigger

with h5py.File(args.file, "r") as data:

    system = initsystem(data)

    plastic = system.plastic()
    N = plastic.size
    nip = system.quad().nip()
    dV = system.quad().AsTensor(2, system.quad().dV())

    # (*) Determine at which increment a push could be applied
    
    eps_kick = data["/run/epsd/kick"][...]
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

    inc_push = np.array(inc_push)

    # (*) Reload specific increment based on target stress and system-spanning increment

    inc = inc_push[np.argmax(target_inc_system >= inc_push)]
    assert np.any(target_inc_system == inc_system)

    system.setU(data["/disp/{0:d}".format(inc)])
    system.addSimpleShearToFixedStress(target_stress)

    # (*) Pin down a fraction of the system

    idx = system.plastic_CurrentIndex()

    assert target_A <= N 
    assert target_element <= N 
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

# (*) Apply push and minimise energy

with h5py.File(args.output, "w") as output:

    output["/disp/0"] = system.u()
    system.triggerElementWithLocalSimpleShear(eps_kick, target_element)
    system.minimise()
    output["/disp/1"] = system.u()
