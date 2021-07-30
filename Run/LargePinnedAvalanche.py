import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import QPot
import h5py
import numpy as np


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

# todo: hardcoded for now
G = 1.0
eps0 = 5.0e-4
sig0 = 2.0 * G * eps0
l0 = np.pi

target_stress = 0.34
target_inc_system = 41
target_A = 200 # number of blocks to keep unpinned
target_element = 0 # element to trigger

with h5py.File("id=000.hdf5", "r") as data:

    system = initsystem(data)

    plastic = system.plastic()
    N = plastic.size
    nip = system.quad().nip()
    dV = system.quad().AsTensor(2, system.quad().dV())

    # (*) Determine at which increment a push could be applied
    
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
        Sig = system.Sig() / sig0
        Eps = system.Eps() / eps0           

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
    system.addSimpleShearToFixedStress(target_stress * sig0)

    # (*) Pin down a fraction of the system

    idx = system.plastic_CurrentIndex()

    # todo: this is not correctly: A should be the free cells!!!
    i = int((N - target_A) / 2)
    pinned = np.zeros((N), dtype=bool)
    pinned[i: i + target_A] = True
    pinned = np.roll(pinned, target_element)

    material = system.material()
    material_plastic = system.material_plastic()

    print(sum(pinned), target_A)

    for i, e in enumerate(plastic):
        if pinned[i]:
            for q in range(nip):
                cusp = material.refCusp([e, q])
                chunk = cusp.refQPotChunked()
                y = chunk.y()[:int(idx[i, q] + 2)] # idx is just left, slicing is up to not including
                y[-1] = np.inf
                chunk.set_y(y) 

                cusp = material_plastic.refCusp([i, q])
                chunk = cusp.refQPotChunked()
                y = chunk.y()[:int(idx[i, q] + 2)] # idx is just left, slicing is up to not including
                y[-1] = np.inf
                chunk.set_y(y)

    # (*) Apply push and minimise energy




    # print(system.plastic_CurrentYieldLeft())
    # print(system.plastic_CurrentYieldRight())
    # print(system.plastic_CurrentYieldRight())

            # print(chunk.y()[int(idx[i, q])], chunk.y()[int(idx[i, q] + 1)], epsp[i, q])
            # y = chunk.y()[:int(idx[i, q] + 1 + 1)]
            # print(y[-1] - epsp[i, q])
            # print(idx[i, q], chunk.i())

            # print(chunk.i())


    # print(inc)

    




    # print(inc_system)
# 


