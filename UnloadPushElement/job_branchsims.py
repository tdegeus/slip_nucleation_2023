
import sys
import os
import re
import subprocess
import h5py
import numpy as np

def check_stress(filename, inc):

    import GooseFEM as fem
    import GMatElastoPlasticQPot as mat

    with h5py.File(filename, 'r') as data:

        coor = data['coor'][...]
        conn = data['conn'][...]
        dofs = data['dofs'][...]

        vector = fem.Vector(conn, dofs)
        quad = fem.Element.Quad4.Quadrature(vector.AsElement(coor))

        material = mat.Cartesian2d.Array2d([conn.shape[0], quad.nip()])

        I = np.zeros(material.shape(), dtype=np.int)
        idx = np.zeros(material.shape(), dtype=np.int)
        elem = data['/elastic/elem'][...]
        I[elem, :] = 1
        idx[elem, :] = np.arange(len(elem)).reshape(-1, 1)
        material.setElastic(I, idx, data['/elastic/K'][...], data['/elastic/G'][...])

        I = np.zeros(material.shape(), dtype=np.int)
        idx = np.zeros(material.shape(), dtype=np.int)
        elem = data['/cusp/elem'][...]
        I[elem, :] = 1
        idx[elem, :] = np.arange(len(elem)).reshape(-1, 1)
        material.setCusp(I, idx, data['/cusp/K'][...], data['/cusp/G'][...], data['/cusp/epsy'][...])

        sig = []

        for i in [inc - 1, inc, inc + 1]:

            disp = data['disp'][str(int(i))][...]
            Eps = quad.SymGradN_vector(vector.AsElement(disp))
            material.setStrain(Eps)
            Sig = material.Stress()

            dV = quad.AsTensor(2, quad.dV())
            sig += [float(mat.Cartesian2d.Sigd(np.average(Sig, weights=dV, axis=(0, 1))))]

        assert sig[0] >= sig[1]
        assert sig[1] <= sig[2]

        return sig[1]





dbase = '../../../data/nx=3^6x2'
N = (3**6) * 2

keys = [
    '/conn',
    '/coor',
    '/cusp/G',
    '/cusp/K',
    '/cusp/elem',
    '/cusp/epsy',
    '/damping/alpha',
    '/damping/eta_d',
    '/damping/eta_v',
    '/dofs',
    '/dofsP',
    '/elastic/G',
    '/elastic/K',
    '/elastic/elem',
    '/rho',
    '/run/dt',
    '/run/epsd/kick',
    '/run/epsd/max',
    '/uuid',
]

with h5py.File(os.path.join(dbase, 'EnsembleInfo.hdf5'), 'r') as data:

    sig0 = float(data['/normalisation/sig0'][...])
    A = data['/avalanche/A'][...]
    idx = np.argwhere(A == N).ravel()
    incs = data['/avalanche/inc'][idx]
    files = data['/files'][...][data['/avalanche/file'][idx]]
    stresses = data['/avalanche/sigd'][idx] * sig0

sigc = 0.15464095 * sig0
push_stresses = np.array([1.0 * sigc, 0.8 * sigc, 0.6 * sigc])

for stress, inc, file in zip(stresses, incs, files):

    # assert np.allclose([stress], [check_stress(os.path.join(dbase, file), inc)])
    print(file, inc, stress)

    for element in [146, 511, 730, 949, 1241, 1387]:

        outfilename = '{0:s}_element={1:d}_inc={2:d}.hdf5'.format(file.split('.hdf5')[0], element, inc)

        with h5py.File(os.path.join(dbase, file), 'r') as data:

            with h5py.File(outfilename, 'w') as output:

                for key in keys:
                    output[key] = data[key][...]

                output['/push/stresses'] = push_stresses
                output['/push/inc'] = inc
                output['/push/element'] = element
                output['/disp/0'] = data['disp'][str(inc)][...]

                dset = output.create_dataset('/stored', (1, ), maxshape=(None, ), dtype=np.int)
                dset[0] = 0

                dset = output.create_dataset('/sigd', (1, ), maxshape=(None, ), dtype=np.float)
                dset[0] = stress

                dset = output.create_dataset('/t', (1, ), maxshape=(None, ), dtype=np.float)
                dset[0] = float(data['/t'][inc])



