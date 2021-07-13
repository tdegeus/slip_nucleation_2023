from setuptools_scm import get_version
import argparse
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import prrng
import tqdm


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


with h5py.File("id=000.hdf5", "r") as data:

    system = initsystem(data)


    data["/run/epsd/kick"][...]
    N = system.N()

