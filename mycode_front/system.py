import numpy as np
import FrictionQPotFEM.UniformSingleLayer2d as model
import h5py
import GooseFEM
import uuid

def initsystem(data: h5py.File) -> model.System:
    r"""
    Read system from file.

    :param data: Open simulation HDF5 archive (read-only).
    :return: The initialised system.
    """

    system = model.System(
        data["/coor"][...],
        data["/conn"][...],
        data["/dofs"][...],
        data["/dofsP"][...],
        data["/elastic/elem"][...],
        data["/cusp/elem"][...],
    )

    system.setMassMatrix(data["/rho"][...])
    system.setDampingMatrix(data["/damping/alpha"][...])
    system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
    system.setPlastic(
        data["/cusp/K"][...], data["/cusp/G"][...], data["/cusp/epsy"][...]
    )
    system.setDt(data["/run/dt"][...])

    return system


def reset_epsy(system: model.System, data: h5py.File):
    r"""
    Reset yield strain history from file.
    This can for example be used to speed-up things by avoiding re-initialising the system.

    :param system: The system (modified: yield strains changed).
    :param data: Open simulation HDF5 archive (read-only).
    """

    e = data["/cusp/epsy"][...]
    epsy = np.empty((e.shape[0], e.shape[1] + 1), dtype=e.dtype)
    epsy[:, 0] = -e[:, 0]
    epsy[:, 1:] = e

    plastic = system.plastic()
    N = plastic.size
    nip = system.quad().nip()
    material = system.material()
    material_plastic = system.material_plastic()

    assert epsy.shape[0] == N

    for i, e in enumerate(plastic):
        for q in range(nip):
            for cusp in [
                material.refCusp([e, q]),
                material_plastic.refCusp([i, q]),
            ]:
                chunk = cusp.refQPotChunked()
                chunk.set_y(epsy[i, :])


def generate(filename: str, N: int):
    """
    Generate input file.

    :param filename: The filename of the input file (overwritten).
    :param N: The number of blocks.
    """

    # parameters
    h = np.pi
    L = h * float(N)

    # define mesh and element sets
    mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)
    nelem = mesh.nelem()
    plastic = mesh.elementsMiddleLayer()
    elastic = np.setdiff1d(np.arange(nelem), plastic)

    # extract node sets to set the boundary conditions
    ndim = mesh.ndim()
    top = mesh.nodesTopEdge()
    bottom = mesh.nodesBottomEdge()
    left = mesh.nodesLeftOpenEdge()
    right = mesh.nodesRightOpenEdge()
    nleft = len(left)
    ntop = len(top)

    # initialize DOF numbers
    dofs = mesh.dofs()

    # periodicity in horizontal direction : eliminate 'dependent' DOFs
    for i in range(nleft):
        for j in range(ndim):
            dofs[right[i], j] = dofs[left[i], j]

    # renumber "dofs" to be sequential
    dofs = GooseFEM.Mesh.renumber(dofs)

    # construct list with prescribed DOFs
    # - allocate
    fixedDofs = np.empty((2 * ntop * ndim), dtype="int")
    # - set DOFs
    for i in range(ntop):
        for j in range(ndim):
            fixedDofs[i * ndim + j] = dofs[bottom[i], j]
            fixedDofs[i * ndim + j + ntop * ndim] = dofs[top[i], j]

    # yield strains
    k = 2.0
    realization = str(uuid.uuid4())
    epsy = 1.0e-5 + 1.0e-3 * np.random.weibull(k, size=1000 * len(plastic)).reshape(
        len(plastic), -1
    )
    epsy[:, 0] = 1.0e-5 + 1.0e-3 * np.random.random(len(plastic))
    epsy = np.cumsum(epsy, axis=1)
    idx = np.min(np.where(np.min(epsy, axis=0) > 0.55)[0])
    epsy = epsy[:, :idx]

    # parameters
    c = 1.0
    G = 1.0
    K = 10.0 * G
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho

    # time step
    dt = 1.0 / (c * qh)
    dt /= 10.0

    with h5py.File(filename, "w") as file:

        file["/coor"] = mesh.coor()
        file["/conn"] = mesh.conn()
        file["/dofs"] = dofs
        file["/iip"] = fixedDofs
        file["/run/epsd/max"] = 0.5
        file["/run/epsd/kick"] = 1.0e-7
        file["/run/dt"] = dt
        file["/rho"] = rho * np.ones(nelem)
        file["alpha"] = alpha * np.ones(nelem)
        file["/cusp/elem"] = plastic
        file["/cusp/K"] = K * np.ones(len(plastic))
        file["/cusp/G"] = G * np.ones(len(plastic))
        file["/cusp/epsy"] = epsy
        file["/elastic/elem"] = elastic
        file["/elastic/K"] = K * np.ones(len(elastic))
        file["/elastic/G"] = G * np.ones(len(elastic))
        file["/uuid"] = realization


def pushincrements(
    system: model.System, data: h5py.File, target_stress: float
) -> (np.ndarray, np.ndarray):
    r"""
    Get a list of increment from which the stress can be reached by elastic loading only.

    :param system: The system (modified: all increments visited).
    :param data: Open simulation HDF5 archive (read-only).
    :param target_stress: The stress at which to push (in real units).
    :return:
        ``inc_system`` List of system spanning avalanches.
        ``inc_push`` List of increment from which the stress can be reached by elastic loading only.
    """

    dV = system.quad().AsTensor(2, system.quad().dV())
    kick = data["/kick"][...].astype(bool)
    incs = data["/stored"][...].astype(int)
    assert np.all(incs == np.arange(incs.size))
    assert kick.shape == incs.shape
    assert np.all(not kick[::2])
    assert np.all(kick[1::2])

    A = np.zeros(incs.shape, dtype=int)
    Strain = np.zeros(incs.shape, dtype=float)
    Stress = np.zeros(incs.shape, dtype=float)

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)

    for inc in incs:

        system.setU(data[f"/disp/{inc:d}"])

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
        ii = inc_system[i] + 1
        jj = inc_system[i + 1]
        s = Stress[ii:jj:2]
        n = incs[ii:jj:2]

        if not np.any(s > target_stress):
            continue

        j = np.argmax(s > target_stress)
        ipush = n[j] - 1

        assert Stress[ipush] <= target_stress
        assert not kick[ipush + 1]

        inc_push += [ipush]
        inc_system_ret += [n[0] - 1]

    inc_push = np.array(inc_push)
    inc_system_ret = np.array(inc_system_ret)

    return inc_system_ret, inc_push
