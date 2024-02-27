import itertools
import pathlib
import re
import shutil
import tempfile

import enstat
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import path
import pytest
import shelephant

from mycode_front import Dynamics
from mycode_front import EventMap
from mycode_front import QuasiStatic
from mycode_front import storage
from mycode_front import tools
from mycode_front import Trigger


@pytest.fixture(scope="module")
def mydata():
    """
    *   Generate a temporary directory.
    *   If all tests are finished: remove temporary directory.
    """
    tmpDir = tempfile.TemporaryDirectory()
    tmp_dir = pathlib.Path(tmpDir.name)
    paths = {
        "quasistatic": {
            "dirname": tmp_dir / "QuasiStatic",
            "realisation": tmp_dir / "QuasiStatic" / "id=0000.h5",
            "info": tmp_dir / "QuasiStatic" / "EnsembleInfo.h5",
        },
        "trigger": {
            "dirname": tmp_dir / "Trigger",
        },
    }
    for ensemble in paths:
        paths[ensemble]["dirname"].mkdir()

    # QuasiStatic

    QuasiStatic.generate(paths["quasistatic"]["realisation"], N=9, dev=True)

    with h5py.File(paths["quasistatic"]["realisation"], "a") as file:
        file["/param/cusp/epsy/nchunk"][...] = 200

    QuasiStatic.cli_run(["--dev", paths["quasistatic"]["realisation"]])
    QuasiStatic.cli_ensembleinfo(
        ["--dev", "-o", paths["quasistatic"]["info"], paths["quasistatic"]["realisation"]]
    )

    # Trigger

    n = 5
    opts = [
        ["--dev", "-f"],
        ["-d", 0.12],
        ["-p", 2],
        ["-o", paths["trigger"]["dirname"]],
        ["--nmax", n],
        [paths["quasistatic"]["info"]],
    ]
    opts = list(itertools.chain(*opts))
    commands = Trigger.cli_job_deltasigma(opts)
    paths["trigger"]["opts"] = opts
    paths["trigger"]["commands"] = commands

    # Trigger: Check that copying worked

    elem = [Trigger.interpret_filename(i)["element"] for i in paths["trigger"]["commands"]]
    elem = np.unique(elem)
    o = [i.split(" ")[-1] for i in paths["trigger"]["commands"]]
    uni = [re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(elem[0]) + r"\4", i) for i in o]
    uni = [str(i) for i in np.unique(uni)]

    for e in elem[1:]:
        for src in uni:
            dst = re.sub(r"(.*)(element)(=[0-9]*)(.*)", r"\1\2=" + str(e) + r"\4", src)
            with h5py.File(src) as s, h5py.File(dst) as d:
                res = g5.compare(s, d)
            assert res["!="] == ["/Trigger/try_element"]
            assert res["->"] == []
            assert res["<-"] == []
            with h5py.File(src) as file:
                assert elem[0] == file["/Trigger/try_element"][1]
            with h5py.File(dst) as file:
                assert e == file["/Trigger/try_element"][1]
                assert -1 == file["/Trigger/element"][1]

    # Trigger: Run

    paths["trigger"]["files"] = []
    for command in paths["trigger"]["commands"]:
        paths["trigger"]["files"].append(Trigger.cli_run(command.split(" ")[1:]))

    paths["trigger"]["pack"] = paths["trigger"]["dirname"] / "ensemblepack.h5"
    paths["trigger"]["info"] = paths["trigger"]["dirname"] / "ensembleinfo.h5"

    Trigger.cli_ensemblepack(["-f", "-o", paths["trigger"]["pack"], *paths["trigger"]["files"]])
    Trigger.cli_ensembleinfo(
        ["--dev", "-f", "-o", paths["trigger"]["info"], paths["trigger"]["pack"]]
    )

    # return

    yield paths

    # cleanup

    tmpDir.cleanup()


def test_quasistatic_historic(mydata):
    """
    Generate + run + check historic output
    """
    historic = shelephant.yaml.read(pathlib.Path(__file__).parent / "data_System_small.yaml")

    with h5py.File(mydata["quasistatic"]["info"]) as file:
        idname = str(mydata["quasistatic"]["realisation"].name)
        epsd = file[f"/full/{idname}/eps"][...]
        sigd = file[f"/full/{idname}/sig"][...]

    ref_eps = historic["epsd"][3:]
    ref_sig = historic["sigd"][3:]

    assert np.allclose(epsd[1:][: len(ref_eps)], ref_eps)
    assert np.allclose(sigd[1:][: len(ref_sig)], ref_sig)


def test_quasistatic_generate(tmp_path):
    """
    Generate a simulation (not a unittest)
    """
    with shelephant.path.cwd(tmp_path):
        QuasiStatic.cli_generate(["--dev", "mygen"])


def test_quasistatic_meta_move(mydata, tmp_path):
    """
    Move meta-data.
    """
    fname = tmp_path / "mycopy.h5"
    shutil.copy(mydata["quasistatic"]["realisation"], fname)
    old = f"/meta/{QuasiStatic.entry_points['cli_run']}"
    new = f"{old}_new"
    QuasiStatic.cli_move_meta(["--dev", old, new, fname])
    with h5py.File(fname) as file:
        assert file[old].attrs["version"] == file[new].attrs["version"]


def test_quasistatic_status(mydata):
    ret = QuasiStatic.cli_status(
        ["-k", f"/meta/{QuasiStatic.entry_points['cli_run']}", mydata["quasistatic"]["realisation"]]
    )
    assert ret == {
        "completed": [str(mydata["quasistatic"]["realisation"])],
        "new": [],
        "partial": [],
    }


def test_quasistatic_interface_state(mydata):
    """
    Check interface state (not a unittest)
    """
    with h5py.File(mydata["quasistatic"]["info"]) as file:
        idname = str(mydata["quasistatic"]["realisation"].name)
        steps = file[f"/full/{idname}/step"][...]

    QuasiStatic.interface_state({mydata["quasistatic"]["realisation"]: steps[-2:]})


def test_quasistatic_branch_fixed_stress(mydata, tmp_path):
    """
    Branch fixed stress.
    """
    with h5py.File(mydata["quasistatic"]["info"]) as file:
        idname = str(mydata["quasistatic"]["realisation"].name)
        A = file[f"/full/{idname}/A"][...]
        N = file["/normalisation/N"][...]
        sig_bot = file["/averages/sig_bottom"][...]
        sig_top = file["/averages/sig_top"][...]

    step_c = np.argwhere(A == N).ravel()[-2]
    stress = (sig_bot + sig_top) / 2
    branchpath = tmp_path / "branch_fixed_stress.h5"

    with h5py.File(mydata["quasistatic"]["realisation"]) as src, h5py.File(branchpath, "w") as dest:
        QuasiStatic.branch_fixed_stress(
            src, dest, "foo", step_c=step_c, stress=stress, normalised=True, dev=True
        )
        system = QuasiStatic.System(dest)
        out = QuasiStatic.basic_output(system, dest, "foo")
        assert np.isclose(out["sig"][0], stress)


def test_quasistatic_rerun_dynamics(mydata, tmp_path):
    """
    Rerun a quasistatic step with Dynamics, check the macroscopic stress.
    """

    commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(
        ["-f", "-o", tmp_path, mydata["quasistatic"]["info"]]
    )
    _, _, outpath, _, step, inpath = commands[0].split(" ")
    step = int(step)
    inpath = pathlib.Path(inpath).name

    with h5py.File(mydata["quasistatic"]["info"]) as file:
        sig0 = file["/normalisation/sig0"][...]
        check = file[f"/full/{inpath}/sig"][step]

    with path.Path(tmp_path):
        Dynamics.cli_run(commands[0].split(" ")[1:] + ["--dev"])

        with h5py.File(outpath) as file:
            sig = GMat.Sigd(storage.symtens2_read(file, "/Dynamics/Sigbar")[-1, ...])
            assert np.isclose(sig / sig0, check)

        Dynamics.cli_average_systemspanning([outpath, "-o", tmp_path / "average.h5", "--dev"])


def test_quasistatic_rerun_dynamics_run_highfrequency(mydata, tmp_path):
    """
    Rerun a quasistatic step with Dynamics, check the macroscopic stress.
    """
    commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(
        ["-f", "-o", tmp_path, mydata["quasistatic"]["info"]]
    )
    _, _, outpath, _, step, inpath = commands[0].split(" ")
    step = int(step)

    with h5py.File(tmp_path / inpath) as file:
        system = QuasiStatic.System(file)
        system.restore_quasistatic_step(file["QuasiStatic"], step - 1)
        check = np.average(system.Sig(), weights=system.dV(rank=2), axis=(0, 1))[0, 1]

    with path.Path(tmp_path):
        Dynamics.cli_run_highfrequency(commands[0].split(" ")[1:] + ["--dev"])

        with h5py.File(outpath) as file:
            sig = file["/DynamicsHighFrequency/fext"][0]
            assert np.isclose(sig, check, rtol=1e-3)


def test_quasistatic_rerun_eventmap(mydata, tmp_path):
    """
    Rerun a quasistatic step with EventMap, check the event size
    """
    commands = QuasiStatic.cli_rerun_dynamics_job_systemspanning(
        ["-f", "-o", tmp_path, mydata["quasistatic"]["info"]]
    )
    _, _, outpath, _, step, inpath = commands[0].split(" ")
    step = int(step)
    inpath = pathlib.Path(inpath).name

    with h5py.File(mydata["quasistatic"]["info"]) as file:
        check = file[f"/full/{inpath}/S"][step]

    with path.Path(tmp_path):
        EventMap.cli_run(commands[0].split(" ")[1:] + ["--dev"])

        with h5py.File(outpath) as file:
            S = file["S"][...]
            assert np.sum(S) == check


def test_quasistatic_state_after_systemspanning(mydata, tmp_path):
    """
    Read the state of systemspanning events (not a unittest)
    """
    QuasiStatic.cli_state_after_systemspanning(
        ["-f", "-o", tmp_path / "state", mydata["quasistatic"]["info"]]
    )


def test_trigger_ensemblepack(mydata, tmp_path):
    """
    Pack the ensemble. Try regenerating one realisation, and read the output.
    """
    t1 = mydata["trigger"]["pack"]
    t2 = tmp_path / "ensemblepack_t.h5"
    p1 = tmp_path / "ensemblepack_a.h5"
    p2 = tmp_path / "ensemblepack_b.h5"

    Trigger.cli_ensemblepack(["-f", "-o", p1, *mydata["trigger"]["files"][:2]])
    Trigger.cli_ensemblepack(["-f", "-o", p2, *mydata["trigger"]["files"][2:]])
    Trigger.cli_ensemblepack_merge(["-f", "-o", t2, p1, p2])
    assert np.abs(t1.stat().st_size - t2.stat().st_size) / t1.stat().st_size < 1e-2

    with h5py.File(t1) as file, h5py.File(t2) as dest:
        res = g5.compare(file, dest)

    for key in res:
        if key != "==":
            assert res[key] == []


def test_trigger_filter(mydata, tmp_path):
    """
    Pack the ensemble, check filtering new generation.
    """

    extra = Trigger.cli_job_deltasigma(
        mydata["trigger"]["opts"] + ["--filter", mydata["trigger"]["pack"]]
    )
    assert not np.any(np.in1d(extra, mydata["trigger"]["commands"]))


def test_trigger_ensembleinfo(mydata, tmp_path):
    """
    Pack the ensemble. Try regenerating one realisation, and read the output.
    """

    clonename = tmp_path / "myclone.h5"
    with h5py.File(mydata["trigger"]["info"]) as file:
        Trigger.restore_from_ensembleinfo(
            ensembleinfo=file,
            index=0,
            destpath=clonename,
            sourcedir=mydata["quasistatic"]["dirname"],
            dev=True,
        )

    Trigger.cli_run(["--dev", "--rerun", clonename])

    with h5py.File(mydata["trigger"]["files"][0]) as source, h5py.File(clonename) as dest:
        paths = list(g5.getdatapaths(source))

        for key in [
            "/meta/Trigger_JobDeltaSigma",
            "/meta/Trigger_Run",
            "/meta/branch_fixed_stress",
        ]:
            if key in paths:
                paths.remove(key)

        assert g5.allequal(source, dest, paths)


def test_trigger_rerun_dynamics_ensemble(mydata, tmp_path):
    """
    Rerun one trigger with Dynamics, check the macroscopic stress.
    """

    opts = [
        ["--dev"],
        ["-f"],
        ["--test"],
        ["-n", 2],
        ["--height", 2],
        ["-s", mydata["quasistatic"]["dirname"]],
        ["-o", tmp_path],
        [mydata["trigger"]["info"]],
    ]
    commands = Trigger.cli_job_rerun_dynamics(list(itertools.chain(*opts)))

    _, _, outpath, _, step, _, _, inpath = commands[0].split(" ")
    step = int(step)

    with h5py.File(inpath) as file:
        u = file["/Trigger/u/0"][...]

    # todo: make filename automatic
    fname = mydata["trigger"]["dirname"] / "deltasigma=0,00000_id=0000_stepc=11_element=0.h5"
    with h5py.File(fname) as file:
        trigger = QuasiStatic.System(file)

        trigger.restore_quasistatic_step(file["Trigger"], 0)
        trigger_i_n = np.copy(trigger.plastic.i)
        if not np.allclose(u, trigger.u):
            print(u)
            print(trigger.u)
        assert np.allclose(u, trigger.u)

        trigger.restore_quasistatic_step(file["Trigger"], 1)

    with path.Path(tmp_path):
        Dynamics.cli_run(commands[0].split(" ")[1:] + ["--dev"])

        with h5py.File(outpath) as file:
            root = file["Dynamics"]
            dynamics = QuasiStatic.System(file)

            udof = np.zeros(dynamics.vector.shape_dofval)
            udof[root["doflist"][...]] = root["u"]["0"][...]
            dynamics.u = dynamics.vector.AsNode(udof)
            i_n = np.copy(dynamics.plastic.i)

            udof = np.zeros(dynamics.vector.shape_dofval)
            udof[root["doflist"][...]] = root["u"][str(file["/Dynamics/inc"].size - 1)][...]
            dynamics.u = dynamics.vector.AsNode(udof)

    assert np.all(np.equal(trigger_i_n, i_n))
    assert np.all(np.equal(trigger.plastic.i, dynamics.plastic.i))


def test_trigger_rerun_dynamics(mydata, tmp_path):
    """
    Rerun one trigger with Dynamics, check the macroscopic stress.
    """
    commands = Trigger.cli_job_deltasigma(
        [
            "--dev",
            "-f",
            mydata["quasistatic"]["info"],
            "-d",
            0.12,
            "-p",
            2,
            "-o",
            tmp_path,
            "--nmax",
            10,
        ]
    )
    triggername = Trigger.cli_run(["--dev"] + commands[-1].split(" ")[1:])
    triggerpack = tmp_path / "tmp_Pack.h5"
    triggerinfo = tmp_path / "tmp_Info.h5"
    Trigger.cli_ensemblepack(["-f", "-o", triggerpack, triggername])
    Trigger.cli_ensembleinfo(["--dev", "-f", "-o", triggerinfo, triggerpack])

    with h5py.File(triggerinfo) as file:
        A = file["A"][0]

    outname = tmp_path / "rerun_trigger.h5"
    Dynamics.cli_run(["--dev", "-f", "--step", 1, "-o", outname, triggername])

    with h5py.File(outname) as file:
        a = file["/Dynamics/A"][-1]
        iiter = file["/Dynamics/inc"].size - 1
        ustore = file[f"/Dynamics/u/{iiter:d}"][...]
        doflist = file["/Dynamics/doflist"][...]

    with h5py.File(triggername) as file:
        system = QuasiStatic.System(file)
        u = file["/Trigger/u/1"][...]
        system.u = u
        Sig = system.plastic.Sig / system.sig0
        Eps = system.plastic.Eps / system.eps0
        Epsp = system.plastic.epsp / system.eps0

    udof = np.zeros(system.vector.shape_dofval)
    udof[doflist] = ustore
    system.u = system.vector.AsNode(udof)

    assert np.allclose(Sig, system.plastic.Sig / system.sig0)
    assert np.allclose(Eps, system.plastic.Eps / system.eps0)
    assert np.allclose(Epsp, system.plastic.epsp / system.eps0)
    assert A == a


def test_trigger_rerun_eventmap(mydata, tmp_path):
    """
    Rerun one trigger with EventMap, check the macroscopic stress.
    """
    commands = Trigger.cli_job_rerun_eventmap(
        [
            "--dev",
            "-f",
            "--amin",
            "2",
            "-n",
            "2",
            "-s",
            mydata["quasistatic"]["dirname"],
            "-o",
            tmp_path,
            mydata["trigger"]["info"],
        ]
    )
    outpath = commands[0].split(" ")[2]
    EventMap.cli_run(["--dev", "-f"] + commands[0].split(" ")[1:])

    # todo: make filename automatic
    with h5py.File(
        mydata["trigger"]["dirname"] / "deltasigma=0,00000_id=0000_stepc=11_element=5.h5"
    ) as file:
        system = QuasiStatic.System(file)
        system.restore_quasistatic_step(file["Trigger"], 0)
        i_n = np.copy(system.plastic.i).astype(np.int64)
        system.restore_quasistatic_step(file["Trigger"], 1)
        check = np.sum((system.plastic.i.astype(np.int64) - i_n)[:, 0])

    with h5py.File(outpath) as file:
        S = file["S"][...]

    assert np.sum(S) == check


def test_dynamics_elements_at_height():
    """
    Identify elements at a certain height above the weak layer.
    """

    N = 3**3
    mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N)
    conn = mesh.conn
    coor = mesh.coor
    e = mesh.elementsMiddleLayer

    assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 0), e))
    assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 1), e + N))
    assert np.all(np.equal(Dynamics.elements_at_height(coor, conn, 2), e + N * 2))


def test_dynamics_partial_storage(mydata, tmp_path):
    """
    Store the displacement filed only for a selection of elements.
    """
    with h5py.File(mydata["quasistatic"]["info"]) as file:
        idname = str(mydata["quasistatic"]["realisation"].name)
        A = file[f"/full/{idname}/A"][...]
        N = file["/normalisation/N"][...]

    step = np.argwhere(A == N).ravel()[-1]

    with h5py.File(mydata["quasistatic"]["realisation"]) as file:
        system = QuasiStatic.System(file)
        u = file[f"/QuasiStatic/u/{step:d}"][...]

    plastic = system.plastic_elem
    system.u = u
    Sig = system.Sig() / system.sig0
    Sig_p = Sig[plastic, ...]

    vector = system.vector
    partial = tools.PartialDisplacement(
        conn=system.conn,
        dofs=system.dofs,
        element_list=plastic,
    )
    dofstore = partial.dof_is_stored()
    doflist = partial.dof_list()

    ustore = vector.AsDofs(u)[dofstore]
    udof = np.zeros(vector.shape_dofval)
    udof[doflist] = ustore
    system.u = vector.AsNode(udof)

    assert not np.allclose(Sig, system.Sig() / system.sig0)
    assert np.allclose(Sig_p, system.Sig()[plastic, ...] / system.sig0)


def test_dynamics_AlignedAverage():
    N = 10
    nip = 4
    elem = np.arange(N)
    nitem = 10
    V = np.random.random((N, nip, 2, 2))
    D = np.random.random((nitem, N + 1, N, nip, 2, 2))
    M = np.random.random((nitem, N + 1, N)) < 0.5

    av = Dynamics.AlignedAverage(shape=[N + 1, N, 2, 2], elements=elem, dV=V)
    check_00 = [enstat.static(shape=[N]) for i in range(N + 1)]
    check_01 = [enstat.static(shape=[N]) for i in range(N + 1)]
    check_10 = [enstat.static(shape=[N]) for i in range(N + 1)]
    check_11 = [enstat.static(shape=[N]) for i in range(N + 1)]

    for i in range(nitem):
        for a in range(N + 1):
            av.add_subsample(i, D[i, a, ...], roll=0, broken=~M[i, a, ...])
            d = np.average(D[i, a, ...], weights=V, axis=1)
            check_00[i].add_sample(d[..., 0, 0], mask=M[i, a, ...])
            check_01[i].add_sample(d[..., 0, 1], mask=M[i, a, ...])
            check_10[i].add_sample(d[..., 1, 0], mask=M[i, a, ...])
            check_11[i].add_sample(d[..., 1, 1], mask=M[i, a, ...])

    res = np.empty([N + 1, N, 2, 2])
    for a in range(N + 1):
        res[a, :, 0, 0] = check_00[a].mean()
        res[a, :, 0, 1] = check_01[a].mean()
        res[a, :, 1, 0] = check_10[a].mean()
        res[a, :, 1, 1] = check_11[a].mean()

    assert np.allclose(av.mean(), res, equal_nan=True)
