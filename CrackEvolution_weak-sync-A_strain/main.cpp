
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <cpppath.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xview.hpp>


#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif


namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;


#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }


class Main {

private:

    // input/output file
    H5Easy::File m_file;

    // mesh parameters
    xt::xtensor<size_t, 2> m_conn;
    xt::xtensor<double, 2> m_coor;
    xt::xtensor<size_t, 2> m_dofs;
    xt::xtensor<size_t, 1> m_iip;

    // mesh dimensions
    size_t m_nelem;
    size_t m_nne;
    size_t m_nnode;
    size_t m_ndim;
    size_t m_nip;

    // numerical quadrature
    QD::Quadrature m_quad;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector;

    // mass matrix
    GF::MatrixDiagonalPartitioned m_M;

    // damping matrix
    GF::MatrixDiagonal m_D;

    // material definition
    GM::Array<2> m_material;

    // convergence check
    GF::Iterate::StopList m_stop;

    // current time & time-step
    double m_t = 0.0;
    double m_dt;

    // event-driven settings
    size_t m_inc = 0;                 // current increment
    double m_deps_kick;               // equivalent strain increment
    xt::xtensor<size_t, 1> m_plastic; // plastic elements

    // nodal displacements, velocities, and accelerations (current and last time-step)
    xt::xtensor<double, 2> m_u;
    xt::xtensor<double, 2> m_v;
    xt::xtensor<double, 2> m_a;
    xt::xtensor<double, 2> m_v_n;
    xt::xtensor<double, 2> m_a_n;

    // element vectors
    xt::xtensor<double, 3> m_ue;
    xt::xtensor<double, 3> m_fe;

    // nodal forces
    xt::xtensor<double, 2> m_felas;
    xt::xtensor<double, 2> m_fdamp;
    xt::xtensor<double, 2> m_fint;
    xt::xtensor<double, 2> m_fext;
    xt::xtensor<double, 2> m_fres;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps;
    xt::xtensor<double, 4> m_Sig;

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadOnly)
    {
        readMesh();
        setMass();
        setDamping();
        setMaterial();
        readParameters();
        computeStrainStress();
    }

public:

    void readParameters()
    {
        m_dt = H5Easy::load<double>(m_file, "/run/dt");
        m_deps_kick = H5Easy::load<double>(m_file, "/run/epsd/kick");
        m_stop = GF::Iterate::StopList(20);
    }

public:

    void readMesh()
    {
        m_conn = H5Easy::load<decltype(m_conn)>(m_file, "/conn");
        m_coor = H5Easy::load<decltype(m_coor)>(m_file, "/coor");
        m_dofs = H5Easy::load<decltype(m_dofs)>(m_file, "/dofs");
        m_iip = H5Easy::load<decltype(m_iip)>(m_file, "/dofsP");

        m_nnode = m_coor.shape(0);
        m_ndim = m_coor.shape(1);
        m_nelem = m_conn.shape(0);
        m_nne = m_conn.shape(1);

        m_vector = GF::VectorPartitioned(m_conn, m_dofs, m_iip);

        m_quad = QD::Quadrature(m_vector.AsElement(m_coor));
        m_nip = m_quad.nip();

        m_u = xt::zeros<double>(m_coor.shape());
        m_v = xt::zeros<double>(m_coor.shape());
        m_a = xt::zeros<double>(m_coor.shape());
        m_v_n = xt::zeros<double>(m_coor.shape());
        m_a_n = xt::zeros<double>(m_coor.shape());

        m_ue = xt::zeros<double>({m_nelem, m_nne, m_ndim});
        m_fe = xt::zeros<double>({m_nelem, m_nne, m_ndim});

        m_felas = xt::zeros<double>(m_coor.shape());
        m_fdamp = xt::zeros<double>(m_coor.shape());
        m_fint = xt::zeros<double>(m_coor.shape());
        m_fext = xt::zeros<double>(m_coor.shape());
        m_fres = xt::zeros<double>(m_coor.shape());

        m_Eps = xt::zeros<double>({m_nelem, m_nip, m_ndim, m_ndim});
        m_Sig = xt::zeros<double>({m_nelem, m_nip, m_ndim, m_ndim});
    }

public:

    void setMass()
    {
        m_M = GF::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);

        auto x = m_vector.AsElement(m_coor);

        QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

        auto val_elem = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/rho");

        MYASSERT(val_elem.size() == m_nelem);

        xt::xtensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});

        for (size_t q = 0; q < nodalQuad.nip(); ++q) {
            xt::view(val_quad, xt::all(), q) = val_elem;
        }

        m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    }

public:

    void setDamping()
    {
        m_D = GF::MatrixDiagonal(m_conn, m_dofs);

        auto x = m_vector.AsElement(m_coor);

        QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

        auto val_elem = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/damping/alpha");

        MYASSERT(val_elem.size() == m_nelem);

        xt::xtensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});

        for (size_t q = 0; q < nodalQuad.nip(); ++q) {
            xt::view(val_quad, xt::all(), q) = val_elem;
        }

        m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    }

public:

    void setMaterial()
    {
        m_material = GM::Array<2>({m_nelem, m_nip});

        {
            auto elem = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/elastic/elem");
            auto k = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/K");
            auto g = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/G");

            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});

            xt::view(I, xt::keep(elem), xt::all()) = 1ul;

            for (size_t q = 0; q < m_nip; ++q) {
                xt::view(idx, xt::keep(elem), q) = xt::arange<size_t>(elem.size());
            }

            m_material.setElastic(I, idx, k, g);
        }

        {
            auto elem = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/cusp/elem");
            auto k = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/K");
            auto g = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/G");
            auto y = H5Easy::load<xt::xtensor<double, 2>>(m_file, "/cusp/epsy");

            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});

            xt::view(I, xt::keep(elem), xt::all()) = 1ul;

            for (size_t q = 0; q < m_nip; ++q) {
                xt::view(idx, xt::keep(elem), q) = xt::arange<size_t>(elem.size());
            }

            m_material.setCusp(I, idx, k, g, y);
        }

        xt::xtensor<double, 2> k = m_material.K();
        xt::xtensor<double, 2> g = m_material.G();
        MYASSERT(xt::mean(k)() == k(0, 0));
        MYASSERT(xt::mean(g)() == g(0, 0));

        m_material.check();

        m_plastic = xt::sort(xt::flatten_indices(xt::argwhere(xt::amin(m_material.isPlastic(), {1}))));
    }

public:

    void timeStep()
    {
        // history

        m_t += m_dt;

        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        // new displacement

        xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.) * m_a;

        // compute strain/strain, and corresponding force

        computeStrainStress();

        m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
        m_vector.assembleNode(m_fe, m_felas);

        // estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // re-estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);
    }

public:

    void computeStrainStress()
    {
        m_vector.asElement(m_u, m_ue);
        m_quad.symGradN_vector(m_ue, m_Eps);
        m_material.setStrain(m_Eps);
        m_material.stress(m_Sig);
    }

public:

    xt::xtensor<size_t, 1> getIncPush()
    {
        // integration point volume
        auto dV = m_quad.DV(2);

        // number of plastic cells
        size_t N = m_plastic.size();

        // basic information for each increment
        auto stored = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/stored");
        auto kick = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/kick");

        // allocate result
        xt::xtensor<size_t, 1> A = xt::zeros<size_t>({xt::amax(stored)[0] + 1});
        xt::xtensor<double, 1> epsd = xt::zeros<double>({xt::amax(stored)[0] + 1});
        xt::xtensor<double, 1> sigd = xt::zeros<double>({xt::amax(stored)[0] + 1});

        // index of the current quadratic potential,
        // for the first integration point per plastic element
        auto idx_n = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);

        // loop over increments
        for (size_t istored = 0; istored < stored.size(); ++istored) {
            // - get increment number
            size_t inc = stored(istored);

            // - restore displacement
            xt::noalias(m_u) = H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", inc));

            // - update strain/strain
            computeStrainStress();

            // - index of the current quadratic potential
            auto idx = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);

            // - macroscopic strain/stress tensor
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsbar = xt::average(m_Eps, dV, {0, 1});
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Sigbar = xt::average(m_Sig, dV, {0, 1});

            // - macroscopic equivalent strain/stress
            epsd(inc) = GM::Epsd(Epsbar)();
            sigd(inc) = GM::Sigd(Sigbar)();

            // - avalanche area
            A(inc) = xt::sum(xt::not_equal(idx, idx_n))[0];

            // - update history
            idx_n = idx;
        }

        // determine increment at which the steady-state starts, by considering the elasto-plastic
        // tangent (stress/strain), w.r.t. the elastic tangent (shear modulus) some 'engineering
        // quantity' is use, which was checked visually
        // - initialise elasto-plastic tangent of each increment
        xt::xtensor<double, 1> K = xt::zeros<size_t>(sigd.shape());
        size_t steadystate = 0;
        // - compute
        for (size_t i = 1; i < K.size(); ++i) {
            K(i) = (sigd(i) - sigd(0)) / (epsd(i) - epsd(0));
        }
        // - set dummy (to take the minimum below)
        K(0) = K(1);
        // - get steady-state increment
        if (xt::any(K <= 0.95 * K(1))) {
            //  - select increment
            steadystate = xt::amin(xt::from_indices(xt::argwhere(K <= 0.95 * K(1))))[0];
            // - make sure to skip at least two increments (artificial: to avoid checks below)
            steadystate = std::max(steadystate, std::size_t(2));
            // - always start the steady-state by elastic loading
            if (kick(steadystate)) {
                steadystate += 1;
            }
        }

        // remove all non-steady-state increments from further consideration
        xt::view(A, xt::range(0, steadystate)) = 0;

        // list with increment of system-spanning avalanches
        return xt::flatten_indices(xt::argwhere(xt::equal(A, N)));
    }

public:

    void triggerElement(size_t element)
    {
        // convert plastic-element to element number
        size_t e = m_plastic(element);

        // make sure to start from quenched state
        m_v.fill(0.0);
        m_a.fill(0.0);
        m_stop.reset();

        // current equivalent deviatoric strain
        auto eps = GM::Epsd(m_Eps);

        // distance to yielding on the positive side
        auto epsy = m_material.CurrentYieldRight();
        auto deps = eps - epsy;

        // find integration point closest to yielding
        // - isolate element
        xt::xtensor<double, 1> deps_e = xt::view(deps, e, xt::all());
        // - get integration point
        auto q = xt::argmin(xt::abs(deps_e))[0];

        // extract (equivalent) deviatoric strain at quadrature-point "(e,q)"
        auto Epsd = xt::view(GM::Deviatoric(m_Eps), e, q);

        // new equivalent deviatoric strain: yield strain + small strain kick
        double eps_new = epsy(e, q) + m_deps_kick / 2.;

        // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
        double dgamma =
            2.0 * (-Epsd(0, 1) + std::sqrt(std::pow(eps_new, 2.0) - std::pow(Epsd(0, 0), 2.0)));

        // apply increment in shear strain as a perturbation to the selected element
        // - nodes belonging to the current element, from connectivity
        auto nodes = xt::view(m_conn, e, xt::all());
        // - displacement-DOFs
        xt::xtensor<double, 1> udofs = m_vector.AsDofs(m_u);
        // - update displacement-DOFs for the element
        for (size_t n = 0; n < m_nne; ++n) {
            udofs(m_dofs(nodes(n), 0)) += dgamma * (m_coor(nodes(n), 1) - m_coor(nodes(0), 1));
        }
        // - convert displacement-DOFs to (periodic) nodal displacement vector
        //   (N.B. storing to nodes directly does not ensure periodicity)
        m_vector.asNode(udofs, m_u);

        // compute strain/stress
        computeStrainStress();
    }

public:

    void run(size_t element, size_t inc_c, const std::string& output, size_t A_step)
    {
        // extract a list with increments at which to start elastic loading
        xt::xtensor<size_t, 1> inc_system = getIncPush();
        MYASSERT(inc_system.size() > 0);
        MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

        // set increment
        m_inc = inc_c;

        // restore displacement
        xt::noalias(m_u) = H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc));
        computeStrainStress();

        // extract "id" from filename (stored to data)
        std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
        size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

        // extract information needed for storage
        size_t N = m_plastic.size();
        auto dV = m_quad.DV(2);
        xt::xtensor<int, 1> idx_n = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);
        xt::xtensor<int, 1> idx = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);

        // perturb the displacement of the set element, to (try to) trigger an avalanche
        triggerElement(element);

        // clear/open the output file
        H5Easy::File data(output, H5Easy::File::Overwrite);

        // storage parameters
        size_t A = 0;
        size_t A_next = 0;
        size_t A_istore = 0;
        bool A_store = true;

        // quench: force equilibrium
        for (size_t iiter = 0;; ++iiter) {

            if (A_store) {

                if (iiter > 0) {
                    idx = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);
                }

                size_t a = xt::sum(xt::not_equal(idx, idx_n))[0];
                A = std::max(A, a);

                if ((A >= A_next && A % A_step == 0) || A == N || iiter == 0) {

                    fmt::print("{0:s}: Saving, sync-A, A = {1:d}\n", m_file.getName(), A);

                    auto jdx = m_material.CurrentIndex();
                    auto epsp = m_material.Epsp();
                    auto epsy_p = m_material.CurrentYieldRight();
                    auto epseq = GM::Epsd(m_Eps);
                    auto x = epsy_p - epseq;

                    xt::xtensor<double, 1> epsp_store = xt::view(epsp, xt::keep(m_plastic), 0);
                    xt::xtensor<size_t, 1> jdx_store = xt::view(jdx, xt::keep(m_plastic), 0);
                    xt::xtensor<double, 1> x_store = xt::view(x, xt::keep(m_plastic), 0);

                    if (iiter == 0) {
                        jdx_store = idx_n;
                    }

                    xt::xtensor<double, 3> Sig_elem = xt::average(m_Sig, dV, {1});
                    xt::xtensor<double, 1> Sig_plas_xx = xt::view(Sig_elem, xt::keep(m_plastic), 0, 0);
                    xt::xtensor<double, 1> Sig_plas_xy = xt::view(Sig_elem, xt::keep(m_plastic), 0, 1);
                    xt::xtensor<double, 1> Sig_plas_yy = xt::view(Sig_elem, xt::keep(m_plastic), 1, 1);

                    xt::xtensor_fixed<double, xt::xshape<2, 2>> Sig_bar = xt::average(m_Sig, dV, {0, 1});

                    H5Easy::dump(data, "/sync-A/stored", A, {A_istore});
                    H5Easy::dump(data, "/sync-A/global/iiter", iiter, {A});
                    H5Easy::dump(data, "/sync-A/global/sig_xx", Sig_bar(0, 0), {A});
                    H5Easy::dump(data, "/sync-A/global/sig_xy", Sig_bar(0, 1), {A});
                    H5Easy::dump(data, "/sync-A/global/sig_yy", Sig_bar(1, 1), {A});
                    H5Easy::dump(data, fmt::format("/sync-A/plastic/{0:d}/sig_xx", A), Sig_plas_xx);
                    H5Easy::dump(data, fmt::format("/sync-A/plastic/{0:d}/sig_xy", A), Sig_plas_xy);
                    H5Easy::dump(data, fmt::format("/sync-A/plastic/{0:d}/sig_yy", A), Sig_plas_yy);
                    H5Easy::dump(data, fmt::format("/sync-A/plastic/{0:d}/idx", A), jdx_store);

                    ++A_istore;
                    A_next = A + A_step;
                }

                if (A == N) {
                    A_store = false;
                }
            }

            timeStep();

            if (m_stop.stop(xt::linalg::norm(m_fres) / xt::linalg::norm(m_fext), 1.e-5)) {
                break;
            }

            if (A == N) {
                break;
            }
        }

        std::string hash = GIT_COMMIT_HASH;
        H5Easy::dump(data, "/git/run", hash);

        H5Easy::dump(data, "/meta/completed", 1);
        H5Easy::dump(data, "/meta/uuid", H5Easy::load<std::string>(m_file, "/uuid"));
        H5Easy::dump(data, "/meta/id", id_num);
        H5Easy::dump(data, "/meta/inc_c", inc_c);
        H5Easy::dump(data, "/meta/element", element);
        H5Easy::dump(data, "/meta/dt", m_dt);
        H5Easy::dump(data, "/meta/plastic", m_plastic);
    }

};


static const char USAGE[] =
    R"(Run
  Extract time evolution of a specific push.

Usage:
  Run [options] --output=N --element=N --file=N --incc=N

Arguments:
      --output=N      Path of the output file.
      --element=N     Element to push.
      --file=N        The path to the simulation file.
      --incc=N        Increment number of the system-spanning avalanche.

Options:
      --Astep=N       Save states at crack sizes A = (0: N: Astep). [default: 1]
  -h, --help          Show help.
      --version       Show version.

(c) Tom de Geus
)";


int main(int argc, const char** argv)
{
    std::string hash = GIT_COMMIT_HASH;

    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, hash);

    std::string output = args["--output"].asString();
    std::string file = args["--file"].asString();
    size_t inc_c = static_cast<size_t>(std::stoi(args["--incc"].asString()));
    size_t element = static_cast<size_t>(std::stoi(args["--element"].asString()));
    size_t A_step = static_cast<size_t>(std::stoi(args["--Astep"].asString()));

    Main sim(file);

    sim.run(element, inc_c, output, A_step);

    return 0;
}
