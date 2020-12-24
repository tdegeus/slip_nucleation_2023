
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
    GF::Iterate::StopList m_stop = GF::Iterate::StopList(20);

    // time evolution
    double m_t = 0.0;   // current time
    double m_dt;        // time step

    // event-driven settings
    size_t m_inc = 0;                 // current increment
    size_t m_iiter = 0;               // current iteration
    int m_kick = 1;                   // yes/no kick with "m_deps_kick"; no: use distance to yielding
    double m_deps_kick;               // equivalent strain increment
    double m_epsd_max;                // maximum equivalent strain
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

    // parameters
    double m_G; // shear modulus (homogeneous)

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadWrite)
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
        m_epsd_max = H5Easy::load<double>(m_file, "/run/epsd/max");
        m_stop = GF::Iterate::StopList(20);
        m_G = m_material.G()(0, 0);
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

    void addEventDrivenStep()
    {
        // displacement perturbation to determine the sign in equivalent strain space
        // --------------------------------------------------------------------------

        // current (equivalent) deviatoric strain
        auto eps = GM::Epsd(m_Eps);
        auto Epsd = GM::Deviatoric(m_Eps);

        // store current displacements
        auto u_real = m_u;

        // add displacements perturbation
        for (size_t n = 0; n < m_nnode; ++n) {
            m_u(n, 0) += m_deps_kick * (m_coor(n, 1) - m_coor(0, 1));
        }
        computeStrainStress();

        // equivalent deviatoric strain after perturbation
        auto eps_pert = GM::Epsd(m_Eps);

        // restore displacements
        xt::noalias(m_u) = u_real;
        computeStrainStress();

        // compute sign
        xt::xtensor<double, 2> sign = xt::sign(eps_pert - eps);

        // determine distance to yielding
        // ------------------------------

        auto idx = m_material.CurrentIndex();
        xt::xtensor<double, 2> epsy_l = xt::abs(m_material.CurrentYieldLeft());
        xt::xtensor<double, 2> epsy_r = xt::abs(m_material.CurrentYieldRight());
        xt::xtensor<double, 2> epsy = xt::where(sign > 0, epsy_r, epsy_l);
        xt::xtensor<double, 2> deps = xt::abs(eps - epsy);

        // select weak layer
        eps = xt::view(eps, xt::keep(m_plastic), xt::all());
        deps = xt::view(deps, xt::keep(m_plastic), xt::all());
        epsy = xt::view(epsy, xt::keep(m_plastic), xt::all());
        sign = xt::view(sign, xt::keep(m_plastic), xt::all());
        auto epsxx = xt::view(Epsd, xt::keep(m_plastic), xt::all(), 0, 0);
        auto epsxy = xt::view(Epsd, xt::keep(m_plastic), xt::all(), 0, 1);

        // determine strain increment
        // --------------------------

        // no kick & current strain sufficiently close the next yield strain: don't move
        if (!m_kick && xt::amin(deps)[0] < m_deps_kick / 2.0) {
            return;
        }

        // set yield strain close to next yield strain
        xt::xtensor<double, 2> eps_new = epsy + sign * (-m_deps_kick / 2.0);

        // or, apply a kick instead
        if (m_kick) {
            eps_new = eps + sign * m_deps_kick;
        }

        // compute shear strain increments
        // - two possible solutions
        xt::xtensor<double, 2> dgamma = 2.0 * (-epsxy + xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
        xt::xtensor<double, 2> dgamma_n = 2.0 * (-epsxy - xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
        // - discard irrelevant solutions
        dgamma_n = xt::where(dgamma_n <= 0.0, dgamma, dgamma_n);
        // - select lowest
        dgamma = xt::where(dgamma_n < dgamma, dgamma_n, dgamma);
        // - select minimal
        double dux = xt::amin(dgamma)();

        // add as affine deformation gradient to the system
        for (size_t n = 0; n < m_nnode; ++n) {
            m_u(n, 0) += dux * (m_coor(n, 1) - m_coor(0, 1));
        }
        computeStrainStress();

        // sanity check
        // ------------

        // get element that was moved
        auto index = xt::unravel_index(xt::argmin(dgamma)(), dgamma.shape());
        size_t e = index[0];
        size_t q = index[1];

        // current equivalent deviatoric strain
        eps = xt::view(GM::Epsd(m_Eps), xt::keep(m_plastic), xt::all());

        // current minima
        auto idx_n = m_material.CurrentIndex();

        // check strain
        if (std::abs(eps(e, q) - eps_new(e, q)) / eps_new(e, q) > 1e-4) {
            throw std::runtime_error("Strain not what it was supposed to be");
        }

        // check that no yielding took place
        if (!m_kick) {
            if (xt::any(xt::not_equal(idx, idx_n))) {
                throw std::runtime_error("Yielding took place where it shouldn't");
            }
        }
    }

public:

    void run()
    {
        if (m_file.exist("/git/run")) {
            std::string hash = GIT_COMMIT_HASH;
            H5Easy::dump(m_file, "/git/run", hash);
        }

        if (m_file.exist("/completed")) {
            fmt::print("Marked completed, skipping\n");
            return;
        }

        if (m_file.exist("/stored")) {
            size_t idx = H5Easy::getSize(m_file, "/stored") - std::size_t(1);
            m_inc = H5Easy::load<decltype(m_inc)>(m_file, "/stored", {idx});
            m_kick = H5Easy::load<decltype(m_kick)>(m_file, "/kick", {idx});
            m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {idx});
            xt::noalias(m_u) = H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc));
            computeStrainStress();
            fmt::print("'{0:s}': Loading, inc = {1:d}\n", m_file.getName(), m_inc);
            m_kick = !m_kick;
        }
        else {
            H5Easy::dump(m_file, "/stored", 0, {0});
            H5Easy::dump(m_file, "/kick", 0, {0});
            H5Easy::dump(m_file, "/t", 0.0, {0});
            H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);
        }

        auto epsd = GM::Epsd(m_Eps);
        double emax;

        for (++m_inc;; ++m_inc) {
            addEventDrivenStep();
            GM::epsd(m_Eps, epsd);
            emax = xt::amax(epsd)();

            if (emax >= m_epsd_max) {
                H5Easy::dump(m_file, "/completed", 1);
                fmt::print("'{0:s}': Completed\n", m_file.getName());
                return;
            }

            if (m_kick) {
                for (m_iiter = 0;; ++m_iiter) {

                    timeStep();

                    if (m_stop.stop(xt::linalg::norm(m_fres) / xt::linalg::norm(m_fext), 1.e-5)) {
                        break;
                    }

                    GM::epsd(m_Eps, epsd);
                    emax = xt::amax(epsd)();

                    if (emax >= m_epsd_max) {
                        H5Easy::dump(m_file, "/completed", 1);
                        fmt::print("'{0:s}': Completed\n", m_file.getName());
                        return;
                    }
                }
            }

            fmt::print(
                "'{0:s}': inc = {1:8d}, kick = {2:1d}, iiter = {3:8d}, max(eps) = {4:16.8e}\n",
                m_file.getName(),
                m_inc,
                m_kick,
                m_iiter,
                emax);

            {
                H5Easy::dump(m_file, "/stored", m_inc, {m_inc});
                H5Easy::dump(m_file, "/kick", m_kick, {m_inc});
                H5Easy::dump(m_file, "/t", m_t, {m_inc});
                H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);
            }

            m_iiter = 0;
            m_kick = !m_kick;
            m_v.fill(0.0);
            m_a.fill(0.0);
            m_stop.reset();
        }
    }
};


static const char USAGE[] =
    R"(Run
    Run simulation until maximum strain.

Usage:
    Run [options] <data.hdf5>

Options:
    -h, --help      Show help.
        --version   Show version.

(c) Tom de Geus
)";


int main(int argc, const char** argv)
{
    std::string hash = GIT_COMMIT_HASH;

    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, hash);

    std::string path = args["<data.hdf5>"].asString();
    Main sim(path);
    sim.run();

    return 0;
}
