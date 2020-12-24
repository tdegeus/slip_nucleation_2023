
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <cpppath.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
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

    std::tuple<xt::xtensor<size_t, 1>, xt::xtensor<size_t, 1>> getIncPush(double stress)
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
        xt::xtensor<size_t, 1> inc_system = xt::flatten_indices(xt::argwhere(xt::equal(A, N)));

        // too few system spanning avalanches -> quit
        if (inc_system.size() < 2) {
            xt::xtensor<size_t, 1> inc_push = xt::zeros<size_t>({0});
            return std::make_tuple(inc_system, inc_push);
        }

        // allocate list with increment numbers at which to push
        xt::xtensor<size_t, 1> inc_push = xt::zeros<size_t>({inc_system.size() - 1});

        // allocate list with all increment numbers
        xt::xtensor<size_t, 1> iinc = xt::arange<size_t>(A.size());

        // consider all system spanning avalanches,
        // that are followed by at least one system spanning avalanche
        for (size_t i = 0; i < inc_system.size() - 1; ++i) {
            // - stress after elastic load, kick/area of these increments for checking, increment
            // numbers
            auto s = xt::view(sigd, xt::range(inc_system(i) + 1, inc_system(i + 1), 2));
            auto k = xt::view(kick, xt::range(inc_system(i) + 1, inc_system(i + 1), 2));
            auto a = xt::view(A, xt::range(inc_system(i) + 1, inc_system(i + 1), 2));
            auto n = xt::view(iinc, xt::range(inc_system(i) + 1, inc_system(i + 1), 2));

            // - skip if the loading pattern was not load-kick-load-kick-... (sanity check)
            if (xt::any(xt::not_equal(k, 0ul))) {
                continue;
            }
            if (xt::any(xt::not_equal(a, 0ul))) {
                continue;
            }

            // - find where the strain(stress) is higher than the target strain(stress)
            //   during that increment the strain(stress) elastically moved from below to above the
            //   target strain(stress); the size of this step can be reduced by an arbitrary size,
            //   without violating equilibrium
            auto idx = xt::flatten_indices(xt::argwhere(s > stress));

            // - no increment found -> skip (sanity check)
            if (idx.size() == 0) {
                continue;
            }

            // - start from the increment before it (the beginning of the elastic loading)
            size_t ipush = n(xt::amin(idx)[0]) - 1;

            // - sanity check
            if (sigd(ipush) > stress) {
                continue;
            }
            if (kick(ipush + 1) != 0) {
                continue;
            }

            // - store
            inc_push(i) = ipush;
        }

        // filter list with increments
        // (zero can never be a valid increment, because of the restriction set above)
        inc_push = xt::filter(inc_push, inc_push > 0ul);

        return std::make_tuple(inc_system, inc_push);
    }

public:

    void moveForwardToFixedStress(double stress)
    {
        // store current minima (for sanity check)
        auto idx_n = m_material.CurrentIndex();

        // integration point volume
        auto dV = m_quad.DV(2);

        // macroscopic (deviatoric) stress/strain tensor
        xt::xtensor_fixed<double, xt::xshape<2, 2>> Sigbar = xt::average(m_Sig, dV, {0, 1});
        xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsbar = xt::average(m_Eps, dV, {0, 1});
        xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsd = GM::Deviatoric(Epsbar);

        // current equivalent deviatoric stress/strain
        double eps = GM::Epsd(Epsbar)();
        double sig = GM::Sigd(Sigbar)();

        // get homogeneous shear modulus
        double G = m_material.G()(0, 0);

        // new equivalent deviatoric strain
        double eps_new = eps + (stress - sig) / (2. * G);

        // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
        double dgamma =
            2.0 * (-Epsd(0, 1) + std::sqrt(std::pow(eps_new, 2.0) - std::pow(Epsd(0, 0), 2.0)));

        // add as affine deformation gradient to the system
        for (size_t n = 0; n < m_nnode; ++n) {
            m_u(n, 0) += dgamma * (m_coor(n, 1) - m_coor(0, 1));
        }

        // compute strain/stress
        computeStrainStress();

        // compute new macroscopic stress (for sanity check)
        Sigbar = xt::average(m_Sig, dV, {0, 1});
        sig = GM::Sigd(Sigbar)();

        // current minima (for sanity check)
        auto idx = m_material.CurrentIndex();

        // check that the stress is what it was set to (sanity check)
        if (std::abs(stress - sig) / sig > 1.e-4) {
            throw std::runtime_error(fmt::format(
                "fname = '{0:s}', stress = {1:16.8e}, inc = {2:d}: Stress incorrect.\n",
                m_file.getName(),
                stress,
                m_inc));
        }

        // check that no yielding took place (sanity check)
        if (xt::any(xt::not_equal(idx, idx_n))) {
            throw std::runtime_error(fmt::format(
                "fname = '{0:s}', stress = {1:16.8e}, inc = {2:d}: Yielding took place where it shouldn't.\n",
                m_file.getName(),
                stress,
                m_inc));
        }
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

    void run(double stress, size_t element, size_t inc_c, const std::string& output)
    {
        // extract a list with increments at which to start elastic loading
        xt::xtensor<size_t, 1> inc_system, inc_push;
        std::tie(inc_system, inc_push) = getIncPush(stress);
        MYASSERT(inc_system.size() > 0);
        MYASSERT(inc_push.size() > 0);
        MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

        // get push increment
        size_t ipush = xt::flatten_indices(xt::argwhere(xt::equal(inc_system, inc_c)))(0);
        m_inc = inc_push(ipush);
        MYASSERT(ipush < inc_push.size());
        MYASSERT(inc_push(ipush) >= inc_c);

        // restore displacement
        xt::noalias(m_u) = H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc));
        computeStrainStress();

        // increase displacement to set "stress"
        moveForwardToFixedStress(stress);

        // extract "id" from filename (stored to data)
        std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
        size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

        // extract information needed for storage
        size_t N = m_plastic.size();
        auto dV = m_quad.DV(2);
        xt::xtensor<int, 1> idx_last = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);
        xt::xtensor<int, 1> idx_n = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);
        xt::xtensor<int, 1> idx = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);

        // perturb the displacement of the set element, to (try to) trigger an avalanche
        triggerElement(element);

        // clear/open the output file
        H5Easy::File data(output, H5Easy::File::Overwrite);

        // storage parameters
        int S = 0;           // avalanche size (maximum size since beginning)
        size_t S_next = 0;   // next storage value
        size_t S_index = 0;  // storage index
        size_t S_step = 50;  // storage step size
        size_t A = 0;        // current crack area  (maximum size since beginning)
        size_t A_next = 0;   // next storage value
        size_t A_index = 0;  // storage index
        size_t A_step = 1;   // storage step size
        bool A_store = true; // store synchronised on "A" -> false when A == N
        size_t t_step = 500; // interval at which to store a global snapshot
        size_t t_factor = 4; // "t_step * t_factor" is interval at which to store local snapshot
        size_t ioverview = 0;
        size_t isnapshot = 0;
        size_t ievent = 0;
        bool last = false;
        bool event_attribute = true;
        bool event = false;
        xt::xtensor_fixed<double, xt::xshape<2, 2>> Sig_bar = xt::average(m_Sig, dV, {0, 1});
        xt::xtensor<double, 3> Sig_elem = xt::average(m_Sig, dV, {1});
        xt::xtensor<double, 2> Sig_plas = xt::empty<double>({3ul, N});
        xt::xtensor<double, 1> sig_weak = xt::empty<double>({3ul});
        xt::xtensor<double, 1> sig_crack = xt::empty<double>({3ul});
        xt::xtensor<double, 1> yielded = xt::empty<double>({N});
        xt::xtensor<double, 2> yielded_2 = xt::empty<double>({3ul, N});

        // quench: force equilibrium
        for (size_t iiter = 0;; ++iiter) {
            if (iiter > 0) {
                idx = xt::view(m_material.CurrentIndex(), xt::keep(m_plastic), 0);
            }

            size_t a = xt::sum(xt::not_equal(idx, idx_n))();
            int s = xt::sum(idx - idx_n)();
            A = std::max(A, a);
            S = std::max(S, s);

            bool save_event = xt::any(xt::not_equal(idx, idx_last));
            bool save_overview = iiter % t_step == 0 || last || iiter == 0;
            bool save_snapshot = ((A >= A_next || A == N) && A_store) ||
                                 S >= (int)S_next ||
                                 iiter % (t_step * t_factor) == 0 ||
                                 last || iiter == 0;

            if (save_event || save_overview | save_snapshot) {
                xt::noalias(yielded) = xt::not_equal(idx, idx_n);
                for (size_t k = 0; k < 3; ++k) {
                    xt::view(yielded_2, k, xt::all()) = yielded;
                }
                xt::noalias(Sig_bar) = xt::average(m_Sig, dV, {0, 1});
                xt::noalias(Sig_elem) = xt::average(m_Sig, dV, {1});
                xt::view(Sig_plas, 0, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 0, 0);
                xt::view(Sig_plas, 1, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 0, 1);
                xt::view(Sig_plas, 2, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 1, 1);
                xt::noalias(sig_weak) = xt::mean(Sig_plas, {1});
                xt::noalias(sig_crack) = xt::average(Sig_plas, yielded_2, {1});
            }

            if (save_event) {
                xt::xtensor<size_t, 1> r = xt::flatten_indices(xt::argwhere(xt::not_equal(idx, idx_last)));
                for (size_t i = 0; i < r.size(); ++i) {
                    H5Easy::dump(data, "/event/step", idx(r(i)) - idx_last(r(i)), {ievent});
                    H5Easy::dump(data, "/event/r", r(i), {ievent});
                    H5Easy::dump(data, "/event/global/iiter", iiter, {ievent});
                    H5Easy::dump(data, "/event/global/S", s, {ievent});
                    H5Easy::dump(data, "/event/global/A", a, {ievent});
                    H5Easy::dump(data, "/event/global/sig", Sig_bar(0, 0), {0, ievent});
                    H5Easy::dump(data, "/event/global/sig", Sig_bar(0, 1), {1, ievent});
                    H5Easy::dump(data, "/event/global/sig", Sig_bar(1, 1), {2, ievent});
                    H5Easy::dump(data, "/event/weak/sig", sig_weak(0), {0, ievent});
                    H5Easy::dump(data, "/event/weak/sig", sig_weak(1), {1, ievent});
                    H5Easy::dump(data, "/event/weak/sig", sig_weak(2), {2, ievent});
                    H5Easy::dump(data, "/event/crack/sig", sig_crack(0), {0, ievent});
                    H5Easy::dump(data, "/event/crack/sig", sig_crack(1), {1, ievent});
                    H5Easy::dump(data, "/event/crack/sig", sig_crack(2), {2, ievent});
                    ievent++;
                }
                xt::noalias(idx_last) = idx;
                event = true;
            }

            // store global snapshot
            if (save_overview || save_snapshot) {
                H5Easy::dump(data, "/overview/global/iiter", iiter, {ioverview});
                H5Easy::dump(data, "/overview/global/S", s, {ioverview});
                H5Easy::dump(data, "/overview/global/A", a, {ioverview});
                H5Easy::dump(data, "/overview/global/sig", Sig_bar(0, 0), {0, ioverview});
                H5Easy::dump(data, "/overview/global/sig", Sig_bar(0, 1), {1, ioverview});
                H5Easy::dump(data, "/overview/global/sig", Sig_bar(1, 1), {2, ioverview});
                H5Easy::dump(data, "/overview/weak/sig", sig_weak(0), {0, ioverview});
                H5Easy::dump(data, "/overview/weak/sig", sig_weak(1), {1, ioverview});
                H5Easy::dump(data, "/overview/weak/sig", sig_weak(2), {2, ioverview});
                H5Easy::dump(data, "/overview/crack/sig", sig_crack(0), {0, ioverview});
                H5Easy::dump(data, "/overview/crack/sig", sig_crack(1), {1, ioverview});
                H5Easy::dump(data, "/overview/crack/sig", sig_crack(2), {2, ioverview});
                ioverview++;
            }

            // store local snapshot
            if (save_snapshot) {
                H5Easy::dump(data, fmt::format("/snapshot/plastic/{0:d}/sig", isnapshot), Sig_plas);
                H5Easy::dump(data, fmt::format("/snapshot/plastic/{0:d}/idx", isnapshot), idx);
                H5Easy::dump(data, "/snapshot/storage/overview", ioverview, {isnapshot});
                H5Easy::dump(data, "/snapshot/storage/snapshot", isnapshot, {isnapshot});
                if ((A >= A_next || A == N) && A_store) {
                    H5Easy::dump(data, "/snapshot/storage/A/values", A, {A_index});
                    H5Easy::dump(data, "/snapshot/storage/A/index", isnapshot, {A_index});
                    A_next += A_step;
                    A_index++;
                    if (A >= N) {
                        A_store = false;
                    }
                }
                if (S >= (int)S_next) {
                    H5Easy::dump(data, "/snapshot/storage/S/values", S, {S_index});
                    H5Easy::dump(data, "/snapshot/storage/S/index", isnapshot, {S_index});
                    S_next += S_step;
                    S_index++;
                }
                if (iiter % (t_step * t_factor) == 0) {
                    size_t j = iiter / (t_step * t_factor);
                    H5Easy::dump(data, "/snapshot/storage/iiter/values", iiter, {j});
                    H5Easy::dump(data, "/snapshot/storage/iiter/index", isnapshot, {j});
                }
                isnapshot++;
            }

            // write info
            if (event && event_attribute) {
                H5Easy::dumpAttribute(data, "/event/step", "desc", std::string("Number of times the block yielded since the last event"));
                H5Easy::dumpAttribute(data, "/event/r", "desc", std::string("Position of the yielding block"));

                H5Easy::dumpAttribute(data, "/event/global/iiter", "desc", std::string("Iteration number for event"));
                H5Easy::dumpAttribute(data, "/event/global/S", "desc", std::string("Avalanche size at time of event"));
                H5Easy::dumpAttribute(data, "/event/global/A", "desc", std::string("Avalanche radius at time of event"));

                H5Easy::dumpAttribute(data, "/event/global/sig", "desc", std::string("Macroscopic stress (xx, xy, yy) at time of event"));
                H5Easy::dumpAttribute(data, "/event/global/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/global/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/global/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(data, "/event/weak/sig", "desc", std::string("Stress averaged on weak layer (xx, xy, yy) at time of event"));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(data, "/event/crack/sig", "desc", std::string("Stress averaged on yielded blocks (xx, xy, yy) at time of event"));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "yy", static_cast<size_t>(2));

                event_attribute = false;
            }

            if (iiter == 0) {
                H5Easy::dumpAttribute(data, "/overview/global/iiter", "desc", std::string("Iteration number"));
                H5Easy::dumpAttribute(data, "/overview/global/S", "desc", std::string("Avalanche size"));
                H5Easy::dumpAttribute(data, "/overview/global/A", "desc", std::string("Avalanche radius"));

                H5Easy::dumpAttribute(data, "/overview/global/sig", "desc", std::string("Macroscopic stress (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(data, "/overview/weak/sig", "desc", std::string("Stress averaged on weak layer (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(data, "/overview/crack/sig", "desc", std::string("Stress averaged on yielded blocks (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(data, fmt::format("/snapshot/plastic/{0:d}/sig", 0), "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, fmt::format("/snapshot/plastic/{0:d}/sig", 0), "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, fmt::format("/snapshot/plastic/{0:d}/sig", 0), "yy", static_cast<size_t>(2));
                H5Easy::dumpAttribute(data, fmt::format("/snapshot/plastic/{0:d}/sig", 0), "desc", std::string("Stress tensor along the weak layer (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, fmt::format("/snapshot/plastic/{0:d}/idx", 0), "desc", std::string("Index of the current local minimum"));

                H5Easy::dump(data, "/snapshot/storage/A/step", A_step);
                H5Easy::dump(data, "/snapshot/storage/S/step", S_step);
                H5Easy::dump(data, "/snapshot/storage/iiter/step", t_step * t_factor);
            }

            if (last) {
                break;
            }

            // time increment
            timeStep();

            // - check for convergence
            if (m_stop.stop(xt::linalg::norm(m_fres) / xt::linalg::norm(m_fext), 1.e-5)) {
                last = true;
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
        H5Easy::dump(data, "/meta/N", N);
    }

};


static const char USAGE[] =
    R"(Run
  Extract time evolution of a specific push.

Usage:
  Run [options] --stress=N --output=N --element=N --file=N --incc=N

Arguments:
      --output=N      Path of the output file.
      --stress=N      Relative stress distance to "sigma_down" at which to measure.
      --element=N     Element to push.
      --file=N        The path to the simulation file.
      --incc=N        Increment number of the system-spanning avalanche.

Options:
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
    double stress = std::stod(args["--stress"].asString());
    size_t inc_c = static_cast<size_t>(std::stoi(args["--incc"].asString()));
    size_t element = static_cast<size_t>(std::stoi(args["--element"].asString()));

    Main sim(file);

    sim.run(stress, element, inc_c, output);

    return 0;
}
