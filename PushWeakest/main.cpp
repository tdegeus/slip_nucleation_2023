
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
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


#ifdef GMATELASTOPLASTICQPOT_ENABLE_ASSERT
#define CHECK_MYSHORTCUT
#endif


class Main {

private:

    // input/output file
    H5Easy::File m_file;

    // mesh parameters
    xt::xtensor<size_t, 2> m_conn;
    xt::xtensor<size_t, 2> m_conn_elastic;
    xt::xtensor<size_t, 2> m_conn_plastic;
    xt::xtensor<double, 2> m_coor;
    xt::xtensor<size_t, 2> m_dofs;
    xt::xtensor<size_t, 1> m_iip;

    // mesh dimensions
    size_t m_N; // == nelem_plas
    size_t m_nelem;
    size_t m_nelem_elas;
    size_t m_nelem_plas;
    size_t m_nne;
    size_t m_nnode;
    size_t m_ndim;
    size_t m_nip;

    // element sets
    xt::xtensor<size_t, 1> m_elastic;
    xt::xtensor<size_t, 1> m_plastic;

    // numerical quadrature
    QD::Quadrature m_quad;
    QD::Quadrature m_quad_elas;
    QD::Quadrature m_quad_plas;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector;
    GF::VectorPartitioned m_vector_elas;
    GF::VectorPartitioned m_vector_plas;

    // mass matrix
    GF::MatrixDiagonalPartitioned m_M;

    // damping matrix
    GF::MatrixDiagonal m_D;

    // material definition
    GM::Array<2> m_material;
    GM::Array<2> m_material_elas;
    GM::Array<2> m_material_plas;

    // convergence check
    GF::Iterate::StopList m_stop = GF::Iterate::StopList(20);

    // time evolution
    double m_t = 0.0;   // current time
    double m_dt;        // time step

    // event-driven settings
    size_t m_inc = 0;    // current increment
    double m_deps_kick;  // equivalent strain increment
    size_t m_nyield;

    // nodal displacements, velocities, and accelerations (current and last time-step)
    xt::xtensor<double, 2> m_u;
    xt::xtensor<double, 2> m_v;
    xt::xtensor<double, 2> m_a;
    xt::xtensor<double, 2> m_v_n;
    xt::xtensor<double, 2> m_a_n;

    // element vectors
    xt::xtensor<double, 3> m_ue;
    xt::xtensor<double, 3> m_fe;
    xt::xtensor<double, 3> m_ue_plas;
    xt::xtensor<double, 3> m_fe_plas;

    // nodal forces
#ifdef CHECK_MYSHORTCUT
    xt::xtensor<double, 2> m_fmaterial;
#endif
    xt::xtensor<double, 2> m_felas;
    xt::xtensor<double, 2> m_fplas;
    xt::xtensor<double, 2> m_fdamp;
    xt::xtensor<double, 2> m_fint;
    xt::xtensor<double, 2> m_fext;
    xt::xtensor<double, 2> m_fres;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps;
    xt::xtensor<double, 4> m_Eps_elas;
    xt::xtensor<double, 4> m_Eps_plas;
    xt::xtensor<double, 4> m_Sig;
    xt::xtensor<double, 4> m_Sig_elas;
    xt::xtensor<double, 4> m_Sig_plas;

    // stiffness matrix
    GF::Matrix m_K_elas;

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadWrite)
    {
        readMesh();
        setMass();
        setDamping();
        setMaterial();
        readParameters();
        computeStrainStress();

#ifdef CHECK_MYSHORTCUT
        std::cout << "Checking simplification" << std::endl;
#endif
    }

public:

    void readParameters()
    {
        m_dt = H5Easy::load<double>(m_file, "/run/dt");
        m_deps_kick = H5Easy::load<double>(m_file, "/run/epsd/kick");
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

        m_elastic = xt::sort(H5Easy::load<decltype(m_elastic)>(m_file, "/elastic/elem"));
        m_plastic = xt::sort(H5Easy::load<decltype(m_plastic)>(m_file, "/cusp/elem"));

        m_nelem_elas = m_elastic.size();
        m_nelem_plas = m_plastic.size();
        m_conn_elastic = xt::view(m_conn, xt::keep(m_elastic), xt::all());
        m_conn_plastic = xt::view(m_conn, xt::keep(m_plastic), xt::all());
        m_N = m_nelem_plas;

        m_vector = GF::VectorPartitioned(m_conn, m_dofs, m_iip);
        m_vector_elas = GF::VectorPartitioned(m_conn_elastic, m_dofs, m_iip);
        m_vector_plas = GF::VectorPartitioned(m_conn_plastic, m_dofs, m_iip);

        m_quad = QD::Quadrature(m_vector.AsElement(m_coor));
        m_quad_elas = QD::Quadrature(m_vector_elas.AsElement(m_coor));
        m_quad_plas = QD::Quadrature(m_vector_plas.AsElement(m_coor));
        m_nip = m_quad.nip();

        m_u = m_vector.AllocateNodevec(0.0);
        m_v = m_vector.AllocateNodevec(0.0);
        m_a = m_vector.AllocateNodevec(0.0);
        m_v_n = m_vector.AllocateNodevec(0.0);
        m_a_n = m_vector.AllocateNodevec(0.0);

        m_ue = m_vector.AllocateElemvec(0.0);
        m_fe = m_vector.AllocateElemvec(0.0);
        m_ue_plas = m_vector_plas.AllocateElemvec(0.0);;
        m_fe_plas = m_vector_plas.AllocateElemvec(0.0);;

#ifdef CHECK_MYSHORTCUT
        m_fmaterial = m_vector.AllocateNodevec(0.0);
#endif
        m_felas = m_vector.AllocateNodevec(0.0);
        m_fplas = m_vector.AllocateNodevec(0.0);
        m_fdamp = m_vector.AllocateNodevec(0.0);
        m_fint = m_vector.AllocateNodevec(0.0);
        m_fext = m_vector.AllocateNodevec(0.0);
        m_fres = m_vector.AllocateNodevec(0.0);

        m_Eps = m_quad.AllocateQtensor<2>(0.0);
        m_Sig = m_quad.AllocateQtensor<2>(0.0);
        m_Eps_elas = m_quad_elas.AllocateQtensor<2>(0.0);
        m_Sig_elas = m_quad_elas.AllocateQtensor<2>(0.0);
        m_Eps_plas = m_quad_plas.AllocateQtensor<2>(0.0);
        m_Sig_plas = m_quad_plas.AllocateQtensor<2>(0.0);
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
        m_material_elas = GM::Array<2>({m_nelem_elas, m_nip});
        m_material_plas = GM::Array<2>({m_nelem_plas, m_nip});

        {
            auto k = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/K");
            auto g = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/G");

            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
            xt::view(I, xt::keep(m_elastic), xt::all()) = 1ul;
            xt::view(idx, xt::keep(m_elastic), xt::all()) = xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});
            m_material.setElastic(I, idx, k, g);

            I = xt::ones<size_t>({m_nelem_elas, m_nip});
            idx = xt::zeros<size_t>({m_nelem_elas, m_nip});
            xt::view(idx, xt::range(0, m_nelem_elas), xt::all()) = xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});
            m_material_elas.setElastic(I, idx, k, g);
        }

        {
            auto k = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/K");
            auto g = H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/G");
            auto y = H5Easy::load<xt::xtensor<double, 2>>(m_file, "/cusp/epsy");
            m_nyield = y.shape(1);

            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
            xt::view(I, xt::keep(m_plastic), xt::all()) = 1ul;
            xt::view(idx, xt::keep(m_plastic), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
            m_material.setCusp(I, idx, k, g, y);

            I = xt::ones<size_t>({m_nelem_plas, m_nip});
            idx = xt::zeros<size_t>({m_nelem_plas, m_nip});
            xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
            m_material_plas.setCusp(I, idx, k, g, y);
        }

        xt::xtensor<double, 2> k = m_material.K();
        xt::xtensor<double, 2> g = m_material.G();
        MYASSERT(xt::mean(k)() == k(0, 0));
        MYASSERT(xt::mean(g)() == g(0, 0));

        m_material.check();
        m_material_elas.check();
        m_material_plas.check();

        m_material.setStrain(m_Eps);
        m_material_elas.setStrain(m_Eps_elas);
        m_material_plas.setStrain(m_Eps_plas);

        m_K_elas = GF::Matrix(m_conn_elastic, m_dofs);
        m_K_elas.assemble(m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material_elas.Tangent()));
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

    void computeStrainStressForcesWeakLayer()
    {
        m_vector_plas.asElement(m_u, m_ue_plas);
        m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
        m_material_plas.setStrain(m_Eps_plas);
        m_material_plas.stress(m_Sig_plas);
        m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
        m_vector_plas.assembleNode(m_fe_plas, m_fplas);
        m_K_elas.dot(m_u, m_felas);
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

        computeStrainStressForcesWeakLayer();

#ifdef CHECK_MYSHORTCUT
        computeStrainStress();
        m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
        m_vector.assembleNode(m_fe, m_fmaterial);

        if (!xt::allclose(m_fmaterial, m_felas + m_fplas)) {
            throw std::runtime_error("Forces");
        }
#endif

        // estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // re-estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);
    }

public:

    void triggerWeakest()
    {
        // displacement perturbation to determine the sign in equivalent strain space
        // --------------------------------------------------------------------------

        // current (equivalent) deviatoric strain
        auto eps = GM::Epsd(m_Eps_plas);
        auto Epsd = GM::Deviatoric(m_Eps_plas);

        // store current displacements
        auto u_real = m_u;

        // add displacements perturbation
        for (size_t n = 0; n < m_nnode; ++n) {
            m_u(n, 0) += m_deps_kick * (m_coor(n, 1) - m_coor(0, 1));
        }
        computeStrainStressForcesWeakLayer();

        // equivalent deviatoric strain after perturbation
        auto eps_pert = GM::Epsd(m_Eps_plas);

        // restore displacements
        xt::noalias(m_u) = u_real;
        computeStrainStressForcesWeakLayer();

        // compute sign
        xt::xtensor<double, 2> sign = xt::sign(eps_pert - eps);

        // determine distance to yielding
        // ------------------------------

        auto idx = m_material_plas.CurrentIndex();
        xt::xtensor<double, 2> epsy_l = xt::abs(m_material_plas.CurrentYieldLeft());
        xt::xtensor<double, 2> epsy_r = xt::abs(m_material_plas.CurrentYieldRight());
        xt::xtensor<double, 2> epsy = xt::where(sign > 0, epsy_r, epsy_l);
        xt::xtensor<double, 2> deps = xt::abs(eps - epsy);
        auto epsxx = xt::view(Epsd, xt::all(), xt::all(), 0, 0);
        auto epsxy = xt::view(Epsd, xt::all(), xt::all(), 0, 1);

        // determine strain increment
        // --------------------------

        // set new strain just after yielding
        xt::xtensor<double, 2> eps_new = epsy + sign * (m_deps_kick / 2.0);

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
        computeStrainStressForcesWeakLayer();

        // sanity check
        // ------------

        // get element that was moved
        auto index = xt::unravel_index(xt::argmin(dgamma)(), dgamma.shape());
        size_t e = index[0];
        size_t q = index[1];

        // current equivalent deviatoric strain
        eps = GM::Epsd(m_Eps_plas);

        // current minima
        auto idx_n = m_material_plas.CurrentIndex();

        // check strain
        if (std::abs(eps(e, q) - eps_new(e, q)) / eps_new(e, q) > 1e-4) {
            throw std::runtime_error("Strain not what it was supposed to be");
        }

        // check that no yielding took place
        if (xt::sum(xt::not_equal(idx, idx_n))() == 0 || xt::sum(xt::not_equal(idx, idx_n))() > 4) {
            throw std::runtime_error("Yielding took place where it shouldn't");
        }
    }

public:

    size_t getMaxStored()
    {
        size_t istored = H5Easy::getSize(m_file, "/stored") - std::size_t(1);
        return H5Easy::load<size_t>(m_file, "/stored", {istored});
    }

public:

    void writeCompleted()
    {
        H5Easy::dump(m_file, "/completed", 1);
    }

public:

    int runIncrement(const std::string& outfilename)
    {
        // make sure to start from quenched state
        m_v.fill(0.0);
        m_a.fill(0.0);
        m_stop.reset();

        // load last increment
        m_inc = getMaxStored();
        m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {m_inc});
        xt::noalias(m_u) = H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc));
        computeStrainStressForcesWeakLayer();
        fmt::print("'{0:s}': Loading, inc = {1:d}\n", m_file.getName(), m_inc);
        m_inc++;

        // extract information needed for storage
        auto dV = m_quad.AsTensor<2>(m_quad.dV());
        auto dV_plas = m_quad_plas.AsTensor<2>(m_quad_plas.dV());
        xt::xtensor<int, 1> idx_last = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx_n = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);

        // trigger weakest point
        triggerWeakest();

        // clear/open the output file
        H5Easy::File data(outfilename, H5Easy::File::Overwrite);

        // storage parameters
        int S = 0;           // avalanche size (maximum size since beginning)
        size_t A = 0;        // current crack area (maximum size since beginning)
        size_t t_step = 500; // interval at which to store a global snapshot
        size_t ioverview = 0;
        size_t ievent = 0;
        bool last = false;
        bool event_attribute = true;
        bool event = false;
        xt::xtensor<double, 2> Sig_bar = xt::average(m_Sig, dV, {0, 1}); // only shape matters
        xt::xtensor<double, 3> Sig_elem = xt::average(m_Sig_plas, dV_plas, {1}); // only shape matters
        xt::xtensor<double, 2> Sig_plas = xt::empty<double>({3ul, m_N});
        xt::xtensor<double, 1> sig_weak = xt::empty<double>({3ul});
        xt::xtensor<double, 1> sig_crack = xt::empty<double>({3ul});
        xt::xtensor<double, 1> yielded = xt::empty<double>({m_N});
        xt::xtensor<double, 2> yielded_broadcast = xt::empty<double>({3ul, m_N});

        // quench: force equilibrium
        for (size_t iiter = 0;; ++iiter) {
            if (iiter > 0) {
                idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
            }

             // break if maximum local strain could be exceeded
            if (xt::amax(idx)() > static_cast<int>(m_nyield) - 10) {
                return -1;
            }

            size_t a = xt::sum(xt::not_equal(idx, idx_n))();
            int s = xt::sum(idx - idx_n)();
            A = std::max(A, a);
            S = std::max(S, s);

            bool save_event = xt::any(xt::not_equal(idx, idx_last));
            bool save_overview = iiter % t_step == 0 || last || iiter == 0;

            if (save_event || save_overview) {
                computeStrainStress();
                xt::noalias(yielded) = xt::not_equal(idx, idx_n);
                for (size_t k = 0; k < 3; ++k) {
                    xt::view(yielded_broadcast, k, xt::all()) = yielded;
                }
                xt::noalias(Sig_bar) = xt::average(m_Sig, dV, {0, 1});
                xt::noalias(Sig_elem) = xt::average(m_Sig_plas, dV_plas, {1});
                xt::view(Sig_plas, 0, xt::all()) = xt::view(Sig_elem, xt::all(), 0, 0);
                xt::view(Sig_plas, 1, xt::all()) = xt::view(Sig_elem, xt::all(), 0, 1);
                xt::view(Sig_plas, 2, xt::all()) = xt::view(Sig_elem, xt::all(), 1, 1);
                xt::noalias(sig_weak) = xt::mean(Sig_plas, {1});
                xt::noalias(sig_crack) = xt::average(Sig_plas, yielded_broadcast, {1});
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
            if (save_overview) {
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
        H5Easy::dump(m_file, "/git/run", hash, {m_inc});
        H5Easy::dump(data, "/git/run", hash);

        H5Easy::dump(m_file, "/stored", m_inc, {m_inc});
        H5Easy::dump(m_file, "/t", m_t, {m_inc});
        H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);

        H5Easy::dump(data, "/meta/completed", 1);
        H5Easy::dump(data, "/meta/uuid", H5Easy::load<std::string>(m_file, "/uuid"));
        H5Easy::dump(data, "/meta/inc", m_inc);
        H5Easy::dump(data, "/meta/dt", m_dt);
        H5Easy::dump(data, "/meta/N", m_N);

        return xt::sum(idx - idx_n)();
    }
};


static const char USAGE[] =
    R"(PushWeakest
    Push the weakest element and quench. Store new state to input-file and write evolution
    to a separate output-file per increment.

Usage:
    PushWeakest [options] --output=N --input=N

Arguments:
    --output=N      Base pat of the output-file: appended with "_ipush={inc:d}.hdf5"
    --input=N       The path to the simulation file.

Options:
    -h, --help      Show help.
        --version   Show version.

(c) Tom de Geus
)";


int main(int argc, const char** argv)
{
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, "v0.0.1");

    std::string output = args["--output"].asString();
    std::string input = args["--input"].asString();

    Main sim(input);

    for (size_t i = 0; i < 200; ++i)
    {
        size_t inc = sim.getMaxStored();
        int S = sim.runIncrement(fmt::format("{0:s}_ipush={1:d}.hdf5", output, inc + 1));
        if (S <= 0) {
            break;
        }
    }
    sim.writeCompleted();

    return 0;
}
