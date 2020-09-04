
#include <docopt/docopt.h>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <GooseFEM/GooseFEM.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <fmt/core.h>
#include <cpppath.h>
#include <highfive/H5Easy.hpp>

// alias namespaces
namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

// hard assertions (not optimised away)
#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line)                                                                                   \
    if (!(expr))                                                                                                          \
    {                                                                                                                     \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + ": assertion failed (" #expr ") \n\t"); \
    }

// =================================================================================================

class Main
{

// -------------------------------------------------------------------------------------------------
// class ('global') variables
// -------------------------------------------------------------------------------------------------

private:

  // input/output file
  H5Easy::File m_file;

  // mesh parameters
  xt::xtensor<size_t,2> m_conn;
  xt::xtensor<double,2> m_coor;
  xt::xtensor<size_t,2> m_dofs;
  xt::xtensor<size_t,1> m_iip;

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
  GM::Matrix m_material;

  // convergence check
  GF::Iterate::StopList m_stop;

  // current time & time-step
  double m_t=0.0;
  double m_dt;

  // event-driven settings
  size_t m_inc=0;                  // current increment
  double m_deps_kick;              // equivalent strain increment
  xt::xtensor<size_t,1> m_plastic; // plastic elements

  // nodal displacements, velocities, and accelerations (current and last time-step)
  xt::xtensor<double,2> m_u;
  xt::xtensor<double,2> m_v;
  xt::xtensor<double,2> m_a;
  xt::xtensor<double,2> m_v_n;
  xt::xtensor<double,2> m_a_n;

  // element vectors
  xt::xtensor<double,3> m_ue;
  xt::xtensor<double,3> m_fe;

  // nodal forces
  xt::xtensor<double,2> m_felas;
  xt::xtensor<double,2> m_fdamp;
  xt::xtensor<double,2> m_fint;
  xt::xtensor<double,2> m_fext;
  xt::xtensor<double,2> m_fres;

  // integration point strain and stress
  xt::xtensor<double,4> m_Eps;
  xt::xtensor<double,4> m_Sig;

// -------------------------------------------------------------------------------------------------

public:

// -------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------

Main(const std::string &fname) : m_file(fname, H5Easy::File::ReadOnly)
{
  readMesh();
  setMass();
  setDamping();
  setMaterial();
  readParameters();
  computeStrainStress();
}

// -------------------------------------------------------------------------------------------------
// read simulation parameters
// -------------------------------------------------------------------------------------------------

void readParameters()
{
  // time step
  m_dt = H5Easy::load<double>(m_file, "/run/dt");

  // kick size
  m_deps_kick = H5Easy::load<double>(m_file, "/run/epsd/kick");

  // initialise stop list
  m_stop = GF::Iterate::StopList(20);
}

// -------------------------------------------------------------------------------------------------
// read mesh
// -------------------------------------------------------------------------------------------------

void readMesh()
{
  // read fields
  m_conn  = H5Easy::load<xt::xtensor<size_t,2>>(m_file, "/conn");
  m_coor  = H5Easy::load<xt::xtensor<double,2>>(m_file, "/coor");
  m_dofs  = H5Easy::load<xt::xtensor<size_t,2>>(m_file, "/dofs");
  m_iip   = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/dofsP");

  // extract sizes
  m_nnode = m_coor.shape(0);
  m_ndim  = m_coor.shape(1);
  m_nelem = m_conn.shape(0);
  m_nne   = m_conn.shape(1);

  // vector-definition: transform nodal vectors <-> DOF values
  m_vector = GF::VectorPartitioned(m_conn, m_dofs, m_iip);

  // numerical quadrature: transform displacements -> strains, integrate stresses -> forces, ...
  m_quad  = QD::Quadrature(m_vector.AsElement(m_coor));
  m_nip   = m_quad.nip();

  // nodal displacements, velocities, and accelerations (current and last time-step)
  m_u     = xt::zeros<double>(m_coor.shape());
  m_v     = xt::zeros<double>(m_coor.shape());
  m_a     = xt::zeros<double>(m_coor.shape());
  m_v_n   = xt::zeros<double>(m_coor.shape());
  m_a_n   = xt::zeros<double>(m_coor.shape());

  // element vectors
  m_ue    = xt::zeros<double>({m_nelem, m_nne, m_ndim});
  m_fe    = xt::zeros<double>({m_nelem, m_nne, m_ndim});

  // nodal forces
  m_felas = xt::zeros<double>(m_coor.shape());
  m_fdamp = xt::zeros<double>(m_coor.shape());
  m_fint  = xt::zeros<double>(m_coor.shape());
  m_fext  = xt::zeros<double>(m_coor.shape());
  m_fres  = xt::zeros<double>(m_coor.shape());

  // integration point strain and stress
  m_Eps   = xt::zeros<double>({m_nelem, m_nip, m_ndim, m_ndim});
  m_Sig   = xt::zeros<double>({m_nelem, m_nip, m_ndim, m_ndim});
}

// -------------------------------------------------------------------------------------------------
// read/set mass matrix
// -------------------------------------------------------------------------------------------------

void setMass()
{
  // allocate
  m_M = GF::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);

  // nodal coordinates as element vector
  xt::xtensor<double,3> x = m_vector.AsElement(m_coor);

  // nodal quadrature
  QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

  // element values
  xt::xtensor<double,1> val_elem = H5Easy::load<xt::xtensor<double,1>>(m_file, "/rho");

  // check size
  MYASSERT(val_elem.size() == m_nelem);

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
  // - copy
  for (size_t q = 0; q < nodalQuad.nip(); ++q)
    xt::view(val_quad, xt::all(), q) = val_elem;

  // compute diagonal matrices
  m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
}

// -------------------------------------------------------------------------------------------------
// read/set damping matrix
// -------------------------------------------------------------------------------------------------

void setDamping()
{
  // allocate
  m_D = GF::MatrixDiagonal(m_conn, m_dofs);

  // nodal coordinates as element vector
  xt::xtensor<double,3> x = m_vector.AsElement(m_coor);

  // nodal quadrature
  QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

  // element values
  xt::xtensor<double,1> val_elem = H5Easy::load<xt::xtensor<double,1>>(m_file, "/damping/alpha");

  // check size
  MYASSERT(val_elem.size() == m_nelem);

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
  // - copy
  for (size_t q = 0; q < nodalQuad.nip(); ++q)
    xt::view(val_quad, xt::all(), q) = val_elem;

  // compute diagonal matrices
  m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
}

// -------------------------------------------------------------------------------------------------
// set material definition
// -------------------------------------------------------------------------------------------------

void setMaterial()
{
  // allocate
  m_material = GM::Matrix(m_nelem, m_nip);

  // add elastic elements
  {
    xt::xtensor<size_t,1> elem = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/elastic/elem");
    xt::xtensor<double,1> k    = H5Easy::load<xt::xtensor<double,1>>(m_file, "/elastic/K"   );
    xt::xtensor<double,1> g    = H5Easy::load<xt::xtensor<double,1>>(m_file, "/elastic/G"   );

    xt::xtensor<size_t,2> I   = xt::zeros<size_t>({m_nelem, m_nip});
    xt::xtensor<size_t,2> idx = xt::zeros<size_t>({m_nelem, m_nip});

    xt::view(I, xt::keep(elem), xt::all()) = 1ul;

    for (size_t q = 0; q < m_nip; ++q)
      xt::view(idx, xt::keep(elem), q) = xt::arange<size_t>(elem.size());

    m_material.setElastic(I, idx, k, g);
  }

  // add plastic-cusp elements
  {
    xt::xtensor<size_t,1> elem = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/cusp/elem");
    xt::xtensor<double,1> k    = H5Easy::load<xt::xtensor<double,1>>(m_file, "/cusp/K"   );
    xt::xtensor<double,1> g    = H5Easy::load<xt::xtensor<double,1>>(m_file, "/cusp/G"   );
    xt::xtensor<double,2> y    = H5Easy::load<xt::xtensor<double,2>>(m_file, "/cusp/epsy");

    xt::xtensor<size_t,2> I   = xt::zeros<size_t>({m_nelem, m_nip});
    xt::xtensor<size_t,2> idx = xt::zeros<size_t>({m_nelem, m_nip});

    xt::view(I, xt::keep(elem), xt::all()) = 1ul;

    for (size_t q = 0; q < m_nip; ++q)
      xt::view(idx, xt::keep(elem), q) = xt::arange<size_t>(elem.size());

    m_material.setCusp(I, idx, k, g, y);
  }

  // check homogeneous elasticity
  xt::xtensor<double,2> k = m_material.K();
  xt::xtensor<double,2> g = m_material.G();
  // -
  MYASSERT(xt::mean(k)[0] == k(0,0));
  MYASSERT(xt::mean(g)[0] == g(0,0));

  // check full material allocation
  m_material.check();

  // plastic elements
  m_plastic = xt::sort(xt::flatten_indices(xt::argwhere(xt::amin(m_material.isPlastic(),{1}))));
}

// -------------------------------------------------------------------------------------------------
// time step using velocity-Verlet algorithm
// -------------------------------------------------------------------------------------------------

void timeStep()
{
  // history

  m_t += m_dt;

  xt::noalias(m_v_n) = m_v;
  xt::noalias(m_a_n) = m_a;

  // new displacement

  xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt,2.) * m_a;

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

// -------------------------------------------------------------------------------------------------
// compute strain and stress based on current displacement
// -------------------------------------------------------------------------------------------------

void computeStrainStress()
{
  m_vector.asElement(m_u, m_ue);
  m_quad.symGradN_vector(m_ue, m_Eps);
  m_material.stress(m_Eps, m_Sig);
}

// -------------------------------------------------------------------------------------------------
// get increments from which to elastically increase the strain
// -------------------------------------------------------------------------------------------------

xt::xtensor<size_t,1> getIncPush()
{
  // integration point volume
  xt::xtensor<double,4> dV = m_quad.DV(2);

  // number of plastic cells
  size_t N = m_plastic.size();

  // basic information for each increment
  xt::xtensor<size_t,1> stored = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/stored");
  xt::xtensor<size_t,1> kick   = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/kick");

  // allocate result
  xt::xtensor<size_t,1> A    = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> epsd = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> sigd = xt::zeros<double>({xt::amax(stored)[0]+1});

  // index of the current quadratic potential,
  // for the first integration point per plastic element
  auto idx_n = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);

  // loop over increments
  for (size_t istored = 0; istored < stored.size(); ++istored)
  {
    // - get increment number
    size_t inc = stored(istored);

    // - restore displacement
    xt::noalias(m_u) = H5Easy::load<xt::xtensor<double,2>>(m_file, "/disp/"+std::to_string(inc));

    // - update strain/strain
    computeStrainStress();

    // - index of the current quadratic potential
    auto idx = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);

    // - macroscopic strain/stress tensor
    xt::xtensor_fixed<double, xt::xshape<2,2>> Epsbar = xt::average(m_Eps, dV, {0,1});
    xt::xtensor_fixed<double, xt::xshape<2,2>> Sigbar = xt::average(m_Sig, dV, {0,1});

    // - macroscopic equivalent strain/stress
    epsd(inc) = GM::Epsd(Epsbar);
    sigd(inc) = GM::Sigd(Sigbar);

    // - avalanche area
    A(inc) = xt::sum(xt::not_equal(idx,idx_n))[0];

    // - update history
    idx_n = idx;
  }

  // determine increment at which the steady-state starts, by considering the elasto-plastic tangent
  // (stress/strain), w.r.t. the elastic tangent (shear modulus)
  // some 'engineering quantity' is use, which was checked visually
  // - initialise elasto-plastic tangent of each increment
  xt::xtensor<double,1> K = xt::zeros<size_t>(sigd.shape());
  size_t steadystate = 0;
  // - compute
  for (size_t i = 1; i < K.size(); ++i)
    K(i) = (sigd(i) - sigd(0)) / (epsd(i) - epsd(0));
  // - set dummy (to take the minimum below)
  K(0) = K(1);
  // - get steady-state increment
  if (xt::any(K <= 0.95 * K(1))) {
    //  - select increment
    steadystate = xt::amin(xt::from_indices(xt::argwhere(K <= 0.95 * K(1))))[0];
    // - make sure to skip at least two increments (artificial: to avoid checks below)
    steadystate = std::max(steadystate, std::size_t(2));
    // - always start the steady-state by elastic loading
    if (kick(steadystate))
      steadystate += 1;
  }

  // remove all non-steady-state increments from further consideration
  xt::view(A, xt::range(0, steadystate)) = 0;

  // list with increment of system-spanning avalanches
  return xt::flatten_indices(xt::argwhere(xt::equal(A,N)));
}

// -------------------------------------------------------------------------------------------------
// trigger one point
// -------------------------------------------------------------------------------------------------

void triggerElement(size_t element)
{
  // convert plastic-element to element number
  size_t e = m_plastic(element);

  // make sure to start from quenched state
  m_v.fill(0.0);
  m_a.fill(0.0);
  m_stop.reset();

  // current equivalent deviatoric strain
  xt::xtensor<double,2> eps = GM::Epsd(m_Eps);

  // distance to yielding on the positive side
  xt::xtensor<size_t,2> idx  = m_material.Find(m_Eps);
  xt::xtensor<double,2> epsy = m_material.Epsy(idx + 1ul);
  xt::xtensor<double,2> deps = eps - epsy;

  // find integration point closest to yielding
  // - isolate element
  xt::xtensor<double,1> deps_e = xt::view(deps, e, xt::all());
  // - get integration point
  auto q = xt::argmin(xt::abs(deps_e))[0];

  // extract (equivalent) deviatoric strain at quadrature-point "(e,q)"
  auto Epsd = xt::view(GM::Deviatoric(m_Eps), e, q);

  // new equivalent deviatoric strain: yield strain + small strain kick
  double eps_new = epsy(e,q) + m_deps_kick/2.;

  // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
  double dgamma = 2. * (-Epsd(0,1) + std::sqrt(std::pow(eps_new,2.) - std::pow(Epsd(0,0),2.)));

  // apply increment in shear strain as a perturbation to the selected element
  // - nodes belonging to the current element, from connectivity
  auto nodes = xt::view(m_conn, e, xt::all());
  // - displacement-DOFs
  xt::xtensor<double,1> udofs = m_vector.AsDofs(m_u);
  // - update displacement-DOFs for the element
  for (size_t n = 0; n < m_nne; ++n)
    udofs(m_dofs(nodes(n),0)) += dgamma * (m_coor(nodes(n),1) - m_coor(nodes(0),1));
  // - convert displacement-DOFs to (periodic) nodal displacement vector
  //   (N.B. storing to nodes directly does not ensure periodicity)
  m_vector.asNode(udofs, m_u);

  // compute strain/stress
  computeStrainStress();
}

// -------------------------------------------------------------------------------------------------
// run
// -------------------------------------------------------------------------------------------------

void run(size_t element, size_t inc_c, const std::string& output)
{
    // extract a list with increments at which to start elastic loading
    xt::xtensor<size_t,1> inc_system = getIncPush();
    MYASSERT(inc_system.size() > 0);
    MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

    // set increment
    m_inc = inc_c;

    // restore displacement
    xt::noalias(m_u) = H5Easy::load<xt::xtensor<double,2>>(m_file, "/disp/"+std::to_string(m_inc));
    computeStrainStress();

    // extract "id" from filename (stored to data)
    std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
    size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

    // extract information needed for storage
    size_t N = m_plastic.size();
    xt::xtensor<double,4> dV = m_quad.DV(2);
    xt::xtensor<int,1> idx_last = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);
    xt::xtensor<int,1> idx_n = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);
    xt::xtensor<int,1> idx = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);

    // perturb the displacement of the set element, to (try to) trigger an avalanche
    triggerElement(element);

    // clear/open the output file
    H5Easy::File data(output, H5Easy::File::Overwrite);

    // storage parameters
    int S = 0; // current avalanche size
    size_t S_next = 0; // next avalanche size to store
    size_t S_index = 0; // index of store avalanche size
    size_t S_step = 50; // avalanche size at which to store
    size_t A = 0; // current crack area
    size_t A_next = 0; // next crack area to store
    size_t A_index = 0; // index of store crack area
    bool A_store = true; // store synchronised on "A"
    size_t t_step = 2000; // time-step interval at which to store
    size_t istore = 0; // storage index
    size_t ievent = 0;
    bool last = false;

    // quench: force equilibrium
    for (size_t iiter = 0; ; ++iiter)
    {
        // update state
        if (iiter > 0) {
            xt::noalias(idx) = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);
        }

        // get the 'crack' size (disallow crack to shrink to avoid storage issues in rare cases)
        size_t a = xt::sum(xt::not_equal(idx, idx_n))[0];
        int s = xt::sum(idx - idx_n)[0];
        A = std::max(A, a);
        S = std::max(S, s);

        // keep last yielding configuration
        if (xt::any(xt::not_equal(idx, idx_last))) {
            xt::xtensor<size_t,1> r = xt::flatten_indices(xt::argwhere(xt::not_equal(idx, idx_last)));
            for (size_t i = 0; i < r.size(); ++i) {
                H5Easy::dump(data, "/event/iiter", iiter, {ievent});
                H5Easy::dump(data, "/event/s", idx(r(i)) - idx_last(r(i)), {ievent});
                H5Easy::dump(data, "/event/r", r(i), {ievent});
                ievent++;
            }
            xt::noalias(idx_last) = idx;
        }

        // store
        if (((A >= A_next || A == N) && A_store) || S >= (int)S_next || iiter % t_step == 0 || last || iiter == 0)
        {
            // macroscopic stress tensor
            xt::xtensor_fixed<double, xt::xshape<2,2>> Sig_bar = xt::average(m_Sig, dV, {0, 1});

            // element stress tensor
            xt::xtensor<double,3> Sig_elem = xt::average(m_Sig, dV, {1});
            xt::xtensor<double,2> Sig_plas = xt::empty<double>({3ul, N});
            xt::view(Sig_plas, 0, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 0, 0);
            xt::view(Sig_plas, 1, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 0, 1);
            xt::view(Sig_plas, 2, xt::all()) = xt::view(Sig_elem, xt::keep(m_plastic), 1, 1);

            // store
            H5Easy::dump(data, "/global/sig", Sig_bar(0, 0), {0, istore});
            H5Easy::dump(data, "/global/sig", Sig_bar(0, 1), {1, istore});
            H5Easy::dump(data, "/global/sig", Sig_bar(1, 1), {2, istore});
            H5Easy::dump(data, "/global/iiter", iiter, {istore});
            H5Easy::dump(data, fmt::format("/plastic/{0:d}/sig", istore), Sig_plas);
            H5Easy::dump(data, fmt::format("/plastic/{0:d}/idx", istore), idx);
            if ((A >= A_next || A == N) && A_store) {
                H5Easy::dump(data, "/storage/A/stored", A, {A_index});
                H5Easy::dump(data, "/storage/A/index", istore, {A_index});
                A_next++;
                A_index++;
                if (A >= N) {
                    A_store = false;
                }
            }
            if (S >= (int)S_next) {
                H5Easy::dump(data, "/storage/S/stored", S, {S_index});
                H5Easy::dump(data, "/storage/S/index", istore, {S_index});
                S_next += S_step;
                S_index++;
            }
            if (iiter % t_step == 0) {
                size_t j = iiter / t_step;
                H5Easy::dump(data, "/storage/iiter/stored", iiter, {j});
                H5Easy::dump(data, "/storage/iiter/index", istore, {j});
            }

            if (iiter == 0) {
                H5Easy::dumpAttribute(data, "/global/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/global/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/global/sig", "yy", static_cast<size_t>(2));
                H5Easy::dumpAttribute(data, "/global/sig", "desc",
                    std::string("Macroscopic stress tensor: each row corresponds to a different component (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, "/global/sig", "shape", std::string("3, nstore"));
                H5Easy::dumpAttribute(data, "/global/iiter", "desc", std::string("Iteration number"));
                H5Easy::dumpAttribute(data, "/global/iiter", "shape", std::string("nstore"));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/sig", istore), "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/sig", istore), "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/sig", istore), "yy", static_cast<size_t>(2));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/sig", istore), "desc",
                    std::string("Stress tensor along the weak layer: each row corresponds to a different component (xx, xy, yy)"));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/sig", istore), "shape", std::string("3, N"));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/idx", istore), "desc",
                    std::string("Index of the current local minimum"));
                H5Easy::dumpAttribute(data, fmt::format("/plastic/{0:d}/idx", istore), "shape", std::string("N"));
                H5Easy::dump(data, "/storage/A/step", 1);
                H5Easy::dump(data, "/storage/S/step", S_step);
                H5Easy::dump(data, "/storage/iiter/step", t_step);
            }

            istore++;
        }

        if (last) {
            break;
        }

        // time increment
        timeStep();

        // - check for convergence
        if (m_stop.stop(xt::linalg::norm(m_fres)/xt::linalg::norm(m_fext), 1.e-5)) {
            last = true;
        }
    }

    H5Easy::dump(data, "/meta/completed", 1);
    H5Easy::dump(data, "/meta/uuid", H5Easy::load<std::string>(m_file, "/uuid"));
    H5Easy::dump(data, "/meta/id", id_num);
    H5Easy::dump(data, "/meta/inc_c", inc_c);
    H5Easy::dump(data, "/meta/element", element);
    H5Easy::dump(data, "/meta/dt", m_dt);
    H5Easy::dump(data, "/meta/plastic", m_plastic);
    H5Easy::dump(data, "/meta/N", N);
}

// -------------------------------------------------------------------------------------------------

};

// =================================================================================================

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
  -h, --help          Show help.
      --version       Show version.

(c) Tom de Geus
)";

// =================================================================================================

int main(int argc, const char** argv)
{
  // parse command-line arguments
  std::map<std::string, docopt::value> args = docopt::docopt(
    USAGE, {argv+1,argv+argc}, true, "v0.0.1"
  );

  // read from command line
  // -
  std::string output = args["--output"].asString();
  std::string file = args["--file"].asString();
  // -
  size_t inc_c = static_cast<size_t>(std::stoi(args["--incc"   ].asString()));
  size_t element = static_cast<size_t>(std::stoi(args["--element"].asString()));

  // initialise simulation
  Main sim(file);

  // run and save
  sim.run(element, inc_c, output);

  return 0;
}
