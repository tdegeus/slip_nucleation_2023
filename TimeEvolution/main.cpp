
#include <docopt/docopt.h>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <GooseFEM/GooseFEM.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <xtensor-io/xhighfive.hpp>
#include <fmt/core.h>
#include <cpppath.h>
#include <xtensor/xhistogram.hpp>
#include <xtensor/xindex_view.hpp>

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
  HighFive::File file;

  // mesh parameters
  xt::xtensor<size_t,2> conn;
  xt::xtensor<double,2> coor;
  xt::xtensor<size_t,2> dofs;
  xt::xtensor<size_t,1> iip;

  // mesh dimensions
  size_t nelem;
  size_t nne;
  size_t nnode;
  size_t ndim;
  size_t nip;

  // numerical quadrature
  QD::Quadrature quad;

  // convert vectors between 'nodevec', 'elemvec', ...
  GF::VectorPartitioned vector;

  // mass matrix
  GF::MatrixDiagonalPartitioned M;

  // damping matrix
  GF::MatrixDiagonal D;

  // material definition
  GM::Matrix material;

  // convergence check
  GF::Iterate::StopList stop;

  // time(-step), time of an avalanche
  double t=0.0;
  double dt;

  // event-driven settings
  size_t inc=0;                  // current increment
  size_t iiter=0;                // last number of iterations
  double deps_kick;              // equivalent strain increment
  xt::xtensor<size_t,1> plastic; // plastic elements

  // nodal displacements, velocities, and accelerations (current and last time-step)
  xt::xtensor<double,2> u;
  xt::xtensor<double,2> v;
  xt::xtensor<double,2> a;
  xt::xtensor<double,2> v_n;
  xt::xtensor<double,2> a_n;

  // element vectors
  xt::xtensor<double,3> ue;
  xt::xtensor<double,3> fe;

  // nodal forces
  xt::xtensor<double,2> felas;
  xt::xtensor<double,2> fdamp;
  xt::xtensor<double,2> fint;
  xt::xtensor<double,2> fext;
  xt::xtensor<double,2> fres;

  // integration point strain and stress
  xt::xtensor<double,4> Eps;
  xt::xtensor<double,4> Sig;

  // parameters
  double G; // shear modulus (homogeneous)

  // push settings
  double stress;                    // stress at which to measure
  xt::xtensor<size_t,1> inc_push;   // increments from which to load elastically to the fixed stress
  xt::xtensor<size_t,1> inc_system; // increments with system spanning avalanches

// -------------------------------------------------------------------------------------------------

public:

// -------------------------------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------------------------------

Main(const std::string &fname) : file(fname, HighFive::File::ReadOnly)
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
  dt = xt::load<double>(file, "/run/dt");

  // kick size
  deps_kick = xt::load<double>(file, "/run/epsd/kick");

  // initialise stop list
  stop = GF::Iterate::StopList(20);

  // extract shear modulus (homogeneous)
  G = material.G()(0,0);
}

// -------------------------------------------------------------------------------------------------
// read mesh
// -------------------------------------------------------------------------------------------------

void readMesh()
{
  // read fields
  conn = xt::load<xt::xtensor<size_t,2>>(file, "/conn");
  coor = xt::load<xt::xtensor<double,2>>(file, "/coor");
  dofs = xt::load<xt::xtensor<size_t,2>>(file, "/dofs");
  iip  = xt::load<xt::xtensor<size_t,1>>(file, "/dofsP");

  // extract sizes
  nnode = coor.shape()[0];
  ndim  = coor.shape()[1];
  nelem = conn.shape()[0];
  nne   = conn.shape()[1];

  // vector-definition: transform nodal vectors <-> DOF values
  vector = GF::VectorPartitioned(conn, dofs, iip);

  // numerical quadrature: transform displacements -> strains, integrate stresses -> forces, ...
  quad = QD::Quadrature(vector.AsElement(coor));
  nip  = quad.nip();

  // nodal displacements, velocities, and accelerations (current and last time-step)
  u   = xt::zeros<double>(coor.shape());
  v   = xt::zeros<double>(coor.shape());
  a   = xt::zeros<double>(coor.shape());
  v_n = xt::zeros<double>(coor.shape());
  a_n = xt::zeros<double>(coor.shape());

  // element vectors
  ue = xt::zeros<double>({nelem, nne, ndim});
  fe = xt::zeros<double>({nelem, nne, ndim});

  // nodal forces
  felas = xt::zeros<double>(coor.shape());
  fdamp = xt::zeros<double>(coor.shape());
  fint  = xt::zeros<double>(coor.shape());
  fext  = xt::zeros<double>(coor.shape());
  fres  = xt::zeros<double>(coor.shape());

  // integration point strain and stress
  Eps = xt::zeros<double>({nelem, nip, ndim, ndim});
  Sig = xt::zeros<double>({nelem, nip, ndim, ndim});
}

// -------------------------------------------------------------------------------------------------
// read/set mass matrix
// -------------------------------------------------------------------------------------------------

void setMass()
{
  // allocate
  M = GF::MatrixDiagonalPartitioned(conn, dofs, iip);

  // nodal coordinates as element vector
  xt::xtensor<double,3> x = vector.AsElement(coor);

  // nodal quadrature
  QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

  // element values
  xt::xtensor<double,1> val_elem = xt::load<xt::xtensor<double,1>>(file, "/rho");

  // check size
  if ( static_cast<size_t>(val_elem.size()) != nelem ) throw std::runtime_error("IOError '/rho'");

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({nelem, nodalQuad.nip()});
  // - copy
  for ( size_t e = 0 ; e < nelem ; ++e )
    for ( size_t q = 0 ; q < nodalQuad.nip() ; ++q )
      val_quad(e,q) = val_elem(e);

  // compute diagonal matrices
  M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
}

// -------------------------------------------------------------------------------------------------
// read/set damping matrix
// -------------------------------------------------------------------------------------------------

void setDamping()
{
  // allocate
  D = GF::MatrixDiagonal(conn, dofs);

  // nodal coordinates as element vector
  xt::xtensor<double,3> x = vector.AsElement(coor);

  // nodal quadrature
  QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());

  // element values
  xt::xtensor<double,1> val_elem = xt::load<xt::xtensor<double,1>>(file, "/damping/alpha");

  // check size
  if ( static_cast<size_t>(val_elem.size()) != nelem ) throw std::runtime_error("IOError '/damping/alpha'");

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({nelem, nodalQuad.nip()});
  // - copy
  for ( size_t e = 0 ; e < nelem ; ++e )
    for ( size_t q = 0 ; q < nodalQuad.nip() ; ++q )
      val_quad(e,q) = val_elem(e);

  // compute diagonal matrices
  D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
}

// -------------------------------------------------------------------------------------------------
// get indicator function
// -------------------------------------------------------------------------------------------------

std::tuple<xt::xtensor<size_t,2>,xt::xtensor<size_t,2>> indicator(const xt::xtensor<size_t,1> &elem)
{
  xt::xtensor<size_t,2> I   = xt::zeros<size_t>({nelem, nip});
  xt::xtensor<size_t,2> idx = xt::zeros<size_t>({nelem, nip});

  for ( size_t e = 0 ; e < elem.size() ; ++e ) {
    for ( size_t q = 0 ; q < nip ; ++q ) {
      I  (elem(e), q) = 1;
      idx(elem(e), q) = e;
    }
  }

  return std::make_tuple(I, idx);
}

// -------------------------------------------------------------------------------------------------
// set material definition
// -------------------------------------------------------------------------------------------------

void setMaterial()
{
  // allocate
  material = GM::Matrix(nelem, nip);

  // add elastic elements
  {
    xt::xtensor<size_t,1> elem = xt::load<xt::xtensor<size_t,1>>(file, "/elastic/elem");
    xt::xtensor<double,1> k    = xt::load<xt::xtensor<double,1>>(file, "/elastic/K"   );
    xt::xtensor<double,1> g    = xt::load<xt::xtensor<double,1>>(file, "/elastic/G"   );

    xt::xtensor<size_t,2> I,idx;

    std::tie(I,idx) = indicator(elem);

    material.setElastic(I, idx, k, g);
  }

  // add plastic-cusp elements
  {
    xt::xtensor<size_t,1> elem = xt::load<xt::xtensor<size_t,1>>(file, "/cusp/elem");
    xt::xtensor<double,1> k    = xt::load<xt::xtensor<double,1>>(file, "/cusp/K"   );
    xt::xtensor<double,1> g    = xt::load<xt::xtensor<double,1>>(file, "/cusp/G"   );
    xt::xtensor<double,2> y    = xt::load<xt::xtensor<double,2>>(file, "/cusp/epsy");

    xt::xtensor<size_t,2> I,idx;

    std::tie(I,idx) = indicator(elem);

    material.setCusp(I, idx, k, g, y);
  }

  // check homogeneous elasticity
  xt::xtensor<double,2> k = material.K();
  xt::xtensor<double,2> g = material.G();
  // -
  if ( xt::mean(k)[0] != k(0,0) ) throw std::runtime_error("IOError: non-homogeneous elasticity detected");
  if ( xt::mean(g)[0] != g(0,0) ) throw std::runtime_error("IOError: non-homogeneous elasticity detected");

  // check full material allocation
  material.check();

  // plastic elements
  plastic = xt::sort(xt::flatten_indices(xt::argwhere(xt::amin(material.isPlastic(),{1}))));
}

// -------------------------------------------------------------------------------------------------
// time step using velocity-Verlet algorithm
// -------------------------------------------------------------------------------------------------

void timeStep()
{
  // history

  t += dt;

  xt::noalias(v_n) = v;
  xt::noalias(a_n) = a;

  // new displacement

  xt::noalias(u) = u + dt * v + 0.5 * std::pow(dt,2.) * a;

  // compute strain/strain, and corresponding force

  computeStrainStress();

  quad.int_gradN_dot_tensor2_dV(Sig, fe);
  vector.assembleNode(fe, felas);

  // estimate new velocity, update corresponding force

  xt::noalias(v) = v_n + dt * a_n;

  D.dot(v, fdamp);

  // compute residual force & solve

  xt::noalias(fint) = felas + fdamp;

  vector.copy_p(fint, fext);

  xt::noalias(fres) = fext - fint;

  M.solve(fres, a);

  // re-estimate new velocity, update corresponding force

  xt::noalias(v) = v_n + .5 * dt * ( a_n + a );

  D.dot(v, fdamp);

  // compute residual force & solve

  xt::noalias(fint) = felas + fdamp;

  vector.copy_p(fint, fext);

  xt::noalias(fres) = fext - fint;

  M.solve(fres, a);

  // new velocity, update corresponding force

  xt::noalias(v) = v_n + .5 * dt * ( a_n + a );

  D.dot(v, fdamp);

  // compute residual force & solve

  xt::noalias(fint) = felas + fdamp;

  vector.copy_p(fint, fext);

  xt::noalias(fres) = fext - fint;

  M.solve(fres, a);
}

// -------------------------------------------------------------------------------------------------
// compute strain and stress based on current displacement
// -------------------------------------------------------------------------------------------------

void computeStrainStress()
{
  vector.asElement(u, ue);
  quad.symGradN_vector(ue, Eps);
  material.stress(Eps, Sig);
}

// -------------------------------------------------------------------------------------------------
// get increments from which to elastically increase the strain
// -------------------------------------------------------------------------------------------------

void getIncPush()
{
  // integration point volume
  xt::xtensor<double,4> dV = quad.DV(2);

  // number of plastic cells
  size_t N = plastic.size();

  // basic information for each increment
  xt::xtensor<size_t,1> stored = xt::load<xt::xtensor<size_t,1>>(file, "/stored");
  xt::xtensor<size_t,1> kick   = xt::load<xt::xtensor<size_t,1>>(file, "/kick");

  // allocate result
  xt::xtensor<size_t,1> A    = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> epsd = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> sigd = xt::zeros<double>({xt::amax(stored)[0]+1});

  // index of the current quadratic potential,
  // for the first integration point per plastic element
  auto idx_n = xt::view(material.Find(Eps), xt::keep(plastic), 0);

  // loop over increments
  for ( size_t istored = 0 ; istored < stored.size() ; ++istored )
  {
    // - get increment number
    size_t inc = stored(istored);

    // - restore displacement
    xt::noalias(u) = xt::load<xt::xtensor<double,2>>(file, "/disp/"+std::to_string(inc));

    // - update strain/strain
    computeStrainStress();

    // - index of the current quadratic potential
    auto idx = xt::view(material.Find(Eps), xt::keep(plastic), 0);

    // - macroscopic strain/stress tensor
    xt::xtensor_fixed<double, xt::xshape<2,2>> Epsbar = xt::average(Eps, dV, {0,1});
    xt::xtensor_fixed<double, xt::xshape<2,2>> Sigbar = xt::average(Sig, dV, {0,1});

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
  // - compute
  for ( size_t i = 1 ; i < K.size() ; ++i )
    K(i) = ( sigd(i) - sigd(0) ) / ( epsd(i) - epsd(0) );
  // - set dummy (to take the minimum below)
  K(0) = K(1);
  // - get steady-state increment
  size_t steadystate = xt::amin(xt::from_indices(xt::argwhere(K<=.95*K(1))))[0];
  // - make sure to skip at least two increment (artificial: to avoid checks below)
  steadystate = std::max(steadystate, std::size_t(2));
  // - always start the steady-state by elastic loading
  if ( kick(steadystate) ) steadystate += 1;

  // remove all non-steady-state increments from further consideration
  xt::view(A, xt::range(0, steadystate)) = 0;

  // list with increment of system-spanning avalanches
  inc_system = xt::flatten_indices(xt::argwhere(xt::equal(A,N)));

  // too few system spanning avalanches -> quit
  if ( inc_system.size() < 2 )
  {
    inc_push = xt::zeros<size_t>({0});
    return;
  }

  // allocate list with increment numbers at which to push
  inc_push = xt::zeros<size_t>({inc_system.size()-1});

  // allocate list with all increment numbers
  xt::xtensor<size_t,1> iinc = xt::arange<size_t>(A.size());

  // consider all system spanning avalanches,
  // that are followed by at least one system spanning avalanche
  for ( size_t i = 0 ; i < inc_system.size()-1 ; ++i )
  {
    // - stress after elastic load, kick/area of these increments for checking, increment numbers
    auto s = xt::view(sigd, xt::range(inc_system(i)+1, inc_system(i+1), 2));
    auto k = xt::view(kick, xt::range(inc_system(i)+1, inc_system(i+1), 2));
    auto a = xt::view(A   , xt::range(inc_system(i)+1, inc_system(i+1), 2));
    auto n = xt::view(iinc, xt::range(inc_system(i)+1, inc_system(i+1), 2));

    // - skip if the loading pattern was not load-kick-load-kick-... (sanity check)
    if ( xt::any(xt::not_equal(k,0ul)) ) continue;
    if ( xt::any(xt::not_equal(a,0ul)) ) continue;

    // - find where the strain(stress) is higher than the target strain(stress)
    //   during that increment the strain(stress) elastically moved from below to above the target
    //   strain(stress); the size of this step can be reduced by an arbitrary size, without
    //   violating equilibrium
    auto idx = xt::flatten_indices(xt::argwhere(s > stress));

    // - no increment found -> skip (sanity check)
    if ( idx.size() == 0 ) continue;

    // - start from the increment before it (the beginning of the elastic loading)
    size_t ipush = n(xt::amin(idx)[0]) - 1;

    // - sanity check
    if ( sigd(ipush  ) >  stress ) continue;
    if ( kick(ipush+1) != 0      ) continue;

    // - store
    inc_push(i) = ipush;
  }

  // filter list with increments
  // (zero can never be a valid increment, because of the restriction set above)
  inc_push = xt::filter(inc_push, inc_push > 0ul);
}

// -------------------------------------------------------------------------------------------------
// move forward elastically to a certain equivalent deviatoric stress
// -------------------------------------------------------------------------------------------------

void moveForwardToFixedStress()
{
  // store current minima (for sanity check)
  auto idx_n = material.Find(Eps);

  // integration point volume
  xt::xtensor<double,4> dV = quad.DV(2);

  // macroscopic (deviatoric) stress/strain tensor
  xt::xtensor_fixed<double, xt::xshape<2,2>> Sigbar = xt::average(Sig, dV, {0,1});
  xt::xtensor_fixed<double, xt::xshape<2,2>> Epsbar = xt::average(Eps, dV, {0,1});
  xt::xtensor_fixed<double, xt::xshape<2,2>> Epsd   = GM::Deviatoric(Epsbar);

  // current equivalent deviatoric stress/strain
  double eps = GM::Epsd(Epsbar);
  double sig = GM::Sigd(Sigbar);

  // new equivalent deviatoric strain
  double eps_new = eps + ( stress - sig ) / ( 2. * G );

  // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
  double dgamma = 2. * ( -Epsd(0,1) + std::sqrt( std::pow(eps_new,2.) - std::pow(Epsd(0,0),2.) ) );

  // add as affine deformation gradient to the system
  for ( size_t n = 0 ; n < nnode ; ++n )
    u(n,0) += dgamma * ( coor(n,1) - coor(0,1) );

  // compute strain/stress
  computeStrainStress();

  // compute new macroscopic stress (for sanity check)
  Sigbar = xt::average(Sig, dV, {0,1});
  sig = GM::Sigd(Sigbar);

  // current minima (for sanity check)
  auto idx = material.Find(Eps);

  // check that the stress is what it was set to (sanity check)
  if ( std::abs(stress-sig)/sig > 1.e-4 )
  {
    throw std::runtime_error(fmt::format(
      "fname = '{0:s}', stress = {1:16.8e}, inc = {2:d}: Stress incorrect.\n",
      file.getName(), stress, inc
    ));
  }

  // check that no yielding took place (sanity check)
  if ( xt::any(xt::not_equal(idx,idx_n)) )
  {
    throw std::runtime_error(fmt::format(
      "fname = '{0:s}', stress = {1:16.8e}, inc = {2:d}: Yielding took place where it shouldn't.\n",
      file.getName(), stress, inc
    ));
  }
}

// -------------------------------------------------------------------------------------------------
// trigger one point
// -------------------------------------------------------------------------------------------------

void triggerElement(size_t element)
{
  // convert plastic-element to element number
  size_t e = plastic(element);

  // make sure to start from quenched state
  v.fill(0.0);
  a.fill(0.0);
  stop.reset();

  // current equivalent deviatoric strain
  xt::xtensor<double,2> eps = GM::Epsd(Eps);

  // distance to yielding on the positive side
  xt::xtensor<size_t,2> idx  = material.Find(Eps);
  xt::xtensor<double,2> epsy = material.Epsy(idx+std::size_t(1));
  xt::xtensor<double,2> deps = eps - epsy;

  // find integration point closest to yielding
  // - isolate element
  xt::xtensor<double,1> deps_e = xt::view(deps, e, xt::all());
  // - get integration point
  auto q = xt::argmin(xt::abs(deps_e))[0];

  // extract (equivalent) deviatoric strain at quadrature-point "(e,q)"
  auto Epsd = xt::view(GM::Deviatoric(Eps), e, q);

  // new equivalent deviatoric strain: yield strain + small strain kick
  double eps_new = epsy(e,q) + deps_kick/2.;

  // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
  double dgamma = 2. * ( -Epsd(0,1) + std::sqrt( std::pow(eps_new,2.) - std::pow(Epsd(0,0),2.) ) );

  // apply increment in shear strain as a perturbation to the selected element
  // - nodes belonging to the current element, from connectivity
  auto elem = xt::view(conn, e, xt::all());
  // - displacement-DOFs
  xt::xtensor<double,1> udofs = vector.AsDofs(u);
  // - update displacement-DOFs for the element
  for ( size_t n = 0 ; n < nne ; ++n )
    udofs(dofs(elem(n),0)) += dgamma * ( coor(elem(n),1) - coor(elem(0),1) );
  // - convert displacement-DOFs to (periodic) nodal displacement vector
  //   (N.B. storing to nodes directly does not ensure periodicity)
  vector.asNode(udofs, u);

  // compute strain/stress
  computeStrainStress();
}

// -------------------------------------------------------------------------------------------------
// apply push and minimise energy
// -------------------------------------------------------------------------------------------------

void runPushAndStop(size_t element, size_t inc_c, const std::string& output)
{
  // id
  std::string id = cpppath::split(cpppath::filename(file.getName()), ".")[0];
  size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

  // basic name
  std::string root = fmt::format("{0:s}_element={1:04d}_incc={2:03d}",
    id, element, inc_c);

  // integration point volume
  xt::xtensor<double,4> dV = quad.DV(2);
  xt::xtensor<double,2> dV_scalar = quad.DV();

  // number of plastic cells
  size_t N = plastic.size();

  // element for which to store
  xt::xtensor<size_t,1> store = xt::concatenate(xt::xtuple(
    plastic - 2 * N,
    plastic - 1 * N,
    plastic,
    plastic + 1 * N,
    plastic + 2 * N
  ));

  // store state
  xt::xtensor<int,1> idx_n = xt::view(material.Find(Eps), xt::keep(plastic), 0);
  xt::xtensor<int,1> idx   = xt::view(material.Find(Eps), xt::keep(plastic), 0);

  // perturb the displacement of the set element, to try to trigger an avalanche
  triggerElement(element);

  // clear output file
  HighFive::File data(output, HighFive::File::Overwrite);

  // quench: force equilibrium
  for ( iiter = 0 ; ; ++iiter )
  {
    if (iiter % 100 == 0)
    {
      // - yield strain to the right side
      xt::xtensor<size_t,2> jdx       = material.Find(Eps);
      xt::xtensor<double,2> epsy_p    = material.Epsy(jdx+size_t(1));
      xt::xtensor<double,2> epseq     = GM::Epsd(Eps);
      xt::xtensor<double,2> x         = epsy_p - epseq;
      xt::xtensor<size_t,1> jdx_store = xt::view(jdx, xt::keep(plastic), 0);
      xt::xtensor<double,1> x_store   = xt::view(x,   xt::keep(plastic), 0);

      // - element stress tensor
      xt::xtensor<double,3> Sig_elem  = xt::average(Sig, dV, {1});
      xt::xtensor<double,3> Sig_store = xt::view(Sig_elem, xt::keep(store), xt::all(), xt::all());

      // - macroscopic stress/strain tensor
      xt::xtensor_fixed<double, xt::xshape<2,2>> Sig_bar = xt::average(Sig, dV, {0,1});

      // - equivalent deviatoric stress/strain
      double sigeq_bar = GM::Sigd(Sig_bar);

      // - store output
      xt::dump(data, fmt::format("/{0:s}/stored"   , root           ), iiter/100, {iiter/100});
      xt::dump(data, fmt::format("/{0:s}/iiter"    , root           ), iiter    , {iiter/100});
      xt::dump(data, fmt::format("/{0:s}/sigbar"   , root           ), sigeq_bar, {iiter/100});
      xt::dump(data, fmt::format("/{0:s}/Sig/{1:d}", root, iiter/100), Sig_store             );
      xt::dump(data, fmt::format("/{0:s}/x/{1:d}"  , root, iiter/100), x_store               );
      xt::dump(data, fmt::format("/{0:s}/idx/{1:d}", root, iiter/100), jdx_store             );
    }

    // - time increment
    timeStep();

    // - check for convergence
    if ( stop.stop(xt::linalg::norm(fres)/xt::linalg::norm(fext), 1.e-5) )
      break;

    // - update state
    idx = xt::view(material.Find(Eps), xt::keep(plastic), 0);

    // - truncate at area == 1
    if ( xt::sum(xt::not_equal(idx, idx_n))[0] == N )
      break;
  }

  xt::dump(data, fmt::format("/{0:s}/completed", root), 1                                   );
  xt::dump(data, fmt::format("/{0:s}/uuid"     , root), xt::load<std::string>(file, "/uuid"));
  xt::dump(data, fmt::format("/{0:s}/id"       , root), id_num                              );
  xt::dump(data, fmt::format("/{0:s}/inc_c"    , root), inc_c                               );
  xt::dump(data, fmt::format("/{0:s}/element"  , root), element                             );
  xt::dump(data, fmt::format("/{0:s}/dt"       , root), dt                                  );
}

// -------------------------------------------------------------------------------------------------
// number of increments at which to push
// -------------------------------------------------------------------------------------------------

size_t numberOfPushIncrements_setStress(double Stress)
{
  // store stress
  stress = Stress;

  // extract a list with increments at which to start elastic loading
  getIncPush();

  // return size
  return inc_push.size();
}

// -------------------------------------------------------------------------------------------------
// run
// -------------------------------------------------------------------------------------------------

void run(size_t element, size_t inc_c, const std::string& output)
{
  MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

  // get push increment
  size_t ipush = xt::flatten_indices(xt::argwhere(xt::equal(inc_system, inc_c)))(0);

  // set increment
  inc = inc_push(ipush);

  // check
  MYASSERT(inc >= inc_c);

  // restore displacement
  xt::noalias(u) = xt::load<xt::xtensor<double,2>>(file, "/disp/"+std::to_string(inc));

  // compute strain/stress
  computeStrainStress();

  // increase displacement to set "stress"
  moveForwardToFixedStress();

  // apply push, quench, measure output parameters
  return runPushAndStop(element, inc_c, output);
}

// -------------------------------------------------------------------------------------------------

};

// =================================================================================================

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
  std::string file   = args["--file"].asString();
  // -
  double stress = std::stod(args["--stress"].asString());
  // -
  size_t inc_c   = static_cast<size_t>(std::stoi(args["--incc"   ].asString()));
  size_t element = static_cast<size_t>(std::stoi(args["--element"].asString()));

  // initialise simulation
  Main sim(file);

  // determine (the number of) pushable increments
  size_t ninc = sim.numberOfPushIncrements_setStress(stress);

  // print progress
  fmt::print("fname = '{0:s}', stress: {1:16.8e}, npush = {2:d}: Starting new run.\n",
    file, stress, ninc);

  // run and save
  sim.run(element, inc_c, output);

  return 0;
}
