
#include <docopt/docopt.h>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <GooseFEM/GooseFEM.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <fmt/core.h>
#include <cpppath.h>
#include <xtensor-io/xhighfive.hpp>

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

struct Data
{
  double dt; // time-step
  double G;
  double rho;
  double eta;
  size_t N;
  size_t steadystate;
  xt::xtensor<size_t,1> incs;
  xt::xtensor<size_t,1> inc_system; // the increment of the last system-spanning event
  xt::xtensor<double,1> sig_system; // equivalent stress of the last system-spanning event
  xt::xtensor<double,1> eps_system; // equivalent stress of the last system-spanning event
  xt::xtensor<size_t,1> kick;
  xt::xtensor<size_t,1> S;
  xt::xtensor<size_t,1> A;
  xt::xtensor<size_t,1> xi;
  xt::xtensor<double,1> depsp;
  xt::xtensor<double,1> dt_avalanche;
  xt::xtensor<double,1> epsd;
  xt::xtensor<double,1> sigd;
};

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

  // time-step
  double dt;

  // event-driven settings
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
  MYASSERT(val_elem.size() == nelem);

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({nelem, nodalQuad.nip()});
  // - copy
  for (size_t e = 0; e < nelem; ++e)
    for (size_t q = 0; q < nodalQuad.nip(); ++q)
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
  MYASSERT(val_elem.size() == nelem);

  // integration point values (constant per element)
  // - allocate
  xt::xtensor<double,2> val_quad = xt::empty<double>({nelem, nodalQuad.nip()});
  // - copy
  for (size_t e = 0; e < nelem; ++e)
    for (size_t q = 0; q < nodalQuad.nip(); ++q)
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

    xt::xtensor<size_t,2> I, idx;

    std::tie(I, idx) = indicator(elem);

    material.setElastic(I, idx, k, g);
  }

  // add plastic-cusp elements
  {
    xt::xtensor<size_t,1> elem = xt::load<xt::xtensor<size_t,1>>(file, "/cusp/elem");
    xt::xtensor<double,1> k    = xt::load<xt::xtensor<double,1>>(file, "/cusp/K"   );
    xt::xtensor<double,1> g    = xt::load<xt::xtensor<double,1>>(file, "/cusp/G"   );
    xt::xtensor<double,2> y    = xt::load<xt::xtensor<double,2>>(file, "/cusp/epsy");

    xt::xtensor<size_t,2> I, idx;

    std::tie(I, idx) = indicator(elem);

    material.setCusp(I, idx, k, g, y);
  }

  // check homogeneous elasticity
  xt::xtensor<double,2> k = material.K();
  xt::xtensor<double,2> g = material.G();
  // -
  MYASSERT(xt::mean(k)[0] == k(0,0));
  MYASSERT(xt::mean(g)[0] == g(0,0));

  // check full material allocation
  material.check();

  // plastic elements
  plastic = xt::sort(xt::flatten_indices(xt::argwhere(xt::amin(material.isPlastic(),{1}))));
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
// run
// -------------------------------------------------------------------------------------------------

Data run()
{
  // initialise output
  Data data;

  // mass density and viscous damping
  data.rho = xt::load<double>(file, "/rho"          , {0});
  data.eta = xt::load<double>(file, "/damping/eta_d", {0});

  // integration point volume
  xt::xtensor<double,4> dV = quad.DV(2);

  // number of plastic cells
  size_t N = plastic.size();

  // basic information for each increment
  xt::xtensor<size_t,1> stored       = xt::load<xt::xtensor<size_t,1>>(file, "/stored");
  xt::xtensor<size_t,1> kick         = xt::load<xt::xtensor<size_t,1>>(file, "/kick");
  xt::xtensor<double,1> dt_avalanche = xt::zeros<double>(stored.shape());

  if ( xt::extensions::exist(file, "/dt/plastic") )
    dt_avalanche = xt::load<xt::xtensor<double,1>>(file, "/dt/plastic");

  // allocate result
  xt::xtensor<size_t,1> S     = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<size_t,1> A     = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<size_t,1> xi    = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> depsp = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> epsd  = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> sigd  = xt::zeros<double>({xt::amax(stored)[0]+1});

  // index of the current quadratic potential and the plastic strain,
  // for the first integration point per plastic element
  auto idx_n  = xt::view(material.Find(Eps), xt::keep(plastic), 0);
  auto epsp_n = xt::view(material.Epsp(Eps), xt::keep(plastic), 0);

  // loop over increments
  for (size_t istored = 0 ; istored < stored.size() ; ++istored)
  {
    // - get increment number
    size_t inc = stored(istored);

    // - restore displacement
    xt::noalias(u) = xt::load<xt::xtensor<double,2>>(file, "/disp/"+std::to_string(inc));

    // - update strain/strain
    computeStrainStress();

    // - index of the current quadratic potential and the plastic strain
    auto idx  = xt::view(material.Find(Eps), xt::keep(plastic), 0);
    auto epsp = xt::view(material.Epsp(Eps), xt::keep(plastic), 0);

    // - macroscopic strain/stress tensor
    xt::xtensor_fixed<double, xt::xshape<2,2>> Epsbar = xt::average(Eps, dV, {0,1});
    xt::xtensor_fixed<double, xt::xshape<2,2>> Sigbar = xt::average(Sig, dV, {0,1});

    // - macroscopic equivalent strain/stress
    epsd(inc) = GM::Epsd(Epsbar);
    sigd(inc) = GM::Sigd(Sigbar);

    // - plastic strain increment
    depsp(inc) = xt::sum(epsp-epsp_n)[0];

    // - avalanche size
    S(inc) = xt::sum(idx-idx_n)[0];

    // - avalanche area
    A(inc) = xt::sum(xt::not_equal(idx,idx_n))[0];

    // - cells where yielding occurred
    xt::xtensor<size_t,1> icell = xt::flatten_indices(xt::argwhere(xt::not_equal(idx_n,idx)));

    // - avalanche linear extension, accounting for periodicity
    if (icell.size() > 0)
    {
      xt::xtensor<size_t,1> rep = {icell(0) + N};
      icell = xt::concatenate(xt::xtuple(icell, rep));
      xi(inc) = N - (xt::amax(xt::diff(icell))[0] - 1);
    }

    // - update history
    idx_n  = idx;
    epsp_n = epsp;
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
  // - always start the steady-state by elastic loading
  if ( kick(steadystate) ) steadystate += 1;

  // per increment, list which increment had the last system-spanning avalanche
  // - list with increment of system-spanning avalanches
  xt::xtensor<size_t,1> isys = xt::flatten_indices(xt::argwhere(xt::equal(A, N)));
  // - allocate
  data.inc_system = xt::zeros<size_t>(stored.shape());
  // - fill
  for (size_t i = 0; i < isys.size()-1; ++i)
    xt::view(data.inc_system, xt::range(isys(i), isys(i+1))) = isys(i);
  // - last entry
  size_t i = isys(isys.size()-1);
  xt::view(data.inc_system, xt::range(i, stored.size())) = i;

  // get information about the last system spanning avalanche
  // - allocate
  data.eps_system = xt::zeros<double>(stored.shape());
  data.sig_system = xt::zeros<double>(stored.shape());
  // - fill
  for (size_t i = 0; i < stored.size(); ++i) {
    data.eps_system(i) = epsd(data.inc_system(i));
    data.sig_system(i) = sigd(data.inc_system(i));
  }

  // return output
  data.dt           = dt;
  data.G            = G;
  data.N            = N;
  data.steadystate  = steadystate;
  data.incs         = stored;
  data.kick         = kick;
  data.S            = S;
  data.A            = A;
  data.xi           = xi;
  data.depsp        = depsp;
  data.dt_avalanche = dt_avalanche;
  data.epsd         = epsd;
  data.sigd         = sigd;

  return data;
}

// -------------------------------------------------------------------------------------------------

};

// =================================================================================================

static const char USAGE[] =
R"(Run
  Read information (avalanche size, stress, strain, ...) of an ensemble.

Output:
  /files              List with files in the ensemble (path relative to the current file).
  /full/file/...      The raw-output (including not-steady-state) per file (see "Fields" below).
  /loading/...        Output after an elastic loading increment (see "Fields" below).
  /avalanche/...      Output after a strain kick, i.e. after an avalanche  (see "Fields" below).
  /normalisation/...  Normalisation factors (see "Normalisation" below).
  /averages/...       Ensemble averages (see "Averages" below).

Fields:
  ../ninc             Number of increments per file (*).
  ../inc              Increment number for each entry (*).
  ../file             Index in "/files" (*)
  ../kick             Increment was: 1 - a kick, or 0 - an elastic step.
  ../steadystate      First steady-state increment (+).
  ../S                Avalanche size.
  ../A                Avalanche area of all avalanches.
  ../xi               Avalanche linear extension of all avalanches.
  ../depsp            Incremental plastic strain.
  ../dt_avalanche     Avalanche duration.
  ../epsd             Current equivalent deviatoric strain (volume average).
  ../sigd             Current equivalent deviatoric stress (volume average).
  (*) Not in raw-output, numbering implicitly sequential.
  (+) Only in raw-output.

Normalisation:
  ../N                Number of plastic elements.
  ../dt               Time-step per increment.
  ../epsy             Ensemble average yield strain (see input).
  ../sigy             Ensemble average yield strain (from input).
  ../G                Shear modulus (homogeneous).
  ../rho              Mass density (homogeneous).
  ../nu               Damping factor in shear (homogeneous).
  ../cs               Shear wave speed (homogeneous).
  ../t0               Time it takes a shear wave to travel the size of an element (see input).

Averages:
  ../sigd_top         Stress just before a system-spanning avalanche.
  ../sigd_bottom      Stress directly after a system-spanning avalanche.

Usage:
  Run [options] <data.hdf5>...

Options:
      --outpath=N     Output file-path. Default: the path common to the input-files. [default: ]
      --outname=N     Output file-name. [default: EnsembleInfo.hdf5]
      --epsy=N        Average yield strain: normalises strain/stress. [default: 5.0e-4]
      --l0=N          Discretisation size. [default: 3.14159]
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

  // list of files
  std::vector<std::string> files = args["<data.hdf5>"].asStringList();

  // skip empty input
  if ( files.size() == 0 ) return 0;

  // output name
  // - command-line options
  std::string outpath = args["--outpath"].asString();
  std::string outname = args["--outname"].asString();
  // - default path
  if ( outpath.size() == 0 ){
    outpath = cpppath::common_dirname(files);
  }
  // - convert name to path
  outname = cpppath::join({outpath, outname});

  // strip files from output-path
  // - allocate
  std::vector<std::string> sims;
  // - fill
  for ( auto &file: files )
    sims.push_back(cpppath::split(file, outpath+"/")[0]);

  // normalisation parameters
  double epsy = std::stod(args["--epsy"].asString());
  double l0   = std::stod(args["--l0"  ].asString());

  // results: ensemble after loading
  xt::xtensor<size_t,1> load_sim          = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> load_incs         = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> load_kick         = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> load_inc_system   = xt::empty<size_t>({0});
  xt::xtensor<double,1> load_eps_system   = xt::empty<double>({0});
  xt::xtensor<double,1> load_sig_system   = xt::empty<double>({0});
  xt::xtensor<size_t,1> load_S            = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> load_A            = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> load_xi           = xt::empty<size_t>({0});
  xt::xtensor<double,1> load_depsp        = xt::empty<double>({0});
  xt::xtensor<double,1> load_dt_avalanche = xt::empty<double>({0});
  xt::xtensor<double,1> load_epsd         = xt::empty<double>({0});
  xt::xtensor<double,1> load_sigd         = xt::empty<double>({0});

  // results: ensemble after avalanche
  xt::xtensor<size_t,1> aval_sim          = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> aval_incs         = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> aval_inc_system   = xt::empty<size_t>({0});
  xt::xtensor<double,1> aval_eps_system   = xt::empty<double>({0});
  xt::xtensor<double,1> aval_sig_system   = xt::empty<double>({0});
  xt::xtensor<size_t,1> aval_kick         = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> aval_S            = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> aval_A            = xt::empty<size_t>({0});
  xt::xtensor<size_t,1> aval_xi           = xt::empty<size_t>({0});
  xt::xtensor<double,1> aval_depsp        = xt::empty<double>({0});
  xt::xtensor<double,1> aval_dt_avalanche = xt::empty<double>({0});
  xt::xtensor<double,1> aval_epsd         = xt::empty<double>({0});
  xt::xtensor<double,1> aval_sigd         = xt::empty<double>({0});

  // number of increments
  xt::xtensor<size_t,1> load_ninc = xt::zeros<size_t>({files.size()});
  xt::xtensor<size_t,1> aval_ninc = xt::zeros<size_t>({files.size()});

  // normalisation
  xt::xtensor<size_t,1> norm_N    = xt::zeros<size_t>({files.size()});
  xt::xtensor<double,1> norm_epsy = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_sigy = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_G    = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_rho  = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_nu   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_cs   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_t0   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_dt   = xt::zeros<double>({files.size()});

  // print progress to screen
  fmt::print("Writing to '{0:s}'\n", outname);

  // open output file
  HighFive::File out(outname, HighFive::File::Overwrite);

  // loop over simulations
  for ( size_t i = 0 ; i < files.size() ; ++i )
  {
    // print progress to screen
    fmt::print("- reading '{0:s}'\n", files[i]);

    // open simulation
    Main sim(files[i]);

    // read results
    Data data = sim.run();

    // alias
    size_t steadystate = data.steadystate;

    // typical stress: ensemble average yield stress
    // (normalisation only)
    double sigy = 2. * data.G * epsy;

    // damping factor
    // (normalisation only)
    double nu = data.eta / data.rho;

    // typical time-scale: the time that it takes one shear wave to travel one cell
    // (normalisation only)
    double cs = std::pow(data.G / (2.0 * data.rho), 0.5);
    double t0 = l0 / cs;

    // normalise relevant fields
    data.sig_system   /= sigy;
    data.eps_system   /= epsy;
    data.sigd         /= sigy;
    data.epsd         /= epsy;
    data.depsp        /= epsy;
    data.dt_avalanche /= t0;

    // store normalisation to ensemble
    norm_N   (i) = data.N;
    norm_epsy(i) = epsy;
    norm_sigy(i) = sigy;
    norm_G   (i) = data.G;
    norm_rho (i) = data.rho;
    norm_nu  (i) = nu;
    norm_cs  (i) = cs;
    norm_t0  (i) = t0;
    norm_dt  (i) = data.dt;

    // storage path
    std::string path = cpppath::normpath(fmt::format("/full/{0:s}", sims[i]));

    // save raw data (in addition to the ensemble data), for fast recovery
    xt::dump(out, path+"/steadystate" , steadystate      );
    xt::dump(out, path+"/kick"        , data.kick        );
    xt::dump(out, path+"/S"           , data.S           );
    xt::dump(out, path+"/A"           , data.A           );
    xt::dump(out, path+"/xi"          , data.xi          );
    xt::dump(out, path+"/depsp"       , data.depsp       );
    xt::dump(out, path+"/dt_avalanche", data.dt_avalanche);
    xt::dump(out, path+"/epsd"        , data.epsd        );
    xt::dump(out, path+"/sigd"        , data.sigd        );

    // total number of increments
    size_t ni = data.incs.size();

    // below the increments are split (based on the event-driven protocol) in elastic loading
    // and (potential) avalanches
    // for simplicity this split is done such that the number of increments is the same for both
    // (hence potentially the last increment has to be discard)
    if ( steadystate % 2 == 0 and ni % 2 != 0 ) ni -= 1;
    if ( steadystate % 2 != 0 and ni % 2 == 0 ) ni -= 1;

    // ensemble after loading
    load_incs         = xt::concatenate(xt::xtuple(load_incs        , xt::view(data.incs        , xt::range(steadystate  , ni, 2))));
    load_kick         = xt::concatenate(xt::xtuple(load_kick        , xt::view(data.kick        , xt::range(steadystate  , ni, 2))));
    load_inc_system   = xt::concatenate(xt::xtuple(load_inc_system  , xt::view(data.inc_system  , xt::range(steadystate  , ni, 2))));
    load_eps_system   = xt::concatenate(xt::xtuple(load_eps_system  , xt::view(data.eps_system  , xt::range(steadystate  , ni, 2))));
    load_sig_system   = xt::concatenate(xt::xtuple(load_sig_system  , xt::view(data.sig_system  , xt::range(steadystate  , ni, 2))));
    load_S            = xt::concatenate(xt::xtuple(load_S           , xt::view(data.S           , xt::range(steadystate  , ni, 2))));
    load_A            = xt::concatenate(xt::xtuple(load_A           , xt::view(data.A           , xt::range(steadystate  , ni, 2))));
    load_xi           = xt::concatenate(xt::xtuple(load_xi          , xt::view(data.xi          , xt::range(steadystate  , ni, 2))));
    load_depsp        = xt::concatenate(xt::xtuple(load_depsp       , xt::view(data.depsp       , xt::range(steadystate  , ni, 2))));
    load_dt_avalanche = xt::concatenate(xt::xtuple(load_dt_avalanche, xt::view(data.dt_avalanche, xt::range(steadystate  , ni, 2))));
    load_epsd         = xt::concatenate(xt::xtuple(load_epsd        , xt::view(data.epsd        , xt::range(steadystate  , ni, 2))));
    load_sigd         = xt::concatenate(xt::xtuple(load_sigd        , xt::view(data.sigd        , xt::range(steadystate  , ni, 2))));

    // ensemble after avalanche
    aval_incs         = xt::concatenate(xt::xtuple(aval_incs        , xt::view(data.incs        , xt::range(steadystate+1, ni, 2))));
    aval_kick         = xt::concatenate(xt::xtuple(aval_kick        , xt::view(data.kick        , xt::range(steadystate+1, ni, 2))));
    aval_inc_system   = xt::concatenate(xt::xtuple(aval_inc_system  , xt::view(data.inc_system  , xt::range(steadystate+1, ni, 2))));
    aval_eps_system   = xt::concatenate(xt::xtuple(aval_eps_system  , xt::view(data.eps_system  , xt::range(steadystate+1, ni, 2))));
    aval_sig_system   = xt::concatenate(xt::xtuple(aval_sig_system  , xt::view(data.sig_system  , xt::range(steadystate+1, ni, 2))));
    aval_S            = xt::concatenate(xt::xtuple(aval_S           , xt::view(data.S           , xt::range(steadystate+1, ni, 2))));
    aval_A            = xt::concatenate(xt::xtuple(aval_A           , xt::view(data.A           , xt::range(steadystate+1, ni, 2))));
    aval_xi           = xt::concatenate(xt::xtuple(aval_xi          , xt::view(data.xi          , xt::range(steadystate+1, ni, 2))));
    aval_depsp        = xt::concatenate(xt::xtuple(aval_depsp       , xt::view(data.depsp       , xt::range(steadystate+1, ni, 2))));
    aval_dt_avalanche = xt::concatenate(xt::xtuple(aval_dt_avalanche, xt::view(data.dt_avalanche, xt::range(steadystate+1, ni, 2))));
    aval_epsd         = xt::concatenate(xt::xtuple(aval_epsd        , xt::view(data.epsd        , xt::range(steadystate+1, ni, 2))));
    aval_sigd         = xt::concatenate(xt::xtuple(aval_sigd        , xt::view(data.sigd        , xt::range(steadystate+1, ni, 2))));

    // store file-index
    load_sim = xt::concatenate(xt::xtuple(load_sim, i * xt::ones<size_t>({(ni - steadystate)/2})));
    aval_sim = xt::concatenate(xt::xtuple(aval_sim, i * xt::ones<size_t>({(ni - steadystate)/2})));

    // get number of increments in each set
    size_t n = load_incs.size();

    // check split: kicks
    MYASSERT(xt::all(xt::equal(load_kick, 0ul)));
    MYASSERT(xt::all(xt::equal(aval_kick, 1ul)));

    // check split: size
    MYASSERT(load_incs.size() == n);
    MYASSERT(aval_incs.size() == n);

    // store number of increments to ensemble
    load_ninc(i) = n;
    aval_ninc(i) = n;
  }

  // check normalisation
  MYASSERT(xt::all(xt::equal(norm_N   , norm_N   (0))));
  MYASSERT(xt::all(xt::equal(norm_epsy, norm_epsy(0))));
  MYASSERT(xt::all(xt::equal(norm_sigy, norm_sigy(0))));
  MYASSERT(xt::all(xt::equal(norm_G   , norm_G   (0))));
  MYASSERT(xt::all(xt::equal(norm_rho , norm_rho (0))));
  MYASSERT(xt::all(xt::equal(norm_nu  , norm_nu  (0))));
  MYASSERT(xt::all(xt::equal(norm_cs  , norm_cs  (0))));
  MYASSERT(xt::all(xt::equal(norm_t0  , norm_t0  (0))));
  MYASSERT(xt::all(xt::equal(norm_dt  , norm_dt  (0))));

  // save ensemble
  xt::dump(out, "/files", sims);

  xt::dump(out, "/normalisation/N"   , static_cast<size_t>(norm_N   (0)));
  xt::dump(out, "/normalisation/epsy", static_cast<double>(norm_epsy(0)));
  xt::dump(out, "/normalisation/sigy", static_cast<double>(norm_sigy(0)));
  xt::dump(out, "/normalisation/G"   , static_cast<double>(norm_G   (0)));
  xt::dump(out, "/normalisation/rho" , static_cast<double>(norm_rho (0)));
  xt::dump(out, "/normalisation/nu"  , static_cast<double>(norm_nu  (0)));
  xt::dump(out, "/normalisation/cs"  , static_cast<double>(norm_cs  (0)));
  xt::dump(out, "/normalisation/t0"  , static_cast<double>(norm_t0  (0)));
  xt::dump(out, "/normalisation/l0"  , l0);
  xt::dump(out, "/normalisation/dt"  , static_cast<double>(norm_dt  (0)));

  xt::dump(out, "/loading/file"          , load_sim         );
  xt::dump(out, "/loading/inc"           , load_incs        );
  xt::dump(out, "/loading/ninc"          , load_ninc        );
  xt::dump(out, "/loading/kick"          , load_kick        );
  xt::dump(out, "/loading/inc_system"    , load_inc_system  );
  xt::dump(out, "/loading/eps_system"    , load_eps_system  );
  xt::dump(out, "/loading/sig_system"    , load_sig_system  );
  xt::dump(out, "/loading/S"             , load_S           );
  xt::dump(out, "/loading/A"             , load_A           );
  xt::dump(out, "/loading/xi"            , load_xi          );
  xt::dump(out, "/loading/depsp"         , load_depsp       );
  xt::dump(out, "/loading/dt_avalanche"  , load_dt_avalanche);
  xt::dump(out, "/loading/epsd"          , load_epsd        );
  xt::dump(out, "/loading/sigd"          , load_sigd        );

  xt::dump(out, "/avalanche/file"        , aval_sim         );
  xt::dump(out, "/avalanche/inc"         , aval_incs        );
  xt::dump(out, "/avalanche/ninc"        , aval_ninc        );
  xt::dump(out, "/avalanche/kick"        , aval_kick        );
  xt::dump(out, "/avalanche/inc_system"  , aval_inc_system  );
  xt::dump(out, "/avalanche/eps_system"  , aval_eps_system  );
  xt::dump(out, "/avalanche/sig_system"  , aval_sig_system  );
  xt::dump(out, "/avalanche/S"           , aval_S           );
  xt::dump(out, "/avalanche/A"           , aval_A           );
  xt::dump(out, "/avalanche/xi"          , aval_xi          );
  xt::dump(out, "/avalanche/depsp"       , aval_depsp       );
  xt::dump(out, "/avalanche/dt_avalanche", aval_dt_avalanche);
  xt::dump(out, "/avalanche/epsd"        , aval_epsd        );
  xt::dump(out, "/avalanche/sigd"        , aval_sigd        );

  // get ensemble averages
  // - size of the weak layer
  size_t N = xt::load<size_t>(out, "/normalisation/N");
  // - area of the avalanche
  xt::xtensor<size_t,1> A = xt::load<xt::xtensor<size_t,1>>(out, "/avalanche/A");
  // - area of the avalanche, and the stress just before and just after each avalanche
  xt::xtensor<double,1> sig_bot = xt::load<xt::xtensor<double,1>>(out, "/avalanche/sigd");
  xt::xtensor<size_t,1> inc_bot = xt::load<xt::xtensor<size_t,1>>(out, "/avalanche/inc" );
  xt::xtensor<double,1> sig_top = xt::load<xt::xtensor<double,1>>(out, "/loading/sigd"  );
  xt::xtensor<size_t,1> inc_top = xt::load<xt::xtensor<size_t,1>>(out, "/loading/inc"   );
  // - filter for system-spanning avalanches
  xt::xtensor<size_t,1> idx = xt::flatten_indices(xt::argwhere(xt::equal(A,N)));
  // - compute averages
  if ( idx.size() > 0 )
  {
    // - extract
    sig_top = xt::view(sig_top, xt::keep(idx));
    sig_bot = xt::view(sig_bot, xt::keep(idx));
    inc_bot = xt::view(inc_bot, xt::keep(idx));
    inc_top = xt::view(inc_top, xt::keep(idx));
    // - check increments numbers
    MYASSERT(xt::all(xt::equal(inc_bot, inc_top+1)));
    // - compute averages
    xt::dump(out, "/averages/sigd_top"   , xt::mean(sig_top)[0]);
    xt::dump(out, "/averages/sigd_bottom", xt::mean(sig_bot)[0]);
  }

  return 0;
}
