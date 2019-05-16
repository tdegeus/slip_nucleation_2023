
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

struct Data
{
  std::string           uuid;            // unique identifier
  double                dt;              // time-step
  double                G;               // homogeneous shear modulus
  double                rho;             // homogeneous mass density
  size_t                N;               // number of plastic elements (along the weak layer)
  size_t                steadystate;     // increment number at which the steady-state begins
  bool                  has_steadystate; // signal if a steady-state has been found
  xt::xtensor<size_t,1> incs;            // increment numbers of the output
  xt::xtensor<size_t,1> inc_system;      // per inc: the increment of the last system-spanning event
  xt::xtensor<double,1> sig_system;      // per inc: eq. stress of the last system-spanning event
  xt::xtensor<double,1> eps_system;      // per inc: eq. strain of the last system-spanning event
  xt::xtensor<size_t,1> kick;            // per inc: kick or not (false == elastic loading)
  xt::xtensor<size_t,1> S;               // per inc: avalanche size
  xt::xtensor<size_t,1> A;               // per inc: avalanche area
  xt::xtensor<size_t,1> xi;              // per inc: avalanche spatial extensor
  xt::xtensor<double,1> depsp;           // per inc: plastic strain increment
  xt::xtensor<double,1> dt_avalanche;    // per inc: duration of activity
  xt::xtensor<double,1> epsd;            // per inc: eq. strain
  xt::xtensor<double,1> sigd;            // per inc: eq. stress
};

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

  // material definition
  GM::Matrix m_material;

  // time-step
  double m_dt;

  // event-driven settings
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
// compute strain and stress based on current displacement
// -------------------------------------------------------------------------------------------------

void computeStrainStress()
{
  m_vector.asElement(m_u, m_ue);
  m_quad.symGradN_vector(m_ue, m_Eps);
  m_material.stress(m_Eps, m_Sig);
}

// -------------------------------------------------------------------------------------------------
// run
// -------------------------------------------------------------------------------------------------

Data run()
{
  // initialise output
  Data data;

  // mass density
  data.rho = H5Easy::load<double>(m_file, "/rho", {0});

  // integration point volume
  xt::xtensor<double,4> dV = m_quad.DV(2);

  // number of plastic cells
  size_t N = m_plastic.size();

  // basic information for each increment
  xt::xtensor<size_t,1> stored = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/stored");
  xt::xtensor<size_t,1> kick   = H5Easy::load<xt::xtensor<size_t,1>>(m_file, "/kick");

  // avalanche duration (if stored)
  if (m_file.exist("/dt/plastic"))
    data.dt_avalanche = H5Easy::load<xt::xtensor<double,1>>(m_file, "/dt/plastic");
  else
    data.dt_avalanche = xt::zeros<double>(stored.shape());

  // allocate result
  xt::xtensor<size_t,1> S     = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<size_t,1> A     = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<size_t,1> xi    = xt::zeros<size_t>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> depsp = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> epsd  = xt::zeros<double>({xt::amax(stored)[0]+1});
  xt::xtensor<double,1> sigd  = xt::zeros<double>({xt::amax(stored)[0]+1});

  // index of the current quadratic potential and the plastic strain,
  // for the first integration point per plastic element
  auto idx_n  = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);
  auto epsp_n = xt::view(m_material.Epsp(m_Eps), xt::keep(m_plastic), 0);

  // loop over increments
  for (size_t istored = 0; istored < stored.size(); ++istored)
  {
    // - get increment number
    size_t inc = stored(istored);

    // - restore displacement
    xt::noalias(m_u) = H5Easy::load<xt::xtensor<double,2>>(m_file, "/disp/"+std::to_string(inc));

    // - update strain/strain
    computeStrainStress();

    // - index of the current quadratic potential and the plastic strain
    auto idx  = xt::view(m_material.Find(m_Eps), xt::keep(m_plastic), 0);
    auto epsp = xt::view(m_material.Epsp(m_Eps), xt::keep(m_plastic), 0);

    // - macroscopic strain/stress tensor
    xt::xtensor_fixed<double, xt::xshape<2,2>> Epsbar = xt::average(m_Eps, dV, {0,1});
    xt::xtensor_fixed<double, xt::xshape<2,2>> Sigbar = xt::average(m_Sig, dV, {0,1});

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
  for (size_t i = 1; i < K.size(); ++i)
    K(i) = (sigd(i) - sigd(0)) / (epsd(i) - epsd(0));
  // - set dummy (to take the minimum below)
  K(0) = K(1);
  // - get steady-state increment
  if (xt::any(K <= 0.95 * K(1))) {
    //  - select increment
    data.steadystate = xt::amin(xt::from_indices(xt::argwhere(K <= 0.95 * K(1))))[0];
    data.has_steadystate = true;
    // - always start the steady-state by elastic loading
    if (kick(data.steadystate))
      data.steadystate += 1;
  }
  // - no steady-state found: no steady-state output
  else {
    data.has_steadystate = false;
  }

  // per increment, list which increment had the last system-spanning avalanche
  // - allocate
  data.inc_system = xt::zeros<size_t>(stored.shape());
  data.eps_system = xt::zeros<double>(stored.shape());
  data.sig_system = xt::zeros<double>(stored.shape());
  // - list with increment of system-spanning avalanches
  xt::xtensor<size_t,1> isys = xt::flatten_indices(xt::argwhere(xt::equal(A, N)));
  // - only continue for non-empty list
  if (isys.size() > 0)
  {
    // - fill: all but last entry
    for (size_t i = 0; i < isys.size()-1; ++i)
      xt::view(data.inc_system, xt::range(isys(i), isys(i+1))) = isys(i);
    // - fill: last entry
    size_t i = isys(isys.size()-1);
    xt::view(data.inc_system, xt::range(i, stored.size())) = i;
    // - get information about the last system spanning avalanche
    for (size_t i = 0; i < stored.size(); ++i) {
      data.eps_system(i) = epsd(data.inc_system(i));
      data.sig_system(i) = sigd(data.inc_system(i));
    }
  }

  // return output
  data.dt    = m_dt;
  data.G     = m_material.G()(0,0);;
  data.N     = N;
  data.incs  = stored;
  data.kick  = kick;
  data.S     = S;
  data.A     = A;
  data.xi    = xi;
  data.depsp = depsp;
  data.epsd  = epsd;
  data.sigd  = sigd;
  data.uuid  = H5Easy::load<std::string>(m_file, "/uuid");

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
  /uuid               List with uuid's of each simulation.
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
  ../eps0             Strain normalisation.
  ../sig0             Stress normalisation.
  ../G                Shear modulus (homogeneous).
  ../rho              Mass density (homogeneous).
  ../cs               Shear wave speed (homogeneous).
  ../t0               Time it takes a shear wave to travel the size of an element.

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
  if (files.size() == 0)
    return 0;

  // output name
  // - command-line options
  std::string outpath = args["--outpath"].asString();
  std::string outname = args["--outname"].asString();
  // - default path
  if (outpath.size() == 0)
    outpath = cpppath::common_dirname(files);
  // - convert name to path
  outname = cpppath::join({outpath, outname});

  // strip files from output-path
  // - allocate
  std::vector<std::string> sims;
  // - fill
  for (auto& m_file: files)
    sims.push_back(cpppath::split(m_file, outpath+"/")[0]);

  // normalisation parameters
  double epsy = std::stod(args["--epsy"].asString());
  double l0   = std::stod(args["--l0"  ].asString());

  // list with uuids
  std::vector<std::string> uuid(sims.size());

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
  xt::xtensor<double,1> norm_eps0 = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_sig0 = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_l0   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_G    = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_rho  = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_cs   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_t0   = xt::zeros<double>({files.size()});
  xt::xtensor<double,1> norm_dt   = xt::zeros<double>({files.size()});

  // print progress to screen
  fmt::print("Writing to '{0:s}'\n", outname);

  // open output file
  H5Easy::File out(outname, H5Easy::File::Overwrite);

  // loop over simulations
  for (size_t i = 0; i < files.size(); ++i )
  {
    // print progress to screen
    fmt::print("- reading '{0:s}'\n", files[i]);

    // open simulation
    Main sim(files[i]);

    // read results
    Data data = sim.run();

    // get uuid, and strain and stress normalisation
    uuid[i] = data.uuid;

    // alias
    size_t steadystate = data.steadystate;

    // typical stress: ensemble average yield stress
    // (normalisation only)
    double sigy = 2. * data.G * epsy;

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
    norm_eps0(i) = epsy;
    norm_sig0(i) = sigy;
    norm_l0  (i) = l0;
    norm_G   (i) = data.G;
    norm_rho (i) = data.rho;
    norm_cs  (i) = cs;
    norm_t0  (i) = t0;
    norm_dt  (i) = data.dt;

    // storage path
    std::string path = cpppath::normpath(fmt::format("/full/{0:s}", sims[i]));

    // save raw data (in addition to the ensemble data), for fast recovery
    H5Easy::dump(out, path+"/steadystate" , steadystate      );
    H5Easy::dump(out, path+"/kick"        , data.kick        );
    H5Easy::dump(out, path+"/S"           , data.S           );
    H5Easy::dump(out, path+"/A"           , data.A           );
    H5Easy::dump(out, path+"/xi"          , data.xi          );
    H5Easy::dump(out, path+"/depsp"       , data.depsp       );
    H5Easy::dump(out, path+"/dt_avalanche", data.dt_avalanche);
    H5Easy::dump(out, path+"/epsd"        , data.epsd        );
    H5Easy::dump(out, path+"/sigd"        , data.sigd        );

    // total number of increments
    size_t ni = data.incs.size();

    // continue only if a steady-state was found
    if (!data.has_steadystate)
      continue;

    // below the increments are split (based on the event-driven protocol) in elastic loading
    // and (potential) avalanches
    // for simplicity this split is done such that the number of increments is the same for both
    // (hence potentially the last increment has to be discard)
    if (steadystate % 2 == 0 && ni % 2 != 0) ni -= 1;
    if (steadystate % 2 != 0 && ni % 2 == 0) ni -= 1;

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
  MYASSERT(xt::all(xt::equal(norm_eps0, norm_eps0(0))));
  MYASSERT(xt::all(xt::equal(norm_sig0, norm_sig0(0))));
  MYASSERT(xt::all(xt::equal(norm_l0  , norm_l0  (0))));
  MYASSERT(xt::all(xt::equal(norm_G   , norm_G   (0))));
  MYASSERT(xt::all(xt::equal(norm_rho , norm_rho (0))));
  MYASSERT(xt::all(xt::equal(norm_cs  , norm_cs  (0))));
  MYASSERT(xt::all(xt::equal(norm_t0  , norm_t0  (0))));
  MYASSERT(xt::all(xt::equal(norm_dt  , norm_dt  (0))));

  // save ensemble
  H5Easy::dump(out, "/files", sims);
  H5Easy::dump(out, "/uuid" , uuid);

  H5Easy::dump(out, "/normalisation/N"   , static_cast<size_t>(norm_N   (0)));
  H5Easy::dump(out, "/normalisation/eps0", static_cast<double>(norm_eps0(0)));
  H5Easy::dump(out, "/normalisation/sig0", static_cast<double>(norm_sig0(0)));
  H5Easy::dump(out, "/normalisation/G"   , static_cast<double>(norm_G   (0)));
  H5Easy::dump(out, "/normalisation/rho" , static_cast<double>(norm_rho (0)));
  H5Easy::dump(out, "/normalisation/cs"  , static_cast<double>(norm_cs  (0)));
  H5Easy::dump(out, "/normalisation/t0"  , static_cast<double>(norm_t0  (0)));
  H5Easy::dump(out, "/normalisation/l0"  , static_cast<double>(norm_l0  (0)));
  H5Easy::dump(out, "/normalisation/dt"  , static_cast<double>(norm_dt  (0)));

  H5Easy::dump(out, "/loading/file"          , load_sim         );
  H5Easy::dump(out, "/loading/inc"           , load_incs        );
  H5Easy::dump(out, "/loading/ninc"          , load_ninc        );
  H5Easy::dump(out, "/loading/kick"          , load_kick        );
  H5Easy::dump(out, "/loading/inc_system"    , load_inc_system  );
  H5Easy::dump(out, "/loading/eps_system"    , load_eps_system  );
  H5Easy::dump(out, "/loading/sig_system"    , load_sig_system  );
  H5Easy::dump(out, "/loading/S"             , load_S           );
  H5Easy::dump(out, "/loading/A"             , load_A           );
  H5Easy::dump(out, "/loading/xi"            , load_xi          );
  H5Easy::dump(out, "/loading/depsp"         , load_depsp       );
  H5Easy::dump(out, "/loading/dt_avalanche"  , load_dt_avalanche);
  H5Easy::dump(out, "/loading/epsd"          , load_epsd        );
  H5Easy::dump(out, "/loading/sigd"          , load_sigd        );

  H5Easy::dump(out, "/avalanche/file"        , aval_sim         );
  H5Easy::dump(out, "/avalanche/inc"         , aval_incs        );
  H5Easy::dump(out, "/avalanche/ninc"        , aval_ninc        );
  H5Easy::dump(out, "/avalanche/kick"        , aval_kick        );
  H5Easy::dump(out, "/avalanche/inc_system"  , aval_inc_system  );
  H5Easy::dump(out, "/avalanche/eps_system"  , aval_eps_system  );
  H5Easy::dump(out, "/avalanche/sig_system"  , aval_sig_system  );
  H5Easy::dump(out, "/avalanche/S"           , aval_S           );
  H5Easy::dump(out, "/avalanche/A"           , aval_A           );
  H5Easy::dump(out, "/avalanche/xi"          , aval_xi          );
  H5Easy::dump(out, "/avalanche/depsp"       , aval_depsp       );
  H5Easy::dump(out, "/avalanche/dt_avalanche", aval_dt_avalanche);
  H5Easy::dump(out, "/avalanche/epsd"        , aval_epsd        );
  H5Easy::dump(out, "/avalanche/sigd"        , aval_sigd        );

  // get ensemble averages
  // - size of the weak layer
  size_t N = H5Easy::load<size_t>(out, "/normalisation/N");
  // - area of the avalanche
  xt::xtensor<size_t,1> A = H5Easy::load<xt::xtensor<size_t,1>>(out, "/avalanche/A");
  // - area of the avalanche, and the stress just before and just after each avalanche
  xt::xtensor<double,1> sig_bot = H5Easy::load<xt::xtensor<double,1>>(out, "/avalanche/sigd");
  xt::xtensor<size_t,1> inc_bot = H5Easy::load<xt::xtensor<size_t,1>>(out, "/avalanche/inc" );
  xt::xtensor<double,1> sig_top = H5Easy::load<xt::xtensor<double,1>>(out, "/loading/sigd"  );
  xt::xtensor<size_t,1> inc_top = H5Easy::load<xt::xtensor<size_t,1>>(out, "/loading/inc"   );
  // - filter for system-spanning avalanches
  xt::xtensor<size_t,1> idx = xt::flatten_indices(xt::argwhere(xt::equal(A,N)));
  // - compute averages
  if (idx.size() > 0)
  {
    // - extract
    sig_top = xt::view(sig_top, xt::keep(idx));
    sig_bot = xt::view(sig_bot, xt::keep(idx));
    inc_bot = xt::view(inc_bot, xt::keep(idx));
    inc_top = xt::view(inc_top, xt::keep(idx));
    // - check increments numbers
    MYASSERT(xt::all(xt::equal(inc_bot, inc_top+1)));
    // - compute averages
    H5Easy::dump(out, "/averages/sigd_top"   , xt::mean(sig_top)[0]);
    H5Easy::dump(out, "/averages/sigd_bottom", xt::mean(sig_bot)[0]);
  }

  return 0;
}