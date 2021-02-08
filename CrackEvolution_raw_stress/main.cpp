
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <GooseFEM/GooseFEM.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <cpppath.h>


#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?"
#endif


#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }


namespace FQF = FrictionQPotFEM::UniformSingleLayer2d;


class Main : public FQF::HybridSystem {

private:

    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_stop = GF::Iterate::StopList(20);
    size_t m_inc;
    double m_deps_kick;

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadOnly)
    {
        this->initHybridSystem(
            H5Easy::load<xt::xtensor<double, 2>>(m_file, "/coor"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/conn"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/dofs"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/dofsP"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/elastic/elem"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/cusp/elem"));

        this->setMassMatrix(H5Easy::load<xt::xtensor<double, 1>>(m_file, "/rho"));
        this->setDampingMatrix(H5Easy::load<xt::xtensor<double, 1>>(m_file, "/damping/alpha"));

        this->setElastic(
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/K"),
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/elastic/G"));

        this->setPlastic(
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/K"),
            H5Easy::load<xt::xtensor<double, 1>>(m_file, "/cusp/G"),
            H5Easy::load<xt::xtensor<double, 2>>(m_file, "/cusp/epsy"));

        this->setDt(H5Easy::load<double>(m_file, "/run/dt"));

        m_deps_kick = H5Easy::load<double>(m_file, "/run/epsd/kick");
    }

public:

    void run(
        double stress,
        size_t element,
        size_t inc_c,
        const std::string& output,
        size_t A_step,
        size_t t_step,
        size_t t_max_fac)
    {
        m_inc = inc_c;
        m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {m_inc});
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));
        this->computeStress();
        this->addSimpleShearToFixedStress(stress);

        // extract "id" from filename (stored to data)
        std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
        size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

        // extract information needed for storage
        size_t N = m_N;
        xt::xtensor<int, 1> idx_n = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);

        // perturb the displacement of the set element, to (try to) trigger an avalanche
        this->triggerElementWithLocalSimpleShear(m_deps_kick, element);

        // clear/open the output file
        H5Easy::File data(output, H5Easy::File::Overwrite);

        // storage parameters
        size_t A = 0;
        size_t A_next = 0;
        size_t t_next = 0;
        size_t A_istore = 0;
        size_t t_istore = 0;
        size_t t_first = 0;
        bool A_store = true;
        bool t_store = false;

        // quench: force equilibrium
        for (size_t iiter = 0;; ++iiter) {

            if (A_store) {

                if (iiter > 0) {
                    idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
                }

                size_t a = xt::sum(xt::not_equal(idx, idx_n))[0];
                A = std::max(A, a);

                if ((A >= A_next && A % A_step == 0) || A == N || iiter == 0) {

                    fmt::print("{0:s}: Saving, sync-A, A = {1:d}\n", m_file.getName(), A);

                    H5Easy::dump(data, "/sync-A/stored", A, {A_istore});
                    H5Easy::dump(data, "/sync-A/global/iiter", iiter, {A});
                    H5Easy::dump(data, fmt::format("/sync-A/{0:d}/u", A), m_u);
                    H5Easy::dump(data, fmt::format("/sync-A/{0:d}/v", A), m_v);
                    H5Easy::dump(data, fmt::format("/sync-A/{0:d}/a", A), m_a);

                    ++A_istore;
                    A_next = A + A_step;
                }

                if (A == N) {
                    A_store = false;
                }
            }

            if ((A_store = false) && (!t_store)) {
                t_store = true;
                t_next = iiter;
                t_first = iiter;
            }

            if (t_store && iiter == t_next) {

                fmt::print("{0:s}: Saving, sync-t, iiter = {1:d}\n", m_file.getName(), iiter);

                H5Easy::dump(data, "/sync-t/stored", t_istore, {t_istore});
                H5Easy::dump(data, "/sync-t/global/iiter", iiter, {t_istore});
                H5Easy::dump(data, fmt::format("/sync-t/{0:d}/u", t_istore), m_u);
                H5Easy::dump(data, fmt::format("/sync-t/{0:d}/v", t_istore), m_v);
                H5Easy::dump(data, fmt::format("/sync-t/{0:d}/a", t_istore), m_a);

                ++t_istore;
                t_next += t_step;
            }

            this->timeStep();

            if (m_stop.stop(this->residual(), 1e-5)) {
                break;
            }

            if (iiter > t_first * t_max_fac && t_store) {
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
        H5Easy::dump(data, "/meta/plastic", this->plastic());
    }

};


static const char USAGE[] =
    R"(CrackEvolution_raw_stress
  Extract time evolution of a specific push.

Usage:
  CrackEvolution_raw_stress [options] --incc=N --element=N --stress=N --file=N --output=N

Arguments:
      --file=N        The path to the simulation file.
      --output=N      Path of the output file.
      --incc=N        Increment number of the system-spanning avalanche.
      --element=N     Element to push.
      --stress=N      Relative stress distance to "sigma_down" at which to measure.

Options:
      --Astep=N       Save states at crack sizes A = (0: N: Astep). [default: 1]
      --tstep=N       Save states at times t = (t0: : tstep). [default: 500]
      --tfac=N        Stop simulation after "tfac * iiter" iterations (iiter when A = N). [default: 100]
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
    size_t A_step = static_cast<size_t>(std::stoi(args["--Astep"].asString()));
    size_t t_step = static_cast<size_t>(std::stoi(args["--tstep"].asString()));
    size_t t_max_fac = static_cast<size_t>(std::stoi(args["--tfac"].asString()));

    Main sim(file);

    sim.run(stress, element, inc_c, output, A_step, t_step, t_max_fac);

    return 0;
}
