
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <GooseFEM/GooseFEM.h>
#include <highfive/H5Easy.hpp>
#include <fmt/core.h>
#include <cpppath.h>
#include <docopt/docopt.h>
#include <cstdio>

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

template <class T>
inline void dump_check(H5Easy::File& file, const std::string& key, const T& data)
{
    if (!file.exist(key)) {
        H5Easy::dump(file, key, data);
    }
    else {
        MYASSERT(H5Easy::load<T>(file, key) == data);
    }
}

class Main : public FQF::HybridSystem {

public:

    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadWrite)
    {
        this->initGeometry(
            H5Easy::load<xt::xtensor<double, 2>>(m_file, "/coor"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/conn"),
            H5Easy::load<xt::xtensor<size_t, 2>>(m_file, "/dofs"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/dofsP"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/elastic/elem"),
            H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/cusp/elem"));

        this->initHybridSystem();

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

    size_t inc()
    {
        return m_inc;
    }

public:

    void restore_last_stored()
    {
        m_inc = H5Easy::load<size_t>(m_file, "/stored", {H5Easy::getSize(m_file, "/stored") - size_t(1)});
        m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {m_inc});
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));
    }

public:

    inline size_t find_weakest_element() {
        auto eps = GM::Epsd(this->plastic_Eps());
        auto epsy = this->plastic_CurrentYieldRight();
        auto deps = epsy - eps;
        auto index = xt::unravel_index(xt::argmin(deps)(), deps.shape());
        return index[0];
    }

public:

    void run_push()
    {
        auto dV = m_quad.AsTensor<2>(m_quad.dV());
        auto dV_plas = m_quad_plas.AsTensor<2>(m_quad_plas.dV());

        this->restore_last_stored();
        this->computeStress();

        H5Easy::dump(m_file, "/debug/stored", 1, {0});
        H5Easy::dump(m_file, "/debug/iter", 0, {0});
        H5Easy::dump(m_file, fmt::format("/debug/disp/{0:d}", 0), m_u, H5Easy::DumpMode::Overwrite);

        size_t trigger_element = this->find_weakest_element();
        double dgamma = this->triggerElementWithLocalSimpleShear(m_deps_kick, trigger_element, false);

        this->quench();
        m_stop.reset();

        H5Easy::dump(m_file, "/debug/trigger_element", trigger_element, H5Easy::DumpMode::Overwrite);

        for (auto& key : {"/debug/corrupt", "/debug/converged", "/debug/completed", "/debug/stored", "/debug/iter"}) {
            if (m_file.exist(key)) {
                auto s = H5Easy::load<xt::xtensor<int, 1>>(m_file, key);
                s.fill(0);
                H5Easy::dump(m_file, key, s, H5Easy::DumpMode::Overwrite);
            }
        }

        size_t step = 50;

        for (size_t iiter = 0;; ++iiter) {

            if (iiter % step == 0) {
                H5Easy::dump(m_file, "/debug/stored", 1, {iiter / step + 1});
                H5Easy::dump(m_file, "/debug/iter", iiter, {iiter / step + 1});
                H5Easy::dump(m_file, fmt::format("/debug/disp/{0:d}", iiter / step + 1), m_u, H5Easy::DumpMode::Overwrite);
            }

            timeStep();

            if (iiter > 30000) {
                H5Easy::dump(m_file, "/debug/completed", 1, {0});
                return;
            }

            if (m_stop.stop(this->residual(), 1e-5)) {
                H5Easy::dump(m_file, "/debug/converged", 1, {0}, H5Easy::DumpMode::Overwrite);
                return;
            }

            if (!m_material_plas.checkYieldBoundRight(5)) {
                H5Easy::dump(m_file, "/debug/corrupt", 1, {0}, H5Easy::DumpMode::Overwrite);
                return;
            }
        }
    }

private:

    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_stop = GF::Iterate::StopList(20);
    size_t m_inc;
    double m_deps_kick;

};

static const char USAGE[] =
    R"(PushWeakest
    Push the element containing the integration point closest to yielding
    (upon a positive strain increase) and compute the force equilibrium.
    Repeat this until nothing is triggered anymore.
    Store new state to input-file and write evolution to a separate output-file per increment.

Usage:
    PushWeakest [options] <data.h5>

Options:
    -h, --help      Show help.
        --version   Show version.

(c) Tom de Geus
)";

int main(int argc, const char** argv)
{
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, "v0.0.1");

    std::string input = args["<data.h5>"].asString();

    Main sim(input);

    sim.restore_last_stored();
    sim.run_push();

    return 0;
}
