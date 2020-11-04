
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
        m_dV = m_quad.AsTensor<2>(m_quad.dV());
    }

public:

    size_t inc()
    {
        return m_inc;
    }

public:

    void set_inc(size_t n)
    {
        m_inc = n;
    }

public:

    void restore_inc()
    {
        m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {m_inc});
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));
    }

public:

    // return +1 : have to pass events by shearing right
    // return -1 : have to pass events by shearing left
    // return 0 : stress can be reached elastically
    int how_to_reach_stress(double target_stress)
    {
        this->restore_inc();
        double df_stress = this->addSimpleShearToFixedStress(target_stress, true);
        double df_elas_neg = this->addSimpleShearEventDriven(1e-2 * m_deps_kick, false, -1.0, true);
        double df_elas_pos = this->addSimpleShearEventDriven(1e-2 * m_deps_kick, false, +1.0, true);
        xt::xtensor<double, 2> Sig_bar = xt::average(this->Sig(), m_dV, {0, 1});

        if (Sig_bar(0, 1) < 0) {
            return +1;
        }

        if (df_stress < 0) {
            if (df_stress < df_elas_neg) {
                return -1;
            }
            else {
                return 0;
            }
        }
        if (df_stress > 0) {
            if (df_stress > df_elas_pos) {
                return +1;
            }
            else {
                return 0;
            }
        }
        return 0;
    }

public:

    void run_push(const std::string& outfilename, double target_stress, size_t push)
    {
        auto dV_plas = m_quad_plas.AsTensor<2>(m_quad_plas.dV());

        this->restore_inc();

        MYASSERT(std::abs(
            GM::Sigd<xt::xtensor<double, 2>>(xt::average(this->Sig(), m_dV, {0, 1}))() -
            H5Easy::load<double>(m_file, "/sigd", {m_inc})) < 1e-8);

        bool target_stress_exact = this->how_to_reach_stress(target_stress) == 0;

        if (target_stress_exact) {
            this->addSimpleShearToFixedStress(target_stress);
        }
        this->computeStress();
        auto u0 = this->u();

        H5Easy::File data(outfilename, H5Easy::File::ReadOnly);
        MYASSERT(m_inc == H5Easy::load<size_t>(data, "/meta/inc"));
        MYASSERT(target_stress_exact == static_cast<bool>(H5Easy::load<int>(data, "/meta/push/target_stress_exact")));
        MYASSERT(H5Easy::load<std::string>(data, "/meta/uuid") == H5Easy::load<std::string>(m_file, "/uuid"));
        MYASSERT(H5Easy::load<size_t>(data, "/meta/push/inc") == H5Easy::load<size_t>(m_file, "/push/inc"));

        H5Easy::dump(m_file, "/push/init/inc", m_inc, {push});
        H5Easy::dump(m_file, fmt::format("/push/init/disp/{0:d}", push), u0);
        if (!m_file.exist("/git/add_init_stress")) {
            std::string hash = GIT_COMMIT_HASH;
            H5Easy::dump(m_file, "/git/add_init_stress", hash);
            std::string comment = "Adding '/push/init/inc' and '/push/init/disp/*', characterising the stress at the beginning of the push.";
            H5Easy::dumpAttribute(m_file, "/git/add_init_stress", "desc", comment);
        }
    }

public:

    xt::xtensor<double, 1> get_target_stresses()
    {
        return H5Easy::load<xt::xtensor<double, 1>>(m_file, "/push/stresses");
    }

private:

    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_stop = GF::Iterate::StopList(20);
    size_t m_inc;
    double m_deps_kick;
    xt::xtensor<double, 4> m_dV;

};

static const char USAGE[] =
    R"(UnloadPushElement
    Unload to a specific force.
    Push a specific element and compute the force equilibrium.
    Store new state to input-file and write evolution to a separate output-file per increment.

Usage:
    UnloadPushElement [options] --output=N --input=N

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
    auto stresses = sim.get_target_stresses();
    sim.set_inc(0);

    for (size_t i = 0; i < stresses.size(); ++i) {
        // unload if needed
        for (size_t u = 0; u < 10; u++) {
            int dir = sim.how_to_reach_stress(stresses(i));
            if (dir == 0) {
                break;
            }
            else if (dir < 0) {
                fmt::print("Event driven unload {0:s}\n", input);
                sim.set_inc(sim.inc() + 2);
            }
            else if (dir > 0) {
                fmt::print("Positive loading would be needed, triggering at current state {0:s}\n", input);
                break;
            }
            if (u == 8) {
                throw std::runtime_error("Target stress not found within max. number of unloading steps");
            }
        }
        // trigger and run
        std::string outname = fmt::format("{0:s}_push={1:d}.hdf5", output, i);
        fmt::print("Verifying to {0:s}\n", outname);
        sim.run_push(outname, stresses(i), i);
    }

    return 0;
}
