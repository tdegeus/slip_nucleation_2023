
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

    void write_completed()
    {
        H5Easy::dump(m_file, "/completed", 1);
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

    int run_push(const std::string& outfilename)
    {
        auto dV = m_quad.AsTensor<2>(m_quad.dV());
        auto dV_plas = m_quad_plas.AsTensor<2>(m_quad_plas.dV());

        this->restore_last_stored();
        this->computeStress();

        H5Easy::File data(outfilename, H5Easy::File::Overwrite);

        int S = 0;              // avalanche size (maximum size since beginning)
        size_t A = 0;           // current avalanche area (maximum size since beginning)
        size_t t_step = 500;    // interval at which to store a global snapshot
        size_t ioverview = 0;   // storage index
        size_t ievent = 0;      // storage index
        bool last = false;      // == true when equilibrium is reached -> store equilibrium configuration
        bool attribute = true;  // signal to store attribute
        bool event = false;     // == true every time a yielding event took place -> write "/event/*"
        xt::xtensor<int, 1> idx_last = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx_n = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 2> Idx_n = xt::view(m_material_plas.CurrentIndex(), xt::all(), xt::all());
        xt::xtensor<double, 2> Sig_bar = xt::average(m_Sig, dV, {0, 1}); // only shape matters
        xt::xtensor<double, 3> Sig_elem = xt::average(m_Sig_plas, dV_plas, {1}); // only shape matters
        xt::xtensor<double, 2> Sig_plas = xt::empty<double>({3ul, m_N});
        xt::xtensor<double, 1> sig_weak = xt::empty<double>({3ul});
        xt::xtensor<double, 1> sig_crack = xt::empty<double>({3ul});
        xt::xtensor<double, 1> yielded = xt::empty<double>({m_N});
        xt::xtensor<double, 2> yielded_broadcast = xt::empty<double>({3ul, m_N});
        MYASSERT(std::abs(GM::Sigd(Sig_bar)() - H5Easy::load<double>(m_file, "/sigd", {m_inc})) < 1e-8);

        size_t trigger_element = this->find_weakest_element();
        this->triggerElementWithLocalSimpleShear(m_deps_kick, trigger_element, false);

        this->quench();
        m_stop.reset();

        for (size_t iiter = 0;; ++iiter) {

            // break if maximum local strain could be exceeded
            if (!m_material_plas.checkYieldBoundRight(5)) {
                H5Easy::dump(data, "/meta/corrupt", 1);
                return INT_MIN;
            }

            if (iiter > 0) {
                idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), 0);
            }

            size_t a = xt::sum(xt::not_equal(idx, idx_n))();
            int s = xt::sum(idx - idx_n)();
            A = std::max(A, a);
            S = std::max(S, s);

            bool save_event = xt::any(xt::not_equal(idx, idx_last));
            bool save_overview = iiter % t_step == 0 || last || iiter == 0;

            if (save_event || save_overview) {
                this->computeStress();
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

            if (event && attribute) {
                H5Easy::dumpAttribute(data, "/event/global/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/event/global/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/event/global/sig", "yy", size_t(2));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "yy", size_t(2));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "yy", size_t(2));
                attribute = false;
            }

            if (iiter == 0) {
                H5Easy::dumpAttribute(data, "/overview/global/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "yy", size_t(2));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "yy", size_t(2));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xx", size_t(0));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xy", size_t(1));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "yy", size_t(2));
            }

            if (last) {
                break;
            }

            timeStep();

            if (m_stop.stop(this->residual(), 1e-5)) {
                last = true;
            }
        }

        m_inc++;

        this->computeStress();
        xt::noalias(Sig_bar) = xt::average(m_Sig, dV, {0, 1});

        std::string hash = GIT_COMMIT_HASH;
        dump_check(m_file, "/git/run", hash);
        dump_check(m_file, "/version/run", FQF::versionInfo());

        H5Easy::dump(m_file, "/stored", m_inc, {m_inc});
        H5Easy::dump(m_file, "/t", m_t, {m_inc});
        H5Easy::dump(m_file, "/sigd", GM::Sigd(Sig_bar)(), {m_inc});
        H5Easy::dump(m_file, fmt::format("/disp/{0:d}", m_inc), m_u);

        dump_check(data, "/git/run", hash);
        dump_check(data, "/version/run", FQF::versionInfo());

        H5Easy::dump(data, "/meta/completed", 1);
        H5Easy::dump(data, "/meta/uuid", H5Easy::load<std::string>(m_file, "/uuid"));
        H5Easy::dump(data, "/meta/push/inc", H5Easy::load<size_t>(m_file, "/push/inc"));
        H5Easy::dump(data, "/meta/push/element", trigger_element);
        H5Easy::dump(data, "/meta/inc", m_inc);
        H5Easy::dump(data, "/meta/dt", m_dt);
        H5Easy::dump(data, "/meta/N", m_N);

        xt::xtensor<int, 2> Idx = xt::view(m_material_plas.CurrentIndex(), xt::all(), xt::all());
        return xt::sum(Idx - Idx_n)();
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

    for (size_t i = 0; i < 200; ++i) {
        // trigger and run
        sim.restore_last_stored();
        std::string outname =  fmt::format("{0:s}_push={1:d}.hdf5", output, sim.inc() + 1);
        fmt::print("Writing to {0:s}\n", outname);
        int S = sim.run_push(outname);
        // remove event output if the potential energy landscape went out-of-bounds somewhere
        if (S == INT_MIN) {
            std::remove(outname.c_str());
            break;
        }
        // stop if the push does not trigger anything anymore
        if (S <= 0) {
            break;
        }
    }

    sim.write_completed();

    return 0;
}
