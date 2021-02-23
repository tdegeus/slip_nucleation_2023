
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>
#include <cpppath.h>
#include <xtensor/xindex_view.hpp>


#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }


namespace FQF = FrictionQPotFEM::UniformSingleLayer2d;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

class Main : public FQF::HybridSystem {

private:

    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_stop = GooseFEM::Iterate::StopList(20);
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

    std::tuple<xt::xtensor<size_t, 1>, xt::xtensor<size_t, 1>> getIncPush(double stress)
    {
        auto dV = m_quad.AsTensor<2>(m_quad.dV());
        size_t N = m_N;

        // basic information for each increment
        auto stored = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/stored");
        auto kick = H5Easy::load<xt::xtensor<size_t, 1>>(m_file, "/kick");
        MYASSERT(stored.size() > 0);

        // allocate result
        xt::xtensor<size_t, 1> A = xt::zeros<size_t>({xt::amax(stored)() + 1});
        xt::xtensor<double, 1> epsd = xt::zeros<double>({xt::amax(stored)() + 1});
        xt::xtensor<double, 1> sigd = xt::zeros<double>({xt::amax(stored)() + 1});

        // restore displacement
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", stored(0))));

        // index of the current quadratic potential,
        // for the first integration point per plastic element
        auto idx_n = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);

        // loop over increments
        for (size_t istored = 0; istored < stored.size(); ++istored) {
            // - get increment number
            size_t inc = stored(istored);

            // - restore displacement
            this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", inc)));

            // - index of the current quadratic potential
            auto idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);

            // - macroscopic strain/stress tensor
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsbar = xt::average(this->Eps(), dV, {0, 1});
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Sigbar = xt::average(this->Sig(), dV, {0, 1});

            // - macroscopic equivalent strain/stress
            epsd(inc) = GM::Epsd(Epsbar)();
            sigd(inc) = GM::Sigd(Sigbar)();

            // - avalanche area
            A(inc) = xt::sum(xt::not_equal(idx, idx_n))();

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

            // - check that the loading pattern was load-kick-load-kick-... (sanity check)
            MYASSERT(xt::all(xt::equal(k, 0)));
            MYASSERT(xt::all(xt::equal(a, 0)));

            // - find where the strain(stress) is higher than the target strain(stress)
            //   during that increment the strain(stress) elastically moved from below to above the
            //   target strain(stress); the size of this step can be reduced by an arbitrary size,
            //   without violating equilibrium
            if (xt::all(s < stress)) {
                continue;
            }
            size_t j = xt::argmax(s >= stress)();

            // - start from the increment before it (the beginning of the elastic loading)
            size_t ipush = n(j) - 1;

            // - sanity check
            MYASSERT(sigd(ipush) <= stress);
            MYASSERT(kick(ipush + 1) == 0);

            // - store
            inc_push(i) = ipush;
        }

        // filter list with increments
        // (zero can never be a valid increment, because of the restriction set above)
        inc_push = xt::filter(inc_push, inc_push > 0ul);

        return std::make_tuple(inc_system, inc_push);
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
        // extract a list with increments at which to start elastic loading
        xt::xtensor<size_t, 1> inc_system, inc_push;
        std::tie(inc_system, inc_push) = getIncPush(stress);
        MYASSERT(inc_system.size() > 0);
        MYASSERT(inc_push.size() > 0);
        MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

        // get push increment
        size_t ipush = xt::argmax(xt::greater_equal(inc_push, inc_c))();
        MYASSERT(ipush < inc_push.size());
        m_inc = inc_push(ipush);
        fmt::print("{0:s}: Saving, restoring inc = {1:d}\n", m_file.getName(), m_inc);

        // restore
        m_t = H5Easy::load<decltype(m_t)>(m_file, "/t", {m_inc});
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));

        // increase displacement to set "stress"
        this->addSimpleShearToFixedStress(stress);

        // extract "id" from filename (stored to data)
        std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
        size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

        // extract information needed for storage
        size_t N = m_N;
        xt::xtensor<int, 1> idx_n = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);

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
                    idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
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

            if ((A_store == false) && (!t_store)) {
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

        H5Easy::dump(data, "/meta/versions/CrackEvolution_raw_stress", std::string(MYVERSION));
        H5Easy::dump(data, "/meta/versions/FrictionQPotFEM", FQF::version_dependencies());
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
        --incc=N        Increment number of the last system-spanning avalanche.
        --element=N     Element to push.
        --stress=N      Stress at which to push.

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
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, std::string(MYVERSION));

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
