#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <cpppath.h>
#include <docopt/docopt.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

namespace FQF = FrictionQPotFEM::UniformSingleLayer2d;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

template <class T>
void DumpWithDescription(
    H5Easy::File& file,
    const std::string& path,
    const T& data,
    const std::string& description)
{
    H5Easy::dump(file, path, data);
    H5Easy::dumpAttribute(file, path, "desc", description);
}

static const char USAGE[] =
    R"(EventEvolution_stress

Description:
    1.  Load a specific system-spanning increment.
    2.  Move forward (advancing possibly events, and loading elastically) to a fixed stress.
    2.  Apply a local push to a specific element.
    3.  Store evolution to output file.

Data:
    -   At every event:
        -   Time-step.
        -   Position of the event.
        -   Stress: macroscopic, on the weak layer, in the crack.
        -   Avalanche properties: size "S", radius "A".
    -   Periodic global snapshots:
        -   Time-step.
        -   Stress: macroscopic, on the weak layer, in the crack.
        -   Avalanche properties: size "S", radius "A".
    -   Periodic snapshots on weak layer:
        -   Stress.
        -   Index of the local potential.

Dependencies:
    -   Simulations ran using "Run" serve as input, and could be necessary to interpret
        some of the output.
    -   Parameters stored "EnsembleInfo" may be needed for normalisation.

Job:
    Use job*.py in the source directory to create the necessary ensemble (based on earlier data).

Usage:
    EventEvolution_stress [options] --output=N --file=N --element=N --incc=N --stress=N

Arguments:
        --output=N      Path of the output file (overwritten).
        --stress=N      Stress at which to measure.
        --file=N        The path to the simulation file (read-only).
        --element=N     Element to push.
        --incc=N        Increment number of the system-spanning avalanche.

Options:
    -h, --help          Show help.
        --version       Show version.

(c) Tom de Geus
)";

class Main : public FQF::System {

private:
    H5Easy::File m_file;
    GooseFEM::Iterate::StopList m_stop = GooseFEM::Iterate::StopList(20);
    size_t m_inc;
    double m_deps_kick;

public:
    Main(const std::string& fname) : m_file(fname, H5Easy::File::ReadOnly)
    {
        this->init(
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
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsbar =
                xt::average(this->Eps(), dV, {0, 1});
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Sigbar =
                xt::average(this->Sig(), dV, {0, 1});

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

            // - skip if the loading pattern was not load-kick-load-kick-... (sanity check)
            if (xt::any(xt::not_equal(k, 0ul))) {
                continue;
            }
            if (xt::any(xt::not_equal(a, 0ul))) {
                continue;
            }

            // - find where the strain(stress) is higher than the target strain(stress)
            //   during that increment the strain(stress) elastically moved from below to above the
            //   target strain(stress); the size of this step can be reduced by an arbitrary size,
            //   without violating equilibrium
            auto idx = xt::flatten_indices(xt::argwhere(s > stress));

            // - no increment found -> skip (sanity check)
            if (idx.size() == 0) {
                continue;
            }

            // - start from the increment before it (the beginning of the elastic loading)
            size_t ipush = n(xt::amin(idx)[0]) - 1;

            // - sanity check
            if (sigd(ipush) > stress) {
                continue;
            }
            if (kick(ipush + 1) != 0) {
                continue;
            }

            // - store
            inc_push(i) = ipush;
        }

        // filter list with increments
        // (zero can never be a valid increment, because of the restriction set above)
        inc_push = xt::filter(inc_push, inc_push > 0ul);

        return std::make_tuple(inc_system, inc_push);
    }

public:
    void run(double stress, size_t element, size_t inc_c, const std::string& output)
    {
        auto dV = m_quad.AsTensor<2>(m_quad.dV());
        auto dV_plas = m_quad_plas.AsTensor<2>(m_quad_plas.dV());
        size_t N = m_N;

        // extract a list with increments at which to start elastic loading
        xt::xtensor<size_t, 1> inc_system, inc_push;
        std::tie(inc_system, inc_push) = getIncPush(stress);
        MYASSERT(inc_system.size() > 0);
        MYASSERT(inc_push.size() > 0);
        MYASSERT(xt::any(xt::equal(inc_system, inc_c)));

        // get push increment
        size_t ipush = xt::flatten_indices(xt::argwhere(xt::greater_equal(inc_push, inc_c)))(0);
        m_inc = inc_push(ipush);
        MYASSERT(ipush < inc_push.size());
        MYASSERT(inc_push(ipush) >= inc_c);

        // clear/open the output file
        H5Easy::File data(output, H5Easy::File::Overwrite);

        // restore increment
        m_inc = inc_c;
        this->setU(H5Easy::load<decltype(m_u)>(m_file, fmt::format("/disp/{0:d}", m_inc)));
        this->addSimpleShearToFixedStress(stress);

        DumpWithDescription(
            data,
            fmt::format("/disp/{0:d}", 0),
            this->u(),
            "Displacement at mechanical equilibrium before pushing.");

        // extract "id" from filename (stored to data)
        std::string id = cpppath::split(cpppath::filename(m_file.getName()), ".")[0];
        size_t id_num = static_cast<size_t>(std::stoi(cpppath::split(id, "=")[1]));

        // extract information needed for storage
        xt::xtensor<int, 1> idx_last = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx_n = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
        xt::xtensor<int, 1> idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);

        // perturb the displacement of the set element, to (try to) trigger an avalanche
        this->triggerElementWithLocalSimpleShear(m_deps_kick, element);

        // storage parameters
        int S = 0; // avalanche size (maximum size since beginning)
        size_t S_next = 0; // next storage value
        size_t S_index = 0; // storage index
        size_t S_step = 50; // storage step size
        size_t A = 0; // current crack area  (maximum size since beginning)
        size_t A_next = 0; // next storage value
        size_t A_index = 0; // storage index
        size_t A_step = 1; // storage step size
        bool A_store = true; // store synchronised on "A" -> false when A == N
        size_t t_step = 500; // interval at which to store a global snapshot
        size_t t_factor = 4; // "t_step * t_factor" is interval at which to store local snapshot
        size_t ioverview = 0;
        size_t isnapshot = 0;
        size_t ievent = 0;
        bool last = false;
        bool event_attribute = true;
        bool event = false;
        xt::xtensor_fixed<double, xt::xshape<2, 2>> Sig_bar = xt::average(this->Sig(), dV, {0, 1});
        xt::xtensor<double, 3> Sig_elem =
            xt::average(this->plastic_Sig(), dV_plas, {1}); // only shape matters
        xt::xtensor<double, 2> Sig_plas = xt::empty<double>({3ul, N});
        xt::xtensor<double, 1> sig_weak = xt::empty<double>({3ul});
        xt::xtensor<double, 1> sig_crack = xt::empty<double>({3ul});
        xt::xtensor<double, 1> yielded = xt::empty<double>({N});
        xt::xtensor<double, 2> yielded_2 = xt::empty<double>({3ul, N});

        // quench: force equilibrium
        for (size_t iiter = 0;; ++iiter) {
            if (iiter > 0) {
                idx = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
            }

            size_t a = xt::sum(xt::not_equal(idx, idx_n))();
            int s = xt::sum(idx - idx_n)();
            A = std::max(A, a);
            S = std::max(S, s);

            bool save_event = xt::any(xt::not_equal(idx, idx_last));
            bool save_overview = iiter % t_step == 0 || last || iiter == 0;
            bool save_snapshot = ((A >= A_next || A == N) && A_store) || S >= (int)S_next ||
                                 iiter % (t_step * t_factor) == 0 || last || iiter == 0;

            if (save_event || save_overview || save_snapshot) {
                xt::noalias(yielded) = xt::not_equal(idx, idx_n);
                for (size_t k = 0; k < 3; ++k) {
                    xt::view(yielded_2, k, xt::all()) = yielded;
                }
                xt::noalias(Sig_bar) = xt::average(this->Sig(), dV, {0, 1});
                xt::noalias(Sig_elem) = xt::average(this->plastic_Sig(), dV_plas, {1});
                xt::view(Sig_plas, 0, xt::all()) = xt::view(Sig_elem, xt::all(), 0, 0);
                xt::view(Sig_plas, 1, xt::all()) = xt::view(Sig_elem, xt::all(), 0, 1);
                xt::view(Sig_plas, 2, xt::all()) = xt::view(Sig_elem, xt::all(), 1, 1);
                xt::noalias(sig_weak) = xt::mean(Sig_plas, {1});
                xt::noalias(sig_crack) = xt::average(Sig_plas, yielded_2, {1});
            }

            if (save_event) {
                xt::xtensor<size_t, 1> r =
                    xt::flatten_indices(xt::argwhere(xt::not_equal(idx, idx_last)));
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
            if (save_overview || save_snapshot) {
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

            // store local snapshot
            if (save_snapshot) {
                H5Easy::dump(data, fmt::format("/snapshot/plastic/{0:d}/sig", isnapshot), Sig_plas);
                H5Easy::dump(data, fmt::format("/snapshot/plastic/{0:d}/idx", isnapshot), idx);
                H5Easy::dump(data, "/snapshot/storage/overview", ioverview, {isnapshot});
                H5Easy::dump(data, "/snapshot/storage/snapshot", isnapshot, {isnapshot});
                if ((A >= A_next || A == N) && A_store) {
                    H5Easy::dump(data, "/snapshot/storage/A/values", A, {A_index});
                    H5Easy::dump(data, "/snapshot/storage/A/index", isnapshot, {A_index});
                    A_next += A_step;
                    A_index++;
                    if (A >= N) {
                        A_store = false;
                    }
                }
                if (S >= (int)S_next) {
                    H5Easy::dump(data, "/snapshot/storage/S/values", S, {S_index});
                    H5Easy::dump(data, "/snapshot/storage/S/index", isnapshot, {S_index});
                    S_next += S_step;
                    S_index++;
                }
                if (iiter % (t_step * t_factor) == 0) {
                    size_t j = iiter / (t_step * t_factor);
                    H5Easy::dump(data, "/snapshot/storage/iiter/values", iiter, {j});
                    H5Easy::dump(data, "/snapshot/storage/iiter/index", isnapshot, {j});
                }
                isnapshot++;
            }

            // write info
            if (event && event_attribute) {
                H5Easy::dumpAttribute(
                    data,
                    "/event/step",
                    "desc",
                    std::string("Number of times the block yielded since the last event"));

                H5Easy::dumpAttribute(
                    data, "/event/r", "desc", std::string("Position of the yielding block"));

                H5Easy::dumpAttribute(
                    data, "/event/global/iiter", "desc", std::string("Iteration number for event"));

                H5Easy::dumpAttribute(
                    data,
                    "/event/global/S",
                    "desc",
                    std::string("Avalanche size at time of event"));

                H5Easy::dumpAttribute(
                    data,
                    "/event/global/A",
                    "desc",
                    std::string("Avalanche 'radius' at time of event"));

                H5Easy::dumpAttribute(
                    data,
                    "/event/global/sig",
                    "desc",
                    std::string("Macroscopic stress (xx, xy, yy) at time of event"));

                H5Easy::dumpAttribute(data, "/event/global/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/global/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/global/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    "/event/weak/sig",
                    "desc",
                    std::string("Stress averaged on weak layer (xx, xy, yy) at time of event"));

                H5Easy::dumpAttribute(data, "/event/weak/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/weak/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    "/event/crack/sig",
                    "desc",
                    std::string("Stress averaged on yielded blocks (xx, xy, yy) at time of event"));

                H5Easy::dumpAttribute(data, "/event/crack/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/event/crack/sig", "yy", static_cast<size_t>(2));

                event_attribute = false;
            }

            if (iiter == 0) {
                H5Easy::dumpAttribute(
                    data,
                    "/overview/global/iiter",
                    "desc",
                    std::string("Iteration (time-step) number"));

                H5Easy::dumpAttribute(
                    data, "/overview/global/S", "desc", std::string("Avalanche size"));

                H5Easy::dumpAttribute(
                    data, "/overview/global/A", "desc", std::string("Avalanche 'radius'"));

                H5Easy::dumpAttribute(
                    data,
                    "/overview/global/sig",
                    "desc",
                    std::string("Macroscopic stress (xx, xy, yy)"));

                H5Easy::dumpAttribute(data, "/overview/global/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/global/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    "/overview/weak/sig",
                    "desc",
                    std::string("Stress averaged on weak layer (xx, xy, yy)"));

                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/weak/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    "/overview/crack/sig",
                    "desc",
                    std::string("Stress averaged on yielded blocks (xx, xy, yy)"));

                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xx", static_cast<size_t>(0));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "xy", static_cast<size_t>(1));
                H5Easy::dumpAttribute(data, "/overview/crack/sig", "yy", static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    fmt::format("/snapshot/plastic/{0:d}/sig", 0),
                    "desc",
                    std::string("Stress tensor along the weak layer (xx, xy, yy)"));

                H5Easy::dumpAttribute(
                    data,
                    fmt::format("/snapshot/plastic/{0:d}/sig", 0),
                    "xx",
                    static_cast<size_t>(0));
                H5Easy::dumpAttribute(
                    data,
                    fmt::format("/snapshot/plastic/{0:d}/sig", 0),
                    "xy",
                    static_cast<size_t>(1));
                H5Easy::dumpAttribute(
                    data,
                    fmt::format("/snapshot/plastic/{0:d}/sig", 0),
                    "yy",
                    static_cast<size_t>(2));

                H5Easy::dumpAttribute(
                    data,
                    fmt::format("/snapshot/plastic/{0:d}/idx", 0),
                    "desc",
                    std::string("Index of the current local minimum"));

                H5Easy::dump(data, "/snapshot/storage/A/step", A_step);
                H5Easy::dump(data, "/snapshot/storage/S/step", S_step);
                H5Easy::dump(data, "/snapshot/storage/iiter/step", t_step * t_factor);
            }

            if (last) {
                break;
            }

            // time increment
            this->timeStep();

            // - check for convergence
            if (m_stop.stop(this->residual(), 1e-5)) {
                last = true;
            }
        }

        DumpWithDescription(
            data,
            fmt::format("/disp/{0:d}", 1),
            this->u(),
            "Displacement at new mechanical equilibrium after pushing.");

        DumpWithDescription(
            data,
            "/meta/uuid",
            H5Easy::load<std::string>(m_file, "/uuid"),
            "uuid of the input simulation.");

        DumpWithDescription(data, "/meta/id", id_num, "id of the input simulation");

        DumpWithDescription(
            data,
            "/meta/inc_c",
            inc_c,
            "Increment of the system-spanning avalanche at which push was applied.");

        DumpWithDescription(
            data,
            "/meta/element",
            element,
            "Element number (along the weak layer) to which push was applied.");

        DumpWithDescription(
            data,
            "/meta/EventEvolution_stress/version",
            std::string(MYVERSION),
            "Code version at compile-time.");

        DumpWithDescription(
            data,
            "/meta/EventEvolution_stress/dependencies",
            FQF::version_dependencies(),
            "Version of key dependencies at compile-time.");

        DumpWithDescription(
            data, "/meta/EventEvolution_stress/completed", 1, "Signal that this program finished.");
    }
};

int main(int argc, const char** argv)
{
    std::map<std::string, docopt::value> args =
        docopt::docopt(USAGE, {argv + 1, argv + argc}, true, std::string(MYVERSION));

    std::string output = args["--output"].asString();
    std::string file = args["--file"].asString();
    double stress = std::stod(args["--stress"].asString());
    size_t inc_c = static_cast<size_t>(std::stoi(args["--incc"].asString()));
    size_t element = static_cast<size_t>(std::stoi(args["--element"].asString()));

    Main sim(file);

    sim.run(stress, element, inc_c, output);

    return 0;
}
