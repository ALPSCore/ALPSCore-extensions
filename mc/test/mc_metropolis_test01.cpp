#include <iostream>

#include "alps/mc/mc_metropolis.hpp"
#include <alps/mc/stop_callback.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/mc/mpiadapter.hpp>
#endif

using namespace alps;

struct dummy_move1 {
    double attempt() {
        return 0.95;
    }
    double accept() {
        return 1.0;
    }
    void reject() { }
};

struct dummy_move2 {
    double attempt() {
        return 0.5;
    }
    double accept() {
        return 1.0;
    }
    void reject() { }
};

struct dummy_measure1 {
    typedef alps::accumulators::accumulator_set mcol_t;
    mcol_t &m_;
    int i = 0;
    dummy_measure1(mcol_t &m) : m_(m) { m_ << alps::accumulators::FullBinningAccumulator<double>("dummy1"); };
    void measure(double sign) {
        assert(sign == 1.0);
        i++;
        m_["dummy1"] << i;
    };

    static constexpr int dummy_check_me = 1231245;
};


int main(int argc, char *argv[]) {
#ifdef ALPS_HAVE_MPI
    MPI_Init(&argc, &argv);
    alps::mpi::communicator world;
#endif

    alps::accumulators::accumulator_set res;
    res << alps::accumulators::FullBinningAccumulator<double>("dummy1");
    res["dummy1"] << 1.0;
    res["dummy1"] << 2.0;

    std::cout << res["dummy1"] << std::endl;

    /*dummy_measure1 m1;
    m1.measure();
    m1.measure();
    m1.measure();

    std::cout << m1.m_["dummy1"] << std::endl;
*/
    alps::params p;
    mc_metropolis::define_parameters(p),
        p["SEED"] = size_t(42);
    p["ncycles"] = 30;
    p["cycle_len"] = 5;

    try {
        mc_metropolis mc(p, 0);
        mc.add_move(dummy_move2(), "dummy2", 2.0);
        mc.add_move(dummy_move1(), "dummy1", 1.0);

        mc.add_measure(dummy_measure1(mc.observables()), "m1");

        // test measurement extraction
        dummy_measure1 const &m2 = mc.extract_measurement<dummy_measure1>("m1");
        std::cout << "1231245 == " << m2.dummy_check_me << std::endl;
        if (m2.dummy_check_me != 1231245) throw std::logic_error("could not extract dummy_measure1 measurement.");

#ifdef ALPS_HAVE_MPI
        alps::mcmpiadapter<mc_metropolis> mc_mpi(p, world, alps::check_schedule(0.001, 2));
        mc_mpi.add_move(dummy_move1(), "dummy1", 1.0);
        mc_mpi.add_move(dummy_move2(), "dummy2", 2.0);
        mc_mpi.add_measure(dummy_measure1(mc_mpi.observables()), "m1");
#endif

        std::cout << std::endl << "no mpi" << std::endl << std::endl;
        mc.run(alps::stop_callback(1));
        auto results1 = mc.collect_results();
        std::cout << "Results (nompi) : " << results1["dummy1"] << std::endl;

#ifdef ALPS_HAVE_MPI
        std::cout << std::endl << "mpi" << std::endl << std::endl;
        mc_mpi.run(alps::stop_callback(1));
        auto results2 = mc_mpi.collect_results();
        if (!world.rank()) std::cout << "Results (mpi) : " << results2["dummy1"] << std::endl;
        MPI_Finalize();
#endif
    }
    catch (std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Fatal Error: Unknown Exception!\n";
        return -2;
    }
}
