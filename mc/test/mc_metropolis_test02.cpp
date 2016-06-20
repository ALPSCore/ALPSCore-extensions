#include <iostream>

#include "alps/mc/mc_metropolis.hpp"
#include <alps/mc/stop_callback.hpp>

#ifdef ALPS_HAVE_MPI
#include <alps/mc/mpiadapter.hpp>
#endif

#include <gtest/gtest.h>

using namespace alps;

// a test to take a simple integral - \int dx dy f(x,y) h(x,y) from 0 to pi
// f(x,y) = sin(x) sin(y)
// h(x,y) = cos(x+y)^2
// integral should be 2.2222222222222222

std::mt19937 rng;

// spl
// probability f
double f(double x, double y) { return sin(x) * sin(y); }
double f_int = 4.0; // that is \int f(x,y) dx dy from 0 to pi
double h(double x, double y) {
    double a = cos(x + y);
    return a * a;
}

struct config_t {
    double x_ = 1e-7, y_ = 1e-7;
};

config_t config;

struct move1 {
    std::uniform_real_distribution<double> distr = std::uniform_real_distribution<double>(1e-8, M_PI);
    double attempt() {
        x = distr(rng);
        y = distr(rng);
        double w = f(x, y) / f(config.x_, config.y_);
        return w;
    }
    double accept() {
        config.x_ = x;
        config.y_ = y;
        return 1.0;
    }
    void reject() { };
    double x, y;
};


struct measure1 {
    typedef alps::accumulators::accumulator_set data_t;
    measure1(data_t &m) : m_(m) { m_ << alps::accumulators::FullBinningAccumulator<double>("fxy"); };
    void measure(double sign) {
        assert(sign == 1.0);
        double fv = f_int * h(config.x_, config.y_);
        m_["fxy"] << fv;
    };
    data_t &m_;
};


TEST(mc_metropolis, test_integral_nompi) {
    alps::params p;
    mc_metropolis::define_parameters(p);

    size_t rnd_seed;
    rnd_seed = std::random_device()();
    std::cout << "random seed for test = " << rnd_seed << std::endl;


    p["SEED"] = rnd_seed;
    p["ncycles"] = 10000;
    p["cycle_len"] = 1;

    mc_metropolis mc(p);
    mc.add_move(move1(), "dummy1", 1.0);
    mc.add_measure(measure1(mc.observables()), "m1");

    std::cout << std::endl << "no mpi" << std::endl << std::endl;
    mc.run(alps::stop_callback(1));

    auto results1 = mc.collect_results();
    std::cout << "Results (nompi) : " << results1["fxy"] << std::endl;
    ASSERT_NEAR(results1["fxy"].mean<double>(), double(20) / double(9), 3 * results1["fxy"].error<double>());
}

#ifdef ALPS_HAVE_MPI

TEST(mc_metropolis, test_integral_mpi) {
    alps::mpi::communicator world;

    world.barrier();
    alps::params p;
    mc_metropolis::define_parameters(p);

    size_t rnd_seed;
    rnd_seed = std::random_device()();
    std::cout << "random seed for test = " << rnd_seed << std::endl;

    p["SEED"] = rnd_seed;
    p["ncycles"] = 10000;
    p["cycle_len"] = 1;

    alps::mcmpiadapter<mc_metropolis> mc(p, world, alps::check_schedule(0.001, 2));
    mc.add_move(move1(), "dummy1", 1.0);
    mc.add_measure(measure1(mc.observables()), "m1");

    std::cout << std::endl << "mpi" << std::endl << std::endl;
    mc.run(alps::stop_callback(1));
    world.barrier();

    //using alps::collect_results;
    if (world.rank() == 0) {
        auto results1 = mc.collect_results();
        std::cout << "Results (mpi) : " << results1["fxy"] << std::endl;
        ASSERT_NEAR(results1["fxy"].mean<double>(), double(20) / double(9), 3 * results1["fxy"].error<double>());
    }
    else mc.collect_results();

}
#endif


int main(int argc, char *argv[]) {
#ifdef ALPS_HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    int out = RUN_ALL_TESTS();
#ifdef ALPS_HAVE_MPI
    MPI_Finalize();
#endif
}

