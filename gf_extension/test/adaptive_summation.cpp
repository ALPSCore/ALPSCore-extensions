#include <gtest/gtest.h>
#include <alps/gf_extension/adaptive_summation.hpp>

TEST(AdaptiveSummation, Scalar) {
  auto f = [](long n) {
    double rtmp = n;
    return 1.0/(rtmp*rtmp);
    //return std::sin(1e-2*rtmp)/(rtmp*rtmp);
  };
  alps::gf_extension::AdaptiveSummation<double, decltype(f)> as(f, 1, long(1E+20));
  auto v =  as.evaluate(1e-10);
  std::cout << v << std::endl;
  std::cout << std::abs(v-M_PI*M_PI/6) << std::endl;
}

