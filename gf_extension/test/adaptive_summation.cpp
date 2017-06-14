#include <complex>
#include <limits>

#include <gtest/gtest.h>
#include <alps/gf_extension/adaptive_summation.hpp>
#include <alps/gf_extension/ir_basis.hpp>

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

TEST(AdaptiveSummation, Tnl_IR) {
  const double Lambda = 10.0;
  const int max_dim = 100;
  alps::gf_extension::fermionic_ir_basis basis(Lambda, max_dim);

  alps::gf_extension::interpolate_Tbar_ol Tbar(basis);

  int dim = basis.dim();

  int num_eval = 0;
  auto f = [&Tbar, &dim, &num_eval](long n) {
    ++num_eval;
    return Tbar(n, dim-1)*Tbar(n, dim-2);
  };

  alps::gf_extension::AdaptiveSummation<std::complex<double>, decltype(f)> as(f, 1, std::numeric_limits<long>::max()/100);
  std::cout << as.evaluate(1e-4) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-6) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-8) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-10) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;

}
