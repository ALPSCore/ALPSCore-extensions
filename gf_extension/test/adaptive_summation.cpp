#include <complex>
#include <limits>

#include <gtest/gtest.h>
#include <alps/gf_extension/adaptive_summation.hpp>
#include <alps/gf_extension/ir_basis.hpp>

TEST(AdaptiveSummation, Scalar) {
  auto f = [](const std::array<std::int64_t,1>& n) {
    double rtmp = n[0];
    return 1.0/(rtmp*rtmp);
  };
  std::int64_t max_n = std::numeric_limits<std::int64_t>::max()/10;
  alps::gf_extension::AdaptiveSummation<1,double, decltype(f)> as(f, 1, max_n);
  auto v =  as.evaluate(1e-5);
  std::cout << " v = " << v << " error " << std::abs(v-M_PI*M_PI/6) << std::endl;
  v =  as.evaluate(1e-8);
  std::cout << " v = " << v << " error " << std::abs(v-M_PI*M_PI/6) << std::endl;
}

TEST(AdaptiveSummation, Orthogonarity) {
  auto f = [&](const std::array<std::int64_t,1>& n) {
    auto iwn = std::complex<double>(0.0, M_PI * (2*n[0]+1));
    return std::conj(1.0/iwn) * (1.0/(iwn*iwn));
  };
  std::int64_t max_n = std::numeric_limits<std::int64_t>::max()/10;
  alps::gf_extension::AdaptiveSummation<1,std::complex<double>, decltype(f)> as(f, -max_n, max_n);
  auto v =  as.evaluate(1e-5);
  std::cout << " v = " << v << " error = " << as.abs_error() << std::endl;
  v =  as.evaluate(1e-8);
  std::cout << " v = " << v << " error = " << as.abs_error() << std::endl;
  v =  as.evaluate(1e-10);
  std::cout << " v = " << v << " error = " << as.abs_error() << std::endl;
  v =  as.evaluate(1e-15);
  std::cout << " v = " << v << " error = " << as.abs_error() << std::endl;
}

TEST(AdaptiveSummation, TwoDimensionsConstant) {
  int num_eval = 0;
  auto f = [&](const std::array<std::int64_t,2>& n) {
    return 1.0;
  };

  std::int64_t max_n = 100;

  std::array<std::pair<std::int64_t,std::int64_t>,2> r;
  r[0] = std::make_pair(1, max_n);
  r[1] = std::make_pair(1, max_n);

  alps::gf_extension::AdaptiveSummation<2,std::complex<double>, decltype(f)> as(f, r);
  auto v =  as.evaluate(1e-5);
  std::cout << " v = " << v << " num_eval " << num_eval << std::endl;
}

TEST(AdaptiveSummation, TwoDimensions) {
  int num_eval = 0;
  auto f = [&](const std::array<std::int64_t,2>& n) {
    double rtmp = n[0];
    double rtmp2 = n[1];
    ++num_eval;
    return 1.0 / (rtmp * rtmp2 * rtmp2);
  };

  std::int64_t max_n = 10000;

  std::array<std::pair<std::int64_t,std::int64_t>,2> r;
  r[0] = std::make_pair(1, max_n);
  r[1] = std::make_pair(1, std::numeric_limits<std::int64_t>::max()/100);

  alps::gf_extension::AdaptiveSummation<2,std::complex<double>, decltype(f)> as(f, r, false);

  double sum1 = 0.0;
  for (int n=1; n<=max_n; ++n) {
    sum1 += 1/(1.*n);
  }

  auto correct_v = sum1 * M_PI*M_PI/6;

  auto v =  as.evaluate(1e-5);
  std::cout << " v = " << v << " " << std::abs(v - correct_v) << " num_eval " << num_eval << std::endl;

  v =  as.evaluate(1e-6);
  std::cout << " v = " << v << " " << std::abs(v - correct_v) << " num_eval " << num_eval << std::endl;

  //v =  as.evaluate(1e-7);
  //std::cout << " v = " << v << " " << std::abs(v - correct_v) << " num_eval " << num_eval << std::endl;

  //v =  as.evaluate(1e-8);
  //std::cout << " v = " << v << " " << std::abs(v - std::pow(M_PI*M_PI/6, 2.0)) << " num_eval " << num_eval << std::endl;
}

TEST(AdaptiveSummation, Tnl_IR) {
  const double Lambda = 10.0;
  const int max_dim = 100;
  alps::gf_extension::fermionic_ir_basis basis(Lambda, max_dim);

  double tol = 1e-10;
  alps::gf_extension::interpolate_Tbar_ol Tbar(basis, tol);

  int dim = basis.dim();

  int num_eval = 0;
  auto f = [&Tbar, &dim, &num_eval](const std::array<std::int64_t,1>& n) {
    ++num_eval;
    return Tbar(n[0], dim-1)*Tbar(n[0], dim-2);
  };

  alps::gf_extension::AdaptiveSummation<1,std::complex<double>, decltype(f)> as(f, 1, std::numeric_limits<std::int64_t>::max()/100);
  std::cout << as.evaluate(1e-4) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-6) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-8) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << as.evaluate(1e-10) << " " << as.abs_error() << " num_eval " << num_eval << std::endl;

  auto f3 = [&Tbar, &dim](const std::array<std::int64_t,1>& n) {
    //return std::pow(std::abs(Tbar(2*n[0]+1, dim-1)), 2);
    return std::conj(Tbar(2*n[0]+1, dim-2)) * Tbar(2*n[0]+1, dim-2);
  };

  alps::gf_extension::AdaptiveSummation<1,std::complex<double>,decltype(f3)> as3(f3, -1E+15, 1E+15);
  std::cout << " norm " << as3.evaluate(1e-8) -1.0 << " " << as3.abs_error() << " num_eval " << num_eval << std::endl;
  std::cout << " norm " << as3.evaluate(1e-10)  -1.0<< " " << as3.abs_error() << " num_eval " << num_eval << std::endl;

  int num_eval2 = 0;
  auto f2 = [&Tbar, &dim, &num_eval2](const std::array<std::int64_t,2>& n) {
    ++num_eval2;
    return std::pow(
        std::abs(
            Tbar(2*n[0]+1, dim-1)*Tbar(2*n[0]+2*n[1]+1, dim-2)
        ),
        2
    );
  };

  std::cout << "debug " << f2(std::array<std::int64_t,2>{1,1}) << std::endl;

  //for (int n = 0; n < 10000; ++n) {
    //std::cout << n << " " << std::abs(Tbar(2*n+1, dim-1)) << std::endl;
  //}

  std::int64_t max_n = std::numeric_limits<std::int64_t>::max()/10;
  std::array<std::pair<std::int64_t,std::int64_t>,2> r;
  //r[0] = std::make_pair(1, std::numeric_limits<std::int64_t>::max()/100);
  //r[1] = std::make_pair(1, std::numeric_limits<std::int64_t>::max()/100);
  r[0] = std::make_pair(-max_n, max_n);
  r[1] = std::make_pair(-max_n, max_n);

  alps::gf_extension::AdaptiveSummation<2,std::complex<double>, decltype(f2)> as2(f2, r);
  std::cout << as2.evaluate(1e-4) << " " << as2.abs_error() << " num_eval " << num_eval2 << std::endl;
  std::cout << as2.evaluate(1e-5) << " " << as2.abs_error() << " num_eval " << num_eval2 << std::endl;
}
