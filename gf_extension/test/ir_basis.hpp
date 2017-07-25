#include <gtest/gtest.h>

#include "alps/gf_extension/ir_basis.hpp"
#include "alps/gf_extension/aux.hpp"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/multi_array.hpp>

#include "../include/alps/gf_extension/piecewise_polynomial.hpp"

#include <boost/math/special_functions/factorials.hpp>

#include "spline.hpp"

void compute_Tnl_legendre(int n_matsubara, int n_legendre, boost::multi_array<std::complex<double>,2> &Tnl) {
  double sign_tmp = 1.0;
  Tnl.resize(boost::extents[n_matsubara][n_legendre]);
  for (int im = 0; im < n_matsubara; ++im) {
    std::complex<double> ztmp(0.0, 1.0);
    for (int il = 0; il < n_legendre; ++il) {
      Tnl[im][il] = sign_tmp * ztmp * std::sqrt(2 * il + 1.0) * boost::math::sph_bessel(il, 0.5 * (2 * im + 1) * M_PI);
      ztmp *= std::complex<double>(0.0, 1.0);
    }
    sign_tmp *= -1;
  }
}

template<alps::gf::statistics::statistics_type s>
struct basis_type {
  typedef alps::gf_extension::fermionic_ir_basis type;
};

template<>
struct basis_type<alps::gf::statistics::BOSONIC> {
  typedef alps::gf_extension::bosonic_ir_basis type;
};



Eigen::Tensor<std::complex<double>,2>
to_Tnl_pn(const Eigen::Tensor<std::complex<double>,2>& Tnl, alps::gf::statistics::statistics_type s) {
  int niw = Tnl.dimension(0);
  int nl = Tnl.dimension(1);
  Eigen::Tensor<std::complex<double>,2> Tnl_pn(2*niw, nl);
  Tnl_pn.setZero();
  if (s==alps::gf::statistics::FERMIONIC) {
    for (int l=0; l<nl; ++l) {
      for (int n=0; n<niw; ++n) {
        Tnl_pn(niw + n, l) = Tnl(n, l);
        Tnl_pn(niw - 1 - n, l) = std::conj(Tnl(n, l));
      }
    }
  } else if (s==alps::gf::statistics::BOSONIC) {
    for (int l=0; l<nl; ++l) {
      for (int n=0; n<niw; ++n) {
        Tnl_pn(niw + n, l) = Tnl(n, l);
      }
      for (int n=1; n<niw; ++n) {
        Tnl_pn(niw - n, l) = std::conj(Tnl(n, l));
      }
    }
  } else {
    throw std::runtime_error("Unknown statistics type");
  }
  return Tnl_pn;
}
