#include "ir_basis.hpp"

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    boost::multi_array<Scalar,3> coeff(boost::extents[n_basis][n_section][k+1]);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

        for (int s = 0; s < n_section; ++s) {
            double rtmp = 1.0;
            for (int l = 0; l < k + 1; ++l) {
                if (n - l < 0) {
                    break;
                }
                if (l > 0) {
                    rtmp /= l;
                    rtmp *= n + 1 - l;
                }
                coeff[s][l] = rtmp * std::pow(section_edges[s], n-l);
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(nfunctions[n].compute_value(x), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++ m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]), (std::pow(1.0,n+m+1)-std::pow(-1.0,n+m+1))/(n+m+1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(4 * nfunctions[n].compute_value(x), (4.0*nfunctions[n]).compute_value(x), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x),
                        (nfunctions[n] + nfunctions[m]).compute_value(x), 1e-8);
            EXPECT_NEAR(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x),
                        (nfunctions[n] - nfunctions[m]).compute_value(x), 1e-8);
        }
    }

    alps::gf::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(nfunctions[1].compute_value(x) * std::sqrt(2.0/3.0), x, 1E-8);
}


TEST(PiecewisePolynomial, Multiply) {
  typedef alps::gf::piecewise_polynomial<double> pp_type;

  const int n_section = 10;
  const int nptr = n_section + 1;
  const int n = 4, m = 3;

  std::vector<double> y(nptr), y2(nptr);
  auto x = alps::gf_extension::linspace(-1.0, 2.0, nptr);
  for (int i = 0; i < nptr; ++i) {
    y[i] = std::pow(x[i], n);
    y2[i] = std::pow(x[i], m);
  }
  auto p1 = alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y);
  auto p2 = alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y2);

  auto p_prod = alps::gf_extension::multiply(p1, p2);

  for (int i = 0; i < nptr; ++i) {
    ASSERT_NEAR(p1.compute_value(x[i])*p2.compute_value(x[i]), p_prod.compute_value(x[i]), 1e-10);
  }
}

TEST(PiecewisePolynomial, Integral) {
  typedef alps::gf::piecewise_polynomial<double> pp_type;

  const int n_section = 2000;
  const int nptr = n_section + 1;
  const int n = 4;
  const double xmin = -2.0, xmax = 1.0;

  std::vector<double> y(nptr);
  auto x = alps::gf_extension::linspace(xmin, xmax, nptr);
  for (int i = 0; i < nptr; ++i) {
    y[i] = std::pow(x[i], n);
  }
  auto p = alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y);

  ASSERT_NEAR(
      alps::gf_extension::integrate(p),
      (std::pow(xmax,n+1)-std::pow(xmin,n+1))/(n+1),
      1e-8
  );
}

/*
TEST(PiecewisePolynomial, IntegralWithExp) {
  typedef alps::gf::piecewise_polynomial<double> pp_type;

  const int n_section = 2000;
  const int nptr = n_section + 1;
  const int n = 0;
  const double xmin = -1.0, xmax = 1.0;

  std::vector<double> y(nptr);
  auto x = alps::gf_extension::linspace(xmin, xmax, nptr);
  for (int i = 0; i < nptr; ++i) {
    y[i] = std::pow(x[i], n);
    //std::cout << "debug_xy " << x[i] << " " << y[i] << std::endl;
  }
  std::vector<pp_type> p {{alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y)}};

  std::vector<double> w;
  w.push_back(0.5*M_PI);

  Eigen::Tensor<std::complex<double>,2> results;
  alps::gf_extension::compute_integral_with_exp(w, p, results);

  std::cout << results(0,0) << std::endl;

  //ASSERT_NEAR(
      //alps::gf_extension::integrate(p),
      //(std::pow(xmax,n+1)-std::pow(xmin,n+1))/(n+1),
      //1e-8
  //);
}
 */

TEST(Legendre, Nodes) {
  double eps = 1e-8;

  std::vector<double> nodes_l0 = alps::gf_extension::detail::compute_legendre_nodes(0);
  ASSERT_TRUE(nodes_l0.size()==0);

  std::vector<double> nodes_l1 = alps::gf_extension::detail::compute_legendre_nodes(1);
  ASSERT_TRUE(nodes_l1.size()==1);
  ASSERT_TRUE(std::abs(nodes_l1[0]-0.0) < eps);

  std::vector<int> l_vec{1, 2, 3, 4, 50, 100, 200};
  for (auto l : l_vec) {
    std::vector<double> nodes = alps::gf_extension::detail::compute_legendre_nodes(l);
    ASSERT_TRUE(nodes.size()==l);
    for (int i=0; i<l; ++i) {
      ASSERT_TRUE(std::abs(boost::math::legendre_p(l, nodes[i])) < eps);
      if (i<l-1) {
        ASSERT_TRUE(std::abs(nodes[i]-nodes[i+1]) > eps);
      }
    }
  }
}

TEST(Legendre, PiecewisePolynomials) {
  double eps = 1e-2;
  int Nl = 100;

  try {
    std::vector<alps::gf::piecewise_polynomial<double>>
        polynomials = alps::gf_extension::construct_cubic_spline_normalized_legendre_polynomials(Nl);

    double de_cutoff = 2.5;
    int N = 1000;
    std::vector<double> tx_vec = alps::gf_extension::detail::linspace<double>(-de_cutoff, de_cutoff, N);
    std::vector<double> x_vec(N);
    for (int i = 0; i < N; ++i) {
      x_vec[i] = std::tanh(0.5 * M_PI * std::sinh(tx_vec[i]));
    }

    for (int l = 0; l < Nl; ++l) {
      for (auto x : x_vec) {
        ASSERT_NEAR(polynomials[l].compute_value(x), boost::math::legendre_p(l, x) * std::sqrt(l+0.5), eps);
      }
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

template<class T>
class HighTTest : public testing::Test {
};

typedef ::testing::Types<alps::gf_extension::fermionic_ir_basis, alps::gf_extension::bosonic_ir_basis> BasisTypes;

TYPED_TEST_CASE(HighTTest, BasisTypes);

TYPED_TEST(HighTTest, BasisTypes) {
  try {
    //construct ir basis
    const double Lambda = 0.01;//high T
    const int max_dim = 100;
    TypeParam basis(Lambda, max_dim);
    ASSERT_TRUE(basis.dim()>3);

    //IR basis functions should match Legendre polynomials
    const int N = 10;
    for (int i = 1; i < N - 1; ++i) {
      const double x = i * (2.0/(N-1)) - 1.0;

      double rtmp;

      //l = 0
      rtmp = basis(0).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(0+0.5)) < 0.02);

      //l = 1
      rtmp = basis(1).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(1+0.5)*x) < 0.02);

      //l = 2
      rtmp = basis(2).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(2+0.5)*(1.5*x*x-0.5)) < 0.02);
    }

    //check parity
    {
      double sign = -1.0;
      double x = 1.0;
      for (int l = 0; l < basis.dim(); ++l) {
        ASSERT_NEAR(basis(l).compute_value(x) + sign * basis(l).compute_value(-x), 0.0, 1e-8);
        sign *= -1;
      }
    }

    //check transformation matrix to Matsubara frequencies
    if (basis.get_statistics() == alps::gf::statistics::FERMIONIC) {

      const int N_iw = 3;

      boost::multi_array<std::complex<double>,2> Tnl_legendre(boost::extents[N_iw][3]),
          Tnl_ir(boost::extents[N_iw][3]);

      compute_Tnl_legendre(N_iw, 3, Tnl_legendre);

      basis.compute_Tnl(0, N_iw-1, Tnl_ir);
      for (int n = 0; n < N_iw; n++) {
        for (int l = 0; l < 3; ++l) {
          ASSERT_NEAR(std::abs(Tnl_ir[n][l] / (Tnl_legendre[n][l]) - 1.0), 0.0, 1e-5);
        }
      }
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(IrBasis, DiscretizationError) {
  try {
    const double Lambda = 10000.0;
    const int max_dim = 30;

    std::vector<int> n_points{500};

    const int nx = 10000;
    std::vector<double> x_points;
    for (int i = 0; i < nx; ++i) {
      x_points.push_back(
          std::tanh(
              0.5 * M_PI * std::sinh(i * 2.0 / (nx - 1) - 1.0)
          )
      );
    }

    for (int nptr: n_points) {
      alps::gf_extension::fermionic_ir_basis basis(Lambda, max_dim, 1e-10, nptr);
      alps::gf_extension::fermionic_ir_basis basis2(Lambda, max_dim, 1e-10, 2 * nptr);

      double max_diff = 0.0;
      for (int b = 0; b < std::max(basis.dim(), basis2.dim()); ++b) {
        for (double x : x_points) {
          max_diff = std::max(
              max_diff,
              std::abs(basis(b).compute_value(x) - basis2(b).compute_value(x))
          );
        }
      }
      ASSERT_NEAR(max_diff, 0.0, 0.01);
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

template<class T>
class SplineTest : public testing::Test {
};

TYPED_TEST_CASE(SplineTest, BasisTypes);

TYPED_TEST(SplineTest, BasisTypes) {
  //construct ir basis
  const double Lambda = 100.0;
  const int max_dim = 100;
  TypeParam basis(Lambda, max_dim);
  ASSERT_TRUE(basis.dim()>3);

  double tol = 1e-8;
  long min_n = 1000;
  long max_n = 10000000000;

  alps::gf_extension::interpolate_Tbar_ol interpolate(basis, tol, min_n, max_n);

  const auto& data_point = interpolate.data_point();

  std::vector<long> o_check;
  for (int i=0; i<data_point.size()-1; ++i) {
    o_check.push_back(
        static_cast<long>(0.5*(data_point[i]+data_point[i+1]))
    );
  }

  auto Tbar_ol = basis.compute_Tbar_ol(o_check);

  for (int i=0; i<o_check.size(); ++i) {
    for (int l = 0; l < basis.dim(); ++l) {
      auto o = o_check[i];
      auto v = interpolate(o_check[i], l);

      //std::cout << "o = " << o << " l= " << l << std::endl;

      ASSERT_NEAR(v.real(), Tbar_ol(i,l).real(), 1e-6);
      ASSERT_NEAR(v.imag(), Tbar_ol(i,l).imag(), 1e-6);

      if ((l+o)%2==0) {
        ASSERT_NEAR(std::abs(Tbar_ol(i,l).imag()), 0.0, 1e-8);
      } else {
        ASSERT_NEAR(std::abs(Tbar_ol(i,l).real()), 0.0, 1e-8);
      }
    }
  }
}

template<class T>
class InsulatingGtau : public testing::Test {
};

TYPED_TEST_CASE(InsulatingGtau, BasisTypes);

// Delta peaks at omega=+/- 1
TYPED_TEST(InsulatingGtau, BasisTypes) {
  try {
    const double Lambda = 300.0, beta = 100.0;
    const int max_dim = 100;
    //typedef basis_type<alps::gf::statistics::FERMIONIC>::type basis_t;
    //alps::gf_extension::fermionic_ir_basis basis(Lambda, max_dim, 1e-10, 501);
    TypeParam basis(Lambda, max_dim, 1e-10, 501);
    ASSERT_TRUE(basis.dim()>0);

    typedef alps::gf::piecewise_polynomial<double> pp_type;

    const int nptr = basis(0).num_sections() + 1;
    std::vector<double> x(nptr), y(nptr);
    for (int i = 0; i < nptr; ++i) {
      x[i] = basis(0).section_edge(i);
      if (basis.get_statistics()==alps::gf::statistics::FERMIONIC) {
        y[i] = std::exp(-0.5*beta)*std::cosh(-0.5*beta*x[i]);
      } else {
        y[i] = std::exp(-0.5*beta)*std::sinh(-0.5*beta*x[i]);
      }
    }
    pp_type gtau(alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y));

    std::vector<double> coeff(basis.dim());
    for (int l = 0; l < basis.dim(); ++l) {
      coeff[l] = gtau.overlap(basis(l)) * beta / std::sqrt(2.0);
    }

    std::vector<double> y_r(nptr, 0.0);
    for (int l = 0; l < 30; ++l) {
      for (int i = 0; i < nptr; ++i) {
        y_r[i] += coeff[l] * (std::sqrt(2.0)/beta) * basis(l).compute_value(x[i]);
      }
    }

    double max_diff = 0.0;
    for (int i = 0; i < nptr; ++i) {
      max_diff = std::max(std::abs(y[i]-y_r[i]), max_diff);
      ASSERT_TRUE(std::abs(y[i]-y_r[i]) < 1e-6);
    }

    //to matsubara freq.
    const int n_iw = 1000;
    boost::multi_array<std::complex<double>,2> Tnl(boost::extents[n_iw][basis.dim()]);
    basis.compute_Tnl(0, n_iw-1, Tnl);
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> Tnl_mat(n_iw, basis.dim()), coeff_vec(basis.dim(),1);
    coeff_vec.setZero();
    for (int l = 0; l < basis.dim(); ++l) {
      coeff_vec(l,0) = coeff[l];
      for (int n = 0; n < n_iw; ++n) {
        Tnl_mat(n,l) = Tnl[n][l];
      }
    }
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> coeff_iw = Tnl_mat * coeff_vec;

    const std::complex<double> zi(0.0, 1.0);
    for (int n = 0; n < n_iw; ++n) {
      double wn;
      if (basis.get_statistics()==alps::gf::statistics::FERMIONIC) {
        wn = (2. * n + 1) * M_PI / beta;
      } else {
        wn = (2. * n ) * M_PI / beta;
      }
      std::complex<double> z = -0.5 / (zi * wn - 1.0) - 0.5 / (zi * wn + 1.0);
      ASSERT_NEAR(std::abs(z-coeff_iw(n)), 0.0, 1e-8);
    }

    //compute Tnl_bar
    {
      std::vector<long> nvec;
      for (int n=0; n<2*n_iw; ++n) {
        nvec.push_back(static_cast<long>(n));
      }
      auto Tbar_ol = basis.compute_Tbar_ol(nvec);
      auto shift = basis.get_statistics()==alps::gf::statistics::FERMIONIC ? 1 : 0;

      for (int l = 0; l < basis.dim(); ++l) {
        for (int n = 0; n < n_iw; ++n) {
          ASSERT_NEAR(std::abs(Tnl_mat(n,l) - Tbar_ol(2*n+shift,l)), 0.0, 1e-8);
        }
      }

    }


  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

/*
TEST(PiecewisePolynomial, IntegrationWithPower) {
  //using dcomplex = std::complex<double>;

  const int k = 100;//order of piecewise polynomial
  const double rnd = 1.2;
  const double xmin = -1.0;
  const double xmax = 1.0;
  const int N = 100;//# of edges

  //rnd * x^n
  for (int n = 0; n < 10; ++n) {
    auto m = n;

    auto edges = alps::gf_extension::linspace(xmin, xmax, N);
    std::vector<double> vals(N);
    for (int i=0; i < edges.size(); ++i) {
      vals[i] = rnd * std::pow(edges[i], n);
    }
    auto p = alps::gf_extension::construct_piecewise_polynomial_cspline<double>(edges, vals);

    auto r = alps::gf_extension::integrate_with_power(m, p);
    std::cout << r << " " << rnd * (std::pow(xmax,n+m+1)-std::pow(xmin, n+m+1))/(n+m+1) << std::endl;
  }
}
*/

namespace g = alps::gf;
namespace ge = alps::gf_extension;

class GfTest : public testing::Test {
 protected:
  typedef alps::gf::piecewise_polynomial<double> pp_type;
  using gl_type = ge::nmesh_three_index_gf<double, g::index_mesh, g::index_mesh>;
  using complex_gl_type = ge::nmesh_three_index_gf<std::complex<double>, g::index_mesh, g::index_mesh>;
  using gtau_type = ge::itime_three_index_gf<double, g::index_mesh, g::index_mesh>;
  using gomega_type = ge::omega_three_index_gf<std::complex<double>, g::index_mesh, g::index_mesh>;

  double Lambda = 300.0;
  double beta = 100.0;
  int max_dim = 100;
  alps::gf_extension::fermionic_ir_basis basis;
  gl_type Gl;
  gtau_type Gtau;
  gomega_type Gomega;

  double compute_gx(double x) const {
    return std::exp(-0.5*beta)*std::cosh(-0.5*beta*x);
  }

  double compute_gtau(double tau) const {
    return compute_gx( 2 * tau/beta - 1);
  }

  std::complex<double> compute_gomega(int n) const {
    const std::complex<double> zi(0.0, 1.0);
    double wn = (2.*n+1)*M_PI/beta;
    return - 0.5/(zi*wn - 1.0) - 0.5/(zi*wn + 1.0);
  }

  GfTest() : basis(alps::gf_extension::fermionic_ir_basis(Lambda, max_dim, 1e-10, 501)),
             Gtau(g::itime_mesh(beta, basis(0).num_sections()+1), g::index_mesh(1), g::index_mesh(1)),
             Gl(g::numerical_mesh<double>(beta, basis.all(), g::statistics::FERMIONIC), g::index_mesh(1), g::index_mesh(1))
  {}

  virtual void SetUp() {
    try {
      // Compute G(tau) for a model system: Delta peaks at omega = +/- 1
      const int nptr = basis(0).num_sections() + 1;
      std::vector<double> x(nptr), y(nptr);
      for (int i = 0; i < nptr; ++i) {
        x[i] = basis(0).section_edge(i);
        y[i] = compute_gx(x[i]);
      }
      pp_type gtau(alps::gf_extension::construct_piecewise_polynomial_cspline<double>(x, y));

      // Then expand G(tau) in terms of the ir basis to compute the coefficients
      std::vector<double> coeff(basis.dim());
      for (int l = 0; l < basis.dim(); ++l) {
        coeff[l] = gtau.overlap(basis(l)) * beta / std::sqrt(2.0);
      }

      // Then transform the data back to imaginary-time domain as y_r[i] and compare it to the original one y[i].
      std::vector<double> y_r(nptr, 0.0);
      for (int l = 0; l < 30; ++l) {
        for (int i = 0; i < nptr; ++i) {
          y_r[i] += coeff[l] * (std::sqrt(2.0)/beta) * basis(l).compute_value(x[i]);
        }
      }

      double max_diff = 0.0;
      for (int i = 0; i < nptr; ++i) {
        max_diff = std::max(std::abs(y[i]-y_r[i]), max_diff);
        ASSERT_TRUE(std::abs(y[i]-y_r[i]) < 1e-6);
      }

      const auto i0 = g::index_mesh::index_type(0);

      for (int i = 0; i < nptr; ++i) {
        double tau = beta * i/(nptr-1);
        Gtau(g::itime_mesh::index_type(i), i0, i0) = compute_gtau(tau);
      }

      // Construct a gf object of numerical mesh
      for (int l = 0; l < basis.dim(); ++l) {
        Gl(g::numerical_mesh<double>::index_type(l), i0, i0) = coeff[l];
      }


    } catch (const std::exception& e) {
      FAIL() << e.what();
    }
  }

  //virtual void TearDown() {}

};


TEST_F(GfTest, Gtau) {
  const auto i0 = g::index_mesh::index_type(0);
  for (int i = 0; i < Gtau.mesh1().extent(); ++i) {
    double tau = Gtau.mesh1().points()[i];
    ASSERT_TRUE(std::abs(Gtau(g::itime_mesh::index_type(i), i0, i0)-compute_gtau(tau)) < 1e-5);
  }
}

TEST_F(GfTest, IrtoTau) {
  const auto i0 = g::index_mesh::index_type(0);

  ge::transformer<gtau_type, gl_type> c(Gtau.mesh1().extent(), Gl.mesh1());

  gtau_type Gtau_tmp(c(Gl));

  for (int i = 0; i < Gtau_tmp.mesh1().extent(); ++i) {
    double tau = Gtau.mesh1().points()[i];
    ASSERT_TRUE(std::abs(Gtau_tmp(g::itime_mesh::index_type(i), i0, i0)-compute_gtau(tau)) < 1e-6);
  }
}

TEST_F(GfTest, IRtoMatsubara) {
  const auto i0 = g::index_mesh::index_type(0);

  const int niw = 1000;

  ge::transformer<gomega_type, gl_type> c(niw, Gl.mesh1());

  gomega_type Gomega_tmp(c(Gl));

  for (int i = 0; i < Gomega_tmp.mesh1().extent(); ++i) {
    ASSERT_TRUE(
        std::abs(
            Gomega_tmp(g::matsubara_positive_mesh::index_type(i), i0, i0)-compute_gomega(i)
        ) < 1e-6
    );
  }
}


/*
TEST_F(GfTest, MatsubaraToIR) {
const auto i0 = g::index_mesh::index_type(0);

const int niw = 10000;

//gomega_type Gomega(g::matsubara_positive_mesh(beta, niw), Gl.mesh2(), Gl.mesh3());

//ge::transformer<complex_gl_type, gomega_type> c(Gl.mesh1(), Gomega.mesh1());

//gomega_type Gomega_tmp(c(Gl));

 for (int i = 0; i < Gomega_tmp.mesh1().extent(); ++i) {
  ASSERT_TRUE(
      std::abs(
          Gomega_tmp(g::matsubara_positive_mesh::index_type(i), i0, i0)-compute_gomega(i)
      ) < 1e-6
  );
}
}
*/



TEST(G2, wtensor) {
  using dcomplex = std::complex<double>;

  double Lambda = 1E+2;
  int max_dim_f = 5;
  int max_dim_b = 5;
  const int niw_sum = 10000;
  const int niw = 10;

  namespace ge = alps::gf_extension;

  ge::fermionic_ir_basis basis_f(Lambda, max_dim_f);
  ge::bosonic_ir_basis basis_b(Lambda, max_dim_b);
  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();

  Eigen::Tensor<dcomplex,2> Tnl_f, Tnl_b;
  basis_f.compute_Tnl(0, niw_sum, Tnl_f);
  basis_b.compute_Tnl(0, niw_sum, Tnl_b);

  auto Tnl_f_pn = to_Tnl_pn(Tnl_f, alps::gf::statistics::FERMIONIC);
  auto Tnl_b_pn = to_Tnl_pn(Tnl_b, alps::gf::statistics::BOSONIC);

  //Compute w(l, l^prime, n) by summing one freq., which will suffer from truncation errors
  Eigen::Tensor<dcomplex,3> w_tensor(niw, dim_b, dim_f);
  auto back_to_range = [](int i) {return std::max(std::min(i, 2*niw_sum),0);};
  for (int lp = 0; lp < dim_f; ++lp) {
    for (int l = 0; l < dim_b; ++l) {
      for (int n = 0; n < niw; ++n) {
        const auto min_m = back_to_range(-n);
        const auto max_m = back_to_range(2*niw_sum -n);
        dcomplex tmp = 0.0;
        for (int m = min_m; m < max_m; ++m) {
          tmp += std::conj(Tnl_b_pn(m,l)) * Tnl_f_pn(m+n, lp);
        }
        w_tensor(n,l,lp) = tmp;

        //if ((l+lp+n)%2==0) {
          //ASSERT_NEAR(std::abs(w_tensor(n,l,lp).real()), 0.0, 1e-5);
        //} else {
          //ASSERT_NEAR(std::abs(w_tensor(n,l,lp).imag()), 0.0, 1e-5);
        //}
      }
    }
  }

  //Alternatively, use the tau formula, which will give more accurate results.
  std::vector<long> n_vec;
  for (int i=0; i < niw; ++i) {
    n_vec.push_back(i);
  }
  auto w_tensor2 = ge::compute_w_tensor(n_vec, basis_f, basis_b);

  //The two results should match approximately.
  for (int lp=0; lp<dim_f; ++lp) {
    for (int l=0; l<dim_b; ++l) {
      for (int n=0; n<niw; ++n) {
        ASSERT_NEAR(abs(w_tensor(n, l, lp)-w_tensor2(n, l, lp)), 0.0,  0.002);
      }
    }
  }
}

class G2WTensorTest : public ::testing::TestWithParam<double> {
};


/*
TEST_P(G2WTensorTest, Spline) {
  using dcomplex = std::complex<double>;

  double Lambda = GetParam();
  int max_dim_f = 20;
  int max_dim_b = 20;
  double ratio = 1.02;
  double ratio2 = std::pow(ratio, 0.511);

  namespace ge = alps::gf_extension;

  ge::fermionic_ir_basis basis_f(Lambda, max_dim_f);
  ge::bosonic_ir_basis basis_b(Lambda, max_dim_b);
  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();
  double max_n = 1E+10;

  //mesh 1
  std::vector<long> n_vec;
  std::vector<long> n_vec_dense;
  for (int i=0; i < 200; ++i) {
    n_vec.push_back(i);
    n_vec_dense.push_back(i);
  }
  while (n_vec.back() < max_n) {
    n_vec.push_back(long(n_vec.back()*ratio));
  }
  while (n_vec_dense.back() < max_n) {
    n_vec_dense.push_back(long(n_vec_dense.back()*ratio2));
  }

  int n_mesh = n_vec.size();
  int n_mesh_dense = n_vec_dense.size();

  auto w_tensor = ge::compute_w_tensor(n_vec, basis_f, basis_b);
  auto w_tensor_dense = ge::compute_w_tensor(n_vec_dense, basis_f, basis_b);


  std::vector<double> y_re_array(n_vec.size()-1);
  std::vector<double> y_imag_array(n_vec.size()-1);
  double max_diff = 0.0;
  for (int lp = 0; lp < dim_f; ++lp) {
    for (int l = 0; l < dim_b; ++l) {

      for (int n = 0; n < n_vec.size()-1; ++n) {
        x_array[n] = std::log(n_vec[n+1]);
        y_re_array[n] = w_tensor(n+1, l, lp).real();
        y_imag_array[n] = w_tensor(n+1, l, lp).imag();
      }
      tk::spline spline_re;
      tk::spline spline_imag;
      spline_re.set_points(x_array, y_re_array);
      spline_imag.set_points(x_array, y_imag_array);

      for (int n=1; n<n_mesh_dense; ++n) {
        auto log_n = std::log(n_vec_dense[n]);
        max_diff = std::max(max_diff, std::abs(
            w_tensor_dense(n,l,lp) - dcomplex(spline_re(log_n), spline_imag(log_n))
          )
        );
      }
    }
  }
  ASSERT_NEAR(max_diff, 0.0, 1e-5);
}
*/

TEST_P(G2WTensorTest, CTensor) {
  using dcomplex = std::complex<double>;

  double Lambda = GetParam();
  int max_dim_f = 15;
  int max_dim_b = 15;

  namespace ge = alps::gf_extension;

  ge::fermionic_ir_basis basis_f(Lambda, max_dim_f, 1e-12);
  ge::bosonic_ir_basis basis_b(Lambda, max_dim_b, 1e-12);
  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();

  std::cout << "dim " << dim_f << " " << dim_b << std::endl;

  Eigen::Tensor<double,6> C_tensor;
  //compute_C_tensor(basis_f, basis_b, C_tensor, 1.02);
  compute_C_tensor(basis_f, basis_b, C_tensor, 1.05, 200);

  Eigen::Tensor<double,6> C_tensor2;
  compute_C_tensor(basis_f, basis_b, C_tensor2, 1.02, 200);

  //std::cout << C_tensor.dimension(0) << std::endl;
  //std::cout << C_tensor.dimension(1) << std::endl;
  //std::cout << C_tensor.dimension(2) << std::endl;
  //std::cout << C_tensor.dimension(3) << std::endl;
  //for (int i=0; i<max_dim_f; ++i) {
    //std::cout << i << " " << C_tensor(i,max_dim_f-1,max_dim_b-1,max_dim_f-1,max_dim_f-1,max_dim_b-1) << " " << C_tensor2(i,max_dim_f-1,max_dim_b-1,max_dim_f-1,max_dim_f-1,max_dim_b-1) << std::endl;
    //std::cout << i << " " << C_tensor(i,max_dim_f-1,max_dim_b-1,max_dim_f-1,max_dim_f-1,0) << " " << C_tensor2(i,max_dim_f-1,max_dim_b-1,max_dim_f-1,max_dim_f-1,0) << std::endl;
  //}
  std::cout << Lambda << " max_diff " << (C_tensor-C_tensor2).abs().maximum();

}

INSTANTIATE_TEST_CASE_P(G2WTensorTestLambda,
                        G2WTensorTest,
                        ::testing::Values(1000.0, 10000.0));
//::testing::Values(10.0, 1000.0, 10000.0));

