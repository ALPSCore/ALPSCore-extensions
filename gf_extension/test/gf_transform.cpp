#include <alps/gf/gf.hpp>
#include <alps/gf_extension/ir_basis.hpp>
#include <alps/gf_extension/transformer.hpp>

#include <gtest/gtest.h>

namespace g = alps::gf;
namespace ge = alps::gf_extension;

class GfTransformTest : public testing::Test {
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

  GfTransformTest() : basis(alps::gf_extension::fermionic_ir_basis(Lambda, max_dim, 1e-10, 501)),
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

TEST_F(GfTransformTest, Gtau) {
  const auto i0 = g::index_mesh::index_type(0);
  for (int i = 0; i < Gtau.mesh1().extent(); ++i) {
    double tau = Gtau.mesh1().points()[i];
    ASSERT_TRUE(std::abs(Gtau(g::itime_mesh::index_type(i), i0, i0)-compute_gtau(tau)) < 1e-5);
  }
}

TEST_F(GfTransformTest, IrtoTau) {
  const auto i0 = g::index_mesh::index_type(0);

  ge::transformer<gtau_type, gl_type> c(Gtau.mesh1().extent(), Gl.mesh1());

  gtau_type Gtau_tmp(c(Gl));

  for (int i = 0; i < Gtau_tmp.mesh1().extent(); ++i) {
    double tau = Gtau.mesh1().points()[i];
    ASSERT_TRUE(std::abs(Gtau_tmp(g::itime_mesh::index_type(i), i0, i0)-compute_gtau(tau)) < 1e-6);
  }
}

TEST_F(GfTransformTest, IRtoMatsubara) {
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
TEST_F(GfTransformTest, MatsubaraToIR) {
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
