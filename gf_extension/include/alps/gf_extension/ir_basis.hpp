#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>

#include <alps/gf/mesh.hpp>
#include "transformer.hpp"
#include "piecewise_polynomial.hpp"

namespace alps {
namespace gf_extension {

  /***
   * Construct a piecewise polynomial by means of cubic spline
   * @param T  we expect T=double
   * @param x_array  values of x in strictly ascending order
   * @param y_array  values of y
   */
  template<typename T>
  alps::gf::piecewise_polynomial<T> construct_piecewise_polynomial_cspline(
      const std::vector<double> &x_array, const std::vector<double> &y_array);

  /**
   * Abstract class representing an analytical continuation kernel
   */
  template<typename T>
  class kernel {
   public:
    virtual ~kernel() {};

    /// return the value of the kernel for given x and y in the [-1,1] interval.
    virtual T operator()(double x, double y) const = 0;

    /// return statistics
    virtual alps::gf::statistics::statistics_type get_statistics() const = 0;

#ifndef SWIG
    /// return a reference to a copy
    virtual boost::shared_ptr<kernel> clone() const = 0;
#endif
  };

#ifdef SWIG
%template(real_kernel) kernel<double>;
#endif

  /**
   * Fermionic kernel
   */
  class fermionic_kernel : public kernel<double> {
   public:
    fermionic_kernel(double Lambda) : Lambda_(Lambda) {}
    virtual ~fermionic_kernel() {};

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (Lambda_ * y > limit) {
        return std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
      } else if (Lambda_ * y < -limit) {
        return std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
      } else {
        return std::exp(-0.5 * Lambda_ * x * y) / (2 * std::cosh(0.5 * Lambda_ * y));
      }
    }

    alps::gf::statistics::statistics_type get_statistics() const {
      return alps::gf::statistics::FERMIONIC;
    }

#ifndef SWIG
    boost::shared_ptr<kernel> clone() const {
      return boost::shared_ptr<kernel>(new fermionic_kernel(Lambda_));
    }
#endif

   private:
    double Lambda_;
  };

  /**
   * Bosonic kernel
   */
  class bosonic_kernel : public kernel<double> {
   public:
    bosonic_kernel(double Lambda) : Lambda_(Lambda) {}
    virtual ~bosonic_kernel() {};

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (std::abs(Lambda_ * y) < 1e-10) {
        return std::exp(-0.5 * Lambda_ * x * y) / Lambda_;
      } else if (Lambda_ * y > limit) {
        return y * std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
      } else if (Lambda_ * y < -limit) {
        return -y * std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
      } else {
        return y * std::exp(-0.5 * Lambda_ * x * y) / (2 * std::sinh(0.5 * Lambda_ * y));
      }
    }

    alps::gf::statistics::statistics_type get_statistics() const {
      return alps::gf::statistics::BOSONIC;
    }

#ifndef SWIG
    boost::shared_ptr<kernel> clone() const {
      return boost::shared_ptr<kernel>(new bosonic_kernel(Lambda_));
    }
#endif

   private:
    double Lambda_;
  };

/**
 * Class template for kernel Ir basis
 * @tparam Scalar scalar type
 * @tparam Kernel kernel type
 */
  template<typename Scalar>
  class ir_basis {
   public:
    /**
     * Constructor
     * @param knl  kernel
     * @param max_dim  max number of basis functions computed.
     * @param cutoff  we drop basis functions corresponding to small singular values  |s_l/s_0~ < cutoff.
     * @param N       dimension of matrices for SVD. 500 may be big enough al least up to Lambda = 10^4.
     */
    ir_basis(const kernel<Scalar>& knl, int max_dim, double cutoff = 1e-10, int N = 501);

   private:
    //typedef alps::gf::piecewise_polynomial<double> pp_type;

    boost::shared_ptr<kernel<Scalar> > p_knl_;
    std::vector<alps::gf::piecewise_polynomial<double> > basis_functions_;

   public:
    /**
     * Compute the values of the basis functions for a given x.
     * @param x    x = 2 * tau/beta - 1  (-1 <= x <= 1)
     * @param val  results
     */
    void value(double x, std::vector<double> &val) const;

    /**
     * Return a reference to the l-th basis function
     * @param l l-th basis function
     * @return  reference to the l-th basis function
     */
    const alps::gf::piecewise_polynomial<double> &operator()(int l) const { return basis_functions_[l]; }

    /**
     * Return a reference to all basis functions
     */
    const std::vector<alps::gf::piecewise_polynomial<double> > all() const { return basis_functions_; }

    /**
     * Return number of basis functions
     * @return  number of basis functions
     */
    int dim() const { return basis_functions_.size(); }

    /// Return statistics
    alps::gf::statistics::statistics_type get_statistics() const {
      return p_knl_->get_statistics();
    }

    /**
     * Construct a mesh
     */
    alps::gf::numerical_mesh<double> construct_mesh(double beta) const {
      return alps::gf::numerical_mesh<double>{beta, all(), get_statistics()};
    }

    /**
     * Compute transformation matrix to Matsubara freq.
     * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
     * @param n_min min Matsubara freq. index
     * @param n_max max Matsubara freq. index
     * @param Tnl max
     */
#ifndef SWIG
    void compute_Tnl(
        int n_min, int n_max,
        boost::multi_array<std::complex<double>, 2> &Tnl
    ) const;

    void compute_Tnl(
        int n_min, int n_max,
        Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) const;

    void compute_Tnl(
        const std::vector<long>& n_vec,
        Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) const {
      alps::gf_extension::compute_transformation_matrix_to_matsubara<double>(n_vec,
                                                                             p_knl_->get_statistics(),
                                                                             basis_functions_,
                                                                             Tnl);
    }
#endif

    Eigen::Tensor<std::complex<double>,2>
    compute_Tnl(const std::vector<long>& n_vec) {
        Eigen::Tensor<std::complex<double>, 2> Tnl;
        compute_Tnl(n_vec, Tnl);
        return Tnl;
    }

  };

#ifdef SWIG
%template(real_ir_basis) ir_basis<double>;
#endif

  /**
   * Fermionic IR basis
   */
  class fermionic_ir_basis : public ir_basis<double> {
   public:
    fermionic_ir_basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501)
        : ir_basis<double>(fermionic_kernel(Lambda), max_dim, cutoff, N) {}
  };

  /**
   * Bosonic IR basis
   */
  class bosonic_ir_basis : public ir_basis<double> {
   public:
    bosonic_ir_basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501)
        : ir_basis<double>(bosonic_kernel(Lambda), max_dim, cutoff, N) {}
  };



  Eigen::Tensor<std::complex<double>, 3>
  compute_w_tensor(
      const std::vector<long> &n_vec,
      const fermionic_ir_basis &basis_f,
      const bosonic_ir_basis &basis_b);

  void
  compute_C_tensor(
      const fermionic_ir_basis &basis_f,
      const bosonic_ir_basis &basis_b,
      Eigen::Tensor<double,6>& C_tensor,
      double ratio_sum=1.01,
      int max_n_exact_sum=10000
  );

}
}
