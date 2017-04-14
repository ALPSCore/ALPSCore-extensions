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
#include <alps/gf_extension/converter.hpp>


namespace alps {
namespace gf_extension {
namespace ir {

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
    /// return the value of the kernel for given x and y in the [-1,1] interval.
    virtual T operator()(double x, double y) const = 0;

    /// return statistics
    virtual alps::gf_extension::statistics get_statistics() const = 0;

    /// return a reference to a copy
    virtual boost::shared_ptr<kernel> clone() const = 0;
  };

  /**
   * Fermionic kernel
   */
  class fermionic_kernel : public kernel<double> {
   public:
    fermionic_kernel(double Lambda) : Lambda_(Lambda) {}

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

    alps::gf_extension::statistics get_statistics() const {
      return fermionic;
    }

    boost::shared_ptr<kernel> clone() const {
      return boost::shared_ptr<kernel>(new fermionic_kernel(Lambda_));
    }

   private:
    double Lambda_;
  };

  /**
   * Bosonic kernel
   */
  class bosonic_kernel : public kernel<double> {
   public:
    bosonic_kernel(double Lambda) : Lambda_(Lambda) {}

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

    alps::gf_extension::statistics get_statistics() const {
      return bosonic;
    }

    boost::shared_ptr<kernel> clone() const {
      return boost::shared_ptr<kernel>(new bosonic_kernel(Lambda_));
    }

   private:
    double Lambda_;
  };

/**
 * Class template for kernel Ir basis
 * @tparam Scalar scalar type
 * @tparam Kernel kernel type
 */
  template<typename Scalar>
  class basis {
   public:
    /**
     * Constructor
     * @param knl  kernel
     * @param max_dim  max number of basis functions computed.
     * @param cutoff  we drop basis functions corresponding to small singular values  |s_l/s_0~ < cutoff.
     * @param N       dimension of matrices for SVD. 500 may be big enough al least up to Lambda = 10^4.
     */
    basis(const kernel<Scalar>& knl, int max_dim, double cutoff = 1e-10, int N = 501);

   private:
    typedef alps::gf::piecewise_polynomial<double> pp_type;

    boost::shared_ptr<kernel<Scalar> > p_knl_;
    std::vector<pp_type> basis_functions_;

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
    const pp_type &operator()(int l) const { return basis_functions_[l]; }

    /**
     * Return a reference to all basis functions
     */
    const std::vector<pp_type> all() const { return basis_functions_; }

    /**
     * Return number of basis functions
     * @return  number of basis functions
     */
    int dim() const { return basis_functions_.size(); }

    /// Return statistics
    alps::gf_extension::statistics get_statistics() const {
      return p_knl_->get_statistics();
    }

    /**
     * Compute transformation matrix to Matsubara freq.
     * The computation may take some time. You may store the result somewhere and do not call this routine frequenctly.
     * @param n_min min Matsubara freq. index
     * @param n_max max Matsubara freq. index
     * @param Tnl max
     */
    void compute_Tnl(
        int n_min, int n_max,
        boost::multi_array<std::complex<double>, 2> &Tnl
    ) const;

    void compute_Tnl(
        int n_min, int n_max,
        Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) const;
  };

  /**
   * Fermionic IR basis
   */
  class fermionic_basis : public basis<double> {
   public:
    fermionic_basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501)
        : basis<double>(fermionic_kernel(Lambda), max_dim, cutoff, N) {}
  };

  /**
   * Bosonic IR basis
   */
  class bosonic_basis : public basis<double> {
   public:
    bosonic_basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501)
        : basis<double>(bosonic_kernel(Lambda), max_dim, cutoff, N) {}
  };

}
}
}
