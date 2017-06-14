#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <set>
#include <assert.h>

#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>

#include <alps/gf/mesh.hpp>
#include "transformer.hpp"
#include "piecewise_polynomial.hpp"

#include "detail/spline.hpp"

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
    alps::gf::numerical_mesh<double> construct_mesh(double beta, int nl) const {
      if (nl > dim()) {
        throw std::invalid_argument("nl cannot be larger dim().");
      }
      std::vector<alps::gf::piecewise_polynomial<double>> bf(nl);
      std::copy(basis_functions_.begin(), basis_functions_.begin()+nl, bf.begin());
      return alps::gf::numerical_mesh<double>{beta, bf, get_statistics()};
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
    compute_Tnl(const std::vector<long>& n_vec) const {
        Eigen::Tensor<std::complex<double>, 2> Tnl;
        compute_Tnl(n_vec, Tnl);
        return Tnl;
    }

    Eigen::Tensor<std::complex<double>,2>
    compute_Tbar_ol(const std::vector<long>& o_vec) const {
      int no = o_vec.size();
      int nl = basis_functions_.size();

      Eigen::Tensor<std::complex<double>,2> Tbar_ol(no, nl);
      alps::gf_extension::compute_Tbar_ol(o_vec, basis_functions_, Tbar_ol);

      return Tbar_ol;
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

  template<typename T>
  std::vector<T>
  to_unique_vec(const std::vector<T>& vec_in) {
    std::set<T> s(vec_in.begin(), vec_in.end());
    std::vector<T> vec_out;
    std::copy(s.begin(), s.end(), std::back_inserter(vec_out));
    return vec_out;
  }

  class interpolate_Tbar_ol {
   public:
    interpolate_Tbar_ol(const ir_basis<double>& basis, double tol=1e-8, long o_max=1000000000000000) : max_exact_o_(200) {
      //if (max_exact_o_ < 200) {
        //throw std::runtime_error("max_exact_o_ should not be smaller than 200.");
      //}

      int nl = basis.dim();

      //Compute values for n <= max_exact_o_
      {
        std::vector<long> o_tmp;
        for (int o=0; o<=max_exact_o_; ++o) {
          o_tmp.push_back(o);
        }
        Tbar_ol_ = basis.compute_Tbar_ol(o_tmp);
      }

      //Construct initial mesh for interpolation
      double ratio = 1.02;
      std::vector<double> weight;
      construct_log_mesh(o_max, max_exact_o_, ratio, o_vec_, weight);

      //Compute values
      while (true) {
        //std::cout << "loop " << std::endl;
        //Construct a unique set of o from o_vec_
        std::vector<long> o_intpl;
        {
          std::set<long> o_set;
          for (auto o : o_vec_) {
            if (o > max_exact_o_) {
              o_set.insert(o);
              o_set.insert(std::min(o+1, o_max));
            }
          }
          o_intpl.resize(0);
          std::copy(o_set.begin(), o_set.end(), std::back_inserter(o_intpl));
        }

        auto Tbar_ol_intpl = basis.compute_Tbar_ol(o_intpl);

        splines_re_.resize(0);
        splines_im_.resize(0);
        splines_re_.resize(nl);
        splines_im_.resize(nl);

        std::vector<double> x_array_even, x_array_odd;
        std::vector<double> y_array;

        for (auto o : o_intpl) {
          if (o%2==0) {
            x_array_even.push_back(std::log(1.*o));
          } else {
            x_array_odd.push_back(std::log(1.*o));
          }
        }

        for (int l=0; l < nl; ++l) {
          //Real part: l+o=even
          y_array.resize(0);
          for (int i = 0; i < o_intpl.size(); ++i) {
            if ((o_intpl[i]+l)%2==0) {
              y_array.push_back(Tbar_ol_intpl(i, l).real());
            }
          }
          if (l%2==0) {
            splines_re_[l].set_points(x_array_even, y_array);
          } else {
            splines_re_[l].set_points(x_array_odd, y_array);
          }

          //Imaginary part: l+o+1=even
          y_array.resize(0);
          for (int i = 0; i < o_intpl.size(); ++i) {
            if ((o_intpl[i]+l)%2==1) {
              y_array.push_back(Tbar_ol_intpl(i, l).imag());
            }
          }
          if (l%2==0) {
            splines_im_[l].set_points(x_array_odd, y_array);
          } else {
            splines_im_[l].set_points(x_array_even, y_array);
          }
        }

        //Check errors and update o_vec_
        {
          bool converged = true;

          std::vector<long> o_check;
          for (int i=0; i<o_intpl.size()-1; ++i) {
            if (o_intpl[i+1] - o_intpl[i] > 2) {
              long omid = static_cast<long>(0.5*(o_intpl[i+1]+o_intpl[i]));
              o_check.push_back(omid);
              o_check.push_back(omid+1);
            }
          }
          auto Tbar_ol_check = basis.compute_Tbar_ol(o_check);
          for (int i=0; i<o_check.size(); ++i) {
            bool flag = false;
            auto o = o_check[i];
            double max_diff = 0.0;
            for (int l=0; l<nl; ++l) {
              auto diff = std::abs(Tbar_ol_check(i,l)-this->operator()(o,l));
              max_diff = std::max(max_diff, diff);
              if (diff > tol) {
                flag = true;
                break;
              }
            }
            //std::cout << o << " max_diff " << max_diff << std::endl;
            if (flag) {
              //std::cout << "new o " << i << " : " << o << std::endl;
              o_vec_.push_back(o);
              converged = false;
            }
          }
          //unique set
          o_vec_ = to_unique_vec(o_vec_);
          std::sort(o_vec_.begin(), o_vec_.end());

          if (converged) {
            //std::cout << "converged " << o_vec_.size() << std::endl;
            break;
          } else {
            //std::cout << "not converged " << o_vec_.size() << std::endl;
          }
        }
      }
    }

    inline std::complex<double> operator()(long o, int l) const {
      if (o >= 0) {
        return get_value_for_positive_o(o, l);
      } else {
        return std::conj(get_value_for_positive_o(-o, l));
      }
    }

    const std::vector<long>& data_point() const {
      return o_vec_;
    }

   private:
    int max_exact_o_;
    std::vector<long> o_vec_;
    std::vector<tk::spline> splines_re_, splines_im_;
    Eigen::Tensor<std::complex<double>,2> Tbar_ol_;

    inline std::complex<double> get_value_for_positive_o(long o, int l) const {
      assert(o>=0);
      if (o <= max_exact_o_) {
        return Tbar_ol_(o, l);
      } else {
        if ((l+o)%2==0) {
          return std::complex<double>(
              splines_re_[l](std::log(1.*o)),
              0.0
          );
        } else {
          return std::complex<double>(
              0.0,
              splines_im_[l](std::log(1.*o))
          );
        }
      }
    }

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
