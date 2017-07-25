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
#include "aux.hpp"

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
   * Construct Piecewise polynomials approximately representing Legenre polynomials normalized to 1 on [-1,1]
   * @param Nl number of Legendre polynomials
   * @return Piecewise polynomials
   */
  inline std::vector<alps::gf::piecewise_polynomial<double>>
  construct_cubic_spline_normalized_legendre_polynomials(int Nl) {
    int Nl_max = 100;
    int M = 40;
    double eps = 1e-10;

    std::vector<double> nodes = detail::compute_legendre_nodes(Nl_max);
    assert(Nl_max%2 == 0);
    std::vector<double> positve_nodes;
    for (auto n: nodes) {
      if (n > 0.0) {
        positve_nodes.push_back(n);
      }
    }
    positve_nodes.push_back(0);
    positve_nodes.push_back(1);
    std::sort(positve_nodes.begin(), positve_nodes.end());

    std::vector<double> x_points;
    for(int i=0; i<positve_nodes.size()-1; ++i) {
      double dx = (positve_nodes[i+1] - positve_nodes[i])/M;
      for (int j=0; j<M; ++j) {
        double x = positve_nodes[i] + dx * j;
        x_points.push_back(x);
        if (std::abs(x) > eps) {
          x_points.push_back(-x);
        }
      }
    }
    x_points.push_back(1);
    x_points.push_back(-1);
    std::sort(x_points.begin(), x_points.end());

    std::vector<alps::gf::piecewise_polynomial<double>> results;
    std::vector<double> y_vals(x_points.size());
    for (int l=0; l<Nl; ++l) {
      for (int j=0; j < x_points.size(); ++j) {
        y_vals[j] = boost::math::legendre_p(l, x_points[j]) * std::sqrt(l+0.5);
      }
      results.push_back(construct_piecewise_polynomial_cspline<double>(x_points, y_vals));
    }

    return results;
  }

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

    /// return lambda
    virtual double Lambda() const = 0;

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

    double Lambda() const {
      return Lambda_;
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

    double Lambda() const {
      return Lambda_;
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
    interpolate_Tbar_ol(const ir_basis<double>& basis, double tol=1e-8, long o_min=0, long o_max=1000000000000000) {
      int nl = basis.dim();

      //Compute values for n <= max_exact_o_
      {
        //std::vector<long> o_tmp;
        //for (int o=0; o<=max_exact_o_; ++o) {
          //o_tmp.push_back(o);
        //}
        //Tbar_ol_ = basis.compute_Tbar_ol(o_tmp);
      }

      if (o_max <= 1) {
        throw std::runtime_error("o_max must be larger than 1.");
      }
      if (o_max > std::numeric_limits<long>::max()/10) {
        throw std::runtime_error("o_max is too large.");
      }
      if (o_min >= o_max) {
        throw std::runtime_error("o_min >= o_max!");
      }

      //log mesh
      double ratio = 1.01;
      o_vec_.push_back(o_min);
      while(true) {
        long new_elem = std::max(o_vec_.back()+1, static_cast<long>(ratio * o_vec_.back()) );
        if (new_elem > o_max) {
          break;
        }
        //std::cout << new_elem << std::endl;
        o_vec_.push_back(new_elem);
      }
      o_vec_.push_back(o_max);
      o_vec_ = to_unique_vec(o_vec_);

      //Construct initial mesh for interpolation
      //double ratio = 1.1;
      //std::vector<double> weight;
      //construct_log_mesh(o_max, max_exact_o_, ratio, o_vec_, weight);

      //Compute values
      while (true) {
        //std::cout << "loop " << std::endl;
        register_to_cache(basis, o_vec_);

        splines_re_.resize(0);
        splines_im_.resize(0);
        splines_re_.resize(nl);
        splines_im_.resize(nl);

        std::vector<double> x_array_even, x_array_odd;
        std::vector<double> y_array;

        for (auto o : o_vec_) {
          if (o==0) {
            continue;
          }
          if (o%2==0) {
            x_array_even.push_back(std::log(1.*o));
          } else {
            x_array_odd.push_back(std::log(1.*o));
          }
        }

        for (int l=0; l < nl; ++l) {
          //Real part: l+o=even
          y_array.resize(0);
          for (int i = 0; i < o_vec_.size(); ++i) {
            if (o_vec_[i]==0) {
              continue;
            }
            if ((o_vec_[i]+l)%2==0) {
              y_array.push_back(get_value_from_cache(o_vec_[i], l).real());
            }
          }
          if (l%2==0) {
            splines_re_[l].set_points(x_array_even, y_array);
          } else {
            splines_re_[l].set_points(x_array_odd, y_array);
          }

          //Imaginary part: l+o+1=even
          y_array.resize(0);
          for (int i = 0; i < o_vec_.size(); ++i) {
            if (o_vec_[i]==0) {
              continue;
            }
            if ((o_vec_[i]+l)%2==1) {
              y_array.push_back(get_value_from_cache(o_vec_[i], l).imag());
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
          for (int i=0; i<o_vec_.size()-1; ++i) {
            //std::cout << "o_vec " << o_vec_[i] << std::endl;
            if (o_vec_[i+1] - o_vec_[i] > 2) {
              long omid = static_cast<long>(0.5*o_vec_[i+1]+0.5*o_vec_[i]);
              o_check.push_back(omid);
              o_check.push_back(omid+1);
            }
          }

          auto Tbar_ol_check = basis.compute_Tbar_ol(o_check);

          double max_diff = 0.0;
          for (int i=0; i<o_check.size(); ++i) {
            bool flag = false;
            auto o = o_check[i];
            for (int l=0; l<nl; ++l) {
              //auto diff = std::abs(Tbar_ol_check(i,l)-this->operator()(o,l));
              auto v1 = Tbar_ol_check(i,l);
              auto v2 = this->operator()(o,l);
              auto max_abs = std::max(std::abs(v1), std::abs(v2));
              //auto diff = std::abs(v1-v2)/std::abs(v1);
              auto diff = std::abs(v1-v2);
              max_diff = std::max(max_diff, diff);
              if (diff > tol) {
                //if (diff > tol) {
                flag = true;
                break;
              }
            }
            if (flag) {
              o_vec_.push_back(o);
              converged = false;
            }
          }
          //std::cout << "diff " << max_diff << std::endl;
          //unique set
          o_vec_ = to_unique_vec(o_vec_);
          std::sort(o_vec_.begin(), o_vec_.end());

          if (converged) {
            break;
          }
        }
      }
    }

    inline std::complex<double> operator()(long o, int l) const {
      if (o >= 0) {
        return get_interpolated_value_for_positive_o(o, l);
      } else {
        return std::conj(get_interpolated_value_for_positive_o(-o, l));
      }
    }

    const std::vector<long>& data_point() const {
      return o_vec_;
    }

   private:
    std::vector<long> o_vec_;
    std::vector<tk::spline> splines_re_, splines_im_;
    Eigen::Tensor<std::complex<double>,2> Tbar_ol_;

    std::map<long,std::vector<std::complex<double>>> cache_;

    void register_to_cache(const ir_basis<double>& basis, const std::vector<long>& o_vec) {
      //compute new values
      std::vector<long> o_vec_new;
      for (auto o : o_vec) {
        if (cache_.find(o) == cache_.end()) {
          o_vec_new.push_back(o);
        }
      }

      //std::cout << "register " << o_vec.size() << " " << o_vec_new.size() << std::endl;

      auto Tbar_ol_new = basis.compute_Tbar_ol(o_vec_new);
      std::vector<std::complex<double>> vec(basis.dim());
      for (int i=0; i<o_vec_new.size(); ++i) {
        for (int l=0; l < basis.dim(); ++l) {
          vec[l] = Tbar_ol_new(i,l);
        }
        cache_[o_vec_new[i]] = vec;
      }
    }

    std::complex<double> get_interpolated_value_for_positive_o(long o, int l) const {
      assert(o>=0);
      auto it = cache_.find(o);
      if (it != cache_.end()) {
        return it->second[l];
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

    std::complex<double> get_value_from_cache(long o, int l) const {
      auto it = cache_.find(o);
      if (it != cache_.end()) {
        return it->second[l];
      } else {
        throw std::runtime_error("No corresponding entry in cache");
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
