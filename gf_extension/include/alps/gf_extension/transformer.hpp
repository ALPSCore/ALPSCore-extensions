#pragma once

#include <alps/gf/gf.hpp>
#include <alps/gf_extension/piecewise_polynomial.hpp>

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

namespace alps {
namespace gf_extension {
  //AVOID USING BOOST_TYPEOF
  template<class T1, class T2>
  struct result_of_overlap {
    typedef std::complex<double> value;
  };

  template<>
  struct result_of_overlap<double, double> {
    typedef double value;
  };

  template<class T>
  void compute_integral_with_exp(
      const std::vector<double> &w,
      const std::vector<alps::gf::piecewise_polynomial<T>> &pp_func,
      Eigen::Tensor<std::complex<double>, 2> &Tnl
  );

  namespace detail {
    template<typename S1, typename S2, unsigned long N>
    void copy_from_tensor(const Eigen::Tensor<S1, N> &tensor, boost::multi_array<S2, N> &marray) {

      assert(marray.num_elements() == tensor.size());

      //From ColumnMajar to RowMajor. This also swaps dimensions.
      Eigen::Tensor<S1, N, Eigen::RowMajor> tensor_tmp = tensor.swap_layout();

      // Swap back dimensions
      std::array<int, N> shuffle;
      for (int i = 0; i < N; ++i) {
        shuffle[i] = N - 1 - i;
      }
      tensor_tmp = tensor_tmp.shuffle(shuffle);

      std::copy(tensor_tmp.data(), tensor_tmp.data() + tensor_tmp.size(), marray.origin());
    }

    template<typename S1, typename S2, std::size_t N1, int N2>
    void copy_to_tensor(const boost::multi_array<S1, N1> &marray, Eigen::Tensor<S2, N2> &tensor) {

      static_assert(N1 == N2, "Imcompatible dimensions");

      assert(marray.num_elements() == tensor.size());

      Eigen::Tensor<S2, N2, Eigen::RowMajor> tensor_tmp(tensor.dimensions());

      std::copy(marray.origin(), marray.origin() + marray.num_elements(), tensor_tmp.data());

      Eigen::Tensor<S2, N2> tensor_tmp2 = tensor_tmp.swap_layout();

      // Swap back dimensions
      std::array<int, N1> shuffle;
      for (int i = 0; i < N1; ++i) {
        shuffle[i] = N1 - 1 - i;
      }

      tensor = tensor_tmp2.shuffle(shuffle);
    }

    template<
        typename T,
        typename M1,
        typename M2,
        typename M3>
    void copy_from_tensor(
        const Eigen::Tensor<T,3>& data,
        alps::gf::three_index_gf<T,M1,M2,M3>& g) {

      assert(data.dimension(0) == g.mesh1().extent());
      assert(data.dimension(1) == g.mesh2().extent());
      assert(data.dimension(2) == g.mesh3().extent());

      for (int i1 = 0; i1 < g.mesh1().extent(); ++i1) {
        for (int i2 = 0; i2 < g.mesh2().extent(); ++i2) {
          for (int i3 = 0; i3 < g.mesh3().extent(); ++i3) {
            g(typename M1::index_type(i1),
              typename M2::index_type(i2),
              typename M3::index_type(i3)) = data(i1, i2, i3);
          }
        }
      }
    }

    template<
        typename T,
        typename M1,
        typename M2,
        typename M3>
    void copy_to_tensor(
        Eigen::Tensor<T,3>& data,
        const alps::gf::three_index_gf<T,M1,M2,M3>& g) {

      data = Eigen::Tensor<T,3>(
          g.mesh1().extent(),
          g.mesh2().extent(),
          g.mesh3().extent());

      for (int i1 = 0; i1 < g.mesh1().extent(); ++i1) {
        for (int i2 = 0; i2 < g.mesh2().extent(); ++i2) {
          for (int i3 = 0; i3 < g.mesh3().extent(); ++i3) {
            data(i1, i2, i3) =
              g(typename M1::index_type(i1),
                typename M2::index_type(i2),
                typename M3::index_type(i3));
          }
        }
      }
    }

    template<
        typename T,
        typename M1,
        typename M2,
        typename M3,
        typename M4,
        typename M5,
        typename M6,
        typename M7>
    void copy_from_tensor(
        const Eigen::Tensor<T,7>& data,
        alps::gf::seven_index_gf<T,M1,M2,M3,M4,M5,M6,M7>& g) {

      assert(data.dimension(0) == g.mesh1().extent());
      assert(data.dimension(1) == g.mesh2().extent());
      assert(data.dimension(2) == g.mesh3().extent());
      assert(data.dimension(3) == g.mesh4().extent());
      assert(data.dimension(4) == g.mesh5().extent());
      assert(data.dimension(5) == g.mesh6().extent());
      assert(data.dimension(6) == g.mesh7().extent());

      for (int i1 = 0; i1 < g.mesh1().extent(); ++i1) {
      for (int i2 = 0; i2 < g.mesh2().extent(); ++i2) {
      for (int i3 = 0; i3 < g.mesh3().extent(); ++i3) {
      for (int i4 = 0; i4 < g.mesh4().extent(); ++i4) {
      for (int i5 = 0; i5 < g.mesh5().extent(); ++i5) {
      for (int i6 = 0; i6 < g.mesh6().extent(); ++i6) {
      for (int i7 = 0; i7 < g.mesh7().extent(); ++i7) {
        g(typename M1::index_type(i1),
          typename M2::index_type(i2),
          typename M3::index_type(i3),
          typename M4::index_type(i4),
          typename M5::index_type(i5),
          typename M6::index_type(i6),
          typename M7::index_type(i7)) = data(i1, i2, i3, i4, i5, i6, i7);
      }
      }
      }
      }
      }
      }
      }
    }

    template<
        typename T,
        typename M1,
        typename M2,
        typename M3,
        typename M4,
        typename M5,
        typename M6,
        typename M7>
    void copy_to_tensor(
        Eigen::Tensor<T,7>& data,
        const alps::gf::seven_index_gf<T,M1,M2,M3,M4,M5,M6,M7>& g) {

      data = Eigen::Tensor<T,7>(
          g.mesh1().extent(),
          g.mesh2().extent(),
          g.mesh3().extent(),
          g.mesh4().extent(),
          g.mesh5().extent(),
          g.mesh6().extent(),
          g.mesh7().extent()
      );

      for (int i1 = 0; i1 < g.mesh1().extent(); ++i1) {
        for (int i2 = 0; i2 < g.mesh2().extent(); ++i2) {
          for (int i3 = 0; i3 < g.mesh3().extent(); ++i3) {
            for (int i4 = 0; i4 < g.mesh4().extent(); ++i4) {
              for (int i5 = 0; i5 < g.mesh5().extent(); ++i5) {
                for (int i6 = 0; i6 < g.mesh6().extent(); ++i6) {
                  for (int i7 = 0; i7 < g.mesh7().extent(); ++i7) {
                    data(i1, i2, i3, i4, i5, i6, i7) =
                      g(typename M1::index_type(i1),
                        typename M2::index_type(i2),
                        typename M3::index_type(i3),
                        typename M4::index_type(i4),
                        typename M5::index_type(i5),
                        typename M6::index_type(i6),
                        typename M7::index_type(i7));
                  }
                }
              }
            }
          }
        }
      }
    }

    template<typename T>
    std::vector<alps::gf::piecewise_polynomial<T>>
    extract_basis_functions(const alps::gf::numerical_mesh<T>& mesh) {
      std::vector<alps::gf::piecewise_polynomial<T>> r;
      for (int b=0; b<mesh.extent(); ++b){
        r.push_back(mesh.basis_function(b));
      }
      return r;
    }

    /// Construct piecewise polynomials representing Matsubara basis functions: exp(-i w_n tau) for n >= 0.
    /// For fermionic cases, w_n = (2*n+1)*pi/beta.
    /// For bosonci cases, w_n = (2*n)*pi/beta.
    /// Caution: when n is large, you need a very dense mesh. You are resposible for this.
    /*
    template<class T>
    void construct_matsubra_basis_functions(
        int n_min, int n_max,
        alps::gf::statistics::statistics_type s,
        const std::vector<double> &section_edges,
        int k,
        std::vector<alps::gf::piecewise_polynomial<std::complex<T> > > &results) {
      typedef alps::gf::piecewise_polynomial<std::complex<T> > pp_type;

      if (n_min < 0) {
        throw std::invalid_argument("n_min cannot be negative.");
      }
      if (n_min > n_max) {
        throw std::invalid_argument("n_min cannot be larger than n_max.");
      }

      const int N = section_edges.size() - 1;

      results.resize(0);

      std::complex<double> z;
      boost::multi_array<std::complex<T>, 2> coeffs(boost::extents[N][k + 1]);

      std::vector<double> pre_factor(k + 1);
      pre_factor[0] = 1.0;
      for (int j = 1; j < k + 1; ++j) {
        pre_factor[j] = pre_factor[j - 1] / j;
      }

      for (int n = n_min; n <= n_max; ++n) {
        if (s == alps::gf::statistics::FERMIONIC) {
          z = -std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (s == alps::gf::statistics::BOSONIC) {
          z = -std::complex<double>(0.0, n) * M_PI;
        }
        for (int section = 0; section < N; ++section) {
          const double x = section_edges[section];
          std::complex<T> exp0 = std::exp(z * (x + 1));
          std::complex<T> z_power = 1.0;
          for (int j = 0; j < k + 1; ++j) {
            coeffs[section][j] = exp0 * z_power * pre_factor[j];
            z_power *= z;
          }
        }
        results.push_back(pp_type(N, section_edges, coeffs));
      }
    }
    */

    /// Construct piecewise polynomials representing Matsubara basis functions: exp(-i w_n tau) for n >= 0.
    /// For fermionic cases, w_n = (2*n+1)*pi/beta.
    /// For bosonci cases, w_n = (2*n)*pi/beta.
    /// Caution: when n is large, you need a very dense mesh. You are resposible for this.
    template<class T>
    void construct_matsubra_basis_functions_coeff(
        int n_min, int n_max,
        alps::gf::statistics::statistics_type s,
        const std::vector<double> &section_edges,
        int k,
        boost::multi_array<std::complex<T>, 3> &coeffs) {

      if (n_min < 0) {
        throw std::invalid_argument("n_min cannot be negative.");
      }
      if (n_min > n_max) {
        throw std::invalid_argument("n_min cannot be larger than n_max.");
      }

      const int N = section_edges.size() - 1;

      std::complex<double> z;
      coeffs.resize(boost::extents[n_max - n_min + 1][N][k + 1]);

      std::vector<double> pre_factor(k + 1);
      pre_factor[0] = 1.0;
      for (int j = 1; j < k + 1; ++j) {
        pre_factor[j] = pre_factor[j - 1] / j;
      }

      for (int n = n_min; n <= n_max; ++n) {
        if (s == alps::gf::statistics::FERMIONIC) {
          z = -std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (s == alps::gf::statistics::BOSONIC) {
          z = -std::complex<double>(0.0, n) * M_PI;
        }
        for (int section = 0; section < N; ++section) {
          const double x = section_edges[section];
          std::complex<T> exp0 = std::exp(z * (x + 1));
          std::complex<T> z_power = 1.0;
          for (int j = 0; j < k + 1; ++j) {
            coeffs[n - n_min][section][j] = exp0 * z_power * pre_factor[j];
            z_power *= z;
          }
        }
      }
    }

    /// Construct piecewise polynomials representing exponential functions: exp(i w_i x)
    template<class T>
    void construct_exp_functions_coeff(
        const std::vector<double> &w,
        const std::vector<double> &section_edges,
        int k,
        boost::multi_array<std::complex<T>, 3> &coeffs) {
      const int N = section_edges.size() - 1;

      std::complex<double> z;
      coeffs.resize(boost::extents[w.size()][N][k + 1]);

      std::vector<double> pre_factor(k + 1);
      pre_factor[0] = 1.0;
      for (int j = 1; j < k + 1; ++j) {
        pre_factor[j] = pre_factor[j - 1] / j;
      }

      for (int n = 0; n < w.size(); ++n) {
        auto z = std::complex<double>(0.0, w[n]);
        for (int section = 0; section < N; ++section) {
          const double x = section_edges[section];
          std::complex<T> exp0 = std::exp(z * (x + 1));
          std::complex<T> z_power = 1.0;
          for (int j = 0; j < k + 1; ++j) {
            coeffs[n][section][j] = exp0 * z_power * pre_factor[j];
            z_power *= z;
          }
        }
      }
    }

/**
 * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
 * @tparam T  scalar type
 * @param n_min min index of Matsubara freq. index (>=0)
 * @param n_max max index of Matsubara freq. index (>=0)
 * @param statis Statistics (fermion or boson)
 * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
    template<class T>
    void compute_transformation_matrix_to_matsubara_impl(
        const std::vector<long> &n_vec,
        alps::gf::statistics::statistics_type statis,
        const std::vector<alps::gf::piecewise_polynomial<T>> &bf_src,
        Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) {
      typedef std::complex<double> dcomplex;
      typedef alps::gf::piecewise_polynomial<std::complex<double> > pp_type;
      typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
      typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

      //if (n_min < 0) {
      //throw std::invalid_argument("n_min cannot be negative.");
      //}
      //if (n_min > n_max) {
      //throw std::invalid_argument("n_min cannot be larger than n_max.");
      //}
      for (int n = 0; n < n_vec.size(); ++n) {
        if (n_vec[n] < 0) {
          throw std::runtime_error("n_vec cannot be negative.");
        }
      }
      for (int n = 0; n < n_vec.size() - 1; ++n) {
        if (n_vec[n] > n_vec[n + 1]) {
          throw std::runtime_error("n_vec must be in ascending order.");
        }
      }

      std::vector<double> w(n_vec.size());

      for (int n = 0; n < n_vec.size(); ++n) {
        if (statis == alps::gf::statistics::FERMIONIC) {
          w[n] = (n_vec[n] + 0.5) * M_PI;
        } else if (statis == alps::gf::statistics::BOSONIC) {
          w[n] = n_vec[n] * M_PI;
        }
      }

      compute_integral_with_exp(w, bf_src, Tnl);

      std::vector<double> inv_norm(bf_src.size());
      for (int l = 0; l < bf_src.size(); ++l) {
        inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
      }
      for (int n = 0; n < w.size(); ++n) {
        for (int l = 0; l < bf_src.size(); ++l) {
          Tnl(n, l) *= inv_norm[l] * std::sqrt(0.5);
        }
      }
    }

  }//namespace detail

  template<typename T>
  std::vector<T> linspace(T minval, T maxval, int N) {
    std::vector<T> r(N);
    for (int i = 0; i < N; ++i) {
      r[i] = i * (maxval - minval) / (N - 1) + minval;
    }
    return r;
  }

/**
 * Compute integral of exponential functions and given piecewise polynomials
 *           \int_{-1}^1 dx exp(i w_i (x+1)) p_j(x),
 *           where w_i are given real double objects and p_j are piecewise polynomials.
 *           The p_j(x) must be defined in the interval [-1,1].
 * @tparam T  scalar type of piecewise polynomials
 * @param w vector of w_i in ascending order
 * @param statis Statistics (fermion or boson)
 * @param p vector of piecewise polynomials
 * @param results  computed results
 */
  template<class T>
  void compute_integral_with_exp(
      const std::vector<double> &w,
      const std::vector<alps::gf::piecewise_polynomial<T>> &pp_func,
      Eigen::Tensor<std::complex<double>, 2> &Tnl
  ) {
    typedef std::complex<double> dcomplex;
    typedef alps::gf::piecewise_polynomial<std::complex<double> > pp_type;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

    //order of polynomials used for representing exponential functions internally.
    const int k_iw = 8;
    const int k = pp_func[0].order();
    const int n_section = pp_func[0].num_sections();

    for (int l = 0; l < pp_func.size(); ++l) {
      if (k != pp_func[l].order()) {
        throw std::runtime_error(
            "Error in compute_transformation_matrix_to_matsubara: basis functions must be pieacewise polynomials of the same order");
      }
      if (pp_func[l].section_edge(0) != -1 || pp_func[l].section_edge(n_section) != 1) {
        throw std::runtime_error("Piecewise polynomials must be defined on [-1,1]");
      }
    }

    const int n_iw = w.size();
    const int n_max = n_iw - 1;

    for (int i = 0; i < w.size() - 1; ++i) {
      if (w[i] > w[i + 1]) {
        throw std::runtime_error("w must be give in ascending order.");
      }
    }

    //Use Taylor expansion for exp(i w_n tau) for w_n*dx < cutoff*M_PI
    const double cutoff = 0.1;

    boost::multi_array<std::complex<double>, 3> exp_coeffs(boost::extents[w.size()][n_section][k_iw + 1]);
    detail::construct_exp_functions_coeff(w, pp_func[0].section_edges(), k_iw, exp_coeffs);

    matrix_t left_mid_matrix(n_iw, k + 1);
    matrix_t left_matrix(n_iw, k_iw + 1);
    matrix_t mid_matrix(k_iw + 1, k + 1);
    matrix_t right_matrix(k + 1, pp_func.size());
    matrix_t r(n_iw, pp_func.size());
    r.setZero();

    std::vector<double> dx_power(k + k_iw + 2);

    for (int s = 0; s < n_section; ++s) {
      double x0 = pp_func[0].section_edge(s);
      double x1 = pp_func[0].section_edge(s + 1);
      double dx = x1 - x0;
      left_mid_matrix.setZero();

      dx_power[0] = 1.0;
      for (int p = 1; p < dx_power.size(); ++p) {
        dx_power[p] = dx * dx_power[p - 1];
      }

      //Use Taylor expansion for exp(i w_n tau) for w_n*dx < cutoff*M_PI
      const double w_max_cs = cutoff * M_PI / dx;
      int n_max_cs = -1;
      for (int i = 0; i < w.size(); ++i) {
        if (w[i] <= w_max_cs) {
          n_max_cs = i;
        }
      }

      //Use Taylor expansion
      if (n_max_cs >= 0) {
        for (int p = 0; p < k_iw + 1; ++p) {
          for (int p2 = 0; p2 < k + 1; ++p2) {
            mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
          }
        }

        for (int n = 0; n < n_max_cs + 1; ++n) {
          for (int p = 0; p < k_iw + 1; ++p) {
            left_matrix(n, p) = exp_coeffs[n][s][p];
          }
        }

        left_mid_matrix.block(0, 0, n_max_cs + 1, k + 1) =
            left_matrix.block(0, 0, n_max_cs + 1, k_iw + 1) * mid_matrix;
      }

      //Otherwise, compute the overlap exactly
      for (int n = std::max(n_max_cs + 1, 0); n <= n_max; ++n) {
        std::complex<double> z = std::complex<double>(0.0, w[n]);

        dcomplex dx_z = dx * z;
        dcomplex dx_z2 = dx_z * dx_z;
        dcomplex dx_z3 = dx_z2 * dx_z;
        dcomplex inv_z = 1.0 / z;
        dcomplex inv_z2 = inv_z * inv_z;
        dcomplex inv_z3 = inv_z2 * inv_z;
        dcomplex inv_z4 = inv_z3 * inv_z;
        dcomplex exp = std::exp(dx * z);
        dcomplex exp0 = std::exp((x0 + 1.0) * z);

        left_mid_matrix(n, 0) = (-1.0 + exp) * inv_z * exp0;
        left_mid_matrix(n, 1) = ((dx_z - 1.0) * exp + 1.0) * inv_z2 * exp0;
        left_mid_matrix(n, 2) = ((dx_z2 - 2.0 * dx_z + 2.0) * exp - 2.0) * inv_z3 * exp0;
        left_mid_matrix(n, 3) = ((dx_z3 - 3.0 * dx_z2 + 6.0 * dx_z - 6.0) * exp + 6.0) * inv_z4 * exp0;
      }

      for (int l = 0; l < pp_func.size(); ++l) {
        for (int p2 = 0; p2 < k + 1; ++p2) {
          right_matrix(p2, l) = pp_func[l].coefficient(s, p2);
        }
      }

      r += left_mid_matrix * right_matrix;
    }

    Tnl = tensor_t(n_iw, pp_func.size());
    for (int n = 0; n < n_iw; ++n) {
      for (int l = 0; l < pp_func.size(); ++l) {
        Tnl(n, l) = r(n, l);
      }
    }
  }


/**
 * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
 * @tparam T  scalar type
 * @param n_min min index of Matsubara freq. index (>=0)
 * @param n_max max index of Matsubara freq. index (>=0)
 * @param statis Statistics (fermion or boson)
 * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
  template<class T>
  void compute_transformation_matrix_to_matsubara(
      int n_min, int n_max,
      alps::gf::statistics::statistics_type statis,
      const std::vector<alps::gf::piecewise_polynomial<T>> &bf_src,
      Eigen::Tensor<std::complex<double>, 2> &Tnl
  ) {
    const int num_n = n_max - n_min + 1;
    const int batch_size = 500;
    Tnl = Eigen::Tensor<std::complex<double>, 2>(num_n, bf_src.size());
    Eigen::Tensor<std::complex<double>, 2> Tnl_batch(batch_size, bf_src.size());
    //TODO: use MPI
    //Split into batches to avoid using too much memory
    for (int ib = 0; ib < num_n / batch_size + 1; ++ib) {
      int n_min_batch = batch_size * ib;
      int n_max_batch = std::min(batch_size * (ib + 1) - 1, n_max);
      if (n_max_batch - n_min_batch < 0) {
        continue;
      }
      std::vector<long> n_vec;
      for (int n = n_min_batch; n <= n_max_batch; ++n) {
        n_vec.push_back(n);
      }
      detail::compute_transformation_matrix_to_matsubara_impl(n_vec, statis, bf_src, Tnl_batch);
      for (int j = 0; j < bf_src.size(); ++j) {
        for (int n = n_min_batch; n <= n_max_batch; ++n) {
          Tnl(n - n_min, j) = Tnl_batch(n - n_min_batch, j);
        }
      }
    }
  }

  /**
 * Compute a transformation matrix from a give orthogonal basis set to Matsubara freq.
 * @tparam T  scalar type
 * @param n indices of Matsubara frequqneices for which matrix elements will be computed (in strictly ascending order).
 *          The Matsubara basis functions look like exp(i PI * (n[i]+1/2)) for fermions, exp(i PI * n[i]) for bosons.
 * @param bf_src orthogonal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
  template<class T>
  void compute_transformation_matrix_to_matsubara(
      const std::vector<long> &n,
      alps::gf::statistics::statistics_type statis,
      const std::vector<alps::gf::piecewise_polynomial<T>> &bf_src,
      Eigen::Tensor<std::complex<double>, 2> &Tnl
  ) {
    typedef std::complex<double> dcomplex;
    typedef alps::gf::piecewise_polynomial<std::complex<double> > pp_type;
    typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    typedef Eigen::Tensor<std::complex<double>, 2> tensor_t;

    if (n.size() == 0) {
      return;
    }

    for (int i = 0; i < n.size() - 1; ++i) {
      if (n[i] > n[i + 1]) {
        throw std::runtime_error("n must be in strictly ascending order!");
      }
    }

    std::vector<double> w;
    if (statis == alps::gf::statistics::FERMIONIC) {
      std::transform(n.begin(), n.end(), std::back_inserter(w), [](double x) { return M_PI * (x + 0.5); });
    } else {
      std::transform(n.begin(), n.end(), std::back_inserter(w), [](double x) { return M_PI * x; });
    }

    compute_integral_with_exp(w, bf_src, Tnl);

    std::vector<double> inv_norm(bf_src.size());
    for (int l = 0; l < bf_src.size(); ++l) {
      inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
    }
    for (int n = 0; n < w.size(); ++n) {
      for (int l = 0; l < bf_src.size(); ++l) {
        Tnl(n, l) *= inv_norm[l] * std::sqrt(0.5);
      }
    }
  }


  /// Compute overlap <left | right> with complex conjugate
  template<class T1, class T2>
  void compute_overlap(
      const std::vector<alps::gf::piecewise_polynomial<T1> > &left_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2> > &right_vectors,
      boost::multi_array<typename result_of_overlap<T1, T2>::value, 2> &results) {
    typedef typename result_of_overlap<T1, T2>::value Tr;

    const int NL = left_vectors.size();
    const int NR = right_vectors.size();
    const int n_sections = left_vectors[0].num_sections();

    const int k1 = left_vectors[0].order();
    const int k2 = right_vectors[0].order();

    if (left_vectors[0].section_edges() != right_vectors[0].section_edges()) {
      throw std::runtime_error("Not supported");
    }

    for (int n = 0; n < NL - 1; ++n) {
      if (left_vectors[n].section_edges() != left_vectors[n + 1].section_edges()) {
        throw std::runtime_error("Not supported");
      }
    }

    for (int n = 0; n < NL; ++n) {
      if (k1 != left_vectors[n].order()) {
        throw std::runtime_error("Left vectors must be piecewise polynomials of the same order.");
      }
    }

    for (int n = 0; n < NR; ++n) {
      if (k2 != right_vectors[n].order()) {
        throw std::runtime_error("Right vectors must be piecewise polynomials of the same order.");
      }
    }

    for (int l = 0; l < NR - 1; ++l) {
      if (right_vectors[l].section_edges() != right_vectors[l + 1].section_edges()) {
        throw std::runtime_error("Not supported");
      }
    }

    std::vector<double> x_min_power(k1 + k2 + 2), dx_power(k1 + k2 + 2);

    Eigen::Matrix<Tr, Eigen::Dynamic, Eigen::Dynamic> mid_matrix(k1 + 1, k2 + 1);
    Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> left_matrix(NL, k1 + 1);
    Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> right_matrix(k2 + 1, NR);
    Eigen::Matrix<Tr, Eigen::Dynamic, Eigen::Dynamic> r(NL, NR);

    r.setZero();
    for (int s = 0; s < n_sections; ++s) {
      dx_power[0] = 1.0;
      const double dx = left_vectors[0].section_edge(s + 1) - left_vectors[0].section_edge(s);
      for (int p = 1; p < dx_power.size(); ++p) {
        dx_power[p] = dx * dx_power[p - 1];
      }

      for (int p = 0; p < k1 + 1; ++p) {
        for (int p2 = 0; p2 < k2 + 1; ++p2) {
          mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
        }
      }

      for (int n = 0; n < NL; ++n) {
        for (int p = 0; p < k1 + 1; ++p) {
          //FIXME: DO NOT USE alps::gf::detail::conjg
          left_matrix(n, p) = alps::gf::detail::conjg(left_vectors[n].coefficient(s, p));
        }
      }

      for (int l = 0; l < NR; ++l) {
        for (int p2 = 0; p2 < k2 + 1; ++p2) {
          right_matrix(p2, l) = right_vectors[l].coefficient(s, p2);
        }
      }

      r += left_matrix * (mid_matrix * right_matrix);
    }

    results.resize(boost::extents[NL][NR]);
    for (int n = 0; n < NL; ++n) {
      for (int l = 0; l < NR; ++l) {
        results[n][l] = r(n, l);
      }
    }
  }


  /// Compute a transformation matrix from a src orthogonal basis set to a dst orthogonal basis set.
  /// The basis vectors are NOT necessarily normalized to 1.
  template<class T1, class T2>
  void compute_transformation_matrix(
      const std::vector<alps::gf::piecewise_polynomial<T1> > &dst_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2> > &src_vectors,
      boost::multi_array<typename result_of_overlap<T1, T2>::value, 2> &results) {
    compute_overlap(dst_vectors, src_vectors, results);

    std::vector<double> coeff1(dst_vectors.size());
    for (int l = 0; l < dst_vectors.size(); ++l) {
      coeff1[l] = 1.0 / std::sqrt(
          static_cast<double>(
              dst_vectors[l].overlap(dst_vectors[l])
          )
      );
    }

    std::vector<double> coeff2(src_vectors.size());
    for (int l = 0; l < src_vectors.size(); ++l) {
      coeff2[l] = 1.0 / std::sqrt(
          static_cast<double>(
              src_vectors[l].overlap(src_vectors[l])
          )
      );
    }

    for (int l1 = 0; l1 < dst_vectors.size(); ++l1) {
      for (int l2 = 0; l2 < src_vectors.size(); ++l2) {
        results[l1][l2] *= coeff1[l1] * coeff2[l2];
      }
    }
  };


  /***
   * Template class transformer. Convert a GF object to a GF object of a different type.
   * Some conversion may not be safe.
   * @tparam G_DST Destination type
   * @tparam G_SRC Source type
   */
  template<typename G_DST, typename G_SRC>
  class transformer {};

  template<typename S, typename M2, typename M3>
  using itime_three_index_gf = alps::gf::three_index_gf<S, alps::gf::itime_mesh, M2, M3>;

  template<typename S, typename M2, typename M3>
  using omega_three_index_gf = alps::gf::three_index_gf<S, alps::gf::matsubara_positive_mesh, M2, M3>;

  template<typename S, typename M2, typename M3>
  using nmesh_three_index_gf = alps::gf::three_index_gf<S, alps::gf::numerical_mesh<double>, M2, M3>;

  /// Numerical mesh to imaginary time (three-index gf)
  template<typename S, typename M2, typename M3>
  class transformer<itime_three_index_gf<S, M2, M3>, nmesh_three_index_gf<S, M2, M3> > {
    using gt_dst = itime_three_index_gf<S, M2, M3>;
    using gt_src = nmesh_three_index_gf<S, M2, M3>;

    using nmesh_type = alps::gf::numerical_mesh<double>;
    using index_type2 = typename M2::index_type;
    using index_type3 = typename M3::index_type;

   public:
    transformer(int ntau, const alps::gf::numerical_mesh<double> &nmesh) : ntau_(ntau), nmesh_(nmesh) {};

    gt_dst operator()(const gt_src &g_in) const {
      if (nmesh_ != g_in.mesh1()) {
        throw std::runtime_error("Given Green's function object has an incompatible numerical mesh.");
      }
      double beta = g_in.mesh1().beta();
      gt_dst g_out(alps::gf::itime_mesh(g_in.mesh1().beta(), ntau_), g_in.mesh2(), g_in.mesh3());

      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      std::vector<double> coeff(dim_in);
      for (int il = 0; il < dim_in; ++il) {
        coeff[il] = sqrt(2.0 / g_in.mesh1().basis_function(il).squared_norm()) / beta;
      }

      std::vector<double> vals(dim_in);

      for (int itau = 0; itau < ntau_; ++itau) {
        double tau = itau * (beta / (ntau_ - 1));
        double x = 2 * tau / beta - 1.0;

        for (int il = 0; il < dim_in; ++il) {
          vals[il] = g_in.mesh1().basis_function(il).compute_value(x);
        }

        for (int index2 = 0; index2 < dim2; ++index2) {
          for (int index3 = 0; index3 < dim3; ++index3) {
            g_out(alps::gf::itime_index(itau), index_type2(index2), index_type3(index3)) = 0.0;
            for (int il = 0; il < dim_in; ++il) {
              g_out(alps::gf::itime_index(itau), index_type2(index2), index_type3(index3)) +=
                  vals[il] * coeff[il] *
                      g_in(
                          alps::gf::numerical_mesh<double>::index_type(il),
                          index_type2(index2),
                          index_type3(index3)
                      );
            }
          }
        }
      }

      return g_out;
    }

   private:
    int ntau_;
    nmesh_type nmesh_;
  };

  /// Numerical mesh to Matsubara mesh (three-index gf)
  template<typename S, typename M2, typename M3>
  class transformer<omega_three_index_gf<std::complex<double>, M2, M3>, nmesh_three_index_gf<S, M2, M3> > {
    using gt_dst = omega_three_index_gf<std::complex<double>, M2, M3>;
    using gt_src = nmesh_three_index_gf<S, M2, M3>;

    using nmesh_type = alps::gf::numerical_mesh<double>;
    using index_type2 = typename M2::index_type;
    using index_type3 = typename M3::index_type;
    using pp_type = alps::gf::piecewise_polynomial<double>;
    static constexpr int num_index = 3;

   public:
    transformer(int niw, const alps::gf::numerical_mesh<double> &nmesh) : niw_(niw), nmesh_(nmesh), Tnl_(0, 0) {

      const int nl = nmesh_.extent();

      Tnl_ = Eigen::Tensor<std::complex<double>, 2>(niw_, nl);
      double beta = nmesh_.beta();

      std::vector<pp_type> basis_functions(nl);
      for (int l = 0; l < nl; ++l) {
        basis_functions[l] = nmesh_.basis_function(l);
      }

      compute_transformation_matrix_to_matsubara(
          0, niw_ - 1,
          nmesh_.statistics(),
          basis_functions,
          Tnl_
      );
    }

    gt_dst operator()(const gt_src &g_in) const {
      if (nmesh_ != g_in.mesh1()) {
        throw std::runtime_error("Given Green's function object has an incompatible numerical mesh.");
      }

      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      Eigen::Tensor<std::complex<double>, num_index> data_l(dim_in, dim2, dim3);
      detail::copy_to_tensor(g_in.data(), data_l);

      std::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
      Eigen::Tensor<std::complex<double>, num_index> data_omega = Tnl_.contract(data_l, product_dims);

      gt_dst g_out(alps::gf::matsubara_positive_mesh(g_in.mesh1().beta(), niw_), g_in.mesh2(), g_in.mesh3());
      for (int i1 = 0; i1 < niw_; ++i1) {
        for (int i2 = 0; i2 < dim2; ++i2) {
          for (int i3 = 0; i3 < dim3; ++i3) {
            g_out(
                alps::gf::matsubara_positive_mesh::index_type(i1),
                typename M2::index_type(i2),
                typename M3::index_type(i3)
            ) = data_omega(i1, i2, i3);
          }
        }
      }

      return g_out;
    }

   private:
    int niw_;
    nmesh_type nmesh_;
    Eigen::Tensor<std::complex<double>, 2> Tnl_;
  };

  template<typename T>
  Eigen::Tensor<std::complex<T>, 3>
  compute_w_tensor(
      const std::vector<long> &n_vec,
      const std::vector<alps::gf::piecewise_polynomial<T>> &basis_f,
      const std::vector<alps::gf::piecewise_polynomial<T>> &basis_b) {
    using dcomplex = std::complex<double>;

    const int dim_f = basis_f.size();
    const int dim_b = basis_b.size();

    std::vector<double> w(n_vec.size());
    for (int n = 0; n < n_vec.size(); ++n) {
      w[n] = M_PI * (n_vec[n] + 0.5);
    }

    std::vector<alps::gf::piecewise_polynomial<double>> prods(dim_f * dim_b);
    for (int lp = 0; lp < dim_f; ++lp) {
      for (int l = 0; l < dim_b; ++l) {
        prods[l + lp * dim_b] = alps::gf_extension::multiply(basis_b[l], basis_f[lp]);
      }
    }

    Eigen::Tensor<dcomplex, 2> integral(n_vec.size(), prods.size());
    const int b_size = 500;
    for (int b = 0; b < n_vec.size() / b_size + 1; ++b) {
      auto n_start = b * b_size;
      auto n_last = std::min((b + 1) * b_size - 1, (int) n_vec.size() - 1);
      if (n_start > n_last) {
        continue;
      }

      std::vector<double> w_batch;
      for (int n = n_start; n <= n_last; ++n) {
        w_batch.push_back(w[n]);
      }
      Eigen::Tensor<dcomplex, 2> sub;
      alps::gf_extension::compute_integral_with_exp(w_batch, prods, sub);
      for (int n = n_start; n <= n_last; ++n) {
        for (int j = 0; j < prods.size(); ++j) {
          integral(n, j) = sub(n - n_start, j);
        }
      }
    }

    auto w_tensor = Eigen::Tensor<dcomplex, 3>(n_vec.size(), dim_b, dim_f);
    for (int lp = 0; lp < dim_f; ++lp) {
      for (int l = 0; l < dim_b; ++l) {
        for (int n = 0; n < n_vec.size(); ++n) {
          w_tensor(n, l, lp) = integral(n, l + lp * dim_b);
        }
      }
    }
    return w_tensor;
  }

  inline
  void construct_log_mesh(long max_n,
                            int max_n_exact_sum,
                            double ratio_sum,
                            std::vector<long>& n_vec,
                            std::vector<double>& weight) {
    n_vec.resize(0);
    weight.resize(0);

    long n_start = 0, dn = 1;
    while (n_start < max_n) {
      long n_mid = (long) std::round(0.5 * (n_start + n_start + dn - 1));
      n_vec.push_back(n_mid);
      weight.push_back(1. * dn);

      n_start += dn;
      if (n_start < max_n_exact_sum) {
        dn = 1;
      } else {
        dn = std::max(long(dn * ratio_sum), dn + 1);
      }
    }
  }

  template<typename T>
  Eigen::Tensor<T, 6>
  compute_C_tensor(
      const std::vector<alps::gf::piecewise_polynomial<T>> &basis_f,
      const std::vector<alps::gf::piecewise_polynomial<T>> &basis_b,
      double ratio_sum = 1.02,
      int max_n_exact_sum = 200
  ) {
    using dcomplex = std::complex<double>;
    namespace ge = alps::gf_extension;

    const int dim_f = basis_f.size();
    const int dim_b = basis_b.size();

    //Construct a mesh
    long max_n = 1E+10;
    std::vector<long> n_vec;
    std::vector<double> weight_sum;
    construct_log_mesh(max_n, max_n_exact_sum, ratio_sum, n_vec, weight_sum);
    int n_mesh = n_vec.size();

    //Compute w tensor
    auto w_tensor = compute_w_tensor(n_vec, basis_f, basis_b);

    //Compute Tnl_f
    Eigen::Tensor<dcomplex, 2> Tnl_f;
    compute_transformation_matrix_to_matsubara(n_vec, alps::gf::statistics::FERMIONIC, basis_f, Tnl_f);

    Eigen::Tensor<dcomplex, 2> Tnl_b;
    compute_transformation_matrix_to_matsubara(n_vec, alps::gf::statistics::BOSONIC, basis_b, Tnl_b);

    std::vector<double> sqrt_weight_sum(weight_sum);
    for (int n = 0; n < n_mesh; ++n) {
      sqrt_weight_sum[n] = std::sqrt(weight_sum[n]);
    }

    Eigen::Tensor<dcomplex, 4> left_mat(dim_f, dim_f, dim_b, n_mesh);//(l1;l2;lp3, n)
    for (int n = 0; n < n_mesh; ++n) {
      for (int lp3 = 0; lp3 < dim_b; ++lp3) {
        for (int l2 = 0; l2 < dim_f; ++l2) {
          auto sign = l2%2==0 ? -1.0 : 1.0;
          for (int l1 = 0; l1 < dim_f; ++l1) {
            left_mat(l1, l2, lp3, n) = sign * std::conj(w_tensor(n, lp3, l1) * Tnl_f(n, l2)) * sqrt_weight_sum[n];
          }
        }
      }
    }

    auto right_mat {left_mat.conjugate()};

    //left_mat(l1, l2, lp3, n)
    //right_mat(lp1, lp2, l3, n)
    //contract => (l1, l2, lp3, lp1, lp2, l3)
    //shuffle => (l1, l2, l3, lp1, lp2, lp3)
    std::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(3, 3)};
    return (-2 * left_mat.contract(right_mat, product_dims).real()).shuffle(
        std::array<int, 6>{{0, 1, 5, 3, 4, 2}}
    );

  }

  inline
  Eigen::Tensor<std::complex<double>, 2>
  compute_Tnl(const std::vector<long>& n_vec,
              const alps::gf::numerical_mesh<double>& nmesh) {
    Eigen::Tensor<std::complex<double>, 2> Tnl;

    int nl = nmesh.extent();

    std::vector<alps::gf::piecewise_polynomial<double>> basis_functions(nl);
    for (int l = 0; l < nl; ++l) {
      basis_functions[l] = nmesh.basis_function(l);
    }

    compute_transformation_matrix_to_matsubara( n_vec, nmesh.statistics(), basis_functions, Tnl);

    return Tnl;
  }

  inline
  Eigen::Tensor<std::complex<double>, 2>
  compute_Tnl(int n_min, int n_max, const alps::gf::numerical_mesh<double>& nmesh) {
    Eigen::Tensor<std::complex<double>, 2> Tnl;

    //double beta = nmesh.beta();
    int nl = nmesh.extent();

    std::vector<alps::gf::piecewise_polynomial<double>> basis_functions(nl);
    for (int l = 0; l < nl; ++l) {
      basis_functions[l] = nmesh.basis_function(l);
    }

    compute_transformation_matrix_to_matsubara(n_min, n_max, nmesh.statistics(), basis_functions, Tnl);

    return Tnl;
  }

  /**
   * Compute a G2 bubule of Hartree type
   * @tparam T       Scalar type of Gl
   * @param Gl       G1
   * @param mesh_f   Fermionic mesh of G2
   * @param mesh_b   Bosonic mesh of G2
   * @return         G2 bubble of Hartree type
   * All numerical_mesh must be ir basis sets with the same value of Lambda
   */
  template<typename T>
  alps::gf::seven_index_gf<
      std::complex<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::index_mesh,
      alps::gf::index_mesh,
      alps::gf::index_mesh,
      alps::gf::index_mesh>
  compute_G2_bubble_H(
      const alps::gf::three_index_gf<T,alps::gf::numerical_mesh<double>,alps::gf::index_mesh,alps::gf::index_mesh>& Gl,
      const alps::gf::numerical_mesh<double>& mesh_f,
      const alps::gf::numerical_mesh<double>& mesh_b
  ) {
    using G2_t = alps::gf::seven_index_gf<
        std::complex<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::index_mesh,
        alps::gf::index_mesh,
        alps::gf::index_mesh,
        alps::gf::index_mesh>;

    double beta = Gl.mesh1().beta();

    if (mesh_f.beta() != beta || mesh_b.beta() != beta) {
      throw std::invalid_argument("All mesh must have the same value of beta.");
    }
    if (Gl.mesh2() != Gl.mesh3()) {
      throw std::invalid_argument("mesh2 and mesh3 of Gl must be identical.");
    }
    if (Gl.mesh1().statistics() != alps::gf::statistics::FERMIONIC) {
      throw std::invalid_argument("mesh1 of Gl must be fermionic.");
    }
    if (mesh_f.statistics() != alps::gf::statistics::FERMIONIC) {
      throw std::invalid_argument("mesh_f must be fermionic.");
    }
    if (mesh_b.statistics() != alps::gf::statistics::BOSONIC) {
      throw std::invalid_argument("mesh_b must be bosonic.");
    }

    G2_t g2(mesh_f, mesh_f, mesh_b, Gl.mesh2(), Gl.mesh2(), Gl.mesh2(), Gl.mesh2());

    int nf = Gl.mesh2().extent();
    int nl_f_G2 = mesh_f.extent();
    int nl_b_G2 = mesh_b.extent();
    int nl_G1 = Gl.mesh1().extent();

    for (int l=0; l<std::min(nl_G1,nl_f_G2); ++l) {
      if (!(mesh_f.basis_function(l) == Gl.mesh1().basis_function(l))) {
        throw std::invalid_argument("mesh_f and Gl are not consistent.");
      }
    }

    if (nl_G1 < nl_f_G2) {
      throw std::invalid_argument("Too few fermionic basis functions for G1.");
    }

    //compute T_nl^B for n=0.
    auto Tnl_b = compute_Tnl(0, 0, mesh_b);

    using nindex_t = alps::gf::numerical_mesh<double>::index_type;
    using iindex_t = alps::gf::index_mesh::index_type;

    for (int f4 = 0; f4 < nf; ++f4) {
      for (int f3 = 0; f3 < nf; ++f3) {
        for (int f2 = 0; f2 < nf; ++f2) {
          for (int f1 = 0; f1 < nf; ++f1) {
            for (int l3=0; l3 < nl_b_G2; ++l3) {
              for (int l2=0; l2 < nl_f_G2; ++l2) {
                auto sign = l2%2 == 0 ? -1.0 : 1.0;
                for (int l1=0; l1 < nl_f_G2; ++l1) {
                  g2(nindex_t(l1),
                     nindex_t(l2),
                     nindex_t(l3),
                     iindex_t(f1),
                     iindex_t(f2),
                     iindex_t(f3),
                     iindex_t(f4)) = beta * sign * std::conj(Tnl_b(0,l3))
                      * Gl(nindex_t(l1), iindex_t(f1), iindex_t(f2))
                      * Gl(nindex_t(l2), iindex_t(f3), iindex_t(f4));
                  //std::cout << l1 << " " << l2 << " " << l3 << " " << std::abs(g2(nindex_t(l1), nindex_t(l2), nindex_t(l3), iindex_t(f1), iindex_t(f2), iindex_t(f3), iindex_t(f4))) << std::endl;
                  //std::cout << l1 << " " << l2 << " " << l3 << " " << std::conj(Tnl_b(0,l3)) << std::endl;
                }
              }
            }
          }
        }
      }
    }

    return g2;
  }

  /**
   * Compute a G2 bubule of Fock type
   * @tparam T       Scalar type of Gl
   * @param Gl       G1
   * @param mesh_f   Fermionic mesh of G2
   * @param mesh_b   Bosonic mesh of G2
   * @return         G2 bubble of Hartree type
   * All numerical_mesh must be ir basis sets with the same value of Lambda
   */
  template<typename T>
  alps::gf::seven_index_gf<
      std::complex<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::numerical_mesh<double>,
      alps::gf::index_mesh,
      alps::gf::index_mesh,
      alps::gf::index_mesh,
      alps::gf::index_mesh>
  compute_G2_bubble_F(
      const alps::gf::three_index_gf<T,alps::gf::numerical_mesh<double>,alps::gf::index_mesh,alps::gf::index_mesh>& Gl,
      const alps::gf::numerical_mesh<double>& mesh_f,
      const alps::gf::numerical_mesh<double>& mesh_b
  ) {
    using G2_t = alps::gf::seven_index_gf<
        std::complex<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::numerical_mesh<double>,
        alps::gf::index_mesh,
        alps::gf::index_mesh,
        alps::gf::index_mesh,
        alps::gf::index_mesh>;

    double beta = Gl.mesh1().beta();

    if (mesh_f.beta() != beta || mesh_b.beta() != beta) {
      throw std::invalid_argument("All mesh must have the same value of beta.");
    }
    if (Gl.mesh2() != Gl.mesh3()) {
      throw std::invalid_argument("mesh2 and mesh3 of Gl must be identical.");
    }
    if (Gl.mesh1().statistics() != alps::gf::statistics::FERMIONIC) {
      throw std::invalid_argument("mesh1 of Gl must be fermionic.");
    }
    if (mesh_f.statistics() != alps::gf::statistics::FERMIONIC) {
      throw std::invalid_argument("mesh_f must be fermionic.");
    }
    if (mesh_b.statistics() != alps::gf::statistics::BOSONIC) {
      throw std::invalid_argument("mesh_b must be bosonic.");
    }

    G2_t g2(mesh_f, mesh_f, mesh_b, Gl.mesh2(), Gl.mesh2(), Gl.mesh2(), Gl.mesh2());

    int nf = Gl.mesh2().extent();
    int nl_f_G2 = mesh_f.extent();
    int nl_b_G2 = mesh_b.extent();
    int nl_G1 = Gl.mesh1().extent();

    for (int l=0; l<std::min(nl_G1,nl_f_G2); ++l) {
      if (!(mesh_f.basis_function(l) == Gl.mesh1().basis_function(l))) {
        throw std::invalid_argument("mesh_f and Gl are not consistent.");
      }
    }

    if (nl_G1 < nl_f_G2) {
      throw std::invalid_argument("Too few fermionic basis functions for G1.");
    }

    //Construct a mesh
    long max_n = 1E+10;
    int max_n_exact_sum = 200;
    double ratio_sum = 1.02;
    std::vector<long> n_vec;
    std::vector<double> weight_sum;
    construct_log_mesh(max_n, max_n_exact_sum, ratio_sum, n_vec, weight_sum);
    int n_mesh = n_vec.size();

    //Compute w tensor
    auto basis_f = detail::extract_basis_functions(mesh_f);
    auto basis_b = detail::extract_basis_functions(mesh_b);
    auto w_tensor = compute_w_tensor(n_vec, basis_f, basis_b);

    //Compute Tnl
    auto Tnl_f = compute_Tnl(n_vec, g2.mesh1());

    Eigen::Tensor<std::complex<double>,3> left_tensor(nl_f_G2, nl_f_G2, n_mesh);
    for (int n=0; n<n_mesh; ++n) {
      for (int l2=0; l2<nl_f_G2; ++l2) {
        auto sign = l2%2==0 ? 1.0 : -1.0;
        for (int l1=0; l1<nl_f_G2; ++l1) {
          left_tensor(l1, l2, n) = sign * std::conj(Tnl_f(n, l1) * Tnl_f(n, l2)) * weight_sum[n] * beta;
        }
      }
    }

    Eigen::Tensor<std::complex<double>,4> right_tensor(n_mesh, nl_b_G2, nl_f_G2, nl_f_G2);
    for (int n=0; n<n_mesh; ++n) {
      for (int l3=0; l3<nl_b_G2; ++l3) {
        for (int lp1=0; lp1<nl_f_G2; ++lp1) {
          for (int lp2=0; lp2<nl_f_G2; ++lp2) {
            right_tensor(n, l3, lp1, lp2) = Tnl_f(n,lp2) * w_tensor(n, l3, lp1);
          }
        }
      }
    }

    std::array<Eigen::IndexPair<int>,1> product_dims {{ Eigen::IndexPair<int>(2,0) }};

    Eigen::Tensor<std::complex<double>,5> trans_tensor
        = 2*left_tensor.contract(right_tensor, product_dims).real().cast<std::complex<double>>();

    Eigen::Tensor<std::complex<double>,3> data_G1;
    detail::copy_to_tensor(data_G1, Gl);
    Eigen::Tensor<std::complex<double>,3> data_G1_slice = data_G1.slice(
        std::array<long,3>{{0,0,0}},
        std::array<long,3>{{nl_f_G2, data_G1.dimension(1), data_G1.dimension(2)}}
    );

    //tmp1: (l1, l2, l3, lp1, lp2) * (lp1, i, l) => (l1, l2, l3, lp2, i, l)
    auto tmp1 = trans_tensor.contract(
        data_G1_slice,
        std::array<Eigen::IndexPair<int>,1>{{Eigen::IndexPair<int>(3,0)}}
    );

    //tmp2: (l1, l2, l3, lp2, i, l) * (lp2, k, j) => (l1, l2, l3, i, l, k, j)
    auto tmp2 = tmp1.contract(
        data_G1_slice,
        std::array<Eigen::IndexPair<int>,1>{{Eigen::IndexPair<int>(3,0)}}
    );

    //tmp3: (l1, l2, l3, i, j, k, l)
    Eigen::Tensor<std::complex<double>,7> tmp3 = tmp2.shuffle(std::array<int,7>{{0, 1, 2,  3, 6, 5, 4}});

    detail::copy_from_tensor(tmp3, g2);

    return g2;
  }

  /***
   * Transform a Hatree-type G2 object to a Fock-type G2
   * @tparam SEVEN_INDEX_GF Type of G2
   */
  template<typename SEVEN_INDEX_GF>
  class transformer_Hartree_to_Fock {
    using M4 = typename SEVEN_INDEX_GF::mesh4_type;
    using M5 = typename SEVEN_INDEX_GF::mesh5_type;
    using M6 = typename SEVEN_INDEX_GF::mesh6_type;
    using M7 = typename SEVEN_INDEX_GF::mesh7_type;

   public:
    transformer_Hartree_to_Fock(
      const alps::gf::numerical_mesh<double>& mesh_f,
      const alps::gf::numerical_mesh<double>& mesh_b) : mesh_f_(mesh_f), mesh_b_(mesh_b) {

      if (mesh_f.statistics() != alps::gf::statistics::FERMIONIC) {
        throw std::invalid_argument("mesh_f must be fermionic.");
      }
      if (mesh_b.statistics() != alps::gf::statistics::BOSONIC) {
        throw std::invalid_argument("mesh_b must be bosonic.");
      }

      std::vector<alps::gf::piecewise_polynomial<double>> basis_f, basis_b;
      for (int l=0; l<mesh_f.extent(); ++l) {
        basis_f.push_back(mesh_f.basis_function(l));
      }
      for (int l=0; l<mesh_b.extent(); ++l) {
        basis_b.push_back(mesh_b.basis_function(l));
      }

      C_tensor_ = compute_C_tensor(basis_f, basis_b).cast<std::complex<double>>();
    }

    SEVEN_INDEX_GF operator()(const SEVEN_INDEX_GF& G2_H) const {
      using dcomplex = std::complex<double>;
      if (G2_H.mesh1() != mesh_f_ || G2_H.mesh2() != mesh_f_ || G2_H.mesh3() != mesh_b_) {
        throw std::runtime_error("mesh mismatch");
      }

      auto G2_F(G2_H);

      int dim_f = G2_H.mesh1().extent();
      int dim_b = G2_H.mesh3().extent();

      int n4 = G2_H.mesh4().extent();
      int n5 = G2_H.mesh5().extent();
      int n6 = G2_H.mesh6().extent();
      int n7 = G2_H.mesh7().extent();

      Eigen::TensorMap<Eigen::Tensor<dcomplex,7>> G2_H_map(const_cast<dcomplex*>(G2_H.data().origin()), n7, n6, n5, n4, dim_b, dim_f, dim_f);

      //G_F_data: dim_f, dim_f, dim_b, f4, f3, f2, f1
      std::array<Eigen::IndexPair<int>,3> product_dims = {
          Eigen::IndexPair<int>(3, 6),
          Eigen::IndexPair<int>(4, 5),
          Eigen::IndexPair<int>(5, 4)
      };

      Eigen::TensorMap<Eigen::Tensor<dcomplex,7>> G2_F_map(const_cast<dcomplex*>(G2_F.data().origin()), n7, n6, n5, n4, dim_b, dim_f, dim_f);

      Eigen::Tensor<dcomplex,7> tmp = C_tensor_.contract(G2_H_map, product_dims);

      //at this point, the indices are (l1, l2, l3, f4, f3, f2, f1)
      //This will be transposed into (f2, f3, f4, f1, l3, l2, l1).

      std::array<int,7> shuffle {{5, 4, 3, 6, 2, 1, 0}};
      Eigen::Tensor<dcomplex,7> tmp2 = tmp.shuffle(shuffle);

      G2_F_map = tmp2;

      return G2_F;
    }

   private:
    Eigen::Tensor<std::complex<double>,6> C_tensor_;
    alps::gf::numerical_mesh<double> mesh_f_, mesh_b_;
  };

  /*
  template<typename S_DST, typename S_SRC, typename M1_DST, typename M1_SRC, typename M2, typename M3>
  //class transformer<alps::gf::three_index_gf<S_DST,M1_DST,M2,M3>, alps::gf::three_index_gf<S_SRC,M1_SRC,M2,M3> > {
  class transformer_base {
    using gt_dst = alps::gf::three_index_gf<S_DST,M1_DST,M2,M3>;
    using gt_src =  alps::gf::three_index_gf<S_SRC,M1_SRC,M2,M3>;

    static constexpr int num_index = 3;

   public:
    transformer(const gt_dst& mesh_dst, const gt_src& mesh_src) :
        mesh_dst_(mesh_dst),
        mesh_src_(mesh_src),
        Tmat_(mesh_dst.extents(),mesh_src.extents()) {
      if (mesh_dst_.beta() != mesh_src_.beta() || mesh_dst_.statistics() != mesh_src_.statistics()) {
        throw std::invalid_argument("mesh_dst and mesh_src are not compatible.");
      }

      //Compute matrix elements of Tmat_ in a derived class
    };

    gt_dst operator()(const gt_src& g_in) {
      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      Eigen::Tensor<std::complex<double>, num_index> data_in(dim_in, dim2, dim3);
      detail::copy_to_tensor(g_in.data(), data_in);

      auto product_dims = { Eigen::IndexPair<int>(1, 0) };
      auto data_out = Tmat_.contract(data_in, product_dims);

      gt_dst g_out(mesh_dst_, g_in.mesh2(), g_in.mesh3());
      for (int i1 = 0; i1 < mesh_dst_.extent(); ++i1) {
        for (int i2 = 0; i2 < dim2; ++i2) {
          for (int i3 = 0; i3 < dim3; ++i3) {
            g_out(
                gt_dst::index_type(i1),
                typename M2::index_type(i2),
                typename M3::index_type(i3)
            ) = data_out(i1, i2, i3);
          }
        }
      }

      return g_out;
    }

   private:
    gt_src mesh_src_;
    gt_dst mesh_dst_;
    Eigen::Tensor<std::complex<double>,2> Tnl_;
  };
  */

  /** Converter for Matsubara mesh to Numerical mesh(three-index gf)
   *
   * We assume an 1/iw_n tail for diagonal components.
   * @tparam M2 Type of the 2nd mesh
   * @tparam M3 Type of the 3rd mesh
   */
/*
  template<typename M2, typename M3>
  class transformer<nmesh_three_index_gf<std::complex<double>,M2,M3>, omega_three_index_gf<std::complex<double>,M2,M3> > {
    using gt_dst = nmesh_three_index_gf<std::complex<double>,M2,M3>;
    using gt_src = omega_three_index_gf<std::complex<double>,M2,M3>;

    using M1_dst = alps::gf::numerical_mesh<double>;
    using M1_src = alps::gf::matsubara_positive_mesh;

    using pp_type = alps::gf::piecewise_polynomial<double>;
    static constexpr int num_index = 3;

   public:
    transformer(const M1_dst& mesh_dst, const M1_src& mesh_src) : mesh_dst_(mesh_dst), mesh_src_(mesh_src), trans_mat_(0,0) {
      if (mesh_dst_.beta() != mesh_src_.beta() || mesh_dst_.statistics() != mesh_src_.statistics()) {
        throw std::invalid_argument("mesh_dst and mesh_src are not compatible.");
      }

      const int niw = mesh_src_.extent();
      const int nl = mesh_dst_.extent();

      trans_mat_ = Eigen::Tensor<std::complex<double>,2>(niw, nl);

      std::vector<pp_type> basis_functions;
      for (int l=0; l < nl; ++l) {
        basis_functions.push_back(mesh_dst_.basis_function(l));
      }

      compute_transformation_matrix_to_matsubara(
          0, niw-1,
          mesh_dst_.statistics(),
          basis_functions,
          trans_mat_
      );

      //Hermite conjugate
      trans_mat_ = trans_mat_.conjugate().shuffle(std::array<int,2>{{1, 0}});
    };

    gt_dst operator()(const gt_src& g_in) {
      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      Eigen::Tensor<std::complex<double>, num_index> data_omega(dim_in, dim2, dim3);
      detail::copy_to_tensor(g_in.data(), data_omega);

      auto product_dims = { Eigen::IndexPair<int>(1, 0) };
      //contributions from positive omega and negative omega
      auto data_l = trans_mat_.contract(data_omega, product_dims)
          + trans_mat_.contract(data_omega, product_dims).shuffle(std::array<Eigen::IndexPair<int>>{{0,2,1}});

      gt_dst g_out(mesh_dst_, g_in.mesh2(), g_in.mesh3());
      for (int i1 = 0; i1 < mesh_dst_.extent(); ++i1) {
        for (int i2 = 0; i2 < dim2; ++i2) {
          for (int i3 = 0; i3 < dim3; ++i3) {
            g_out(
                gt_dst::index_type(i1),
                typename M2::index_type(i2),
                typename M3::index_type(i3)
            ) = data_l(i1, i2, i3);
          }
        }
      }

      return g_out;
    }

   private:
    M1_src mesh_src_;
    M1_dst mesh_dst_;
    Eigen::Tensor<std::complex<double>,2> trans_mat_;
  };
    */

}//namespace gf_extension
}//namespace alps
