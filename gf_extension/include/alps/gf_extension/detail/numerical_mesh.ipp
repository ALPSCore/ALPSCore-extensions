#include "../numerical_mesh.hpp"

namespace alps {
namespace gf_extension {
  namespace detail {
    /// Construct piecewise polynomials representing Matsubara basis functions: exp(-i w_n tau).
    /// For fermionic cases, w_n = (2*n+1)*pi/beta.
    /// For bosonci cases, w_n = (2*n)*pi/beta.
    /// Caution: when n is large, you need a very dense mesh. You are resposible for this.
    template<class T>
    void construct_matsubra_basis_functions(
        int n_min, int n_max,
        statistics s,
        const std::vector<double> &section_edges,
        int k,
        std::vector<alps::gf::piecewise_polynomial<std::complex<T> > > &results) {
      typedef alps::gf::piecewise_polynomial<std::complex<T> > pp_type;

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
        if (s == fermionic) {
          z = -std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (s == bosonic) {
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
  }//namespace detail

/**
 * Compute a transformation matrix from a give orthonormal basis set to Matsubara freq.
 * @tparam T  scalar type
 * @param n_min min index of Matsubara freq. index
 * @param n_max max index of Matsubara freq. index
 * @param statis Statistics (fermion or boson)
 * @param bf_src orthonormal basis functions. They must be piecewise polynomials of the same order.
 * @param Tnl  computed transformation matrix
 */
  template<class T>
  void compute_transformation_matrix_to_matsubara(
      int n_min, int n_max,
      statistics statis,
      const std::vector <alps::gf::piecewise_polynomial<T>> &bf_src,
      boost::multi_array<std::complex<double>, 2> &Tnl
  ) {
    typedef std::complex<double> dcomplex;
    typedef alps::gf::piecewise_polynomial<std::complex < double> > pp_type;
    typedef Eigen::Matrix <std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

    //order of polynomials used for representing Matsubara basis functions internally.
    const int k_iw = 16;

    const int k = bf_src[0].order();

    for (int l = 0; l < bf_src.size(); ++l) {
      if (k != bf_src[l].order()) {
        throw std::runtime_error(
            "Error in compute_transformation_matrix_to_matsubara: basis functions must be pieacewise polynomials of the same order");
      }
    }

    std::vector <pp_type> matsubara_functions;

    detail::construct_matsubra_basis_functions(n_min, n_max, statis, bf_src[0].section_edges(), k_iw, matsubara_functions);

    const int n_section = bf_src[0].num_sections();
    const int n_iw = n_max - n_min + 1;

    matrix_t left_mid_matrix(n_iw, k + 1);
    matrix_t left_matrix(n_iw, k_iw + 1);
    matrix_t mid_matrix(k_iw + 1, k + 1);
    matrix_t right_matrix(k + 1, bf_src.size());
    matrix_t r(n_iw, bf_src.size());
    r.setZero();

    std::vector<double> dx_power(k + k_iw + 2);

    const double cutoff = 0.1;
    for (int s = 0; s < n_section; ++s) {
      double x0 = bf_src[0].section_edge(s);
      double x1 = bf_src[0].section_edge(s + 1);
      double dx = x1 - x0;

      dx_power[0] = 1.0;
      for (int p = 1; p < dx_power.size(); ++p) {
        dx_power[p] = dx * dx_power[p - 1];
      }

      //Use Taylor expansion for exp(i w_n tau) for M_PI*(n+0.5)*dx < cutoff*M_PI
      int n_max_cs = std::max(std::min(static_cast<int>(cutoff / dx - 0.5), n_max), 0);

      for (int p = 0; p < k_iw + 1; ++p) {
        for (int p2 = 0; p2 < k + 1; ++p2) {
          mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
        }
      }

      for (int n = 0; n < n_max_cs - n_min + 1; ++n) {
        for (int p = 0; p < k_iw + 1; ++p) {
          left_matrix(n, p) = alps::gf::detail::conjg(matsubara_functions[n].coefficient(s, p));
        }
      }

      left_mid_matrix.block(0, 0, n_max_cs - n_min + 1, k + 1) =
          left_matrix.block(0, 0, n_max_cs - n_min + 1, k_iw + 1) * mid_matrix;

      //Compute the overlap exactly for M_PI*(n+0.5)*dx > cutoff*M_PI
      for (int n = n_max_cs + 1; n <= n_max; ++n) {
        std::complex<double> z;
        if (statis == fermionic) {
          z = std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (statis == bosonic) {
          z = std::complex<double>(0.0, n) * M_PI;
        }

        dcomplex dx_z = dx * z;
        dcomplex dx_z2 = dx_z * dx_z;
        dcomplex dx_z3 = dx_z2 * dx_z;
        dcomplex inv_z = 1.0 / z;
        dcomplex inv_z2 = inv_z * inv_z;
        dcomplex inv_z3 = inv_z2 * inv_z;
        dcomplex inv_z4 = inv_z3 * inv_z;
        dcomplex exp = std::exp(dx * z);
        dcomplex exp0 = std::exp((x0 + 1.0) * z);

        left_mid_matrix(n - n_min, 0) = (-1.0 + exp) * inv_z * exp0;
        left_mid_matrix(n - n_min, 1) = ((dx_z - 1.0) * exp + 1.0) * inv_z2 * exp0;
        left_mid_matrix(n - n_min, 2) = ((dx_z2 - 2.0 * dx_z + 2.0) * exp - 2.0) * inv_z3 * exp0;
        left_mid_matrix(n - n_min, 3) = ((dx_z3 - 3.0 * dx_z2 + 6.0 * dx_z - 6.0) * exp + 6.0) * inv_z4 * exp0;
      }

      for (int l = 0; l < bf_src.size(); ++l) {
        for (int p2 = 0; p2 < k + 1; ++p2) {
          right_matrix(p2, l) = bf_src[l].coefficient(s, p2);
        }
      }

      r += left_mid_matrix * right_matrix;
    }

    Tnl.resize(boost::extents[n_iw][bf_src.size()]);
    std::vector<double> inv_norm(bf_src.size());
    for (int l = 0; l < bf_src.size(); ++l) {
      inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
    }
    for (int n = 0; n < n_iw; ++n) {
      for (int l = 0; l < bf_src.size(); ++l) {
        // 0.5 is the inverse of the norm of exp(i w_n tau)
        Tnl[n][l] = r(n, l) * inv_norm[l] * std::sqrt(0.5);
      }
    }
  }

  /// Compute overlap <left | right> with complex conjugate
  template<class T1, class T2>
  void compute_overlap(
      const std::vector<alps::gf::piecewise_polynomial<T1> > &left_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2> > &right_vectors,
      boost::multi_array<typename result_of_overlap<T1,T2>::value, 2> &results) {
    typedef typename result_of_overlap<T1,T2>::value Tr;

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

    for (int n = 0; n < NL ; ++n) {
      if (k1 != left_vectors[n].order()) {
        throw std::runtime_error("Left vectors must be piecewise polynomials of the same order.");
      }
    }

    for (int n = 0; n < NR ; ++n) {
      if (k2 != right_vectors[n].order()) {
        throw std::runtime_error("Right vectors must be piecewise polynomials of the same order.");
      }
    }

    for (int l = 0; l < NR - 1; ++l) {
      if (right_vectors[l].section_edges() != right_vectors[l + 1].section_edges()) {
        throw std::runtime_error("Not supported");
      }
    }

    std::vector<double> x_min_power(k1+k2+2), dx_power(k1+k2+2);

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

}//namespace gf_extension
}//namespace alps
