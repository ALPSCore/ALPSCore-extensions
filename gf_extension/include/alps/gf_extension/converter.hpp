#pragma once

#include <alps/gf/gf.hpp>
//#include <alps/gf/mesh.hpp>

#include <Eigen/Core>

#include <Eigen/CXX11/Tensor>

namespace alps {
namespace gf_extension {
  /// Statistics of Matsubara frequencies
  enum statistics { fermionic, bosonic };

  //AVOID USING BOOST_TYPEOF
  template <class T1, class T2>
  struct result_of_overlap {
    typedef std::complex<double> value;
  };

  template <>
  struct result_of_overlap<double,double> {
    typedef double value;
  };


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
      alps::gf::statistics::statistics_type statis,
      const std::vector <alps::gf::piecewise_polynomial<T> > &bf_src,
      Eigen::Tensor<std::complex<double>, 2> &Tnl
  );

  /// Compute overlap <left | right> with complex conjugate
  template<class T1, class T2>
  void compute_overlap(
      const std::vector<alps::gf::piecewise_polynomial<T1> > &left_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2> > &right_vectors,
      boost::multi_array<typename result_of_overlap<T1,T2>::value, 2> &results);

  /// Compute a transformation matrix from a src orthogonal basis set to a dst orthogonal basis set.
  /// The basis vectors are NOT necessarily normalized to 1.
  template<class T1, class T2>
  void compute_transformation_matrix(
      const std::vector<alps::gf::piecewise_polynomial<T1> > &dst_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2> > &src_vectors,
      boost::multi_array<typename result_of_overlap<T1,T2>::value, 2> &results) {
    compute_overlap(dst_vectors, src_vectors, results);

    std::vector<double> coeff1(dst_vectors.size());
    for (int l = 0; l < dst_vectors.size(); ++l) {
      coeff1[l] = 1.0/std::sqrt(
          static_cast<double>(
              dst_vectors[l].overlap(dst_vectors[l])
          )
      );
    }

    std::vector<double> coeff2(src_vectors.size());
    for (int l = 0; l < src_vectors.size(); ++l) {
      coeff2[l] = 1.0/std::sqrt(
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
   * Template class converter. Convert a GF object to a GF object of a different type.
   * Some conversion may not be safe.
   * @tparam G_DST Destination type
   * @tparam G_SRC Source type
   */
  template<typename G_DST, typename G_SRC> class converter {};

  template<typename S, typename M2, typename M3>
  using itime_three_index_gf = alps::gf::three_index_gf<S, alps::gf::itime_mesh, M2, M3>;

  template<typename S, typename M2, typename M3>
  using omega_three_index_gf = alps::gf::three_index_gf<S, alps::gf::matsubara_positive_mesh, M2, M3>;

  template<typename S, typename M2, typename M3>
  using nmesh_three_index_gf = alps::gf::three_index_gf<S, alps::gf::numerical_mesh<double>, M2, M3>;

  /// Numerical mesh to imaginary time (three-index gf)
  template<typename S, typename M2, typename M3>
  class converter<itime_three_index_gf<S,M2,M3>, nmesh_three_index_gf<S,M2,M3> >;

  /// Numerical mesh to Matsubara frequencies (three-index gf)
  template<typename S, typename M2, typename M3>
  class converter<omega_three_index_gf<S,M2,M3>, nmesh_three_index_gf<S,M2,M3> >;

}//namespace gf_extension
}//namespace alps


/************************************************************
****    Below, implementations follow.                     **
*************************************************************/

namespace alps {
namespace gf_extension {

  namespace detail {
    template<typename S1, typename S2, unsigned long N>
    void copy_from_tensor(const Eigen::Tensor<S1,N>& tensor, boost::multi_array<S2,N>& marray) {

      assert(marray.num_elements() == tensor.size());

      //From ColumnMajar to RowMajor. This also swaps dimensions.
      Eigen::Tensor<S1,N,Eigen::RowMajor> tensor_tmp = tensor.swap_layout();

      // Swap back dimensions
      std::array<int,N> shuffle;
      for (int i=0; i<N; ++i) {
        shuffle[i] = N - 1 - i;
      }
      tensor_tmp = tensor_tmp.shuffle(shuffle);

      std::copy(tensor_tmp.data(), tensor_tmp.data() + tensor_tmp.size(), marray.origin());
    }

    template<typename S1, typename S2, unsigned long N>
    void copy_to_tensor(const boost::multi_array<S1,N>& marray, Eigen::Tensor<S2,N>& tensor) {

      assert(marray.num_elements() == tensor.size());

      Eigen::Tensor<S2,N,Eigen::RowMajor> tensor_tmp(tensor.dimensions());

      std::copy(marray.origin(), marray.origin()+marray.num_elements(), tensor_tmp.data());

      Eigen::Tensor<S2,N> tensor_tmp2 = tensor_tmp.swap_layout();

      // Swap back dimensions
      std::array<int,N> shuffle;
      for (int i=0; i<N; ++i) {
        shuffle[i] = N - 1 - i;
      }

      tensor = tensor_tmp2.shuffle(shuffle);
    }

    /// Construct piecewise polynomials representing Matsubara basis functions: exp(-i w_n tau).
    /// For fermionic cases, w_n = (2*n+1)*pi/beta.
    /// For bosonci cases, w_n = (2*n)*pi/beta.
    /// Caution: when n is large, you need a very dense mesh. You are resposible for this.
    template<class T>
    void construct_matsubra_basis_functions(
        int n_min, int n_max,
        alps::gf::statistics::statistics_type s,
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
  }//namespace detail

  /// Numerical mesh to imaginary time (three-index gf)
  template<typename S, typename M2, typename M3>
  class converter<itime_three_index_gf<S,M2,M3>, nmesh_three_index_gf<S,M2,M3> > {
    using gt_dst = itime_three_index_gf<S,M2,M3>;
    using gt_src = nmesh_three_index_gf<S,M2,M3>;

    using index_type2 = typename M2::index_type;
    using index_type3 = typename M3::index_type;

   public:
    converter(double beta, int ntau) : beta_(beta), ntau_(ntau) {};

    gt_dst operator()(const gt_src& g_in) const {
      if (beta_ != g_in.mesh1().beta()) {
        throw std::runtime_error("gf object with a wrong beta has been passed to the converter");
      }
      gt_dst g_out(alps::gf::itime_mesh(g_in.mesh1().beta(), ntau_), g_in.mesh2(), g_in.mesh3());

      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      std::vector<double> coeff(dim_in);
      for (int il = 0; il < dim_in; ++il) {
        coeff[il] = sqrt(2.0/g_in.mesh1().basis_function(il).squared_norm())/beta_;
      }

      std::vector<double> vals(dim_in);

      for (int itau = 0; itau < ntau_ ; ++itau) {
        double tau = itau * (beta_ / (ntau_-1) );
        double x = 2 * tau / beta_ - 1.0;

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
    double beta_;
    int ntau_;
  };

  /// Numerical mesh to Matsubara frequencies (three-index gf)
  template<typename S, typename M2, typename M3>
  class converter<omega_three_index_gf<std::complex<double>,M2,M3>, nmesh_three_index_gf<S,M2,M3> > {
    using gt_dst = omega_three_index_gf<std::complex<double>,M2,M3>;
    using gt_src = nmesh_three_index_gf<S,M2,M3>;

    using nmesh_type = alps::gf::numerical_mesh<double>;
    using index_type2 = typename M2::index_type;
    using index_type3 = typename M3::index_type;
    using pp_type = alps::gf::piecewise_polynomial<double>;
    static constexpr int num_index = 3;

   public:
    converter(double beta, int niw) : beta_(beta), niw_(niw), Tnl_(0,0) {
    };

    void compute_Tnl() {
      const int nl = p_nmesh_->extent();

      Tnl_ = Eigen::Tensor<std::complex<double>,2>(niw_, nl);

      pp_type b;
      b = p_nmesh_->basis_function(0);

      std::vector<pp_type> basis_functions(nl);
      for (int l=0; l < nl; ++l) {
        basis_functions[l] = p_nmesh_->basis_function(l);
      }

      compute_transformation_matrix_to_matsubara(
          0, niw_-1,
          p_nmesh_->statistics(),
          basis_functions,
          Tnl_
      );
    }

    gt_dst operator()(const gt_src& g_in) {
      if (beta_ != g_in.mesh1().beta()) {
        throw std::runtime_error("gf object with a wrong beta has been passed to the converter");
      }

      if (!p_nmesh_ || *p_nmesh_ != g_in.mesh1()) {
        p_nmesh_.reset(new nmesh_type(g_in.mesh1()));
        compute_Tnl();
      }

      int dim_in = g_in.mesh1().extent();
      int dim2 = g_in.mesh2().extent();
      int dim3 = g_in.mesh3().extent();

      Eigen::Tensor<std::complex<double>, num_index> data_l(dim_in, dim2, dim3);
      detail::copy_to_tensor(g_in.data(), data_l);

      std::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
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
    double beta_;
    int niw_;
    mutable Eigen::Tensor<std::complex<double>,2> Tnl_;
    std::shared_ptr<nmesh_type> p_nmesh_;
  };


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
      alps::gf::statistics::statistics_type statis,
      const std::vector <alps::gf::piecewise_polynomial<T>> &bf_src,
      Eigen::Tensor<std::complex<double>,2> &Tnl
  ) {
    typedef std::complex<double> dcomplex;
    typedef alps::gf::piecewise_polynomial<std::complex < double> > pp_type;
    typedef Eigen::Matrix <std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    typedef Eigen::Tensor<std::complex<double>,2> tensor_t;

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
        if (statis == alps::gf::statistics::FERMIONIC) {
          z = std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (statis == alps::gf::statistics::BOSONIC) {
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

    Tnl = tensor_t(n_iw, bf_src.size());
    std::vector<double> inv_norm(bf_src.size());
    for (int l = 0; l < bf_src.size(); ++l) {
      inv_norm[l] = 1. / std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
    }
    for (int n = 0; n < n_iw; ++n) {
      for (int l = 0; l < bf_src.size(); ++l) {
        // 0.5 is the inverse of the norm of exp(i w_n tau)
        Tnl(n,l) = r(n, l) * inv_norm[l] * std::sqrt(0.5);
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

