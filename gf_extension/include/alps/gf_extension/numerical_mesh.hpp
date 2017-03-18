#pragma once

#include <alps/gf/mesh.hpp>

#include <Eigen/Core>

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
      statistics statis,
      const std::vector <alps::gf::piecewise_polynomial<T> > &bf_src,
      boost::multi_array<std::complex<double>, 2> &Tnl
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

}//namespace gf_extension
}//namespace alps
