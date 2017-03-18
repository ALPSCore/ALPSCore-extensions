#include <alps/gf_extension/numerical_mesh.hpp>
#include <alps/gf_extension/detail/numerical_mesh.ipp>

#include <alps/gf_extension/ir/ir_basis.hpp>
#include <alps/gf_extension/ir/detail/ir_basis.ipp>

namespace alps {
namespace gf_extension {
  template void compute_transformation_matrix_to_matsubara<double>(
      int n_min, int n_max,
      statistics statis,
      const std::vector <alps::gf::piecewise_polynomial<double> > &bf_src,
      boost::multi_array<std::complex<double>, 2> &Tnl
  );

  namespace ir {
    template class basis<double>;

    template
    alps::gf::piecewise_polynomial<double> construct_piecewise_polynomial_cspline(
        const std::vector<double> &x_array, const std::vector<double> &y_array);
  }
}
}
