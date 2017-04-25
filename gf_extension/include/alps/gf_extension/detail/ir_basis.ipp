#include "../ir_basis.hpp"

#include "spline.hpp"

extern "C" void dgesvd_(const char *jobu, const char *jobvt,
                        const int *m, const int *n, double *a, const int *lda,
                        double *s, double *u, const int *ldu,
                        double *vt, const int *ldvt,
                        double *work, const int *lwork, int *info);

extern "C" void dgesdd_(const char *jobz,
                        const int *m, const int *n, double *a, const int *lda,
                        double *s, double *u, const int *ldu,
                        double *vt, const int *ldvt,
                        double *work, const int *lwork, const int *iwork, int *info);

namespace alps {
  namespace gf_extension {
    namespace detail {
      template<typename T>
      inline std::vector<T> linspace(T minval, T maxval, int N) {
        std::vector<T> r(N);
        for (int i = 0; i < N; ++i) {
          r[i] = i * (maxval - minval) / (N - 1) + minval;
        }
        return r;
      }

      template<class Matrix, class Vector>
      void svd_square_matrix(Matrix &K, int n, Vector &S, Matrix &Vt, Matrix &U) {
        char jobu = 'S';
        char jobvt = 'S';
        int lda = n;
        int ldu = n;
        int ldvt = n;

        double *vt = Vt.data();
        double *u = U.data();
        double *s = S.data();

        double dummywork;
        int lwork = -1;
        int info = 0;

        double *A = K.data();
        std::vector<int> iwork(8 * n);

        //get optimal workspace
        dgesdd_(&jobu, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &dummywork, &lwork, &iwork[0], &info);

        lwork = int(dummywork) + 32;
        Vector work(lwork);

        dgesdd_(&jobu, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &work[0], &lwork, &iwork[0], &info);
        if (info != 0) {
          throw std::runtime_error("SVD failed to converge!");
        }
      }
    }//namespace detail

    /***
     * Construct a piecewise polynomial by means of cubic spline
     * @param T  we expect T=double
     * @param x_array  values of x
     * @param y_array  values of y
     */
    template<typename T>
    alps::gf::piecewise_polynomial<T> construct_piecewise_polynomial_cspline(
        const std::vector<double> &x_array, const std::vector<double> &y_array) {

      const int n_points = x_array.size();
      const int n_section = n_points - 1;

      boost::multi_array<double, 2> coeff(boost::extents[n_section][4]);

      // Cubic spline interpolation
      tk::spline spline;
      spline.set_points(x_array, y_array);

      // Construct piecewise_polynomial
      for (int s = 0; s < n_section; ++s) {
        for (int p = 0; p < 4; ++p) {
          coeff[s][p] = spline.get_coeff(s, p);
        }
      }
      return alps::gf::piecewise_polynomial<T>(n_section, x_array, coeff);
    };

    /// do a svd for the given parity sector (even or odd)
    template<typename T>
    void do_svd(const kernel <T> &knl, int parity, int N, double cutoff_singular_values,
                std::vector<double> &singular_values,
                std::vector<alps::gf::piecewise_polynomial<double> > &basis_functions
    ) {
      typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

      double de_cutoff = 2.5;

      //DE mesh for x
      std::vector<double> tx_vec = detail::linspace<double>(0.0, de_cutoff, N);
      std::vector<double> weight_x(N), x_vec(N);
      for (int i = 0; i < N; ++i) {
        x_vec[i] = std::tanh(0.5 * M_PI * std::sinh(tx_vec[i]));
        //sqrt of the weight of DE formula
        weight_x[i] = std::sqrt(0.5 * M_PI * std::cosh(tx_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(tx_vec[i]));
      }

      //DE mesh for y
      std::vector<double> ty_vec = detail::linspace<double>(-de_cutoff, 0.0, N);
      std::vector<double> y_vec(N), weight_y(N);
      for (int i = 0; i < N; ++i) {
        y_vec[i] = std::tanh(0.5 * M_PI * std::sinh(ty_vec[i])) + 1.0;
        //sqrt of the weight of DE formula
        weight_y[i] = std::sqrt(0.5 * M_PI * std::cosh(ty_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(ty_vec[i]));
      }

      matrix_t K(N, N);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          K(i, j) = weight_x[i] * (knl(x_vec[i], y_vec[j]) + parity * knl(x_vec[i], -y_vec[j])) * weight_y[j];
        }
      }

      //Perform SVD
      Eigen::VectorXd svalues(N);
      matrix_t U(N, N), Vt(N, N);
      detail::svd_square_matrix(K, N, svalues, Vt, U);

      //Count non-zero SV
      int dim = N;
      for (int i = 1; i < N; ++i) {
        if (std::abs(svalues(i) / svalues(0)) < cutoff_singular_values) {
          dim = i;
          break;
        }
      }

      //Rescale U and V
      U.conservativeResize(N, dim);
      for (int l = 0; l < dim; ++l) {
        for (int i = 0; i < N; ++i) {
          U(i, l) /= weight_x[i];
        }
        if (U(N - 1, l) < 0.0) {
          U.col(l) *= -1;
        }
      }

      singular_values.resize(dim);
      for (int l = 0; l < dim; ++l) {
        singular_values[l] = svalues(l);
      }

      //cubic spline interpolation
      const int n_points = 2 * N + 1;
      const int n_section = n_points + 1;
      std::vector<double> x_array(n_points), y_array(n_points);

      //set up x values
      for (int itau = 0; itau < N; ++itau) {
        x_array[-itau + n_points / 2] = -x_vec[itau];
        x_array[itau + n_points / 2] = x_vec[itau];
      }
      x_array.front() = -1.0;
      x_array.back() = 1.0;

      // spline interpolation
      for (int l = 0; l < dim; ++l) {
        //set up y values
        for (int itau = 0; itau < N; ++itau) {
          y_array[-itau + n_points / 2] = parity * U(itau, l);
          y_array[itau + n_points / 2] = U(itau, l);
        }
        if (parity == -1) {
          y_array[n_points / 2] = 0.0;
        }
        y_array.front() = parity * U(N - 1, l);
        y_array.back() = U(N - 1, l);

        basis_functions.push_back(construct_piecewise_polynomial_cspline<double>(x_array, y_array));
      }

      orthonormalize(basis_functions);
      assert(singular_values.size() == basis_functions.size());
    }

    template<typename Scalar>
    ir_basis<Scalar>::ir_basis(const kernel <Scalar> &knl, int max_dim, double cutoff, int N) : p_knl_(knl.clone()) {
      std::vector<double> even_svalues, odd_svalues, svalues;
      std::vector<pp_type> even_basis_functions, odd_basis_functions;

      do_svd<Scalar>(*p_knl_, 1, N, cutoff, even_svalues, even_basis_functions);
      do_svd<Scalar>(*p_knl_, -1, N, cutoff, odd_svalues, odd_basis_functions);

      //Merge
      basis_functions_.resize(0);
      assert(even_basis_functions.size() == even_svalues.size());
      assert(odd_basis_functions.size() == odd_svalues.size());
      for (int pair = 0; pair < std::max(even_svalues.size(), odd_svalues.size()); ++pair) {
        if (pair < even_svalues.size()) {
          svalues.push_back(even_svalues[pair]);
          basis_functions_.push_back(even_basis_functions[pair]);
        }
        if (pair < odd_svalues.size()) {
          svalues.push_back(odd_svalues[pair]);
          basis_functions_.push_back(odd_basis_functions[pair]);
        }
      }

      assert(even_svalues.size() + odd_svalues.size() == svalues.size());

      //use max_dim
      if (svalues.size() > max_dim) {
        svalues.resize(max_dim);
        basis_functions_.resize(max_dim);
      }

      //Check
      for (int i = 0; i < svalues.size() - 1; ++i) {
        if (svalues[i] < svalues[i + 1]) {
          //FIXME: SHOULD NOT THROW IN A CONSTRUCTOR
          throw std::runtime_error("Even and odd basis functions do not appear alternately.");
        }
      }
    };

    template<typename Scalar>
    void
    ir_basis<Scalar>::value(double x, std::vector<double> &val) const {
      assert(val.size() >= basis_functions_.size());
      assert(x >= -1.00001 && x <= 1.00001);

      const int dim = basis_functions_.size();

      if (dim > val.size()) {
        val.resize(dim);
      }
      const int section = basis_functions_[0].find_section(x);
      for (int l = 0; l < dim; l++) {
        val[l] = basis_functions_[l].compute_value(x, section);
      }
    }

    template<typename Scalar>
    void
    ir_basis<Scalar>::compute_Tnl(
        int n_min, int n_max,
        boost::multi_array<std::complex<double>, 2> &Tnl
    ) const {
      const int niw = n_max - n_min + 1;
      Eigen::Tensor<std::complex<double>, 2> Tnl_tmp(niw, basis_functions_.size());
      compute_Tnl(n_min, n_max, Tnl_tmp);
      Tnl.resize(boost::extents[niw][basis_functions_.size()]);
      for (int i = 0; i < niw; ++i) {
        for (int l = 0; l < basis_functions_.size(); ++l) {
          Tnl[i][l] = Tnl_tmp(i, l);
        }
      }
    };

    template<typename Scalar>
    void
    ir_basis<Scalar>::compute_Tnl(
        int n_min, int n_max,
        Eigen::Tensor<std::complex<double>, 2> &Tnl
    ) const {
      alps::gf_extension::compute_transformation_matrix_to_matsubara<double>(n_min,
                                                                             n_max,
                                                                             p_knl_->get_statistics(),
                                                                             basis_functions_,
                                                                             Tnl);
    }

    Eigen::Tensor<std::complex<double>, 3>
    compute_w_tensor(
        const std::vector<double> &n_vec,
        const fermionic_ir_basis &basis_f,
        const bosonic_ir_basis &basis_b) {
      using dcomplex = std::complex<double>;

      const int dim_f = basis_f.dim();
      const int dim_b = basis_b.dim();


      std::vector<double> w(n_vec.size());
      for (int n = 0; n < n_vec.size(); ++n) {
        w[n] = M_PI * (n_vec[n] + 0.5);
      }

      std::vector<alps::gf::piecewise_polynomial<double>> prods(dim_f * dim_b);
      for (int lp = 0; lp < dim_f; ++lp) {
        for (int l = 0; l < dim_b; ++l) {
          prods[l + lp * dim_b] = alps::gf_extension::multiply(basis_b.all()[l], basis_f.all()[lp]);
        }
      }

      Eigen::Tensor<dcomplex, 2> integral(n_vec.size(), prods.size());
      const int b_size = 500;
      for (int b = 0; b < n_vec.size()/b_size+1; ++b) {
        auto n_start = b * b_size;
        auto n_last = std::min((b+1) * b_size-1, (int) n_vec.size()-1);
        if (n_start > n_last) {
          continue;
        }

        std::vector<double> w_batch;
        for (int n=n_start; n<=n_last; ++n) {
          w_batch.push_back(w[n]);
        }
        Eigen::Tensor<dcomplex,2> sub;
        alps::gf_extension::compute_integral_with_exp(w_batch, prods, sub);
        for (int n=n_start; n<=n_last; ++n) {
          for (int j=0; j<prods.size(); ++j) {
            integral(n,j) = sub(n-n_start,j);
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

  }//namespace gf_extension
}//namespace alps