#pragma once

#include <algorithm>

#include <boost/math/special_functions/legendre.hpp>

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

      //Compute nodes (zeros) of Legendre polynomials
      inline std::vector<double> compute_legendre_nodes(int l) {
        double eps = 1e-10;
        if (l > 200) {
          throw std::runtime_error("l > 200 in compute_legendre_nodes");
        }

        std::vector<double> nodes;

        auto leg_diff = [](int l, double x) {
          return l * (x * boost::math::legendre_p(l, x) - boost::math::legendre_p(l - 1, x)) / (x * x - 1);
        };

        //i-th zero
        for (int i = 0; i < l / 2; i++) {
          //initial guess
          double x = std::cos(M_PI * (i + 1 - 0.25) / (l + 0.5));

          //Newton-Raphson iteration
          while (true) {
            double leg = boost::math::legendre_p(l, x);
            double x_new = x - 0.1 * leg / leg_diff(l, x);
            if (std::abs(x_new - x) < eps && std::abs(leg) < eps) {
              break;
            }
            x = x_new;
          }

          nodes.push_back(x);
          nodes.push_back(-x);
        }

        if (l % 2 == 1) {
          nodes.push_back(0.0);
        }

        std::sort(nodes.begin(), nodes.end());

        return nodes;
      }

    }//namespace detail
  }
}
