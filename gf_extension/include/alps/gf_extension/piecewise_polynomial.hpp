/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <complex>
#include <cmath>
#include <vector>

#include <alps/gf/piecewise_polynomial.hpp>

//#include <boost/multi_array.hpp>
//#include <boost/lexical_cast.hpp>
//#include <boost/typeof/typeof.hpp>

//#include <alps/hdf5/archive.hpp>
//#include <alps/hdf5/complex.hpp>
//#include <alps/hdf5/vector.hpp>
//#include <alps/hdf5/multi_array.hpp>

#include <alps/gf/mpi_bcast.hpp>

/**
 * @brief Class representing a pieacewise polynomial and utilities
 */
namespace alps {
  namespace gf_extension {

    template<typename T>
    alps::gf::piecewise_polynomial<T>
    multiply(const alps::gf::piecewise_polynomial<T> &f1, const alps::gf::piecewise_polynomial<T> &f2) {
      if (f1.section_edges() != f2.section_edges()) {
        throw std::runtime_error("Two pieacewise_polynomial objects with different sections cannot be multiplied.");
      }

      const int k1 = f1.order();
      const int k2 = f2.order();
      const int k = k1 + k2;

      alps::gf::piecewise_polynomial<T> r(k, f1.section_edges());
      for (int s=0; s < f1.num_sections(); ++s) {
        for (int p = 0; p <= k; p++) {
          r.coefficient(s, p) = 0.0;
        }
        for (int p1 = 0; p1 <= k1; ++p1) {
          for (int p2 = 0; p2 <= k2; ++p2) {
            r.coefficient(s, p1+p2) += f1.coefficient(s, p1) * f2.coefficient(s, p2);
          }
        }
      }
      return r;
    }

    template<typename T>
    T
    integrate(const alps::gf::piecewise_polynomial<T> &y) {
      const int k = y.order();

      std::vector<T> rvec(k+1, 0.0);
      for (int s=0; s < y.num_sections(); ++s) {
        auto dx = y.section_edge(s+1) - y.section_edge(s);
        auto dx_power = dx;
        for (int p=0; p<=k; ++p) {
          rvec[p] += dx_power * y.coefficient(s, p);
          dx_power *= dx;
        }
      }

      double r = 0.0;
      //double coeff = 1.0;
      for (int p=0; p<=k; ++p) {
        //coeff /= p+1;
        //r += coeff * rvec[p];
        r += rvec[p]/(p+1);
      }
      return r;
    }



  }
}

