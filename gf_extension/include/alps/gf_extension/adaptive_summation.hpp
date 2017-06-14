#pragma once

#include <set>
#include <unordered_map>

#include <boost/lexical_cast.hpp> //for debug

namespace alps {
namespace gf_extension {

template<typename T, typename F>
class AdaptiveSummation {
 public:
  AdaptiveSummation(const F& f, long n_min, long n_max) : f_(f), nodes_(), f_cache_() {
    if (n_max > std::numeric_limits<long>::max()/10) {
      throw std::runtime_error("n_max is too large.");
    }
    auto n = create_node(n_min, n_max);
    sum_abs_error_ = n.error;
    nodes_.insert(n);
  }

  double abs_error() const {return sum_abs_error_;}

  T evaluate(double eps_abs, int min_points=100) {
    int loop = 0;
    int check_interval = 100;

    while (true) {
      auto it_begin = nodes_.begin();
      assert(it_begin->n_max != it_begin->n_min);
      long n_middle = it_begin->n_max == it_begin->n_min+1 ? it_begin->n_min : static_cast<long>((it_begin->n_max+it_begin->n_min)/2);
      node n1 = create_node(it_begin->n_min, n_middle);
      node n2 = create_node(n_middle+1, it_begin->n_max);

      nodes_.erase(it_begin);
      nodes_.insert(n1);
      nodes_.insert(n2);

      if (nodes_.size() > min_points &&
          loop%check_interval==0 && nodes_.begin()->error < eps_abs) {
        sum_abs_error_ = 0.0;
        for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
          sum_abs_error_ += it->error;
          if (sum_abs_error_ > eps_abs) {
            break;
          }
        }
        if (sum_abs_error_ < eps_abs) {
          break;
        }
      }

      ++ loop;
    }

    sum_abs_error_ = 0.0;
    for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
      sum_abs_error_ += it->error;
    }

    T sum = nodes_.begin()->value;
    auto it_second = nodes_.begin();
    ++it_second;
    for (auto it = it_second; it != nodes_.end(); ++it) {
      sum += it->value;
    }
    return sum;
  }


 private:
  struct node {
    long n_min, n_max;
    T value;
    double error;

    bool operator>(const node& other) const {
      if (error == other.error) {
        return n_min < other.n_min;
      } else {
        return error > other.error;
      }
    }
  };

  inline T approximate_sum(long n_min, long n_max) const {
    assert(n_min <= n_max);
    if (n_min == n_max) {
      return get_f_value(n_min);
    } else {
      return 0.5*static_cast<double>(n_max-n_min+1) * (get_f_value(n_min) + get_f_value(n_max));
    }
  }

  node create_node(long n_min, long n_max) {
    assert(n_min <= n_max);
    node n;
    n.n_max = n_max;
    n.n_min = n_min;
    n.value = approximate_sum(n_min, n_max);

    //error estimate
    if (n_max == n_min) {
      n.error = 0.0;
    } else {
      long n_middle = n_max == n_min+1 ? n_min : static_cast<long>(0.5*(n_max + n_min));
      n.error = std::abs(n.value -  approximate_sum(n_min, n_middle) - approximate_sum(n_middle+1, n_max));
    }

    if (n.error < 0.0) {
      throw std::runtime_error("error is negative!");
    }

    return n;
  }

  T get_f_value(long n) const {
    auto it =  f_cache_.find(n);
    if (it != f_cache_.end()) {
      return it->second;
    } else {
      T v = f_(n);
      f_cache_[n] = v;
      return v;
    }
  }

 private:
  F f_;
  std::set<node,std::greater<node>> nodes_;
  mutable std::unordered_map<long,T> f_cache_;
  double sum_abs_error_;
};

}
}
