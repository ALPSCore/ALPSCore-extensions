#pragma once

#include <set>
#include <unordered_map>

namespace alps {
namespace gf_extension {

template<typename T, typename F>
class AdaptiveSummation {
 public:
  AdaptiveSummation(const F& f, long n_min, long n_max) : f_(f) {
    auto n = create_node(n_min, n_max);
    sum_abs_error_ = n.error;
    nodes_.insert(n);
  }

  T evaluate(double eps_abs, int min_points=100) {
    int loop = 0;
    while (sum_abs_error_ > eps_abs || nodes_.size() < min_points) {
      //std::cout << loop << " " << sum_abs_error_ << std::endl;
      auto it_begin = nodes_.begin();
      assert(it_begin->n_max != it_begin->n_min);
      long n_middle = it_begin->n_max == it_begin->n_min+1 ? it_begin->n_min : static_cast<long>((it_begin->n_max+it_begin->n_min)/2);
      node n1 = create_node(it_begin->n_min, n_middle);
      node n2 = create_node(n_middle+1, it_begin->n_max);

      auto de = n1.error + n2.error - it_begin->error;

      nodes_.erase(it_begin);
      nodes_.insert(n1);
      nodes_.insert(n2);

      if ( std::abs(sum_abs_error_ + de) < 1e-10*std::abs(sum_abs_error_) ) {
        sum_abs_error_ = 0.0;
        for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
          if (std::abs(it->error) < 1e-15*std::abs(sum_abs_error_)) {
            break;
          }
          sum_abs_error_ += it->error;
        }
      } else {
        sum_abs_error_ += de;
      }

      ++ loop;
    }

    T sum = nodes_.begin()->value;
    auto it_second = nodes_.begin();
    std::cout << " " << nodes_.begin()->n_min << " " << nodes_.begin()->n_max << " v = " << nodes_.begin()->value << " +/- " << nodes_.begin()->error << std::endl;
    ++it_second;
    for (auto it = it_second; it != nodes_.end(); ++it) {
      std::cout << " " << it->n_min << " " << it->n_max << " v = " << it->value << " +/- " << it->error << std::endl;
      sum += it->value;
    }
    std::cout << "size " << nodes_.size() << std::endl;
    return sum;
  }


 private:
  struct node {
    long n_min, n_max;
    T value, error;

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
      //long n_middle = static_cast<long>(0.5*(n_max + n_min));
      //return static_cast<double>(n_max-n_min+1) * get_f_value(n_middle);
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
      ////std::cout << " debug " << n_min << " " << n_max << " " << n_middle << std::endl;
      //std::cout << " debug " << n.value << std::endl;
      //std::cout << " debug " << approximate_sum(n_min, n_middle) << std::endl;
      //std::cout << " debug " << approximate_sum(n_middle+1, n_max) << std::endl;
      n.error = std::abs(n.value -  approximate_sum(n_min, n_middle) - approximate_sum(n_middle+1, n_max));
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
