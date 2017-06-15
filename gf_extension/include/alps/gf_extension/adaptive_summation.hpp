#pragma once

#include <cinttypes>

#include <set>
#include <unordered_map>

#include <boost/functional/hash.hpp> //for bash_combine

#include <boost/lexical_cast.hpp> //for debug

namespace alps {
namespace gf_extension {

namespace detail {
  struct container_hasher {
    template<class T>
    std::size_t operator()(const T& c) const {
      std::size_t seed = 0;
      for(const auto& elem : c) {
        boost::hash_combine(seed, std::hash<typename T::value_type>()(elem));
      }
      return seed;
    }
  };

  template<int D, typename T, typename F> struct center_value_helper;

  template<typename T, typename F>
  struct
  center_value_helper<1,T,F> {
    T operator()(const F& f, const std::array<std::pair<std::int64_t,std::int64_t>,1>& min_max) const {
      return 0.5*(
          f(std::array<std::int64_t,1>{min_max[0].first})+
              f(std::array<std::int64_t,1>{min_max[0].second})
      );
    }
  };

  template<typename T, typename F>
  struct
  center_value_helper<2,T,F> {
    T operator()(const F& f, const std::array<std::pair<std::int64_t,std::int64_t>,2>& min_max) const {
      return 0.25 * (
          f(std::array<std::int64_t,2>{{min_max[0].first, min_max[1].first}})+
              f(std::array<std::int64_t,2>{{min_max[0].first, min_max[1].second}})+
              f(std::array<std::int64_t,2>{{min_max[0].second, min_max[1].first}})+
              f(std::array<std::int64_t,2>{{min_max[0].second, min_max[1].second}})
      );
    }
  };

}

template<int D, typename T, typename F>
class AdaptiveSummation {
 public:
  AdaptiveSummation(const F& f, const std::array<std::pair<std::int64_t,std::int64_t>,D>& min_max, bool use_cache = true) : f_(f), nodes_(), use_cache_(use_cache), f_cache_() {
    for (int d=0; d<D; ++d) {
      if (min_max[d].second > std::numeric_limits<std::int64_t>::max()/10) {
        throw std::runtime_error("n_max is too large.");
      }
    }
    std::cout << " min_max " << min_max[0].first << "  " << min_max[0].second << std::endl;
    auto n = create_node(min_max);
    sum_abs_error_ = n.error;
    nodes_.insert(n);

    /*
    volume_ = 1;
    for (int d=0; d<D; ++d) {
      volume_ *= min_max[d].second - min_max[d].first + 1;
    }
    std::cout << "initial volue " <<  volume_ << std::endl;
    */

    sanity_check();
  }

  AdaptiveSummation(const F& f, std::int64_t n_min, std::int64_t n_max, bool use_cache = true) :
      AdaptiveSummation(f, std::array<std::pair<std::int64_t,std::int64_t>,1>{std::pair<std::int64_t,std::int64_t>{n_min, n_max}}, use_cache) {
    if (D != 1) {
      throw std::runtime_error("This constructor is only for D=1!");
    }
  }

  double abs_error() const {return sum_abs_error_;}

  T evaluate(double eps_abs, int min_points=1000) {
    int loop = 0;
    int check_interval = 100;

    while (true) {
      sanity_check();
      node n1, n2;
      std::tie(n1, n2) = split_node(*nodes_.begin());

      nodes_.erase(nodes_.begin());
      nodes_.insert(n1);
      nodes_.insert(n2);

      if (nodes_.begin()->error == 0.0) {
        break;
      }

      if (loop % 100 == 0) {
        T sum = nodes_.begin()->value;
        auto it_second = nodes_.begin();
        ++it_second;
        for (auto it = it_second; it != nodes_.end(); ++it) {
          sum += it->value;
        }
        std::cout << "loop " << loop << " " << nodes_.size() << " " << sum << " +/- " << nodes_.begin()->error << std::endl;
      }

      if (nodes_.size() > min_points &&
          loop%check_interval==0 && nodes_.begin()->error < eps_abs) {
        sum_abs_error_ = 0.0;
        for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
          sum_abs_error_ += it->error;
          if (sum_abs_error_ > eps_abs) {
            break;
          }
        }
        //std::cout << "sum_abs " << sum_abs_error_ << std::endl;
        if (sum_abs_error_ < eps_abs) {
          break;
        }
      }
      //std::cout << loop << " " << nodes_.size() << " " << nodes_.begin()->error << std::endl;
      //int i = 0;
      //for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
        ////std::cout << i << " " << it->min_max[0].first << "  " << it->min_max[0].second << " : " << it->value << " " << it->error << std::endl;
        //++i;
      //}

      ++ loop;
    }

    /*
    int i = 0;
    for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
      std::cout << " node " << i << " "
                << it->min_max[0].first << "  " << it->min_max[0].second << " : "
                << it->value << " " << it->error << std::endl;
      ++i;
    }
     */

    sum_abs_error_ = 0.0;
    for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
      sum_abs_error_ += it->error;
    }

    //FIXME: SUM SMALL VALUES FIRST
    T sum = nodes_.begin()->value;
    auto it_second = nodes_.begin();
    ++it_second;
    for (auto it = it_second; it != nodes_.end(); ++it) {
      sum += it->value;
    }
    return sum;
  }


 private:
  using range_t = std::array<std::pair<std::int64_t,std::int64_t>,D>;

  struct node {
    std::array<std::pair<std::int64_t,std::int64_t>,D> min_max;
    T value;
    double error;
    int d_largest_error;

    bool operator>(const node& other) const {
      if (error == other.error) {
        return min_max < other.min_max;
      } else {
        return error > other.error;
      }
    }

    std::int64_t volume() const {
      std::int64_t v = 1;
      for (int d=0; d < D; ++d) {
        v *= min_max[d].second - min_max[d].first + 1;
      }
      return v;
    }
  };

  inline void sanity_check() const {
    /*
    std::int64_t sum_volume = 0.0;
    for (auto it=nodes_.begin(); it!=nodes_.end(); ++it) {
      sum_volume += it->volume();
    }
    if (sum_volume != volume_) {
      std::cout << "volume " << sum_volume << " " << volume_ << std::endl;
      throw std::runtime_error("Something got wrong.");
    }
     */
  }

  inline T approximate_sum(const range_t& min_max) const {
    T v = 0;
    double volume = 1.0;
    for (int d=0; d<D; ++d) {
      volume *= min_max[d].second - min_max[d].first + 1;
    }

    auto f_tmp = [&] (const std::array<std::int64_t,D>& n){
      return get_f_value(n);
    };

    return volume * detail::center_value_helper<D,T,decltype(f_tmp)>()(f_tmp, min_max);
    /*
    if (D==1) {
      return volume * 0.5*(
          get_f_value(std::array<std::int64_t,1>{min_max[0].first})+
          get_f_value(std::array<std::int64_t,1>{min_max[0].second})
      );
    } else if (D==2) {
      return 0.0;
    } else {
      throw std::runtime_error("D>2 is not implemented");
    }
    */
  }

  inline bool can_be_split(const range_t& min_max, int d) const {
    return min_max[d].second > min_max[d].first;
  }

  inline std::pair<range_t,range_t>
  split_range(const range_t& min_max, int d) const {
    range_t min_max1 = min_max;
    range_t min_max2 = min_max;

    assert(can_be_split(min_max,d));

    std::int64_t n_min = min_max[d].first;
    std::int64_t n_max = min_max[d].second;
    if (n_max == n_min+1) {
      min_max1[d].second = n_min;
      min_max2[d].first = n_max;
    } else {
      std::int64_t n_middle = (n_max + n_min)/2;
      min_max1[d].second = n_middle;
      min_max2[d].first = n_middle+1;
    }

    return std::make_pair(min_max1, min_max2);
  }

  inline
  std::pair<node,node>
  split_node(const node& n) const {
    //Split the range astd::int64_t the direction in which the error is the largest.
    range_t min_max1, min_max2;
    std::tie(min_max1, min_max2) = split_range(n.min_max, n.d_largest_error);

    return std::make_pair(create_node(min_max1), create_node(min_max2));
  }

  node create_node(const range_t& min_max) const {
    node n;
    n.min_max = min_max;
    n.value = approximate_sum(min_max);

    //error estimate
    n.error = 0.0;
    range_t min_max1, min_max2;
    n.d_largest_error = 0;
    for (int d=0; d<D; ++d) {
      if (!can_be_split(min_max, d)) {
        continue;
      }
      std::tie(min_max1, min_max2) = split_range(min_max, d);
      auto error_d = std::abs(n.value -  approximate_sum(min_max1) - approximate_sum(min_max2));
      if (error_d > n.error) {
        n.error = error_d;
        n.d_largest_error = d;
      }
    }

    return n;
  }

  //D=1 version
  //T get_f_value(std::int64_t n) const {
    //return get_f_value(std::array<std::int64_t,1>{{n}});
  //}

  T get_f_value(const std::array<std::int64_t,D>& n) const {
    if (!use_cache_) {
      return f_(n);
    }

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
  bool use_cache_;
  mutable std::unordered_map<std::array<std::int64_t,D>,T,detail::container_hasher> f_cache_;
  double sum_abs_error_;
  //std::int64_t volume_;
};


}
}
