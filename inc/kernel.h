#ifndef KERNEL_H_
#define KERNEL_H_

#include <Eigen/Dense>

template <int dim>
struct kernel_result {
  double w;
  Eigen::Matrix<double, dim, 1> grad_w;
  kernel_result() : w(0.), grad_w(Eigen::Matrix<double, dim, 1>::Zero()) {}
};

#endif /* KERNEL_H_ */

extern template struct kernel_result<2>;
extern template struct kernel_result<3>;
