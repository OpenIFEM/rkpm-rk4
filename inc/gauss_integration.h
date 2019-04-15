#ifndef GAUSS_INTEGRATION_H_
#define GAUSS_INTEGRATION_H_

#include "element.h"
#include "particle.h"

#include <assert.h>
#include <Eigen/Dense>
#include <tuple>
#include <type_traits>
#include <vector>

template <int dim>
struct vol_quad_info;

template <>
struct vol_quad_info<2> {
  static constexpr unsigned int quad_pts_per_cell = 4;
};

template <>
struct vol_quad_info<3> {
  static constexpr unsigned int quad_pts_per_cell = 8;
};

// 2nd order Gauss quadrature
template <int dim>
class gauss_int {
 private:
  // vector of size dim
  using vec = Eigen::Matrix<double, dim, 1>;
  // matrix of shape dim * dim
  using mat = Eigen::Matrix<double, dim, dim>;
  // vector of size no_support_vertices
  using val_t = Eigen::Matrix<double, vol_quad_info<dim>::quad_pts_per_cell, 1>;
  // matrix of shape dim * no_support_vertices
  using der_t =
      Eigen::Matrix<double, dim, vol_quad_info<dim>::quad_pts_per_cell>;

  std::vector<element<dim>> m_elements;
  std::vector<vec> m_local_gp_pos;
  std::vector<double> m_local_gp_weight;

 public:
  gauss_int(std::vector<element<dim>> elements);

  void update_gauss_points(particle<dim> **particles,
                           particle<dim> **gauss_points);
};

#endif /* GAUSS_INTEGRATION_H_ */

extern template class gauss_int<2>;
extern template class gauss_int<3>;
