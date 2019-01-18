#ifndef GAUSS_FACE_INTEGRATION_H_
#define GAUSS_FACE_INTEGRATION_H_

#include "element.h"
#include "particle.h"

#include <assert.h>
#include <Eigen/Dense>
#include <tuple>
#include <type_traits>
#include <vector>

template <int dim>
struct face_quad_info {
  static_assert(dim == 2 || dim == 3, "Not implemented!");
};

template <>
struct face_quad_info<2> {
  static constexpr unsigned int quad_pts_per_face = 2;
  static constexpr unsigned int face_nodes[4][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}};
};

template <>
struct face_quad_info<3> {
  static constexpr unsigned int quad_pts_per_face = 4;
  static constexpr unsigned int face_nodes[6][4] = {{0, 1, 2, 3}, {4, 7, 6, 5},
                                                    {0, 4, 5, 1}, {1, 5, 6, 2},
                                                    {2, 6, 7, 3}, {3, 7, 4, 0}};
};

template <int dim>
class gauss_face_int {
 private:
  // vector of size (dim - 1)
  using vec = Eigen::Matrix<double, dim - 1, 1>;
  // matrix of shape dim * no_support_vertices
  using mat =
      Eigen::Matrix<double, dim, face_quad_info<dim>::quad_pts_per_face>;
  // vector of size no_support_vertices
  using val_t =
      Eigen::Matrix<double, face_quad_info<dim>::quad_pts_per_face, 1>;
  // matrix of shape (dim - 1) * no_support_vertices
  using der_t =
      Eigen::Matrix<double, dim - 1, face_quad_info<dim>::quad_pts_per_face>;

  std::vector<element<dim>> m_elements;
  std::vector<vec> m_local_gp_pos;
  std::vector<double> m_local_gp_weight;

 public:
  gauss_face_int(std::vector<element<dim>> elements);

  void update_face_gauss_points(particle<dim> **particles,
                                particle<dim> **gauss_points,
                                unsigned orientation);
};

#endif /* GAUSS_INTEGRATION_H_ */

extern template class gauss_face_int<2>;
extern template class gauss_face_int<3>;
