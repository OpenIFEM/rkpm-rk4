#include "gauss_face_integration.h"

namespace {
void face_shape_function(
    Eigen::Matrix<double, 1, 1> xi,
    Eigen::Matrix<double, face_quad_info<2>::quad_pts_per_face, 1> &N,
    Eigen::Matrix<double, 1, face_quad_info<2>::quad_pts_per_face> &N_xi) {
  N[0] = 0.5 * (1 - xi[0]);
  N[1] = 0.5 * (1 + xi[0]);
  N_xi[0] = -0.5;
  N_xi[1] = 0.5;
}

void face_shape_function(
    Eigen::Matrix<double, 2, 1> xi,
    Eigen::Matrix<double, face_quad_info<3>::quad_pts_per_face, 1> &N,
    Eigen::Matrix<double, 2, face_quad_info<3>::quad_pts_per_face> &N_xi) {
  N[0] = 0.25 * (1 - xi[0]) * (1 - xi[1]);
  N[1] = 0.25 * (1 + xi[0]) * (1 - xi[1]);
  N[2] = 0.25 * (1 + xi[0]) * (1 + xi[1]);
  N[3] = 0.25 * (1 - xi[0]) * (1 + xi[1]);
  N_xi(0, 0) = -0.25 * (1 - xi[1]);
  N_xi(1, 0) = -0.25 * (1 - xi[0]);
  N_xi(0, 1) = 0.25 * (1 - xi[1]);
  N_xi(1, 1) = -0.25 * (1 + xi[0]);
  N_xi(0, 2) = 0.25 * (1 + xi[1]);
  N_xi(1, 2) = 0.25 * (1 + xi[0]);
  N_xi(0, 3) = -0.25 * (1 + xi[1]);
  N_xi(1, 3) = 0.25 * (1 - xi[0]);
}

void get_face_weight_pos(std::vector<Eigen::Matrix<double, 1, 1>> &p,
                         std::vector<double> &w) {
  p.resize(2);
  w.resize(2, 0.);
  p[0] << -0.577350269189626;
  p[1] << 0.577350269189626;
  w[0] = 1.0;
  w[1] = 1.0;
}

void get_face_weight_pos(std::vector<Eigen::Matrix<double, 2, 1>> &p,
                         std::vector<double> &w) {
  p.resize(4);
  w.resize(4, 0.);
  p[0] << -0.577350269189626, -0.577350269189626;
  p[1] << 0.577350269189626, -0.577350269189626;
  p[2] << 0.577350269189626, 0.577350269189626;
  p[3] << -0.577350269189626, 0.577350269189626;
  w[0] = 1.0;
  w[1] = 1.0;
  w[2] = 1.0;
  w[3] = 1.0;
}

template <int dim>
double compute_J(Eigen::Matrix<double, dim - 1, dim> x_xi) {
  static_assert(dim == 2 || dim == 3, "Not implemented!");
  return 0;
}

template <>
double compute_J(Eigen::Matrix<double, 2, 3> x_xi) {
  return (x_xi.row(0).cross(x_xi.row(1))).norm();
}

template <>
double compute_J(Eigen::Matrix<double, 1, 2> x_xi) {
  return x_xi.norm();
}
}  // namespace

template <int dim>
void gauss_face_int<dim>::update_face_gauss_points(particle<dim> **particles,
                                                   particle<dim> **gauss_points,
                                                   unsigned int orientation) {
  unsigned int global_gp_iter = 0;

  for (auto it = m_elements.begin(); it != m_elements.end(); ++it) {
    // Coordinates of supporting nodes, shape is different from N_xi
    mat m_node;
    for (unsigned int i = 0; i < face_quad_info<dim>::quad_pts_per_face; ++i)
      m_node.col(i) =
          particles[it->nodes[face_quad_info<dim>::face_nodes[orientation][i]]]
              ->x;

    for (unsigned int i = 0; i < face_quad_info<dim>::quad_pts_per_face; i++) {
      vec local_pos = m_local_gp_pos[i];
      double local_w = m_local_gp_weight[i];

      val_t N;
      der_t N_xi;  // N_xi(i, j) = \partial{N_j}/\partial{xi_i}
      face_shape_function(local_pos, N, N_xi);

      // position
      Eigen::Matrix<double, dim, 1> global_pos = m_node * N;

      gauss_points[global_gp_iter]->x = global_pos;
      gauss_points[global_gp_iter]->X = global_pos;

      // x_xi(i, j) = m_node(j, k) * N_xi(i, k)
      Eigen::Matrix<double, dim - 1, dim> x_xi = N_xi * m_node.transpose();

      // weight
      double J = compute_J(x_xi);
      double global_gp_weigth = local_w * J;
      assert(global_gp_weigth > 0.);

      gauss_points[global_gp_iter]->quad_weight = global_gp_weigth;

      // Jacobian x_xi is non-square, it is not invertable.
      // However, it has a right inverse: (A^T * A)^-1 * A^T
      Eigen::Matrix<double, dim, dim - 1> xi_x =
          x_xi.transpose() * (x_xi * x_xi.transpose()).inverse();
      // N_x(i, j) = xi_x(i, k) * N_xi(k, j)
      Eigen::Matrix<double, dim, face_quad_info<dim>::quad_pts_per_face> N_x =
          xi_x * N_xi;

      gauss_points[global_gp_iter]->num_nbh =
          face_quad_info<dim>::quad_pts_per_face;
      for (unsigned int I = 0; I < face_quad_info<dim>::quad_pts_per_face;
           I++) {
        gauss_points[global_gp_iter]->nbh[I] =
            it->nodes[face_quad_info<dim>::face_nodes[orientation][I]];
        gauss_points[global_gp_iter]->w[I].w = N(I);
        gauss_points[global_gp_iter]->w[I].grad_w = N_x.col(I);
      }

      global_gp_iter++;
    }
  }
}

template <int dim>
gauss_face_int<dim>::gauss_face_int(std::vector<element<dim>> elements)
    : m_elements(elements) {
  m_local_gp_pos.resize(face_quad_info<dim>::quad_pts_per_face);
  m_local_gp_weight.resize(face_quad_info<dim>::quad_pts_per_face, 0.);

  std::vector<vec> pt;
  std::vector<double> w;
  get_face_weight_pos(pt, w);

  for (unsigned int i = 0; i < face_quad_info<dim>::quad_pts_per_face; i++) {
    m_local_gp_pos[i] = pt[i];
    m_local_gp_weight[i] = w[i];
  }
}

template class gauss_face_int<2>;
template class gauss_face_int<3>;
