#include "gauss_integration.h"

namespace {
void shape_function(
    Eigen::Matrix<double, 2, 1> xi,
    Eigen::Matrix<double, vol_quad_info<2>::quad_pts_per_cell, 1> &N,
    Eigen::Matrix<double, 2, vol_quad_info<2>::quad_pts_per_cell> &N_xi) {
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

void shape_function(
    Eigen::Matrix<double, 3, 1> xi,
    Eigen::Matrix<double, vol_quad_info<3>::quad_pts_per_cell, 1> &N,
    Eigen::Matrix<double, 3, vol_quad_info<3>::quad_pts_per_cell> &N_xi) {
  N[0] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 - xi[2]);
  N[1] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[2]);
  N[2] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[2]);
  N[3] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 - xi[2]);
  N[4] = 0.125 * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[2]);
  N[5] = 0.125 * (1 + xi[0]) * (1 - xi[1]) * (1 + xi[2]);
  N[6] = 0.125 * (1 + xi[0]) * (1 + xi[1]) * (1 + xi[2]);
  N[7] = 0.125 * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[2]);

  N_xi(0, 0) = -0.125 * (1 - xi[1]) * (1 - xi[2]);
  N_xi(1, 0) = -0.125 * (1 - xi[0]) * (1 - xi[2]);
  N_xi(2, 0) = -0.125 * (1 - xi[0]) * (1 - xi[1]);
  N_xi(0, 1) = 0.125 * (1 - xi[1]) * (1 - xi[2]);
  N_xi(1, 1) = -0.125 * (1 + xi[0]) * (1 - xi[2]);
  N_xi(2, 1) = -0.125 * (1 + xi[0]) * (1 - xi[1]);
  N_xi(0, 2) = 0.125 * (1 + xi[1]) * (1 - xi[2]);
  N_xi(1, 2) = 0.125 * (1 + xi[0]) * (1 - xi[2]);
  N_xi(2, 2) = -0.125 * (1 + xi[0]) * (1 + xi[1]);
  N_xi(0, 3) = -0.125 * (1 + xi[1]) * (1 - xi[2]);
  N_xi(1, 3) = 0.125 * (1 - xi[0]) * (1 - xi[2]);
  N_xi(2, 3) = -0.125 * (1 - xi[0]) * (1 + xi[1]);
  N_xi(0, 4) = -0.125 * (1 - xi[1]) * (1 + xi[2]);
  N_xi(1, 4) = -0.125 * (1 - xi[0]) * (1 + xi[2]);
  N_xi(2, 4) = 0.125 * (1 - xi[0]) * (1 - xi[1]);
  N_xi(0, 5) = 0.125 * (1 - xi[1]) * (1 + xi[2]);
  N_xi(1, 5) = -0.125 * (1 + xi[0]) * (1 + xi[2]);
  N_xi(2, 5) = 0.125 * (1 + xi[0]) * (1 - xi[1]);
  N_xi(0, 6) = 0.125 * (1 + xi[1]) * (1 + xi[2]);
  N_xi(1, 6) = 0.125 * (1 + xi[0]) * (1 + xi[2]);
  N_xi(2, 6) = 0.125 * (1 + xi[0]) * (1 + xi[1]);
  N_xi(0, 7) = -0.125 * (1 + xi[1]) * (1 + xi[2]);
  N_xi(1, 7) = 0.125 * (1 - xi[0]) * (1 + xi[2]);
  N_xi(2, 7) = 0.125 * (1 - xi[0]) * (1 + xi[1]);
}

void get_weight_pos(std::vector<Eigen::Matrix<double, 2, 1>> &pt,
                    std::vector<double> &w) {
  pt.resize(4);
  w.resize(4);
  pt[0] << -0.577350269189626, -0.577350269189626;
  pt[1] << 0.577350269189626, -0.577350269189626;
  pt[2] << 0.577350269189626, 0.577350269189626;
  pt[3] << -0.577350269189626, 0.577350269189626;
  w[0] = 1.;
  w[1] = 1.;
  w[2] = 1.;
  w[3] = 1.;
}

void get_weight_pos(std::vector<Eigen::Matrix<double, 3, 1>> &pt,
                    std::vector<double> &w) {
  pt.resize(8);
  w.resize(8);
  pt[0] << -0.577350269189626, -0.577350269189626, -0.577350269189626;
  pt[1] << 0.577350269189626, -0.577350269189626, -0.577350269189626;
  pt[2] << 0.577350269189626, 0.577350269189626, -0.577350269189626;
  pt[3] << -0.577350269189626, 0.577350269189626, -0.577350269189626;
  pt[4] << -0.577350269189626, -0.577350269189626, 0.577350269189626;
  pt[5] << 0.577350269189626, -0.577350269189626, 0.577350269189626;
  pt[6] << 0.577350269189626, 0.577350269189626, 0.577350269189626;
  pt[7] << -0.577350269189626, 0.577350269189626, 0.577350269189626;
  w[0] = 1.;
  w[1] = 1.;
  w[2] = 1.;
  w[3] = 1.;
  w[4] = 1.;
  w[5] = 1.;
  w[6] = 1.;
  w[7] = 1.;
}
}  // namespace

template <int dim>
void gauss_int<dim>::update_gauss_points(particle<dim> **particles,
                                         particle<dim> **gauss_points) {
  unsigned int global_gp_iter = 0;

  for (auto it = m_elements.begin(); it != m_elements.end(); ++it) {
    // Coordinates of nodes in current element, m_node(i, j) = x_i^j
    der_t m_node;
    for (unsigned int i = 0; i < it->nodes.size(); ++i) {
      m_node.col(i) = particles[it->nodes[i]]->x;
    }

    for (unsigned int i = 0; i < vol_quad_info<dim>::quad_pts_per_cell; i++) {
      vec local_pos = m_local_gp_pos[i];
      double local_w = m_local_gp_weight[i];

      val_t N;
      der_t N_xi;  // N_xi(i, j) = \partial{N_j}/\partial{xi_i}
      shape_function(local_pos, N, N_xi);

      // position
      vec global_pos = m_node * N;

      gauss_points[global_gp_iter]->x = global_pos;
      gauss_points[global_gp_iter]->X = global_pos;

      // x_xi(i, j) = m_node(j, k) * N_xi(i, k)
      mat x_xi = N_xi * m_node.transpose();

      // weight
      double J = x_xi.determinant();
      double global_gp_weigth = local_w * J;
      assert(global_gp_weigth > 0.);

      gauss_points[global_gp_iter]->quad_weight = global_gp_weigth;

      // function values
      mat xi_x = x_xi.inverse();
      // N_x(i, j) = xi_x(i, k) * N_xi(k, j)
      der_t N_x = xi_x * N_xi;

      gauss_points[global_gp_iter]->num_nbh = it->nodes.size();
      for (unsigned int I = 0; I < it->nodes.size(); I++) {
        gauss_points[global_gp_iter]->nbh[I] = it->nodes[I];
        gauss_points[global_gp_iter]->w[I].w = N[I];
        gauss_points[global_gp_iter]->w[I].grad_w = N_x.col(I);
      }

      global_gp_iter++;
    }
  }
}

template <int dim>
gauss_int<dim>::gauss_int(std::vector<element<dim>> elements)
    : m_elements(elements) {
  m_local_gp_pos.resize(vol_quad_info<dim>::quad_pts_per_cell);
  m_local_gp_weight.resize(vol_quad_info<dim>::quad_pts_per_cell, 0.);

  std::vector<vec> pt;
  std::vector<double> w;
  get_weight_pos(pt, w);

  for (unsigned int i = 0; i < vol_quad_info<dim>::quad_pts_per_cell; i++) {
    m_local_gp_pos[i] = pt[i];
    m_local_gp_weight[i] = w[i];
  }
}

template class gauss_int<2>;
template class gauss_int<3>;
