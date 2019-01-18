#include "boundary_conditions.h"

template <int dim>
void boundary_conditions<dim>::add_boundary_condition(
    const std::vector<unsigned int> boundary_indexes,
    std::function<void(particle<dim> *)> boundary_fun) {
  m_boundary_conditions.push_back(
      std::tuple<std::vector<unsigned int>,
                 std::function<void(particle<dim> *)>>(boundary_indexes,
                                                       boundary_fun));
}

template <int dim>
void boundary_conditions<dim>::add_neumann_boundary_condition(
    const std::vector<unsigned int> boundary_indexes,
    std::function<void(particle<dim> *)> boundary_fun) {
  m_neumann_boundary_conditions.push_back(
      std::tuple<std::vector<unsigned int>,
                 std::function<void(particle<dim> *)>>(boundary_indexes,
                                                       boundary_fun));
}

template <int dim>
void boundary_conditions<dim>::carry_out_boundary_conditions(
    particle<dim> **particles, unsigned int np) const {
  for (auto it = m_boundary_conditions.begin();
       it != m_boundary_conditions.end(); ++it) {
    std::vector<unsigned int> cur_indices = std::get<0>(*it);
    auto cur_bc = std::get<1>(*it);

    for (auto jt = cur_indices.begin(); jt != cur_indices.end(); ++jt) {
      unsigned int idx = *jt;
      cur_bc(particles[idx]);
    }
  }
}

template <int dim>
void boundary_conditions<dim>::carry_out_neumann_boundary_conditions(
    particle<dim> **particles, unsigned int np) const {
  for (auto it = m_neumann_boundary_conditions.begin();
       it != m_neumann_boundary_conditions.end(); ++it) {
    std::vector<unsigned int> cur_indices = std::get<0>(*it);
    auto cur_bc = std::get<1>(*it);

    for (auto jt = cur_indices.begin(); jt != cur_indices.end(); ++jt) {
      unsigned int idx = *jt;
      cur_bc(particles[idx]);
    }
  }
}

template class boundary_conditions<2>;
template class boundary_conditions<3>;
