#ifndef BOUNDARY_CONDITIONS_H_
#define BOUNDARY_CONDITIONS_H_

#include <functional>
#include <tuple>
#include <vector>

#include "particle.h"

template <int dim>
class boundary_conditions {
 public:
  boundary_conditions(){};

  void add_boundary_condition(
      const std::vector<unsigned int> boundary_indexes,
      std::function<void(particle<dim> *)> boundary_fun);

  void add_neumann_boundary_condition(
      const std::vector<unsigned int> boundary_indexes,
      std::function<void(particle<dim> *)> boundary_fun);

  void carry_out_boundary_conditions(particle<dim> **particles,
                                     unsigned int np) const;

  void carry_out_neumann_boundary_conditions(particle<dim> **particles,
                                             unsigned int np) const;

 private:
  std::vector<std::tuple<std::vector<unsigned int>,
                         std::function<void(particle<dim> *)>>>
      m_boundary_conditions;
  std::vector<std::tuple<std::vector<unsigned int>,
                         std::function<void(particle<dim> *)>>>
      m_neumann_boundary_conditions;
};

#endif /* BOUNDARY_CONDITIONS_H_ */

extern template class boundary_conditions<2>;
extern template class boundary_conditions<3>;
