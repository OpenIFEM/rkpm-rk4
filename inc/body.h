#ifndef BODY_H_
#define BODY_H_

#include <stdio.h>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include "boundary_conditions.h"
#include "gauss_face_integration.h"
#include "gauss_integration.h"
#include "particle.h"
#include "simulation_data.h"
#include "simulation_time.h"

template <int dim>
class body {
 private:
  class rk4;
  struct action;
  struct derive_quad_coordinates;
  struct derive_face_quad_coordinates;
  struct derive_quad_deformation;
  struct derive_quad_velocity_gradient_ref;
  struct derive_quad_velocity_gradient_cur;
  struct derive_quad_density;
  struct derive_quad_pressure;
  struct derive_quad_jaumann_stress_rate;
  struct derive_quad_nominal_stress;
  struct derive_part_nominal_stress_divergence;
  struct derive_part_cauchy_stress;
  struct derive_part_traction;
  struct derive_part_rhs;
  struct derive_part_coordinate_derivative;

  simulation_data m_data;
  boundary_conditions<dim> m_bc;

  unsigned int m_num_part;
  particle<dim>** m_particles;
  particle<dim>** m_cur_particles;

  unsigned int m_num_quad;
  particle<dim>** m_quad_points;
  particle<dim>** m_cur_quad_points;

  unsigned int m_num_face_quad;
  particle<dim>** m_face_quad_points;
  particle<dim>** m_cur_face_quad_points;

  std::unique_ptr<rk4> m_time_stepper;

  // actions on body
  std::vector<std::unique_ptr<action>> m_actions;

  double m_c = 0.0;  // damping

  void compute_time_derivatives(particle<dim>** particles,
                                unsigned int num_part,
                                particle<dim>** quad_points,
                                unsigned int num_quad,
                                particle<dim>** face_quad_points = 0,
                                unsigned int num_face_quad = 0);

 public:
  ~body();

  void smooth_vel();

  simulation_data get_sim_data() const { return m_data; }

  particle<dim>** get_cur_particles() const { return m_cur_particles; }

  particle<dim>** get_particles() const { return m_particles; }

  unsigned int get_num_part() const { return m_num_part; }

  particle<dim>** get_cur_quad_points() const { return m_cur_quad_points; }

  particle<dim>** get_quad_points() const { return m_quad_points; }

  unsigned int get_num_quad_points() const { return m_num_quad; }

  particle<dim>** get_face_quad_points() const { return m_face_quad_points; }

  particle<dim>** get_cur_face_quad_points() const {
    return m_cur_face_quad_points;
  }

  unsigned int get_num_face_quad_points() const { return m_num_face_quad; }

  double get_damping() const { return m_c; }

  body(particle<dim>** particles, unsigned int n, simulation_data data,
       double dt, boundary_conditions<dim> bc = boundary_conditions<dim>(),
       particle<dim>** quad_points = nullptr, unsigned int nq = 0,
       particle<dim>** face_quad_points = nullptr, unsigned int nfq = 0,
       double damping = 0.);

  void step();
};

#endif /* BODY_H_ */

extern template class body<2>;
extern template class body<3>;
