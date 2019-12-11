#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <assert.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>

#include "kernel.h"
#include "particle.h"

template <int dim>
class utilities {
 public:
  using mat_low = Eigen::Matrix<double, dim, dim>;
  using mat_high = Eigen::Matrix<double, dim + 1, dim + 1>;
  using vec_low = Eigen::Matrix<double, dim, 1>;
  using vec_high = Eigen::Matrix<double, dim + 1, 1>;

  // find neighbors for quad points
  static void find_neighbors(particle<dim> **particles, unsigned int num_part,
                             particle<dim> **gauss_points, unsigned int num_gp,
                             double h);
  // find neighbors for particle points
  static void find_neighbors(particle<dim> **particles, unsigned int num_part,
                             double h);
  // compute rkpm shape functions at particles
  static void precomp_rkpm(particle<dim> **particles, unsigned int n);
  // compute rkpm shape functions at quad points
  static void precomp_rkpm(particle<dim> **from, particle<dim> **to,
                           unsigned int nto);
  // output
  static void vtk_write_particle(particle<dim> **particles,
                                 unsigned int num_part,
                                 unsigned int step,
                                 std::string prefix = "out");
  static void vtk_write_face_quad_points(particle<dim> **face_quad_points,
                                         unsigned int num_face_quad_points,
                                         unsigned int step);
  static void vtk_write_quad_points(particle<dim> **quad_points,
                                    unsigned int num_quad_points,
                                    unsigned int step);
  // debug
  static void print_particle_kernel(particle<dim> **particles,
                                    unsigned int num_part);
  static void print_quad_points_kernel(particle<dim> **quad_points,
                                       unsigned int num_quad_points);
  static void print_face_quad_points_kernel(particle<dim> **face_quad_points,
                                            unsigned int num_face_quad_points);

 private:
  // helper functions for precomp_rkpm
  static kernel_result<dim> cubic_spline(Eigen::Matrix<double, dim, 1> xi,
                                         Eigen::Matrix<double, dim, 1> xj,
                                         double h);
  static void compute_rkpm_correctors(particle<dim> *pi,
                                      particle<dim> **particles, mat_high &C);
  static kernel_result<dim> compute_rkpm_shape_function(particle<dim> *pi,
                                                        particle<dim> *pj,
                                                        mat_high C);
};

#endif

extern template class utilities<2>;
extern template class utilities<3>;
