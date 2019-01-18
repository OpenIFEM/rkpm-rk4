#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <assert.h>
#include <string.h>
#include <Eigen/Dense>
#include <type_traits>
#include "kernel.h"

#define MAX_NBH 200

template <int dim>
class particle {
 public:
  using vec = Eigen::Matrix<double, dim, 1>;    // column vector
  using mat = Eigen::Matrix<double, dim, dim>;  // square matrix
  particle();
  particle(unsigned int idx);
  virtual ~particle();

  virtual void reset();
  virtual void copy_into(particle *p) const;

  unsigned int idx = 0, hash = 0;
  double m = 0., rho = 0., h = 0., quad_weight = 0., p = 0.;
  double rho_t = 0.;

  vec X, x;  // original and current coordinates
  vec v;     // velocity
  vec x_t, v_t;
  vec previous_v;
  vec a, t, n;
  vec div_P;  // stress divergence

  mat S;       // stress
  mat grad_v;  // grad_v[i][j]: dvjdxi
  mat S_t;
  mat H, Fdot, P;

  unsigned int num_nbh = 0.;
  unsigned int nbh[MAX_NBH];
  kernel_result<dim> w[MAX_NBH];
};

#endif /* PARTICLE_H_ */

extern template class particle<2>;
extern template class particle<3>;
