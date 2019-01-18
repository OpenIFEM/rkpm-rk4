#include "utilities.h"

//----------------------------------------------------------------------

template <int dim>
kernel_result<dim> utilities<dim>::cubic_spline(
    Eigen::Matrix<double, dim, 1> xi, Eigen::Matrix<double, dim, 1> xj,
    double h) {
  double h1 = 1. / h;
  double rij = (xi - xj).norm();

  double fac = 10 * (M_1_PI) / 7.0 * h1 * h1;
  double q = rij * h1;

  kernel_result<dim> w;

  if (q < 2.) {
    if (q >= 1.) {
      w.w = fac * (0.25 * (2 - q) * (2 - q) * (2 - q));
      if (rij > 1e-12) {
        double der = -0.75 * (2 - q) * (2 - q) * h1 / rij;
        w.grad_w = (xi - xj) * der * fac;
      }
    } else {
      w.w = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
      if (rij > 1e-12) {
        double der = -3.0 * q * (1 - 0.75 * q) * h1 / rij;
        w.grad_w = (xi - xj) * der * fac;
      }
    }
  }

  return w;
}

template <int dim>
void utilities<dim>::compute_rkpm_correctors(particle<dim> *pi,
                                             particle<dim> **particles,
                                             mat_high &C) {
  vec_low xi = pi->x;
  double hi = pi->h;
  unsigned int num_nbh = pi->num_nbh;

  mat_high M(mat_high::Zero());
  std::vector<mat_high> M_der(dim, mat_high::Zero());

  for (unsigned int j = 0; j < num_nbh; j++) {
    unsigned int jdx = pi->nbh[j];

    particle<dim> *pj = particles[jdx];

    double mj = pj->m;
    double rhoj = pj->rho;

    vec_low xj = pj->x;

    vec_low r = xi - xj;
    mat_low grad_r(mat_low::Identity());

    kernel_result<dim> w = cubic_spline(xi, xj, hi);

    /*
     * Moment matrix, symmetric
     * |w.w*mj/rhoj   w.w*mj/rhoj*r^T       |
     * |w.w*mj/rhoj*r w.w*mj/rhoj*r\cross{r}|
     *
     * Boy, I love Eigen...
     */
    M(0, 0) += w.w * mj / rhoj;
    M.block(0, 1, 1, dim) += r.transpose() * w.w * mj / rhoj;
    M.block(1, 0, dim, 1) += r * w.w * mj / rhoj;
    M.block(1, 1, dim, dim) += r * r.transpose() * w.w * mj / rhoj;

    // M_der[n] denotes \partial{M}/\partial{x_n}
    for (unsigned int n = 0; n < dim; ++n) {
      M_der[n](0, 0) += w.grad_w[n] * mj / rhoj;
      M_der[n].block(0, 1, 1, dim) +=
          (w.grad_w(n) * r.transpose() + w.w * grad_r.row(n)) * mj / rhoj;
      M_der[n].block(1, 0, dim, 1) +=
          (w.grad_w(n) * r + w.w * grad_r.col(n)) * mj / rhoj;
      M_der[n].block(1, 1, dim, dim) +=
          (r * r.transpose() * w.grad_w(n) +
           w.w * (grad_r.col(n) * r.transpose() + r * grad_r.row(n))) *
          mj / rhoj;
    }
  }

  vec_high P(vec_high::Zero());
  P[0] = 1.;
  // Since the system is at most 4x4, it makes sense to solve for inverse
  assert(M.determinant() > 0.);
  mat_high M_inv = M.inverse();
  // Solve for C
  C.col(0) = M_inv * P;
  for (unsigned int n = 0; n < dim; ++n) {
    vec_high rhs = -M_der[n] * C.col(0);
    C.col(n + 1) = M_inv * rhs;
  }
}

// W_bar = phi * W where W_bar is rkpm kernel and W is SPH kernel
// phi = C_0 + C_1 \cdot (xi - xj) where C_1 and (xi - xj) are dim-dimensional
// vector
template <int dim>
kernel_result<dim> utilities<dim>::compute_rkpm_shape_function(
    particle<dim> *pi, particle<dim> *pj, mat_high C) {
  vec_low xi = pi->x;
  double hi = pi->h;
  vec_low xj = pj->x;
  vec_low r = xi - xj;
  mat_low grad_r(mat_low::Identity());
  kernel_result<dim> w = cubic_spline(xi, xj, hi);

  // Correction to shape function value
  vec_low C1 = C.block(1, 0, dim, 1);
  double corr = r.transpose() * C1 + C(0, 0);
  // Its derivatives
  vec_low grad_corr = C.block(1, 1, dim, dim).transpose() * r +
                      grad_r * C.block(1, 0, dim, 1) +
                      C.block(0, 1, 1, dim).transpose();

  kernel_result<dim> N;
  N.w = w.w * corr;
  N.grad_w = grad_corr * w.w + w.grad_w * corr;

  return N;
}

template <int dim>
void utilities<dim>::precomp_rkpm(particle<dim> **particles, unsigned int n) {
  for (unsigned int i = 0; i < n; i++) {
    particle<dim> *pi = particles[i];

    mat_high C(mat_high::Zero());
    compute_rkpm_correctors(pi, particles, C);

    for (unsigned int j = 0; j < pi->num_nbh; j++) {
      unsigned int jdx = pi->nbh[j];
      particle<dim> *pj = particles[jdx];

      pi->w[j] = compute_rkpm_shape_function(pi, pj, C);
    }
  }
}

template <int dim>
void utilities<dim>::precomp_rkpm(particle<dim> **from, particle<dim> **to,
                                  unsigned int nto) {
  for (unsigned int i = 0; i < nto; i++) {
    particle<dim> *pi = to[i];

    mat_high C(mat_high::Zero());
    compute_rkpm_correctors(pi, from, C);

    unsigned int num_nbh = pi->num_nbh;
    for (unsigned int j = 0; j < num_nbh; j++) {
      unsigned int jdx = pi->nbh[j];
      particle<dim> *pj = from[jdx];

      pi->w[j] = compute_rkpm_shape_function(pi, pj, C);
    }
  }
}

template <int dim>
void utilities<dim>::find_neighbors(particle<dim> **particles,
                                    unsigned int num_part,
                                    particle<dim> **gauss_points,
                                    unsigned int num_gp, double h) {
  for (unsigned int i = 0; i < num_gp; i++) {
    particle<dim> *p1 = gauss_points[i];
    unsigned int nbh_iter = 0;

    double hi = h;

    for (unsigned int j = 0; j < num_part; j++) {
      particle<dim> *p2 = particles[j];
      if ((p1->x - p2->x).norm() < 2 * hi) {
        gauss_points[i]->nbh[nbh_iter] = j;
        nbh_iter++;
      }
    }

    assert(nbh_iter < MAX_NBH);
    assert(nbh_iter > 0);

    gauss_points[i]->num_nbh = nbh_iter;
  }
}

template <int dim>
void utilities<dim>::find_neighbors(particle<dim> **particles,
                                    unsigned int num_part, double h) {
  for (unsigned int i = 0; i < num_part; i++) {
    particle<dim> *p1 = particles[i];
    unsigned int nbh_iter = 0;

    double hi = h;

    for (unsigned int j = 0; j < num_part; j++) {
      particle<dim> *p2 = particles[j];
      if ((p1->x - p2->x).norm() < 2 * hi) {
        particles[i]->nbh[nbh_iter] = j;
        nbh_iter++;
      }
    }

    assert(nbh_iter < MAX_NBH);
    assert(nbh_iter > 0);

    particles[i]->num_nbh = nbh_iter;
  }
}

//----------------------------------------------------------------------

template <int dim>
void utilities<dim>::print_particle_kernel(particle<dim> **particles,
                                           unsigned int num_part) {
  FILE *fp = fopen("particle_kernel.txt", "w+");
  // particle id, neighbor id, w, grad_w
  for (unsigned int i = 0; i < num_part; ++i) {
    for (unsigned int j = 0; j < particles[i]->num_nbh; ++j) {
      fprintf(fp, "%4d %4d %6.3f", i, particles[i]->nbh[j],
              particles[i]->w[j].w);
      for (unsigned int d = 0; d < dim; ++d) {
        fprintf(fp, " %6.3f", particles[i]->w[j].grad_w[d]);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

template <int dim>
void utilities<dim>::print_quad_points_kernel(particle<dim> **quad_points,
                                              unsigned int num_quad_points) {
  FILE *fp = fopen("volume_kernel.txt", "w+");
  // particle id, neighbor id, w, grad_w
  for (unsigned int i = 0; i < num_quad_points; ++i) {
    for (unsigned int j = 0; j < quad_points[i]->num_nbh; ++j) {
      fprintf(fp, "%4d %4d %6.3f", i, quad_points[i]->nbh[j],
              quad_points[i]->w[j].w);
      for (unsigned int d = 0; d < dim; ++d) {
        fprintf(fp, " %6.3f", quad_points[i]->w[j].grad_w[d]);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

template <int dim>
void utilities<dim>::print_face_quad_points_kernel(
    particle<dim> **face_quad_points, unsigned int num_face_quad_points) {
  FILE *fp = fopen("face_kernel.txt", "w+");
  // particle id, neighbor id, w, grad_w
  for (unsigned int i = 0; i < num_face_quad_points; ++i) {
    for (unsigned int j = 0; j < face_quad_points[i]->num_nbh; ++j) {
      fprintf(fp, "%4d %4d %6.3f", i, face_quad_points[i]->nbh[j],
              face_quad_points[i]->w[j].w);
      for (unsigned int d = 0; d < dim; ++d) {
        fprintf(fp, " %6.3f", face_quad_points[i]->w[j].grad_w[d]);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

//----------------------------------------------------------------------

template <int dim>
void utilities<dim>::vtk_write_particle(particle<dim> **particles,
                                        unsigned int num_part,
                                        unsigned int step) {
  char buf[256];
  sprintf(buf, "out_%06d.vtk", step);
  FILE *fp = fopen(buf, "w+");

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "rkpm-rk4\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "\n");

  fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(fp, "POINTS %d float\n", num_part);
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f %f %f\n", particles[i]->x[0], particles[i]->x[1],
            (dim == 3 ? particles[i]->x[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELLS %d %d\n", num_part, 2 * num_part);
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%d %d\n", 1, i);
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELL_TYPES %d\n", num_part);
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%d\n", 1);
  }
  fprintf(fp, "\n");

  fprintf(fp, "POINT_DATA %d\n", num_part);
  fprintf(fp, "SCALARS h float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f\n", particles[i]->h);
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS density float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f\n", particles[i]->rho);
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS idx int 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%d\n", particles[i]->idx);
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS Sxx float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f\n", particles[i]->S(0, 0));
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS Sxy float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f\n", particles[i]->S(0, 1));
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS Syy float 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f\n", particles[i]->S(1, 1));
  }
  fprintf(fp, "\n");

  if (dim == 3) {
    fprintf(fp, "SCALARS Szz float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < num_part; i++) {
      fprintf(fp, "%f\n", particles[i]->S(2, 2));
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS Sxz float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < num_part; i++) {
      fprintf(fp, "%f\n", particles[i]->S(0, 2));
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS Syz float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (unsigned int i = 0; i < num_part; i++) {
      fprintf(fp, "%f\n", particles[i]->S(1, 2));
    }
    fprintf(fp, "\n");
  }

  fprintf(fp, "VECTORS velocity float\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f %f %f\n", particles[i]->v[0], particles[i]->v[1],
            (dim == 3 ? particles[i]->v[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "VECTORS displacement float\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f %f %f\n", particles[i]->x[0] - particles[i]->X[0],
            particles[i]->x[1] - particles[i]->X[1],
            (dim == 3 ? particles[i]->x[2] - particles[i]->X[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "VECTORS acceleration float\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f %f %f\n", particles[i]->a[0], particles[i]->a[1],
            (dim == 3 ? particles[i]->a[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "VECTORS fext float\n");
  for (unsigned int i = 0; i < num_part; i++) {
    fprintf(fp, "%f %f %f\n", particles[i]->t[0], particles[i]->t[1],
            (dim == 3 ? particles[i]->t[2] : 0.));
  }
  fprintf(fp, "\n");

  fclose(fp);
}

template <int dim>
void utilities<dim>::vtk_write_face_quad_points(
    particle<dim> **face_quad_points, unsigned int num_face_quad_points,
    unsigned int step) {
  char buf[256];
  sprintf(buf, "face_quad_%06d.vtk", step);
  FILE *fp = fopen(buf, "w+");

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "rkpm-rk4\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "\n");

  fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(fp, "POINTS %d float\n", num_face_quad_points);
  for (unsigned int i = 0; i < num_face_quad_points; i++) {
    fprintf(fp, "%f %f %f\n", face_quad_points[i]->x[0],
            face_quad_points[i]->x[1],
            (dim == 3 ? face_quad_points[i]->x[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELLS %d %d\n", num_face_quad_points, 2 * num_face_quad_points);
  for (unsigned int i = 0; i < num_face_quad_points; i++) {
    fprintf(fp, "%d %d\n", 1, i);
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELL_TYPES %d\n", num_face_quad_points);
  for (unsigned int i = 0; i < num_face_quad_points; i++) {
    fprintf(fp, "%d\n", 1);
  }
  fprintf(fp, "\n");

  fprintf(fp, "POINT_DATA %d\n", num_face_quad_points);
  fprintf(fp, "VECTORS traction float\n");
  for (unsigned int i = 0; i < num_face_quad_points; i++) {
    fprintf(fp, "%f %f %f\n", face_quad_points[i]->t[0],
            face_quad_points[i]->t[1],
            (dim == 3 ? face_quad_points[i]->t[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS idx int 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_face_quad_points; i++) {
    fprintf(fp, "%d\n", face_quad_points[i]->idx);
  }
  fprintf(fp, "\n");

  fclose(fp);
}

template <int dim>
void utilities<dim>::vtk_write_quad_points(particle<dim> **quad_points,
                                           unsigned int num_quad_points,
                                           unsigned int step) {
  char buf[256];
  sprintf(buf, "volume_quad_%06d.vtk", step);
  FILE *fp = fopen(buf, "w+");

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "rkpm-rk4\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "\n");

  fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(fp, "POINTS %d float\n", num_quad_points);
  for (unsigned int i = 0; i < num_quad_points; i++) {
    fprintf(fp, "%f %f %f\n", quad_points[i]->x[0], quad_points[i]->x[1],
            (dim == 3 ? quad_points[i]->x[2] : 0.));
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELLS %d %d\n", num_quad_points, 2 * num_quad_points);
  for (unsigned int i = 0; i < num_quad_points; i++) {
    fprintf(fp, "%d %d\n", 1, i);
  }
  fprintf(fp, "\n");

  fprintf(fp, "CELL_TYPES %d\n", num_quad_points);
  for (unsigned int i = 0; i < num_quad_points; i++) {
    fprintf(fp, "%d\n", 1);
  }
  fprintf(fp, "\n");

  fprintf(fp, "POINT_DATA %d\n", num_quad_points);
  fprintf(fp, "SCALARS idx int 1\n");
  fprintf(fp, "LOOKUP_TABLE default\n");
  for (unsigned int i = 0; i < num_quad_points; i++) {
    fprintf(fp, "%d\n", quad_points[i]->idx);
  }
  fprintf(fp, "\n");
  fclose(fp);
}

template class utilities<2>;
template class utilities<3>;
