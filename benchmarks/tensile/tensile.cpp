#include "body.h"
#include "simulation_time.h"
#include "utilities.h"

static void left_bndry_fun_tl(particle<2> *p) {
  p->x[0] = p->X[0] - 10 * simulation_time::getInstance().get_time();
  p->x[1] = p->X[1];

  p->v[0] = -10.;
  p->v[1] = 0.;

  p->previous_v[0] = -10.;
  p->previous_v[1] = 0.;
  p->a[0] = 0.;
  p->a[1] = 0.;

  p->v_t[0] = 0.;
  p->v_t[1] = 0.;
}

static void right_bndry_fun_tl(particle<2> *p) {
  p->x[0] = p->X[0] + 10 * simulation_time::getInstance().get_time();
  p->x[1] = p->X[1];

  p->v[0] = 10.;
  p->v[1] = 0.;

  p->previous_v[0] = 10.;
  p->previous_v[1] = 0.;
  p->a[0] = 0.;
  p->a[1] = 0.;

  p->v_t[0] = 0.;
  p->v_t[1] = 0.;
}

std::shared_ptr<body<2>> tensile_tl_weak(unsigned int nbox) {
  // material constants
  double E = 1e7;
  double nu = 0.4;
  double rho0 = 1.;

  // problem dimensions
  double L = 1.;

  double dx = L / (nbox - 1);
  double hdx = 1.3;

  double dt = 1e-6;

  // n particles
  unsigned n = nbox * nbox;
  particle<2> **particles = new particle<2> *[n];

  std::vector<unsigned int> left_bndry;
  std::vector<unsigned int> right_bndry;

  unsigned int part_iter = 0;
  for (unsigned int i = 0; i < nbox; i++) {
    for (unsigned int j = 0; j < nbox; j++) {
      // Create particles and assign coordinates to them
      double px = i * dx;
      double py = j * dx;
      particle<2> *p = new particle<2>(part_iter);

      p->x[0] = px;
      p->x[1] = py;

      p->X[0] = px;
      p->X[1] = py;

      particles[part_iter] = p;

      // Push particle id to left_bndry if i = 0
      if (i == 0) {
        left_bndry.push_back(part_iter);
      }

      // Push particle id to left_bndry if i = nbox - 1
      if (i == nbox - 1) {
        right_bndry.push_back(part_iter);
      }
      part_iter++;
    }
  }

  // Create the elements: there are n particles, and (nbox-1)*(nbox-1) elements
  std::vector<element<2>> elements = element_map_rectangle(nbox - 1, nbox - 1);

  // Each element has 4 Gauss quadrature points
  unsigned num_gp = 2 * 2 * elements.size();
  // Gauss points are also particles
  particle<2> **gauss_points = new particle<2> *[num_gp];
  for (unsigned int i = 0; i < num_gp; i++) {
    gauss_points[i] = new particle<2>(i);
  }

  // Create a Gauss integration rule
  gauss_int<2> gi(elements);
  // Update gauss_points using particles so that it knows
  // the read coordintaes, derivatives etc.
  gi.update_gauss_points(particles, gauss_points);

  // Face integration points
  unsigned num_fgp = 4 * 2 * (nbox - 1);
  particle<2> **gauss_face_points = new particle<2> *[num_fgp];
  for (unsigned int i = 0; i < num_fgp; i++) {
    gauss_face_points[i] = new particle<2>(i);
  }
  std::vector<std::vector<element<2>>> boundaries =
      boundary_element_rectangle(nbox - 1, nbox - 1);
  for (unsigned int i = 0; i < 4; ++i) {
    gauss_face_int<2> gfi(boundaries[i]);
    gfi.update_face_gauss_points(particles, gauss_face_points + i * num_fgp / 4,
                                 i);
  }

  // bc keeps track of particles and the functions to be applied on them
  boundary_conditions<2> bc;
  bc.add_boundary_condition(left_bndry, &left_bndry_fun_tl);
  bc.add_boundary_condition(right_bndry, &right_bndry_fun_tl);

  for (unsigned int i = 0; i < n; i++) {
    particles[i]->rho = rho0;
    particles[i]->h = hdx * dx;
    particles[i]->m = dx * dx * rho0;
    particles[i]->quad_weight = particles[i]->m / particles[i]->rho;
  }

  for (unsigned int i = 0; i < num_gp; i++) {
    gauss_points[i]->h = hdx * dx;
    gauss_points[i]->rho = rho0;
  }

  for (unsigned int i = 0; i < num_fgp; i++) {
    gauss_face_points[i]->h = hdx * dx;
    gauss_face_points[i]->rho = rho0;
  }

  utilities<2>::find_neighbors(particles, n, hdx * dx);
  utilities<2>::find_neighbors(particles, n, gauss_points, num_gp, hdx * dx);
  utilities<2>::find_neighbors(particles, n, gauss_face_points, num_fgp,
                               hdx * dx);

  utilities<2>::precomp_rkpm(particles, gauss_points, num_gp);
  utilities<2>::precomp_rkpm(particles, gauss_face_points, num_fgp);
  utilities<2>::precomp_rkpm(particles, n);

  physical_constants physical_constants(nu, E, rho0);
  simulation_data sim_data(
      physical_constants,
      correction_constants(constants_monaghan(),
                           constants_artificial_viscosity(), 0, true));

  auto b =
      std::make_shared<body<2>>(particles, n, sim_data, dt, bc, gauss_points,
                                num_gp, gauss_face_points, num_fgp);

  // This specific test contains only one body, which has n particles
  return b;
}

int main() {
  auto body = tensile_tl_weak(21);

  simulation_time *time = &simulation_time::getInstance();

  unsigned int step = 0;
  unsigned int freq = 100;

  while (step < 10001) {
    body->step();

    if (step % freq == 0) {
      utilities<2>::vtk_write_particle(body->get_particles(),
                                       body->get_num_part(), step);
    }

    printf("%d: %f\n", step, time->get_time());
    step++;
  }

  return EXIT_SUCCESS;
}
