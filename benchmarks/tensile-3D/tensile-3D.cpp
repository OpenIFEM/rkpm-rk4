#include "body.h"
#include "simulation_time.h"
#include "utilities.h"

static void left_bndry_fun_tl(particle<3> *p) {
  p->x[0] = p->X[0] + 10 * simulation_time::getInstance().get_time();
  p->x[1] = p->X[1];
  p->x[2] = p->X[2];

  p->v[0] = 10.;
  p->v[1] = 0.;
  p->v[2] = 0.;

  p->previous_v[0] = 10.;
  p->previous_v[1] = 0.;
  p->previous_v[2] = 0.;
  p->a[0] = 0.;
  p->a[1] = 0.;
  p->a[2] = 0.;

  p->v_t[0] = 0.;
  p->v_t[1] = 0.;
  p->v_t[2] = 0.;
}

static void right_bndry_fun_tl(particle<3> *p) {
  p->x[0] = p->X[0] - 10 * simulation_time::getInstance().get_time();
  p->x[1] = p->X[1];
  p->x[2] = p->X[2];

  p->v[0] = -10.;
  p->v[1] = 0.;
  p->v[2] = 0.;

  p->previous_v[0] = -10.;
  p->previous_v[1] = 0.;
  p->previous_v[2] = 0.;
  p->a[0] = 0.;
  p->a[1] = 0.;
  p->a[2] = 0.;

  p->v_t[0] = 0.;
  p->v_t[1] = 0.;
  p->v_t[2] = 0.;
}

std::shared_ptr<body<3>> tensile_tl_weak(unsigned int nbox) {
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
  unsigned n = nbox * nbox * nbox;
  particle<3> **particles = new particle<3> *[n];

  std::vector<unsigned int> left_bndry;
  std::vector<unsigned int> right_bndry;

  unsigned int part_iter = 0;
  for (unsigned int k = 0; k < nbox; k++) {
    for (unsigned int j = 0; j < nbox; j++) {
      for (unsigned int i = 0; i < nbox; i++) {
        // Create particles and assign coordinates to them
        double px = i * dx;
        double py = j * dx;
        double pz = k * dx;
        particle<3> *p = new particle<3>(part_iter);

        p->x[0] = px;
        p->x[1] = py;
        p->x[2] = pz;

        p->X[0] = px;
        p->X[1] = py;
        p->X[2] = pz;

        particles[part_iter] = p;

        if (i == 0) {
          right_bndry.push_back(part_iter);
        }
        if (i == nbox - 1) {
          left_bndry.push_back(part_iter);
        }

        part_iter++;
      }
    }
  }

  std::vector<element<3>> elements =
      element_map_rectangle(nbox - 1, nbox - 1, nbox - 1);

  // Each element has 4 Gauss quadrature points
  unsigned num_gp = 8 * elements.size();
  // Gauss points are also particles
  particle<3> **gauss_points = new particle<3> *[num_gp];
  for (unsigned int i = 0; i < num_gp; i++) {
    gauss_points[i] = new particle<3>(i);
  }

  // Create a Gauss integration rule
  gauss_int<3> gi(elements);
  // Update gauss_points using particles so that it knows
  // the read coordintaes, derivatives etc.
  gi.update_gauss_points(particles, gauss_points);

  // Face integration points
  unsigned num_fgp = 24 * (nbox - 1) * (nbox - 1);
  particle<3> **gauss_face_points = new particle<3> *[num_fgp];
  for (unsigned int i = 0; i < num_fgp; i++) {
    gauss_face_points[i] = new particle<3>(i);
  }
  std::vector<std::vector<element<3>>> boundaries =
      boundary_element_rectangle(nbox - 1, nbox - 1, nbox - 1);
  unsigned shift = 0;
  for (unsigned int i = 0; i < 6; ++i) {
    gauss_face_int<3> gfi(boundaries[i]);
    gfi.update_face_gauss_points(particles, gauss_face_points + shift, i);
    shift += 4 * (nbox - 1) * (nbox - 1);
  }

  // bc keeps track of particles and the functions to be applied on them
  boundary_conditions<3> bc;
  bc.add_boundary_condition(left_bndry, &left_bndry_fun_tl);
  bc.add_boundary_condition(right_bndry, &right_bndry_fun_tl);

  for (unsigned int i = 0; i < n; i++) {
    particles[i]->rho = rho0;
    particles[i]->h = hdx * dx;
    particles[i]->m = dx * dx * dx * rho0;
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

  utilities<3>::find_neighbors(particles, n, hdx * dx);
  utilities<3>::find_neighbors(particles, n, gauss_points, num_gp, hdx * dx);
  utilities<3>::find_neighbors(particles, n, gauss_face_points, num_fgp,
                               hdx * dx);

  utilities<3>::precomp_rkpm(particles, gauss_points, num_gp);
  utilities<3>::precomp_rkpm(particles, gauss_face_points, num_fgp);
  utilities<3>::precomp_rkpm(particles, n);

  physical_constants physical_constants(nu, E, rho0);
  simulation_data sim_data(
      physical_constants,
      correction_constants(constants_monaghan(),
                           constants_artificial_viscosity(), 0, true));

  auto b =
      std::make_shared<body<3>>(particles, n, sim_data, dt, bc, gauss_points,
                                num_gp, gauss_face_points, num_fgp);

  // This specific test contains only one body, which has n particles
  return b;
}

int main() {
  FILE *fp = fopen("thickness.txt", "w+");
  fprintf(fp, "time thickness\n");
  auto body = tensile_tl_weak(15);

  simulation_time *time = &simulation_time::getInstance();

  unsigned int step = 0;
  unsigned int freq = 100;

  while (step < 65001) {
    body->step();

    if (step % freq == 0) {
      fprintf(fp, "%4.3e %6.5e\n", time->get_time(),
              std::abs(body->get_particles()[1792]->x[1] -
                       body->get_particles()[1582]->x[1]));
      fflush(fp);
      utilities<3>::vtk_write_particle(body->get_particles(),
                                       body->get_num_part(), step);
    }

    printf("%d: %f\n", step, time->get_time());
    step++;
  }
  fclose(fp);

  return EXIT_SUCCESS;
}
