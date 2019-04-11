#include "body.h"
#include <Eigen/Dense>

/* --------------------------------------------------------------------------*/

template <int dim>
class body<dim>::rk4 {
 private:
  particle<dim> **m_k1 = 0;
  particle<dim> **m_k2 = 0;
  particle<dim> **m_k3 = 0;
  particle<dim> **m_k4 = 0;

  particle<dim> **m_k1_q = 0;
  particle<dim> **m_k2_q = 0;
  particle<dim> **m_k3_q = 0;
  particle<dim> **m_k4_q = 0;

  particle<dim> **m_k1_fq = 0;
  particle<dim> **m_k2_fq = 0;
  particle<dim> **m_k3_fq = 0;
  particle<dim> **m_k4_fq = 0;

  void rk_add(particle<dim> **k_in, particle<dim> **p_init,
              particle<dim> **k_out, unsigned int n, double dt) const {
    for (unsigned int i = 0; i < n; i++) {
      p_init[i]->copy_into(k_out[i]);

      k_out[i]->x += k_in[i]->x_t * dt;
      k_out[i]->v += k_in[i]->v_t * dt;
      k_out[i]->rho += k_in[i]->rho_t * dt;
      k_out[i]->S += k_in[i]->S_t * dt;
    }
  }

  void do_step_weak(body<dim> *body) {
    simulation_time *time = &simulation_time::getInstance();
    double m_dt = time->m_dt;

    for (unsigned int i = 0; i < body->m_num_part; i++) {
      body->m_particles[i]->copy_into(m_k1[i]);
    }

    for (unsigned int i = 0; i < body->m_num_quad; i++) {
      body->m_quad_points[i]->copy_into(m_k1_q[i]);
    }

    for (unsigned int i = 0; i < body->m_num_face_quad; i++) {
      body->m_face_quad_points[i]->copy_into(m_k1_fq[i]);
    }

    body->compute_time_derivatives(m_k1, body->m_num_part, m_k1_q,
                                   body->m_num_quad, m_k1_fq,
                                   body->m_num_face_quad);
    rk_add(m_k1, body->m_particles, m_k2, body->m_num_part, 0.5 * m_dt);
    rk_add(m_k1_q, body->m_quad_points, m_k2_q, body->m_num_quad, 0.5 * m_dt);
    rk_add(m_k1_fq, body->m_face_quad_points, m_k2_fq, body->m_num_face_quad,
           0.5 * m_dt);

    body->compute_time_derivatives(m_k2, body->m_num_part, m_k2_q,
                                   body->m_num_quad, m_k2_fq,
                                   body->m_num_face_quad);
    rk_add(m_k2, body->m_particles, m_k3, body->m_num_part, 0.5 * m_dt);
    rk_add(m_k2_q, body->m_quad_points, m_k3_q, body->m_num_quad, 0.5 * m_dt);
    rk_add(m_k2_fq, body->m_face_quad_points, m_k3_fq, body->m_num_face_quad,
           0.5 * m_dt);

    body->compute_time_derivatives(m_k3, body->m_num_part, m_k3_q,
                                   body->m_num_quad, m_k3_fq,
                                   body->m_num_face_quad);
    rk_add(m_k3, body->m_particles, m_k4, body->m_num_part, m_dt);
    rk_add(m_k3_q, body->m_quad_points, m_k4_q, body->m_num_quad, m_dt);
    rk_add(m_k3_fq, body->m_face_quad_points, m_k4_fq, body->m_num_face_quad,
           m_dt);

    body->compute_time_derivatives(m_k4, body->m_num_part, m_k4_q,
                                   body->m_num_quad, m_k4_fq,
                                   body->m_num_face_quad);

    for (unsigned int i = 0; i < body->m_num_part; i++) {
      body->m_particles[i]->x +=
          m_dt * 1. / 6. *
          (m_k1[i]->x_t + 2 * m_k2[i]->x_t + 2 * m_k3[i]->x_t + m_k4[i]->x_t);
      body->m_particles[i]->rho += m_dt * 1. / 6. *
                                   (m_k1[i]->rho_t + 2 * m_k2[i]->rho_t +
                                    2 * m_k3[i]->rho_t + m_k4[i]->rho_t);

      body->m_particles[i]->v +=
          m_dt * 1. / 6. *
          (m_k1[i]->v_t + 2 * m_k2[i]->v_t + 2 * m_k3[i]->v_t + m_k4[i]->v_t);

      body->m_particles[i]->S = body->m_cur_particles[i]->S;

      body->m_particles[i]->t = body->m_cur_particles[i]->t;
    }

    for (unsigned int i = 0; i < body->m_num_quad; i++) {
      body->m_quad_points[i]->x += m_dt * 1. / 6. *
                                   (m_k1_q[i]->x_t + 2 * m_k2_q[i]->x_t +
                                    2 * m_k3_q[i]->x_t + m_k4_q[i]->x_t);
      body->m_quad_points[i]->rho += m_dt * 1. / 6. *
                                     (m_k1_q[i]->rho_t + 2 * m_k2_q[i]->rho_t +
                                      2 * m_k3_q[i]->rho_t + m_k4_q[i]->rho_t);

      body->m_quad_points[i]->S += m_dt * 1. / 6. *
                                   (m_k1_q[i]->S_t + 2 * m_k2_q[i]->S_t +
                                    2 * m_k3_q[i]->S_t + m_k4_q[i]->S_t);
    }

    for (unsigned int i = 0; i < body->m_num_face_quad; i++) {
      body->m_face_quad_points[i]->x += m_dt * 1. / 6. *
                                        (m_k1_fq[i]->x_t + 2 * m_k2_fq[i]->x_t +
                                         2 * m_k3_fq[i]->x_t + m_k4_fq[i]->x_t);
      body->m_face_quad_points[i]->rho +=
          m_dt * 1. / 6. *
          (m_k1_fq[i]->rho_t + 2 * m_k2_fq[i]->rho_t + 2 * m_k3_fq[i]->rho_t +
           m_k4_fq[i]->rho_t);
    }
  }

 public:
  rk4() {}

  rk4(unsigned int num_part, double dt) {
    m_k1 = new particle<dim> *[num_part];
    m_k2 = new particle<dim> *[num_part];
    m_k3 = new particle<dim> *[num_part];
    m_k4 = new particle<dim> *[num_part];

    for (unsigned int i = 0; i < num_part; i++) {
      m_k1[i] = new particle<dim>();
      m_k2[i] = new particle<dim>();
      m_k3[i] = new particle<dim>();
      m_k4[i] = new particle<dim>();
    }

    m_k1_q = 0;
    m_k2_q = 0;
    m_k3_q = 0;
    m_k4_q = 0;

    simulation_time *time = &simulation_time::getInstance();
    time->m_dt = dt;
  }

  rk4(unsigned int num_part, unsigned int num_quad, double dt) {
    m_k1 = new particle<dim> *[num_part];
    m_k2 = new particle<dim> *[num_part];
    m_k3 = new particle<dim> *[num_part];
    m_k4 = new particle<dim> *[num_part];

    for (unsigned int i = 0; i < num_part; i++) {
      m_k1[i] = new particle<dim>();
      m_k2[i] = new particle<dim>();
      m_k3[i] = new particle<dim>();
      m_k4[i] = new particle<dim>();
    }

    m_k1_q = new particle<dim> *[num_quad];
    m_k2_q = new particle<dim> *[num_quad];
    m_k3_q = new particle<dim> *[num_quad];
    m_k4_q = new particle<dim> *[num_quad];

    for (unsigned int i = 0; i < num_quad; i++) {
      m_k1_q[i] = new particle<dim>();
      m_k2_q[i] = new particle<dim>();
      m_k3_q[i] = new particle<dim>();
      m_k4_q[i] = new particle<dim>();
    }

    simulation_time *time = &simulation_time::getInstance();
    time->m_dt = dt;
  }

  rk4(unsigned int num_part, unsigned int num_quad, unsigned int num_face_quad,
      double dt) {
    m_k1 = new particle<dim> *[num_part];
    m_k2 = new particle<dim> *[num_part];
    m_k3 = new particle<dim> *[num_part];
    m_k4 = new particle<dim> *[num_part];

    for (unsigned int i = 0; i < num_part; i++) {
      m_k1[i] = new particle<dim>();
      m_k2[i] = new particle<dim>();
      m_k3[i] = new particle<dim>();
      m_k4[i] = new particle<dim>();
    }

    m_k1_q = new particle<dim> *[num_quad];
    m_k2_q = new particle<dim> *[num_quad];
    m_k3_q = new particle<dim> *[num_quad];
    m_k4_q = new particle<dim> *[num_quad];

    for (unsigned int i = 0; i < num_quad; i++) {
      m_k1_q[i] = new particle<dim>();
      m_k2_q[i] = new particle<dim>();
      m_k3_q[i] = new particle<dim>();
      m_k4_q[i] = new particle<dim>();
    }

    m_k1_fq = new particle<dim> *[num_face_quad];
    m_k2_fq = new particle<dim> *[num_face_quad];
    m_k3_fq = new particle<dim> *[num_face_quad];
    m_k4_fq = new particle<dim> *[num_face_quad];

    for (unsigned int i = 0; i < num_face_quad; i++) {
      m_k1_fq[i] = new particle<dim>();
      m_k2_fq[i] = new particle<dim>();
      m_k3_fq[i] = new particle<dim>();
      m_k4_fq[i] = new particle<dim>();
    }

    simulation_time *time = &simulation_time::getInstance();
    time->m_dt = dt;
  }

  void step(body<dim> *body) {
    simulation_time *time = &simulation_time::getInstance();

    do_step_weak(body);

    time->increment_time();

    body->m_bc.carry_out_boundary_conditions(body->m_particles,
                                             body->m_num_part);
    body->smooth_vel();

    for (unsigned int i = 0; i < body->m_num_part; i++) {
      body->m_particles[i]->a =
          (body->m_particles[i]->v - body->m_particles[i]->previous_v) /
          time->m_dt;
      body->m_particles[i]->previous_v = body->m_particles[i]->v;
    }
  }
};

/* --------------------------------------------------------------------------*/

template <int dim>
struct body<dim>::action {
  virtual void operator()(body<dim> &b) = 0;
};

// Derive current coordinates of quad points, only for visualization purpose
template <int dim>
struct body<dim>::derive_quad_coordinates : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_quad_points()[i];

      Eigen::Matrix<double, dim, 1> x(Eigen::Matrix<double, dim, 1>::Zero());

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        auto xj = pj->x;

        double quad_weight = pj->quad_weight;

        x += xj * w.w * quad_weight;
      }

      pi->x = x;
    }
  }
};

// Derive current coordinates of face quad points, only for visualization
// purpose
template <int dim>
struct body<dim>::derive_face_quad_coordinates : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_face_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_face_quad_points()[i];

      Eigen::Matrix<double, dim, 1> x(Eigen::Matrix<double, dim, 1>::Zero());

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        auto xj = pj->x;

        double quad_weight = pj->quad_weight;

        x += xj * w.w * quad_weight;
      }

      pi->x = x;
    }
  }
};

// H = F - I where F is deformation tensor
// Derive H at quad points
template <int dim>
struct body<dim>::derive_quad_deformation : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_quad_points()[i];

      Eigen::Matrix<double, dim, dim> Hi;
      Hi.setZero();
      auto xi = pi->x;
      auto Xi = pi->X;
      auto ui = xi - Xi;

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        auto xj = pj->x;
        auto Xj = pj->X;
        auto uj = xj - Xj;

        double quad_weight = pj->quad_weight;

        Hi += (uj - ui) * w.grad_w.transpose() * quad_weight;
      }

      pi->H = Hi;
    }
  }
};

// Derive Fdot (dvdX) at quad points
template <int dim>
struct body<dim>::derive_quad_velocity_gradient_ref : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_quad_points()[i];

      Eigen::Matrix<double, dim, dim> Fdoti;
      Fdoti.setZero();
      auto vi = pi->v;

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        auto vj = pj->v;
        double quad_weight = pj->quad_weight;

        Fdoti += (vj - vi) * w.grad_w.transpose() * quad_weight;
      }

      pi->Fdot = Fdoti;
    }
  }
};

// Derive velocity gradient L (dvdx) at quad points
template <int dim>
struct body<dim>::derive_quad_velocity_gradient_cur : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *p = b.get_cur_quad_points()[i];
      // deformation tensor
      auto F = p->H + Eigen::Matrix<double, dim, dim>::Identity();
      p->grad_v = p->Fdot * F.inverse();
    }
  }
};

// Derive current density at quad points
template <int dim>
struct body<dim>::derive_quad_density : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *p = b.get_cur_quad_points()[i];
      p->rho_t -= p->rho * p->grad_v.trace();
    }
  }
};

// Derive hydrostatic pressure at quad points
template <int dim>
struct body<dim>::derive_quad_pressure : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    double rho0 = b.get_sim_data().get_physical_constants().rho0();
    double K = b.get_sim_data().get_physical_constants().K();
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *p = b.get_cur_quad_points()[i];
      double c0 = sqrt(K / rho0);
      p->p = c0 * c0 * (p->rho - rho0);
    }
  }
};

// Derive Jaumann stress rate at quad points
template <int dim>
struct body<dim>::derive_quad_jaumann_stress_rate : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    double G = b.get_sim_data().get_physical_constants().G();
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *p = b.get_cur_quad_points()[i];
      auto epsdot = 0.5 * (p->grad_v + p->grad_v.transpose());
      auto omega = 0.5 * (p->grad_v - p->grad_v.transpose());
      auto S = p->S;
      auto I = Eigen::Matrix<double, dim, dim>::Identity();
      double trace_epsdot = epsdot.trace();
      auto S_t =
          2 * G * (epsdot - 1. / 3. * trace_epsdot * I) - omega * S + S * omega;
      p->S_t += S_t;
    }
  }
};

// Derive nominal stress (PK1^T) at quad points
template <int dim>
struct body<dim>::derive_quad_nominal_stress : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *p = b.get_cur_quad_points()[i];
      auto F = p->H + Eigen::Matrix<double, dim, dim>::Identity();
      double J = F.determinant();
      assert(J > 0.0);
      auto sigma = p->S - p->p * Eigen::Matrix<double, dim, dim>::Identity();
      auto Finv = F.inverse();
      p->P = J * Finv * sigma;
    }
  }
};

// Derive weak form nominal stress divergence (PK1^T) at particle points
// Note that notations and sign of divergence are opposite, however will be
// fixed in derive_part_rhs
template <int dim>
struct body<dim>::derive_part_nominal_stress_divergence
    : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_part(); i++) {
      b.get_cur_particles()[i]->div_P.setZero();
    }
    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_quad_points()[i];
      double quad_weight = pi->quad_weight;
      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];
        double vol_node = pj->quad_weight;
        pj->div_P -= pi->P.transpose() * w.grad_w * quad_weight * vol_node;
      }
    }
  }
};

// Derive Cauchy stress at partical points, only for visualization purpose
// Note that in a particle, S stands for the deviatoric part of Cauchy stress,
// and p stands for the hydrostatic pressure, in output they are combined.
template <int dim>
struct body<dim>::derive_part_cauchy_stress : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_part(); i++) {
      b.get_cur_particles()[i]->S.setZero();
    }

    for (unsigned int i = 0; i < b.get_num_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_quad_points()[i];

      auto Si = pi->S;
      auto pressure = pi->p;

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        double vol_node = pj->quad_weight;

        pj->S += (Si - pressure * decltype(Si)::Identity()) * w.w * vol_node;
      }
    }
  }
};

// Derive weak form traction at particle points
template <int dim>
struct body<dim>::derive_part_traction : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_part(); i++) {
      b.get_cur_particles()[i]->t.setZero();
    }

    for (unsigned int i = 0; i < b.get_num_face_quad_points(); i++) {
      particle<dim> *pi = b.get_cur_face_quad_points()[i];

      auto t = pi->t;

      double quad_weight = pi->quad_weight;

      for (unsigned int j = 0; j < pi->num_nbh; j++) {
        unsigned int jdx = pi->nbh[j];
        kernel_result<dim> w = pi->w[j];
        particle<dim> *pj = b.get_cur_particles()[jdx];

        double vol_node = pj->quad_weight;

        pj->t += t * w.w * quad_weight * vol_node;
      }
    }
  }
};

// Derive weak form RHS at particle points,
// which contains internal, external and damping force
template <int dim>
struct body<dim>::derive_part_rhs : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_part(); i++) {
      particle<dim> *p = b.get_cur_particles()[i];
      double m = p->m;
      double c = b.get_damping();  // damping
      p->v_t += 1. / m * p->div_P + p->t / m - c * p->v;
    }
  }
};

// Derive coordinate time derivative (dxdt) at particle points for time
// stepping, which are the same as velocity.
template <int dim>
struct body<dim>::derive_part_coordinate_derivative : public body<dim>::action {
  virtual void operator()(body<dim> &b) {
    for (unsigned int i = 0; i < b.get_num_part(); i++) {
      particle<dim> *p = b.get_cur_particles()[i];

      p->x_t += p->v;
    }
  }
};

/* --------------------------------------------------------------------------*/

template <int dim>
body<dim>::~body(){};

template <int dim>
body<dim>::body(particle<dim> **particles, unsigned int n, simulation_data data,
                double dt, boundary_conditions<dim> bc,
                particle<dim> **quad_points, unsigned int nq,
                particle<dim> **face_quad_points, unsigned int nfq,
                double damping)
    : m_data(data),
      m_bc(bc),
      m_num_part(n),
      m_particles(particles),
      m_cur_particles(particles),
      m_num_quad(nq),
      m_quad_points(quad_points),
      m_cur_quad_points(quad_points),
      m_num_face_quad(nfq),
      m_face_quad_points(face_quad_points),
      m_cur_face_quad_points(face_quad_points),
      m_c(damping) {
  m_time_stepper = std::make_unique<rk4>(n, nq, nfq, dt);
  m_actions.emplace_back(new derive_quad_coordinates());
  m_actions.emplace_back(new derive_face_quad_coordinates());
  m_actions.emplace_back(new derive_quad_deformation());
  m_actions.emplace_back(new derive_quad_velocity_gradient_ref());
  m_actions.emplace_back(new derive_quad_velocity_gradient_cur());
  m_actions.emplace_back(new derive_quad_density());
  m_actions.emplace_back(new derive_quad_pressure());
  m_actions.emplace_back(new derive_quad_jaumann_stress_rate());
  m_actions.emplace_back(new derive_quad_nominal_stress());
  m_actions.emplace_back(new derive_part_nominal_stress_divergence());
  m_actions.emplace_back(new derive_part_cauchy_stress());
  m_actions.emplace_back(new derive_part_traction());
  m_actions.emplace_back(new derive_part_rhs());
  m_actions.emplace_back(new derive_part_coordinate_derivative());
}

template <int dim>
void body<dim>::smooth_vel() {
  if (!this->get_sim_data().get_correction_constants().do_smooth()) return;

  static Eigen::Matrix<double, dim, 1> *vs = 0;

  if (vs == 0) {
    vs = (Eigen::Matrix<double, dim, 1> *)calloc(
        get_num_part(), sizeof(Eigen::Matrix<double, dim, 1>));
  }

  for (unsigned int i = 0; i < get_num_part(); i++) {
    particle<dim> *pi = get_particles()[i];

    Eigen::Matrix<double, dim, 1> vsi;
    vsi.setZero();

    for (unsigned int j = 0; j < pi->num_nbh; j++) {
      unsigned int jdx = pi->nbh[j];
      kernel_result<dim> w = pi->w[j];
      particle<dim> *pj = get_particles()[jdx];

      auto vj = pj->v;
      double quad_weight = pj->quad_weight;

      vsi += vj * w.w * quad_weight;
    }

    vs[i] = vsi;
  }

  for (unsigned int i = 0; i < get_num_part(); i++) {
    get_particles()[i]->v = vs[i];
  }
}

template <int dim>
void body<dim>::compute_time_derivatives(particle<dim> **particles,
                                         unsigned int num_part,
                                         particle<dim> **quad_points,
                                         unsigned int num_quad,
                                         particle<dim> **face_quad_points,
                                         unsigned int num_face_quad) {
  for (unsigned int i = 0; i < num_part; i++) {
    particles[i]->reset();
  }

  for (unsigned int i = 0; i < num_quad; i++) {
    quad_points[i]->reset();
  }

  m_bc.carry_out_boundary_conditions(particles, num_part);
  m_bc.carry_out_neumann_boundary_conditions(face_quad_points, num_face_quad);

  m_cur_particles = particles;
  m_cur_quad_points = quad_points;
  m_cur_face_quad_points = face_quad_points;
  for (auto &act : m_actions) {
    act->operator()(*this);  // apply action
  }
}

template <int dim>
void body<dim>::step() {
  m_time_stepper->step(this);
}

template class body<2>;
template class body<3>;
