#include "particle.h"

template <int dim>
particle<dim>::particle() {
  X.setZero();
  x.setZero();
  v.setZero();
  S.setZero();
  grad_v.setZero();
  v_t.setZero();
  S_t.setZero();
  previous_v.setZero();
  a.setZero();
  t.setZero();
  n.setZero();
  H.setZero();
  Fdot.setZero();
  P.setZero();
  div_P.setZero();
  memset(nbh, 0, sizeof(unsigned int) * MAX_NBH);
  memset(w, 0, sizeof(kernel_result<dim>) * MAX_NBH);
};

template <int dim>
particle<dim>::particle(unsigned int idx) : particle() {
  this->idx = idx;
  memset(nbh, 0, sizeof(unsigned int) * MAX_NBH);
  memset(w, 0, sizeof(kernel_result<dim>) * MAX_NBH);
}

template <int dim>
particle<dim>::~particle(){};

template <int dim>
void particle<dim>::reset() {
  x_t.setZero();
  v_t.setZero();
  t.setZero();
  S_t.setZero();
  rho_t = 0.;
}

template <int dim>
void particle<dim>::copy_into(particle *p) const {
  p->idx = idx;
  p->hash = hash;
  p->m = m;
  p->rho = rho;
  p->h = h;
  p->quad_weight = quad_weight;
  p->p = this->p;
  p->x = x;
  p->X = X;
  p->v = v;
  p->S = S;
  p->num_nbh = num_nbh;
  memcpy(p->nbh, nbh, sizeof(unsigned int) * num_nbh);
  memcpy(p->w, w, sizeof(kernel_result<dim>) * num_nbh);
  p->t = t;
  p->a = a;
  p->n = n;
  p->previous_v = previous_v;
  p->P = this->P;
}

template class particle<2>;
template class particle<3>;
