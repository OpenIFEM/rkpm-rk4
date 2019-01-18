#include "simulation_time.h"

simulation_time& simulation_time::getInstance() {
  static simulation_time instance;
  return instance;
}

double simulation_time::get_time() const { return m_time; }

double simulation_time::get_dt() const { return m_dt; }

simulation_time::simulation_time() : m_time(0.), m_dt(0.) {}

void simulation_time::increment_time() { m_time += m_dt; }
