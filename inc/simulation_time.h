#ifndef SIMULATION_TIME_H_
#define SIMULATION_TIME_H_

class simulation_time {
  template <int dim>
  friend class body;

 public:
  static simulation_time &getInstance();
  simulation_time(simulation_time const &) = delete;
  void operator=(simulation_time const &) = delete;
  double get_time() const;
  double get_dt() const;

 private:
  simulation_time();
  double m_time;
  double m_dt;
  void increment_time();
};

#endif /* SIMULATION_TIME_H_ */
